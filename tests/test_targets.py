import errno
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any

import pytest

from mlx_model_doctor.errors import TargetError
from mlx_model_doctor.safetensors_header import SafetensorsHeader
from mlx_model_doctor.targets import LocalTarget
from tests.fakes import write_hf_cache_snapshot


def test_local_target_lists_sizes_and_reads_files(tmp_path: Path) -> None:
    model = tmp_path / "model"
    nested = model / "nested"
    nested.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    (model / "weights.safetensors").write_bytes(b"weights")
    (nested / "tokenizer.json").write_text('{"tokens":[]}', encoding="utf-8")

    target = LocalTarget(model)

    assert target.name == str(model)
    assert target.source == "local"
    assert target.list_files() == (
        "config.json",
        "nested/tokenizer.json",
        "weights.safetensors",
    )
    assert target.exists("config.json")
    assert not target.exists("missing.json")
    assert target.size("weights.safetensors") == len(b"weights")
    assert target.size("missing.json") is None
    assert target.read_bytes("weights.safetensors") == b"weights"
    assert target.read_text("nested/tokenizer.json") == '{"tokens":[]}'


def test_local_target_honors_max_bytes_for_byte_and_text_reads(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokens.txt").write_text("abcdef", encoding="utf-8")

    target = LocalTarget(model)

    assert target.read_bytes("tokens.txt", max_bytes=3) == b"abc"
    assert target.read_text("tokens.txt", max_bytes=4) == "abcd"


def test_local_target_rejects_paths_outside_root(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    target = LocalTarget(model)

    with pytest.raises(TargetError, match="outside"):
        target.read_text("../secret.txt")


def test_local_target_list_files_excludes_symlinks_outside_root(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (model / "outside_link").symlink_to(outside)

    target = LocalTarget(model)

    assert target.list_files() == ()
    with pytest.raises(TargetError, match="outside"):
        target.read_text("outside_link")


def test_local_target_allows_symlinks_resolving_inside_root(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    target_file = model / "config.json"
    target_file.write_text("{}", encoding="utf-8")
    (model / "config-link.json").symlink_to(target_file)

    target = LocalTarget(model)

    assert target.list_files() == ("config-link.json", "config.json")
    assert target.read_text("config-link.json") == "{}"


def test_local_target_rejects_outside_root_exists_checks(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    target = LocalTarget(model)

    with pytest.raises(TargetError, match="outside"):
        target.exists("../secret.txt")


def test_local_target_rejects_outside_root_size_checks(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    target = LocalTarget(model)

    with pytest.raises(TargetError, match="outside"):
        target.size("../secret.txt")


def test_local_target_requires_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(TargetError, match="directory"):
        LocalTarget(tmp_path / "missing")


def test_local_target_wraps_malformed_paths(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    target = LocalTarget(model)

    with pytest.raises(TargetError, match="Invalid local target path"):
        target.exists("bad\0path")


def _write_safetensors(path: Path, header: dict[str, object], data: bytes = b"") -> None:
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)


def test_local_target_reads_single_file_header_without_reading_weights(tmp_path: Path) -> None:
    model = tmp_path / "m"
    model.mkdir()
    _write_safetensors(
        model / "model.safetensors",
        {"w": {"dtype": "BF16", "shape": [2, 4], "data_offsets": [0, 16]}},
        data=b"\x00" * 16,
    )
    header = LocalTarget(model).safetensors_header()
    assert isinstance(header, SafetensorsHeader)
    assert header.sharded is False
    assert header.weight_map == {"w": "model.safetensors"}
    entry = header.tensor("w")
    assert entry is not None
    assert entry.shape == (2, 4)
    assert header.files[0].data_section_length == 16  # file_size threaded from stat


def test_local_target_returns_none_without_safetensors(tmp_path: Path) -> None:
    model = tmp_path / "m"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    assert LocalTarget(model).safetensors_header() is None


def test_local_target_uses_index_weight_map_for_sharded_model(tmp_path: Path) -> None:
    model = tmp_path / "m"
    model.mkdir()
    _write_safetensors(
        model / "model-00001-of-00002.safetensors",
        {"a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}},
        data=b"\x00" * 4,
    )
    _write_safetensors(
        model / "model-00002-of-00002.safetensors",
        {"b": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}},
        data=b"\x00" * 4,
    )
    (model / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    header = LocalTarget(model).safetensors_header()
    assert header is not None
    assert header.sharded is True
    assert header.weight_map == {
        "a": "model-00001-of-00002.safetensors",
        "b": "model-00002-of-00002.safetensors",
    }


from mlx_model_doctor.targets import _canonical_shard_paths  # noqa: E402


def test_canonical_shard_paths_top_level_only_without_index() -> None:
    files = ["config.json", "model.safetensors", "vae/model.safetensors"]
    assert _canonical_shard_paths(files, None) == ["model.safetensors"]


def test_canonical_shard_paths_prefers_index_named_shards() -> None:
    files = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "vae/diffusion.safetensors",
    ]
    weight_map = {
        "w1": "model-00001-of-00002.safetensors",
        "w2": "model-00002-of-00002.safetensors",
    }
    assert _canonical_shard_paths(files, weight_map) == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]


def test_canonical_shard_paths_ignores_nested_when_index_only_names_top_level() -> None:
    files = ["model.safetensors", "vae/model.safetensors"]
    assert _canonical_shard_paths(files, {"w": "model.safetensors"}) == ["model.safetensors"]


def test_canonical_shard_paths_falls_back_when_weight_map_has_no_matching_files() -> None:
    files = ["model.safetensors", "vae/model.safetensors"]
    weight_map = {"w": "stale-missing.safetensors"}
    assert _canonical_shard_paths(files, weight_map) == ["model.safetensors"]


def test_local_target_header_excludes_nested_component_shards(tmp_path: Path) -> None:
    tensor = {"w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}
    _write_safetensors(tmp_path / "model.safetensors", tensor)
    (tmp_path / "vae").mkdir()
    _write_safetensors(tmp_path / "vae" / "model.safetensors", {"vae.w": tensor["w"]})

    header = LocalTarget(tmp_path).safetensors_header()

    assert header is not None
    names = list(header.tensor_names())
    assert "w" in names
    assert "vae.w" not in names


# --- Hugging Face cache snapshots (blob symlinks) -------------------------------

_CONFIG = b'{"model_type":"llama"}'
_TOKENIZER = b'{"version":"1.0","model":{}}'
_NESTED = b'{"x":1}'


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_sha1(data: bytes) -> str:
    """The git blob id, the 40-hex name huggingface_hub gives a non-LFS file's blob."""
    return hashlib.sha1(b"blob %d\x00" % len(data) + data, usedforsecurity=False).hexdigest()


def _cache_snapshot(tmp_path: Path, *, commit: str = "0" * 40) -> Path:
    return write_hf_cache_snapshot(
        tmp_path / "hub", "org/model", {"config.json": _CONFIG}, commit=commit
    )


def test_local_target_reads_hf_cache_snapshot_symlinks(tmp_path: Path) -> None:
    # Bug caught: the traversal guard rejects a snapshot's relative link into the
    # sibling blobs/ directory, so every file of a cached model reads as outside.
    snapshot = write_hf_cache_snapshot(
        tmp_path / "hub",
        "org/model",
        {"config.json": _CONFIG, "tokenizer.json": _TOKENIZER, "nested/x.json": _NESTED},
    )

    target = LocalTarget(snapshot)

    assert target.list_files() == ("config.json", "nested/x.json", "tokenizer.json")
    assert target.exists("config.json")
    assert target.size("tokenizer.json") == len(_TOKENIZER)
    assert target.read_text("config.json") == _CONFIG.decode()
    assert target.read_text("nested/x.json") == _NESTED.decode()
    assert target.read_bytes("tokenizer.json", offset=2, length=3) == _TOKENIZER[2:5]


def test_hf_cache_snapshot_reads_a_git_sha1_named_blob(tmp_path: Path) -> None:
    # Bug caught: the blob-name rule accepts only 64-hex (sha256) names, so the 40-hex
    # git-sha1 blobs huggingface_hub stores for non-LFS files are rejected.
    snapshot = _cache_snapshot(tmp_path)
    blob = "c" * 40
    (snapshot.parent.parent / "blobs" / blob).write_bytes(b"MIT")
    (snapshot / "LICENSE").symlink_to("../../blobs/" + blob)

    target = LocalTarget(snapshot)

    assert target.read_text("LICENSE") == "MIT"
    assert target.list_files() == ("LICENSE", "config.json")


@pytest.mark.parametrize(
    "outside_name",
    ["outside.txt", "f" * 64, "blobs/sub/" + "f" * 64],
    ids=["plain", "hash", "nested"],
)
def test_hf_cache_snapshot_still_rejects_symlink_escaping_the_cache(
    tmp_path: Path, outside_name: str
) -> None:
    """Regression fence (green before and after the cache allowance).

    Bug caught: the allowance accepts any out-of-root target once the root looks like a
    cache snapshot, or any hash-named file not sitting directly in ``blobs/`` (outside the
    cache, or nested below ``blobs/``).
    """
    snapshot = _cache_snapshot(tmp_path)
    repo = snapshot.parent.parent
    outside = (repo if outside_name.startswith("blobs/") else tmp_path) / outside_name
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("secret", encoding="utf-8")
    (snapshot / "secret.json").symlink_to(outside)

    target = LocalTarget(snapshot)

    with pytest.raises(TargetError, match="outside"):
        target.read_text("secret.json")
    assert "secret.json" not in target.list_files()


def test_hf_cache_snapshot_rejects_symlink_into_another_snapshot(tmp_path: Path) -> None:
    """Regression fence (green before and after the cache allowance).

    Bug caught: the allowance widened from ``blobs/`` to the whole ``models--*`` repo dir,
    so a file planted in another snapshot reads through a link.
    """
    snapshot = _cache_snapshot(tmp_path)
    other = snapshot.parent / ("1" * 40)
    other.mkdir()
    (other / "secret.json").write_text("other snapshot", encoding="utf-8")
    (snapshot / "secret.json").symlink_to("../" + "1" * 40 + "/secret.json")

    target = LocalTarget(snapshot)

    with pytest.raises(TargetError, match="outside"):
        target.read_text("secret.json")
    assert "secret.json" not in target.list_files()


def test_hf_cache_snapshot_rejects_symlinked_blobs_directory(tmp_path: Path) -> None:
    """Regression fence (green before and after the cache allowance).

    Bug caught: ``blobs`` is accepted when it is itself a symlink to another directory.
    """
    snapshot = _cache_snapshot(tmp_path)
    blobs = snapshot.parent.parent / "blobs"
    elsewhere = tmp_path / "elsewhere"
    blobs.rename(elsewhere)
    blobs.symlink_to(elsewhere, target_is_directory=True)
    assert (elsewhere / _sha256(_CONFIG)).is_file()

    target = LocalTarget(snapshot)

    with pytest.raises(TargetError, match="outside"):
        target.read_text("config.json")
    assert target.list_files() == ()


@pytest.mark.parametrize(
    "blob_name",
    ["id_rsa", "a" * 63, "a" * 65, "a" * 39, "a" * 41, "a" * 64 + ".incomplete", "A" * 64],
    ids=["id_rsa", "63-hex", "65-hex", "39-hex", "41-hex", "incomplete", "uppercase"],
)
def test_hf_cache_snapshot_rejects_non_hash_blob_name(tmp_path: Path, blob_name: str) -> None:
    """Regression fence (green before and after the cache allowance).

    Bug caught: the blob-name check dropped or loosened (a prefix match instead of a full
    match), so any file placed in ``blobs/`` reads through a link.
    """
    snapshot = _cache_snapshot(tmp_path)
    (snapshot.parent.parent / "blobs" / blob_name).write_text("key", encoding="utf-8")
    (snapshot / "key.json").symlink_to("../../blobs/" + blob_name)

    target = LocalTarget(snapshot)

    with pytest.raises(TargetError, match="outside"):
        target.read_text("key.json")
    assert "key.json" not in target.list_files()


def test_hf_cache_snapshot_rejects_directory_symlink_into_blobs(tmp_path: Path) -> None:
    """Regression fence (green before and after the cache allowance).

    Bug caught: only the fully resolved path is checked, not the lexical parent, so a
    directory link into ``blobs/`` exposes every blob under a new name.
    """
    snapshot = _cache_snapshot(tmp_path)
    (snapshot / "sub").symlink_to("../../blobs", target_is_directory=True)

    target = LocalTarget(snapshot)

    with pytest.raises(TargetError, match="outside"):
        target.read_text("sub/" + _sha256(_CONFIG))
    assert not any(name.startswith("sub/") for name in target.list_files())


@pytest.mark.parametrize("form", ["dotdot", "absolute"])
def test_hf_cache_snapshot_rejects_requested_path_naming_a_blob(tmp_path: Path, form: str) -> None:
    """Regression fence (green before and after the cache allowance).

    Bug caught: a lexical ``..`` or absolute request for a blob rides the allowance meant
    only for a snapshot's own leaf links.
    """
    snapshot = _cache_snapshot(tmp_path)
    blob = snapshot.parent.parent / "blobs" / _sha256(_CONFIG)
    request = "../../blobs/" + blob.name if form == "dotdot" else str(blob)

    target = LocalTarget(snapshot)

    with pytest.raises(TargetError, match="outside"):
        target.read_text(request)
    with pytest.raises(TargetError, match="outside"):
        target.exists(request)


def test_hf_cache_snapshot_rejects_blob_linking_onward_outside(tmp_path: Path) -> None:
    """Regression fence (green before and after the cache allowance).

    Bug caught: only the first hop (the blob name) is checked, not the fully resolved
    path, so a blob that is itself a link to an outside file reads through.
    """
    snapshot = _cache_snapshot(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    blob = "b" * 64
    (snapshot.parent.parent / "blobs" / blob).symlink_to(outside)
    (snapshot / "secret.json").symlink_to("../../blobs/" + blob)

    target = LocalTarget(snapshot)

    with pytest.raises(TargetError, match="outside"):
        target.read_text("secret.json")
    assert "secret.json" not in target.list_files()


@pytest.mark.parametrize(
    "commit",
    ["main", "0" * 39, "0" * 41, "A" * 40],
    ids=["ref-name", "39-hex", "41-hex", "uppercase"],
)
def test_cache_shaped_tree_with_non_hash_snapshot_name_gets_no_allowance(
    tmp_path: Path, commit: str
) -> None:
    """Regression fence (green before and after the cache allowance).

    Bug caught: the snapshot-name check dropped or loosened, so any directory under
    ``snapshots/`` gets the blob allowance.
    """
    snapshot = _cache_snapshot(tmp_path, commit=commit)

    target = LocalTarget(snapshot)

    with pytest.raises(TargetError, match="outside"):
        target.read_text("config.json")
    assert target.list_files() == ()


@pytest.mark.parametrize(
    ("repo_dir", "snapshots_dir"),
    [("datasets--org--model", "snapshots"), ("models--org--model", "snapshot")],
    ids=["not-a-models-dir", "not-snapshots"],
)
def test_cache_shaped_tree_off_the_hub_layout_gets_no_allowance(
    tmp_path: Path, repo_dir: str, snapshots_dir: str
) -> None:
    """Regression fence (green before and after the cache allowance).

    Bug caught: the ``snapshots`` parent or ``models--`` grandparent check dropped, so a
    look-alike tree outside the model cache layout gets the blob allowance.
    """
    repo = tmp_path / repo_dir
    blobs = repo / "blobs"
    snapshot = repo / snapshots_dir / ("0" * 40)
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (blobs / _sha256(_CONFIG)).write_bytes(_CONFIG)
    (snapshot / "config.json").symlink_to("../../blobs/" + _sha256(_CONFIG))

    target = LocalTarget(snapshot)

    with pytest.raises(TargetError, match="outside"):
        target.read_text("config.json")
    assert target.list_files() == ()


def test_hf_cache_snapshot_dangling_blob_link_is_absent(tmp_path: Path) -> None:
    # Bug caught: a link to a blob that was never downloaded maps to "outside" or
    # raises, instead of reading as an absent file (an incomplete cache download).
    snapshot = _cache_snapshot(tmp_path)
    (snapshot / "tokenizer.json").symlink_to("../../blobs/" + "d" * 64)

    target = LocalTarget(snapshot)

    assert target.exists("tokenizer.json") is False
    assert target.size("tokenizer.json") is None
    assert target.list_files() == ("config.json",)


@pytest.mark.parametrize(
    ("kind", "blob"),
    [("fifo", "e" * 40), ("directory", "e" * 40), ("directory", "e" * 64)],
    ids=["fifo-40-hex", "directory-40-hex", "directory-64-hex"],
)
def test_hf_cache_snapshot_one_hop_blob_that_is_not_a_regular_file_is_refused(
    tmp_path: Path, kind: str, blob: str
) -> None:
    # Bug caught: the one-hop endpoint checks only the blob's name, not that it is a
    # regular file, so a hash-named FIFO or directory in blobs/ passes the guard.
    snapshot = _cache_snapshot(tmp_path)
    entry = snapshot.parent.parent / "blobs" / blob
    if kind == "fifo":
        os.mkfifo(entry)
    else:
        entry.mkdir()
    (snapshot / "odd.json").symlink_to("../../blobs/" + blob)

    target = LocalTarget(snapshot)

    # exists first: should a regression accept the FIFO, a read of it would block.
    with pytest.raises(TargetError, match="outside"):
        target.exists("odd.json")
    with pytest.raises(TargetError, match="outside"):
        target.size("odd.json")
    with pytest.raises(TargetError, match="outside"):
        target.read_bytes("odd.json")


def test_hf_cache_snapshot_one_hop_blob_that_cannot_be_inspected_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug caught: every lstat error on the one-hop blob maps to "absent" (accepted), not
    # only a missing blob, so a blob the guard cannot inspect passes unchecked.
    snapshot = _cache_snapshot(tmp_path)
    target = LocalTarget(snapshot)
    real_lstat = Path.lstat

    def lstat(self: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        if self.name == _sha256(_CONFIG):
            raise PermissionError(errno.EACCES, "Permission denied", str(self))
        return real_lstat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", lstat)

    with pytest.raises(TargetError, match="outside"):
        target.exists("config.json")
    with pytest.raises(TargetError, match="outside"):
        target.read_text("config.json")


def test_hf_cache_snapshot_link_into_another_repos_blobs_is_refused(tmp_path: Path) -> None:
    """Regression fence (green before and after the regular-file rule).

    Bug caught: the one-hop endpoint accepts a blob in any ``models--*/blobs`` of the
    cache, not only this repo's own, so a link into another cached repo reads through.
    """
    snapshot = _cache_snapshot(tmp_path)
    write_hf_cache_snapshot(tmp_path / "hub", "org/other", {"secret.json": _SECRET})
    other_blob = tmp_path / "hub" / "models--org--other" / "blobs" / _sha256(_SECRET)
    assert other_blob.is_file()
    (snapshot / "secret.json").symlink_to("../../../models--org--other/blobs/" + other_blob.name)

    target = LocalTarget(snapshot)

    with pytest.raises(TargetError, match="outside"):
        target.read_text("secret.json")
    with pytest.raises(TargetError, match="outside"):
        target.exists("secret.json")
    assert "secret.json" not in target.list_files()


def test_symlink_loop_in_the_model_root_never_raises_runtime_error(tmp_path: Path) -> None:
    # Bug caught: on Python 3.11 and 3.12 ``resolve`` raises RuntimeError for a symlink
    # loop and ``_path`` lets it escape; every read must give a value or a TargetError.
    model = tmp_path / "model"
    model.mkdir()
    (model / "a").symlink_to("b")
    (model / "b").symlink_to("a")
    target = LocalTarget(model)
    calls = {
        "exists": lambda: target.exists("a"),
        "size": lambda: target.size("a"),
        "read_bytes": lambda: target.read_bytes("a"),
    }

    outcomes: dict[str, object] = {}
    for name, call in calls.items():
        try:
            outcomes[name] = call()
        except TargetError:
            outcomes[name] = TargetError

    assert outcomes["exists"] in (False, TargetError)
    assert outcomes["size"] in (None, TargetError)
    assert outcomes["read_bytes"] is TargetError
    assert target.list_files() == ()


@pytest.mark.parametrize("which", ["requested-path", "lexical-parent"])
def test_resolve_runtime_error_is_an_invalid_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    # Bug caught: ``_path`` maps only OSError and ValueError from either resolve call, so
    # the RuntimeError Python 3.11 and 3.12 raise for a loop escapes as a raw error.
    snapshot = _cache_snapshot(tmp_path)
    target = LocalTarget(snapshot)
    trigger = snapshot / "config.json" if which == "requested-path" else snapshot
    real_resolve = Path.resolve

    def resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        if self == trigger:
            raise RuntimeError(f"Symlink loop from {str(self)!r}")
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)

    with pytest.raises(TargetError, match="Invalid local target path"):
        target.read_text("config.json")
    with pytest.raises(TargetError, match="Invalid local target path"):
        target.exists("config.json")


def test_hf_cache_snapshot_exists_maps_oserror_to_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug caught: the cache branch moves is_file out of the OSError mapping, so a name
    # Python 3.11-3.13 cannot stat (ENAMETOOLONG) raises instead of reading as absent.
    target = LocalTarget(_cache_snapshot(tmp_path))
    real_is_file = Path.is_file

    def is_file(self: Path, *args: Any, **kwargs: Any) -> bool:
        if self.name == _sha256(_CONFIG):
            raise OSError(errno.ENAMETOOLONG, "File name too long", str(self))
        return real_is_file(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_file", is_file)

    assert target.exists("config.json") is False


def test_hf_cache_blobs_dir_that_cannot_be_resolved_gives_no_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug caught: an OSError while checking blobs/ (no permission) escapes LocalTarget's
    # constructor as a raw OSError instead of leaving the snapshot without the allowance.
    snapshot = _cache_snapshot(tmp_path)
    blobs = snapshot.parent.parent / "blobs"
    real_resolve = Path.resolve

    def resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        if self == blobs:
            raise PermissionError(errno.EACCES, "Permission denied", str(self))
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    target = LocalTarget(snapshot)

    with pytest.raises(TargetError, match="outside"):
        target.read_text("config.json")


def test_hf_cache_lexical_parent_resolve_error_is_an_invalid_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug caught: the second resolve (the requested path's lexical parent) sits outside
    # the OSError mapping, so a blob-linked name raises a raw OSError, not a TargetError.
    snapshot = _cache_snapshot(tmp_path)
    target = LocalTarget(snapshot)
    real_resolve = Path.resolve

    def resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        if self == snapshot:
            raise OSError(errno.ELOOP, "Too many levels of symbolic links", str(self))
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)

    with pytest.raises(TargetError, match="Invalid local target path"):
        target.read_text("config.json")


# --- Shared blob store of huggingface_hub 1.32 and later (second hop for Xet files) -

# A store entry is named by a 64-hex Xet hash under its two-character prefix directory.
_XET = "ab" + "0123456789abcdef" * 3 + "0123456789abcd"
_SECRET = b'{"secret":true}'


def _shared_snapshot(tmp_path: Path) -> Path:
    # Every file is routed through the store: these tests read the second hop.
    files = {"config.json": _CONFIG, "tokenizer.json": _TOKENIZER, "nested/x.json": _NESTED}
    return write_hf_cache_snapshot(
        tmp_path / "hub", "org/model", files, shared_blobs=True, store_files=files
    )


def _hub(snapshot: Path) -> Path:
    return snapshot.parent.parent.parent


def _plant(snapshot: Path, link_target: str, name: str = "secret.json") -> None:
    """Link ``name`` -> repo ``blobs/<etag>`` -> ``link_target`` (relative to repo blobs/)."""
    etag = _sha256(name.encode())
    (snapshot.parent.parent / "blobs" / etag).symlink_to(link_target)
    (snapshot / name).symlink_to("../../blobs/" + etag)


def _plant_store_file(snapshot: Path, store_relative: str, data: bytes = _SECRET) -> None:
    entry = _hub(snapshot) / "blobs" / store_relative
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_bytes(data)
    _plant(snapshot, "../../blobs/" + store_relative)


def _assert_refused(target: LocalTarget, name: str = "secret.json") -> None:
    with pytest.raises(TargetError, match="outside"):
        target.read_text(name)
    with pytest.raises(TargetError, match="outside"):
        target.exists(name)
    assert name not in target.list_files()


def test_local_target_reads_hf_shared_blob_snapshot(tmp_path: Path) -> None:
    # Bug caught: the guard accepts only the one-hop blobs/<hash> endpoint, so every
    # Xet file of huggingface_hub 1.32+ (blobs/<etag> -> <hub>/blobs/<xx>/<xet>) is outside.
    snapshot = _shared_snapshot(tmp_path)
    repo_blob = snapshot.parent.parent / "blobs" / _sha256(_TOKENIZER)
    assert repo_blob.is_symlink()  # the layout really takes two hops

    target = LocalTarget(snapshot)

    assert target.list_files() == ("config.json", "nested/x.json", "tokenizer.json")
    assert target.exists("tokenizer.json")
    assert target.size("tokenizer.json") == len(_TOKENIZER)
    assert target.read_text("config.json") == _CONFIG.decode()
    assert target.read_text("nested/x.json") == _NESTED.decode()
    assert target.read_bytes("tokenizer.json", offset=2, length=3) == _TOKENIZER[2:5]


def test_hf_shared_blob_snapshot_reads_safetensors_header(tmp_path: Path) -> None:
    # Bug caught: the header path (list_files + size + bounded reads) breaks on a
    # shard stored in the shared blob store.
    raw = json.dumps({"w": {"dtype": "BF16", "shape": [2, 4], "data_offsets": [0, 16]}})
    shard = struct.pack("<Q", len(raw)) + raw.encode() + b"\x00" * 16
    snapshot = write_hf_cache_snapshot(
        tmp_path / "hub", "org/model", {"model.safetensors": shard}, shared_blobs=True
    )

    header = LocalTarget(snapshot).safetensors_header()

    assert header is not None
    assert header.weight_map == {"w": "model.safetensors"}
    assert header.files[0].data_section_length == 16


def test_hf_snapshot_mixes_one_hop_and_shared_blob_files(tmp_path: Path) -> None:
    # Bug caught: enabling the shared-store endpoint drops the one-hop endpoint (or the
    # reverse), as on a real 1.32+ cache where small files stay one hop.
    snapshot = _cache_snapshot(tmp_path)
    store = _hub(snapshot) / "blobs"
    store.mkdir()
    (store / ".huggingface-shared-blobs").write_text("1\n", encoding="utf-8")
    _plant_store_file(snapshot, f"{_XET[:2]}/{_XET}")

    target = LocalTarget(snapshot)

    assert target.read_text("secret.json") == _SECRET.decode()
    assert target.read_text("config.json") == _CONFIG.decode()
    assert target.list_files() == ("config.json", "secret.json")


def test_hf_shared_blob_store_without_marker_is_refused(tmp_path: Path) -> None:
    # Bug caught: any cache-root blobs/ is trusted as a shared store without its marker.
    snapshot = _shared_snapshot(tmp_path)
    marker = _hub(snapshot) / "blobs" / ".huggingface-shared-blobs"
    marker.rename(tmp_path / "marker-moved")

    _assert_refused(LocalTarget(snapshot), "config.json")


@pytest.mark.parametrize(
    "content",
    [b"", b"1", b"2\n", b"1\r\n", b"1\n\n", b" 1\n", b"10\n"],
    ids=["empty", "no-newline", "version-2", "crlf", "trailing", "leading-space", "10"],
)
def test_hf_shared_blob_marker_content_must_be_exactly_layout_1(
    tmp_path: Path, content: bytes
) -> None:
    # Bug caught: the marker content check dropped or loosened (strip, prefix, newline
    # translation), so an unknown store layout version is followed.
    snapshot = _shared_snapshot(tmp_path)
    (_hub(snapshot) / "blobs" / ".huggingface-shared-blobs").write_bytes(content)

    _assert_refused(LocalTarget(snapshot), "config.json")


def test_hf_shared_blob_marker_that_is_a_symlink_is_refused(tmp_path: Path) -> None:
    # Bug caught: the marker is stat-ed through a link instead of lstat-ed, so a link to
    # any file holding "1\n" marks a directory as a shared store.
    snapshot = _shared_snapshot(tmp_path)
    marker = _hub(snapshot) / "blobs" / ".huggingface-shared-blobs"
    elsewhere = tmp_path / "marker-elsewhere"
    marker.rename(elsewhere)
    marker.symlink_to(elsewhere)
    assert marker.read_text(encoding="utf-8") == "1\n"

    _assert_refused(LocalTarget(snapshot), "config.json")


def test_hf_shared_blob_store_that_is_a_symlink_is_refused(tmp_path: Path) -> None:
    # Bug caught: the store is taken as the resolved <hub>/blobs, so a store dir that is
    # a link to a marked directory elsewhere is trusted.
    snapshot = _shared_snapshot(tmp_path)
    store = _hub(snapshot) / "blobs"
    elsewhere = tmp_path / "elsewhere"
    store.rename(elsewhere)
    store.symlink_to(elsewhere, target_is_directory=True)
    assert (store / ".huggingface-shared-blobs").read_text(encoding="utf-8") == "1\n"

    _assert_refused(LocalTarget(snapshot), "config.json")


@pytest.mark.parametrize(
    "store_relative",
    [
        f"{_XET[:2]}/{_XET}.refs",
        f"cd/{_XET}",
        f"{_XET[:2]}/{_XET[:2]}/{_XET}",
        f"{_XET[:2]}/{_XET[:40]}",
        f"{_XET[:2].upper()}/{_XET.upper()}",
    ],
    ids=["refs-manifest", "prefix-mismatch", "nested-deeper", "40-hex", "uppercase"],
)
def test_hf_shared_blob_store_rejects_off_layout_entries(
    tmp_path: Path, store_relative: str
) -> None:
    # Bug caught: the entry shape check (64-hex full match, prefix dir == name[:2],
    # prefix dir directly in the store) dropped or loosened.
    snapshot = _shared_snapshot(tmp_path)
    _plant_store_file(snapshot, store_relative)

    _assert_refused(LocalTarget(snapshot))


def test_hf_shared_blob_store_of_another_cache_root_is_refused(tmp_path: Path) -> None:
    # Bug caught: any marked store is accepted, not only the one in this snapshot's own
    # cache root, so a link into a second cache reads through.
    snapshot = _shared_snapshot(tmp_path)
    other = write_hf_cache_snapshot(
        tmp_path / "hub2", "org/other", {"config.json": _CONFIG}, shared_blobs=True
    )
    entry = _hub(other) / "blobs" / _XET[:2] / _XET
    entry.parent.mkdir(exist_ok=True)
    entry.write_bytes(_SECRET)
    _plant(snapshot, str(entry))

    _assert_refused(LocalTarget(snapshot))


def test_hf_shared_blob_store_symlink_resolving_outside_is_refused(tmp_path: Path) -> None:
    """Regression fence (green before and after the shared-store endpoint).

    Bug caught: only the store entry's name is checked, not the fully resolved path, so
    a store entry that is itself a link to an outside file reads through.
    """
    snapshot = _shared_snapshot(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_bytes(_SECRET)
    entry = _hub(snapshot) / "blobs" / _XET[:2] / _XET
    entry.parent.mkdir(exist_ok=True)
    entry.symlink_to(outside)
    _plant(snapshot, f"../../blobs/{_XET[:2]}/{_XET}")

    _assert_refused(LocalTarget(snapshot))


def test_hf_shared_blob_store_entry_that_is_a_directory_is_refused(tmp_path: Path) -> None:
    # Bug caught: the regular-file check on the store entry dropped, so a hash-named
    # directory in the store passes the guard (and reads fail as I/O errors instead).
    snapshot = _shared_snapshot(tmp_path)
    (_hub(snapshot) / "blobs" / _XET[:2] / _XET).mkdir(parents=True)
    _plant(snapshot, f"../../blobs/{_XET[:2]}/{_XET}")

    _assert_refused(LocalTarget(snapshot))


@pytest.mark.parametrize("form", ["dir-link", "dotdot", "absolute"])
def test_hf_shared_blob_requested_path_must_stay_in_root(tmp_path: Path, form: str) -> None:
    # Bug caught: the shared-store endpoint skips the lexical-parent rule, so a directory
    # link into the store, or a ``..``/absolute request, reads any store entry.
    snapshot = _shared_snapshot(tmp_path)
    entry = _hub(snapshot) / "blobs" / _XET[:2] / _XET
    entry.parent.mkdir(exist_ok=True)
    entry.write_bytes(_SECRET)
    (snapshot / "sub").symlink_to(f"../../../blobs/{_XET[:2]}", target_is_directory=True)
    request = {
        "dir-link": f"sub/{_XET}",
        "dotdot": f"../../../blobs/{_XET[:2]}/{_XET}",
        "absolute": str(entry),
    }[form]

    target = LocalTarget(snapshot)

    with pytest.raises(TargetError, match="outside"):
        target.read_text(request)
    with pytest.raises(TargetError, match="outside"):
        target.exists(request)
    assert not any(name.startswith("sub/") for name in target.list_files())


@pytest.mark.parametrize("commit", ["main", "A" * 40], ids=["ref-name", "uppercase"])
def test_hf_shared_blob_store_needs_a_snapshot_root(tmp_path: Path, commit: str) -> None:
    # Bug caught: the shared-store endpoint is enabled for any root three levels below a
    # marked store, not only for a Hugging Face cache snapshot.
    snapshot = write_hf_cache_snapshot(
        tmp_path / "hub",
        "org/model",
        {"config.json": _CONFIG},
        commit=commit,
        shared_blobs=True,
        store_files=["config.json"],
    )

    _assert_refused(LocalTarget(snapshot), "config.json")


def test_hf_shared_blob_missing_store_entry_is_refused(tmp_path: Path) -> None:
    # Bug caught: a store entry that cannot be lstat-ed (garbage-collected, or never
    # published) passes the guard because the OSError maps to "accepted"; or the refusal
    # says "outside the model root", which reads like a security refusal.
    snapshot = _shared_snapshot(tmp_path)
    (_hub(snapshot) / "blobs" / _XET[:2]).mkdir()
    _plant(snapshot, f"../../blobs/{_XET[:2]}/{_XET}")

    target = LocalTarget(snapshot)

    missing = "secret.json is missing from the Hugging Face cache store"
    with pytest.raises(TargetError, match=missing):
        target.read_text("secret.json")
    with pytest.raises(TargetError, match=missing):
        target.exists("secret.json")
    with pytest.raises(TargetError, match=missing):
        target.size("secret.json")
    assert "secret.json" not in target.list_files()


@pytest.mark.parametrize("form", ["dotdot", "absolute"])
def test_hf_shared_blob_missing_store_entry_requested_from_outside_stays_outside(
    tmp_path: Path, form: str
) -> None:
    # Bug caught: the missing-store-entry message skips the lexical-parent rule, so a
    # ``..`` or absolute request for a store path gets it instead of the outside refusal.
    snapshot = _shared_snapshot(tmp_path)
    entry = _hub(snapshot) / "blobs" / _XET[:2] / _XET
    entry.parent.mkdir(exist_ok=True)
    request = f"../../../blobs/{_XET[:2]}/{_XET}" if form == "dotdot" else str(entry)

    target = LocalTarget(snapshot)

    with pytest.raises(TargetError, match="outside the model root"):
        target.read_text(request)
    with pytest.raises(TargetError, match="outside the model root"):
        target.exists(request)


def test_hf_shared_blob_snapshot_link_straight_to_the_store_is_accepted(tmp_path: Path) -> None:
    """Regression fence (green before and after the regular-file rule).

    Bug caught: the second hop is pinned to a link from this repo's own ``blobs/``, so a
    snapshot link straight to a store entry (the same content-addressed, cache-wide
    file) is refused.
    """
    snapshot = _shared_snapshot(tmp_path)
    entry = _hub(snapshot) / "blobs" / _XET[:2] / _XET
    entry.parent.mkdir(exist_ok=True)
    entry.write_bytes(_SECRET)
    (snapshot / "direct.json").symlink_to(f"../../../blobs/{_XET[:2]}/{_XET}")

    target = LocalTarget(snapshot)

    assert target.exists("direct.json")
    assert target.read_text("direct.json") == _SECRET.decode()
    assert "direct.json" in target.list_files()


def test_hf_snapshot_opened_through_a_symlinked_hub_reads_both_layouts(tmp_path: Path) -> None:
    """Regression fence (green before and after the regular-file rule).

    Bug caught: ``blobs/`` or the shared store is located from the root as given rather
    than the resolved root, so a cache reached through a linked folder loses both the
    one-hop and the second-hop allowance.
    """
    weights = b"\x00" * 24
    snapshot = write_hf_cache_snapshot(
        tmp_path / "hub",
        "org/model",
        {"config.json": _CONFIG, "model.safetensors": weights},
        shared_blobs=True,
    )
    repo_blobs = snapshot.parent.parent / "blobs"
    assert not (repo_blobs / _git_sha1(_CONFIG)).is_symlink()  # one hop
    assert (repo_blobs / _sha256(weights)).is_symlink()  # two hops
    (tmp_path / "linkhub").symlink_to(tmp_path / "hub", target_is_directory=True)

    target = LocalTarget(tmp_path / "linkhub" / snapshot.relative_to(tmp_path / "hub"))

    assert target.read_text("config.json") == _CONFIG.decode()
    assert target.read_bytes("model.safetensors") == weights
    assert target.list_files() == ("config.json", "model.safetensors")
