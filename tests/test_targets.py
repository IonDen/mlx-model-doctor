import json
import struct
from pathlib import Path

import pytest

from mlx_model_doctor.errors import TargetError
from mlx_model_doctor.safetensors_header import SafetensorsHeader
from mlx_model_doctor.targets import LocalTarget


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
