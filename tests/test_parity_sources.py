"""Source resolution: pin each of base/adapter/fused to a local snapshot once (F10).

The default lane is offline: a local directory resolves to itself with no
downloader call, and a Hugging Face id resolves through an injected fake
downloader (so ``assert_no_network`` holds). The real ``snapshot_download``
path is exercised only under the ``network`` marker.
"""

from pathlib import Path

import pytest

from mlx_model_doctor.errors import ModelDoctorError
from mlx_model_doctor.parity.sources import ResolvedSources, resolve_sources


class RecordingDownloader:
    """Fake snapshot downloader: maps each repo id to a fresh local dir, counting calls."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self.calls: list[str] = []

    def snapshot(self, repo_id: str) -> str:
        self.calls.append(repo_id)
        directory = self._root / repo_id.replace("/", "__")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text("{}", encoding="utf-8")
        return str(directory)


class ForbiddenDownloader:
    """Downloader whose ``snapshot`` must never be called on the local pass-through path."""

    def snapshot(self, repo_id: str) -> str:
        raise AssertionError(f"snapshot must not be called for a local path: {repo_id!r}")


def _make_dir(tmp_path: Path, name: str) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    return directory


def test_local_paths_pass_through_without_a_download(tmp_path: Path) -> None:
    """Local directories resolve to themselves; the downloader is never touched.

    A regression that snapshot-downloaded a local path would trip
    ``ForbiddenDownloader``; one that dropped ``source="local"`` would fail the
    source assertions.
    """
    base = _make_dir(tmp_path, "base")
    adapter = _make_dir(tmp_path, "adapter")
    fused = _make_dir(tmp_path, "fused")

    sources = resolve_sources(
        str(base),
        str(adapter),
        str(fused),
        allow_network=False,
        downloader=ForbiddenDownloader(),
    )

    assert isinstance(sources, ResolvedSources)
    assert sources.base.source == "local"
    assert sources.base.path == str(base.resolve())
    assert sources.base.original_ref == str(base)
    assert sources.adapter.source == "local"
    assert sources.adapter.path == str(adapter.resolve())
    assert sources.fused.source == "local"
    assert sources.fused.path == str(fused.resolve())


def test_hf_id_resolves_to_a_pinned_snapshot_exactly_once(tmp_path: Path) -> None:
    """A Hugging Face id snapshots once to a pinned local dir; locals pass through.

    Asserts the single-resolution contract (F10): the HF base is downloaded
    exactly once, so the base/base-repeat/reference loads that share it never
    re-resolve a drifting default branch.
    """
    downloader = RecordingDownloader(tmp_path / "snapshots")
    adapter = _make_dir(tmp_path, "adapter")
    fused = _make_dir(tmp_path, "fused")

    sources = resolve_sources(
        "org/base-model",
        str(adapter),
        str(fused),
        allow_network=True,
        downloader=downloader,
    )

    assert downloader.calls == ["org/base-model"]
    assert sources.base.source == "hf"
    assert sources.base.original_ref == "org/base-model"
    assert Path(sources.base.path).is_dir()
    assert sources.adapter.source == "local"
    assert sources.fused.source == "local"


def test_hf_adapter_id_resolves_to_a_local_directory(tmp_path: Path) -> None:
    """An HF adapter id resolves to a local dir (``load_adapters`` needs a local path)."""
    downloader = RecordingDownloader(tmp_path / "snapshots")
    base = _make_dir(tmp_path, "base")
    fused = _make_dir(tmp_path, "fused")

    sources = resolve_sources(
        str(base),
        "org/my-adapter",
        str(fused),
        allow_network=True,
        downloader=downloader,
    )

    assert downloader.calls == ["org/my-adapter"]
    assert sources.adapter.source == "hf"
    assert sources.adapter.original_ref == "org/my-adapter"
    assert Path(sources.adapter.path).is_dir()


def test_hf_source_without_network_raises_a_human_facing_download_hint(tmp_path: Path) -> None:
    """A Hugging Face id with ``allow_network=False`` is a setup failure, not a silent
    download -- and the message must tell the user how to fix it (download it and pass
    a local path) without leaking the internal ``allow_network``/test-marker seam.
    """
    adapter = _make_dir(tmp_path, "adapter")
    fused = _make_dir(tmp_path, "fused")

    with pytest.raises(ModelDoctorError) as exc_info:
        resolve_sources(
            "org/base-model",
            str(adapter),
            str(fused),
            allow_network=False,
            downloader=ForbiddenDownloader(),
        )

    message = str(exc_info.value)
    assert "hf download org/base-model --local-dir" in message
    assert "network marker" not in message
    assert "in tests" not in message


@pytest.mark.network
def test_real_hf_snapshot_resolution(tmp_path: Path) -> None:
    """The real ``snapshot_download`` path resolves an HF id to a local snapshot.

    Network-gated (``--run-network``); only ``base`` is an HF id so a single small
    model is snapshotted.
    """
    adapter = _make_dir(tmp_path, "adapter")
    fused = _make_dir(tmp_path, "fused")

    sources = resolve_sources(
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
        str(adapter),
        str(fused),
        allow_network=True,
    )

    assert sources.base.source == "hf"
    snapshot = Path(sources.base.path)
    assert snapshot.is_dir()
    assert (snapshot / "config.json").is_file()


def test_default_downloader_never_fetches_repository_python_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Bug: the parity snapshot downloads the whole repository, "*.py" included, so a
    # repository's code lands on disk beside the files the worker loads.
    from fnmatch import fnmatch

    import huggingface_hub

    from mlx_model_doctor.parity.sources import DefaultSnapshotDownloader

    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return "/cache/models--org--base/snapshots/" + "0" * 40

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    path = DefaultSnapshotDownloader().snapshot("org/base")

    assert path == "/cache/models--org--base/snapshots/" + "0" * 40
    (call,) = calls
    assert call["repo_id"] == "org/base"
    ignored = call.get("ignore_patterns") or []
    allowed = call.get("allow_patterns")
    assert isinstance(ignored, list)

    def fetched(name: str) -> bool:
        if any(fnmatch(name, p) for p in ignored):
            return False
        return allowed is None or any(fnmatch(name, p) for p in allowed)  # type: ignore[union-attr]

    assert not fetched("modeling_x.py")
    for name in ("config.json", "model.safetensors", "adapters.safetensors", "tokenizer.json"):
        assert fetched(name), name


def test_default_downloader_fetches_only_the_loader_and_adapter_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Bug: the parity snapshot uses a deny list (ignore "*.py"), so compiled modules,
    # shared libraries and any other file the repository ships are fetched; or the
    # allow list drops a file the parity load needs (tokenizer.model, *.jinja, the
    # adapter weights).
    import huggingface_hub

    from mlx_model_doctor.parity.sources import DefaultSnapshotDownloader

    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return "/cache/models--org--base/snapshots/" + "0" * 40

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    DefaultSnapshotDownloader().snapshot("org/base")

    # mlx-lm 0.31.3 utils._download's patterns minus "*.py", plus the two files
    # mlx-lm 0.31.3 tuner/utils.py load_adapters reads.
    assert calls == [
        {
            "repo_id": "org/base",
            "allow_patterns": [
                "*.json",
                "model*.safetensors",
                "tokenizer.model",
                "*.tiktoken",
                "tiktoken.model",
                "*.txt",
                "*.jsonl",
                "*.jinja",
                "adapter_config.json",
                "adapters.safetensors",
            ],
        }
    ]


def test_default_downloader_failure_is_a_target_error_without_cache_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: a failing parity snapshot download escapes resolve_sources as a raw OSError
    # (a Python traceback and exit 1, not a tool error), with the Hugging Face cache
    # path and the user's home in its text.
    import huggingface_hub

    from mlx_model_doctor.errors import TargetError
    from tests.fakes import use_fake_home

    home = tmp_path / "Users" / "John Smith"
    use_fake_home(monkeypatch, home=str(home))
    lock = home / ".cache" / "huggingface" / "hub" / ".locks" / "models--o--n" / "a.lock"
    error = OSError(f"No space left: '{lock}'")

    def failing_snapshot_download(**_kwargs: object) -> str:
        raise error

    monkeypatch.setattr(huggingface_hub, "snapshot_download", failing_snapshot_download)

    with pytest.raises(TargetError) as exc_info:
        resolve_sources("o/n", "o/adapter", "o/fused", allow_network=True)

    assert str(exc_info.value) == (
        "Could not download a snapshot of Hugging Face model o/n: No space left: 'a.lock'"
    )
    assert exc_info.value.target == "o/n"
    assert exc_info.value.source == "hf"
    assert exc_info.value.__cause__ is error


def test_default_downloader_failure_that_is_not_an_os_error_is_a_target_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: the download handler catches only OSError, so a repo id the Hub rejects
    # (HFValidationError, a ValueError) escapes raw with the user's home in its text.
    import huggingface_hub
    from huggingface_hub.errors import HFValidationError

    from mlx_model_doctor.errors import TargetError
    from tests.fakes import use_fake_home

    assert issubclass(HFValidationError, ValueError)
    assert not issubclass(HFValidationError, OSError)
    home = tmp_path / "Users" / "John Smith"
    use_fake_home(monkeypatch, home=str(home))
    error = HFValidationError(f"Cannot have -- or .. in repo_id: '{home}/m..x/n'.")

    def failing_snapshot_download(**_kwargs: object) -> str:
        raise error

    monkeypatch.setattr(huggingface_hub, "snapshot_download", failing_snapshot_download)

    with pytest.raises(TargetError) as exc_info:
        resolve_sources("o/n", "o/adapter", "o/fused", allow_network=True)

    assert str(exc_info.value) == (
        "Could not download a snapshot of Hugging Face model o/n: "
        "Cannot have -- or .. in repo_id: 'n'."
    )
    assert exc_info.value.target == "o/n"
    assert exc_info.value.source == "hf"
    assert exc_info.value.__cause__ is error
