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


def test_hf_source_without_network_raises(tmp_path: Path) -> None:
    """A Hugging Face id with ``allow_network=False`` is a setup failure, not a silent download."""
    adapter = _make_dir(tmp_path, "adapter")
    fused = _make_dir(tmp_path, "fused")

    with pytest.raises(ModelDoctorError, match="network"):
        resolve_sources(
            "org/base-model",
            str(adapter),
            str(fused),
            allow_network=False,
            downloader=ForbiddenDownloader(),
        )


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
