"""One-shot source resolution to pinned local snapshots (F10).

Each of base / adapter / fused is resolved **once**, up front, to a pinned local
directory the four model loads share: a local path is used as-is; a Hugging Face
repo id is snapshot-downloaded (a pinned commit) to a local directory through the
:class:`SnapshotDownloader` seam. Resolving up front is what stops the comparison
from measuring a drifting default-branch checkpoint (the same base snapshot backs
the base, base-repeat, and base+adapter reference loads), and it also gives
``mlx_lm.load_adapters`` -- which requires a local directory -- a real path for an
HF adapter id.

The Hugging Face download is deliberately behind ``allow_network``: with it off,
an id that is not an existing local directory is a setup failure
(:class:`~mlx_model_doctor.errors.ModelDoctorError`), never a silent network
fetch, so the default (offline) test lane cannot accidentally reach the Hub.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from mlx_model_doctor.errors import ModelDoctorError
from mlx_model_doctor.parity.report import ResolvedIdentity


class SnapshotDownloader(Protocol):
    """Boundary for resolving a Hugging Face repo id to a local snapshot directory."""

    def snapshot(self, repo_id: str) -> str:
        """Download ``repo_id`` to a pinned local directory and return its path."""


class DefaultSnapshotDownloader:
    """Real :class:`SnapshotDownloader` backed by ``huggingface_hub.snapshot_download``."""

    def snapshot(self, repo_id: str) -> str:
        """Snapshot-download a full repository and return its local cache path."""
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=repo_id)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedSources:
    """The three pinned model identities a parity run compares (F10)."""

    base: ResolvedIdentity
    adapter: ResolvedIdentity
    fused: ResolvedIdentity


def _resolve_one(
    ref: str, *, allow_network: bool, downloader: SnapshotDownloader
) -> ResolvedIdentity:
    """Resolve one reference to a pinned local snapshot identity.

    An existing local directory is used as-is (``source="local"``); anything else
    is treated as a Hugging Face repo id and snapshot-downloaded when
    ``allow_network`` is set (``source="hf"``), or rejected as a setup failure.
    """
    local_dir = Path(ref).expanduser()
    if local_dir.is_dir():
        return ResolvedIdentity(path=str(local_dir.resolve()), original_ref=ref, source="local")
    if not allow_network:
        raise ModelDoctorError(
            f"cannot resolve source {ref!r}: it is not an existing local directory, and "
            "resolving it as a Hugging Face repo id requires network access "
            "(pass allow_network=True; downloads are gated by the 'network' marker in tests). "
            "If this was meant to be a local path, check it for a typo."
        )
    snapshot_path = downloader.snapshot(ref)
    return ResolvedIdentity(path=snapshot_path, original_ref=ref, source="hf")


def resolve_sources(
    base: str,
    adapter: str,
    fused: str,
    *,
    allow_network: bool,
    downloader: SnapshotDownloader | None = None,
) -> ResolvedSources:
    """Resolve base / adapter / fused each once to a pinned local snapshot (F10).

    Local directories pass through unchanged; Hugging Face repo ids are
    snapshot-downloaded exactly once each through ``downloader`` (defaulting to
    :class:`DefaultSnapshotDownloader`), gated by ``allow_network``. Raises
    :class:`~mlx_model_doctor.errors.ModelDoctorError` for a non-local reference
    when ``allow_network`` is ``False``.
    """
    resolver = downloader if downloader is not None else DefaultSnapshotDownloader()
    return ResolvedSources(
        base=_resolve_one(base, allow_network=allow_network, downloader=resolver),
        adapter=_resolve_one(adapter, allow_network=allow_network, downloader=resolver),
        fused=_resolve_one(fused, allow_network=allow_network, downloader=resolver),
    )
