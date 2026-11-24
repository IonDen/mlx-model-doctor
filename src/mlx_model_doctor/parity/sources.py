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

from mlx_model_doctor.errors import ModelDoctorError, TargetError
from mlx_model_doctor.parity.report import ResolvedIdentity
from mlx_model_doctor.redaction import redact_local_paths

# The text smoke list (mlx-lm 0.31.3 utils._download minus "*.py") plus the two files
# mlx-lm 0.31.3 tuner/utils.py load_adapters reads from an adapter folder.
_PARITY_SNAPSHOT_PATTERNS = (
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
)


class SnapshotDownloader(Protocol):
    """Boundary for resolving a Hugging Face repo id to a local snapshot directory."""

    def snapshot(self, repo_id: str) -> str:
        """Download ``repo_id`` to a pinned local directory and return its path."""


class DefaultSnapshotDownloader:
    """Real :class:`SnapshotDownloader` backed by ``huggingface_hub.snapshot_download``."""

    def snapshot(self, repo_id: str) -> str:
        """Snapshot-download the files a parity load reads and return the folder."""
        from huggingface_hub import snapshot_download

        # allow_patterns: huggingface_hub 1.19.0 snapshot_download keyword. The same
        # downloader fetches base, fused and adapter repositories, so the list is the
        # mlx-lm one plus the adapter files. It does not download .py files.
        try:
            return snapshot_download(
                repo_id=repo_id, allow_patterns=list(_PARITY_SNAPSHOT_PATTERNS)
            )
        except Exception as exc:
            raise TargetError(
                f"Could not download a snapshot of Hugging Face model {repo_id}: "
                f"{redact_local_paths(str(exc))}",
                target=repo_id,
                source="hf",
            ) from exc


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedSources:
    """The three pinned model identities a parity run compares (F10)."""

    base: ResolvedIdentity
    adapter: ResolvedIdentity
    fused: ResolvedIdentity


def _local_dir_hint(ref: str) -> str:
    """Return a plausible local directory name to suggest for ``hf download --local-dir``."""
    return Path(ref).name or ref


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
            f"cannot use '{ref}': it is not a local directory. Download it first (e.g. "
            f"`hf download {ref} --local-dir ./{_local_dir_hint(ref)}`) and pass that path. "
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
