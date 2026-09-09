"""Serial parity worker orchestration + parent-supervised subprocess launcher (F2/F9)."""

import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from mlx_model_doctor.errors import WorkerArtifactError
from mlx_model_doctor.parity.worker import WorkerSpec, read_worker_json

# The exact prefix ``worker.main`` prints on success (see ``worker.py``'s ``main``).
_SENTINEL_PREFIX = "::PARITY_WORKER::ok"

_DEFAULT_TIMEOUT_S = 330.0
_DEFAULT_TERMINATE_GRACE_S = 5.0


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkerOutcome:
    """The observable outcome of running one ``WorkerSpec`` to completion (F2/F9).

    ``status`` is ``"ok"`` (a validated result was read back), ``"error"`` (the
    subprocess crashed, timed out, or its artifact failed validation), or
    ``"skipped"`` (the spec was never attempted). ``argmax``/``adapter_applied``/
    ``peak_bytes`` are null whenever ``status`` is not ``"ok"``.
    """

    role: str
    argmax: list[int] | None
    adapter_applied: bool | None
    peak_bytes: int | None
    status: Literal["ok", "error", "skipped"]
    error: str | None


def _error_outcome(role: str, message: str) -> WorkerOutcome:
    """Build an ``error`` outcome for ``role`` with no result fields."""
    return WorkerOutcome(
        role=role,
        argmax=None,
        adapter_applied=None,
        peak_bytes=None,
        status="error",
        error=message,
    )


class Launcher(Protocol):
    """Boundary for running a single worker spec to completion."""

    def run_one(self, spec: WorkerSpec) -> WorkerOutcome:
        """Run ``spec`` to completion and return its outcome."""


def run_parity_workers(specs: Sequence[WorkerSpec], *, launcher: Launcher) -> list[WorkerOutcome]:
    """Run each spec strictly one at a time, isolating a launcher crash (F2).

    Specs are run serially: the next spec is never started before the previous
    one's ``launcher.run_one`` call has returned, so at most one model is ever
    resident. An exception raised by ``launcher.run_one`` itself (a launcher bug,
    distinct from a worker-reported failure, which a well-behaved ``Launcher``
    already folds into an ``error`` outcome) is caught and turned into an
    ``error`` outcome so it can never abort the remaining specs.
    """
    outcomes: list[WorkerOutcome] = []
    for spec in specs:
        try:
            outcome = launcher.run_one(spec)
        except Exception as exc:
            outcome = _error_outcome(spec.role, f"launcher crashed: {type(exc).__name__}: {exc}")
        outcomes.append(outcome)
    return outcomes


def _default_argv_prefix() -> tuple[str, ...]:
    """Return the real worker's spawn command: ``python -m mlx_model_doctor.parity.worker``."""
    return (sys.executable, "-m", "mlx_model_doctor.parity.worker")


def _build_worker_argv(spec: WorkerSpec, out_path: Path) -> list[str]:
    """Build the ``--flag value`` argv the worker's argument parser expects (see ``worker.main``).

    Serializes ``spec.token_ids`` (one or more sequences) as ``;``-separated
    groups of ``,``-separated integers -- the exact format
    ``worker._parse_token_id_sequences`` parses back.
    """
    argv = ["--model-path", spec.model_path]
    if spec.adapter_path is not None:
        argv += ["--adapter-path", spec.adapter_path]
    argv += [
        "--token-ids",
        ";".join(",".join(str(token_id) for token_id in sequence) for sequence in spec.token_ids),
        "--fixture-id",
        spec.fixture_id,
        "--role",
        spec.role,
        "--out",
        str(out_path),
    ]
    return argv


@dataclass(frozen=True, slots=True, kw_only=True)
class SubprocessLauncher:
    """Parent-supervised ``Launcher`` that spawns the real worker as a subprocess (F2/F9).

    Owns bounded termination: a worker that does not exit within ``timeout_s`` is
    ``terminate()``-d, given ``terminate_grace_s`` to exit cooperatively, then
    ``kill()``-ed and reaped — so a wedged or nonresponsive worker can never block
    the orchestrator past that bound. ``argv_prefix`` defaults to the real worker
    entry point and is overridable so tests can spawn a trivial stub script instead
    (no MLX in the default test lane).
    """

    vocab_size: int
    timeout_s: float = _DEFAULT_TIMEOUT_S
    terminate_grace_s: float = _DEFAULT_TERMINATE_GRACE_S
    argv_prefix: tuple[str, ...] = field(default_factory=_default_argv_prefix)

    def run_one(self, spec: WorkerSpec) -> WorkerOutcome:
        """Spawn the worker for ``spec``, wait bounded, and validate its result artifact."""
        with tempfile.TemporaryDirectory(prefix="mlx-model-doctor-parity-") as tmp_dir:
            out_path = Path(tmp_dir) / "result.json"
            argv = [*self.argv_prefix, *_build_worker_argv(spec, out_path)]

            try:
                process = subprocess.Popen(
                    argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
            except OSError as exc:
                return _error_outcome(spec.role, f"failed to launch worker subprocess: {exc}")

            try:
                stdout, stderr = process.communicate(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                self._terminate_and_reap(process)
                return _error_outcome(
                    spec.role,
                    f"worker subprocess for role {spec.role!r} did not exit within "
                    f"{self.timeout_s}s and was terminated",
                )

            if process.returncode != 0:
                return _error_outcome(
                    spec.role,
                    f"worker subprocess for role {spec.role!r} exited with code "
                    f"{process.returncode}: {stderr.strip()}",
                )
            if _SENTINEL_PREFIX not in stdout:
                return _error_outcome(
                    spec.role,
                    f"worker subprocess for role {spec.role!r} exited 0 without reporting "
                    f"success (missing {_SENTINEL_PREFIX!r} sentinel); stdout={stdout!r}",
                )

            try:
                result = read_worker_json(
                    out_path,
                    expect_fixture=spec.fixture_id,
                    expect_role=spec.role,
                    vocab_size=self.vocab_size,
                    length=sum(len(sequence) for sequence in spec.token_ids),
                )
            except (WorkerArtifactError, ValueError) as exc:
                return _error_outcome(spec.role, str(exc))

            return WorkerOutcome(
                role=result.role,
                argmax=result.argmax,
                adapter_applied=result.adapter_applied,
                peak_bytes=result.peak_bytes,
                status="ok",
                error=None,
            )

    def _terminate_and_reap(self, process: "subprocess.Popen[str]") -> None:
        """Escalate TERM -> KILL against a bounded wait and reap the process either way (F2)."""
        process.terminate()
        try:
            process.communicate(timeout=self.terminate_grace_s)
            return
        except subprocess.TimeoutExpired:
            pass
        process.kill()
        process.communicate()
