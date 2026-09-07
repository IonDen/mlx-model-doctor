"""Tests for the serial parity orchestrator + parent-supervised subprocess launcher (F2/F9)."""

import sys
import time
from pathlib import Path

import mlx_model_doctor.parity.orchestrator as orchestrator_module
from mlx_model_doctor.parity.orchestrator import (
    SubprocessLauncher,
    WorkerOutcome,
    run_parity_workers,
)
from mlx_model_doctor.parity.worker import WorkerSpec
from tests.parity_fakes import (
    RaisingLauncher,
    TrackingLauncher,
    write_exit_nonzero_stub,
    write_malformed_output_stub,
    write_missing_output_stub,
    write_nonresponsive_stub,
    write_slow_cooperative_stub,
    write_success_stub,
    write_wrong_length_stub,
)


def _spec(**overrides: object) -> WorkerSpec:
    defaults: dict[str, object] = {
        "model_path": "/models/base",
        "adapter_path": None,
        "token_ids": (1, 2, 3),
        "fixture_id": "fx-1",
        "role": "base",
    }
    defaults.update(overrides)
    return WorkerSpec(**defaults)  # type: ignore[arg-type]


def _outcome(**overrides: object) -> WorkerOutcome:
    defaults: dict[str, object] = {
        "role": "base",
        "argmax": [1, 2, 3],
        "adapter_applied": None,
        "peak_bytes": 100,
        "status": "ok",
        "error": None,
    }
    defaults.update(overrides)
    return WorkerOutcome(**defaults)  # type: ignore[arg-type]


# --- WorkerOutcome ----------------------------------------------------------------


def test_worker_outcome_holds_the_expected_fields() -> None:
    outcome = _outcome(
        role="adapter", argmax=[9], adapter_applied=True, peak_bytes=42, status="ok", error=None
    )
    assert outcome.role == "adapter"
    assert outcome.argmax == [9]
    assert outcome.adapter_applied is True
    assert outcome.peak_bytes == 42
    assert outcome.status == "ok"
    assert outcome.error is None


def test_worker_outcome_supports_error_with_null_result_fields() -> None:
    outcome = _outcome(
        role="fused",
        argmax=None,
        adapter_applied=None,
        peak_bytes=None,
        status="error",
        error="boom",
    )
    assert outcome.status == "error"
    assert outcome.argmax is None
    assert outcome.error == "boom"


# --- run_parity_workers: strict serialization + crash isolation -------------------


def test_run_parity_workers_calls_launcher_once_per_spec_in_order() -> None:
    specs = [_spec(role="base"), _spec(role="adapter"), _spec(role="fused")]
    outcomes = [_outcome(role=s.role) for s in specs]  # type: ignore[attr-defined]
    launcher = TrackingLauncher(outcomes)

    result = run_parity_workers(specs, launcher=launcher)

    assert result == outcomes
    assert launcher.calls == specs


def test_run_parity_workers_runs_strictly_serially() -> None:
    """A mutant that ran specs concurrently (e.g. a thread pool) must fail this test.

    ``TrackingLauncher`` sleeps briefly inside ``run_one`` (see ``tests/parity_fakes.py``)
    so an overlapping call is actually observable as ``active > 1`` — without that delay
    the GIL makes the increment/decrement window too narrow for a thread-pool mutant to
    ever be caught here.
    """
    specs = [_spec(role=f"role-{i}") for i in range(4)]
    outcomes = [_outcome(role=s.role) for s in specs]  # type: ignore[attr-defined]
    launcher = TrackingLauncher(outcomes)

    run_parity_workers(specs, launcher=launcher)

    assert launcher.max_concurrent == 1


def test_run_parity_workers_returns_empty_list_for_no_specs() -> None:
    launcher = TrackingLauncher([])

    assert run_parity_workers([], launcher=launcher) == []
    assert launcher.calls == []


def test_run_parity_workers_isolates_a_launcher_crash_from_other_specs() -> None:
    """One worker's launcher call raising must not abort the remaining specs (F2)."""
    specs = [_spec(role="base"), _spec(role="broken"), _spec(role="fused")]
    outcomes_by_role = {
        "base": _outcome(role="base"),
        "fused": _outcome(role="fused"),
    }
    launcher = RaisingLauncher(outcomes_by_role, raising_roles=["broken"])

    result = run_parity_workers(specs, launcher=launcher)

    assert len(result) == 3
    assert result[0] == outcomes_by_role["base"]
    assert result[2] == outcomes_by_role["fused"]
    broken = result[1]
    assert broken.status == "error"
    assert broken.role == "broken"
    assert broken.error is not None
    assert "broken" in broken.error
    assert launcher.calls == ["base", "broken", "fused"]  # the crash didn't stop the loop


# --- _build_worker_argv (pure helper) ----------------------------------------------


def test_build_worker_argv_omits_adapter_path_when_none() -> None:
    argv = orchestrator_module._build_worker_argv(
        _spec(adapter_path=None, token_ids=(1, 2)), Path("/tmp/out.json")
    )

    assert "--adapter-path" not in argv
    assert argv == [
        "--model-path",
        "/models/base",
        "--token-ids",
        "1,2",
        "--fixture-id",
        "fx-1",
        "--role",
        "base",
        "--out",
        "/tmp/out.json",
    ]


def test_build_worker_argv_includes_adapter_path_when_set() -> None:
    argv = orchestrator_module._build_worker_argv(
        _spec(adapter_path="/adapters/a"), Path("/tmp/out.json")
    )

    assert "--adapter-path" in argv
    assert argv[argv.index("--adapter-path") + 1] == "/adapters/a"


# --- SubprocessLauncher: defaults ---------------------------------------------------


def test_subprocess_launcher_default_argv_prefix_targets_the_real_worker_module() -> None:
    launcher = SubprocessLauncher(vocab_size=10)

    assert launcher.argv_prefix == (sys.executable, "-m", "mlx_model_doctor.parity.worker")


# --- SubprocessLauncher: offline stub-script scenarios ------------------------------


def test_subprocess_launcher_success_stub_returns_ok_outcome(tmp_path: Path) -> None:
    stub = write_success_stub(tmp_path / "stub.py")
    launcher = SubprocessLauncher(
        vocab_size=10, timeout_s=10.0, argv_prefix=(sys.executable, str(stub))
    )

    outcome = launcher.run_one(_spec(token_ids=(1, 2, 3), fixture_id="fx-1", role="base"))

    assert outcome.status == "ok"
    assert outcome.error is None
    assert outcome.argmax == [0, 0, 0]
    assert outcome.peak_bytes == 123
    assert outcome.adapter_applied is None
    assert outcome.role == "base"


def test_subprocess_launcher_nonzero_exit_returns_error(tmp_path: Path) -> None:
    stub = write_exit_nonzero_stub(tmp_path / "stub.py")
    launcher = SubprocessLauncher(
        vocab_size=10, timeout_s=10.0, argv_prefix=(sys.executable, str(stub))
    )

    outcome = launcher.run_one(_spec())

    assert outcome.status == "error"
    assert outcome.error is not None
    assert "7" in outcome.error


def test_subprocess_launcher_missing_output_file_returns_error(tmp_path: Path) -> None:
    """A worker that claims success (sentinel + exit 0) but never wrote its artifact is an error."""
    stub = write_missing_output_stub(tmp_path / "stub.py")
    launcher = SubprocessLauncher(
        vocab_size=10, timeout_s=10.0, argv_prefix=(sys.executable, str(stub))
    )

    outcome = launcher.run_one(_spec())

    assert outcome.status == "error"
    assert outcome.error is not None


def test_subprocess_launcher_malformed_output_file_returns_error(tmp_path: Path) -> None:
    stub = write_malformed_output_stub(tmp_path / "stub.py")
    launcher = SubprocessLauncher(
        vocab_size=10, timeout_s=10.0, argv_prefix=(sys.executable, str(stub))
    )

    outcome = launcher.run_one(_spec())

    assert outcome.status == "error"
    assert outcome.error is not None


def test_subprocess_launcher_wrong_length_artifact_returns_error(tmp_path: Path) -> None:
    """Artifact validation is applied on read (F9): a wrong-length argmax is an error."""
    stub = write_wrong_length_stub(tmp_path / "stub.py")
    launcher = SubprocessLauncher(
        vocab_size=10, timeout_s=10.0, argv_prefix=(sys.executable, str(stub))
    )

    outcome = launcher.run_one(_spec(token_ids=(1, 2, 3)))  # stub always writes a length-1 argmax

    assert outcome.status == "error"
    assert outcome.error is not None


def test_subprocess_launcher_missing_sentinel_is_an_error(tmp_path: Path) -> None:
    """A mutant dropping the sentinel check (trusting returncode 0 alone) must fail this."""
    stub = tmp_path / "stub.py"
    stub.write_text(
        """
import json
import sys

out_path = sys.argv[sys.argv.index("--out") + 1]
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "fixture_id": "fx-1",
            "role": "base",
            "argmax": [0, 0, 0],
            "peak_bytes": 1,
            "adapter_applied": None,
        },
        handle,
    )
sys.exit(0)
""",
        encoding="utf-8",
    )
    launcher = SubprocessLauncher(
        vocab_size=10, timeout_s=10.0, argv_prefix=(sys.executable, str(stub))
    )

    outcome = launcher.run_one(_spec())

    assert outcome.status == "error"
    assert outcome.error is not None


def test_subprocess_launcher_launch_failure_returns_error() -> None:
    launcher = SubprocessLauncher(
        vocab_size=10, timeout_s=5.0, argv_prefix=("/no/such/executable-xyz",)
    )

    outcome = launcher.run_one(_spec())

    assert outcome.status == "error"
    assert outcome.error is not None


# --- SubprocessLauncher: parent-owned bounded termination (F2) ---------------------


def test_subprocess_launcher_terminates_a_nonresponsive_worker_within_the_bound(
    tmp_path: Path,
) -> None:
    """A worker that ignores SIGTERM and sleeps must still be reaped within ~timeout (F2)."""
    stub = write_nonresponsive_stub(tmp_path / "stub.py")
    launcher = SubprocessLauncher(
        vocab_size=10,
        timeout_s=0.3,
        terminate_grace_s=0.3,
        argv_prefix=(sys.executable, str(stub)),
    )

    start = time.monotonic()
    outcome = launcher.run_one(_spec())
    elapsed = time.monotonic() - start

    assert outcome.status == "error"
    assert outcome.error is not None
    # Generous margin over timeout_s + terminate_grace_s for CI scheduling jitter; the
    # stub sleeps 30s, so any bound well under that proves it was actually killed.
    assert elapsed < 10.0


def test_subprocess_launcher_reaps_a_slow_but_cooperative_worker_via_term_alone(
    tmp_path: Path,
) -> None:
    """A worker that responds to plain SIGTERM must be reaped without ever needing KILL."""
    stub = write_slow_cooperative_stub(tmp_path / "stub.py")
    launcher = SubprocessLauncher(
        vocab_size=10,
        timeout_s=0.2,
        terminate_grace_s=5.0,
        argv_prefix=(sys.executable, str(stub)),
    )

    start = time.monotonic()
    outcome = launcher.run_one(_spec())
    elapsed = time.monotonic() - start

    assert outcome.status == "error"
    assert outcome.error is not None
    # The stub sleeps 30s and honors default SIGTERM handling, so TERM alone (well
    # within the 5s grace window) must reap it — a much smaller bound than the sleep.
    assert elapsed < 10.0
