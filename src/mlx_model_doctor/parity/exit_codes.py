"""Pure, total exit-code decision for adapter/fused-model parity reports (F4).

``parity_exit_code`` is deliberately separate from
:mod:`mlx_model_doctor.exit_codes` (``exit_code_for``/``exit_code_for_error``,
which stay untouched) -- the parity run has its own decision table over
:class:`~mlx_model_doctor.parity.report.ParityReport`, not a
:class:`~mlx_model_doctor.report.DoctorReport`.

The table is total and evaluated top-down, first match wins (spec Sec.3/Sec.9):

1. **Tool/setup error -> 2:** a crashed parity check, a malformed
   adapter-config/shape-mismatch fail, or a blocking embedded base/fused
   ``text`` report failure. (A missing ``[mlx-lm]`` extra also lands here in
   practice, via a worker reporting a dependency-import failure -- caught by
   step 3's worker-status check, which returns the same exit code.)
2. **Determined-bad static incompatibility -> 1:** a tokenizer mismatch or a
   statically-confirmed missing/uncovered LoRA target. These are exit 1 EVEN
   when ``verdict is None`` -- the oracle was deliberately void-skipped, not
   failed, so this step must be checked *before* step 3's null-verdict
   catch-all, not folded into it.
3. **Runtime cannot-determine -> 2:** any ``worker_status`` value of
   ``"error"``; ``adapter_applied`` is ``None``/``False``; ``verdict`` is
   ``INCONCLUSIVE``; or ``verdict is None`` for any reason not already
   covered by step 2.
4. **Determined-bad runtime -> 1:** the oracle verdict is
   ``FAIL_TRACKS_BASE``/``FAIL_GROSS``.
5. **Pass -> 0:** none of the above (in practice, ``verdict is PASS`` with no
   blocking failure).

No blocking defect or worker error can ever fall through to a pass, even
alongside a ``PASS`` verdict or a statically-confirmed defect that would
otherwise resolve to 1 on its own.
"""

from mlx_model_doctor.parity.oracle import ParityVerdict
from mlx_model_doctor.parity.report import ParityReport

# check_id values mirror the dataclass defaults declared in
# mlx_model_doctor.parity.checks (AdapterConfigCheck, TargetCoverageCheck,
# FusedTargetConsistencyCheck, TokenizerIdentityCheck). Duplicated here as
# literal keys -- rather than importing the check classes -- keeps this
# module's decision table a flat, readable set of check-id constants.
_MALFORMED_ADAPTER_CONFIG_CHECK_ID = "parity/adapter.config_well_formed"
_MISSING_TARGET_CHECK_IDS = frozenset({"parity/target.coverage", "parity/fused.target_consistency"})
_TOKENIZER_MISMATCH_CHECK_ID = "parity/tokenizer.identity"

# Mirrors mlx_model_doctor.parity.checks.run_parity_checks's crash-isolation
# message format (``message=f"check crashed: {exc}"``), so a crashed check --
# which could not even determine an answer -- can be told apart from an
# ordinary, confidently-reached ``fail``.
_CRASH_MESSAGE_PREFIX = "check crashed:"

_DETERMINED_BAD_VERDICTS = frozenset({ParityVerdict.FAIL_TRACKS_BASE, ParityVerdict.FAIL_GROSS})


def _has_failed_check(report: ParityReport, check_id: str) -> bool:
    """Return whether a parity check result for ``check_id`` is a ``fail``."""
    return any(result.check_id == check_id and result.status == "fail" for result in report.results)


def _any_check_crashed(report: ParityReport) -> bool:
    """Return whether any parity check result records a crash rather than an answer."""
    return any(
        result.status == "fail" and result.message.startswith(_CRASH_MESSAGE_PREFIX)
        for result in report.results
    )


def _has_missing_target_failure(report: ParityReport) -> bool:
    """Return whether a statically-confirmed missing/uncovered target check failed."""
    return any(_has_failed_check(report, check_id) for check_id in _MISSING_TARGET_CHECK_IDS)


def _has_blocking_embedded_failure(report: ParityReport) -> bool:
    """Return whether the embedded base or fused ``text`` report recorded a failure."""
    return bool(report.base_report.summary["fail"]) or bool(report.fused_report.summary["fail"])


def _tool_or_setup_error(report: ParityReport) -> bool:
    """Step 1: return whether a tool/setup-level error occurred (exit 2)."""
    return (
        _any_check_crashed(report)
        or _has_failed_check(report, _MALFORMED_ADAPTER_CONFIG_CHECK_ID)
        or _has_blocking_embedded_failure(report)
    )


def _determined_bad_static(report: ParityReport) -> bool:
    """Step 2: return whether a statically-confirmed defect voided the oracle (exit 1).

    Checked *before* step 3's null-verdict catch-all: a tokenizer mismatch or
    a missing/uncovered target deliberately void-skips the oracle (leaving
    ``verdict=None``) rather than failing it, and is still a determined, real
    defect -- not a "cannot-determine" outcome.
    """
    return _has_missing_target_failure(report) or _has_failed_check(
        report, _TOKENIZER_MISMATCH_CHECK_ID
    )


def _runtime_cannot_determine(report: ParityReport) -> bool:
    """Step 3: return whether the run cannot determine a verdict at runtime (exit 2)."""
    return (
        any(status == "error" for status in report.worker_status.values())
        or report.adapter_applied is not True
        or report.verdict is None
        or report.verdict is ParityVerdict.INCONCLUSIVE
    )


def _determined_bad_runtime(report: ParityReport) -> bool:
    """Step 4: return whether the oracle itself confirmed a defect (exit 1)."""
    return report.verdict in _DETERMINED_BAD_VERDICTS


def parity_exit_code(report: ParityReport) -> int:
    """Return the total process exit code for a completed adapter-parity report (F4).

    Evaluated top-down, first match wins -- see the module docstring for the
    full five-step table. In short: a tool/setup error (step 1) or a
    statically-confirmed defect that void-skipped the oracle (step 2) is
    decided before any runtime null-verdict/worker-status check (step 3), so
    a tokenizer mismatch or missing target never gets mistaken for a mere
    "cannot-determine" just because it left ``verdict`` null.
    """
    if _tool_or_setup_error(report):
        return 2
    if _determined_bad_static(report):
        return 1
    if _runtime_cannot_determine(report):
        return 2
    if _determined_bad_runtime(report):
        return 1
    return 0
