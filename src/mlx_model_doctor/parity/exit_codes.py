"""Pure, total exit-code decision for adapter/fused-model parity reports (F4).

``parity_exit_code`` is deliberately separate from
:mod:`mlx_model_doctor.exit_codes` (``exit_code_for``/``exit_code_for_error``,
which stay untouched) -- the parity run has its own decision table over
:class:`~mlx_model_doctor.parity.report.ParityReport`, not a
:class:`~mlx_model_doctor.report.DoctorReport`. The table is total and
evaluated in strict precedence order, cannot-determine (2) before
determined-bad (1) before pass (0), so a blocking defect or worker error can
never fall through to a pass -- even alongside a ``PASS`` verdict or a
statically-confirmed defect that would otherwise resolve to 1 on its own.
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


def _cannot_determine(report: ParityReport) -> bool:
    """Return whether the run cannot determine a parity verdict at all (F4, exit 2)."""
    return (
        any(status == "error" for status in report.worker_status.values())
        or report.adapter_applied is not True
        or report.verdict is None
        or report.verdict is ParityVerdict.INCONCLUSIVE
        or _any_check_crashed(report)
        or _has_failed_check(report, _MALFORMED_ADAPTER_CONFIG_CHECK_ID)
        or _has_blocking_embedded_failure(report)
    )


def _determined_bad(report: ParityReport) -> bool:
    """Return whether the run confirms a real, statically- or oracle-determined defect (exit 1)."""
    return (
        report.verdict in _DETERMINED_BAD_VERDICTS
        or _has_missing_target_failure(report)
        or _has_failed_check(report, _TOKENIZER_MISMATCH_CHECK_ID)
    )


def parity_exit_code(report: ParityReport) -> int:
    """Return the total process exit code for a completed adapter-parity report (F4).

    - ``2`` (cannot-determine): a worker errored, ``adapter_applied`` is
      ``False``/``None``, ``verdict`` is ``None``/``INCONCLUSIVE``, a parity
      check crashed, the adapter config is malformed, or an embedded
      base/fused ``text`` report recorded a blocking failure.
    - ``1`` (determined-bad): none of the above, and either the oracle
      verdict is ``FAIL_TRACKS_BASE``/``FAIL_GROSS``, a target-coverage or
      fused-consistency check statically confirmed a missing/uncovered
      target, or the tokenizer-identity check confirmed a mismatch.
    - ``0`` (pass): none of the above -- an eligible run with no blocking
      failure (in practice, ``verdict is ParityVerdict.PASS``).
    """
    if _cannot_determine(report):
        return 2
    if _determined_bad(report):
        return 1
    return 0
