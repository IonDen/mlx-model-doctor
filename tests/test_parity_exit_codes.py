"""Tests for the total parity exit-code decision table (F4).

Every test builds a REAL ``ParityReport`` (via ``tests.parity_fakes.build_parity_report``,
never a bypass flag) and isolates exactly one branch of the decision table, so a
regression in ``parity_exit_code`` -- removing a clause, or reordering the
precedence check -- makes the matching test fail for the stated reason.
"""

from mlx_model_doctor.parity.exit_codes import parity_exit_code
from mlx_model_doctor.parity.oracle import ParityVerdict
from tests.parity_fakes import build_parity_report, embedded_doctor_report, parity_check_result


def test_pass_verdict_with_no_blocking_failure_is_zero() -> None:
    """A clean, eligible PASS run exits 0."""
    report = build_parity_report(verdict=ParityVerdict.PASS)
    assert parity_exit_code(report) == 0


def test_fail_tracks_base_verdict_is_one() -> None:
    """FAIL_TRACKS_BASE is a determined defect (adapter effect lost), not a void: exit 1.

    Regresses to 0 if FAIL_TRACKS_BASE is dropped from the determined-bad verdict set.
    """
    report = build_parity_report(verdict=ParityVerdict.FAIL_TRACKS_BASE)
    assert parity_exit_code(report) == 1


def test_fail_gross_verdict_is_one() -> None:
    """FAIL_GROSS (tracks neither base nor adapter) is a determined defect: exit 1."""
    report = build_parity_report(verdict=ParityVerdict.FAIL_GROSS)
    assert parity_exit_code(report) == 1


def test_tokenizer_mismatch_is_one_even_with_pass_shaped_verdict() -> None:
    """A confirmed tokenizer-identity failure is a real, determined defect: exit 1.

    ``verdict`` is deliberately left PASS here to isolate the tokenizer-mismatch
    clause: if that clause were removed, this would regress to 0 instead of 1.
    """
    report = build_parity_report(
        verdict=ParityVerdict.PASS,
        results=(
            parity_check_result(
                check_id="parity/tokenizer.identity", status="fail", severity="high"
            ),
        ),
    )
    assert parity_exit_code(report) == 1


def test_uncovered_target_coverage_failure_is_one_even_with_pass_shaped_verdict() -> None:
    """A statically-confirmed uncovered LoRA target is a real defect: exit 1.

    ``verdict`` is deliberately left PASS to isolate the target-coverage clause.
    """
    report = build_parity_report(
        verdict=ParityVerdict.PASS,
        results=(
            parity_check_result(check_id="parity/target.coverage", status="fail", severity="high"),
        ),
    )
    assert parity_exit_code(report) == 1


def test_fused_target_inconsistency_failure_is_one_even_with_pass_shaped_verdict() -> None:
    """A statically-confirmed fused-target inconsistency is a real defect: exit 1."""
    report = build_parity_report(
        verdict=ParityVerdict.PASS,
        results=(
            parity_check_result(
                check_id="parity/fused.target_consistency", status="fail", severity="high"
            ),
        ),
    )
    assert parity_exit_code(report) == 1


def test_adapter_applied_false_is_two_even_with_pass_shaped_verdict() -> None:
    """``adapter_applied=False`` means no valid reference: cannot-determine, exit 2.

    ``verdict`` is deliberately left PASS to isolate the ``adapter_applied``
    clause: without it, this would fall through to 0.
    """
    report = build_parity_report(verdict=ParityVerdict.PASS, adapter_applied=False)
    assert parity_exit_code(report) == 2


def test_adapter_applied_none_is_two_even_with_pass_shaped_verdict() -> None:
    """``adapter_applied=None`` (unsupported adapter type) is also cannot-determine, exit 2."""
    report = build_parity_report(verdict=ParityVerdict.PASS, adapter_applied=None)
    assert parity_exit_code(report) == 2


def test_null_verdict_is_two_even_with_adapter_applied_true() -> None:
    """A null verdict (a blocking phase short-circuit) is cannot-determine, exit 2.

    ``adapter_applied`` is deliberately left True to isolate the verdict-null
    clause from the ``adapter_applied`` clause.
    """
    report = build_parity_report(verdict=None, adapter_applied=True)
    assert parity_exit_code(report) == 2


def test_inconclusive_verdict_is_two_even_with_adapter_applied_true() -> None:
    """INCONCLUSIVE (can't discriminate from noise) is cannot-determine, exit 2."""
    report = build_parity_report(verdict=ParityVerdict.INCONCLUSIVE, adapter_applied=True)
    assert parity_exit_code(report) == 2


def test_worker_error_is_two_even_with_pass_shaped_verdict() -> None:
    """Any worker_status value of 'error' is cannot-determine, exit 2.

    ``verdict``/``adapter_applied`` are deliberately left PASS-shaped to prove
    the worker_status clause is checked independently of the verdict.
    """
    report = build_parity_report(
        verdict=ParityVerdict.PASS,
        adapter_applied=True,
        worker_status={"base": "ok", "adapter": "ok", "fused": "error"},
    )
    assert parity_exit_code(report) == 2


def test_crashed_check_is_two_not_zero_even_with_pass_shaped_verdict() -> None:
    """A crashed parity check could not even determine an answer: cannot-determine, exit 2.

    Regresses to 0 if the crash-message check is dropped from ``_cannot_determine``.
    """
    report = build_parity_report(
        verdict=ParityVerdict.PASS,
        results=(
            parity_check_result(
                check_id="parity/fused.target_consistency",
                status="fail",
                severity="high",
                message="check crashed: ValueError('boom')",
            ),
        ),
    )
    assert parity_exit_code(report) == 2


def test_malformed_adapter_config_is_two_not_zero_even_with_pass_shaped_verdict() -> None:
    """A malformed adapter_config.json blocks static analysis: cannot-determine, exit 2.

    Regresses to 0 if the adapter-config-well-formed check_id is dropped from
    ``_cannot_determine``.
    """
    report = build_parity_report(
        verdict=ParityVerdict.PASS,
        results=(
            parity_check_result(
                check_id="parity/adapter.config_well_formed", status="fail", severity="high"
            ),
        ),
    )
    assert parity_exit_code(report) == 2


def test_blocking_embedded_base_failure_is_two_not_zero() -> None:
    """A failure inside the embedded base ``text`` report blocks the whole run: exit 2."""
    report = build_parity_report(
        verdict=ParityVerdict.PASS, base_report=embedded_doctor_report("base", status="fail")
    )
    assert parity_exit_code(report) == 2


def test_blocking_embedded_fused_failure_is_two_not_zero() -> None:
    """A failure inside the embedded fused ``text`` report blocks the whole run: exit 2."""
    report = build_parity_report(
        verdict=ParityVerdict.PASS, fused_report=embedded_doctor_report("fused", status="fail")
    )
    assert parity_exit_code(report) == 2


def test_precedence_static_defect_and_worker_error_is_two() -> None:
    """cannot-determine outranks determined-bad: a worker error alongside a
    statically-confirmed defect (which alone would be 1) must still exit 2.

    Regresses to 1 if cannot-determine is evaluated after determined-bad.
    """
    report = build_parity_report(
        verdict=ParityVerdict.PASS,
        results=(
            parity_check_result(
                check_id="parity/tokenizer.identity", status="fail", severity="high"
            ),
        ),
        worker_status={"base": "ok", "adapter": "ok", "fused": "error"},
    )
    assert parity_exit_code(report) == 2


def test_precedence_crash_beats_static_fail() -> None:
    """A crashed check alongside an independent, real static defect still exits 2.

    Regresses to 1 if the crash check does not preempt the determined-bad path.
    """
    report = build_parity_report(
        verdict=ParityVerdict.PASS,
        results=(
            parity_check_result(check_id="parity/target.coverage", status="fail", severity="high"),
            parity_check_result(
                check_id="parity/fused.target_consistency",
                status="fail",
                severity="high",
                message="check crashed: RuntimeError('boom')",
            ),
        ),
    )
    assert parity_exit_code(report) == 2
