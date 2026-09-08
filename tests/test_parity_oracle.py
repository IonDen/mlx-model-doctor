import pytest

from mlx_model_doctor.parity.oracle import (
    PARITY_FLOOR,
    PARITY_K,
    ROLE_BASE,
    ROLE_FUSED,
    ROLE_NOISE,
    ROLE_REFERENCE,
    argmax_agreement,
    assemble_verdict_inputs,
    decide_verdict,
    first_divergence,
    flip_count,
)
from mlx_model_doctor.parity.oracle import (
    ParityVerdict as V,
)
from mlx_model_doctor.parity.orchestrator import WorkerOutcome


def _ok_outcome(role: str, argmax: list[int]) -> WorkerOutcome:
    """Build an ``ok`` worker outcome carrying a hand-chosen argmax vector."""
    return WorkerOutcome(
        role=role,
        argmax=argmax,
        adapter_applied=None,
        peak_bytes=1,
        status="ok",
        error=None,
    )


class TestLengthGuard:
    """Test that oracle functions reject empty or unequal-length inputs."""

    def test_agreement_requires_equal_nonzero_length(self) -> None:
        assert argmax_agreement([1, 0, 1], [1, 0, 1]) == 1.0
        with pytest.raises(ValueError, match="equal, nonzero"):
            argmax_agreement([1], [1, 2])  # F9: no silent zip
        with pytest.raises(ValueError, match="equal, nonzero"):
            argmax_agreement([], [])  # F9: no empty=1.0

    def test_first_divergence_requires_equal_nonzero_length(self) -> None:
        with pytest.raises(ValueError, match="equal, nonzero"):
            first_divergence([1], [1, 2])
        with pytest.raises(ValueError, match="equal, nonzero"):
            first_divergence([], [])

    def test_flip_count_requires_equal_nonzero_length(self) -> None:
        with pytest.raises(ValueError, match="equal, nonzero"):
            flip_count([1], [1, 2])
        with pytest.raises(ValueError, match="equal, nonzero"):
            flip_count([], [])


class TestOracleFunctionsPositivePath:
    """Test oracle functions with real diverging data (catches loop mutations)."""

    def test_argmax_agreement_full_match(self) -> None:
        assert argmax_agreement([1, 0, 1], [1, 0, 1]) == 1.0

    def test_argmax_agreement_partial_match(self) -> None:
        # 2 out of 3 match; catches wrong operator or denominator logic
        assert argmax_agreement([1, 0, 1], [1, 1, 1]) == 2 / 3

    def test_argmax_agreement_no_match(self) -> None:
        # 0 out of 3 match
        assert argmax_agreement([1, 0, 1], [0, 1, 0]) == 0.0

    def test_first_divergence_at_middle(self) -> None:
        # Divergence at position 1; catches off-by-one or dropped return
        assert first_divergence([1, 0, 1], [1, 1, 1]) == 1

    def test_first_divergence_at_start(self) -> None:
        # Divergence at position 0
        assert first_divergence([0, 0, 0], [1, 0, 0]) == 0

    def test_first_divergence_identical(self) -> None:
        # No divergence; catches missing final return None
        assert first_divergence([1, 1], [1, 1]) is None

    def test_flip_count_single_diff(self) -> None:
        # 1 position differs; catches wrong operator or loop logic
        assert flip_count([1, 0, 1], [1, 1, 1]) == 1

    def test_flip_count_all_diff(self) -> None:
        # All 3 positions differ
        assert flip_count([1, 0, 1], [0, 1, 0]) == 3

    def test_flip_count_identical(self) -> None:
        # No differences
        assert flip_count([1, 0, 1], [1, 0, 1]) == 0


class TestVerdictMatrix:
    """Binary-exact test matrix with k=0.5, floor=0.75; agreements are eighths (exact in float)."""

    def test_pass(self) -> None:
        # delta 0.375 >= required 0.25, agree_fa>=floor
        assert (
            decide_verdict(agree_fa=0.875, agree_fb=0.5, gap=0.5, noise=0.125, k=0.5, floor=0.75)
            is V.PASS
        )

    def test_pass_boundary_delta_exactly_required_ge_vs_gt(self) -> None:
        # delta==0.25
        assert (
            decide_verdict(agree_fa=0.875, agree_fb=0.625, gap=0.5, noise=0.125, k=0.5, floor=0.75)
            is V.PASS
        )

    def test_pass_boundary_agree_fa_exactly_floor(self) -> None:
        assert (
            decide_verdict(agree_fa=0.75, agree_fb=0.375, gap=0.5, noise=0.125, k=0.5, floor=0.75)
            is V.PASS
        )

    def test_fail_tracks_base(self) -> None:
        # -delta 0.375, agree_fb>=floor
        assert (
            decide_verdict(agree_fa=0.5, agree_fb=0.875, gap=0.5, noise=0.125, k=0.5, floor=0.75)
            is V.FAIL_TRACKS_BASE
        )

    def test_copied_base_reference_case_F1(self) -> None:  # noqa: N802
        # active adapter, fused==base
        assert (
            decide_verdict(agree_fa=0.5, agree_fb=1.0, gap=0.5, noise=0.0, k=0.5, floor=0.75)
            is V.FAIL_TRACKS_BASE
        )

    def test_fail_gross_both_low(self) -> None:
        assert (
            decide_verdict(agree_fa=0.5, agree_fb=0.5, gap=0.5, noise=0.125, k=0.5, floor=0.75)
            is V.FAIL_GROSS
        )

    def test_inconclusive_required_within_noise(self) -> None:
        assert (
            decide_verdict(agree_fa=0.875, agree_fb=0.75, gap=0.25, noise=0.125, k=0.5, floor=0.75)
            is V.INCONCLUSIVE
        )

    def test_inconclusive_preference_within_noise_F7(self) -> None:  # noqa: N802
        # high agree_fa, |delta|<=noise
        assert (
            decide_verdict(agree_fa=0.875, agree_fb=0.75, gap=0.5, noise=0.125, k=0.5, floor=0.75)
            is V.INCONCLUSIVE
        )

    def test_inconclusive_preference_exceeds_noise_but_no_threshold(self) -> None:
        # |delta|=0.25 in (noise 0.125, required 0.375)
        assert (
            decide_verdict(agree_fa=0.75, agree_fb=1.0, gap=0.75, noise=0.125, k=0.5, floor=0.75)
            is V.INCONCLUSIVE
        )

    def test_minus_plus_mutant_killer(self) -> None:
        # correct=FAIL_TRACKS_BASE; '+' would PASS
        assert (
            decide_verdict(agree_fa=0.75, agree_fb=1.0, gap=0.5, noise=0.125, k=0.5, floor=0.75)
            is V.FAIL_TRACKS_BASE
        )

    def test_required_equals_noise_boundary_le_vs_lt(self) -> None:
        # required==noise -> INCONCLUSIVE
        assert (
            decide_verdict(agree_fa=0.875, agree_fb=0.5, gap=0.25, noise=0.125, k=0.5, floor=0.75)
            is V.INCONCLUSIVE
        )


class TestAssembleVerdictInputs:
    """Reduce four worker outcomes to (agree_fa, agree_fb, gap, noise) with exact eighths."""

    def test_reduces_four_outcomes_to_the_four_oracle_metrics(self) -> None:
        # base all-1s; reference differs from base at the first 4 of 8 positions
        # (agree=4/8=0.5 -> gap=0.5); base_repeat differs from base at 1 of 8
        # (agree=7/8 -> noise=0.125); fused == reference (agree_fa=1.0) and
        # differs from base at 4/8 (agree_fb=0.5). Wrong role mapping,
        # a `1 - agree` sign flip on gap, or the wrong noise pair each move a value.
        outcomes = [
            _ok_outcome(ROLE_BASE, [1, 1, 1, 1, 1, 1, 1, 1]),
            _ok_outcome(ROLE_REFERENCE, [2, 2, 2, 2, 1, 1, 1, 1]),
            _ok_outcome(ROLE_NOISE, [1, 1, 1, 1, 1, 1, 1, 0]),
            _ok_outcome(ROLE_FUSED, [2, 2, 2, 2, 1, 1, 1, 1]),
        ]
        vi = assemble_verdict_inputs(outcomes)
        assert vi.gap == 0.5
        assert vi.noise == 0.125
        assert vi.agree_fa == 1.0
        assert vi.agree_fb == 0.5

    def test_agree_fa_is_fused_vs_reference_not_fused_vs_base(self) -> None:
        # fused tracks base exactly, diverges from reference at 4/8:
        # agree_fa (fused vs reference) = 0.5, agree_fb (fused vs base) = 1.0.
        # A swapped agree_fa/agree_fb assignment flips these two values.
        outcomes = [
            _ok_outcome(ROLE_BASE, [1, 1, 1, 1, 1, 1, 1, 1]),
            _ok_outcome(ROLE_REFERENCE, [2, 2, 2, 2, 1, 1, 1, 1]),
            _ok_outcome(ROLE_NOISE, [1, 1, 1, 1, 1, 1, 1, 1]),
            _ok_outcome(ROLE_FUSED, [1, 1, 1, 1, 1, 1, 1, 1]),
        ]
        vi = assemble_verdict_inputs(outcomes)
        assert vi.agree_fa == 0.5
        assert vi.agree_fb == 1.0

    def test_missing_reference_role_raises(self) -> None:
        outcomes = [
            _ok_outcome(ROLE_BASE, [1, 1]),
            _ok_outcome(ROLE_NOISE, [1, 1]),
            _ok_outcome(ROLE_FUSED, [1, 1]),
        ]
        with pytest.raises(ValueError, match="reference"):
            assemble_verdict_inputs(outcomes)

    def test_none_argmax_on_a_required_role_raises(self) -> None:
        outcomes = [
            _ok_outcome(ROLE_BASE, [1, 1]),
            _ok_outcome(ROLE_NOISE, [1, 1]),
            _ok_outcome(ROLE_FUSED, [1, 1]),
            WorkerOutcome(
                role=ROLE_REFERENCE,
                argmax=None,
                adapter_applied=None,
                peak_bytes=None,
                status="error",
                error="boom",
            ),
        ]
        with pytest.raises(ValueError, match="reference"):
            assemble_verdict_inputs(outcomes)


class TestConstants:
    """Test that constants meet minimum bounds."""

    def test_parity_k_at_least_0_4(self) -> None:
        assert PARITY_K >= 0.4

    def test_parity_floor_at_least_0_75(self) -> None:
        assert PARITY_FLOOR >= 0.75
