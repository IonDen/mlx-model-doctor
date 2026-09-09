from typing import ClassVar

import pytest

from mlx_model_doctor.parity.oracle import (
    PARITY_GROSS_FLOOR,
    PARITY_K,
    PARITY_PASS_FLOOR,
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


class TestScoredPositions:
    """``scored`` restricts agreement/divergence/flip metrics to a given index subset."""

    # a differs from b at positions 1 and 3 (full agreement 2/4 = 0.5).
    _A: ClassVar[list[int]] = [1, 0, 1, 0]
    _B: ClassVar[list[int]] = [1, 1, 1, 1]

    def test_argmax_agreement_over_scored_subset_differs_from_full(self) -> None:
        assert argmax_agreement(self._A, self._B) == 0.5
        # positions 0 and 2 agree in both sequences -> 1.0, differs from the full 0.5.
        assert argmax_agreement(self._A, self._B, scored=[0, 2]) == 1.0

    def test_flip_count_over_scored_subset_differs_from_full(self) -> None:
        assert flip_count(self._A, self._B) == 2
        assert flip_count(self._A, self._B, scored=[0, 2]) == 0

    def test_first_divergence_over_scored_subset_differs_from_full(self) -> None:
        assert first_divergence(self._A, self._B) == 1
        # restricting to [0, 2] excludes both diverging positions -> None.
        assert first_divergence(self._A, self._B, scored=[0, 2]) is None

    def test_first_divergence_scored_subset_iterates_in_given_order(self) -> None:
        # scored=[3, 1]: position 3 is checked before position 1, so it is
        # reported first even though 1 < 3 -- catches a mutant that sorts or
        # re-indexes ``scored`` instead of iterating it in the given order.
        assert first_divergence(self._A, self._B, scored=[3, 1]) == 3
        assert first_divergence(self._A, self._B, scored=[1, 3]) == 1

    def test_scored_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="scored"):
            argmax_agreement(self._A, self._B, scored=[])
        with pytest.raises(ValueError, match="scored"):
            first_divergence(self._A, self._B, scored=[])
        with pytest.raises(ValueError, match="scored"):
            flip_count(self._A, self._B, scored=[])

    def test_scored_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="scored"):
            argmax_agreement(self._A, self._B, scored=[0, 4])  # len is 4, max valid index 3
        with pytest.raises(ValueError, match="scored"):
            first_divergence(self._A, self._B, scored=[-1])
        with pytest.raises(ValueError, match="scored"):
            flip_count(self._A, self._B, scored=[0, 4])


class TestVerdictMatrix:
    """Binary-exact matrix (eighths/quarters) with k=0.5, pass_floor=0.875, gross_floor=0.5.

    ``pass_floor`` and ``gross_floor`` are deliberately different values (rather than
    reusing one shared floor) so a case can distinguish "used the right floor in the
    right branch" from "used a floor" -- a parameter swap between them would flip a
    verdict on at least one case here.
    """

    def test_pass(self) -> None:
        # delta 0.5 >= required 0.25, agree_fa (1.0) >= pass_floor (0.875)
        assert (
            decide_verdict(
                agree_fa=1.0,
                agree_fb=0.5,
                gap=0.5,
                noise=0.125,
                k=0.5,
                pass_floor=0.875,
                gross_floor=0.5,
            )
            is V.PASS
        )

    def test_pass_boundary_delta_exactly_required_and_agree_fa_exactly_pass_floor(self) -> None:
        # delta==required==0.25 (ge vs gt) and agree_fa==pass_floor==0.875 (ge vs gt)
        assert (
            decide_verdict(
                agree_fa=0.875,
                agree_fb=0.625,
                gap=0.5,
                noise=0.125,
                k=0.5,
                pass_floor=0.875,
                gross_floor=0.5,
            )
            is V.PASS
        )

    def test_pass_requires_true_pass_floor_not_gross_floor(self) -> None:
        # agree_fa (0.625) clears gross_floor (0.5) but NOT pass_floor (0.875); delta
        # (0.5) clears required (0.25). A mutant that swapped gross_floor in for
        # pass_floor here would wrongly return PASS instead of INCONCLUSIVE.
        assert (
            decide_verdict(
                agree_fa=0.625,
                agree_fb=0.125,
                gap=0.5,
                noise=0.125,
                k=0.5,
                pass_floor=0.875,
                gross_floor=0.5,
            )
            is V.INCONCLUSIVE
        )

    def test_fail_tracks_base(self) -> None:
        # -delta 0.5 >= required 0.25, agree_fb (1.0) >= gross_floor (0.5). Also
        # exercises the FAIL_GROSS `<` boundary: agree_fa==gross_floor exactly does
        # NOT count as "below" it, so FAIL_GROSS is correctly skipped.
        assert (
            decide_verdict(
                agree_fa=0.5,
                agree_fb=1.0,
                gap=0.5,
                noise=0.125,
                k=0.5,
                pass_floor=0.875,
                gross_floor=0.5,
            )
            is V.FAIL_TRACKS_BASE
        )

    def test_copied_base_reference_case_F1(self) -> None:  # noqa: N802
        # active adapter, fused==base
        assert (
            decide_verdict(
                agree_fa=0.5,
                agree_fb=1.0,
                gap=0.5,
                noise=0.0,
                k=0.5,
                pass_floor=0.875,
                gross_floor=0.5,
            )
            is V.FAIL_TRACKS_BASE
        )

    def test_fail_gross_both_low(self) -> None:
        assert (
            decide_verdict(
                agree_fa=0.375,
                agree_fb=0.25,
                gap=0.5,
                noise=0.125,
                k=0.5,
                pass_floor=0.875,
                gross_floor=0.5,
            )
            is V.FAIL_GROSS
        )

    def test_fail_gross_boundary_both_exactly_at_gross_floor_not_triggered(self) -> None:
        # both agree_fa and agree_fb == gross_floor exactly: `<` (not `<=`) means
        # neither counts as "below" it, so FAIL_GROSS must not fire.
        assert (
            decide_verdict(
                agree_fa=0.5,
                agree_fb=0.5,
                gap=0.5,
                noise=0.0,
                k=0.5,
                pass_floor=0.875,
                gross_floor=0.5,
            )
            is V.INCONCLUSIVE
        )

    def test_inconclusive_required_within_noise(self) -> None:
        assert (
            decide_verdict(
                agree_fa=0.875,
                agree_fb=0.75,
                gap=0.25,
                noise=0.125,
                k=0.5,
                pass_floor=0.875,
                gross_floor=0.5,
            )
            is V.INCONCLUSIVE
        )

    def test_inconclusive_preference_within_noise_F7(self) -> None:  # noqa: N802
        # high agree_fa, |delta|<=noise
        assert (
            decide_verdict(
                agree_fa=0.875,
                agree_fb=0.75,
                gap=0.5,
                noise=0.125,
                k=0.5,
                pass_floor=0.875,
                gross_floor=0.5,
            )
            is V.INCONCLUSIVE
        )

    def test_q4_partial_blend_mirrors_real_calibration(self) -> None:
        # Mirrors the real default-q4 fuse: fa and fb both land mid-range and close
        # to each other (a partial degradation, not a clean win for either side) --
        # the gap between them never clears `required`, so the oracle correctly
        # declines to call it rather than defaulting to FAIL.
        assert (
            decide_verdict(
                agree_fa=0.75,
                agree_fb=0.875,
                gap=0.5,
                noise=0.0,
                k=0.5,
                pass_floor=0.875,
                gross_floor=0.5,
            )
            is V.INCONCLUSIVE
        )

    def test_minus_plus_mutant_killer(self) -> None:
        # correct=FAIL_TRACKS_BASE; '+' would PASS
        assert (
            decide_verdict(
                agree_fa=0.75,
                agree_fb=1.0,
                gap=0.5,
                noise=0.125,
                k=0.5,
                pass_floor=0.875,
                gross_floor=0.5,
            )
            is V.FAIL_TRACKS_BASE
        )

    def test_required_equals_noise_boundary_le_vs_lt(self) -> None:
        # required==noise -> INCONCLUSIVE
        assert (
            decide_verdict(
                agree_fa=0.875,
                agree_fb=0.5,
                gap=0.25,
                noise=0.125,
                k=0.5,
                pass_floor=0.875,
                gross_floor=0.5,
            )
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

    def test_scored_restricts_all_four_metrics(self) -> None:
        # base/noise/reference/fused chosen so gap, noise, agree_fa, and agree_fb
        # each take a different value over the full 8 positions than over
        # scored=[0, 1, 2, 3] -- catches a metric that stayed on the full
        # sequence instead of being routed through `scored` (all four internal
        # argmax_agreement calls must consume the same scored set).
        outcomes = [
            _ok_outcome(ROLE_BASE, [0, 0, 0, 0, 1, 1, 1, 1]),
            _ok_outcome(ROLE_REFERENCE, [0, 0, 1, 1, 0, 0, 0, 0]),
            _ok_outcome(ROLE_NOISE, [0, 0, 0, 1, 1, 1, 1, 1]),
            _ok_outcome(ROLE_FUSED, [0, 1, 1, 1, 1, 1, 0, 0]),
        ]
        full = assemble_verdict_inputs(outcomes)
        assert full.gap == 0.75
        assert full.noise == 0.125
        assert full.agree_fa == 0.625
        assert full.agree_fb == 0.375

        scored = assemble_verdict_inputs(outcomes, scored=[0, 1, 2, 3])
        assert scored.gap == 0.5
        assert scored.noise == 0.25
        assert scored.agree_fa == 0.75
        assert scored.agree_fb == 0.25

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
    """Test that the calibrated (2026-09-09) constants meet their calibrated bounds."""

    def test_parity_k_at_least_0_2(self) -> None:
        assert PARITY_K >= 0.2

    def test_parity_pass_floor_at_least_0_85(self) -> None:
        assert PARITY_PASS_FLOOR >= 0.85

    def test_parity_gross_floor_at_most_0_6(self) -> None:
        assert PARITY_GROSS_FLOOR <= 0.6
