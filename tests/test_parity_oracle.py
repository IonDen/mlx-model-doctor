import pytest

from mlx_model_doctor.parity.oracle import (
    PARITY_FLOOR,
    PARITY_K,
    argmax_agreement,
    decide_verdict,
    first_divergence,
    flip_count,
)
from mlx_model_doctor.parity.oracle import (
    ParityVerdict as V,
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


class TestConstants:
    """Test that constants meet minimum bounds."""

    def test_parity_k_at_least_0_4(self) -> None:
        assert PARITY_K >= 0.4

    def test_parity_floor_at_least_0_75(self) -> None:
        assert PARITY_FLOOR >= 0.75
