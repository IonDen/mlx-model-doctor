"""Three-way relative parity oracle (pure; per-position argmax int sequences)."""

from enum import Enum

PARITY_K = 0.5  # margin fraction of the base-vs-adapter gap (finalized in Task 13)
PARITY_FLOOR = 0.8  # min fused-vs-adapter agreement for PASS (finalized in Task 13)


class ParityVerdict(Enum):
    """Verdict on adapter/fused parity relative to base model outputs."""

    PASS = "pass"
    FAIL_TRACKS_BASE = "fail_tracks_base"
    FAIL_GROSS = "fail_gross"
    INCONCLUSIVE = "inconclusive"


def _check(a: list[int], b: list[int]) -> None:
    """Check that sequences are equal length and nonzero; raise ValueError otherwise."""
    if len(a) != len(b) or not a:
        raise ValueError("parity sequences must be equal, nonzero length")


def argmax_agreement(a: list[int], b: list[int]) -> float:
    """Compute agreement ratio: fraction of positions where a and b are equal."""
    _check(a, b)
    return sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a)


def first_divergence(a: list[int], b: list[int]) -> int | None:
    """Return the first position where a and b differ, or None if identical."""
    _check(a, b)
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        if x != y:
            return i
    return None


def flip_count(a: list[int], b: list[int]) -> int:
    """Return the number of positions where a and b differ."""
    _check(a, b)
    return sum(1 for x, y in zip(a, b, strict=True) if x != y)


def decide_verdict(
    *,
    agree_fa: float,
    agree_fb: float,
    gap: float,
    noise: float,
    k: float,
    floor: float,
) -> ParityVerdict:
    """Compute three-way parity verdict from agreement metrics and noise bounds."""
    required = k * gap
    if required <= noise:  # can't discriminate
        return ParityVerdict.INCONCLUSIVE
    if agree_fa < floor and agree_fb < floor:  # tracks neither
        return ParityVerdict.FAIL_GROSS
    delta = agree_fa - agree_fb
    if abs(delta) <= noise:  # preference within noise (F7)
        return ParityVerdict.INCONCLUSIVE
    if delta >= required and agree_fa >= floor:
        return ParityVerdict.PASS
    if -delta >= required and agree_fb >= floor:
        return ParityVerdict.FAIL_TRACKS_BASE
    return ParityVerdict.INCONCLUSIVE  # not a catch-all FAIL_GROSS
