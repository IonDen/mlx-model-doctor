"""Three-way relative parity oracle (pure; per-position argmax int sequences)."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

PARITY_K = 0.5  # margin fraction of the base-vs-adapter gap (finalized in Task 13)
PARITY_FLOOR = 0.8  # min fused-vs-adapter agreement for PASS (finalized in Task 13)

# The four worker roles the reducer consumes. ``ROLE_NOISE`` is a second load of
# the base model whose disagreement with the first base load measures the
# cross-process argmax noise floor; ``ROLE_REFERENCE`` is the base+adapter load
# (the only source of the adapter-applied signal, F1).
ROLE_BASE = "base"
ROLE_NOISE = "base_repeat"
ROLE_REFERENCE = "reference"
ROLE_FUSED = "fused"


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


class _RoleArgmax(Protocol):
    """Minimal ``(role, argmax)`` surface the reducer needs from a worker outcome.

    Declared as read-only properties so a frozen ``WorkerOutcome`` satisfies it
    structurally without the reducer importing the orchestrator.
    """

    @property
    def role(self) -> str:
        """The worker's role name."""

    @property
    def argmax(self) -> list[int] | None:
        """The worker's per-position argmax vector, or ``None`` when it did not run."""


@dataclass(frozen=True, slots=True, kw_only=True)
class VerdictInputs:
    """The four scalar inputs :func:`decide_verdict` consumes.

    Built by :func:`assemble_verdict_inputs` from the four worker outcomes, kept
    separate from :func:`decide_verdict` so the reduction (which argmax pair maps
    to which metric) is unit-testable independently of the verdict thresholds.
    """

    agree_fa: float
    agree_fb: float
    gap: float
    noise: float


def _require_argmax(by_role: Mapping[str, _RoleArgmax], role: str) -> list[int]:
    """Return the argmax vector for ``role``, or raise if it is absent or null."""
    outcome = by_role.get(role)
    if outcome is None or outcome.argmax is None:
        raise ValueError(
            f"assemble_verdict_inputs requires a usable argmax for worker role {role!r}"
        )
    return outcome.argmax


def assemble_verdict_inputs(outcomes: Sequence[_RoleArgmax]) -> VerdictInputs:
    """Reduce the four worker outcomes to the oracle's scalar inputs (pure, F1/F8).

    Maps each role's argmax vector to a metric: ``gap`` is the base-vs-adapter
    behavioral gap on this fixture (``1 - agree(base, reference)``); ``noise`` is
    the cross-process argmax noise floor from the base-repeat pair
    (``1 - agree(base, base_repeat)``); ``agree_fa`` is the fused-vs-base+adapter
    agreement and ``agree_fb`` the fused-vs-base agreement. Requires all four
    roles present with equal, nonzero-length argmax vectors (delegated to
    :func:`argmax_agreement`); raises ``ValueError`` otherwise.
    """
    by_role = {outcome.role: outcome for outcome in outcomes}
    base = _require_argmax(by_role, ROLE_BASE)
    base_repeat = _require_argmax(by_role, ROLE_NOISE)
    reference = _require_argmax(by_role, ROLE_REFERENCE)
    fused = _require_argmax(by_role, ROLE_FUSED)
    return VerdictInputs(
        agree_fa=argmax_agreement(fused, reference),
        agree_fb=argmax_agreement(fused, base),
        gap=1.0 - argmax_agreement(base, reference),
        noise=1.0 - argmax_agreement(base, base_repeat),
    )


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
