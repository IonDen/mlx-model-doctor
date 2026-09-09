"""Three-way relative parity oracle (pure; per-position argmax int sequences)."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

# Calibrated 2026-09-09 against mlx 0.32.0 / mlx-lm 0.31.3 on Qwen2.5-0.5B-Instruct-4bit +
# a wikisql LoRA; fp16 (--dequantize) fuse fa~0.97 -> PASS, default-q4 fuse fa~0.78/fb~0.82
# -> INCONCLUSIVE (partial degradation), fused==base -> FAIL_TRACKS_BASE (see
# docs/superpowers/reviews/2026-09-09-mlx-model-doctor-adapter-parity-calibration.md).
PARITY_K = 0.3  # margin fraction of the base-vs-adapter gap
PARITY_PASS_FLOOR = 0.9  # min fused-vs-adapter agreement for PASS
PARITY_GROSS_FLOOR = 0.5  # max fused-vs-{adapter,base} agreement for FAIL_GROSS

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


def _resolve_positions(length: int, scored: list[int] | None) -> Sequence[int]:
    """Return the indices to iterate: ``scored`` when given (validated), else all of them.

    ``scored`` must be nonzero-length and every index must be in ``range(length)``;
    raises ``ValueError`` otherwise (same discipline as :func:`_check`).
    """
    if scored is None:
        return range(length)
    if not scored or any(idx < 0 or idx >= length for idx in scored):
        raise ValueError("scored positions must be a nonzero-length list of valid indices")
    return scored


def argmax_agreement(a: list[int], b: list[int], scored: list[int] | None = None) -> float:
    """Compute agreement ratio: fraction of positions where a and b are equal.

    When ``scored`` is given, the ratio is computed over only those indices.
    """
    _check(a, b)
    positions = _resolve_positions(len(a), scored)
    return sum(1 for i in positions if a[i] == b[i]) / len(positions)


def first_divergence(a: list[int], b: list[int], scored: list[int] | None = None) -> int | None:
    """Return the first position where a and b differ, or None if identical.

    When ``scored`` is given, only those indices are considered, checked in the
    order ``scored`` lists them.
    """
    _check(a, b)
    positions = _resolve_positions(len(a), scored)
    for i in positions:
        if a[i] != b[i]:
            return i
    return None


def flip_count(a: list[int], b: list[int], scored: list[int] | None = None) -> int:
    """Return the number of positions where a and b differ.

    When ``scored`` is given, only those indices are counted.
    """
    _check(a, b)
    positions = _resolve_positions(len(a), scored)
    return sum(1 for i in positions if a[i] != b[i])


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
    pass_floor: float,
    gross_floor: float,
) -> ParityVerdict:
    """Compute three-way parity verdict from agreement metrics and noise bounds.

    ``pass_floor`` and ``gross_floor`` are separate thresholds: ``pass_floor`` is
    the higher bar a PASS must clear on ``agree_fa``; ``gross_floor`` is the lower
    bar below which fused tracks neither the base+adapter reference nor the base
    (FAIL_GROSS), and the bar FAIL_TRACKS_BASE must clear on ``agree_fb``.
    """
    required = k * gap
    if required <= noise:  # can't discriminate
        return ParityVerdict.INCONCLUSIVE
    if agree_fa < gross_floor and agree_fb < gross_floor:  # tracks neither
        return ParityVerdict.FAIL_GROSS
    delta = agree_fa - agree_fb
    if abs(delta) <= noise:  # preference within noise (F7)
        return ParityVerdict.INCONCLUSIVE
    if delta >= required and agree_fa >= pass_floor:
        return ParityVerdict.PASS
    if -delta >= required and agree_fb >= gross_floor:
        return ParityVerdict.FAIL_TRACKS_BASE
    return ParityVerdict.INCONCLUSIVE  # not a catch-all FAIL_GROSS
