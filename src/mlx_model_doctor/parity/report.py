"""Adapter/fused-model parity report model + JSON rendering (F1/F4/F6/F8/F9/F11).

:class:`ParityReport` is the companion report ``check_adapter_parity`` produces --
kept separate from :class:`~mlx_model_doctor.report.DoctorReport` /
``report.v1`` (which stay untouched) per the design's "verify-only, separate
report" decision. Its runtime fields (``verdict`` and the agreement/divergence
metrics) are deliberately **nullable**: a blocking phase short-circuit (a
blocking static defect, a tokenizer mismatch, a worker crash, an
unsupported/incomplete adapter reference) leaves them ``None`` with the reason
recorded in :attr:`ParityReport.reasons` and the responsible phase/role named in
:attr:`ParityReport.phase_outcomes`/:attr:`ParityReport.worker_status` -- there is
no total run/result state that forces a stand-in verdict (F4).
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from mlx_model_doctor.parity.context import TokenizerFingerprint
from mlx_model_doctor.parity.deltamap import TensorDelta, TensorDeltaKlass, aggregate_delta_summary
from mlx_model_doctor.parity.fixtures import FixtureRef
from mlx_model_doctor.parity.oracle import ParityVerdict
from mlx_model_doctor.report import CheckResult, DoctorReport, render_json

SourceKind = Literal["local", "hf"]

# A run proceeds in named phases (spec §9); each settles into one of these once
# the run reaches (or short-circuits past) it.
PhaseOutcome = Literal["ok", "blocking_fail", "skipped", "error", "unsupported"]

# A single worker load's outcome (mirrors WorkerOutcome.status in orchestrator.py).
WorkerStatusValue = Literal["ok", "error", "skipped"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedIdentity:
    """One resolved model target's pinned-snapshot identity (F10).

    ``path`` is the pinned local snapshot path every worker load actually uses;
    ``original_ref`` is what the caller supplied verbatim (a local path or a
    Hugging Face repo id); ``source`` records which kind ``original_ref`` was.
    """

    path: str
    original_ref: str
    source: SourceKind


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeProvenance:
    """Hardware/dtype/precision context a run was measured under (F8).

    Every field is ``None`` when not recorded, never guessed or defaulted.
    """

    chip: str | None = None
    os: str | None = None
    backend: str | None = None
    weight_dtype: str | None = None
    compute_note: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ParityReport:
    """Full result of an adapter-vs-fused-model parity run.

    Grouped, per the design (spec §3), into: identities (which pinned
    snapshots were compared, under what tokenizer/fixture/tool/hardware
    context), static results (the embedded base/fused ``text`` reports plus
    the cross-target parity checks), runtime metrics (the three-way oracle's
    verdict and agreement/divergence numbers -- all nullable, F4/F7), and run
    state (per-phase and per-worker outcomes, peak memory, and the reasons a
    null runtime field was left null).
    """

    schema_version: str = "parity/1.0"

    # --- identities ---------------------------------------------------------
    base: ResolvedIdentity
    adapter: ResolvedIdentity
    fused: ResolvedIdentity
    tokenizer_fingerprint: TokenizerFingerprint
    fixture: FixtureRef
    tool_version: str
    mlx_version: str | None
    mlx_lm_version: str | None
    provenance: RuntimeProvenance

    # --- static (reused single-target pipeline + cross-target checks) ------
    base_report: DoctorReport
    fused_report: DoctorReport
    results: tuple[CheckResult, ...]

    # --- runtime (nullable -- null exactly when not measured, F4/F7) -------
    verdict: ParityVerdict | None
    agree_fa: float | None
    agree_fb: float | None
    gap: float | None
    noise: float | None
    first_divergence: int | None
    flip_count: int | None
    adapter_applied: bool | None
    delta_map: tuple[TensorDelta, ...]

    # --- run state (F4) ------------------------------------------------------
    phase_outcomes: dict[str, PhaseOutcome]
    worker_status: dict[str, WorkerStatusValue]
    peak_bytes: dict[str, int | None]
    reasons: tuple[str, ...] = field(default_factory=tuple)
    # Populated for exactly the roles whose worker_status is "error" -- the
    # worker's captured cause (stderr, or the watchdog abort marker's reason),
    # so a worker failure's cause is never silently dropped from the report.
    worker_errors: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy caller-owned collections so the report is stable."""
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "delta_map", tuple(self.delta_map))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "phase_outcomes", dict(self.phase_outcomes))
        object.__setattr__(self, "worker_status", dict(self.worker_status))
        object.__setattr__(self, "peak_bytes", dict(self.peak_bytes))
        object.__setattr__(self, "worker_errors", dict(self.worker_errors))


def _identity_to_dict(identity: ResolvedIdentity) -> dict[str, object]:
    """Convert a resolved model identity to a JSON-serializable dict."""
    return {
        "path": identity.path,
        "original_ref": identity.original_ref,
        "source": identity.source,
    }


def _tokenizer_fingerprint_to_dict(fingerprint: TokenizerFingerprint) -> dict[str, object]:
    """Convert a tokenizer fingerprint to a JSON-serializable dict."""
    return {
        "vocab_size": fingerprint.vocab_size,
        "special_tokens_digest": fingerprint.special_tokens_digest,
        "token_id_map_digest": fingerprint.token_id_map_digest,
        "chat_template_digest": fingerprint.chat_template_digest,
    }


def _fixture_to_dict(fixture: FixtureRef) -> dict[str, object]:
    """Convert a fixture reference's identity/contract fields to a JSON-serializable dict."""
    return {
        "id": fixture.id,
        "input_digest": fixture.input_digest,
        "max_length": fixture.max_length,
        "scored_positions": list(fixture.scored_positions),
    }


def _provenance_to_dict(provenance: RuntimeProvenance) -> dict[str, object]:
    """Convert runtime provenance to a JSON-serializable dict."""
    return {
        "chip": provenance.chip,
        "os": provenance.os,
        "backend": provenance.backend,
        "weight_dtype": provenance.weight_dtype,
        "compute_note": provenance.compute_note,
    }


def _result_to_dict(result: CheckResult) -> dict[str, object]:
    """Convert a check result to a JSON-serializable dict (mirrors report.v1's result shape)."""
    return {
        "check_id": result.check_id,
        "title": result.title,
        "status": result.status,
        "severity": result.severity,
        "message": result.message,
        "remediation": result.remediation,
        "details": dict(result.details),
        "duration_s": result.duration_s,
    }


def _delta_to_dict(delta: TensorDelta) -> dict[str, object]:
    """Convert a tensor delta to a JSON-serializable dict."""
    return {"tensor": delta.tensor, "klass": delta.klass, "reason": delta.reason}


def _embedded_report(report: DoctorReport) -> dict[str, object]:
    """Render an embedded DoctorReport through its own renderer, then parse it back.

    Mirrors ``mlx_model_doctor.sampling._report_to_dict`` -- reusing
    ``render_json`` (rather than re-implementing the report.v1 shape here) is
    what keeps the embedded report byte-for-byte identical to a standalone
    ``check`` report, so the same ``report.v1`` schema validates it via a
    ``$ref``.
    """
    return cast("dict[str, object]", json.loads(render_json(report)))


def parity_report_to_dict(report: ParityReport) -> dict[str, object]:
    """Convert a ParityReport into the JSON-serializable dict parity.v1 describes."""
    verdict_value = report.verdict.value if report.verdict is not None else None
    return {
        "schema_version": report.schema_version,
        "base": _identity_to_dict(report.base),
        "adapter": _identity_to_dict(report.adapter),
        "fused": _identity_to_dict(report.fused),
        "tokenizer_fingerprint": _tokenizer_fingerprint_to_dict(report.tokenizer_fingerprint),
        "fixture": _fixture_to_dict(report.fixture),
        "tool_version": report.tool_version,
        "mlx_version": report.mlx_version,
        "mlx_lm_version": report.mlx_lm_version,
        "provenance": _provenance_to_dict(report.provenance),
        "base_report": _embedded_report(report.base_report),
        "fused_report": _embedded_report(report.fused_report),
        "results": [_result_to_dict(result) for result in report.results],
        "verdict": verdict_value,
        "agree_fa": report.agree_fa,
        "agree_fb": report.agree_fb,
        "gap": report.gap,
        "noise": report.noise,
        "first_divergence": report.first_divergence,
        "flip_count": report.flip_count,
        "adapter_applied": report.adapter_applied,
        "delta_map": [_delta_to_dict(delta) for delta in report.delta_map],
        "phase_outcomes": dict(report.phase_outcomes),
        "worker_status": dict(report.worker_status),
        "peak_bytes": dict(report.peak_bytes),
        "reasons": list(report.reasons),
        "worker_errors": dict(report.worker_errors),
    }


def render_parity_json(report: ParityReport) -> str:
    """Render a ParityReport as stable JSON."""
    return json.dumps(parity_report_to_dict(report), indent=2, sort_keys=True)


def _fmt_float(value: float | None) -> str:
    """Format an optional metric value for display, or 'n/a' when unmeasured."""
    return "n/a" if value is None else f"{value:.4f}"


def _fmt_int(value: int | None) -> str:
    """Format an optional integer position for display, or 'none' when there isn't one."""
    return "none" if value is None else str(value)


def _fmt_bool(value: bool | None) -> str:
    """Format an optional boolean for display, or 'n/a' when unmeasured."""
    return "n/a" if value is None else str(value).lower()


def _verdict_label(verdict: ParityVerdict | None) -> str:
    """Return the display label for a (possibly null) parity verdict."""
    return "not determined" if verdict is None else verdict.value


def _failing_static_checks(report: ParityReport) -> list[tuple[str, CheckResult]]:
    """Return every non-passing check across the parity, base, and fused check sets.

    Foregrounds the failure class for a human reader: which specific check
    (its ``check_id``/``title``) is responsible, and whether it came from the
    cross-target parity checks or an embedded base/fused ``text`` report.
    """
    failing: list[tuple[str, CheckResult]] = []
    failing.extend(
        ("parity", result) for result in report.results if result.status in {"fail", "warn"}
    )
    failing.extend(
        ("base", result)
        for result in report.base_report.results
        if result.status in {"fail", "warn"}
    )
    failing.extend(
        ("fused", result)
        for result in report.fused_report.results
        if result.status in {"fail", "warn"}
    )
    return failing


# The summary line's class breakdown -- deliberately excludes "missing" (always
# named individually below, never bulk-summarized: a dropped tensor is rare and
# always worth a human's attention, unlike the routinely-large other classes).
_DELTA_SUMMARY_KLASSES: tuple[TensorDeltaKlass, ...] = (
    "changed",
    "unchanged",
    "non_comparable",
    "unexpected",
)

# Cap on individually-named delta-map lines in the text render, so a real
# model's few dozen adapter targets stay readable even if every one of them
# is notable.
_MAX_NOTABLE_DELTA_LINES = 20


def _delta_map_summary_line(deltas: Sequence[TensorDelta]) -> str:
    """Return the one-line ``Delta map: N tensors (...)`` count summary."""
    counts = aggregate_delta_summary(deltas).counts_by_klass
    parts: list[str] = []
    for klass in _DELTA_SUMMARY_KLASSES:
        count = counts.get(klass, 0)
        if count:
            parts.append(f"{count} {klass}")
    breakdown = f" ({', '.join(parts)})" if parts else ""
    return f"Delta map: {len(deltas)} tensors{breakdown}"


def _delta_map_notable(deltas: Sequence[TensorDelta]) -> list[TensorDelta]:
    """Return the delta-map tensors worth naming individually in the text render.

    ``changed``/``unexpected``/``missing`` are always worth surfacing -- each is
    a small, meaningful signal (a target that changed, an out-of-scope byte
    difference, a tensor that vanished). A bulk ``non_comparable`` sweep (e.g.
    every weight after a ``--dequantize`` fuse) is noise unless it landed on an
    adapter target, which is still worth a second look.
    """
    return [
        delta
        for delta in deltas
        if delta.klass in ("changed", "unexpected", "missing")
        or (delta.klass == "non_comparable" and delta.is_target)
    ]


def _delta_map_lines(report: ParityReport) -> list[str]:
    """Return the text-render lines for the delta map.

    A count summary plus the notable tensors, capped so a large model's
    routine bulk never floods the report. The complete per-tensor list stays
    in ``render_parity_json``.
    """
    if not report.delta_map:
        return []
    lines = ["", _delta_map_summary_line(report.delta_map)]
    notable = _delta_map_notable(report.delta_map)
    if notable:
        lines.append("")
        shown = notable[:_MAX_NOTABLE_DELTA_LINES]
        for delta in shown:
            suffix = f" -- {delta.reason}" if delta.reason else ""
            lines.append(f"  {delta.klass} {delta.tensor}{suffix}")
        remaining = len(notable) - len(shown)
        if remaining > 0:
            lines.append(f"  … ({remaining} more)")
    return lines


def render_parity_text(report: ParityReport) -> str:
    """Render a ParityReport as plain text.

    Foregrounds the verdict (or its null state plus ``reasons``), the
    agreement rates, gap, noise, first divergence, flip count, and
    adapter-applied signal, any failing static or embedded checks (the
    failure class), any worker failure's captured cause, and a summary of the
    delta map plus the notable tensors within it.
    """
    lines = [
        f"MLX Model Doctor (parity): {report.adapter.original_ref} -> {report.fused.original_ref}",
        "",
        f"Verdict: {_verdict_label(report.verdict)}",
    ]
    if report.verdict is None:
        lines.extend(f"  Reason: {reason}" for reason in report.reasons)
    lines.extend(
        [
            "",
            "Agreement:",
            f"  fused vs adapter (agree_fa): {_fmt_float(report.agree_fa)}",
            f"  fused vs base    (agree_fb): {_fmt_float(report.agree_fb)}",
            f"  gap:                         {_fmt_float(report.gap)}",
            f"  noise:                       {_fmt_float(report.noise)}",
            f"  first divergence:            {_fmt_int(report.first_divergence)}",
            f"  flip count:                  {_fmt_int(report.flip_count)}",
            f"  adapter applied:             {_fmt_bool(report.adapter_applied)}",
        ]
    )
    failing = _failing_static_checks(report)
    if failing:
        lines.extend(["", "Failing checks:"])
        for source, result in failing:
            lines.append(
                f"  {result.status.upper()} [{source}] {result.check_id}: {result.message}"
            )
    if report.worker_errors:
        lines.extend(["", "Worker errors:"])
        for role, error in sorted(report.worker_errors.items()):
            lines.append(f"  {role}: {error}")
    lines.extend(_delta_map_lines(report))
    return "\n".join(lines)


def render_parity_markdown(report: ParityReport) -> str:
    """Render a ParityReport as Markdown, mirroring ``render_parity_text``'s content."""
    lines = [
        f"# MLX Model Doctor (parity): {report.adapter.original_ref} "
        f"vs {report.fused.original_ref}",
        "",
        f"**Verdict:** {_verdict_label(report.verdict)}",
        "",
    ]
    if report.verdict is None and report.reasons:
        lines.extend(f"> {reason}" for reason in report.reasons)
        lines.append("")
    lines.extend(
        [
            "| Metric | Value |",
            "|---|---:|",
            f"| agree_fa | {_fmt_float(report.agree_fa)} |",
            f"| agree_fb | {_fmt_float(report.agree_fb)} |",
            f"| gap | {_fmt_float(report.gap)} |",
            f"| noise | {_fmt_float(report.noise)} |",
            f"| first_divergence | {_fmt_int(report.first_divergence)} |",
            "",
        ]
    )
    failing = _failing_static_checks(report)
    if failing:
        lines.extend(["## Failing checks", ""])
        for source, result in failing:
            lines.extend(
                [
                    f"### {result.status.upper()} [{source}] {result.check_id}",
                    "",
                    f"**{result.title}.** {result.message}",
                    "",
                ]
            )
    if report.worker_errors:
        lines.extend(["## Worker errors", ""])
        for role, error in sorted(report.worker_errors.items()):
            lines.append(f"- **{role}:** {error}")
        lines.append("")
    if report.delta_map:
        lines.extend(["## Delta map", "", "| Tensor | Class | Reason |", "|---|---|---|"])
        lines.extend(
            f"| {delta.tensor} | {delta.klass} | {delta.reason or ''} |"
            for delta in report.delta_map
        )
    return "\n".join(lines)
