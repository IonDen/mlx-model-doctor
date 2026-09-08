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
from dataclasses import dataclass, field
from typing import Literal, cast

from mlx_model_doctor.parity.context import TokenizerFingerprint
from mlx_model_doctor.parity.deltamap import TensorDelta
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

    def __post_init__(self) -> None:
        """Copy caller-owned collections so the report is stable."""
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "delta_map", tuple(self.delta_map))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "phase_outcomes", dict(self.phase_outcomes))
        object.__setattr__(self, "worker_status", dict(self.worker_status))
        object.__setattr__(self, "peak_bytes", dict(self.peak_bytes))


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
    }


def render_parity_json(report: ParityReport) -> str:
    """Render a ParityReport as stable JSON."""
    return json.dumps(parity_report_to_dict(report), indent=2, sort_keys=True)
