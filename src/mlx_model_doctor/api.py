"""Public Python API."""

import platform
import sys
from dataclasses import dataclass
from importlib import metadata, util
from pathlib import Path
from typing import cast

from mlx_model_doctor.context import CheckContext, CheckOptions
from mlx_model_doctor.errors import DependencyError
from mlx_model_doctor.parity.checks import (
    AdapterConfigCheck,
    FusedTargetConsistencyCheck,
    ParityCheck,
    TargetCoverageCheck,
    TokenizerIdentityCheck,
    resolve_targets_against_base,
    run_parity_checks,
)
from mlx_model_doctor.parity.context import ParityContext, ParityTargets, tokenizers_match
from mlx_model_doctor.parity.deltamap import TensorDelta, delta_map
from mlx_model_doctor.parity.fixtures import DEFAULT_FIXTURE_ID, FixtureRef, get_fixture
from mlx_model_doctor.parity.oracle import (
    PARITY_GROSS_FLOOR,
    PARITY_K,
    PARITY_PASS_FLOOR,
    ROLE_BASE,
    ROLE_FUSED,
    ROLE_NOISE,
    ROLE_REFERENCE,
    ParityVerdict,
    assemble_verdict_inputs,
    decide_verdict,
    first_divergence,
    flip_count,
)
from mlx_model_doctor.parity.orchestrator import Launcher, SubprocessLauncher, run_parity_workers
from mlx_model_doctor.parity.report import (
    ParityReport,
    PhaseOutcome,
    RuntimeProvenance,
    WorkerStatusValue,
)
from mlx_model_doctor.parity.sources import ResolvedSources, SnapshotDownloader, resolve_sources
from mlx_model_doctor.parity.worker import WorkerSpec
from mlx_model_doctor.plugins import DoctorPlugin, get_plugin
from mlx_model_doctor.report import CheckResult, DoctorReport, zero_check_reason_for
from mlx_model_doctor.runners.smoke import run_smoke_checks
from mlx_model_doctor.runners.static import run_static_checks
from mlx_model_doctor.targets import HfHubProtocol, HfTarget, LocalTarget, ModelTarget

_ADAPTER_CONFIG_CHECK_ID = "parity/adapter.config_well_formed"
_TOKENIZER_CHECK_ID = "parity/tokenizer.identity"
_MISSING_TARGET_CHECK_IDS = frozenset({"parity/target.coverage", "parity/fused.target_consistency"})

# The four cross-target static checks, in stable execution order (F3/F5/F6).
# Cast mirrors ``plugins.text``'s ModelCheck tuple: the frozen check dataclasses
# implement the ``ParityCheck`` Protocol structurally, which a heterogeneous
# tuple literal does not convey to the type checker on its own.
_PARITY_CHECKS: tuple[ParityCheck, ...] = cast(
    "tuple[ParityCheck, ...]",
    (
        AdapterConfigCheck(),
        TargetCoverageCheck(),
        FusedTargetConsistencyCheck(),
        TokenizerIdentityCheck(),
    ),
)

# Fallback ceiling for worker-artifact vocab-range validation when the base
# config declares no usable vocab_size (real-launcher path only).
_DEFAULT_VOCAB_CEILING = 1_000_000


def check_local_model(
    path: str | Path,
    *,
    options: CheckOptions | None = None,
    plugin_name: str = "text",
) -> DoctorReport:
    """Check a local model repository."""
    return _check_target(LocalTarget(path), options=options, plugin_name=plugin_name)


def check_hf_model(
    repo_id: str,
    *,
    options: CheckOptions | None = None,
    plugin_name: str = "text",
    hub: HfHubProtocol | None = None,
) -> DoctorReport:
    """Check a Hugging Face model repository."""
    return _check_target(HfTarget(repo_id, hub=hub), options=options, plugin_name=plugin_name)


def _check_target(
    target: ModelTarget,
    *,
    options: CheckOptions | None,
    plugin_name: str,
) -> DoctorReport:
    plugin = get_plugin(plugin_name)
    ctx = CheckContext(
        target=target,
        options=options if options is not None else _default_options(),
    )
    results = _run_plugin_checks(ctx, plugin)
    return DoctorReport(
        target=target.name,
        source=target.source,
        plugin=plugin.name,
        results=results,
        zero_check_reason=None if results else zero_check_reason_for(plugin.name),
    )


def _run_plugin_checks(ctx: CheckContext, plugin: DoctorPlugin) -> list[CheckResult]:
    static_results = run_static_checks(ctx, plugin.static_checks())
    weight_checks = plugin.weight_checks() if ctx.options.include_weights else ()
    weight_results = run_static_checks(ctx, weight_checks)
    smoke_checks = plugin.smoke_checks() if ctx.options.smoke else ()
    return [
        *static_results,
        *weight_results,
        *run_smoke_checks(ctx, smoke_checks, static_results),
    ]


def _default_options() -> CheckOptions:
    return CheckOptions(
        max_memory_bytes=None,
        context_length=4096,
        include_weights=True,
        smoke=False,
        verbosity="normal",
    )


# --- Adapter/fused-model parity API (F1/F4/F10) ---------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ParityOptions:
    """Injectable seams for :func:`check_adapter_parity`.

    ``launcher`` runs the four teacher-forced worker loads; when ``None`` a real
    parent-supervised :class:`SubprocessLauncher` is built (the offline tests
    inject a fake so no MLX/subprocess is needed). ``check_options`` tunes the
    embedded ``text`` pipeline and the cross-target checks. ``fixture_id`` selects
    the pinned token-id fixture, unless ``fixture`` is set, in which case it is used
    directly instead of resolving ``fixture_id`` through :func:`get_fixture` -- the
    seam a caller building a real-repository fixture (e.g. via
    :func:`~mlx_model_doctor.parity.fixtures.build_fixture_from_prompts`) uses to
    supply it. ``allow_network``/``downloader`` gate and back the Hugging Face
    snapshot resolution (F10).
    """

    launcher: Launcher | None = None
    check_options: CheckOptions | None = None
    fixture_id: str = DEFAULT_FIXTURE_ID
    fixture: "tuple[FixtureRef, tuple[tuple[int, ...], ...]] | None" = None
    allow_network: bool = False
    downloader: SnapshotDownloader | None = None


def check_adapter_parity(
    *,
    base: str,
    adapter: str,
    fused: str,
    options: ParityOptions | None = None,
) -> ParityReport:
    """Verify that a fused model preserved the behavior a LoRA adapter learned.

    Resolves base/adapter/fused once to pinned local snapshots (F10), embeds the
    ``text`` reports for base and fused, runs the cross-target static checks, and
    -- unless a blocking static defect or tokenizer mismatch gates it -- runs the
    four teacher-forced worker loads and the three-way relative oracle. The
    adapter-applied signal comes from the base+adapter reference load only (F1);
    the byte-compare delta map is diagnostic. Worker/IPC failures are folded into
    a returned crash-carrying report; only setup failures (a missing ``mlx-lm``,
    an unreadable target, an HF source without ``allow_network``) raise
    :class:`~mlx_model_doctor.errors.ModelDoctorError`.
    """
    options = options if options is not None else ParityOptions()
    check_options = (
        options.check_options if options.check_options is not None else _default_options()
    )

    sources = resolve_sources(
        base, adapter, fused, allow_network=options.allow_network, downloader=options.downloader
    )

    base_target = LocalTarget(sources.base.path)
    adapter_target = LocalTarget(sources.adapter.path)
    fused_target = LocalTarget(sources.fused.path)

    base_report = _check_target(base_target, options=check_options, plugin_name="text")
    fused_report = _check_target(fused_target, options=check_options, plugin_name="text")

    pctx = ParityContext(
        targets=ParityTargets(base=base_target, adapter=adapter_target, fused=fused_target),
        options=check_options,
    )
    results, crashed = run_parity_checks(pctx, _PARITY_CHECKS)

    fixture_ref, token_ids = (
        options.fixture if options.fixture is not None else get_fixture(options.fixture_id)
    )
    tokenizer_fingerprint = pctx.base_tokenizer_fingerprint()
    fixture_match = tokenizers_match(fixture_ref.tokenizer_fingerprint, tokenizer_fingerprint)
    fixture_mismatch = not fixture_match.matched
    delta_result = _build_delta_map(pctx, base_target, fused_target, sources)

    reasons: list[str] = []
    phase_outcomes: dict[str, PhaseOutcome] = {"delta_map": delta_result.phase}
    if delta_result.reason is not None and delta_result.phase == "error":
        reasons.append(delta_result.reason)

    embedded_fail = bool(base_report.summary["fail"]) or bool(fused_report.summary["fail"])
    adapter_config_fail = _has_fail(results, _ADAPTER_CONFIG_CHECK_ID)
    missing_target_fail = any(_has_fail(results, cid) for cid in _MISSING_TARGET_CHECK_IDS)
    tokenizer_mismatch = _has_fail(results, _TOKENIZER_CHECK_ID)

    if crashed:
        phase_outcomes["static"] = "error"
        reasons.append("a cross-target parity check crashed; the run cannot be trusted.")
    elif embedded_fail or adapter_config_fail or missing_target_fail:
        phase_outcomes["static"] = "blocking_fail"
        reasons.append("a blocking static defect prevented a valid parity run.")
    else:
        phase_outcomes["static"] = "ok"

    phase_outcomes["tokenizer_gate"] = "blocking_fail" if tokenizer_mismatch else "ok"

    if fixture_mismatch:
        reasons.append(
            "the parity fixture was built for a different tokenizer than the base model "
            f"({fixture_match.reason}); build a fixture for this model's tokenizer with "
            "build_fixture_from_prompts and pass it via ParityOptions.fixture instead of "
            "the built-in default."
        )

    runtime_gated = (
        crashed
        or embedded_fail
        or adapter_config_fail
        or missing_target_fail
        or tokenizer_mismatch
        or fixture_mismatch
    )

    runtime = (
        _skipped_runtime(reasons, tokenizer_mismatch=tokenizer_mismatch)
        if runtime_gated
        else _run_runtime(pctx, sources, fixture_ref, token_ids, options, reasons)
    )
    phase_outcomes.update(runtime.phase_outcomes)

    return ParityReport(
        base=sources.base,
        adapter=sources.adapter,
        fused=sources.fused,
        tokenizer_fingerprint=tokenizer_fingerprint,
        fixture=fixture_ref,
        tool_version=_tool_version(),
        mlx_version=_dist_version("mlx"),
        mlx_lm_version=_dist_version("mlx-lm"),
        provenance=_runtime_provenance(),
        base_report=base_report,
        fused_report=fused_report,
        results=tuple(results),
        verdict=runtime.verdict,
        agree_fa=runtime.agree_fa,
        agree_fb=runtime.agree_fb,
        gap=runtime.gap,
        noise=runtime.noise,
        first_divergence=runtime.first_divergence,
        flip_count=runtime.flip_count,
        adapter_applied=runtime.adapter_applied,
        delta_map=delta_result.deltas,
        phase_outcomes=phase_outcomes,
        worker_status=runtime.worker_status,
        peak_bytes=runtime.peak_bytes,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class _RuntimeOutcome:
    """The runtime-phase result folded into a :class:`ParityReport`."""

    verdict: ParityVerdict | None
    agree_fa: float | None
    agree_fb: float | None
    gap: float | None
    noise: float | None
    first_divergence: int | None
    flip_count: int | None
    adapter_applied: bool | None
    worker_status: dict[str, WorkerStatusValue]
    peak_bytes: dict[str, int | None]
    phase_outcomes: dict[str, PhaseOutcome]


_ALL_ROLES = (ROLE_BASE, ROLE_NOISE, ROLE_REFERENCE, ROLE_FUSED)


def _skipped_runtime(reasons: list[str], *, tokenizer_mismatch: bool) -> _RuntimeOutcome:
    """Build the runtime outcome for a run gated before any worker ran."""
    if tokenizer_mismatch:
        reasons.append("base and fused tokenizers differ; the oracle's outputs are not comparable.")
    worker_status: dict[str, WorkerStatusValue] = dict.fromkeys(_ALL_ROLES, "skipped")
    peak_bytes: dict[str, int | None] = dict.fromkeys(_ALL_ROLES)
    return _RuntimeOutcome(
        verdict=None,
        agree_fa=None,
        agree_fb=None,
        gap=None,
        noise=None,
        first_divergence=None,
        flip_count=None,
        adapter_applied=None,
        worker_status=worker_status,
        peak_bytes=peak_bytes,
        phase_outcomes={"reference": "skipped", "oracle": "skipped"},
    )


def _voided_runtime(
    *,
    adapter_applied: bool | None,
    worker_status: dict[str, WorkerStatusValue],
    peak_bytes: dict[str, int | None],
    phases: dict[str, PhaseOutcome],
) -> _RuntimeOutcome:
    """Build a runtime outcome whose oracle was not computed (null verdict/metrics)."""
    return _RuntimeOutcome(
        verdict=None,
        agree_fa=None,
        agree_fb=None,
        gap=None,
        noise=None,
        first_divergence=None,
        flip_count=None,
        adapter_applied=adapter_applied,
        worker_status=worker_status,
        peak_bytes=peak_bytes,
        phase_outcomes=phases,
    )


def _run_runtime(
    pctx: ParityContext,
    sources: ResolvedSources,
    fixture_ref: FixtureRef,
    token_ids: tuple[tuple[int, ...], ...],
    options: ParityOptions,
    reasons: list[str],
) -> _RuntimeOutcome:
    """Run the four worker loads and the three-way oracle (F1)."""
    launcher = _build_launcher(pctx, options)
    specs = _build_worker_specs(sources, fixture_ref, token_ids)
    outcomes = run_parity_workers(specs, launcher=launcher)
    by_role = {outcome.role: outcome for outcome in outcomes}
    worker_status: dict[str, WorkerStatusValue] = {
        outcome.role: outcome.status for outcome in outcomes
    }
    peak_bytes: dict[str, int | None] = {outcome.role: outcome.peak_bytes for outcome in outcomes}

    reference = by_role.get(ROLE_REFERENCE)
    reference_ok = reference is not None and reference.status == "ok"
    # adapter_applied is derived from the base+adapter reference ONLY (F1). An
    # error outcome carries adapter_applied=None, so this is already None exactly
    # when the reference worker itself errored or is absent.
    adapter_applied = reference.adapter_applied if reference is not None else None

    if any(outcome.status == "error" for outcome in outcomes):
        reasons.append("one or more parity workers failed; the oracle could not be computed.")
        # Keep the reference-derived signal: a crash in a non-reference worker
        # (base_repeat/fused) must not erase a valid reference verdict (F1). The
        # run is still cannot-determine (exit 2) via the worker-status error.
        return _voided_runtime(
            adapter_applied=adapter_applied,
            worker_status=worker_status,
            peak_bytes=peak_bytes,
            phases={"reference": "ok" if reference_ok else "error", "oracle": "error"},
        )
    if adapter_applied is None:
        reasons.append(
            "the base+adapter reference could not be validated; adapter application is "
            "indeterminate (inconclusive)."
        )
        return _voided_runtime(
            adapter_applied=None,
            worker_status=worker_status,
            peak_bytes=peak_bytes,
            phases={"reference": "unsupported", "oracle": "skipped"},
        )
    if adapter_applied is False:
        reasons.append(
            "the base+adapter reference shows no effective adapter contribution; "
            "the fuse cannot be scored."
        )
        return _voided_runtime(
            adapter_applied=False,
            worker_status=worker_status,
            peak_bytes=peak_bytes,
            phases={"reference": "blocking_fail", "oracle": "skipped"},
        )

    scored = list(fixture_ref.scored_positions)
    vi = assemble_verdict_inputs(outcomes, scored=scored)
    verdict = decide_verdict(
        agree_fa=vi.agree_fa,
        agree_fb=vi.agree_fb,
        gap=vi.gap,
        noise=vi.noise,
        k=PARITY_K,
        pass_floor=PARITY_PASS_FLOOR,
        gross_floor=PARITY_GROSS_FLOOR,
    )
    if verdict is ParityVerdict.INCONCLUSIVE:
        reasons.append(
            "the fused model's outputs do not clearly track the base+adapter reference; "
            "adapter parity could not be confirmed (the fuse may partially degrade the "
            "adapter)."
        )
    reference_argmax = by_role[ROLE_REFERENCE].argmax
    fused_argmax = by_role[ROLE_FUSED].argmax
    first_div: int | None = None
    flips: int | None = None
    if reference_argmax is not None and fused_argmax is not None:
        first_div = first_divergence(fused_argmax, reference_argmax, scored)
        flips = flip_count(fused_argmax, reference_argmax, scored)
    return _RuntimeOutcome(
        verdict=verdict,
        agree_fa=vi.agree_fa,
        agree_fb=vi.agree_fb,
        gap=vi.gap,
        noise=vi.noise,
        first_divergence=first_div,
        flip_count=flips,
        adapter_applied=True,
        worker_status=worker_status,
        peak_bytes=peak_bytes,
        phase_outcomes={"reference": "ok", "oracle": "ok"},
    )


def _build_launcher(pctx: ParityContext, options: ParityOptions) -> Launcher:
    """Return the injected launcher, or build a real subprocess launcher (F2)."""
    if options.launcher is not None:
        return options.launcher
    _require_mlx_lm_available()
    return SubprocessLauncher(vocab_size=_base_vocab_size(pctx))


def _build_worker_specs(
    sources: ResolvedSources, fixture_ref: FixtureRef, token_ids: tuple[tuple[int, ...], ...]
) -> list[WorkerSpec]:
    """Build the four worker specs over the pinned paths (base thrice, fused once)."""
    base_path = sources.base.path
    return [
        WorkerSpec(
            model_path=base_path,
            adapter_path=None,
            token_ids=token_ids,
            fixture_id=fixture_ref.id,
            role=ROLE_BASE,
        ),
        WorkerSpec(
            model_path=base_path,
            adapter_path=None,
            token_ids=token_ids,
            fixture_id=fixture_ref.id,
            role=ROLE_NOISE,
        ),
        WorkerSpec(
            model_path=base_path,
            adapter_path=sources.adapter.path,
            token_ids=token_ids,
            fixture_id=fixture_ref.id,
            role=ROLE_REFERENCE,
        ),
        WorkerSpec(
            model_path=sources.fused.path,
            adapter_path=None,
            token_ids=token_ids,
            fixture_id=fixture_ref.id,
            role=ROLE_FUSED,
        ),
    ]


@dataclass(frozen=True, slots=True, kw_only=True)
class _DeltaMapResult:
    """The diagnostic delta map plus its phase outcome and an optional failure reason."""

    deltas: tuple[TensorDelta, ...]
    phase: PhaseOutcome
    reason: str | None = None


def _build_delta_map(
    pctx: ParityContext,
    base_target: LocalTarget,
    fused_target: LocalTarget,
    sources: ResolvedSources,
) -> _DeltaMapResult:
    """Build the diagnostic delta map, tracking the ``delta_map`` phase (F11, spec Sec.9).

    A byte-compare runs only for a local base/fused pair; an HF pair is
    ``"skipped"`` (note-only), a missing header is ``"skipped"``, a completed
    compare is ``"ok"``, and a caught read/parse failure is ``"error"`` with a
    reason -- never a silent empty result.
    """
    if sources.base.source != "local" or sources.fused.source != "local":
        return _DeltaMapResult(deltas=_hf_non_comparable_delta(pctx), phase="skipped")
    base_hdr = pctx.base.safetensors_header()
    fused_hdr = pctx.fused.safetensors_header()
    if base_hdr is None or fused_hdr is None:
        return _DeltaMapResult(
            deltas=(),
            phase="skipped",
            reason="delta map skipped: base or fused has no readable safetensors header.",
        )
    targets = _delta_targets(pctx)
    try:
        deltas = tuple(
            delta_map(
                base_target, fused_target, base_hdr=base_hdr, fused_hdr=fused_hdr, targets=targets
            )
        )
    except (ValueError, OSError) as exc:
        return _DeltaMapResult(
            deltas=(), phase="error", reason=f"delta map could not compare tensor bytes: {exc}"
        )
    return _DeltaMapResult(deltas=deltas, phase="ok")


def _hf_non_comparable_delta(pctx: ParityContext) -> tuple[TensorDelta, ...]:
    """Emit a per-tensor ``not byte-comparable (HF)`` note for a Hugging Face pair (F11)."""
    base_hdr = pctx.base.safetensors_header()
    if base_hdr is None:
        return ()
    return tuple(
        TensorDelta(tensor=name, klass="non_comparable", reason="not byte-comparable (HF)")
        for name in sorted(base_hdr.tensor_names())
    )


def _delta_targets(pctx: ParityContext) -> tuple[str, ...]:
    """Return the resolver's expected-change target set, or empty when unresolved."""
    outcome = resolve_targets_against_base(pctx)
    if isinstance(outcome, str):
        return ()
    return tuple(sorted(outcome.covered))


def _has_fail(results: "tuple[CheckResult, ...] | list[CheckResult]", check_id: str) -> bool:
    """Return whether a parity check result for ``check_id`` is a ``fail``."""
    return any(result.check_id == check_id and result.status == "fail" for result in results)


def _base_vocab_size(pctx: ParityContext) -> int:
    """Return the base config's ``vocab_size`` for artifact validation, or a safe ceiling."""
    config = pctx.base.config_json()
    if config is not None:
        value = config.get("vocab_size")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return _DEFAULT_VOCAB_CEILING


def _require_mlx_lm_available() -> None:
    """Raise a clear dependency error when ``mlx-lm`` is not importable (setup failure)."""
    if util.find_spec("mlx_lm") is None:
        raise DependencyError(
            missing_package="mlx_lm",
            extra_name="mlx-lm",
            executable=sys.executable,
            message="mlx-lm is required to run adapter-parity workers; install the 'mlx-lm' extra.",
        )


def _runtime_provenance() -> RuntimeProvenance:
    """Record honest host provenance; measurement-device fields stay null when not measured."""
    return RuntimeProvenance(
        chip=platform.processor() or None,
        os=platform.platform(),
        backend=None,
        weight_dtype=None,
        compute_note=(
            "A PASS verdict reflects scoped top-token agreement, not proof of all learned behavior."
        ),
    )


def _tool_version() -> str:
    """Return the installed package version, or ``'unknown'`` when unavailable."""
    try:
        return metadata.version("mlx-model-doctor")
    except metadata.PackageNotFoundError:
        return "unknown"


def _dist_version(name: str) -> str | None:
    """Return an installed distribution's version, or ``None`` when it is absent."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
