"""Generic smoke check runner with a memory pre-flight gate."""

import importlib
from collections.abc import Mapping, Sequence
from typing import cast

from mlx_model_doctor.checks.base import ModelCheck
from mlx_model_doctor.checks.memory import (
    MEMORY_LOWER_BOUND_KIND_DETAIL,
    MODEL_RUNTIME_MEMORY_LOWER_BOUND_KIND,
)
from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.memory import smoke_budget_from_device_info
from mlx_model_doctor.report import CheckResult
from mlx_model_doctor.runners.core import run_checks


def run_smoke_checks(
    ctx: CheckContext,
    checks: Sequence[ModelCheck],
    prior_results: Sequence[CheckResult],
) -> list[CheckResult]:
    """Run smoke checks when requested, isolating unexpected check crashes."""
    if not ctx.options.smoke:
        return []

    gate_result = _memory_gate_result(ctx, checks, prior_results)
    if gate_result is not None:
        return [gate_result]

    return run_checks(ctx, checks)


def _memory_gate_result(
    ctx: CheckContext,
    checks: Sequence[ModelCheck],
    prior_results: Sequence[CheckResult],
) -> CheckResult | None:
    estimate = _memory_lower_bound(prior_results)
    budget = _smoke_budget_bytes(ctx)
    if estimate is None or budget is None or estimate.lower_bound_bytes <= budget.bytes:
        return None

    namespace = _result_namespace(prior_results) or _check_namespace(checks) or "text"
    return CheckResult(
        check_id=f"{namespace}/smoke.memory_budget",
        title="Smoke memory budget",
        status="fail",
        severity="high",
        message="Smoke check refused because estimated lower-bound memory exceeds the smoke budget.",
        remediation="Use a smaller model, stronger quantization, or a higher explicit --max-memory budget.",
        details={
            "estimate_source": estimate.estimate_source,
            "memory_lower_bound_bytes": estimate.lower_bound_bytes,
            "smoke_budget_bytes": budget.bytes,
            "smoke_budget_source": budget.source,
        },
    )


class _MemoryEstimate:
    def __init__(self, *, lower_bound_bytes: int, estimate_source: str) -> None:
        self.lower_bound_bytes = lower_bound_bytes
        self.estimate_source = estimate_source


class _SmokeBudget:
    def __init__(self, *, bytes: int, source: str) -> None:
        self.bytes = bytes
        self.source = source


def _memory_lower_bound(results: Sequence[CheckResult]) -> _MemoryEstimate | None:
    for result in results:
        if (
            result.details.get(MEMORY_LOWER_BOUND_KIND_DETAIL)
            != MODEL_RUNTIME_MEMORY_LOWER_BOUND_KIND
        ):
            continue
        lower_bound = result.details.get("lower_bound_bytes")
        if type(lower_bound) is not int or lower_bound <= 0:
            return None
        estimate_source = result.details.get("estimate_source")
        return _MemoryEstimate(
            lower_bound_bytes=lower_bound,
            estimate_source=estimate_source if isinstance(estimate_source, str) else "unknown",
        )
    return None


def _smoke_budget_bytes(ctx: CheckContext) -> _SmokeBudget | None:
    if ctx.options.max_memory_bytes is not None:
        return _SmokeBudget(
            bytes=ctx.options.max_memory_bytes,
            source="options.max_memory_bytes",
        )

    device_info = _mlx_device_info()
    if device_info is None:
        return None
    budget = smoke_budget_from_device_info(device_info)
    if budget is None:
        return None
    return _SmokeBudget(
        bytes=budget,
        source="mlx.max_recommended_working_set_size",
    )


def _mlx_device_info() -> Mapping[str, object] | None:
    try:
        mx = importlib.import_module("mlx.core")
    except ImportError:
        return None
    try:
        device_info = mx.device_info()
    except Exception:
        return None
    if isinstance(device_info, Mapping):
        return cast("Mapping[str, object]", device_info)
    return None


def _result_namespace(results: Sequence[CheckResult]) -> str | None:
    for result in results:
        namespace, separator, _name = result.check_id.partition("/")
        if separator:
            return namespace
    return None


def _check_namespace(checks: Sequence[ModelCheck]) -> str | None:
    for check in checks:
        namespace, separator, _name = check.check_id.partition("/")
        if separator:
            return namespace
    return None
