"""Checks for conservative model memory estimates."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from mlx_model_doctor.checks.quantization import config_has_mixed_precision_quant
from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.errors import TargetError, raise_for_hf_target_error
from mlx_model_doctor.report import CheckResult

EstimateSource = Literal["config", "file_sizes", "unknown"]
MEMORY_LOWER_BOUND_KIND_DETAIL = "memory_lower_bound_kind"
MODEL_RUNTIME_MEMORY_LOWER_BOUND_KIND = "model_runtime"
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
_FP16_BYTES = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryEstimate:
    """A conservative lower-bound memory estimate."""

    lower_bound_bytes: int
    estimate_source: EstimateSource
    weight_lower_bound_bytes: int = 0
    kv_cache_lower_bound_bytes: int = 0
    unavailable_weight_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MemoryEstimateCheck:
    """Estimate a conservative lower bound for model memory use."""

    check_id: str = "text/memory.estimate"
    title: str = "Memory estimate"

    def run(self, ctx: CheckContext) -> CheckResult:
        """Return a lower-bound memory estimate for the configured context length."""
        context_length = ctx.options.context_length
        config = ctx.config_json()
        estimate: MemoryEstimate | None
        if config is not None and config_has_mixed_precision_quant(config):
            estimate = _mixed_precision_estimate(ctx, config, context_length)
        elif config is not None:
            estimate = _config_estimate(config, context_length)
            if estimate is None:
                estimate = _file_size_estimate(ctx)
        else:
            estimate = _file_size_estimate(ctx)
        if estimate is None:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message="Memory estimate skipped because of insufficient metadata.",
                details={"estimate_source": "unknown", "context_length": context_length},
            )
        if estimate.estimate_source == "unknown":
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message="Memory estimate skipped because weight file sizes are unavailable.",
                details=_estimate_details(estimate, context_length, ctx.options.max_memory_bytes),
            )

        details = _estimate_details(estimate, context_length, ctx.options.max_memory_bytes)
        max_memory_bytes = ctx.options.max_memory_bytes
        if max_memory_bytes is not None and estimate.lower_bound_bytes > max_memory_bytes:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message="Estimated lower bound memory exceeds the configured budget.",
                remediation=(
                    "Use a smaller model, stronger quantization, or a shorter context length before loading."
                ),
                details=details,
            )

        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message="Estimated lower bound memory is advisory and may be below runtime use.",
            details=details,
        )


def _estimate_details(
    estimate: MemoryEstimate,
    context_length: int,
    max_memory_bytes: int | None,
) -> dict[str, object]:
    details: dict[str, object] = {
        "estimate_source": estimate.estimate_source,
        "context_length": context_length,
        "lower_bound_bytes": estimate.lower_bound_bytes,
        "weight_lower_bound_bytes": estimate.weight_lower_bound_bytes,
        "kv_cache_lower_bound_bytes": estimate.kv_cache_lower_bound_bytes,
    }
    # A partial weight sum (some shards unsized) understates the bound, so it is not a
    # trustworthy smoke-gate floor: report it diagnostically but withhold the gate marker
    # that runners/smoke.py keys off. Mirrors the mixed-precision posture in 0032.
    if estimate.lower_bound_bytes > 0 and not estimate.unavailable_weight_paths:
        details[MEMORY_LOWER_BOUND_KIND_DETAIL] = MODEL_RUNTIME_MEMORY_LOWER_BOUND_KIND
    if estimate.estimate_source == "file_sizes":
        details["measured_bytes"] = estimate.weight_lower_bound_bytes
    if estimate.unavailable_weight_paths:
        details["unavailable_weight_paths"] = estimate.unavailable_weight_paths
    if max_memory_bytes is not None:
        details["max_memory_bytes"] = max_memory_bytes
    return details


def _config_estimate(config: dict[str, object], context_length: int) -> MemoryEstimate | None:
    hidden_size = _positive_int(config, "hidden_size")
    num_layers = _positive_int(config, "num_hidden_layers")
    vocab_size = _positive_int(config, "vocab_size")
    intermediate_size = _positive_int(config, "intermediate_size")
    if hidden_size is None or num_layers is None:
        return None

    bytes_per_weight = _bytes_per_weight(config)
    parameterish_count = num_layers * (4 * hidden_size * hidden_size)
    if vocab_size is not None:
        parameterish_count += vocab_size * hidden_size
    if intermediate_size is not None:
        parameterish_count += num_layers * (3 * hidden_size * intermediate_size)
    weight_lower_bound = int(parameterish_count * bytes_per_weight)
    kv_cache_lower_bound = _kv_cache_bytes(config, context_length, hidden_size, num_layers)
    lower_bound = weight_lower_bound + kv_cache_lower_bound
    if lower_bound <= 0:
        return None
    return MemoryEstimate(
        lower_bound_bytes=lower_bound,
        estimate_source="config",
        weight_lower_bound_bytes=weight_lower_bound,
        kv_cache_lower_bound_bytes=kv_cache_lower_bound,
    )


def _bytes_per_weight(config: dict[str, object]) -> float:
    quantization = config.get("quantization")
    if isinstance(quantization, Mapping):
        bits = _positive_number(quantization.get("bits"))
        if bits is not None:
            return bits / 8
    return float(_FP16_BYTES)


def _kv_cache_bytes(
    config: dict[str, object],
    context_length: int,
    hidden_size: int,
    num_layers: int,
) -> int:
    return 2 * num_layers * context_length * _kv_hidden_size(config, hidden_size) * _FP16_BYTES


def _kv_hidden_size(config: dict[str, object], hidden_size: int) -> int:
    num_key_value_heads = _positive_int(config, "num_key_value_heads")
    head_dim = _positive_int(config, "head_dim")
    if head_dim is None:
        num_attention_heads = _positive_int(config, "num_attention_heads")
        if num_attention_heads is not None and hidden_size % num_attention_heads == 0:
            head_dim = hidden_size // num_attention_heads
    if num_key_value_heads is not None and head_dim is not None:
        return num_key_value_heads * head_dim
    return hidden_size


def _measure_weights(ctx: CheckContext) -> tuple[int, tuple[str, ...]] | None:
    try:
        files = ctx.target.list_files()
    except TargetError as exc:
        raise_for_hf_target_error(exc)
        return None
    return _measured_weight_bytes(ctx, files)


def _file_size_estimate(ctx: CheckContext) -> MemoryEstimate | None:
    measured = _measure_weights(ctx)
    if measured is None:
        return None
    measured_bytes, unavailable_weight_paths = measured
    if measured_bytes <= 0:
        if unavailable_weight_paths:
            return MemoryEstimate(
                lower_bound_bytes=0,
                estimate_source="unknown",
                unavailable_weight_paths=unavailable_weight_paths,
            )
        return None
    return MemoryEstimate(
        lower_bound_bytes=measured_bytes,
        estimate_source="file_sizes",
        weight_lower_bound_bytes=measured_bytes,
        unavailable_weight_paths=unavailable_weight_paths,
    )


def _kv_cache_for_config(config: dict[str, object], context_length: int) -> int:
    hidden_size = _positive_int(config, "hidden_size")
    num_layers = _positive_int(config, "num_hidden_layers")
    if hidden_size is None or num_layers is None:
        return 0
    return _kv_cache_bytes(config, context_length, hidden_size, num_layers)


def _mixed_precision_estimate(
    ctx: CheckContext, config: dict[str, object], context_length: int
) -> MemoryEstimate:
    measured = _measure_weights(ctx)
    if measured is None:
        return MemoryEstimate(lower_bound_bytes=0, estimate_source="unknown")
    measured_bytes, unavailable_weight_paths = measured
    if measured_bytes <= 0 or unavailable_weight_paths:
        return MemoryEstimate(
            lower_bound_bytes=0,
            estimate_source="unknown",
            unavailable_weight_paths=unavailable_weight_paths,
        )
    kv_cache = _kv_cache_for_config(config, context_length)
    return MemoryEstimate(
        lower_bound_bytes=measured_bytes + kv_cache,
        estimate_source="file_sizes",
        weight_lower_bound_bytes=measured_bytes,
        kv_cache_lower_bound_bytes=kv_cache,
    )


def _measured_weight_bytes(ctx: CheckContext, files: Sequence[str]) -> tuple[int, tuple[str, ...]]:
    weight_files = [path for path in files if _is_weight_file(path)]
    safetensors = [path for path in weight_files if path.endswith(".safetensors")]
    selected = safetensors or weight_files
    total = 0
    unavailable_paths: list[str] = []
    for path in selected:
        try:
            size = ctx.target.size(path)
        except TargetError as exc:
            raise_for_hf_target_error(exc)
            unavailable_paths.append(path)
            continue
        if size is None:
            unavailable_paths.append(path)
            continue
        total += size
    return total, tuple(unavailable_paths)


def _is_weight_file(path: str) -> bool:
    return path.endswith(_WEIGHT_SUFFIXES)


def _positive_int(config: dict[str, object], key: str) -> int | None:
    value = config.get(key)
    if type(value) is int and value > 0:
        return value
    return None


def _positive_number(value: object) -> float | None:
    if (type(value) is int or type(value) is float) and value > 0:
        return float(value)
    return None
