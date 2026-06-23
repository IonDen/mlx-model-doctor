import json
from dataclasses import replace

from mlx_model_doctor.checks.memory import MemoryEstimateCheck
from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.errors import TargetError
from tests.fakes import FakeTarget, check_options, context_for_files

BASE_CONFIG = {
    "hidden_size": 4096,
    "num_hidden_layers": 2,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 1000,
    "intermediate_size": 11008,
    "quantization": {"bits": 4},
}

MIXED_CONFIG = {
    **BASE_CONFIG,
    "quantization": {"bits": 4, "group_size": 64, "model.layers.0.mlp": {"bits": 8}},
}


def test_memory_estimate_check_uses_config_and_includes_kv_cache_term() -> None:
    result = MemoryEstimateCheck().run(_context_for_config(BASE_CONFIG, context_length=16))

    hidden_size = 4096
    layers = 2
    vocab_size = 1000
    intermediate_size = 11008
    bits = 4
    bytes_per_weight = bits / 8
    kv_hidden_size = 8 * 128
    expected_weights = int(
        (
            vocab_size * hidden_size
            + layers * (4 * hidden_size * hidden_size + 3 * hidden_size * intermediate_size)
        )
        * bytes_per_weight
    )
    expected_kv = 2 * layers * 16 * kv_hidden_size * 2

    assert result.status == "warn"
    assert result.severity == "low"
    assert result.details["estimate_source"] == "config"
    assert result.details["context_length"] == 16
    assert result.details["memory_lower_bound_kind"] == "model_runtime"
    assert result.details["kv_cache_lower_bound_bytes"] == expected_kv
    assert result.details["lower_bound_bytes"] == expected_weights + expected_kv


def test_memory_estimate_check_warns_high_when_lower_bound_exceeds_budget() -> None:
    result = MemoryEstimateCheck().run(
        _context_for_config(BASE_CONFIG, max_memory_bytes=1024, context_length=16)
    )

    assert result.status == "warn"
    assert result.severity == "high"
    assert "lower bound" in result.message
    assert result.remediation is not None
    assert "context" in result.remediation


def test_memory_estimate_check_warns_low_when_lower_bound_is_within_budget() -> None:
    result = MemoryEstimateCheck().run(
        _context_for_config(BASE_CONFIG, max_memory_bytes=10**12, context_length=16)
    )

    assert result.status == "warn"
    assert result.severity == "low"
    assert "lower bound" in result.message


def test_memory_estimate_check_uses_file_size_fallback_when_config_is_insufficient() -> None:
    result = MemoryEstimateCheck().run(
        context_for_files(
            {
                "config.json": b'{"hidden_size":4096}',
                "model-00001-of-00002.safetensors": b"a" * 10,
                "model-00002-of-00002.safetensors": b"b" * 20,
            }
        )
    )

    assert result.status == "warn"
    assert result.severity == "low"
    assert result.details["estimate_source"] == "file_sizes"
    assert result.details["lower_bound_bytes"] == 30
    # A fully-measured no-config sum is a trustworthy smoke-gate lower bound.
    assert result.details["memory_lower_bound_kind"] == "model_runtime"


def test_memory_estimate_check_uses_config_when_vocab_size_is_missing() -> None:
    config = {key: value for key, value in BASE_CONFIG.items() if key != "vocab_size"}

    result = MemoryEstimateCheck().run(_context_for_config(config, context_length=16))

    assert result.status == "warn"
    assert result.severity == "low"
    assert result.details["estimate_source"] == "config"
    assert result.details["lower_bound_bytes"] > 0
    assert result.details["weight_lower_bound_bytes"] > 0
    assert result.details["kv_cache_lower_bound_bytes"] == 131072


def test_memory_estimate_check_uses_config_when_intermediate_size_is_missing() -> None:
    config = {key: value for key, value in BASE_CONFIG.items() if key != "intermediate_size"}

    result = MemoryEstimateCheck().run(_context_for_config(config, context_length=16))

    assert result.status == "warn"
    assert result.severity == "low"
    assert result.details["estimate_source"] == "config"
    assert result.details["lower_bound_bytes"] > 0
    assert result.details["weight_lower_bound_bytes"] > 0
    assert result.details["kv_cache_lower_bound_bytes"] == 131072


def test_memory_estimate_check_keeps_measurable_file_sizes_when_one_size_fails() -> None:
    target = UnavailableSizeTarget(
        files={
            "config.json": b'{"model_type":"llama"}',
            "model-00001-of-00002.safetensors": b"a" * 10,
            "model-00002-of-00002.safetensors": b"b" * 20,
        }
    )

    result = MemoryEstimateCheck().run(CheckContext(target=target, options=check_options()))

    assert result.status == "warn"
    assert result.severity == "low"
    assert result.details["estimate_source"] == "file_sizes"
    assert result.details["lower_bound_bytes"] == 10
    assert result.details["unavailable_weight_paths"] == ("model-00002-of-00002.safetensors",)
    # The partial sum is reported diagnostically only; it must not be smoke-gate-trusted.
    assert "memory_lower_bound_kind" not in result.details


def test_memory_estimate_check_reports_listed_weight_with_unavailable_size() -> None:
    target = UnavailableNoneSizeTarget(
        files={
            "config.json": b'{"model_type":"llama"}',
            "model-00001-of-00002.safetensors": b"a" * 10,
            "model-00002-of-00002.safetensors": b"b" * 20,
        },
        unavailable_paths=("model-00002-of-00002.safetensors",),
    )

    result = MemoryEstimateCheck().run(CheckContext(target=target, options=check_options()))

    assert result.status == "warn"
    assert result.severity == "low"
    assert result.details["estimate_source"] == "file_sizes"
    assert result.details["lower_bound_bytes"] == 10
    assert result.details["measured_bytes"] == 10
    assert result.details["unavailable_weight_paths"] == ("model-00002-of-00002.safetensors",)
    # The partial sum is reported diagnostically only; it must not be smoke-gate-trusted.
    assert "memory_lower_bound_kind" not in result.details


def test_memory_estimate_check_skip_reports_all_listed_weights_with_unavailable_sizes() -> None:
    target = UnavailableNoneSizeTarget(
        files={
            "config.json": b'{"model_type":"llama"}',
            "model-00001-of-00002.safetensors": b"a" * 10,
            "model-00002-of-00002.safetensors": b"b" * 20,
        },
        unavailable_paths=(
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ),
    )

    result = MemoryEstimateCheck().run(CheckContext(target=target, options=check_options()))

    assert result.status == "skip"
    assert result.severity == "info"
    assert result.details["estimate_source"] == "unknown"
    assert result.details["context_length"] == 4096
    assert result.details["unavailable_weight_paths"] == (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )


def test_memory_estimate_check_skips_when_no_estimate_is_available() -> None:
    result = MemoryEstimateCheck().run(
        context_for_files({"config.json": b'{"model_type":"llama"}'})
    )

    assert result.status == "skip"
    assert result.severity == "info"
    assert "insufficient metadata" in result.message
    assert result.details["estimate_source"] == "unknown"
    assert result.details["context_length"] == 4096


def test_memory_estimate_check_context_length_affects_lower_bound() -> None:
    short_context = MemoryEstimateCheck().run(_context_for_config(BASE_CONFIG, context_length=16))
    long_context = MemoryEstimateCheck().run(_context_for_config(BASE_CONFIG, context_length=32))

    assert long_context.details["lower_bound_bytes"] > short_context.details["lower_bound_bytes"]


def _context_for_config(
    config: dict[str, object],
    *,
    context_length: int = 4096,
    max_memory_bytes: int | None = None,
) -> CheckContext:
    options = replace(
        check_options(),
        context_length=context_length,
        max_memory_bytes=max_memory_bytes,
    )
    return CheckContext(
        target=context_for_files({"config.json": json.dumps(config).encode()}).target,
        options=options,
    )


class UnavailableSizeTarget(FakeTarget):
    def size(self, path: str) -> int | None:
        if path == "model-00002-of-00002.safetensors":
            raise TargetError("could not stat file", target=path, source=self.source)
        return super().size(path)


class UnavailableNoneSizeTarget(FakeTarget):
    unavailable_paths: tuple[str, ...]

    def __init__(self, *, files: dict[str, bytes], unavailable_paths: tuple[str, ...]) -> None:
        super().__init__(files=files)
        self.unavailable_paths = unavailable_paths

    def size(self, path: str) -> int | None:
        if path in self.unavailable_paths:
            return None
        return super().size(path)


def test_memory_estimate_uses_file_sizes_for_mixed_precision_when_all_weights_sized() -> None:
    ctx = context_for_files(
        {
            "config.json": json.dumps(MIXED_CONFIG).encode(),
            "model-00001-of-00002.safetensors": b"a" * 100,
            "model-00002-of-00002.safetensors": b"b" * 200,
        },
        options=replace(check_options(), context_length=16),
    )

    result = MemoryEstimateCheck().run(ctx)

    assert result.status == "warn"
    assert result.details["estimate_source"] == "file_sizes"
    assert result.details["memory_lower_bound_kind"] == "model_runtime"
    assert result.details["weight_lower_bound_bytes"] == 300
    assert result.details["measured_bytes"] == 300
    expected_kv = 2 * 2 * 16 * (8 * 128) * 2  # 2 * layers * ctx_len * kv_hidden * fp16_bytes
    assert result.details["kv_cache_lower_bound_bytes"] == expected_kv
    assert result.details["lower_bound_bytes"] == 300 + expected_kv


def test_memory_estimate_mixed_precision_unverified_when_a_shard_size_is_missing() -> None:
    target = UnavailableNoneSizeTarget(
        files={
            "config.json": json.dumps(MIXED_CONFIG).encode(),
            "model-00001-of-00002.safetensors": b"a" * 100,
            "model-00002-of-00002.safetensors": b"b" * 200,
        },
        unavailable_paths=("model-00002-of-00002.safetensors",),
    )

    result = MemoryEstimateCheck().run(CheckContext(target=target, options=check_options()))

    assert result.status == "skip"
    assert result.severity == "info"
    assert result.details["estimate_source"] == "unknown"
    assert "memory_lower_bound_kind" not in result.details
    assert result.details["unavailable_weight_paths"] == ("model-00002-of-00002.safetensors",)


def test_memory_estimate_mixed_precision_unverified_when_no_weight_files() -> None:
    ctx = context_for_files({"config.json": json.dumps(MIXED_CONFIG).encode()})

    result = MemoryEstimateCheck().run(ctx)

    assert result.status == "skip"
    assert result.details["estimate_source"] == "unknown"
    assert "memory_lower_bound_kind" not in result.details
    assert "unavailable_weight_paths" not in result.details
    assert result.details["context_length"] == 4096


def test_memory_estimate_stays_config_for_harmless_per_layer_overrides() -> None:
    base = MemoryEstimateCheck().run(_context_for_config(BASE_CONFIG, context_length=16))

    harmless_quant_blocks = [
        {"bits": 4, "model.layers.0.mlp": {}},  # empty override
        {"bits": 4, "model.layers.0.mlp": {"mode": "affine"}},  # mode-only
        {"bits": 4, "model.layers.0.mlp": {"bits": 4}},  # same-bits
    ]
    for quant in harmless_quant_blocks:
        config = {**BASE_CONFIG, "quantization": quant}
        result = MemoryEstimateCheck().run(_context_for_config(config, context_length=16))
        assert result.details["estimate_source"] == "config", quant
        assert result.details["lower_bound_bytes"] == base.details["lower_bound_bytes"], quant
        assert result.details["memory_lower_bound_kind"] == "model_runtime", quant


def test_memory_estimate_stays_config_for_unquantized_config() -> None:
    config = {key: value for key, value in BASE_CONFIG.items() if key != "quantization"}

    result = MemoryEstimateCheck().run(_context_for_config(config, context_length=16))

    assert result.details["estimate_source"] == "config"


def test_promoted_smoke_gate_detail_keys_are_stable() -> None:
    # Contract pin: these three details keys are the documented cross-check seam the
    # --smoke memory gate reads (runners/smoke.py). A rename/retype is a breaking change.
    result = MemoryEstimateCheck().run(_context_for_config(BASE_CONFIG, context_length=16))

    assert isinstance(result.details["lower_bound_bytes"], int)
    assert isinstance(result.details["estimate_source"], str)
    # Only present on a trusted bound (lower_bound_bytes > 0 and no unsized shards).
    assert result.details["memory_lower_bound_kind"] == "model_runtime"
