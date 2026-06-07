"""Tests for the MLX-compatibility signal check."""

import json

from mlx_model_doctor.checks.compat import (
    MlxCompatSignalCheck,
    _header_has_quantized_tensors,
)
from mlx_model_doctor.context import CheckOptions
from mlx_model_doctor.safetensors_header import FileHeader, SafetensorsHeader, TensorEntry
from tests.fakes import context_for_files

CHECK = MlxCompatSignalCheck()


def _options(*, include_weights):
    return CheckOptions(
        max_memory_bytes=None,
        context_length=4096,
        include_weights=include_weights,
        smoke=False,
        verbosity="normal",
    )


def _quantized_header():
    def _entry(dtype):
        return TensorEntry(dtype=dtype, shape=(2, 2), data_offsets=(0, 0), stored_element_count=1)

    file_header = FileHeader(
        filename="model.safetensors",
        tensors={"model.layer.scales": _entry("U8"), "model.layer.weight": _entry("U32")},
        metadata={},
        header_length=None,
        file_size=None,
    )
    return SafetensorsHeader(
        files=(file_header,), weight_map={}, sharded=False, stored_count_by_dtype={}
    )


def _run(files, **kw):
    return CHECK.run(context_for_files(files, **kw))


def _quant_config():
    return json.dumps({"quantization": {"bits": 4, "group_size": 64}}).encode()


def test_pass_lists_strong_signals():
    r = _run({"config.json": _quant_config()}, source="hf", name="mlx-community/Foo-4bit")
    assert r.check_id == "text/compat.mlx_signal"
    assert r.status == "pass"
    assert r.severity == "info"
    assert set(r.details["signals"]) == {"author:mlx-community", "config:quantization", "repo-name"}


def test_author_signal_hf_only_isolated():
    r = _run({}, source="hf", name="mlx-community/clean")
    assert r.status == "pass"
    assert r.details["signals"] == ("author:mlx-community",)


def test_tag_and_library_from_listing_metadata():
    r = _run({}, source="hf", name="org/m", tags=frozenset({"mlx"}), library_name="mlx-lm")
    assert set(r.details["signals"]) == {"tag:mlx", "library:mlx-lm"}


def test_weak_only_is_hedged_pass():
    r = _run({}, source="hf", name="someorg/Model-4bit")
    assert r.status == "pass"
    assert r.details["signals"] == ("repo-name",)
    assert "no mlx-specific" in r.message.lower() or "weak" in r.message.lower()


def test_zero_signals_is_pass_with_empty_tuple():
    r = _run({}, source="hf", name="someorg/plain")
    assert r.status == "pass"
    assert r.severity == "info"
    assert r.details["signals"] == ()


def test_local_path_name_basename_signal():
    r = _run({}, source="local", name="/Users/x/models/Foo-mlx-8bit")
    assert r.details["signals"] == ("repo-name",)


def test_local_parent_dir_token_does_not_false_match():
    r = _run({}, source="local", name="/Users/mlxuser/models/clean-name")
    assert r.details["signals"] == ()


def test_config_absent_does_not_crash_and_name_signal_fires():
    r = _run({}, source="hf", name="someorg/Foo-mlx")
    assert r.status == "pass"
    assert r.details["signals"] == ("repo-name",)


# --- _header_has_quantized_tensors helper (duck-typed stub) ---


class _Entry:
    def __init__(self, dtype):
        self.dtype = dtype


class _StubHeader:
    def __init__(self, tensors):
        self._tensors = tensors

    def tensor_names(self):
        return iter(self._tensors)

    def tensor(self, name):
        return self._tensors.get(name)


def test_header_has_quantized_tensors_true():
    h = _StubHeader({"layer.scales": _Entry("F16"), "layer.weight": _Entry("U32")})
    assert _header_has_quantized_tensors(h) is True


def test_header_has_quantized_tensors_false_without_u32():
    h = _StubHeader({"layer.scales": _Entry("F16"), "layer.weight": _Entry("F16")})
    assert _header_has_quantized_tensors(h) is False


def test_header_has_quantized_tensors_false_no_scales():
    assert _header_has_quantized_tensors(_StubHeader({"layer.weight": _Entry("F16")})) is False


def test_weights_signal_present_when_include_weights():
    # A real quantized header (.scales + U32 .weight) yields weights:mlx-quant end-to-end.
    ctx = context_for_files(
        {}, source="hf", name="org/plain", options=_options(include_weights=True)
    )
    ctx.target._safetensors_header = _quantized_header()
    assert "weights:mlx-quant" in CHECK.run(ctx).details["signals"]


def test_weights_signal_absent_when_skip_weights():
    # Same quantized header present, but --skip-weights means the header is never consulted.
    ctx = context_for_files(
        {}, source="hf", name="org/plain", options=_options(include_weights=False)
    )
    ctx.target._safetensors_header = _quantized_header()
    assert "weights:mlx-quant" not in CHECK.run(ctx).details["signals"]
