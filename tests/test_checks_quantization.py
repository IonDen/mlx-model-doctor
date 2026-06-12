import json

from mlx_model_doctor.checks.quantization import (
    MlxQuantizationModeCheck,
    MlxQuantShapeCheck,
    QuantizationMetadataCheck,
    _effective_quant,
    _resolve_quant_field,
)
from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.safetensors_header import FileHeader, SafetensorsHeader, TensorEntry
from tests.fakes import FakeTarget, check_options, context_for_files


def test_quantization_metadata_check_warns_for_non_mlx_quantization_config() -> None:
    config = {"quantization_config": {"bits": 4, "group_size": 128}}

    result = QuantizationMetadataCheck().run(_context_for_config(config))

    assert result.status == "warn"
    assert result.severity == "medium"
    assert "non-MLX" in result.message
    assert result.remediation is not None


def test_quantization_metadata_check_passes_for_mlx_quantization_dict() -> None:
    config = {"quantization": {"bits": 4, "group_size": 64}}

    result = QuantizationMetadataCheck().run(_context_for_config(config))

    assert result.status == "pass"
    assert result.severity == "info"
    assert result.details["bits"] == 4
    assert result.details["group_size"] == 64


def test_quantization_metadata_check_warns_when_mlx_bits_are_missing() -> None:
    _assert_incomplete_mlx_quantization_warns({"group_size": 64})


def test_quantization_metadata_check_warns_when_mlx_quantization_is_empty() -> None:
    _assert_incomplete_mlx_quantization_warns({})


def test_quantization_metadata_check_warns_when_mlx_bits_are_invalid() -> None:
    _assert_incomplete_mlx_quantization_warns({"bits": 0, "group_size": 64})


def test_quantization_metadata_check_warns_when_metadata_is_missing() -> None:
    result = QuantizationMetadataCheck().run(_context_for_config({"model_type": "llama"}))

    assert result.status == "warn"
    assert result.severity == "low"
    assert "quantization" in result.message
    assert result.remediation is not None


def test_quantization_metadata_check_skips_when_config_unavailable() -> None:
    for files in ({}, {"config.json": b"{not-json"}, {"config.json": b"null"}):
        result = QuantizationMetadataCheck().run(context_for_files(files))

        assert result.status == "skip"
        assert result.severity == "info"
        assert "config" in result.message
        assert "unavailable" in result.message


def test_quantization_metadata_check_warns_for_invalid_mlx_quantization_value() -> None:
    result = QuantizationMetadataCheck().run(_context_for_config({"quantization": "4bit"}))

    assert result.status == "warn"
    assert result.severity == "medium"
    assert "quantization" in result.message
    assert result.remediation is not None


def _context_for_config(config: dict[str, object]):
    return context_for_files({"config.json": json.dumps(config).encode()})


def _assert_incomplete_mlx_quantization_warns(quantization: dict[str, object]) -> None:
    result = QuantizationMetadataCheck().run(_context_for_config({"quantization": quantization}))

    assert result.status == "warn"
    assert result.severity == "medium"
    assert "incomplete" in result.message
    assert result.remediation is not None
    assert result.details["quantization"] == quantization


def test_quant_mode_passes_for_valid_affine() -> None:
    result = MlxQuantizationModeCheck().run(
        _context_for_config({"quantization": {"bits": 4, "group_size": 64}})
    )
    assert result.status == "pass"


def test_quant_mode_passes_for_valid_fixed_mode() -> None:
    result = MlxQuantizationModeCheck().run(
        _context_for_config({"quantization": {"mode": "mxfp4", "bits": 4, "group_size": 32}})
    )
    assert result.status == "pass"


def test_quant_mode_fails_for_unknown_mode() -> None:
    result = MlxQuantizationModeCheck().run(
        _context_for_config({"quantization": {"mode": "int8", "bits": 8, "group_size": 64}})
    )
    assert result.status == "fail"
    assert result.severity == "high"
    assert "mode" in result.message.lower()


def test_quant_mode_warns_for_off_table_affine_bits() -> None:
    result = MlxQuantizationModeCheck().run(
        _context_for_config({"quantization": {"bits": 7, "group_size": 64}})
    )
    assert result.status == "warn"


def test_quant_mode_warns_for_noncanonical_fixed_mode() -> None:
    result = MlxQuantizationModeCheck().run(
        _context_for_config({"quantization": {"mode": "mxfp4", "bits": 8, "group_size": 32}})
    )
    assert result.status == "warn"


def test_quant_mode_skips_when_not_quantized() -> None:
    result = MlxQuantizationModeCheck().run(_context_for_config({"model_type": "llama"}))
    assert result.status == "skip"


def test_quant_mode_skips_for_non_mlx_quantization_config() -> None:
    result = MlxQuantizationModeCheck().run(
        _context_for_config({"quantization_config": {"bits": 4, "group_size": 128}})
    )
    assert result.status == "skip"


# ---------------------------------------------------------------------------
# MlxQuantShapeCheck helpers
# ---------------------------------------------------------------------------


def _t(dtype: str, shape: tuple[int, ...]) -> TensorEntry:
    return TensorEntry(dtype=dtype, shape=shape, data_offsets=(0, 0), stored_element_count=1)


def _quant_header(tensors: dict[str, TensorEntry]) -> SafetensorsHeader:
    fh = FileHeader(
        filename="model.safetensors",
        tensors=tensors,
        metadata={},
        header_length=10,
        file_size=None,
    )
    return SafetensorsHeader(
        files=(fh,),
        weight_map=dict.fromkeys(tensors, "model.safetensors"),
        sharded=False,
        stored_count_by_dtype={},
    )


def _quant_ctx(header: SafetensorsHeader | None, quant: dict[str, object] | None) -> CheckContext:
    config: dict[str, object] = {"quantization": quant} if quant is not None else {}
    files = {"config.json": json.dumps(config).encode()}
    return CheckContext(
        target=FakeTarget(files=files, _safetensors_header=header),
        options=check_options(),
    )


# ---------------------------------------------------------------------------
# MlxQuantShapeCheck tests
# ---------------------------------------------------------------------------


def test_quant_shape_pass_for_consistent_layer() -> None:
    # bits=4, group_size=64: in=512 -> packed_last=512*4/32=64, scales_last=512/64=8
    header = _quant_header({"l.weight": _t("U32", (256, 64)), "l.scales": _t("BF16", (256, 8))})
    assert (
        MlxQuantShapeCheck().run(_quant_ctx(header, {"bits": 4, "group_size": 64})).status == "pass"
    )


def test_quant_shape_fail_on_scales_weight_mismatch() -> None:
    # scales_last=4 -> in_s=256, but packed_last=64 -> in_w=512: mismatch
    header = _quant_header({"l.weight": _t("U32", (256, 64)), "l.scales": _t("BF16", (256, 4))})
    result = MlxQuantShapeCheck().run(_quant_ctx(header, {"bits": 4, "group_size": 64}))
    assert result.status == "fail"
    assert result.severity == "high"
    assert result.details["inconsistent_layers"]


def test_quant_shape_warn_declared_but_unapplied() -> None:
    header = _quant_header({"model.embed_tokens.weight": _t("BF16", (256, 512))})  # no .scales
    result = MlxQuantShapeCheck().run(_quant_ctx(header, {"bits": 4, "group_size": 64}))
    assert result.status == "warn"
    assert result.details["no_quantized_tensors"] is True


def test_quant_shape_warn_on_unknown_bits() -> None:
    header = _quant_header({"l.weight": _t("U32", (256, 64)), "l.scales": _t("BF16", (256, 8))})
    result = MlxQuantShapeCheck().run(_quant_ctx(header, {"bits": 7, "group_size": 64}))
    assert result.status == "warn"
    assert result.details["unknown_bits"] == 7


def test_quant_shape_skip_without_quant_or_header() -> None:
    header = _quant_header({"l.weight": _t("U32", (256, 64)), "l.scales": _t("BF16", (256, 8))})
    assert MlxQuantShapeCheck().run(_quant_ctx(header, None)).status == "skip"
    assert (
        MlxQuantShapeCheck().run(_quant_ctx(None, {"bits": 4, "group_size": 64})).status == "skip"
    )


def test_quant_shape_skip_when_bits_missing() -> None:
    # quantization mapping present but no usable bits -> skip (not a false pass/fail).
    header = _quant_header({"l.weight": _t("U32", (256, 64)), "l.scales": _t("BF16", (256, 8))})
    result = MlxQuantShapeCheck().run(_quant_ctx(header, {"group_size": 64}))
    assert result.status == "skip"


def test_quant_shape_skips_non_u32_weight_with_scales() -> None:
    # A BF16 weight carrying a stray .scales sibling must NOT be measured as packed-U32:
    # the dtype != "U32" guard skips it, so the layer is not flagged as inconsistent.
    # (Dropping that guard would measure the BF16 weight and emit a spurious fail.)
    header = _quant_header({"l.weight": _t("BF16", (256, 64)), "l.scales": _t("BF16", (256, 4))})
    result = MlxQuantShapeCheck().run(_quant_ctx(header, {"bits": 4, "group_size": 64}))
    assert result.status == "pass"


def test_resolve_quant_field_absent_uses_default() -> None:
    assert _resolve_quant_field({}, "bits", 4) == 4


def test_resolve_quant_field_present_valid_wins() -> None:
    assert _resolve_quant_field({"bits": 8}, "bits", 4) == 8


def test_resolve_quant_field_present_invalid_returns_none() -> None:
    # Present-but-invalid must NOT fall back to the default; it returns None so the
    # layer is reported unverified rather than silently validated with the default.
    assert _resolve_quant_field({"bits": 0}, "bits", 4) is None
    assert _resolve_quant_field({"bits": "8"}, "bits", 4) is None
    assert _resolve_quant_field({"bits": -1}, "bits", 4) is None


def test_effective_quant_override_hit() -> None:
    quant = {"bits": 4, "group_size": 32, "lm_head": {"bits": 8, "group_size": 64}}
    assert _effective_quant(quant, "lm_head", 4, 32) == (8, 64)


def test_effective_quant_no_override_uses_defaults() -> None:
    quant = {"bits": 4, "group_size": 32}
    assert _effective_quant(quant, "model.layers.0.mlp.experts.down_proj", 4, 32) == (4, 32)


def test_effective_quant_partial_override_falls_back_per_field() -> None:
    quant = {"bits": 4, "group_size": 32, "x": {"bits": 8}}  # group_size absent -> default
    assert _effective_quant(quant, "x", 4, 32) == (8, 32)


def test_effective_quant_present_invalid_field_is_none() -> None:
    quant = {"bits": 4, "group_size": 32, "x": {"bits": 0, "group_size": 64}}
    assert _effective_quant(quant, "x", 4, 32) == (None, 64)


def test_effective_quant_non_mapping_override_uses_defaults() -> None:
    quant = {"bits": 4, "group_size": 32, "x": "weird"}
    assert _effective_quant(quant, "x", 4, 32) == (4, 32)
