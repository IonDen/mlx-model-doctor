import json

from mlx_model_doctor.checks.quantization import (
    MlxQuantizationModeCheck,
    MlxQuantShapeCheck,
    QuantizationMetadataCheck,
    _classify_quant,
    _effective_mode_params,
    _effective_quant,
    _is_per_layer_override,
    _resolve_quant_field,
    config_has_mixed_precision_quant,
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
    assert result.details["unverified_layers"] == ("l",)


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


def test_quant_shape_pass_for_mixed_precision_per_layer() -> None:
    # Real gpt-oss-20b-MXFP4-Q8 shape vectors: lm_head is 8-bit affine (BF16 scales),
    # experts are mxfp4 (U8 scales). Both are measured via their U32 weight (only weight
    # dtype is gated; U8 is the scales dtype, not gated). The flat b4/gs32 formula mis-fails
    # lm_head (5760 != 1440); per-layer resolution validates lm_head at b8/gs64 (2880 == 2880).
    header = _quant_header(
        {
            "lm_head.weight": _t("U32", (201088, 720)),
            "lm_head.scales": _t("BF16", (201088, 45)),
            "model.layers.0.mlp.experts.down_proj.weight": _t("U32", (32, 2880, 360)),
            "model.layers.0.mlp.experts.down_proj.scales": _t("U8", (32, 2880, 90)),
        }
    )
    quant = {
        "bits": 4,
        "group_size": 32,
        "mode": "mxfp4",
        "lm_head": {"bits": 8, "group_size": 64, "mode": "affine"},
    }
    assert MlxQuantShapeCheck().run(_quant_ctx(header, quant)).status == "pass"


def test_quant_shape_fail_under_resolved_per_layer_params() -> None:
    # Differential RED->GREEN: the shapes are CONSISTENT under the scalar default
    # (b4/gs64: 64*32//4=512 == 8*64=512), so the old global-formula code returns pass.
    # The per-layer override (b8/gs64) makes it inconsistent (64*32//8=256 != 8*64=512)
    # -> fail. This isolates per-layer resolution and also catches a "silently skip
    # overridden layers" regression (which would wrongly return pass).
    header = _quant_header({"l.weight": _t("U32", (256, 64)), "l.scales": _t("BF16", (256, 8))})
    quant = {"bits": 4, "group_size": 64, "l": {"bits": 8, "group_size": 64}}
    result = MlxQuantShapeCheck().run(_quant_ctx(header, quant))
    assert result.status == "fail"
    assert result.details["inconsistent_layers"] == ("l",)


def test_quant_shape_pass_for_uniform_mxfp4_no_overrides() -> None:
    # Spec #2 regression: a uniform mxfp4 model (U8 scales, no per-layer overrides)
    # passes under the scalar defaults. 360*32//4=2880 == 90*32=2880.
    header = _quant_header(
        {
            "model.layers.0.mlp.experts.down_proj.weight": _t("U32", (32, 2880, 360)),
            "model.layers.0.mlp.experts.down_proj.scales": _t("U8", (32, 2880, 90)),
        }
    )
    quant = {"bits": 4, "group_size": 32, "mode": "mxfp4"}
    assert MlxQuantShapeCheck().run(_quant_ctx(header, quant)).status == "pass"


def test_quant_shape_pass_for_override_only_completes_missing_default() -> None:
    # has_per_layer=True with an incomplete scalar default: the override supplies both
    # fields, so the layer is checked (not skipped). 64*32//4=512 == 8*64=512 -> pass.
    header = _quant_header({"l.weight": _t("U32", (256, 64)), "l.scales": _t("BF16", (256, 8))})
    quant = {"l": {"bits": 4, "group_size": 64}}  # no scalar bits/group_size
    assert MlxQuantShapeCheck().run(_quant_ctx(header, quant)).status == "pass"


def test_quant_shape_skip_survives_stray_non_override_mapping() -> None:
    # A non-override nested mapping in the quantization block must NOT flip has_per_layer
    # and suppress the incomplete-metadata skip. {"group_size":64} (no bits) + a U32 layer
    # with no override for it -> still skip, despite the stray "foo" mapping.
    header = _quant_header({"l.weight": _t("U32", (256, 64)), "l.scales": _t("BF16", (256, 8))})
    quant = {"group_size": 64, "foo": {"bar": 1}}
    assert MlxQuantShapeCheck().run(_quant_ctx(header, quant)).status == "skip"


def test_quant_shape_warn_on_invalid_present_override() -> None:
    # Shapes ARE consistent with the default (b4/gs64 -> 512 == 512), but the override
    # declares an explicit invalid field. The layer must be reported unverified (warn),
    # NOT silently validated against the default (which would pass), NOT failed. Covers
    # Finding 2 for zero, non-int, and invalid group_size override values.
    header = _quant_header({"l.weight": _t("U32", (256, 64)), "l.scales": _t("BF16", (256, 8))})
    bad_overrides = (
        {"bits": 0, "group_size": 64},
        {"bits": "8", "group_size": 64},
        {"bits": 8, "group_size": 0},
    )
    for bad_override in bad_overrides:
        quant = {"bits": 4, "group_size": 64, "l": bad_override}
        result = MlxQuantShapeCheck().run(_quant_ctx(header, quant))
        assert result.status == "warn", bad_override
        assert result.details["unverified_layers"] == ("l",), bad_override


# ---------------------------------------------------------------------------
# _classify_quant (pure)
# ---------------------------------------------------------------------------


def test_classify_quant_ok_for_valid_affine_and_fixed_modes() -> None:
    assert _classify_quant("affine", 64, 4).kind == "ok"
    assert _classify_quant("mxfp4", 32, 4).kind == "ok"
    assert _classify_quant("mxfp8", 32, 8).kind == "ok"
    assert _classify_quant("nvfp4", 16, 4).kind == "ok"


def test_classify_quant_absent_fields_do_not_flag() -> None:
    # None means "field not present" -> never a finding (matches the scalar "key in quant" guards).
    assert _classify_quant("affine", None, None).kind == "ok"
    assert _classify_quant("mxfp4", None, None).kind == "ok"


def test_classify_quant_non_string_mode_warns() -> None:
    assert _classify_quant(4, None, None).kind == "non_string_mode"


def test_classify_quant_unknown_mode_fails() -> None:
    assert _classify_quant("int8", 8, 64).kind == "unknown_mode"


def test_classify_quant_off_table_affine() -> None:
    assert _classify_quant("affine", 64, 7).kind == "off_table_affine"
    assert _classify_quant("affine", 48, 4).kind == "off_table_affine"


def test_classify_quant_fixed_mode_mismatch() -> None:
    assert _classify_quant("mxfp4", 64, 8).kind == "fixed_mismatch"


def test_classify_quant_crash_safe_on_unhashable_affine_values() -> None:
    # Set membership against a frozenset raises TypeError on unhashable values; the classifier
    # must treat them as off-table (warn), never raise.
    assert _classify_quant("affine", 64, []).kind == "off_table_affine"
    assert _classify_quant("affine", 64, {"x": 1}).kind == "off_table_affine"
    assert _classify_quant("affine", [], 4).kind == "off_table_affine"


def test_classify_quant_hashable_wrong_type_still_off_table() -> None:
    # The existing string-bits behavior is preserved: "8" is hashable and off-table.
    assert _classify_quant("affine", 64, "8").kind == "off_table_affine"


# ---------------------------------------------------------------------------
# _is_per_layer_override + _effective_mode_params (pure)
# ---------------------------------------------------------------------------


def test_is_per_layer_override_accepts_subset_keyed_mappings() -> None:
    assert _is_per_layer_override({"mode": "affine", "bits": 8, "group_size": 64}) is True
    assert _is_per_layer_override({"bits": 8, "group_size": 64}) is True  # mode-less
    assert _is_per_layer_override({}) is True  # empty -> resolves to defaults


def test_is_per_layer_override_rejects_foreign_keys_and_scalars() -> None:
    assert _is_per_layer_override({"foo": 1}) is False
    assert _is_per_layer_override({"mode": "x", "extra": 1}) is False  # foreign key present
    assert _is_per_layer_override(4) is False
    assert _is_per_layer_override("affine") is False


def test_effective_mode_params_mode_less_defaults_to_affine() -> None:
    # The flagship-repo case: a mode-less override is interpreted affine by MLX, NOT the scalar mode.
    assert _effective_mode_params({"bits": 8, "group_size": 64}) == ("affine", 64, 8)


def test_effective_mode_params_reads_override_only() -> None:
    assert _effective_mode_params({"mode": "affine", "bits": 8, "group_size": 64}) == (
        "affine",
        64,
        8,
    )
    assert _effective_mode_params({"bits": 8}) == ("affine", None, 8)  # partial
    assert _effective_mode_params({}) == ("affine", None, None)  # empty


def test_quant_mode_scalar_unhashable_bits_warns_not_crashes() -> None:
    # Pre-refactor, `quant["bits"] not in _AFFINE_BITS` raises TypeError on []. After the
    # refactor the scalar path goes through crash-safe _classify_quant -> structured warn.
    result = MlxQuantizationModeCheck().run(
        _context_for_config({"quantization": {"bits": [], "group_size": 64}})
    )
    assert result.status == "warn"
    assert result.severity == "medium"
    assert "off-table" in result.message


def test_quant_mode_scalar_explicit_null_fields_classify_as_absent() -> None:
    # A present-but-null bits/group_size is treated as absent (no value) by the mode check, so
    # the (mode, bits, group_size) table relation passes. QuantizationMetadataCheck separately
    # warns that bits is not a positive number, so a null-bits config is still surfaced overall.
    affine = MlxQuantizationModeCheck().run(
        _context_for_config({"quantization": {"bits": None, "group_size": None}})
    )
    assert affine.status == "pass"
    fixed = MlxQuantizationModeCheck().run(
        _context_for_config({"quantization": {"mode": "mxfp4", "group_size": None, "bits": None}})
    )
    assert fixed.status == "pass"
    assert "mxfp4" in fixed.message


# ---------------------------------------------------------------------------
# MlxQuantizationModeCheck — per-layer overrides (0030)
# ---------------------------------------------------------------------------


def test_quant_mode_per_layer_unknown_mode_fails() -> None:
    # RED->GREEN: today the override is ignored -> pass. After the fix -> fail naming the layer.
    quant = {"mode": "mxfp4", "bits": 4, "group_size": 32, "model.layers.0": {"mode": "int3"}}
    result = MlxQuantizationModeCheck().run(_context_for_config({"quantization": quant}))
    assert result.status == "fail"
    assert result.severity == "high"
    assert result.details == {"invalid_mode_layers": ("model.layers.0",)}


def test_quant_mode_per_layer_off_table_bits_warns() -> None:
    quant = {"mode": "mxfp4", "bits": 4, "group_size": 32, "l": {"mode": "affine", "bits": 7}}
    result = MlxQuantizationModeCheck().run(_context_for_config({"quantization": quant}))
    assert result.status == "warn"
    assert result.details == {"off_table_layers": ("l",)}


def test_quant_mode_per_layer_fixed_mismatch_warns() -> None:
    quant = {"mode": "mxfp4", "bits": 4, "group_size": 32, "l": {"mode": "mxfp4", "bits": 8}}
    result = MlxQuantizationModeCheck().run(_context_for_config({"quantization": quant}))
    assert result.status == "warn"
    assert result.details == {"off_table_layers": ("l",)}


def test_quant_mode_per_layer_non_string_mode_warns() -> None:
    quant = {"mode": "mxfp4", "bits": 4, "group_size": 32, "l": {"mode": 4}}
    result = MlxQuantizationModeCheck().run(_context_for_config({"quantization": quant}))
    assert result.status == "warn"
    assert result.details == {"off_table_layers": ("l",)}


def test_quant_mode_per_layer_unhashable_value_warns_not_crashes() -> None:
    quant = {"mode": "mxfp4", "bits": 4, "group_size": 32, "l": {"bits": []}}
    result = MlxQuantizationModeCheck().run(_context_for_config({"quantization": quant}))
    assert result.status == "warn"
    assert result.details == {"off_table_layers": ("l",)}


def test_quant_mode_fail_beats_warn_across_scalar_and_layers() -> None:
    # Scalar off-table (warn) + a per-layer unknown mode (fail) -> overall fail, both reflected.
    quant = {"bits": 7, "group_size": 64, "model.layers.0": {"mode": "int3"}}
    result = MlxQuantizationModeCheck().run(_context_for_config({"quantization": quant}))
    assert result.status == "fail"
    assert result.details == {
        "invalid_mode_layers": ("model.layers.0",),
        "scalar_default_invalid": True,
    }


def test_quant_mode_scalar_offender_in_mixed_config_uses_distinct_key() -> None:
    # Scalar off-table, overrides valid -> warn, flagged via the distinct boolean key
    # (no "<default>" sentinel mixed into a layer tuple).
    quant = {"bits": 7, "group_size": 64, "l": {"mode": "affine", "bits": 8, "group_size": 64}}
    result = MlxQuantizationModeCheck().run(_context_for_config({"quantization": quant}))
    assert result.status == "warn"
    assert result.details == {"scalar_default_invalid": True}


def test_quant_mode_multiple_offending_layers_are_sorted() -> None:
    quant = {
        "mode": "mxfp4",
        "bits": 4,
        "group_size": 32,
        "model.layers.1": {"mode": "int3"},
        "model.layers.0": {"mode": "int3"},
    }
    result = MlxQuantizationModeCheck().run(_context_for_config({"quantization": quant}))
    assert result.status == "fail"
    assert result.details == {"invalid_mode_layers": ("model.layers.0", "model.layers.1")}


def test_quant_mode_passes_for_valid_mixed_precision() -> None:
    # Guard (green on old AND new code — not RED->GREEN): scalar mxfp4 canonical + valid affine
    # 8/64 overrides (one explicit, one mode-less). Asserts the exact pass-details contract.
    quant = {
        "mode": "mxfp4",
        "bits": 4,
        "group_size": 32,
        "lm_head": {"mode": "affine", "bits": 8, "group_size": 64},
        "model.layers.0.mlp.gate": {"bits": 8, "group_size": 64},
    }
    result = MlxQuantizationModeCheck().run(_context_for_config({"quantization": quant}))
    assert result.status == "pass"
    assert result.details == {"per_layer_overrides": 2}


def test_quant_mode_empty_and_stray_mappings_add_no_finding() -> None:
    # Empty override {} resolves to (affine, None, None) -> ok. A foreign-key mapping is not an
    # override (subset filter) -> ignored. Scalar mxfp4 canonical -> overall pass.
    quant = {"mode": "mxfp4", "bits": 4, "group_size": 32, "l": {}, "foo": {"bar": 1}}
    result = MlxQuantizationModeCheck().run(_context_for_config({"quantization": quant}))
    assert result.status == "pass"
    assert result.details == {"per_layer_overrides": 1}  # only {} counts; "foo" is filtered out


def test_quant_mode_stray_foreign_mapping_preserves_scalar_fast_path() -> None:
    # A scalar-only config carrying ONLY a foreign-key mapping takes the legacy scalar path,
    # byte-identical to a plain valid config (the subset filter skips "foo").
    quant = {"bits": 4, "group_size": 64, "foo": {"bar": 1}}
    result = MlxQuantizationModeCheck().run(_context_for_config({"quantization": quant}))
    assert result.status == "pass"
    assert result.details == {"mode": "affine", "group_size": 64, "bits": 4}


def test_quant_mode_scalar_fail_with_valid_overrides() -> None:
    # Scalar unknown mode (fail) + all per-layer overrides valid -> overall fail; details carry
    # only the scalar flag (fail_layers empty, so no invalid_mode_layers tuple).
    quant = {"mode": "int3", "bits": 4, "group_size": 32, "l": {"bits": 8, "group_size": 64}}
    result = MlxQuantizationModeCheck().run(_context_for_config({"quantization": quant}))
    assert result.status == "fail"
    assert result.severity == "high"
    assert result.details == {"scalar_default_invalid": True}


# ---------------------------------------------------------------------------
# config_has_mixed_precision_quant (pure predicate)
# ---------------------------------------------------------------------------


def test_mixed_precision_predicate_false_for_scalar_only() -> None:
    assert (
        config_has_mixed_precision_quant({"quantization": {"bits": 4, "group_size": 64}}) is False
    )


def test_mixed_precision_predicate_true_for_differing_bits_override() -> None:
    config = {"quantization": {"bits": 4, "group_size": 64, "model.layers.0.mlp": {"bits": 8}}}
    assert config_has_mixed_precision_quant(config) is True


def test_mixed_precision_predicate_false_for_same_bits_override() -> None:
    config = {
        "quantization": {
            "bits": 4,
            "group_size": 64,
            "model.layers.0.mlp": {"bits": 4, "group_size": 64},
        }
    }
    assert config_has_mixed_precision_quant(config) is False


def test_mixed_precision_predicate_false_for_mode_only_override() -> None:
    config = {"quantization": {"bits": 4, "model.layers.0.mlp": {"mode": "affine"}}}
    assert config_has_mixed_precision_quant(config) is False


def test_mixed_precision_predicate_false_for_empty_override() -> None:
    config = {"quantization": {"bits": 4, "model.layers.0.mlp": {}}}
    assert config_has_mixed_precision_quant(config) is False


def test_mixed_precision_predicate_false_when_no_quantization_block() -> None:
    assert config_has_mixed_precision_quant({"model_type": "llama"}) is False


def test_mixed_precision_predicate_false_for_non_mapping_quantization() -> None:
    assert config_has_mixed_precision_quant({"quantization": "nope"}) is False


def test_mixed_precision_predicate_true_when_scalar_bits_absent_but_override_has_bits() -> None:
    config = {"quantization": {"group_size": 64, "model.layers.0.mlp": {"bits": 8}}}
    assert config_has_mixed_precision_quant(config) is True
