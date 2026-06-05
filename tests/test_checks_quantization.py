import json

from mlx_model_doctor.checks.quantization import MlxQuantizationModeCheck, QuantizationMetadataCheck
from tests.fakes import context_for_files


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
