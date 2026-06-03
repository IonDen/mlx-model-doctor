import json
from pathlib import Path

import pytest

from mlx_model_doctor.api import check_local_model
from mlx_model_doctor.context import CheckOptions
from mlx_model_doctor.errors import ModelDoctorError


def test_check_local_model_returns_text_report_for_valid_local_model(tmp_path: Path) -> None:
    model = write_local_model(tmp_path)

    report = check_local_model(model)

    assert report.target == str(model.resolve())
    assert report.source == "local"
    assert report.plugin == "text"
    assert report.summary["fail"] == 0
    assert {result.check_id for result in report.results} == {
        "text/files.required",
        "text/config.json",
        "text/config.model_type",
        "text/tokenizer.files",
        "text/tokenizer.special_tokens",
        "text/safetensors.index",
        "text/quantization.metadata",
        "text/memory.estimate",
    }


def test_check_local_model_reports_missing_config_without_crashing(tmp_path: Path) -> None:
    model = tmp_path / "missing-config"
    model.mkdir()

    report = check_local_model(model)

    assert report.plugin == "text"
    assert report.summary["fail"] >= 1
    assert any(result.check_id == "text/files.required" for result in report.results)


def test_check_local_model_propagates_custom_options_to_memory_estimate(
    tmp_path: Path,
) -> None:
    model = write_local_model(tmp_path)
    options = CheckOptions(
        max_memory_bytes=1,
        context_length=8,
        include_weights=True,
        smoke=False,
        verbosity="normal",
    )

    report = check_local_model(model, options=options)

    memory = result_by_id(report.results, "text/memory.estimate")
    assert memory.status == "warn"
    assert memory.severity == "high"
    assert memory.details["context_length"] == 8
    assert memory.details["max_memory_bytes"] == 1


def test_check_local_model_rejects_unknown_plugin_with_package_error(tmp_path: Path) -> None:
    model = write_local_model(tmp_path)

    with pytest.raises(ModelDoctorError, match="Unknown plugin"):
        check_local_model(model, plugin_name="vision")


def write_local_model(root: Path) -> Path:
    model = root / "model"
    model.mkdir()
    config = {
        "model_type": "llama",
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": 256,
        "intermediate_size": 512,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "quantization": {"bits": 4, "group_size": 64},
    }
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    return model


def result_by_id(results, check_id: str):
    return next(result for result in results if result.check_id == check_id)
