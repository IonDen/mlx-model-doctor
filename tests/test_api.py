import json
from pathlib import Path

import pytest

from mlx_model_doctor.api import check_hf_model, check_local_model
from mlx_model_doctor.context import CheckOptions
from mlx_model_doctor.errors import ModelDoctorError, TargetError


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


def test_check_hf_model_returns_text_report_for_valid_fake_repo() -> None:
    hub = FakeHub(files=valid_hf_files())

    report = check_hf_model("org/model", hub=hub)

    assert report.target == "org/model"
    assert report.source == "hf"
    assert report.plugin == "text"
    assert report.summary["fail"] == 0
    assert result_by_id(report.results, "text/config.json").status == "pass"
    assert result_by_id(report.results, "text/tokenizer.files").status == "pass"


def test_check_hf_model_reports_missing_config_without_crashing() -> None:
    hub = FakeHub(files={"tokenizer.json": b"{}"})

    report = check_hf_model("org/missing-config", hub=hub)

    assert report.plugin == "text"
    assert report.source == "hf"
    assert report.summary["fail"] >= 1
    assert result_by_id(report.results, "text/files.required").status == "fail"


def test_check_hf_model_reports_bad_config_metadata_without_crashing() -> None:
    hub = FakeHub(files={"config.json": b"{not-json", "tokenizer.json": b"{}"})

    report = check_hf_model("org/bad-config", hub=hub)

    assert result_by_id(report.results, "text/config.json").status == "fail"


def test_check_hf_model_propagates_custom_options_to_memory_estimate() -> None:
    options = CheckOptions(
        max_memory_bytes=1,
        context_length=8,
        include_weights=True,
        smoke=False,
        verbosity="normal",
    )
    hub = FakeHub(files=valid_hf_files())

    report = check_hf_model("org/model", options=options, hub=hub)

    memory = result_by_id(report.results, "text/memory.estimate")
    assert memory.status == "warn"
    assert memory.severity == "high"
    assert memory.details["context_length"] == 8
    assert memory.details["max_memory_bytes"] == 1


def test_check_hf_model_rejects_unknown_plugin_with_package_error() -> None:
    hub = FakeHub(files=valid_hf_files())

    with pytest.raises(ModelDoctorError, match="Unknown plugin"):
        check_hf_model("org/model", plugin_name="vision", hub=hub)


def test_check_hf_model_maps_model_info_error_to_target_error() -> None:
    hub = FakeHub(model_info_error=RuntimeError("gated repo"))

    with pytest.raises(TargetError, match="Could not inspect Hugging Face model") as exc_info:
        check_hf_model("org/gated", hub=hub)

    assert exc_info.value.source == "hf"


def test_check_hf_model_propagates_download_target_error() -> None:
    hub = FakeHub(
        files=valid_hf_files(),
        download_errors={"config.json": RuntimeError("network unavailable")},
    )

    with pytest.raises(TargetError, match="Could not read Hugging Face model file") as exc_info:
        check_hf_model("org/model", hub=hub)

    assert exc_info.value.source == "hf"


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


def valid_hf_files() -> dict[str, bytes]:
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
    return {"config.json": json.dumps(config).encode(), "tokenizer.json": b"{}"}


class FakeSibling:
    def __init__(self, *, rfilename: str, size: int | None) -> None:
        self.rfilename = rfilename
        self.size = size


class FakeModelInfo:
    def __init__(self, *, siblings: tuple[FakeSibling, ...]) -> None:
        self.siblings = siblings


class FakeHub:
    def __init__(
        self,
        *,
        files: dict[str, bytes] | None = None,
        model_info_error: Exception | None = None,
        download_errors: dict[str, Exception] | None = None,
    ) -> None:
        self.files = files if files is not None else {}
        self.model_info_error = model_info_error
        self.download_errors = download_errors if download_errors is not None else {}

    def model_info(self, repo_id: str, *, files_metadata: bool) -> FakeModelInfo:
        if self.model_info_error is not None:
            raise self.model_info_error
        return FakeModelInfo(
            siblings=tuple(
                FakeSibling(rfilename=path, size=len(data))
                for path, data in self.files.items()
            )
        )

    def download_bytes(self, repo_id: str, filename: str) -> bytes:
        if filename in self.download_errors:
            raise self.download_errors[filename]
        return self.files[filename]
