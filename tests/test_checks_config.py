import json

from mlx_model_doctor.checks.config import ConfigJsonCheck, ModelTypeCheck
from mlx_model_doctor.checks.tokenizer import SpecialTokensCheck
from mlx_model_doctor.context import _MAX_METADATA_BYTES
from mlx_model_doctor.errors import TargetError
from mlx_model_doctor.report import DoctorReport, render_json
from tests.fakes import FakeTarget, check_options, context_for_files


def test_config_json_check_fails_on_invalid_json() -> None:
    result = ConfigJsonCheck().run(context_for_files({"config.json": b"{not-json"}))

    assert result.check_id == "text/config.json"
    assert result.status == "fail"
    assert result.severity == "high"
    assert "invalid JSON" in result.message
    assert result.remediation is not None


def test_config_json_check_fails_on_non_object_json() -> None:
    result = ConfigJsonCheck().run(context_for_files({"config.json": b"null"}))

    assert result.status == "fail"
    assert result.severity == "high"
    assert "JSON object" in result.message
    assert result.remediation is not None


def test_config_json_check_fails_cleanly_when_config_missing() -> None:
    result = ConfigJsonCheck().run(context_for_files({}))

    assert result.status == "fail"
    assert result.severity == "high"
    assert "config.json" in result.message
    assert result.remediation is not None


def test_config_json_check_fails_cleanly_on_target_error() -> None:
    result = ConfigJsonCheck().run(
        context_for_target(TargetErrorTarget(files={"config.json": b"{}"}))
    )

    assert result.status == "fail"
    assert result.severity == "high"
    assert "Could not read" in result.message
    assert result.remediation is not None


def test_config_json_check_passes_on_valid_object() -> None:
    result = ConfigJsonCheck().run(context_for_files({"config.json": b'{"model_type":"llama"}'}))

    assert result.status == "pass"
    assert result.severity == "info"
    assert "valid JSON object" in result.message


def test_model_type_check_warns_when_missing_from_valid_config() -> None:
    result = ModelTypeCheck().run(context_for_files({"config.json": b"{}"}))

    assert result.check_id == "text/config.model_type"
    assert result.status == "warn"
    assert result.severity == "medium"
    assert "model_type" in result.message
    assert result.remediation is not None


def test_model_type_check_warns_for_empty_or_non_string_value() -> None:
    for config in (
        b'{"model_type":""}',
        b'{"model_type":[]}',
    ):
        result = ModelTypeCheck().run(context_for_files({"config.json": config}))

        assert result.status == "warn"
        assert result.severity == "medium"
        assert "model_type" in result.message


def test_model_type_check_passes_when_present() -> None:
    result = ModelTypeCheck().run(context_for_files({"config.json": b'{"model_type":"llama"}'}))

    assert result.status == "pass"
    assert result.severity == "info"
    assert "llama" in result.message
    assert result.details["model_type"] == "llama"


def test_model_type_check_skips_when_config_unavailable() -> None:
    for files in ({}, {"config.json": b"{not-json"}, {"config.json": b"null"}):
        result = ModelTypeCheck().run(context_for_files(files))

        assert result.status == "skip"
        assert result.severity == "info"
        assert "config" in result.message
        assert "unavailable" in result.message


def test_missing_config_checks_render_clean_report_shape() -> None:
    ctx = context_for_files({})
    report = DoctorReport(
        target=ctx.target.name,
        source=ctx.target.source,
        plugin="text",
        results=[
            ConfigJsonCheck().run(ctx),
            ModelTypeCheck().run(ctx),
            SpecialTokensCheck().run(ctx),
        ],
    )

    data = json.loads(render_json(report))

    assert data["summary"] == {"pass": 0, "warn": 0, "fail": 1, "skip": 2}
    assert [result["status"] for result in data["results"]] == ["fail", "skip", "skip"]
    assert all("runner" not in result["message"].lower() for result in data["results"])


def context_for_target(target: FakeTarget):
    from mlx_model_doctor.context import CheckContext

    return CheckContext(target=target, options=check_options())


class TargetErrorTarget(FakeTarget):
    def exists(self, path: str) -> bool:
        return True

    def read_text(self, path: str, *, max_bytes: int | None = None) -> str:
        raise TargetError("read failed", target=path, source=self.source)


def test_invalid_config_checks_render_clean_report_shape() -> None:
    ctx = context_for_files({"config.json": b"{not-json"})
    report = DoctorReport(
        target=ctx.target.name,
        source=ctx.target.source,
        plugin="text",
        results=[
            ConfigJsonCheck().run(ctx),
            ModelTypeCheck().run(ctx),
            SpecialTokensCheck().run(ctx),
        ],
    )

    data = json.loads(render_json(report))

    assert data["summary"] == {"pass": 0, "warn": 0, "fail": 1, "skip": 2}
    assert [result["status"] for result in data["results"]] == ["fail", "skip", "skip"]
    assert all("runner" not in result["message"].lower() for result in data["results"])


def test_config_json_check_fails_for_oversized_config_without_reading() -> None:
    oversized = b"{}" + b" " * (_MAX_METADATA_BYTES + 1)
    result = ConfigJsonCheck().run(context_for_files({"config.json": oversized}))
    assert result.status == "fail"
    assert result.severity == "high"
    assert "too large" in result.message
