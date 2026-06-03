import json
from pathlib import Path

import pytest

from mlx_model_doctor.report import (
    CheckResult,
    DoctorReport,
    render_json,
    render_markdown,
    render_text,
)


def test_pass_result_must_have_info_severity() -> None:
    with pytest.raises(ValueError, match=r"pass.*info"):
        CheckResult(
            check_id="text/files.required",
            title="Required files",
            status="pass",
            severity="low",
            message="ok",
        )


def test_report_summary_counts_each_status() -> None:
    report = DoctorReport(
        target="fixture",
        source="local",
        plugin="text",
        results=[
            CheckResult(
                check_id="text/a",
                title="A",
                status="pass",
                severity="info",
                message="ok",
            ),
            CheckResult(
                check_id="text/b",
                title="B",
                status="warn",
                severity="medium",
                message="suspicious",
                remediation="check config",
            ),
        ],
    )

    assert report.summary == {"pass": 1, "warn": 1, "fail": 0, "skip": 0}


def test_report_copies_mutable_inputs() -> None:
    details = {"path": "config.json"}
    result = CheckResult(
        check_id="text/config.json",
        title="Config",
        status="warn",
        severity="medium",
        message="config warning",
        details=details,
    )
    results = [result]
    report = DoctorReport(target="fixture", source="local", plugin="text", results=results)

    details["path"] = "tokenizer.json"
    results.append(
        CheckResult(
            check_id="text/extra",
            title="Extra",
            status="fail",
            severity="high",
            message="extra failure",
        )
    )

    data = json.loads(render_json(report))

    assert report.summary == {"pass": 0, "warn": 1, "fail": 0, "skip": 0}
    assert data["results"][0]["details"] == {"path": "config.json"}
    with pytest.raises(TypeError):
        result.details["path"] = "other.json"


def test_invalid_status_and_severity_are_rejected() -> None:
    with pytest.raises(ValueError, match="status"):
        CheckResult(
            check_id="text/bad",
            title="Bad",
            status="unknown",
            severity="info",
            message="bad",
        )
    with pytest.raises(ValueError, match="severity"):
        CheckResult(
            check_id="text/bad",
            title="Bad",
            status="warn",
            severity="urgent",
            message="bad",
        )


def test_check_id_requires_non_empty_namespace_and_name() -> None:
    for check_id in ("plain", "/", "text/", "/required"):
        with pytest.raises(ValueError, match="check_id"):
            CheckResult(
                check_id=check_id,
                title="Bad",
                status="warn",
                severity="medium",
                message="bad",
            )


def test_json_renderer_has_schema_version_and_results() -> None:
    report = DoctorReport(
        target="fixture",
        source="local",
        plugin="text",
        results=[
            CheckResult(
                check_id="text/files.required",
                title="Required files",
                status="fail",
                severity="high",
                message="Missing config.json",
                remediation="Add config.json to the model repo.",
            )
        ],
    )

    data = json.loads(render_json(report))

    assert data["schema_version"] == "1.0"
    assert data["summary"]["fail"] == 1
    assert data["results"][0]["remediation"] == "Add config.json to the model repo."


def test_json_renderer_normalizes_non_json_detail_values() -> None:
    report = DoctorReport(
        target="fixture",
        source="local",
        plugin="text",
        results=[
            CheckResult(
                check_id="text/details",
                title="Details",
                status="warn",
                severity="medium",
                message="has details",
                details={"path": Path("config.json"), "tags": {"mlx", "text"}},
            )
        ],
        environment={"venv": Path(".venv")},
    )

    data = json.loads(render_json(report))

    assert data["environment"]["venv"] == ".venv"
    assert data["results"][0]["details"]["path"] == "config.json"
    assert data["results"][0]["details"]["tags"] == ["mlx", "text"]


def test_markdown_and_text_include_failed_check_title_and_message() -> None:
    report = DoctorReport(
        target="fixture",
        source="local",
        plugin="text",
        results=[
            CheckResult(
                check_id="text/files.required",
                title="Required files",
                status="fail",
                severity="high",
                message="Missing config.json",
            )
        ],
    )

    assert "Required files" in render_markdown(report)
    assert "Missing config.json" in render_markdown(report)
    assert "Required files" in render_text(report)
    assert "Missing config.json" in render_text(report)
