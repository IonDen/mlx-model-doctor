import json
from pathlib import Path

import pytest

from mlx_model_doctor.report import (
    CheckResult,
    DoctorReport,
    github_output_lines,
    render_github,
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


def test_render_github_emits_error_and_warning_annotations() -> None:
    report = DoctorReport(
        target="org/model",
        source="hf",
        plugin="text",
        results=[
            CheckResult(
                check_id="text/files.required",
                title="Required files",
                status="fail",
                severity="high",
                message="Missing config.json",
            ),
            CheckResult(
                check_id="text/tokenizer.files",
                title="Tokenizer files",
                status="warn",
                severity="medium",
                message="No tokenizer",
            ),
            CheckResult(
                check_id="text/config.json",
                title="Config",
                status="pass",
                severity="info",
                message="ok",
            ),
            # check_id with a colon in the name segment — exercises _gh_escape_property's
            # colon→%3A replacement, which is unreachable from the name-only checks above.
            CheckResult(
                check_id="text/files.required:strict",
                title="Required files (strict)",
                status="fail",
                severity="high",
                message="Missing config.json (strict mode)",
            ),
        ],
    )

    out = render_github(report)

    assert "::error title=text/files.required::Required files: Missing config.json" in out
    assert "::warning title=text/tokenizer.files::Tokenizer files: No tokenizer" in out
    assert "::notice title=mlx-model-doctor::org/model — pass=1 warn=1 fail=2 skip=0" in out
    assert "text/config.json" not in out  # pass results are not annotated
    # colon in check_id is property-escaped to %3A in the title= field
    assert "title=text/files.required%3Astrict" in out
    assert "%3A" in out


def test_render_github_escapes_workflow_command_metacharacters() -> None:
    report = DoctorReport(
        target="x",
        source="local",
        plugin="text",
        results=[
            CheckResult(
                check_id="text/a",
                title="T",
                status="fail",
                severity="high",
                message="bad: 50% off\nsecond line",
            )
        ],
    )

    out = render_github(report)
    lines = out.splitlines()

    assert len(lines) == 2  # one error annotation + the notice summary, no raw newline injected
    assert lines[0].startswith("::error title=text/a::")
    assert "%25" in lines[0]  # percent escaped
    assert "%0A" in lines[0]  # newline escaped, stays on one line
    assert lines[0].endswith("second line")


def test_github_output_lines_includes_counts_schema_and_exit_code() -> None:
    report = DoctorReport(
        target="x",
        source="local",
        plugin="text",
        results=[
            CheckResult(
                check_id="text/a",
                title="A",
                status="fail",
                severity="high",
                message="m",
            )
        ],
    )

    lines = github_output_lines(report, exit_code=1).splitlines()

    assert "pass=0" in lines
    assert "warn=0" in lines
    assert "fail=1" in lines
    assert "skip=0" in lines
    assert "schema-version=1.0" in lines
    assert "exit-code=1" in lines
