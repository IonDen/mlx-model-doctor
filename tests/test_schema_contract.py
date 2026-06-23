"""Contract tests for the published --format json report schema.

The schema is opaque about `details`/`environment` (additionalProperties: true);
every other object is closed (additionalProperties: false). The completeness guard
is ANCHORED to hard-coded key sets so additive OR removal drift in render_json (or
the schema) fails CI — deriving the expected set from the live output would be
tautological.
"""

import json
from importlib.resources import files
from typing import Literal

from jsonschema import Draft202012Validator

from mlx_model_doctor.report import CheckResult, DoctorReport, render_json

SCHEMA = json.loads(
    (files("mlx_model_doctor") / "schema" / "report.v1.schema.json").read_text(encoding="utf-8")
)

EXPECTED_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "target",
        "source",
        "plugin",
        "summary",
        "environment",
        "zero_check_reason",
        "results",
    }
)
EXPECTED_RESULT_KEYS = frozenset(
    {"check_id", "title", "status", "severity", "message", "remediation", "details", "duration_s"}
)
EXPECTED_SUMMARY_KEYS = frozenset({"pass", "warn", "fail", "skip"})


def _result(check_id: str, status: str, severity: str, **kw: object) -> CheckResult:
    return CheckResult(
        check_id=check_id, title="t", status=status, severity=severity, message="m", **kw
    )


def _report(
    results: list[CheckResult], source: Literal["local", "hf"] = "local", **kw: object
) -> DoctorReport:
    return DoctorReport(target="x", source=source, plugin="text", results=results, **kw)


# Representative reports covering every shape the schema must accept.
REPORTS = {
    "pass": _report([_result("text/a.b", "pass", "info")]),
    "warn": _report([_result("text/a.b", "warn", "medium")]),
    "fail": _report([_result("text/a.b", "fail", "high", remediation="do x")]),
    "skip": _report([_result("text/a.b", "skip", "info")]),
    "zero_check": _report([], zero_check_reason="no checks ran"),
    # Non-null duration_s + a nested details sub-object exercise the nullable
    # rule and the open `details` node against a real nested payload.
    "rich": _report(
        [
            _result(
                "text/a.b",
                "pass",
                "info",
                duration_s=1.23,
                details={"counts": {"x": 1, "y": 2}, "lower_bound_bytes": 4096},
            )
        ]
    ),
    # Exercise the second `source` enum value and the `low` severity so enum
    # shrinkage on either closed field fails validation.
    "hf_source": _report([_result("text/a.b", "pass", "info")], source="hf"),
    "warn_low": _report([_result("text/a.b", "warn", "low")]),
    "fail_medium": _report([_result("text/a.b", "fail", "medium")]),
    "fail_low": _report([_result("text/a.b", "fail", "low")]),
}


def test_schema_is_itself_valid() -> None:
    Draft202012Validator.check_schema(SCHEMA)


def test_representative_reports_validate() -> None:
    validator = Draft202012Validator(SCHEMA)
    for name, report in REPORTS.items():
        payload = json.loads(render_json(report))
        errors = sorted(validator.iter_errors(payload), key=str)
        assert not errors, f"{name} failed validation: {[e.message for e in errors]}"


def test_payload_keys_match_anchor() -> None:
    payload = json.loads(render_json(REPORTS["rich"]))
    assert set(payload) == EXPECTED_TOP_LEVEL_KEYS
    assert set(payload["summary"]) == EXPECTED_SUMMARY_KEYS
    assert set(payload["results"][0]) == EXPECTED_RESULT_KEYS
    assert set(json.loads(render_json(REPORTS["zero_check"]))) == EXPECTED_TOP_LEVEL_KEYS


def test_schema_properties_match_anchor() -> None:
    assert set(SCHEMA["properties"]) == EXPECTED_TOP_LEVEL_KEYS
    assert set(SCHEMA["properties"]["summary"]["properties"]) == EXPECTED_SUMMARY_KEYS
    assert set(SCHEMA["$defs"]["result"]["properties"]) == EXPECTED_RESULT_KEYS


def test_top_level_and_result_objects_are_closed() -> None:
    assert SCHEMA["additionalProperties"] is False
    assert SCHEMA["$defs"]["result"]["additionalProperties"] is False
    assert SCHEMA["properties"]["summary"]["additionalProperties"] is False


def test_open_objects_allow_additional() -> None:
    assert SCHEMA["properties"]["environment"]["additionalProperties"] is True
    assert SCHEMA["$defs"]["result"]["properties"]["details"]["additionalProperties"] is True
