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
from referencing import Registry, Resource

from mlx_model_doctor.report import CheckResult, DoctorReport, render_json
from mlx_model_doctor.sampling import (
    SampleBatchReport,
    SampledModelResult,
    render_sample_batch_json,
)

SCHEMA = json.loads(
    (files("mlx_model_doctor") / "schema" / "report.v1.schema.json").read_text(encoding="utf-8")
)
BATCH_SCHEMA = json.loads(
    (files("mlx_model_doctor") / "schema" / "sample-batch.v1.schema.json").read_text(
        encoding="utf-8"
    )
)
# The batch schema embeds a full `check` report under each checked item via a $ref
# to the report schema's $id; a registry makes that cross-file reference resolvable.
BATCH_REGISTRY = Registry().with_resources(
    [
        (SCHEMA["$id"], Resource.from_contents(SCHEMA)),
        (BATCH_SCHEMA["$id"], Resource.from_contents(BATCH_SCHEMA)),
    ]
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


# --- sample hf batch output contract -----------------------------------------

EXPECTED_BATCH_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "source", "author", "task", "limit", "plugin", "summary", "items"}
)
EXPECTED_BATCH_SUMMARY_KEYS = frozenset({"checked", "tool-error"})
EXPECTED_BATCH_ITEM_KEYS = frozenset({"repo_id", "signal", "status", "error", "report"})


def _batch(items: list[SampledModelResult], *, task: str | None = None, limit: int = 5) -> dict:
    batch = SampleBatchReport(
        author="mlx-community", task=task, limit=limit, plugin="text", items=items
    )
    return json.loads(render_sample_batch_json(batch))


_CHECKED_ITEM = SampledModelResult(
    repo_id="mlx-community/m",
    signal="config:quantization",
    status="checked",
    report=_report([_result("text/a.b", "pass", "info")], source="hf"),
)
_TOOL_ERROR_ITEM = SampledModelResult(
    repo_id="mlx-community/broken",
    signal="author:mlx-community",
    status="tool-error",
    error="Could not inspect Hugging Face model",
)

BATCH_PAYLOADS = {
    "mixed": _batch([_CHECKED_ITEM, _TOOL_ERROR_ITEM]),
    "empty": _batch([], task="text-generation", limit=0),
    "checked_only": _batch([_CHECKED_ITEM]),
    "tool_error_only": _batch([_TOOL_ERROR_ITEM]),
}


def test_batch_schema_is_itself_valid() -> None:
    Draft202012Validator.check_schema(BATCH_SCHEMA)


def test_batch_payload_advertises_the_published_v1_schema_family() -> None:
    # The serialized batch stamps its own schema_version; it must name the family the
    # published v1 schema covers (sample-batch/1.x) so a consumer selects this schema.
    # A major bump (e.g. sample-batch/2.0) shipped without a new schema file fails here.
    for name, payload in BATCH_PAYLOADS.items():
        assert payload["schema_version"].startswith("sample-batch/1."), name


def test_representative_batches_validate() -> None:
    validator = Draft202012Validator(BATCH_SCHEMA, registry=BATCH_REGISTRY)
    for name, payload in BATCH_PAYLOADS.items():
        errors = sorted(validator.iter_errors(payload), key=str)
        assert not errors, f"{name} failed validation: {[e.message for e in errors]}"


def test_batch_payload_keys_match_anchor() -> None:
    payload = BATCH_PAYLOADS["mixed"]
    assert set(payload) == EXPECTED_BATCH_TOP_LEVEL_KEYS
    assert set(payload["summary"]) == EXPECTED_BATCH_SUMMARY_KEYS
    checked = next(item for item in payload["items"] if item["status"] == "checked")
    tool_error = next(item for item in payload["items"] if item["status"] == "tool-error")
    assert set(checked) == EXPECTED_BATCH_ITEM_KEYS
    # `report` is present only on checked items; a tool-error item omits it.
    assert set(tool_error) == EXPECTED_BATCH_ITEM_KEYS - {"report"}


def test_batch_schema_properties_match_anchor() -> None:
    assert set(BATCH_SCHEMA["properties"]) == EXPECTED_BATCH_TOP_LEVEL_KEYS
    assert set(BATCH_SCHEMA["properties"]["summary"]["properties"]) == EXPECTED_BATCH_SUMMARY_KEYS
    assert set(BATCH_SCHEMA["$defs"]["item"]["properties"]) == EXPECTED_BATCH_ITEM_KEYS


def test_batch_objects_are_closed() -> None:
    assert BATCH_SCHEMA["additionalProperties"] is False
    assert BATCH_SCHEMA["properties"]["summary"]["additionalProperties"] is False
    assert BATCH_SCHEMA["$defs"]["item"]["additionalProperties"] is False


def test_batch_embedded_report_is_validated_against_the_report_schema() -> None:
    # Inject a stray key into a checked item's embedded report: the $ref to the
    # closed report schema must reject it, proving the cross-file ref is live.
    validator = Draft202012Validator(BATCH_SCHEMA, registry=BATCH_REGISTRY)
    payload = json.loads(json.dumps(BATCH_PAYLOADS["checked_only"]))
    payload["items"][0]["report"]["UNEXPECTED"] = 1
    errors = list(validator.iter_errors(payload))
    assert any("UNEXPECTED" in e.message for e in errors), errors
