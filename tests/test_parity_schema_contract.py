"""Contract tests for the published parity.v1 --format json report schema.

Mirrors ``tests/test_schema_contract.py``'s technique: the schema is opaque about
``details``/``environment`` (inherited via the embedded report.v1 payload) but every
other object here is closed (``additionalProperties: false``). The completeness
guard is ANCHORED to hard-coded key sets so additive OR removal drift in
``render_parity_json`` (or the schema) fails CI. The per-result object reuses the
*same* ``EXPECTED_RESULT_KEYS`` frozenset the report.v1 contract test anchors to,
since ``ParityReport.results`` holds plain ``CheckResult`` objects, not embedded
reports. ``base_report``/``fused_report`` are different: those are whole embedded
``DoctorReport`` payloads, validated via a `$ref` to ``report.v1``'s own `$id`
through a ``referencing.Registry`` -- the same cross-file technique
``sample-batch.v1.schema.json`` uses for its ``items[].report`` field.
"""

import json
from importlib.resources import files

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from mlx_model_doctor.parity.deltamap import TensorDelta
from mlx_model_doctor.parity.fixtures import DEFAULT_FIXTURE_ID, get_fixture
from mlx_model_doctor.parity.oracle import ParityVerdict
from mlx_model_doctor.parity.report import (
    ParityReport,
    ResolvedIdentity,
    RuntimeProvenance,
    parity_report_to_dict,
    render_parity_json,
)
from mlx_model_doctor.report import CheckResult, DoctorReport
from tests.test_schema_contract import EXPECTED_RESULT_KEYS
from tests.test_schema_contract import SCHEMA as REPORT_SCHEMA

PARITY_SCHEMA = json.loads(
    (files("mlx_model_doctor") / "schema" / "parity.v1.schema.json").read_text(encoding="utf-8")
)
# The parity schema embeds two full `check` reports (base_report/fused_report) via a
# $ref to the report schema's $id; a registry makes that cross-file reference resolvable.
REGISTRY = Registry().with_resources(
    [
        (REPORT_SCHEMA["$id"], Resource.from_contents(REPORT_SCHEMA)),
        (PARITY_SCHEMA["$id"], Resource.from_contents(PARITY_SCHEMA)),
    ]
)

EXPECTED_PARITY_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "base",
        "adapter",
        "fused",
        "tokenizer_fingerprint",
        "fixture",
        "tool_version",
        "mlx_version",
        "mlx_lm_version",
        "provenance",
        "base_report",
        "fused_report",
        "results",
        "verdict",
        "agree_fa",
        "agree_fb",
        "gap",
        "noise",
        "first_divergence",
        "flip_count",
        "adapter_applied",
        "delta_map",
        "phase_outcomes",
        "worker_status",
        "peak_bytes",
        "reasons",
    }
)
EXPECTED_IDENTITY_KEYS = frozenset({"path", "original_ref", "source"})
EXPECTED_TOKENIZER_FINGERPRINT_KEYS = frozenset(
    {"vocab_size", "special_tokens_digest", "token_id_map_digest", "chat_template_digest"}
)
EXPECTED_FIXTURE_KEYS = frozenset({"id", "input_digest", "max_length", "scored_positions"})
EXPECTED_PROVENANCE_KEYS = frozenset({"chip", "os", "backend", "weight_dtype", "compute_note"})
EXPECTED_TENSOR_DELTA_KEYS = frozenset({"tensor", "klass", "reason"})


def _identity(role: str, *, source: str = "local") -> ResolvedIdentity:
    original = f"mlx-community/{role}-model" if source == "hf" else f"/models/{role}"
    return ResolvedIdentity(path=f"/snapshots/{role}", original_ref=original, source=source)  # type: ignore[arg-type]


_FIXTURE_REF, _FIXTURE_TOKEN_IDS = get_fixture(DEFAULT_FIXTURE_ID)


def _check_result(status: str = "pass", severity: str = "info", **kw: object) -> CheckResult:
    return CheckResult(
        check_id="parity/adapter.config_well_formed",
        title="Adapter config well-formed",
        status=status,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        message="m",
        **kw,
    )


def _doctor_report(name: str, *, source: str = "local", status: str = "pass") -> DoctorReport:
    return DoctorReport(
        target=name,
        source=source,  # type: ignore[arg-type]
        plugin="text",
        results=[
            CheckResult(
                check_id="text/a.b",
                title="t",
                status=status,  # type: ignore[arg-type]
                severity="info" if status in {"pass", "skip"} else "high",
                message="m",
            )
        ],
    )


def _report(**overrides: object) -> ParityReport:
    defaults: dict[str, object] = {
        "base": _identity("base"),
        "adapter": _identity("adapter"),
        "fused": _identity("fused"),
        "tokenizer_fingerprint": _FIXTURE_REF.tokenizer_fingerprint,
        "fixture": _FIXTURE_REF,
        "tool_version": "0.9.0",
        "mlx_version": "0.32.0",
        "mlx_lm_version": "0.31.3",
        "provenance": RuntimeProvenance(
            chip="Apple M1 Max",
            os="macOS 15",
            backend="metal",
            weight_dtype="bfloat16",
            compute_note="scoped top-token agreement",
        ),
        "base_report": _doctor_report("base"),
        "fused_report": _doctor_report("fused"),
        "results": (_check_result(),),
        "verdict": ParityVerdict.PASS,
        "agree_fa": 0.95,
        "agree_fb": 0.4,
        "gap": 0.6,
        "noise": 0.02,
        "first_divergence": 3,
        "flip_count": 2,
        "adapter_applied": True,
        "delta_map": (
            TensorDelta(tensor="model.layers.0.self_attn.q_proj.weight", klass="changed"),
        ),
        "phase_outcomes": {
            "static": "ok",
            "reference": "ok",
            "tokenizer_gate": "ok",
            "oracle": "ok",
        },
        "worker_status": {"base": "ok", "adapter": "ok", "fused": "ok", "base_repeat": "ok"},
        "peak_bytes": {"base": 1024, "adapter": 512, "fused": 1024, "base_repeat": 1024},
        "reasons": (),
    }
    defaults.update(overrides)
    return ParityReport(**defaults)  # type: ignore[arg-type]


# Representative reports covering every branch of the (nullable) runtime state.
REPORTS = {
    "pass": _report(),
    "fail_tracks_base": _report(
        verdict=ParityVerdict.FAIL_TRACKS_BASE,
        agree_fa=0.3,
        agree_fb=0.92,
        adapter=_identity("adapter", source="hf"),
        delta_map=(
            TensorDelta(tensor="model.layers.0.self_attn.q_proj.weight", klass="unchanged"),
        ),
    ),
    "fail_gross": _report(
        verdict=ParityVerdict.FAIL_GROSS,
        agree_fa=0.1,
        agree_fb=0.15,
        delta_map=(
            TensorDelta(tensor="lm_head.weight", klass="missing", reason="omitted by the fuse"),
            TensorDelta(
                tensor="model.layers.1.mlp.up_proj.weight",
                klass="unexpected",
                reason="a non-target byte-differs",
            ),
        ),
    ),
    "inconclusive": _report(
        verdict=ParityVerdict.INCONCLUSIVE,
        agree_fa=0.81,
        agree_fb=0.80,
        delta_map=(
            TensorDelta(
                tensor="model.layers.0.self_attn.q_proj.weight",
                klass="non_comparable",
                reason="a --dequantize fuse changed the whole model's representation",
            ),
        ),
    ),
    # A blocking phase short-circuit: verdict/metrics/adapter_applied are all null,
    # worker_status/phase_outcomes/peak_bytes record why, and reasons explains it.
    "null_verdict_blocking_worker_error": _report(
        verdict=None,
        agree_fa=None,
        agree_fb=None,
        gap=None,
        noise=None,
        first_divergence=None,
        flip_count=None,
        adapter_applied=None,
        mlx_version=None,
        mlx_lm_version=None,
        delta_map=(),
        results=(_check_result(status="fail", severity="high"), _check_result(status="skip")),
        phase_outcomes={
            "static": "ok",
            "reference": "ok",
            "tokenizer_gate": "ok",
            "oracle": "error",
            "delta_map": "skipped",
        },
        worker_status={"base": "ok", "adapter": "ok", "fused": "error", "base_repeat": "skipped"},
        peak_bytes={"base": 1024, "adapter": 512, "fused": None, "base_repeat": None},
        reasons=("fused worker crashed; verdict cannot be computed",),
    ),
    # A tokenizer-mismatch void-skip: unsupported adapter reference (F4).
    "null_verdict_unsupported_adapter": _report(
        verdict=None,
        agree_fa=None,
        agree_fb=None,
        gap=None,
        noise=None,
        first_divergence=None,
        flip_count=None,
        adapter_applied=None,
        delta_map=(),
        phase_outcomes={
            "static": "ok",
            "reference": "unsupported",
            "tokenizer_gate": "skipped",
            "oracle": "skipped",
            "delta_map": "skipped",
        },
        worker_status={"base": "ok", "adapter": "ok", "fused": "ok", "base_repeat": "ok"},
        reasons=("adapter fine_tune_type is unsupported; no valid reference could be built",),
    ),
}


def test_schema_is_itself_valid() -> None:
    Draft202012Validator.check_schema(PARITY_SCHEMA)


def test_representative_reports_validate() -> None:
    validator = Draft202012Validator(PARITY_SCHEMA, registry=REGISTRY)
    for name, report in REPORTS.items():
        payload = json.loads(render_parity_json(report))
        errors = sorted(validator.iter_errors(payload), key=str)
        assert not errors, f"{name} failed validation: {[e.message for e in errors]}"


def test_null_verdict_and_metrics_validate() -> None:
    # Names the exact bug this guards: a runtime field typed non-nullable (missing
    # "null" from its schema `type` array) would reject this report even though a
    # blocking phase short-circuit is a normal, spec-required outcome (F4).
    validator = Draft202012Validator(PARITY_SCHEMA, registry=REGISTRY)
    payload = json.loads(render_parity_json(REPORTS["null_verdict_blocking_worker_error"]))
    assert payload["verdict"] is None
    assert payload["agree_fa"] is None
    assert payload["agree_fb"] is None
    assert payload["gap"] is None
    assert payload["noise"] is None
    assert payload["first_divergence"] is None
    assert payload["flip_count"] is None
    assert payload["adapter_applied"] is None
    assert payload["peak_bytes"]["fused"] is None
    errors = list(validator.iter_errors(payload))
    assert not errors, [e.message for e in errors]


def test_payload_keys_match_anchor() -> None:
    payload = json.loads(render_parity_json(REPORTS["pass"]))
    assert set(payload) == EXPECTED_PARITY_TOP_LEVEL_KEYS
    assert set(payload["results"][0]) == EXPECTED_RESULT_KEYS
    assert set(payload["base"]) == EXPECTED_IDENTITY_KEYS
    assert set(payload["adapter"]) == EXPECTED_IDENTITY_KEYS
    assert set(payload["fused"]) == EXPECTED_IDENTITY_KEYS
    assert set(payload["tokenizer_fingerprint"]) == EXPECTED_TOKENIZER_FINGERPRINT_KEYS
    assert set(payload["fixture"]) == EXPECTED_FIXTURE_KEYS
    assert set(payload["provenance"]) == EXPECTED_PROVENANCE_KEYS
    assert set(payload["delta_map"][0]) == EXPECTED_TENSOR_DELTA_KEYS
    # The null-metrics report has the exact same key set -- nulling values never
    # drops or adds a top-level key.
    null_payload = json.loads(render_parity_json(REPORTS["null_verdict_blocking_worker_error"]))
    assert set(null_payload) == EXPECTED_PARITY_TOP_LEVEL_KEYS


def test_schema_properties_match_anchor() -> None:
    assert set(PARITY_SCHEMA["properties"]) == EXPECTED_PARITY_TOP_LEVEL_KEYS
    assert set(PARITY_SCHEMA["$defs"]["result"]["properties"]) == EXPECTED_RESULT_KEYS
    assert set(PARITY_SCHEMA["$defs"]["identity"]["properties"]) == EXPECTED_IDENTITY_KEYS
    assert (
        set(PARITY_SCHEMA["$defs"]["tokenizer_fingerprint"]["properties"])
        == EXPECTED_TOKENIZER_FINGERPRINT_KEYS
    )
    assert set(PARITY_SCHEMA["$defs"]["fixture"]["properties"]) == EXPECTED_FIXTURE_KEYS
    assert set(PARITY_SCHEMA["$defs"]["provenance"]["properties"]) == EXPECTED_PROVENANCE_KEYS
    assert set(PARITY_SCHEMA["$defs"]["tensor_delta"]["properties"]) == EXPECTED_TENSOR_DELTA_KEYS


def test_top_level_and_named_objects_are_closed() -> None:
    assert PARITY_SCHEMA["additionalProperties"] is False
    for def_name in (
        "result",
        "identity",
        "tokenizer_fingerprint",
        "fixture",
        "provenance",
        "tensor_delta",
    ):
        assert PARITY_SCHEMA["$defs"][def_name]["additionalProperties"] is False, def_name


def test_worker_state_maps_accept_arbitrary_role_or_phase_keys() -> None:
    # phase_outcomes/worker_status/peak_bytes are keyed by dynamic role/phase names
    # (not a fixed schema) -- a report using a role name never seen in these tests
    # ("base_repeat", chosen here) must still validate, proving the schema doesn't
    # pin the key set the way it pins the identity/result/etc. objects above.
    validator = Draft202012Validator(PARITY_SCHEMA, registry=REGISTRY)
    payload = json.loads(render_parity_json(REPORTS["pass"]))
    assert "base_repeat" in payload["worker_status"]
    assert "base_repeat" in payload["peak_bytes"]
    errors = list(validator.iter_errors(payload))
    assert not errors, [e.message for e in errors]


def test_verdict_enum_matches_parity_verdict_values() -> None:
    # A future ParityVerdict member added without a matching schema enum entry
    # would go undetected by the representative-report tests above (which only
    # exercise the members REPORTS happens to use) -- this pins the full set.
    schema_values = {v for v in PARITY_SCHEMA["properties"]["verdict"]["enum"] if v is not None}
    assert schema_values == {v.value for v in ParityVerdict}
    assert None in PARITY_SCHEMA["properties"]["verdict"]["enum"]


def test_delta_klass_enum_matches_tensor_delta_klasses() -> None:
    from mlx_model_doctor.parity.deltamap import _ALL_KLASSES

    assert set(PARITY_SCHEMA["$defs"]["tensor_delta"]["properties"]["klass"]["enum"]) == set(
        _ALL_KLASSES
    )


def test_ref_liveness_for_embedded_base_report() -> None:
    # Inject a stray key into the embedded base_report: the $ref to the closed
    # report.v1 schema must reject it, proving the cross-file ref is live (F4).
    validator = Draft202012Validator(PARITY_SCHEMA, registry=REGISTRY)
    payload = json.loads(render_parity_json(REPORTS["pass"]))
    payload["base_report"]["UNEXPECTED"] = 1
    errors = list(validator.iter_errors(payload))
    assert any("UNEXPECTED" in e.message for e in errors), errors


def test_ref_liveness_for_embedded_fused_report() -> None:
    # Same proof for fused_report -- a separate schema property with its own $ref,
    # so base_report validating correctly does not by itself prove this one does.
    validator = Draft202012Validator(PARITY_SCHEMA, registry=REGISTRY)
    payload = json.loads(render_parity_json(REPORTS["pass"]))
    payload["fused_report"]["UNEXPECTED"] = 1
    errors = list(validator.iter_errors(payload))
    assert any("UNEXPECTED" in e.message for e in errors), errors


def test_parity_report_to_dict_matches_render_parity_json() -> None:
    report = REPORTS["pass"]
    assert parity_report_to_dict(report) == json.loads(render_parity_json(report))
