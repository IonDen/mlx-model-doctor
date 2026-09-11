"""Tests for the ParityReport text/markdown renderers.

Each test builds a REAL ``ParityReport`` (via ``tests.parity_fakes.build_parity_report``)
and asserts on rendered substrings that name the exact regression the test would
catch -- a dropped verdict/reason, a metric silently omitted, a failing check not
surfaced, or a delta-map tensor not named.
"""

import json

from mlx_model_doctor.parity.deltamap import TensorDelta
from mlx_model_doctor.parity.oracle import ParityVerdict
from mlx_model_doctor.parity.report import (
    render_parity_json,
    render_parity_markdown,
    render_parity_text,
)
from tests.parity_fakes import build_parity_report, embedded_doctor_report, parity_check_result


def test_text_renders_pass_verdict_and_agreement_metrics() -> None:
    report = build_parity_report(
        verdict=ParityVerdict.PASS,
        agree_fa=0.97,
        agree_fb=0.31,
        gap=0.66,
        noise=0.01,
        first_divergence=5,
        flip_count=4,
        adapter_applied=True,
    )
    text = render_parity_text(report)
    assert "Verdict: pass" in text
    # Regresses if agree_fa/agree_fb/gap/noise are dropped or the wrong field
    # is substituted for another (each value is chosen to be distinguishable).
    assert "0.9700" in text
    assert "0.3100" in text
    assert "0.6600" in text
    assert "0.0100" in text
    assert "first divergence:            5" in text
    # flip_count/adapter_applied were previously omitted from the text render
    # entirely -- pins that they now appear in the Agreement/summary block.
    assert "flip count:                  4" in text
    assert "adapter applied:             true" in text


def test_text_renders_null_verdict_and_reasons() -> None:
    report = build_parity_report(
        verdict=None,
        adapter_applied=None,
        agree_fa=None,
        agree_fb=None,
        gap=None,
        noise=None,
        first_divergence=None,
        reasons=("fused worker crashed; verdict cannot be computed",),
    )
    text = render_parity_text(report)
    assert "Verdict: not determined" in text
    assert "fused worker crashed; verdict cannot be computed" in text
    # Null metrics must render as an explicit sentinel, not be silently dropped.
    assert "n/a" in text
    assert "first divergence:            none" in text
    assert "adapter applied:             n/a" in text


def test_text_omits_reasons_line_when_verdict_is_not_null() -> None:
    report = build_parity_report(verdict=ParityVerdict.PASS, reasons=())
    text = render_parity_text(report)
    assert "Reason:" not in text


def test_text_surfaces_worker_errors() -> None:
    report = build_parity_report(worker_errors={"fused": "boom"})
    text = render_parity_text(report)
    assert "Worker errors:" in text
    assert "fused: boom" in text


def test_text_omits_worker_errors_section_when_empty() -> None:
    report = build_parity_report(worker_errors={})
    text = render_parity_text(report)
    assert "Worker errors:" not in text


def test_markdown_surfaces_worker_errors() -> None:
    report = build_parity_report(worker_errors={"fused": "boom"})
    markdown = render_parity_markdown(report)
    assert "## Worker errors" in markdown
    assert "**fused:** boom" in markdown


def test_markdown_omits_worker_errors_section_when_empty() -> None:
    report = build_parity_report(worker_errors={})
    markdown = render_parity_markdown(report)
    assert "## Worker errors" not in markdown


def test_text_surfaces_failing_parity_check() -> None:
    report = build_parity_report(
        results=(
            parity_check_result(
                check_id="parity/tokenizer.identity",
                status="fail",
                severity="high",
                message="tokenizers do not match",
            ),
        )
    )
    text = render_parity_text(report)
    assert "Failing checks:" in text
    assert "[parity] parity/tokenizer.identity" in text
    assert "tokenizers do not match" in text


def test_text_surfaces_failing_embedded_base_and_fused_checks() -> None:
    report = build_parity_report(
        base_report=embedded_doctor_report("base", status="fail"),
        fused_report=embedded_doctor_report("fused", status="fail"),
    )
    text = render_parity_text(report)
    assert "[base] text/a.b" in text
    assert "[fused] text/a.b" in text


def test_text_omits_failing_checks_section_when_all_pass() -> None:
    report = build_parity_report()
    text = render_parity_text(report)
    assert "Failing checks:" not in text


def test_text_summarizes_delta_map_counts_and_names_notable_tensors() -> None:
    """A real model's delta map is hundreds of tensors -- the text render must
    summarize the bulk classes and name only what a human needs to check: a
    ``changed`` target (the positive proof a fuse applied) and a ``missing``
    tensor (always a real finding), never every ``unchanged`` tensor.
    """
    report = build_parity_report(
        delta_map=(
            TensorDelta(tensor="lm_head.weight", klass="missing", reason="omitted by the fuse"),
            TensorDelta(
                tensor="model.layers.0.self_attn.q_proj.weight", klass="changed", is_target=True
            ),
            TensorDelta(tensor="model.layers.0.self_attn.k_proj.weight", klass="unchanged"),
            TensorDelta(tensor="model.layers.0.self_attn.v_proj.weight", klass="unchanged"),
        )
    )
    text = render_parity_text(report)
    assert "Delta map: 4 tensors (1 changed, 2 unchanged)" in text
    assert "missing lm_head.weight -- omitted by the fuse" in text
    assert "changed model.layers.0.self_attn.q_proj.weight" in text
    # The bulk unchanged tensors must NOT be dumped one line each.
    assert "model.layers.0.self_attn.k_proj.weight" not in text
    assert "model.layers.0.self_attn.v_proj.weight" not in text


def test_text_hides_bulk_non_comparable_but_shows_an_on_target_one() -> None:
    """A --dequantize fuse makes every tensor ``non_comparable`` -- noise unless
    it landed on an adapter target, which is still worth a human's attention.
    """
    deltas = (
        TensorDelta(
            tensor="model.layers.0.self_attn.q_proj.weight",
            klass="non_comparable",
            reason="representation changed",
            is_target=True,
        ),
        *(
            TensorDelta(tensor=f"model.layers.{i}.mlp.up_proj.weight", klass="non_comparable")
            for i in range(50)
        ),
    )
    report = build_parity_report(delta_map=deltas)
    text = render_parity_text(report)
    # The summary counts every non_comparable tensor (51: the target plus the
    # 50-tensor bulk sweep); only the on-target one is ALSO named individually.
    assert "Delta map: 51 tensors (51 non_comparable)" in text
    assert "non_comparable model.layers.0.self_attn.q_proj.weight" in text
    assert "model.layers.0.mlp.up_proj.weight" not in text


def test_text_caps_notable_delta_lines_and_reports_the_remainder() -> None:
    deltas = tuple(
        TensorDelta(
            tensor=f"model.layers.{i}.self_attn.q_proj.weight", klass="changed", is_target=True
        )
        for i in range(25)
    )
    report = build_parity_report(delta_map=deltas)
    text = render_parity_text(report)
    assert text.count("changed model.layers.") == 20
    assert "… (5 more)" in text


def test_text_omits_delta_map_section_when_empty() -> None:
    report = build_parity_report(delta_map=())
    text = render_parity_text(report)
    assert "Delta map:" not in text


def test_json_delta_map_still_lists_every_tensor_when_text_summarizes() -> None:
    """The text renderer's summarization must never leak into the JSON contract,
    which keeps the complete per-tensor list (``render_parity_json`` is untouched).
    """
    deltas = tuple(
        TensorDelta(
            tensor=f"model.layers.{i}.self_attn.q_proj.weight", klass="changed", is_target=True
        )
        for i in range(30)
    )
    report = build_parity_report(delta_map=deltas)
    payload = json.loads(render_parity_json(report))
    assert len(payload["delta_map"]) == 30


def test_markdown_renders_verdict_metrics_table_and_failing_checks() -> None:
    report = build_parity_report(
        verdict=ParityVerdict.FAIL_GROSS,
        agree_fa=0.12,
        agree_fb=0.15,
        results=(
            parity_check_result(
                check_id="parity/target.coverage",
                status="fail",
                severity="high",
                message="uncovered target",
            ),
        ),
    )
    markdown = render_parity_markdown(report)
    assert "**Verdict:** fail_gross" in markdown
    assert "| agree_fa | 0.1200 |" in markdown
    assert "| agree_fb | 0.1500 |" in markdown
    assert "## Failing checks" in markdown
    assert "### FAIL [parity] parity/target.coverage" in markdown
    assert "uncovered target" in markdown


def test_markdown_renders_null_verdict_reasons_as_blockquote() -> None:
    report = build_parity_report(verdict=None, reasons=("adapter reference is unsupported",))
    markdown = render_parity_markdown(report)
    assert "**Verdict:** not determined" in markdown
    assert "> adapter reference is unsupported" in markdown


def test_markdown_renders_delta_map_table() -> None:
    report = build_parity_report(
        delta_map=(
            TensorDelta(
                tensor="model.layers.1.mlp.up_proj.weight",
                klass="unexpected",
                reason="a non-target byte-differs",
            ),
        )
    )
    markdown = render_parity_markdown(report)
    assert "## Delta map" in markdown
    assert (
        "| model.layers.1.mlp.up_proj.weight | unexpected | a non-target byte-differs |" in markdown
    )


def test_markdown_omits_delta_map_section_when_empty() -> None:
    report = build_parity_report(delta_map=())
    markdown = render_parity_markdown(report)
    assert "## Delta map" not in markdown
