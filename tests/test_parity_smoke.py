"""Real-weights smoke test exercising the adapter-parity product path (F1/F2/F9).

Drives :func:`mlx_model_doctor.api.check_adapter_parity` exactly the way a
real caller does: a fixture built from real ``(prompt, completion)`` pairs via
:func:`mlx_model_doctor.parity.prompts.build_prompts_fixture`, and
``options.launcher`` left ``None`` so the real ``SubprocessLauncher`` runs
each of the four teacher-forced worker loads in its OWN subprocess. Confirms
the 2026-09-09 fuse-degradation calibration still holds against real
Qwen2.5-0.5B-Instruct-4bit weights plus a wikisql LoRA adapter: a faithful
fp16 (``--dequantize``) fuse PASSes, the default re-quantized q4 fuse is
honestly INCONCLUSIVE (partial degradation), and reverting the fuse to the
bare base model FAILs as tracking the base.

Because every model load happens in its own subprocess worker, the pytest
PARENT process never imports ``mlx`` or ``transformers`` at all -- so the
MLX/Metal interpreter-teardown segfault that motivated an earlier
spawned-worker/``os._exit`` workaround in this file cannot occur here, and
that scaffolding has been removed.

Skips (does not fail) unless MMD_PARITY_SMOKE_BASE / _ADAPTER / _FUSED_FP16 /
_FUSED_Q4 all name existing local directories with the calibration artifacts.
This is a ``smoke``-marked test -- gated off by default via ``--run-smoke``
(see ``tests/conftest.py``) -- and additionally gated by these env vars so it
only actually loads models on a machine that has the calibration artifacts.

The one-line bug this pins: a future oracle/constant/scoring/worker change
silently reclassifies the real default-q4 fuse as PASS or FAIL, or the
faithful fp16 fuse as not-PASS, through the real product path (CLI/API +
subprocess workers).
"""

import json
import os
from pathlib import Path

import pytest

from mlx_model_doctor.api import ParityOptions, check_adapter_parity
from mlx_model_doctor.parity.oracle import ParityVerdict
from mlx_model_doctor.parity.prompts import build_prompts_fixture

_ENV_BASE = "MMD_PARITY_SMOKE_BASE"
_ENV_ADAPTER = "MMD_PARITY_SMOKE_ADAPTER"
_ENV_FUSED_FP16 = "MMD_PARITY_SMOKE_FUSED_FP16"
_ENV_FUSED_Q4 = "MMD_PARITY_SMOKE_FUSED_Q4"

# Held-out text-to-SQL QA pairs (wikisql-style), never used to train the
# adapter -- the same fixture the 2026-09-09 calibration scored.
_QA: tuple[tuple[str, str], ...] = (
    (
        "What is the total number of years for the player from Duke?",
        "SELECT COUNT(Years) FROM table WHERE College = 'Duke'",
    ),
    (
        "Which city has the highest population in the state of Texas?",
        "SELECT City FROM table WHERE State = 'Texas' ORDER BY Population DESC LIMIT 1",
    ),
    (
        "Name the coach for the team with 12 wins.",
        "SELECT Coach FROM table WHERE Wins = 12",
    ),
    (
        "What is the score when the opponent was the Lakers?",
        "SELECT Score FROM table WHERE Opponent = 'Lakers'",
    ),
    (
        "How many losses does the team from Ohio have?",
        "SELECT COUNT(Losses) FROM table WHERE State = 'Ohio'",
    ),
)


def _resolve_smoke_dirs() -> dict[str, str] | None:
    """Return the four calibration model directories from env, or None if unavailable.

    All four env vars must point at existing local directories -- unlike the
    prior worker-based version, ``build_prompts_fixture`` needs a
    locally-resolvable base to load its tokenizer, so the base is required to
    be an existing directory too, not a bare Hugging Face repo id.
    """
    base = os.environ.get(_ENV_BASE)
    adapter = os.environ.get(_ENV_ADAPTER)
    fp16_dir = os.environ.get(_ENV_FUSED_FP16)
    q4_dir = os.environ.get(_ENV_FUSED_Q4)
    if base is None or adapter is None or fp16_dir is None or q4_dir is None:
        return None
    if not (
        Path(base).is_dir()
        and Path(adapter).is_dir()
        and Path(fp16_dir).is_dir()
        and Path(q4_dir).is_dir()
    ):
        return None
    return {"base": base, "adapter": adapter, "fp16": fp16_dir, "q4": q4_dir}


def _write_prompts_json(path: Path) -> None:
    """Write the held-out QA pairs as the JSON array ``build_prompts_fixture`` reads."""
    payload = [{"prompt": question, "completion": answer} for question, answer in _QA]
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.smoke
def test_real_weights_pins_2026_09_09_fuse_degradation_calibration(tmp_path: Path) -> None:
    """Real Qwen + wikisql LoRA through the real product path: fp16 PASSes, q4 is INCONCLUSIVE, revert FAILs."""
    dirs = _resolve_smoke_dirs()
    if dirs is None:
        pytest.skip(
            f"requires {_ENV_BASE}, {_ENV_ADAPTER}, {_ENV_FUSED_FP16}, and {_ENV_FUSED_Q4} "
            "pointing at existing local calibration-artifact directories (2026-09-09 "
            "fuse-degradation run)"
        )

    prompts_path = tmp_path / "prompts.json"
    _write_prompts_json(prompts_path)
    fixture = build_prompts_fixture(dirs["base"], str(prompts_path))

    reports = {}
    for label, fused_path in [("fp16", dirs["fp16"]), ("q4", dirs["q4"]), ("revert", dirs["base"])]:
        report = check_adapter_parity(
            base=dirs["base"],
            adapter=dirs["adapter"],
            fused=fused_path,
            options=ParityOptions(fixture=fixture),
        )
        print(
            f"{label}: fa={report.agree_fa} fb={report.agree_fb} gap={report.gap} verdict={report.verdict}"
        )
        reports[label] = report

    # Calibrated 2026-09-09, mlx 0.32.0 / mlx-lm 0.31.3, Qwen2.5-0.5B-Instruct-4bit + wikisql LoRA, 5-example fixture (per-example forwards), gap≈0.338:
    #   fp16 --dequantize -> PASS ; default q4 -> INCONCLUSIVE (partial degradation) ; fused==base -> FAIL_TRACKS_BASE
    assert reports["fp16"].verdict == ParityVerdict.PASS
    assert reports["q4"].verdict == ParityVerdict.INCONCLUSIVE
    assert reports["revert"].verdict == ParityVerdict.FAIL_TRACKS_BASE
