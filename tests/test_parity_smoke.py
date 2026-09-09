"""Real-weights smoke test pinning the 2026-09-09 fuse-degradation calibration.

Loads real Qwen2.5-0.5B-Instruct-4bit weights plus a wikisql LoRA adapter and
feeds a held-out text-to-SQL fixture through the production parity oracle
(:mod:`mlx_model_doctor.parity.oracle`) to confirm the three-way verdict the
calibration run produced still holds: a faithful fp16 (``--dequantize``) fuse
PASSes, the default re-quantized q4 fuse is honestly INCONCLUSIVE (partial
degradation, tracking neither the adapter nor the base decisively), and
reverting the fuse to the bare base model FAILs as tracking the base.

Skips (does not fail) unless MMD_PARITY_SMOKE_BASE / _ADAPTER / _FUSED_FP16 /
_FUSED_Q4 point at real local calibration artifacts. This is a
``smoke``-marked test -- gated off by default via ``--run-smoke`` (see
``tests/conftest.py``) -- and additionally gated by these env vars so it only
actually loads models on a machine that has the calibration artifacts.

All four model loads and the verdict computation run inside a spawned child
process (see ``_smoke_worker``): loading four real models sequentially in one
process was observed to segfault at MLX/Metal interpreter teardown even
though every assertion had already passed, and a 139 exit poisons pytest's
return code for the whole ``--run-smoke`` invocation. Isolating the loads in
a child that reports its result over a queue and then calls ``os._exit(0)``
(skipping normal interpreter/MLX teardown) keeps that crash from ever
reaching the parent pytest process.

Calibrated 2026-09-09, mlx 0.32.0 / mlx-lm 0.31.3, Qwen2.5-0.5B-Instruct-4bit + wikisql LoRA, gap=0.338:
  fp16 --dequantize: agree_fa=0.974 agree_fb=0.636 -> PASS
  default q4:        agree_fa=0.779 agree_fb=0.818 -> INCONCLUSIVE (partial degradation)
  revert (fused==base): agree_fa=0.662 agree_fb=1.000 -> FAIL_TRACKS_BASE
"""

import gc
import multiprocessing
import os
from pathlib import Path
from queue import Empty

import pytest

from mlx_model_doctor.parity.oracle import (
    PARITY_GROSS_FLOOR,
    PARITY_K,
    PARITY_PASS_FLOOR,
    ParityVerdict,
    argmax_agreement,
    decide_verdict,
)

_ENV_BASE = "MMD_PARITY_SMOKE_BASE"
_ENV_ADAPTER = "MMD_PARITY_SMOKE_ADAPTER"
_ENV_FUSED_FP16 = "MMD_PARITY_SMOKE_FUSED_FP16"
_ENV_FUSED_Q4 = "MMD_PARITY_SMOKE_FUSED_Q4"

# Small retained-allocator cache bound, independent of the process-wide
# wired/memory caps `tests/conftest.py` installs at import in the parent
# (`install_mlx_memory_caps`) -- the spawned child does NOT inherit that, so
# it installs its own caps before any load. Qwen2.5-0.5B fits comfortably
# under 1 GiB, and a small cap plus the explicit `del` + `gc.collect()` +
# `mx.clear_cache()` between loads below keep only one model's buffers
# resident at a time.
_CACHE_LIMIT_BYTES = 1 * 1024**3

# How long the parent waits for the child to report a result / exit, before
# failing instead of hanging forever.
_RESULT_TIMEOUT_S = 300
_JOIN_TIMEOUT_S = 30

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


def _resolve_smoke_paths() -> dict[str, str] | None:
    """Return the four calibration model paths from env, or None if unavailable.

    ``MMD_PARITY_SMOKE_BASE`` may be an unresolved Hugging Face repo id
    (``mlx_lm.load`` resolves it through the Hub cache), so only
    non-emptiness is required for it; the other three name local directories
    the calibration run produced and must exist on disk.
    """
    base = os.environ.get(_ENV_BASE)
    adapter = os.environ.get(_ENV_ADAPTER)
    fp16_dir = os.environ.get(_ENV_FUSED_FP16)
    q4_dir = os.environ.get(_ENV_FUSED_Q4)
    if not base or not adapter or not fp16_dir or not q4_dir:
        return None
    if not (Path(adapter).exists() and Path(fp16_dir).exists() and Path(q4_dir).exists()):
        return None
    return {"base": base, "adapter": adapter, "fp16_dir": fp16_dir, "q4_dir": q4_dir}


def _build_examples(tokenizer) -> list[tuple[list[int], range]]:
    """Build ``(full_ids, scored_range)`` pairs for each held-out QA example.

    ``full_ids`` is the teacher-forced token-id list for the full
    user+assistant turn; ``scored_range`` is the slice of positions
    predicting the assistant-content tokens (completion-only scoring),
    computed as ``range(len(prompt_ids) - 1, len(full_ids) - 1)``.
    """
    examples: list[tuple[list[int], range]] = []
    for question, answer in _QA:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True,
        )
        full_ids = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            add_generation_prompt=False,
        )
        scored_range = range(len(prompt_ids) - 1, len(full_ids) - 1)
        examples.append((full_ids, scored_range))
    return examples


def _score_model(mx, model, examples: list[tuple[list[int], range]]) -> list[int]:
    """Run one teacher-forced forward pass per example, concatenate completion-only argmax."""
    scored: list[int] = []
    for full_ids, scored_range in examples:
        am = mx.argmax(model(mx.array([full_ids]))[0], axis=-1)
        mx.eval(am)
        scored.extend(int(am[i]) for i in scored_range)
    return scored


def _smoke_worker(paths: dict[str, str], out: "multiprocessing.Queue[dict[str, str]]") -> None:
    """Load the four calibration models and compute verdict names in an isolated child process.

    Runs entirely inside a spawned child so the MLX/Metal interpreter-teardown
    segfault (observed after sequentially loading four real models in one
    process) can never reach the parent pytest process: the result is sent
    back over ``out`` and the child then calls ``os._exit(0)`` to bypass
    normal Python/MLX interpreter teardown, rather than returning and falling
    into it. Any exception is reported back as ``{"error": repr(exc)}``
    instead of propagating, so the parent never hangs waiting on the queue.
    """
    try:
        # The child does not inherit pytest's conftest, so it installs the
        # MLX memory caps itself before any model load.
        from mlx_model_doctor.memory import install_mlx_memory_caps

        install_mlx_memory_caps()

        import mlx.core as mx
        from mlx_lm import load

        mx.set_cache_limit(_CACHE_LIMIT_BYTES)

        base = paths["base"]
        adapter = paths["adapter"]
        fp16_dir = paths["fp16_dir"]
        q4_dir = paths["q4_dir"]

        reference_model, tokenizer = load(base, adapter_path=adapter)
        examples = _build_examples(tokenizer)
        reference_tokens = _score_model(mx, reference_model, examples)
        del reference_model
        gc.collect()
        mx.clear_cache()

        fp16_model, _tokenizer = load(fp16_dir)
        fp16_tokens = _score_model(mx, fp16_model, examples)
        del fp16_model
        gc.collect()
        mx.clear_cache()

        q4_model, _tokenizer = load(q4_dir)
        q4_tokens = _score_model(mx, q4_model, examples)
        del q4_model
        gc.collect()
        mx.clear_cache()

        base_model, _tokenizer = load(base)
        base_tokens = _score_model(mx, base_model, examples)
        del base_model
        gc.collect()
        mx.clear_cache()

        gap = 1.0 - argmax_agreement(base_tokens, reference_tokens)
        noise = 0.0  # a second same-process base load is deterministic; no repeat load needed

        def _verdict(fused_tokens: list[int]) -> ParityVerdict:
            return decide_verdict(
                agree_fa=argmax_agreement(fused_tokens, reference_tokens),
                agree_fb=argmax_agreement(fused_tokens, base_tokens),
                gap=gap,
                noise=noise,
                k=PARITY_K,
                pass_floor=PARITY_PASS_FLOOR,
                gross_floor=PARITY_GROSS_FLOOR,
            )

        # Verdict names (strings), not raw floats, are sent back: a real
        # re-run's agreement numbers will vary slightly, but the honest
        # three-way classification must not.
        result = {
            "fp16": _verdict(fp16_tokens).name,
            "q4": _verdict(q4_tokens).name,
            "revert": _verdict(base_tokens).name,
        }
        out.put(result)
    except Exception as exc:
        out.put({"error": repr(exc)})
    finally:
        out.close()
        out.join_thread()
        os._exit(0)  # skip MLX/Metal interpreter teardown entirely


@pytest.mark.smoke
def test_real_weights_pins_2026_09_09_fuse_degradation_calibration() -> None:
    """Real Qwen + wikisql LoRA: fp16 fuse PASSes, q4 fuse is INCONCLUSIVE, revert FAILs.

    The one-line bug this pins: a future oracle/constant/scoring change that
    silently reclassifies the real default-q4 fuse as PASS or FAIL instead of
    the honest INCONCLUSIVE, or the fp16 faithful fuse as anything but PASS.
    """
    paths = _resolve_smoke_paths()
    if paths is None:
        pytest.skip(
            f"requires {_ENV_BASE}, {_ENV_ADAPTER}, {_ENV_FUSED_FP16}, and {_ENV_FUSED_Q4} "
            "pointing at real calibration artifacts (2026-09-09 fuse-degradation run)"
        )

    ctx = multiprocessing.get_context("spawn")
    out: multiprocessing.Queue[dict[str, str]] = ctx.Queue()
    process = ctx.Process(target=_smoke_worker, args=(paths, out))
    process.start()
    try:
        result = out.get(timeout=_RESULT_TIMEOUT_S)
    except Empty:
        pytest.fail(
            f"smoke worker reported no result within {_RESULT_TIMEOUT_S}s "
            "(the child process appears stuck rather than crashed)"
        )
    finally:
        process.join(timeout=_JOIN_TIMEOUT_S)

    # Rely on the returned result, not the child's exit code: the child
    # deliberately calls `os._exit(0)` to skip MLX/Metal teardown, so a
    # nonzero exit code here would not by itself mean the run was bad -- and
    # the four assertions below already give a precise failure signal.
    if "error" in result:
        pytest.fail(result["error"])

    assert result["fp16"] == ParityVerdict.PASS.name
    assert result["q4"] == ParityVerdict.INCONCLUSIVE.name
    assert result["revert"] == ParityVerdict.FAIL_TRACKS_BASE.name
