"""Networked live-Hub tests (``@pytest.mark.network``, opt-in via ``--run-network``).

Offline unit tests for the same sampling helpers live in ``test_sampling.py``.
"""

import json
import tomllib
from pathlib import Path

import pytest


def _live_records() -> tuple[dict[str, str], ...]:
    records: list[dict[str, str]] = []
    for path in sorted((Path(__file__).parent / "live").glob("known-*.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        records.extend(data["models"])
    return tuple(records)


@pytest.mark.network
@pytest.mark.parametrize("record", list(_live_records()))
def test_live_records_exercise_check_hf_tool_behavior(record: dict[str, str], capsys) -> None:
    from mlx_model_doctor import cli

    code = cli.main(["check", "hf", record["repo"], "--format", "json"])
    captured = capsys.readouterr()

    assert "Traceback" not in captured.err
    if code in {0, 1}:
        payload = json.loads(captured.out)
        assert payload["target"] == record["repo"]
        assert "summary" in payload
        return

    assert code == 2
    assert captured.err.startswith("Error: ")


@pytest.mark.network
def test_hf_safetensors_header_is_readable_without_download() -> None:
    from mlx_model_doctor.targets import HfTarget

    header = HfTarget("mlx-community/Qwen2.5-0.5B-Instruct-4bit").safetensors_header()
    assert header is not None
    entry = next(iter(header.files[0].tensors.values()))
    assert entry.dtype
    assert entry.shape
    assert entry.data_offsets[1] >= entry.data_offsets[0]


@pytest.mark.network
def test_hf_known_good_repo_has_no_failures_with_weight_checks() -> None:
    from mlx_model_doctor.api import check_hf_model

    report = check_hf_model("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    assert report.summary["fail"] == 0


def _vlm_result(report):
    return next(r for r in report.results if r.check_id == "text/vlm.image_processor")


@pytest.mark.network
def test_internvl3_passes_via_resolution_signal():
    # InternVL3 omits image_processor_type but declares a feature-extractor / custom
    # processor path, so it must PASS (not merely "not fail" — a broken gate that
    # skipped it would also avoid failing).
    from mlx_model_doctor.api import check_hf_model

    report = check_hf_model("mlx-community/InternVL3-2B-4bit")
    assert _vlm_result(report).status == "pass"


@pytest.mark.network
def test_qwen2_audio_skips_as_non_vlm():
    from mlx_model_doctor.api import check_hf_model

    report = check_hf_model("mlx-community/Qwen2-Audio-7B-Instruct-4bit")
    assert _vlm_result(report).status == "skip"


@pytest.mark.network
def test_qwen2_5_vl_passes():
    from mlx_model_doctor.api import check_hf_model

    report = check_hf_model("mlx-community/Qwen2.5-VL-3B-Instruct-bf16")
    assert _vlm_result(report).status == "pass"


def _quant_shape_result(report):
    return next(r for r in report.results if r.check_id == "text/quantization.shape")


@pytest.mark.network
def test_mxfp4_mixed_precision_repo_quant_shape_passes() -> None:
    # gpt-oss-20b-MXFP4-Q8 mixes mxfp4 experts with 8-bit affine dense layers; the
    # per-layer shape check must PASS it (not merely "not fail" — a regression that turned
    # every overridden layer "unverified"/warn, or tripped the skip branch, would also avoid
    # failing). Reproduced the false-fail on the flat-formula code: fail/high
    # inconsistent_layers=('lm_head', ...). All resolved layers are consistent (verified).
    from mlx_model_doctor.api import check_hf_model

    report = check_hf_model("mlx-community/gpt-oss-20b-MXFP4-Q8")
    assert _quant_shape_result(report).status == "pass"


@pytest.mark.network
def test_nvfp4_mixed_precision_repo_quant_shape_passes() -> None:
    # Like the mxfp4 case, this is a genuine old-fail->new-pass guard, not a vacuous pass:
    # the flat-formula code returned fail/high inconsistent_layers=(...mlp.gate,
    # ...shared_expert_gate) for this repo (those gate layers are 8-bit overrides stored as
    # U32 weights), so the check does measure these layers; per-layer resolution makes them
    # consistent.
    from mlx_model_doctor.api import check_hf_model

    report = check_hf_model("mlx-community/Qwen3.6-35B-A3B-nvfp4")
    assert _quant_shape_result(report).status == "pass"


def _quant_mode_result(report):
    return next(r for r in report.results if r.check_id == "text/quantization.mode")


@pytest.mark.network
def test_mxfp4_mixed_precision_repo_quant_mode_passes() -> None:
    # GUARD (not RED->GREEN): the mode check already passed this repo on the old code because it
    # only read the canonical scalar default (mxfp4/32/4). After per-layer validation it still
    # passes — the per-layer overrides resolve to affine 8/64, which is in-table. Asserting on the
    # specific check id (not the overall report) so a different check skipping can't mask a regression.
    from mlx_model_doctor.api import check_hf_model

    report = check_hf_model("mlx-community/gpt-oss-20b-MXFP4-Q8")
    assert _quant_mode_result(report).status == "pass"


@pytest.mark.network
def test_nvfp4_mixed_precision_repo_quant_mode_passes() -> None:
    # GUARD (see above): scalar nvfp4 (16/4) canonical; the 80 mode-less {group_size:64, bits:8}
    # overrides resolve to affine 8/64 (their .biases tensors confirm affine) -> in-table -> pass.
    from mlx_model_doctor.api import check_hf_model

    report = check_hf_model("mlx-community/Qwen3.6-35B-A3B-nvfp4")
    assert _quant_mode_result(report).status == "pass"
