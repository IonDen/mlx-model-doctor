"""Tests for the VLM image-processor check."""

import json

import pytest

from mlx_model_doctor.checks.vlm import VlmImageProcessorCheck
from tests.fakes import context_for_files

CHECK = VlmImageProcessorCheck()


def _files(config=None, preproc=None):
    out = {}
    if config is not None:
        out["config.json"] = json.dumps(config).encode()
    if preproc is not None:
        out["preprocessor_config.json"] = json.dumps(preproc).encode()
    return out


def _run(config=None, preproc=None, source="hf", name="org/m"):
    return CHECK.run(context_for_files(_files(config, preproc), source=source, name=name))


def test_non_vlm_skips():
    r = _run(config={"model_type": "llama"})
    assert r.check_id == "text/vlm.image_processor"
    assert r.status == "skip"
    assert r.severity == "info"


def test_gate_vision_config_with_valid_type_passes():
    r = _run(
        config={"vision_config": {"x": 1}},
        preproc={"image_processor_type": "Qwen2VLImageProcessor"},
    )
    assert r.status == "pass"
    assert r.details["image_processor_type"] == "Qwen2VLImageProcessor"
    assert r.details["source"] == "preprocessor_config.json"


def test_gate_preprocessor_image_keys_only():
    r = _run(preproc={"image_mean": [0.5], "image_processor_type": "CLIPImageProcessor"})
    assert r.status == "pass"


def test_type_in_config_fallback():
    r = _run(config={"vision_config": {}, "image_processor_type": "Phi3VImageProcessor"})
    assert r.status == "pass"
    assert r.details["source"] == "config.json"


def test_precedence_preprocessor_wins():
    r = _run(
        config={"vision_config": {}, "image_processor_type": "FromConfig"},
        preproc={"image_processor_type": "FromPreproc"},
    )
    assert r.status == "pass"
    assert r.details["image_processor_type"] == "FromPreproc"
    assert r.details["source"] == "preprocessor_config.json"


def test_missing_type_fails():
    r = _run(config={"vision_config": {}})
    assert r.status == "fail"
    assert r.severity == "high"
    assert "image_processor_type" in (r.remediation or "")


def test_missing_type_preproc_only_gate_fails():
    # VLM gated solely via preprocessor image keys (CLIP-style image_mean/std,
    # no config), with no image_processor_type and no exemption -> fail.
    r = _run(preproc={"image_mean": [0.5], "image_std": [0.5]})
    assert r.status == "fail"
    assert r.severity == "high"


def test_feature_extractor_type_exempts():
    r = _run(
        config={"vision_config": {}}, preproc={"feature_extractor_type": "CLIPFeatureExtractor"}
    )
    assert r.status == "pass"


def test_auto_map_image_processor_exempts():
    r = _run(config={"vision_config": {}, "auto_map": {"AutoImageProcessor": "x--y"}})
    assert r.status == "pass"
    assert r.remediation is None  # the pass comes from the exemption path, not a fallback


def test_auto_map_feature_extractor_exempts():
    r = _run(preproc={"image_mean": [0.5], "auto_map": {"AutoFeatureExtractor": "x--y"}})
    assert r.status == "pass"
    assert r.remediation is None


def test_generic_auto_map_does_not_exempt():
    r = _run(config={"vision_config": {}, "auto_map": {"AutoModelForCausalLM": "x--y"}})
    assert r.status == "fail"


def test_audio_repo_with_feature_extractor_preprocessor_skips():
    r = _run(preproc={"feature_extractor_type": "WhisperFeatureExtractor"})
    assert r.status == "skip"


@pytest.mark.parametrize("bad", [0, [], {}, None, ""])
def test_empty_or_nonstring_type_fails(bad):
    r = _run(config={"vision_config": {}}, preproc={"image_processor_type": bad})
    assert r.status == "fail"


def test_unknown_future_type_string_passes():
    r = _run(
        config={"vision_config": {}},
        preproc={"image_processor_type": "SomeFuture2030ImageProcessor"},
    )
    assert r.status == "pass"
