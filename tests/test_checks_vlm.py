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


def test_missing_type_no_signal_warns():
    # VLM with no image_processor_type and no resolution signal: we can't confirm the
    # standard AutoImageProcessor path resolves it offline, so warn rather than fail.
    r = _run(config={"vision_config": {}})
    assert r.status == "warn"
    assert r.severity == "medium"
    assert "image_processor_type" in (r.remediation or "")


def test_missing_type_preproc_only_gate_warns():
    # VLM gated solely via preprocessor image keys (CLIP-style image_mean/std),
    # no image_processor_type and no resolution signal -> warn.
    r = _run(preproc={"image_mean": [0.5], "image_std": [0.5]})
    assert r.status == "warn"
    assert r.severity == "medium"


def test_feature_extractor_type_exempts():
    r = _run(
        config={"vision_config": {}}, preproc={"feature_extractor_type": "CLIPFeatureExtractor"}
    )
    assert r.status == "pass"
    assert r.details["resolution"] == "feature_extractor_type"


def test_config_only_auto_map_does_not_exempt_warns():
    # A config-only auto_map (no preprocessor_config.json) is NOT a reliable signal:
    # transformers loads the preprocessor config first and raises before reaching it.
    r = _run(config={"vision_config": {}, "auto_map": {"AutoImageProcessor": "x--y"}})
    assert r.status == "warn"


def test_preprocessor_auto_map_image_processor_exempts():
    r = _run(
        config={"vision_config": {}},
        preproc={"image_mean": [0.5], "auto_map": {"AutoImageProcessor": "x--y"}},
    )
    assert r.status == "pass"
    assert r.details["resolution"] == "preprocessor_config.json auto_map"
    assert r.remediation is None


def test_auto_map_feature_extractor_exempts():
    r = _run(preproc={"image_mean": [0.5], "auto_map": {"AutoFeatureExtractor": "x--y"}})
    assert r.status == "pass"
    assert r.remediation is None


def test_processor_class_exempts():
    # mlx-vlm loads via AutoProcessor; a processor_class resolves the image processor.
    r = _run(config={"vision_config": {}, "processor_class": "InternVLChatProcessor"})
    assert r.status == "pass"
    assert r.details["resolution"] == "processor_class"


def test_empty_feature_extractor_type_warns():
    # A blank feature_extractor_type is not a resolution signal.
    r = _run(config={"vision_config": {}}, preproc={"feature_extractor_type": ""})
    assert r.status == "warn"


def test_generic_auto_map_does_not_exempt():
    # A generic auto_map (no AutoImageProcessor/AutoFeatureExtractor) is not a resolution signal.
    r = _run(config={"vision_config": {}, "auto_map": {"AutoModelForCausalLM": "x--y"}})
    assert r.status == "warn"


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
