"""Tests for the pure MLX-signal core."""

from mlx_model_doctor.compat import mlx_signals


def _signals(**kw):
    base = {
        "name": None,
        "source": None,
        "tags": frozenset(),
        "library_name": None,
        "config": None,
        "has_quantized_tensors": False,
    }
    base.update(kw)
    return mlx_signals(**base)


def test_tag_mlx_from_hf_tags():
    assert _signals(source="hf", name="org/m", tags=frozenset({"MLX", "text"})) == ("tag:mlx",)


def test_library_mlx_lm():
    assert _signals(source="hf", name="org/m", library_name="mlx-lm") == ("library:mlx-lm",)


def test_library_mlx():
    assert _signals(source="hf", name="org/m", library_name="MLX") == ("library:mlx",)


def test_author_mlx_community_hf_only():
    assert _signals(source="hf", name="mlx-community/Foo") == ("author:mlx-community",)


def test_author_not_applied_for_local_paths():
    assert _signals(source="local", name="/Users/x/mlx-community/Foo") == ()


def test_config_quantization_mapping_with_positive_bits():
    cfg = {"quantization": {"bits": 4, "group_size": 64}}
    assert _signals(source="local", name="/x/m", config=cfg) == ("config:quantization",)


def test_config_quantization_excludes_hf_quantization_config():
    cfg = {"quantization_config": {"quant_method": "gptq", "bits": 4}}
    assert _signals(source="hf", name="org/m-gptq", config=cfg) == ()


def test_config_quantization_requires_positive_bits():
    assert _signals(source="local", name="/x/m", config={"quantization": {"group_size": 64}}) == ()
    assert _signals(source="local", name="/x/m", config={"quantization": True}) == ()
    assert _signals(source="local", name="/x/m", config={"quantization": {"bits": 0}}) == ()


def test_weights_mlx_quant_signal():
    assert _signals(source="hf", name="org/m", has_quantized_tensors=True) == ("weights:mlx-quant",)


def test_repo_name_is_weak_and_basename_for_local():
    assert _signals(source="local", name="/Users/x/models/Foo-mlx-8bit") == ("repo-name",)
    assert _signals(source="local", name="/Users/mlxuser/models/clean-name") == ()


def test_repo_name_hf_uses_repo_basename():
    # The weak repo-name signal matches the repo basename, not the org.
    assert _signals(source="hf", name="someorg/Llama-3-4bit") == ("repo-name",)


def test_repo_name_hf_ignores_org_only_token():
    # A token in the org component (not the repo name) must NOT fire repo-name;
    # the org is captured by author:* signals instead, keeping the meaning clean.
    assert _signals(source="hf", name="mlxorg/clean-model") == ()


def test_priority_order_tag_first_then_repo_name():
    out = _signals(source="hf", name="mlx-community/Foo-4bit", tags=frozenset({"mlx"}))
    assert out[0] == "tag:mlx"
    assert set(out) == {"tag:mlx", "author:mlx-community", "repo-name"}


def test_no_signals():
    assert _signals(source="hf", name="someorg/plain-model") == ()
