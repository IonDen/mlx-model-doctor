from mlx_model_doctor.plugins import BUILTIN_PLUGINS, get_plugin
from mlx_model_doctor.plugins.vlm import VlmModelPlugin


def test_builtin_plugins_include_text_and_vlm() -> None:
    assert tuple(BUILTIN_PLUGINS) == ("text", "vlm")
    assert get_plugin("vlm").name == "vlm"


def test_vlm_plugin_check_inventory_is_ordered() -> None:
    plugin = VlmModelPlugin()

    assert [check.check_id for check in plugin.static_checks()] == [
        "vlm/files.required",
        "vlm/config.json",
        "vlm/config.model_type",
        "vlm/compat.mlx_signal",
        "vlm/tokenizer.files",
        "vlm/tokenizer.special_tokens",
        "vlm/chat_template.presence",
        "vlm/chat_template.special_tokens",
        "vlm/safetensors.index",
        "vlm/quantization.metadata",
        "vlm/quantization.mode",
        "vlm/image_processor",
        "vlm/image_token.wiring",
        "vlm/generation_config.tokens",
        "vlm/memory.estimate",
    ]
    assert [check.check_id for check in plugin.weight_checks()] == [
        "vlm/safetensors.offsets",
        "vlm/weights.param_count",
        "vlm/weights.tied_embedding",
        "vlm/quantization.shape",
    ]
    assert plugin.smoke_checks() == ()
