import json

from mlx_model_doctor.checks.chat_template import (
    ChatTemplatePresenceCheck,
    ChatTemplateSpecialTokensCheck,
)
from tests.fakes import context_for_files


def _ctx(files: dict[str, bytes]):
    return context_for_files(files)


def test_presence_passes_when_template_in_tokenizer_config() -> None:
    files = {"tokenizer_config.json": json.dumps({"chat_template": "{{ x }}"}).encode()}
    result = ChatTemplatePresenceCheck().run(_ctx(files))
    assert result.status == "pass"
    assert result.severity == "info"


def test_presence_passes_when_template_only_in_jinja_sibling() -> None:
    files = {
        "tokenizer_config.json": json.dumps({"eos_token": "<|im_end|>"}).encode(),
        "chat_template.jinja": b"{{ messages }}",
    }
    result = ChatTemplatePresenceCheck().run(_ctx(files))
    assert result.status == "pass"


def test_presence_warns_when_template_absent_in_both() -> None:
    files = {"tokenizer_config.json": json.dumps({"eos_token": "<|im_end|>"}).encode()}
    result = ChatTemplatePresenceCheck().run(_ctx(files))
    assert result.status == "warn"
    assert result.severity == "low"
    assert "base" in result.message.lower() or "non-chat" in result.message.lower()


def test_presence_skips_when_no_tokenizer_metadata() -> None:
    result = ChatTemplatePresenceCheck().run(_ctx({}))
    assert result.status == "skip"
    assert result.severity == "info"


def test_presence_warns_when_tokenizer_config_present_but_unparseable() -> None:
    result = ChatTemplatePresenceCheck().run(_ctx({"tokenizer_config.json": b"{not-json"}))
    assert result.status == "warn"
    assert "parse" in result.message.lower() or "read" in result.message.lower()


def _tokenizer_config(eos: str, added: list[str]) -> bytes:
    decoder = {str(i): {"content": c} for i, c in enumerate(added)}
    return json.dumps({"eos_token": eos, "added_tokens_decoder": decoder}).encode()


def test_special_tokens_pass_when_template_literal_is_registered() -> None:
    files = {
        "tokenizer_config.json": _tokenizer_config("<|im_end|>", ["<|im_end|>", "<|im_start|>"]),
        "chat_template.jinja": b"{{ m }}<|im_end|>",
    }
    result = ChatTemplateSpecialTokensCheck().run(context_for_files(files))
    assert result.status == "pass"


def test_special_tokens_warns_on_template_literal_not_registered() -> None:
    files = {
        "tokenizer_config.json": _tokenizer_config("<|im_end>", ["<|im_end>"]),
        "chat_template.jinja": b"{{ m }}<|im_end|>",
    }
    result = ChatTemplateSpecialTokensCheck().run(context_for_files(files))
    assert result.status == "warn"
    assert result.severity == "medium"
    assert list(result.details["unregistered"]) == ["<|im_end|>"]


def test_special_tokens_skips_without_template() -> None:
    files = {"tokenizer_config.json": _tokenizer_config("<|im_end|>", ["<|im_end|>"])}
    result = ChatTemplateSpecialTokensCheck().run(context_for_files(files))
    assert result.status == "skip"


def test_presence_passes_for_list_form_chat_template() -> None:
    files = {
        "tokenizer_config.json": json.dumps(
            {"chat_template": [{"name": "default", "template": "{{ m }}<|im_end|>"}]}
        ).encode()
    }
    result = ChatTemplatePresenceCheck().run(_ctx(files))
    assert result.status == "pass"


def test_special_tokens_detects_typo_in_list_form_template() -> None:
    # template (list form) emits <|im_end|> but registered EOS has the typo <|im_end>
    files = {
        "tokenizer_config.json": json.dumps(
            {
                "chat_template": [{"name": "default", "template": "{{ m }}<|im_end|>"}],
                "eos_token": "<|im_end>",
                "added_tokens_decoder": {"0": {"content": "<|im_end>"}},
            }
        ).encode()
    }
    result = ChatTemplateSpecialTokensCheck().run(_ctx(files))
    assert result.status == "warn"
    assert list(result.details["unregistered"]) == ["<|im_end|>"]


def test_special_tokens_no_false_warn_for_jinja_variable_only_template() -> None:
    files = {
        "tokenizer_config.json": _tokenizer_config("<|im_end|>", ["<|im_end|>"]),
        "chat_template.jinja": b"{{ bos_token }}{{ messages }}{{ eos_token }}",
    }
    result = ChatTemplateSpecialTokensCheck().run(_ctx(files))
    assert result.status == "pass"


def test_presence_warns_medium_when_tokenizer_config_unparseable_with_empty_jinja() -> None:
    files = {"tokenizer_config.json": b"{not-json", "chat_template.jinja": b"   "}
    result = ChatTemplatePresenceCheck().run(_ctx(files))
    assert result.status == "warn"
    assert result.severity == "medium"
    assert "parse" in result.message.lower()
