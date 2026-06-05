import json

from mlx_model_doctor.checks.chat_template import ChatTemplatePresenceCheck
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
