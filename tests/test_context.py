import json

import pytest

from mlx_model_doctor.context import _MAX_METADATA_BYTES, CheckContext, CheckOptions
from mlx_model_doctor.errors import TargetError
from tests.fakes import FakeTarget, context_for_files


def options() -> CheckOptions:
    return CheckOptions(
        max_memory_bytes=None,
        context_length=4096,
        include_weights=False,
        smoke=False,
        verbosity="normal",
    )


def test_context_parses_config_json_once() -> None:
    target = FakeTarget(files={"config.json": b'{"model_type":"llama"}'})
    ctx = CheckContext(target=target, options=options())

    assert ctx.config_json() == {"model_type": "llama"}
    assert ctx.config_json() is ctx.config_json()


def test_context_returns_none_for_missing_invalid_or_non_object_config() -> None:
    assert CheckContext(target=FakeTarget(files={}), options=options()).config_json() is None
    assert (
        CheckContext(
            target=FakeTarget(files={"config.json": b"not json"}),
            options=options(),
        ).config_json()
        is None
    )
    assert (
        CheckContext(
            target=FakeTarget(files={"config.json": b"null"}),
            options=options(),
        ).config_json()
        is None
    )


def test_context_returns_none_for_invalid_utf8_or_target_errors() -> None:
    assert (
        CheckContext(
            target=FakeTarget(files={"config.json": b"\xff"}),
            options=options(),
        ).config_json()
        is None
    )
    assert CheckContext(target=TargetErrorTarget(files={}), options=options()).config_json() is None


def test_context_propagates_hf_target_errors() -> None:
    ctx = CheckContext(
        target=TargetErrorTarget(files={}, _source="hf"),
        options=options(),
    )

    with pytest.raises(TargetError, match="read failed") as exc_info:
        ctx.config_json()

    assert exc_info.value.source == "hf"


def test_context_does_not_hide_unexpected_target_bugs() -> None:
    ctx = CheckContext(target=KeyErrorTarget(files={"config.json": b"{}"}), options=options())

    with pytest.raises(KeyError, match="adapter bug") as exc_info:
        ctx.config_json()

    assert exc_info.value.args == ("adapter bug",)


def test_context_caches_none_config_result() -> None:
    target = CountingTarget(files={"config.json": b"null"})
    ctx = CheckContext(target=target, options=options())

    assert ctx.config_json() is None
    assert ctx.config_json() is None
    assert target.reads == 1


class CountingTarget(FakeTarget):
    __slots__ = ("reads",)

    def __init__(self, files: dict[str, bytes]) -> None:
        super().__init__(files=files)
        self.reads = 0

    def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        self.reads += 1
        return super().read_bytes(path, max_bytes=max_bytes)


class TargetErrorTarget(FakeTarget):
    def exists(self, path: str) -> bool:
        return True

    def read_text(self, path: str, *, max_bytes: int | None = None) -> str:
        raise TargetError("read failed", target=path, source=self.source)


class KeyErrorTarget(FakeTarget):
    def exists(self, path: str) -> bool:
        return True

    def read_text(self, path: str, *, max_bytes: int | None = None) -> str:
        raise KeyError("adapter bug")


# ---------------------------------------------------------------------------
# New accessor tests (Task 1)
# ---------------------------------------------------------------------------


def test_tokenizer_config_json_parses_present_file() -> None:
    ctx = context_for_files(
        {"tokenizer_config.json": json.dumps({"eos_token": "<|im_end|>"}).encode()}
    )
    assert ctx.tokenizer_config_json() == {"eos_token": "<|im_end|>"}


def test_metadata_accessors_return_none_when_absent() -> None:
    ctx = context_for_files({})
    assert ctx.tokenizer_config_json() is None
    assert ctx.special_tokens_map_json() is None
    assert ctx.generation_config_json() is None
    assert ctx.chat_template_text() is None


def test_metadata_accessors_return_none_for_malformed_json() -> None:
    ctx = context_for_files({"generation_config.json": b"{not-json"})
    assert ctx.generation_config_json() is None
    assert ctx.target.exists("generation_config.json") is True


def test_chat_template_text_reads_jinja_sibling() -> None:
    ctx = context_for_files({"chat_template.jinja": b"{{ bos_token }}<|im_end|>"})
    assert ctx.chat_template_text() == "{{ bos_token }}<|im_end|>"


def test_json_accessor_refuses_oversized_file_without_parsing() -> None:
    oversized = b"{}" + b" " * (_MAX_METADATA_BYTES + 1)
    ctx = context_for_files({"tokenizer_config.json": oversized})
    assert ctx.tokenizer_config_json() is None
    assert ctx.target.exists("tokenizer_config.json") is True


def test_config_json_still_parses_and_is_cached() -> None:
    ctx = context_for_files({"config.json": json.dumps({"model_type": "llama"}).encode()})
    assert ctx.config_json() == {"model_type": "llama"}
    assert ctx.config_json() is ctx.config_json()
