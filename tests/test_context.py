import json
from collections.abc import Sequence

import pytest

from mlx_model_doctor.context import _MAX_METADATA_BYTES, CheckContext, CheckOptions
from mlx_model_doctor.errors import TargetError
from mlx_model_doctor.safetensors_header import SafetensorsHeader
from tests.fakes import FakeTarget, check_options, context_for_files


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

    def size(self, path: str) -> int | None:
        # Known (not None) so the L7 unknown-size guard doesn't short-circuit
        # before the read this fixture exists to exercise.
        return 0

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


# ---------------------------------------------------------------------------
# safetensors_header accessor tests
# ---------------------------------------------------------------------------


def test_context_caches_safetensors_header_one_call() -> None:
    header = SafetensorsHeader(files=(), weight_map={}, sharded=False, stored_count_by_dtype={})

    class CountingHeaderTarget(FakeTarget):
        calls: int = 0

        def safetensors_header(self) -> SafetensorsHeader | None:
            CountingHeaderTarget.calls += 1
            return header

    ctx = CheckContext(target=CountingHeaderTarget(files={}), options=check_options())
    assert ctx.safetensors_header() is header
    assert ctx.safetensors_header() is header
    assert CountingHeaderTarget.calls == 1


def test_context_routes_hf_header_target_error() -> None:
    from mlx_model_doctor.errors import TargetError

    class HfErrorTarget(FakeTarget):
        def safetensors_header(self) -> SafetensorsHeader | None:
            raise TargetError("boom", target="org/repo", source="hf")

    ctx = CheckContext(target=HfErrorTarget(files={}, _source="hf"), options=check_options())
    with pytest.raises(TargetError) as exc_info:
        ctx.safetensors_header()
    # The re-raise must preserve source="hf" so a Hub/network failure is never
    # mistaken for a clean "file absent" (which degrades to None on the local path).
    assert exc_info.value.source == "hf"


def test_context_exposes_local_header_parse_error() -> None:
    from mlx_model_doctor.safetensors_header import SafetensorsHeaderError

    class BadHeaderTarget(FakeTarget):
        def safetensors_header(self) -> SafetensorsHeader | None:
            raise SafetensorsHeaderError("model.safetensors: corrupt")

    ctx = CheckContext(target=BadHeaderTarget(files={}), options=check_options())
    assert ctx.safetensors_header() is None
    assert ctx.safetensors_header_error() == "model.safetensors: corrupt"


def test_context_exposes_local_header_target_error_message() -> None:
    from mlx_model_doctor.errors import TargetError

    class LocalIoErrorTarget(FakeTarget):
        def safetensors_header(self) -> SafetensorsHeader | None:
            raise TargetError("disk read failed", target="/m", source="local")

    ctx = CheckContext(target=LocalIoErrorTarget(files={}), options=check_options())
    assert ctx.safetensors_header() is None
    assert ctx.safetensors_header_error() == "disk read failed"


def test_context_no_header_error_when_absent() -> None:
    # No safetensors present at all -> header None, but NO error (it's a skip, not a fail).
    ctx = CheckContext(target=FakeTarget(files={}), options=check_options())
    assert ctx.safetensors_header() is None
    assert ctx.safetensors_header_error() is None


# ---------------------------------------------------------------------------
# preprocessor_config_json accessor tests (Task 4)
# ---------------------------------------------------------------------------


def test_preprocessor_config_json_parses() -> None:
    ctx = context_for_files({"preprocessor_config.json": b'{"image_processor_type": "X"}'})
    assert ctx.preprocessor_config_json() == {"image_processor_type": "X"}


def test_preprocessor_config_json_absent_returns_none() -> None:
    assert context_for_files({}).preprocessor_config_json() is None


def test_preprocessor_config_json_malformed_returns_none() -> None:
    ctx = context_for_files({"preprocessor_config.json": b"{not json"})
    assert ctx.preprocessor_config_json() is None


def test_preprocessor_config_json_oversized_returns_none() -> None:
    blob = b'{"image_processor_type": "X"}' + b" " * (_MAX_METADATA_BYTES + 1)
    ctx = context_for_files({"preprocessor_config.json": blob})
    assert ctx.preprocessor_config_json() is None


def test_preprocessor_config_json_propagates_hf_target_error() -> None:
    ctx = CheckContext(
        target=TargetErrorTarget(files={}, _source="hf"),
        options=check_options(),
    )

    with pytest.raises(TargetError, match="read failed") as exc_info:
        ctx.preprocessor_config_json()

    assert exc_info.value.source == "hf"


def test_preprocessor_config_json_swallows_local_target_error() -> None:
    ctx = CheckContext(
        target=TargetErrorTarget(files={}, _source="local"),
        options=check_options(),
    )
    assert ctx.preprocessor_config_json() is None


# ---------------------------------------------------------------------------
# Unknown file size guard (L7)
# ---------------------------------------------------------------------------


class UnknownSizeRecordingTarget:
    def __init__(self) -> None:
        self.read_attempted = False

    @property
    def source(self) -> str:
        return "hf"

    def exists(self, path: str) -> bool:
        return path == "config.json"

    def list_files(self) -> Sequence[str]:
        return ("config.json",)

    def size(self, path: str) -> int | None:
        return None  # sibling metadata lacks a size

    def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        self.read_attempted = True
        return b"{}"

    def read_text(self, path: str, *, max_bytes: int | None = None) -> str:
        self.read_attempted = True
        return "{}"

    def safetensors_header(self) -> SafetensorsHeader | None:
        return None


def test_unknown_size_metadata_is_not_read() -> None:
    target = UnknownSizeRecordingTarget()
    ctx = CheckContext(target=target, options=check_options())

    assert ctx.config_json() is None
    assert target.read_attempted is False
