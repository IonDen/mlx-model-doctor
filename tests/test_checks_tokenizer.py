from mlx_model_doctor.checks.tokenizer import SpecialTokensCheck, TokenizerFilesCheck
from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.errors import TargetError
from tests.fakes import FakeTarget, check_options, context_for_files


def test_tokenizer_files_check_warns_when_no_tokenizer_artifacts() -> None:
    result = TokenizerFilesCheck().run(context_for_files({"config.json": b"{}"}))

    assert result.check_id == "text/tokenizer.files"
    assert result.status == "warn"
    assert result.severity == "medium"
    assert "tokenizer" in result.message
    assert result.remediation is not None


def test_tokenizer_files_check_passes_when_tokenizer_json_exists() -> None:
    result = TokenizerFilesCheck().run(
        context_for_files({"config.json": b"{}", "tokenizer.json": b"{}"})
    )

    assert result.status == "pass"
    assert result.severity == "info"
    assert result.details["present"] == ("tokenizer.json",)


def test_tokenizer_files_check_warns_cleanly_on_target_error() -> None:
    result = TokenizerFilesCheck().run(
        CheckContext(target=ExistsErrorTarget(files={}), options=check_options())
    )

    assert result.status == "warn"
    assert result.severity == "medium"
    assert "Could not inspect" in result.message
    assert result.remediation is not None


def test_special_tokens_check_warns_when_pad_equals_eos() -> None:
    result = SpecialTokensCheck().run(
        context_for_files({"config.json": b'{"pad_token_id":2,"eos_token_id":2}'})
    )

    assert result.check_id == "text/tokenizer.special_tokens"
    assert result.status == "warn"
    assert result.severity == "medium"
    assert "pad_token_id" in result.message
    assert "eos_token_id" in result.message
    assert result.remediation is not None


def test_special_tokens_check_warns_for_non_integer_ids() -> None:
    for config in (
        b'{"pad_token_id":"2","eos_token_id":2}',
        b'{"pad_token_id":true,"eos_token_id":2}',
    ):
        result = SpecialTokensCheck().run(context_for_files({"config.json": config}))

        assert result.status == "warn"
        assert result.severity == "medium"
        assert "token" in result.message.lower()


def test_special_tokens_check_passes_when_ids_differ() -> None:
    result = SpecialTokensCheck().run(
        context_for_files({"config.json": b'{"pad_token_id":0,"eos_token_id":2}'})
    )

    assert result.status == "pass"
    assert result.severity == "info"
    assert result.details == {"pad_token_id": 0, "eos_token_id": 2}


def test_special_tokens_check_skips_when_config_unavailable() -> None:
    for files in ({}, {"config.json": b"{not-json"}, {"config.json": b"null"}):
        result = SpecialTokensCheck().run(context_for_files(files))

        assert result.status == "skip"
        assert result.severity == "info"
        assert "config" in result.message
        assert "unavailable" in result.message


class ExistsErrorTarget(FakeTarget):
    def exists(self, path: str) -> bool:
        raise TargetError("exists failed", target=path, source=self.source)
