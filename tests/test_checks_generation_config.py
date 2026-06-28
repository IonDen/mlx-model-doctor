import json

from mlx_model_doctor.checks.generation_config import GenerationConfigTokensCheck
from tests.fakes import context_for_files


def _ctx(files: dict[str, object]):
    return context_for_files({k: json.dumps(v).encode() for k, v in files.items()})


def test_warns_when_generation_config_has_eos_without_pad() -> None:
    files = {"config.json": {"model_type": "llama"}, "generation_config.json": {"eos_token_id": 2}}
    result = GenerationConfigTokensCheck().run(_ctx(files))
    assert result.status == "warn"
    assert result.severity == "low"  # eos-without-pad is informational, not policy-failing
    assert "pad" in result.message.lower()


def test_warns_on_cross_file_eos_disagreement() -> None:
    files = {
        "config.json": {"eos_token_id": 2, "pad_token_id": 0},
        "generation_config.json": {"eos_token_id": 7, "pad_token_id": 0},
    }
    result = GenerationConfigTokensCheck().run(_ctx(files))
    assert result.status == "warn"
    assert result.severity == "medium"  # a genuine cross-file disagreement
    assert "eos" in result.message.lower()


def test_scalar_vs_list_eos_is_not_a_disagreement() -> None:
    files = {
        "config.json": {"eos_token_id": 2, "pad_token_id": 0},
        "generation_config.json": {"eos_token_id": [2], "pad_token_id": 0},
    }
    result = GenerationConfigTokensCheck().run(_ctx(files))
    assert result.status == "pass"


def test_passes_when_consistent() -> None:
    files = {
        "config.json": {"eos_token_id": 2, "pad_token_id": 0},
        "generation_config.json": {"eos_token_id": 2, "pad_token_id": 0},
    }
    result = GenerationConfigTokensCheck().run(_ctx(files))
    assert result.status == "pass"


def test_skips_when_no_token_ids_anywhere() -> None:
    result = GenerationConfigTokensCheck().run(_ctx({"config.json": {"model_type": "llama"}}))
    assert result.status == "skip"


def test_warns_when_generation_config_present_but_unparseable() -> None:
    ctx = context_for_files(
        {"config.json": b'{"model_type": "llama"}', "generation_config.json": b"{not-json"}
    )
    result = GenerationConfigTokensCheck().run(ctx)
    assert result.status == "warn"
    assert result.severity == "medium"
    assert "parse" in result.message.lower() or "read" in result.message.lower()


def test_warns_on_cross_file_bos_disagreement() -> None:
    files = {
        "config.json": {"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0},
        "generation_config.json": {"bos_token_id": 9, "eos_token_id": 2, "pad_token_id": 0},
    }
    result = GenerationConfigTokensCheck().run(_ctx(files))
    assert result.status == "warn"
    assert result.severity == "medium"  # a genuine cross-file disagreement
    assert "bos" in result.message.lower()


def test_bool_token_id_is_treated_as_absent() -> None:
    # eos_token_id=True must NOT count as token id 1 (no eos-without-pad warn)
    files = {
        "config.json": {"model_type": "llama"},
        "generation_config.json": {"eos_token_id": True},
    }
    result = GenerationConfigTokensCheck().run(_ctx(files))
    assert result.status == "skip"
