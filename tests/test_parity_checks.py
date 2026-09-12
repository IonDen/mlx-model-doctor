"""Tests for the tokenizer-identity fingerprint (F6) and the parity checks.

The first part of this file is the "tokenizer part" named by the task plan:
pure tests of :mod:`mlx_model_doctor.parity.context`'s
``tokenizer_fingerprint``/``tokenizers_match``. The rest exercises
``ParityContext``/``ParityCheck``/``run_parity_checks`` and the four
cross-target static checks in :mod:`mlx_model_doctor.parity.checks` (F3/F4/
F5/F6), built entirely on in-memory ``FakeTarget`` repositories so it stays
offline and MLX/numpy-free.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from mlx_model_doctor.errors import ModelDoctorError, TargetError
from mlx_model_doctor.parity.checks import (
    AdapterConfigCheck,
    FusedTargetConsistencyCheck,
    TargetCoverageCheck,
    TokenizerIdentityCheck,
    run_parity_checks,
)
from mlx_model_doctor.parity.context import (
    MatchResult,
    ParityContext,
    ParityTargets,
    tokenizer_fingerprint,
    tokenizers_match,
)
from mlx_model_doctor.report import CheckResult
from mlx_model_doctor.safetensors_header import FileHeader, SafetensorsHeader, TensorEntry
from tests.fakes import FakeTarget, check_options

_BASE_TOKENIZER_CONFIG: dict[str, object] = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
}

_BASE_VOCAB: dict[str, int] = {"<s>": 0, "</s>": 1, "<pad>": 2, "hello": 3, "world": 4}


def _tokenizer_json(vocab: dict[str, int]) -> dict[str, object]:
    return {"model": {"type": "BPE", "vocab": vocab}}


class TestTokenizerFingerprintComponents:
    """Each fingerprint component is None exactly when its source is unusable."""

    def test_missing_tokenizer_json_leaves_vocab_and_map_none(self) -> None:
        fp = tokenizer_fingerprint(
            tokenizer_config=_BASE_TOKENIZER_CONFIG,
            tokenizer_json=None,
            chat_template="a template",
        )
        assert fp.vocab_size is None
        assert fp.token_id_map_digest is None
        assert fp.special_tokens_digest is not None
        assert fp.chat_template_digest is not None

    def test_missing_tokenizer_config_leaves_special_tokens_none(self) -> None:
        fp = tokenizer_fingerprint(
            tokenizer_config=None,
            tokenizer_json=_tokenizer_json(_BASE_VOCAB),
            chat_template=None,
        )
        assert fp.special_tokens_digest is None
        assert fp.vocab_size == len(_BASE_VOCAB)
        assert fp.token_id_map_digest is not None
        assert fp.chat_template_digest is None

    def test_vocab_size_matches_vocab_length(self) -> None:
        fp = tokenizer_fingerprint(
            tokenizer_config=None,
            tokenizer_json=_tokenizer_json(_BASE_VOCAB),
            chat_template=None,
        )
        assert fp.vocab_size == 5

    def test_malformed_vocab_value_is_unavailable(self) -> None:
        # A vocab entry whose id isn't an int can't be hashed as an id map.
        malformed = {"model": {"vocab": {"a": "not-an-int"}}}
        fp = tokenizer_fingerprint(
            tokenizer_config=None, tokenizer_json=malformed, chat_template=None
        )
        assert fp.vocab_size is None
        assert fp.token_id_map_digest is None

    def test_bool_token_id_is_treated_as_malformed(self) -> None:
        # bool is a subclass of int in Python; `isinstance(x, int)` would
        # wrongly accept it as a valid token id.
        malformed = {"model": {"vocab": {"a": True}}}
        fp = tokenizer_fingerprint(
            tokenizer_config=None, tokenizer_json=malformed, chat_template=None
        )
        assert fp.vocab_size is None
        assert fp.token_id_map_digest is None

    def test_non_mapping_model_is_unavailable(self) -> None:
        fp = tokenizer_fingerprint(
            tokenizer_config=None, tokenizer_json={"model": "not-a-mapping"}, chat_template=None
        )
        assert fp.vocab_size is None
        assert fp.token_id_map_digest is None

    def test_empty_vocab_is_unavailable(self) -> None:
        fp = tokenizer_fingerprint(
            tokenizer_config=None, tokenizer_json=_tokenizer_json({}), chat_template=None
        )
        assert fp.vocab_size is None
        assert fp.token_id_map_digest is None

    def test_fully_populated_inputs_produce_every_digest(self) -> None:
        fp = tokenizer_fingerprint(
            tokenizer_config=_BASE_TOKENIZER_CONFIG,
            tokenizer_json=_tokenizer_json(_BASE_VOCAB),
            chat_template="a template",
        )
        assert fp.vocab_size == len(_BASE_VOCAB)
        assert fp.special_tokens_digest is not None
        assert fp.token_id_map_digest is not None
        assert fp.chat_template_digest is not None


class TestTokenizersMatch:
    """tokenizers_match reports which aspect differs, in a fixed order."""

    def test_identical_fingerprints_match(self) -> None:
        fp_a = tokenizer_fingerprint(
            tokenizer_config=_BASE_TOKENIZER_CONFIG,
            tokenizer_json=_tokenizer_json(_BASE_VOCAB),
            chat_template="template",
        )
        fp_b = tokenizer_fingerprint(
            tokenizer_config=dict(_BASE_TOKENIZER_CONFIG),
            tokenizer_json=_tokenizer_json(dict(_BASE_VOCAB)),
            chat_template="template",
        )
        assert tokenizers_match(fp_a, fp_b) == MatchResult(matched=True, reason=None)

    def test_same_size_vocab_permutation_is_a_mismatch(self) -> None:
        # F6: a two-token ID swap preserves vocab size and special tokens --
        # the fingerprint must still catch it via the ordinary token-id map.
        permuted_vocab = dict(_BASE_VOCAB)
        permuted_vocab["hello"], permuted_vocab["world"] = (
            permuted_vocab["world"],
            permuted_vocab["hello"],
        )
        fp_a = tokenizer_fingerprint(
            tokenizer_config=_BASE_TOKENIZER_CONFIG,
            tokenizer_json=_tokenizer_json(_BASE_VOCAB),
            chat_template="template",
        )
        fp_b = tokenizer_fingerprint(
            tokenizer_config=_BASE_TOKENIZER_CONFIG,
            tokenizer_json=_tokenizer_json(permuted_vocab),
            chat_template="template",
        )
        assert fp_a.vocab_size == fp_b.vocab_size
        assert fp_a.special_tokens_digest == fp_b.special_tokens_digest
        assert tokenizers_match(fp_a, fp_b) == MatchResult(
            matched=False, reason="token_id_map_differs"
        )

    def test_changed_chat_template_is_a_mismatch(self) -> None:
        fp_a = tokenizer_fingerprint(
            tokenizer_config=_BASE_TOKENIZER_CONFIG,
            tokenizer_json=_tokenizer_json(_BASE_VOCAB),
            chat_template="template one",
        )
        fp_b = tokenizer_fingerprint(
            tokenizer_config=_BASE_TOKENIZER_CONFIG,
            tokenizer_json=_tokenizer_json(_BASE_VOCAB),
            chat_template="template two",
        )
        assert tokenizers_match(fp_a, fp_b) == MatchResult(
            matched=False, reason="chat_template_differs"
        )

    def test_vocab_size_mismatch_is_reported_first(self) -> None:
        fp_a = tokenizer_fingerprint(
            tokenizer_config=_BASE_TOKENIZER_CONFIG,
            tokenizer_json=_tokenizer_json(_BASE_VOCAB),
            chat_template="template",
        )
        smaller_vocab = {k: v for k, v in _BASE_VOCAB.items() if k != "world"}
        fp_b = tokenizer_fingerprint(
            tokenizer_config=_BASE_TOKENIZER_CONFIG,
            tokenizer_json=_tokenizer_json(smaller_vocab),
            chat_template="template",
        )
        assert tokenizers_match(fp_a, fp_b) == MatchResult(
            matched=False, reason="vocab_size_differs"
        )

    def test_special_tokens_mismatch_is_reported(self) -> None:
        fp_a = tokenizer_fingerprint(
            tokenizer_config=_BASE_TOKENIZER_CONFIG,
            tokenizer_json=_tokenizer_json(_BASE_VOCAB),
            chat_template="template",
        )
        other_config = dict(_BASE_TOKENIZER_CONFIG)
        other_config["eos_token"] = "<different-eos>"
        fp_b = tokenizer_fingerprint(
            tokenizer_config=other_config,
            tokenizer_json=_tokenizer_json(_BASE_VOCAB),
            chat_template="template",
        )
        assert tokenizers_match(fp_a, fp_b) == MatchResult(
            matched=False, reason="special_tokens_differ"
        )


class TestUnavailableRepresentationNeverFalseMatches:
    """An uncomparable aspect must report a distinct reason, never a match."""

    def test_both_sides_missing_chat_template_is_not_a_false_match(self) -> None:
        # A naive `None == None` compare would call two untrackable
        # templates identical; it must instead report "unavailable".
        fp_a = tokenizer_fingerprint(
            tokenizer_config=_BASE_TOKENIZER_CONFIG,
            tokenizer_json=_tokenizer_json(_BASE_VOCAB),
            chat_template=None,
        )
        fp_b = tokenizer_fingerprint(
            tokenizer_config=_BASE_TOKENIZER_CONFIG,
            tokenizer_json=_tokenizer_json(_BASE_VOCAB),
            chat_template=None,
        )
        result = tokenizers_match(fp_a, fp_b)
        assert result.matched is False
        assert result.reason == "chat_template_unavailable"

    def test_one_sided_missing_chat_template_is_unavailable(self) -> None:
        fp_a = tokenizer_fingerprint(
            tokenizer_config=_BASE_TOKENIZER_CONFIG,
            tokenizer_json=_tokenizer_json(_BASE_VOCAB),
            chat_template="template",
        )
        fp_b = tokenizer_fingerprint(
            tokenizer_config=_BASE_TOKENIZER_CONFIG,
            tokenizer_json=_tokenizer_json(_BASE_VOCAB),
            chat_template=None,
        )
        assert tokenizers_match(fp_a, fp_b) == MatchResult(
            matched=False, reason="chat_template_unavailable"
        )

    def test_missing_vocab_mapping_on_both_sides_is_not_a_false_match(self) -> None:
        fp_a = tokenizer_fingerprint(
            tokenizer_config=_BASE_TOKENIZER_CONFIG, tokenizer_json=None, chat_template="template"
        )
        fp_b = tokenizer_fingerprint(
            tokenizer_config=_BASE_TOKENIZER_CONFIG, tokenizer_json=None, chat_template="template"
        )
        result = tokenizers_match(fp_a, fp_b)
        assert result.matched is False
        assert result.reason == "vocab_unavailable"

    def test_one_sided_missing_vocab_mapping_is_unavailable(self) -> None:
        fp_a = tokenizer_fingerprint(
            tokenizer_config=_BASE_TOKENIZER_CONFIG,
            tokenizer_json=_tokenizer_json(_BASE_VOCAB),
            chat_template="template",
        )
        fp_b = tokenizer_fingerprint(
            tokenizer_config=_BASE_TOKENIZER_CONFIG, tokenizer_json=None, chat_template="template"
        )
        assert tokenizers_match(fp_a, fp_b) == MatchResult(
            matched=False, reason="vocab_unavailable"
        )

    def test_missing_special_tokens_on_both_sides_is_not_a_false_match(self) -> None:
        fp_a = tokenizer_fingerprint(
            tokenizer_config=None,
            tokenizer_json=_tokenizer_json(_BASE_VOCAB),
            chat_template="template",
        )
        fp_b = tokenizer_fingerprint(
            tokenizer_config=None,
            tokenizer_json=_tokenizer_json(_BASE_VOCAB),
            chat_template="template",
        )
        result = tokenizers_match(fp_a, fp_b)
        assert result.matched is False
        assert result.reason == "special_tokens_unavailable"


# --- ParityContext / ParityCheck / run_parity_checks (F3/F4/F5/F6) ----------

_BLOCK_LINEAR_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

_PARITY_VOCAB: dict[str, int] = {"<s>": 0, "</s>": 1, "hello": 2, "world": 3}


def _base_tensor_names(num_blocks: int = 2) -> tuple[str, ...]:
    """Build a tiny 2-block llama-family base model's safetensors tensor names."""
    names = ["model.embed_tokens.weight"]
    for i in range(num_blocks):
        names.extend(f"model.layers.{i}.{suffix}.weight" for suffix in _BLOCK_LINEAR_SUFFIXES)
    names.append("model.norm.weight")
    names.append("lm_head.weight")
    return tuple(names)


def _entry() -> TensorEntry:
    return TensorEntry(dtype="F32", shape=(1,), data_offsets=(0, 4), stored_element_count=1)


def _header(names: Sequence[str]) -> SafetensorsHeader:
    tensors = {name: _entry() for name in names}
    file_header = FileHeader(
        filename="model.safetensors",
        tensors=tensors,
        metadata={},
        header_length=10,
        file_size=None,
    )
    return SafetensorsHeader(
        files=(file_header,),
        weight_map={},
        sharded=False,
        stored_count_by_dtype={"F32": len(tensors)},
    )


def _header_from_entries(entries: dict[str, TensorEntry]) -> SafetensorsHeader:
    """Build a SafetensorsHeader from explicit per-tensor entries (dtype/shape control)."""
    file_header = FileHeader(
        filename="model.safetensors",
        tensors=entries,
        metadata={},
        header_length=10,
        file_size=None,
    )
    return SafetensorsHeader(
        files=(file_header,), weight_map={}, sharded=False, stored_count_by_dtype={}
    )


def _quantized_target_entries(target: str) -> dict[str, TensorEntry]:
    """Build the header entries a quantized nn.QuantizedLinear target module has.

    Verified against installed mlx 0.32.0: nn.QuantizedLinear's parameters are
    ``{weight (uint32, packed), scales, biases, bias}`` -- a quantized module
    HAS a (packed) ``.weight`` tensor alongside ``.scales``/``.biases``, not
    ``.scales``/``.biases`` "instead of" one.
    """
    return {
        f"{target}.weight": TensorEntry(
            dtype="U32", shape=(4, 1), data_offsets=(0, 16), stored_element_count=4
        ),
        f"{target}.scales": TensorEntry(
            dtype="F16", shape=(4, 1), data_offsets=(16, 24), stored_element_count=4
        ),
        f"{target}.biases": TensorEntry(
            dtype="F16", shape=(4, 1), data_offsets=(24, 32), stored_element_count=4
        ),
    }


def _adapter_config_bytes(**overrides: object) -> bytes:
    config: dict[str, object] = {
        "num_layers": -1,
        "fine_tune_type": "lora",
        "lora_parameters": {"keys": ["self_attn.q_proj"]},
    }
    config.update(overrides)
    return json.dumps(config).encode("utf-8")


def _tokenizer_files(
    vocab: dict[str, int], *, chat_template: str = "a template"
) -> dict[str, bytes]:
    tokenizer_config = {"bos_token": "<s>", "eos_token": "</s>", "chat_template": chat_template}
    tokenizer_json = {"model": {"type": "BPE", "vocab": vocab}}
    return {
        "tokenizer_config.json": json.dumps(tokenizer_config).encode("utf-8"),
        "tokenizer.json": json.dumps(tokenizer_json).encode("utf-8"),
    }


def _base_target(
    *, model_type: str = "llama", header: SafetensorsHeader | None = None
) -> FakeTarget:
    return FakeTarget(
        files={"config.json": json.dumps({"model_type": model_type}).encode("utf-8")},
        name="base",
        _safetensors_header=header if header is not None else _header(_base_tensor_names()),
    )


def _adapter_target(config_bytes: bytes | None) -> FakeTarget:
    files = {} if config_bytes is None else {"adapter_config.json": config_bytes}
    return FakeTarget(files=files, name="adapter")


def _fused_target(
    *, header: SafetensorsHeader | None = None, tokenizer_files: dict[str, bytes] | None = None
) -> FakeTarget:
    return FakeTarget(files=dict(tokenizer_files or {}), name="fused", _safetensors_header=header)


def _pctx(*, base: FakeTarget, adapter: FakeTarget, fused: FakeTarget) -> ParityContext:
    return ParityContext(
        targets=ParityTargets(base=base, adapter=adapter, fused=fused),
        options=check_options(),
    )


class _TargetErrorTarget(FakeTarget):
    """FakeTarget whose reads always raise TargetError (source configurable)."""

    def exists(self, path: str) -> bool:
        return True

    def size(self, path: str) -> int | None:
        return 0

    def read_text(self, path: str, *, max_bytes: int | None = None) -> str:
        raise TargetError("read failed", target=path, source=self.source)


class TestParityContextAdapterConfigParsing:
    """ParityContext parses adapter_config.json once and exposes its fields."""

    def test_convenience_properties_reflect_the_parsed_config(self) -> None:
        pctx = _pctx(
            base=_base_target(),
            adapter=_adapter_target(_adapter_config_bytes()),
            fused=_fused_target(),
        )
        assert pctx.num_layers == -1
        assert pctx.fine_tune_type == "lora"
        assert pctx.lora_parameter_keys == ("self_attn.q_proj",)

    def test_convenience_properties_are_none_without_a_config(self) -> None:
        pctx = _pctx(base=_base_target(), adapter=_adapter_target(None), fused=_fused_target())
        assert pctx.num_layers is None
        assert pctx.fine_tune_type is None
        assert pctx.lora_parameter_keys is None

    def test_non_json_adapter_config_is_treated_as_malformed(self) -> None:
        pctx = _pctx(
            base=_base_target(), adapter=_adapter_target(b"not-json-at-all"), fused=_fused_target()
        )
        assert pctx.adapter_config is None

    def test_base_model_type_is_none_without_a_base_config(self) -> None:
        pctx = _pctx(
            base=FakeTarget(files={}, name="base"),
            adapter=_adapter_target(_adapter_config_bytes()),
            fused=_fused_target(),
        )
        assert pctx.base_model_type() is None

    def test_hf_adapter_config_read_error_propagates(self) -> None:
        with pytest.raises(TargetError):
            _pctx(
                base=_base_target(),
                adapter=_TargetErrorTarget(files={}, name="adapter", _source="hf"),
                fused=_fused_target(),
            )

    def test_local_adapter_config_read_error_is_swallowed(self) -> None:
        pctx = _pctx(
            base=_base_target(),
            adapter=_TargetErrorTarget(files={}, name="adapter", _source="local"),
            fused=_fused_target(),
        )
        assert pctx.adapter_config is None


class TestAdapterConfigCheck:
    """AdapterConfigCheck validates presence and shape of adapter_config.json."""

    def test_well_formed_config_passes(self) -> None:
        pctx = _pctx(
            base=_base_target(),
            adapter=_adapter_target(_adapter_config_bytes()),
            fused=_fused_target(),
        )
        result = AdapterConfigCheck().run(pctx)
        assert result.status == "pass"

    def test_missing_adapter_config_fails(self) -> None:
        pctx = _pctx(base=_base_target(), adapter=_adapter_target(None), fused=_fused_target())
        result = AdapterConfigCheck().run(pctx)
        assert result.status == "fail"
        assert "Missing" in result.message

    def test_malformed_num_layers_fails_and_names_the_field(self) -> None:
        pctx = _pctx(
            base=_base_target(),
            adapter=_adapter_target(_adapter_config_bytes(num_layers="oops")),
            fused=_fused_target(),
        )
        result = AdapterConfigCheck().run(pctx)
        assert result.status == "fail"
        assert "num_layers" in result.message
        problems = result.details["problems"]
        assert any("num_layers" in problem for problem in problems)

    def test_malformed_lora_keys_fails_and_names_the_field(self) -> None:
        pctx = _pctx(
            base=_base_target(),
            adapter=_adapter_target(_adapter_config_bytes(lora_parameters={"keys": "not-a-list"})),
            fused=_fused_target(),
        )
        result = AdapterConfigCheck().run(pctx)
        assert result.status == "fail"
        assert "lora_parameters.keys" in result.message


class TestTargetCoverageCheck:
    """TargetCoverageCheck resolves LoRA target keys against the base model's tensors."""

    def test_uncovered_key_fails_and_names_it(self) -> None:
        pctx = _pctx(
            base=_base_target(),
            adapter=_adapter_target(
                _adapter_config_bytes(lora_parameters={"keys": ["self_attn.typo_proj"]})
            ),
            fused=_fused_target(),
        )
        result = TargetCoverageCheck().run(pctx)
        assert result.status == "fail"
        assert "self_attn.typo_proj" in result.message
        assert result.details["uncovered_keys"] == ("self_attn.typo_proj",)

    def test_covered_targets_pass(self) -> None:
        pctx = _pctx(
            base=_base_target(),
            adapter=_adapter_target(_adapter_config_bytes()),
            fused=_fused_target(),
        )
        result = TargetCoverageCheck().run(pctx)
        assert result.status == "pass"

    def test_full_fine_tune_is_skipped_not_failed(self) -> None:
        pctx = _pctx(
            base=_base_target(),
            adapter=_adapter_target(_adapter_config_bytes(fine_tune_type="full")),
            fused=_fused_target(),
        )
        result = TargetCoverageCheck().run(pctx)
        assert result.status == "skip"

    def test_unverified_architecture_is_skipped_not_failed(self) -> None:
        pctx = _pctx(
            base=_base_target(model_type="made-up-arch"),
            adapter=_adapter_target(_adapter_config_bytes()),
            fused=_fused_target(),
        )
        result = TargetCoverageCheck().run(pctx)
        assert result.status == "skip"

    def test_missing_adapter_config_is_skipped_not_failed(self) -> None:
        pctx = _pctx(base=_base_target(), adapter=_adapter_target(None), fused=_fused_target())
        result = TargetCoverageCheck().run(pctx)
        assert result.status == "skip"


class TestFusedTargetConsistencyCheck:
    """FusedTargetConsistencyCheck checks the fused header against resolved targets."""

    def test_omitted_target_and_leftover_lora_factor_both_fail_and_are_named(self) -> None:
        fused_names = [
            name
            for name in _base_tensor_names()
            if name != "model.layers.0.self_attn.q_proj.weight"
        ]
        fused_names.append("model.layers.0.self_attn.o_proj.lora_a")
        fused_names.append("model.layers.0.self_attn.o_proj.lora_b")
        pctx = _pctx(
            base=_base_target(),
            adapter=_adapter_target(
                _adapter_config_bytes(
                    lora_parameters={"keys": ["self_attn.q_proj", "self_attn.o_proj"]}
                )
            ),
            fused=_fused_target(header=_header(fused_names)),
        )
        result = FusedTargetConsistencyCheck().run(pctx)
        assert result.status == "fail"
        assert "model.layers.0.self_attn.q_proj" in result.details["omitted_targets"]
        assert "model.layers.0.self_attn.o_proj.lora_a" in result.details["leftover_lora_factors"]

    def test_consistent_fused_header_passes(self) -> None:
        pctx = _pctx(
            base=_base_target(),
            adapter=_adapter_target(_adapter_config_bytes()),
            fused=_fused_target(header=_header(_base_tensor_names())),
        )
        result = FusedTargetConsistencyCheck().run(pctx)
        assert result.status == "pass"

    def test_missing_fused_header_is_skipped_not_failed(self) -> None:
        pctx = _pctx(
            base=_base_target(),
            adapter=_adapter_target(_adapter_config_bytes()),
            fused=_fused_target(),
        )
        result = FusedTargetConsistencyCheck().run(pctx)
        assert result.status == "skip"

    def test_quantized_target_present_in_both_headers_is_not_omitted(self) -> None:
        # A reviewer worried this check would false-positive on quantized
        # models by assuming a quantized target has .scales/.biases "instead
        # of" .weight. That premise is wrong (see _quantized_target_entries):
        # this proves a quantized target with its packed .weight present in
        # both headers is correctly NOT reported omitted.
        target = "model.layers.0.self_attn.q_proj"
        base_entries = {
            "model.embed_tokens.weight": _entry(),
            **_quantized_target_entries(target),
            "model.norm.weight": _entry(),
            "lm_head.weight": _entry(),
        }
        fused_entries = dict(base_entries)
        pctx = _pctx(
            base=_base_target(header=_header_from_entries(base_entries)),
            adapter=_adapter_target(_adapter_config_bytes()),
            fused=_fused_target(header=_header_from_entries(fused_entries)),
        )
        result = FusedTargetConsistencyCheck().run(pctx)
        assert result.status == "pass"
        assert "exposes all 1 targeted weight" in result.message


class TestTokenizerIdentityCheck:
    """TokenizerIdentityCheck compares base vs fused tokenizer fingerprints."""

    def test_matching_tokenizers_pass(self) -> None:
        pctx = _pctx(
            base=FakeTarget(files={**_tokenizer_files(_PARITY_VOCAB)}, name="base"),
            adapter=_adapter_target(_adapter_config_bytes()),
            fused=FakeTarget(files={**_tokenizer_files(_PARITY_VOCAB)}, name="fused"),
        )
        result = TokenizerIdentityCheck().run(pctx)
        assert result.status == "pass"

    def test_mismatched_tokenizers_fail_and_flag_the_oracle_void(self) -> None:
        permuted_vocab = {"<s>": 0, "</s>": 1, "hello": 3, "world": 2}
        pctx = _pctx(
            base=FakeTarget(files={**_tokenizer_files(_PARITY_VOCAB)}, name="base"),
            adapter=_adapter_target(_adapter_config_bytes()),
            fused=FakeTarget(files={**_tokenizer_files(permuted_vocab)}, name="fused"),
        )
        result = TokenizerIdentityCheck().run(pctx)
        assert result.status == "fail"
        assert result.details["void_oracle"] is True
        assert result.details["match_reason"] == "token_id_map_differs"

    def test_chat_template_difference_with_matching_vocab_warns_and_does_not_void_oracle(
        self,
    ) -> None:
        # F6 relaxation: the oracle feeds the SAME pre-computed token ids to both
        # models, so only the vocab (token<->id map) is load-bearing. A changed
        # chat template with an unchanged vocab must warn, not fail -- and must
        # NOT void the oracle.
        pctx = _pctx(
            base=FakeTarget(
                files={**_tokenizer_files(_PARITY_VOCAB, chat_template="{{ x }}")}, name="base"
            ),
            adapter=_adapter_target(_adapter_config_bytes()),
            fused=FakeTarget(
                files={**_tokenizer_files(_PARITY_VOCAB, chat_template="{{ y }}")}, name="fused"
            ),
        )
        result = TokenizerIdentityCheck().run(pctx)
        assert result.status == "warn"
        assert result.details["void_oracle"] is False
        assert result.details["match_reason"] == "chat_template_differs"

    def test_special_tokens_difference_with_matching_vocab_warns_and_does_not_void_oracle(
        self,
    ) -> None:
        # A real `mlx_lm.fuse` output re-serializes tokenizer_config.json (e.g.
        # base `additional_special_tokens` vs fused `extra_special_tokens`) --
        # metadata that never reaches the fixed-id forward pass. Vocab and chat
        # template are unchanged here; only the special-token fields differ.
        base_files = _tokenizer_files(_PARITY_VOCAB)
        fused_config = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "chat_template": "a template",
        }
        fused_files = {
            "tokenizer_config.json": json.dumps(fused_config).encode("utf-8"),
            "tokenizer.json": json.dumps({"model": {"type": "BPE", "vocab": _PARITY_VOCAB}}).encode(
                "utf-8"
            ),
        }
        pctx = _pctx(
            base=FakeTarget(files={**base_files}, name="base"),
            adapter=_adapter_target(_adapter_config_bytes()),
            fused=FakeTarget(files=fused_files, name="fused"),
        )
        result = TokenizerIdentityCheck().run(pctx)
        assert result.status == "warn"
        assert result.details["void_oracle"] is False
        assert result.details["match_reason"] == "special_tokens_differ"


@dataclass(frozen=True, slots=True)
class _CrashingParityCheck:
    check_id: str = "parity/boom"
    title: str = "Crashes"

    def run(self, pctx: ParityContext) -> CheckResult:
        raise RuntimeError("boom")


@dataclass(frozen=True, slots=True)
class _ToolErrorParityCheck:
    check_id: str = "parity/tool-error"
    title: str = "Tool-level failure"

    def run(self, pctx: ParityContext) -> CheckResult:
        raise ModelDoctorError("tool-level failure")


class TestRunParityChecks:
    """run_parity_checks isolates crashes and flags them distinctly from an ordinary fail."""

    def test_crashing_check_sets_crash_flag_and_is_isolated_as_fail(self) -> None:
        pctx = _pctx(
            base=_base_target(),
            adapter=_adapter_target(_adapter_config_bytes()),
            fused=_fused_target(),
        )
        results, crashed = run_parity_checks(pctx, [_CrashingParityCheck()])
        assert crashed is True
        assert len(results) == 1
        assert results[0].status == "fail"
        assert results[0].severity == "high"
        assert results[0].check_id == "parity/boom"
        assert "boom" in results[0].message

    def test_ordinary_fail_does_not_set_the_crash_flag(self) -> None:
        pctx = _pctx(
            base=_base_target(),
            adapter=_adapter_target(
                _adapter_config_bytes(lora_parameters={"keys": ["self_attn.typo_proj"]})
            ),
            fused=_fused_target(),
        )
        results, crashed = run_parity_checks(pctx, [TargetCoverageCheck()])
        assert crashed is False
        assert results[0].status == "fail"

    def test_model_doctor_error_propagates_instead_of_being_isolated(self) -> None:
        pctx = _pctx(
            base=_base_target(),
            adapter=_adapter_target(_adapter_config_bytes()),
            fused=_fused_target(),
        )
        with pytest.raises(ModelDoctorError):
            run_parity_checks(pctx, [_ToolErrorParityCheck()])

    def test_clean_repositories_pass_every_check(self) -> None:
        base = FakeTarget(
            files={
                "config.json": json.dumps({"model_type": "llama"}).encode("utf-8"),
                **_tokenizer_files(_PARITY_VOCAB),
            },
            name="base",
            _safetensors_header=_header(_base_tensor_names()),
        )
        fused = FakeTarget(
            files={**_tokenizer_files(_PARITY_VOCAB)},
            name="fused",
            _safetensors_header=_header(_base_tensor_names()),
        )
        pctx = _pctx(base=base, adapter=_adapter_target(_adapter_config_bytes()), fused=fused)

        results, crashed = run_parity_checks(
            pctx,
            [
                AdapterConfigCheck(),
                TargetCoverageCheck(),
                FusedTargetConsistencyCheck(),
                TokenizerIdentityCheck(),
            ],
        )

        assert crashed is False
        assert [r.status for r in results] == ["pass", "pass", "pass", "pass"]
