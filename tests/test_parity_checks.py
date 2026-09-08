"""Tests for the tokenizer-identity fingerprint (F6).

This is the "tokenizer part" of ``test_parity_checks.py`` named by the task
plan: pure tests of :mod:`mlx_model_doctor.parity.context`'s
``tokenizer_fingerprint``/``tokenizers_match``. A later task adds
``ParityCheck``/``ParityContext`` tests to this same file.
"""

from mlx_model_doctor.parity.context import (
    MatchResult,
    tokenizer_fingerprint,
    tokenizers_match,
)

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
