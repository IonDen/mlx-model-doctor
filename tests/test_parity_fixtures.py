"""Tests for pinned, tokenizer-bound parity-oracle fixtures (F6/F9)."""

import pytest

from mlx_model_doctor.parity.context import TokenizerFingerprint, tokenizer_fingerprint
from mlx_model_doctor.parity.fixtures import (
    DEFAULT_FIXTURE_ID,
    FixtureRef,
    UnknownFixtureError,
    build_fixture_from_prompts,
    compute_input_digest,
    get_fixture,
    reference_tokenizer_files,
    token_ids_in_range,
)


class _ScriptedChatTemplate:
    """Fake ``apply_chat_template`` returning pre-scripted token ids per call.

    Responses are keyed by ``(first_message_content, add_generation_prompt)`` --
    the prompt text is present in both the prompt-only and full calls for a given
    pair, so this distinguishes "the prompt-only render for pair N" from "the
    full render for pair N" without needing call-order bookkeeping.
    """

    def __init__(self, responses: dict[tuple[str, bool], list[int]]) -> None:
        self._responses = responses
        self.calls: list[tuple[list[dict[str, str]], bool]] = []

    def __call__(self, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
        self.calls.append((messages, add_generation_prompt))
        key = (messages[0]["content"], add_generation_prompt)
        return list(self._responses[key])


def _fingerprint() -> TokenizerFingerprint:
    return tokenizer_fingerprint(tokenizer_config=None, tokenizer_json=None, chat_template=None)


class TestGetFixture:
    """get_fixture(id) returns a pinned (FixtureRef, token_sequences) pair."""

    def test_default_fixture_returns_ref_and_nonempty_token_sequences(self) -> None:
        ref, token_sequences = get_fixture(DEFAULT_FIXTURE_ID)
        assert isinstance(ref, FixtureRef)
        assert ref.id == DEFAULT_FIXTURE_ID
        assert isinstance(token_sequences, tuple)
        assert len(token_sequences) > 0
        for sequence in token_sequences:
            assert isinstance(sequence, tuple)
            assert len(sequence) > 0
            assert all(type(token_id) is int for token_id in sequence)

    def test_default_fixture_is_a_single_sequence(self) -> None:
        """Pinned shape: today's default fixture is exactly one sequence."""
        _, token_sequences = get_fixture(DEFAULT_FIXTURE_ID)
        assert len(token_sequences) == 1

    def test_default_fixture_max_length_matches_total_flattened_length(self) -> None:
        ref, token_sequences = get_fixture(DEFAULT_FIXTURE_ID)
        total = sum(len(sequence) for sequence in token_sequences)
        assert ref.max_length == total

    def test_default_fixture_scored_positions_are_within_bounds(self) -> None:
        ref, token_sequences = get_fixture(DEFAULT_FIXTURE_ID)
        total = sum(len(sequence) for sequence in token_sequences)
        assert len(ref.scored_positions) > 0
        assert all(0 <= pos < total for pos in ref.scored_positions)

    def test_default_fixture_input_digest_matches_compute_input_digest(self) -> None:
        ref, token_sequences = get_fixture(DEFAULT_FIXTURE_ID)
        assert ref.input_digest == compute_input_digest(token_sequences)

    def test_default_fixture_is_bound_to_a_complete_tokenizer_fingerprint(self) -> None:
        ref, _ = get_fixture(DEFAULT_FIXTURE_ID)
        fp = ref.tokenizer_fingerprint
        assert fp.vocab_size is not None
        assert fp.special_tokens_digest is not None
        assert fp.token_id_map_digest is not None
        assert fp.chat_template_digest is not None

    def test_unknown_fixture_id_raises_unknown_fixture_error(self) -> None:
        with pytest.raises(UnknownFixtureError):
            get_fixture("does-not-exist")

    def test_unknown_fixture_error_str_is_a_clear_message_not_a_quoted_key_repr(self) -> None:
        """``UnknownFixtureError`` subclasses ``KeyError``, whose ``__str__`` would otherwise
        wrap the id in quotes (``"'foo'"``) with no explanation -- a real CLI regression
        (``Error: 'foo'``) this pins against.
        """
        error = UnknownFixtureError("foo")
        assert str(error) == (
            "unknown parity fixture 'foo'; the only built-in fixture is 'default-v1' "
            "— use --prompts to build a fixture for your own model"
        )


class TestComputeInputDigest:
    """compute_input_digest is a deterministic, order/value-sensitive hash over sequences."""

    def test_deterministic_for_same_sequences(self) -> None:
        assert compute_input_digest(((1, 2, 3),)) == compute_input_digest(((1, 2, 3),))

    def test_differs_for_different_values(self) -> None:
        assert compute_input_digest(((1, 2, 3),)) != compute_input_digest(((1, 2, 4),))

    def test_differs_for_different_order_within_a_sequence(self) -> None:
        assert compute_input_digest(((1, 2, 3),)) != compute_input_digest(((3, 2, 1),))

    def test_differs_for_different_order_across_sequences(self) -> None:
        assert compute_input_digest(((1, 2), (3, 4))) != compute_input_digest(((3, 4), (1, 2)))

    def test_differs_when_a_token_moves_between_sequences(self) -> None:
        """Same flattened concatenation ([1, 2, 3, 4]), different sequence structure."""
        moved_into_first = ((1, 2, 3), (4,))
        moved_into_second = ((1, 2), (3, 4))
        assert compute_input_digest(moved_into_first) != compute_input_digest(moved_into_second)


class TestTokenIdsInRange:
    """A fixture's token ids must be validatable as in-range for a vocab size, across sequences."""

    def test_default_fixture_ids_are_in_range_for_its_own_vocab_size(self) -> None:
        ref, token_sequences = get_fixture(DEFAULT_FIXTURE_ID)
        vocab_size = ref.tokenizer_fingerprint.vocab_size
        assert vocab_size is not None
        assert token_ids_in_range(token_sequences, vocab_size=vocab_size) is True

    def test_out_of_range_high_id_is_rejected(self) -> None:
        assert token_ids_in_range(((0, 1, 100),), vocab_size=10) is False

    def test_negative_id_is_rejected(self) -> None:
        assert token_ids_in_range(((0, -1, 2),), vocab_size=10) is False

    def test_id_equal_to_vocab_size_is_out_of_range(self) -> None:
        # ids are 0-indexed: vocab_size - 1 is the last valid id.
        assert token_ids_in_range(((0, 9),), vocab_size=10) is True
        assert token_ids_in_range(((0, 10),), vocab_size=10) is False

    def test_empty_token_sequences_are_vacuously_in_range(self) -> None:
        assert token_ids_in_range((), vocab_size=10) is True

    def test_a_single_empty_sequence_is_vacuously_in_range(self) -> None:
        assert token_ids_in_range(((),), vocab_size=10) is True

    def test_checks_every_sequence_not_just_the_first(self) -> None:
        """The first sequence is fully in-range; only the second has a bad id."""
        assert token_ids_in_range(((0, 1), (2, 100)), vocab_size=10) is False


class TestBuildFixtureFromPrompts:
    """build_fixture_from_prompts turns (prompt, completion) pairs into a fixture (F9)."""

    def test_single_pair_builds_the_full_teacher_forced_sequence(self) -> None:
        template = _ScriptedChatTemplate(
            {
                ("hello", True): [1, 2, 3],
                ("hello", False): [1, 2, 3, 4, 5],
            }
        )
        fp = _fingerprint()

        ref, sequences = build_fixture_from_prompts(
            apply_chat_template=template,
            pairs=[("hello", "world")],
            tokenizer_fingerprint=fp,
        )

        assert sequences == ((1, 2, 3, 4, 5),)
        assert ref.id == "user-prompts"
        assert ref.max_length == 5
        assert ref.tokenizer_fingerprint is fp

    def test_single_pair_scores_only_the_completion_prefix_positions(self) -> None:
        """prompt_ids has 3 ids, full_ids has 5: the completion occupies positions 3-4,
        so the positions whose logits PREDICT those tokens are 2-3 (index len(prompt)-1
        through len(full)-2), not the prompt's own positions and not position 4 itself
        (nothing exists to predict beyond the last token).
        """
        template = _ScriptedChatTemplate(
            {
                ("hello", True): [1, 2, 3],
                ("hello", False): [1, 2, 3, 4, 5],
            }
        )
        ref, _ = build_fixture_from_prompts(
            apply_chat_template=template,
            pairs=[("hello", "world")],
            tokenizer_fingerprint=_fingerprint(),
        )
        assert ref.scored_positions == (2, 3)

    def test_two_pairs_concatenate_in_order_with_scored_positions_offset(self) -> None:
        """Catches a scored-position offset bug that only shows up past the first
        sequence: the second pair's scored range must be shifted by the FIRST
        sequence's length, not computed as if it started the concatenation.
        """
        template = _ScriptedChatTemplate(
            {
                ("hello", True): [1, 2, 3],
                ("hello", False): [1, 2, 3, 4, 5],
                ("bye", True): [10, 11],
                ("bye", False): [10, 11, 12, 13],
            }
        )
        ref, sequences = build_fixture_from_prompts(
            apply_chat_template=template,
            pairs=[("hello", "world"), ("bye", "now")],
            tokenizer_fingerprint=_fingerprint(),
        )

        assert sequences == ((1, 2, 3, 4, 5), (10, 11, 12, 13))
        assert ref.max_length == 9
        # pair 1: range(2, 4) -> (2, 3); pair 2: range(1, 3) offset by 5 -> (6, 7)
        assert ref.scored_positions == (2, 3, 6, 7)

    def test_calls_apply_chat_template_with_the_documented_message_shapes(self) -> None:
        template = _ScriptedChatTemplate(
            {
                ("hi", True): [1],
                ("hi", False): [1, 2],
            }
        )
        build_fixture_from_prompts(
            apply_chat_template=template,
            pairs=[("hi", "there")],
            tokenizer_fingerprint=_fingerprint(),
        )
        assert template.calls == [
            ([{"role": "user", "content": "hi"}], True),
            (
                [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "there"},
                ],
                False,
            ),
        ]

    def test_input_digest_matches_compute_input_digest_of_the_sequences(self) -> None:
        template = _ScriptedChatTemplate({("hi", True): [1], ("hi", False): [1, 2]})
        ref, sequences = build_fixture_from_prompts(
            apply_chat_template=template,
            pairs=[("hi", "there")],
            tokenizer_fingerprint=_fingerprint(),
        )
        assert ref.input_digest == compute_input_digest(sequences)

    def test_custom_fixture_id_is_used(self) -> None:
        template = _ScriptedChatTemplate({("hi", True): [1], ("hi", False): [1, 2]})
        ref, _ = build_fixture_from_prompts(
            apply_chat_template=template,
            pairs=[("hi", "there")],
            tokenizer_fingerprint=_fingerprint(),
            fixture_id="my-fixture",
        )
        assert ref.id == "my-fixture"

    def test_empty_completion_raises_value_error_naming_the_pair_index(self) -> None:
        """The second pair's completion adds no tokens beyond the prompt render --
        an unusable (empty) scored range -- and must be rejected, not silently
        produce a sequence with nothing to score.
        """
        template = _ScriptedChatTemplate(
            {
                ("hello", True): [1, 2, 3],
                ("hello", False): [1, 2, 3, 4, 5],
                ("bye", True): [10, 11],
                ("bye", False): [10, 11],  # no additional tokens
            }
        )
        with pytest.raises(ValueError, match=r"\b1\b"):
            build_fixture_from_prompts(
                apply_chat_template=template,
                pairs=[("hello", "world"), ("bye", "")],
                tokenizer_fingerprint=_fingerprint(),
            )

    def test_empty_pairs_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            build_fixture_from_prompts(
                apply_chat_template=_ScriptedChatTemplate({}),
                pairs=[],
                tokenizer_fingerprint=_fingerprint(),
            )

    def test_non_prefix_full_render_raises_value_error_naming_the_pair_index(self) -> None:
        """A tokenizer that is not append-stable at the prompt/completion boundary --
        the full render's own first len(prompt_ids) ids diverge from the prompt-only
        render, e.g. a BPE/SentencePiece merge across the boundary -- must be
        rejected: ``scored_positions`` is derived purely from ``len(prompt_ids)``, so
        silently trusting a non-prefix render would score the wrong logit positions.
        """
        template = _ScriptedChatTemplate(
            {
                ("hello", True): [1, 2, 3],
                # full_ids[:3] == [1, 2, 9] != prompt_ids == [1, 2, 3]: token 3 got
                # merged/retokenized away by the completion's presence, not just
                # appended to.
                ("hello", False): [1, 2, 9, 4, 5],
            }
        )
        with pytest.raises(ValueError, match=r"\b0\b"):
            build_fixture_from_prompts(
                apply_chat_template=template,
                pairs=[("hello", "world")],
                tokenizer_fingerprint=_fingerprint(),
            )


class TestReferenceTokenizerFiles:
    """reference_tokenizer_files exposes the default fixture's own reference tokenizer."""

    def test_returns_a_config_json_and_chat_template_that_reproduce_the_default_fingerprint(
        self,
    ) -> None:
        """A repository writing exactly these three artifacts must fingerprint
        identically to DEFAULT_FIXTURE_ID's own tokenizer_fingerprint -- that
        identity is the entire point of exposing this helper to callers/tests.
        """
        tokenizer_config, tokenizer_json, chat_template = reference_tokenizer_files()
        recomputed = tokenizer_fingerprint(
            tokenizer_config=tokenizer_config,
            tokenizer_json=tokenizer_json,
            chat_template=chat_template,
        )
        default_ref, _ = get_fixture(DEFAULT_FIXTURE_ID)
        assert recomputed == default_ref.tokenizer_fingerprint

    def test_returns_independent_copies_not_the_live_module_state(self) -> None:
        """Mutating one call's result must not leak into a later call -- callers
        (e.g. a test building a repo per test case) must be free to adapt the
        returned mappings without polluting other tests.
        """
        tokenizer_config, tokenizer_json, _ = reference_tokenizer_files()
        tokenizer_config["chat_template"] = "mutated"
        tokenizer_json["model"] = "mutated"

        tokenizer_config_again, tokenizer_json_again, _ = reference_tokenizer_files()
        assert tokenizer_config_again["chat_template"] != "mutated"
        assert tokenizer_json_again["model"] != "mutated"
