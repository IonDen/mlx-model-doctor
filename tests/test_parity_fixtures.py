"""Tests for pinned, tokenizer-bound parity-oracle fixtures (F6/F9)."""

import pytest

from mlx_model_doctor.parity.fixtures import (
    DEFAULT_FIXTURE_ID,
    FixtureRef,
    UnknownFixtureError,
    compute_input_digest,
    get_fixture,
    token_ids_in_range,
)


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
