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
    """get_fixture(id) returns a pinned (FixtureRef, token_ids) pair."""

    def test_default_fixture_returns_ref_and_nonempty_token_ids(self) -> None:
        ref, token_ids = get_fixture(DEFAULT_FIXTURE_ID)
        assert isinstance(ref, FixtureRef)
        assert ref.id == DEFAULT_FIXTURE_ID
        assert isinstance(token_ids, tuple)
        assert len(token_ids) > 0
        assert all(type(token_id) is int for token_id in token_ids)

    def test_default_fixture_max_length_matches_its_token_ids(self) -> None:
        ref, token_ids = get_fixture(DEFAULT_FIXTURE_ID)
        assert ref.max_length == len(token_ids)

    def test_default_fixture_scored_positions_are_within_bounds(self) -> None:
        ref, token_ids = get_fixture(DEFAULT_FIXTURE_ID)
        assert len(ref.scored_positions) > 0
        assert all(0 <= pos < len(token_ids) for pos in ref.scored_positions)

    def test_default_fixture_input_digest_matches_compute_input_digest(self) -> None:
        ref, token_ids = get_fixture(DEFAULT_FIXTURE_ID)
        assert ref.input_digest == compute_input_digest(token_ids)

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
    """compute_input_digest is a deterministic, order/value-sensitive hash."""

    def test_deterministic_for_same_ids(self) -> None:
        assert compute_input_digest((1, 2, 3)) == compute_input_digest((1, 2, 3))

    def test_differs_for_different_values(self) -> None:
        assert compute_input_digest((1, 2, 3)) != compute_input_digest((1, 2, 4))

    def test_differs_for_different_order(self) -> None:
        assert compute_input_digest((1, 2, 3)) != compute_input_digest((3, 2, 1))


class TestTokenIdsInRange:
    """A fixture's token ids must be validatable as in-range for a vocab size."""

    def test_default_fixture_ids_are_in_range_for_its_own_vocab_size(self) -> None:
        ref, token_ids = get_fixture(DEFAULT_FIXTURE_ID)
        vocab_size = ref.tokenizer_fingerprint.vocab_size
        assert vocab_size is not None
        assert token_ids_in_range(token_ids, vocab_size=vocab_size) is True

    def test_out_of_range_high_id_is_rejected(self) -> None:
        assert token_ids_in_range((0, 1, 100), vocab_size=10) is False

    def test_negative_id_is_rejected(self) -> None:
        assert token_ids_in_range((0, -1, 2), vocab_size=10) is False

    def test_id_equal_to_vocab_size_is_out_of_range(self) -> None:
        # ids are 0-indexed: vocab_size - 1 is the last valid id.
        assert token_ids_in_range((0, 9), vocab_size=10) is True
        assert token_ids_in_range((0, 10), vocab_size=10) is False

    def test_empty_token_ids_are_vacuously_in_range(self) -> None:
        assert token_ids_in_range((), vocab_size=10) is True
