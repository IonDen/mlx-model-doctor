"""Pinned, tokenizer-bound token-id fixtures for the parity oracle (F6/F9).

A fixture supplies the *identical* teacher-forced token-id sequences every
worker feeds to base/adapter/fused, one forward per sequence, in the same
model load (see :mod:`mlx_model_doctor.parity.worker`). Their per-sequence
argmax vectors are concatenated in sequence order into a single flat vector
that :mod:`mlx_model_doctor.parity.oracle` scores (``scored_positions`` index
into that concatenation). Binding the fixture to a
:class:`~mlx_model_doctor.parity.context.TokenizerFingerprint` makes that
binding explicit and checkable: the sequences only have the meaning they were
built for alongside a tokenizer that produces the exact same fingerprint, so
a fixture built for one tokenizer is never silently fed to a repository with
a different (or unverifiable) one -- the caller compares
:attr:`FixtureRef.tokenizer_fingerprint` against the target's own computed
fingerprint (via :func:`~mlx_model_doctor.parity.context.tokenizers_match`)
before trusting the fixture's ids.

``DEFAULT_FIXTURE_ID`` is a small, fully-inline fixture for the resolver's
supported architecture family (see
:mod:`mlx_model_doctor.parity.resolver`): its
``tokenizer_fingerprint`` is derived, via
:func:`~mlx_model_doctor.parity.context.tokenizer_fingerprint`, from a
minimal reference tokenizer definition committed alongside it below -- it is
a deliberately small, documented stand-in, not scraped from a live
repository. Wiring it against a *specific* real base/fused repository's
tokenizer is left to the caller: when a target's own fingerprint does not
match ``DEFAULT_FIXTURE_ID``'s, construct a ``FixtureRef`` (via
``tokenizer_fingerprint()``) and a matching token-id tuple for that
tokenizer directly, rather than forcing the default -- the documented
user-fixture path.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

from mlx_model_doctor.errors import ModelDoctorError
from mlx_model_doctor.parity.context import TokenizerFingerprint, tokenizer_fingerprint


@dataclass(frozen=True, slots=True, kw_only=True)
class FixtureRef:
    """Identity and tokenizer-binding contract for a pinned oracle-input fixture.

    ``max_length`` is the TOTAL number of token ids across all of the
    fixture's sequences (i.e. the length of the flat, per-sequence-argmax
    concatenation a worker produces); ``scored_positions`` are indices into
    that same flat concatenation whose per-position argmax should be included
    when comparing model outputs.
    """

    id: str
    input_digest: str
    max_length: int
    scored_positions: tuple[int, ...]
    tokenizer_fingerprint: TokenizerFingerprint


class UnknownFixtureError(ModelDoctorError, KeyError):
    """Raised by get_fixture for a fixture id with no pinned registration."""


def compute_input_digest(token_sequences: Sequence[Sequence[int]]) -> str:
    """Return a stable sha256 hex digest of an ordered sequence of token-id sequences.

    Order-sensitive and value-sensitive both WITHIN and ACROSS sequences:
    reordering or changing any id, or moving an id from one sequence to
    another, changes the digest. Used both to build a
    ``FixtureRef.input_digest`` and to let a caller verify a worker's
    reported ids against the fixture it claims to have used.
    """
    encoded = json.dumps(
        [list(sequence) for sequence in token_sequences], separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def token_ids_in_range(token_sequences: Sequence[Sequence[int]], *, vocab_size: int) -> bool:
    """Return whether every token id across every sequence lies in ``[0, vocab_size)``.

    No sequences, or only empty sequences, are vacuously in range.
    """
    return all(0 <= token_id < vocab_size for sequence in token_sequences for token_id in sequence)


def _flatten_length(token_sequences: Sequence[Sequence[int]]) -> int:
    """Return the total number of token ids across all of ``token_sequences``."""
    return sum(len(sequence) for sequence in token_sequences)


# --- Built-in default fixture ------------------------------------------------
#
# A small, self-contained reference tokenizer (not a real repository's) so
# this module stays offline and pure. Its vocabulary is intentionally tiny;
# the fingerprint below is computed from it via `tokenizer_fingerprint`, the
# same function a real ParityContext calls against an actual repository, so
# the two are directly comparable through `tokenizers_match`.
_REFERENCE_VOCAB: dict[str, int] = {
    "<s>": 0,
    "</s>": 1,
    "<pad>": 2,
    "<unk>": 3,
    **{f"<tok_{i}>": i for i in range(4, 64)},
}

_REFERENCE_TOKENIZER_JSON: dict[str, object] = {
    "model": {"type": "BPE", "vocab": _REFERENCE_VOCAB},
}

_REFERENCE_CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"
)

_REFERENCE_TOKENIZER_CONFIG: dict[str, object] = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "chat_template": _REFERENCE_CHAT_TEMPLATE,
}

DEFAULT_FIXTURE_ID = "default-v1"

# BOS, a short run of ordinary tokens, EOS -- teacher-forced, no decode loop.
# A single-sequence fixture today; get_fixture's contract supports any number
# of (nonempty) sequences.
_DEFAULT_TOKEN_SEQUENCES: tuple[tuple[int, ...], ...] = ((0, 10, 22, 45, 7, 33, 1),)

_DEFAULT_FIXTURE = FixtureRef(
    id=DEFAULT_FIXTURE_ID,
    input_digest=compute_input_digest(_DEFAULT_TOKEN_SEQUENCES),
    max_length=_flatten_length(_DEFAULT_TOKEN_SEQUENCES),
    scored_positions=tuple(range(_flatten_length(_DEFAULT_TOKEN_SEQUENCES))),
    tokenizer_fingerprint=tokenizer_fingerprint(
        tokenizer_config=_REFERENCE_TOKENIZER_CONFIG,
        tokenizer_json=_REFERENCE_TOKENIZER_JSON,
        chat_template=_REFERENCE_CHAT_TEMPLATE,
    ),
)

_FIXTURES: dict[str, tuple[FixtureRef, tuple[tuple[int, ...], ...]]] = {
    DEFAULT_FIXTURE_ID: (_DEFAULT_FIXTURE, _DEFAULT_TOKEN_SEQUENCES),
}


def get_fixture(fixture_id: str) -> tuple[FixtureRef, tuple[tuple[int, ...], ...]]:
    """Return the pinned ``(FixtureRef, token_sequences)`` pair for ``fixture_id``.

    ``token_sequences`` is a tuple of one or more token-id sequences; a
    worker runs one forward per sequence (one model load) and concatenates
    the per-sequence argmax, in sequence order, into the flat vector
    ``FixtureRef.scored_positions`` indexes into.

    Only :data:`DEFAULT_FIXTURE_ID` is built in today. For a repository whose
    own tokenizer fingerprint does not match it, build a fixture for that
    tokenizer directly (see the module docstring) instead of calling this
    function -- the documented user-fixture path.

    Raises:
        UnknownFixtureError: ``fixture_id`` has no registered fixture.
    """
    try:
        return _FIXTURES[fixture_id]
    except KeyError:
        raise UnknownFixtureError(fixture_id) from None
