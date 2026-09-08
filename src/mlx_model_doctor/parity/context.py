"""Tokenizer identity fingerprinting for the adapter-parity checker (F6).

A plain vocab-size + special-token check cannot detect two real
tokenizer incompatibilities between a base and a fused (or fixture-bound)
repository: a same-size-vocab token-ID *permutation* (two ordinary tokens
swap IDs; vocab size and special tokens are untouched) and a *changed chat
template* (identical tokens, a different prompt format). Both silently
produce numerically different model outputs for what looks like the same
input. :func:`tokenizer_fingerprint` adds a digest of the ordinary
token-ID mapping and a digest of the effective chat template on top of the
vocab-size/special-token summary, so :func:`tokenizers_match` catches both.

Pure and offline: every input is already-parsed tokenizer data (the
``tokenizer_config.json`` mapping, the ``tokenizer.json`` mapping, and the
resolved chat-template text) -- this module never loads a tokenizer or reads
a :class:`~mlx_model_doctor.targets.ModelTarget` itself. That is what makes
it exercisable with synthetic data in tests. A later task adds
``ParityContext`` (which resolves those inputs from real repositories) to
this module.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

_SPECIAL_TOKEN_KEYS = (
    "bos_token",
    "eos_token",
    "unk_token",
    "pad_token",
    "sep_token",
    "cls_token",
    "mask_token",
    "additional_special_tokens",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenizerFingerprint:
    """A bounded, offline-computable summary of tokenizer identity.

    Each field is ``None`` exactly when its source data was absent or
    unusable -- never guessed or defaulted -- so :func:`tokenizers_match`
    can tell "this aspect could not be compared" apart from "this aspect
    matched".
    """

    vocab_size: int | None
    special_tokens_digest: str | None
    token_id_map_digest: str | None
    chat_template_digest: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class MatchResult:
    """The outcome of comparing two :class:`TokenizerFingerprint` values."""

    matched: bool
    reason: str | None


def _digest(payload: object) -> str:
    """Return a stable sha256 hex digest of a JSON-serializable payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _special_tokens(tokenizer_config: Mapping[str, object] | None) -> dict[str, object] | None:
    """Return the special-token fields present in tokenizer_config, or None."""
    if tokenizer_config is None:
        return None
    collected = {
        key: tokenizer_config[key] for key in _SPECIAL_TOKEN_KEYS if key in tokenizer_config
    }
    return collected or None


def _vocab_mapping(tokenizer_json: Mapping[str, object] | None) -> dict[str, int] | None:
    """Return the ordinary token -> id mapping from tokenizer_json, or None.

    ``None`` when the mapping is absent, empty, or malformed (a non-mapping
    ``model``/``vocab``, or an entry with a non-string token or a value that
    is not exactly ``int`` -- ``type(x) is int`` excludes ``bool``, since
    ``isinstance(True, int)`` is ``True`` in Python). Callers must treat
    ``None`` as "cannot compare", never as an empty match.
    """
    if tokenizer_json is None:
        return None
    model = tokenizer_json.get("model")
    if not isinstance(model, Mapping):
        return None
    vocab = model.get("vocab")
    if not isinstance(vocab, Mapping) or not vocab:
        return None
    mapping: dict[str, int] = {}
    for token, token_id in vocab.items():
        if not isinstance(token, str) or type(token_id) is not int:
            return None
        mapping[token] = token_id
    return mapping


def tokenizer_fingerprint(
    *,
    tokenizer_config: Mapping[str, object] | None,
    tokenizer_json: Mapping[str, object] | None,
    chat_template: str | None,
) -> TokenizerFingerprint:
    """Compute a bounded tokenizer-identity fingerprint from parsed tokenizer data.

    ``tokenizer_config``/``tokenizer_json`` are the parsed
    ``tokenizer_config.json``/``tokenizer.json`` mappings (``None`` when
    absent); ``chat_template`` is the resolved template text (``None`` when
    absent). Beyond vocab size and the special-token fields, this hashes the
    ordinary token-ID mapping and the chat template, so a same-size
    permutation or a changed template yields a different fingerprint.
    """
    vocab_mapping = _vocab_mapping(tokenizer_json)
    special_tokens = _special_tokens(tokenizer_config)
    return TokenizerFingerprint(
        vocab_size=len(vocab_mapping) if vocab_mapping is not None else None,
        special_tokens_digest=_digest(special_tokens) if special_tokens is not None else None,
        token_id_map_digest=_digest(vocab_mapping) if vocab_mapping is not None else None,
        chat_template_digest=_digest(chat_template) if chat_template is not None else None,
    )


def tokenizers_match(a: TokenizerFingerprint, b: TokenizerFingerprint) -> MatchResult:
    """Compare two fingerprints, reporting the first aspect that differs.

    An aspect neither side (or only one side) could compute compares as
    *unavailable* -- a distinct, named reason -- rather than as a match: an
    uncomparable representation must never look identical, even when both
    sides are identically unable to produce it. ``vocab_size`` and
    ``token_id_map_digest`` are derived from the same ordinary token-id
    mapping (see :func:`tokenizer_fingerprint`), so they are unavailable
    together; one ``vocab_unavailable`` check covers both.
    """
    if (
        a.vocab_size is None
        or b.vocab_size is None
        or a.token_id_map_digest is None
        or b.token_id_map_digest is None
    ):
        return MatchResult(matched=False, reason="vocab_unavailable")
    if a.vocab_size != b.vocab_size:
        return MatchResult(matched=False, reason="vocab_size_differs")
    if a.token_id_map_digest != b.token_id_map_digest:
        return MatchResult(matched=False, reason="token_id_map_differs")

    if a.special_tokens_digest is None or b.special_tokens_digest is None:
        return MatchResult(matched=False, reason="special_tokens_unavailable")
    if a.special_tokens_digest != b.special_tokens_digest:
        return MatchResult(matched=False, reason="special_tokens_differ")

    if a.chat_template_digest is None or b.chat_template_digest is None:
        return MatchResult(matched=False, reason="chat_template_unavailable")
    if a.chat_template_digest != b.chat_template_digest:
        return MatchResult(matched=False, reason="chat_template_differs")

    return MatchResult(matched=True, reason=None)
