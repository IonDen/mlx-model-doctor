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

The fingerprinting core above (:class:`TokenizerFingerprint`,
:func:`tokenizer_fingerprint`, :func:`tokenizers_match`) is pure and
offline: every input is already-parsed tokenizer data (the
``tokenizer_config.json`` mapping, the ``tokenizer.json`` mapping, and the
resolved chat-template text), so it is exercisable with synthetic data in
tests without ever loading a tokenizer or reading a
:class:`~mlx_model_doctor.targets.ModelTarget`. :class:`ParityContext` below
is the part of this module that reads real repositories: it composes one
:class:`~mlx_model_doctor.context.CheckContext` per target (base, adapter,
fused), parses the adapter's ``adapter_config.json``, and computes a
target's tokenizer fingerprint from its actual repository files on demand.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field

from mlx_model_doctor.checks.chat_template import _template_string
from mlx_model_doctor.context import _MAX_METADATA_BYTES, CheckContext, CheckOptions
from mlx_model_doctor.errors import TargetError, raise_for_hf_target_error
from mlx_model_doctor.targets import ModelTarget

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


# --- Cross-target parity context (reads real repositories) ------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ParityTargets:
    """The three model targets an adapter-parity check compares.

    ``base`` is the unmodified base model, ``adapter`` is the standalone
    LoRA/DoRA adapter repository (``adapter_config.json`` plus
    ``adapters.safetensors``), and ``fused`` is the base model with the
    adapter merged into its weights (what an ``mlx_lm.fuse`` run produces).
    """

    base: ModelTarget
    adapter: ModelTarget
    fused: ModelTarget


def _read_adapter_config(target: ModelTarget) -> Mapping[str, object] | None:
    """Read and parse adapter_config.json from a target, or None if absent/unusable.

    A raw target read: :class:`~mlx_model_doctor.context.CheckContext` has no
    ``adapter_config.json`` accessor of its own, so this mirrors its own
    guarded-read pattern directly -- check existence and size before reading,
    and never let a local-target read failure masquerade as a genuine
    Hugging Face one (:func:`~mlx_model_doctor.errors.raise_for_hf_target_error`
    still propagates a real Hub error).
    """
    try:
        if not target.exists("adapter_config.json"):
            return None
        size = target.size("adapter_config.json")
        if size is None or size > _MAX_METADATA_BYTES:
            return None
        text = target.read_text("adapter_config.json", max_bytes=_MAX_METADATA_BYTES)
    except TargetError as exc:
        raise_for_hf_target_error(exc)
        return None
    except (FileNotFoundError, UnicodeError):
        return None
    try:
        parsed: object = json.loads(text)
    except (json.JSONDecodeError, RecursionError, UnicodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _read_tokenizer_json(target: ModelTarget) -> Mapping[str, object] | None:
    """Read and parse tokenizer.json from a target, or None if absent/unusable.

    Also a raw target read, for the same reason as :func:`_read_adapter_config`:
    :class:`~mlx_model_doctor.context.CheckContext` reads
    ``tokenizer_config.json`` but never the separate ``tokenizer.json`` vocab
    file :func:`tokenizer_fingerprint` needs.
    """
    try:
        if not target.exists("tokenizer.json"):
            return None
        size = target.size("tokenizer.json")
        if size is None or size > _MAX_METADATA_BYTES:
            return None
        text = target.read_text("tokenizer.json", max_bytes=_MAX_METADATA_BYTES)
    except TargetError as exc:
        raise_for_hf_target_error(exc)
        return None
    except (FileNotFoundError, UnicodeError):
        return None
    try:
        parsed: object = json.loads(text)
    except (json.JSONDecodeError, RecursionError, UnicodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _adapter_config_str_field(config: Mapping[str, object] | None, key: str) -> str | None:
    """Return a top-level string field from a parsed adapter_config, or None."""
    if config is None:
        return None
    value = config.get(key)
    return value if isinstance(value, str) else None


def _adapter_config_int_field(config: Mapping[str, object] | None, key: str) -> int | None:
    """Return a top-level int field from a parsed adapter_config, or None.

    Excludes ``bool`` (a subclass of ``int``), matching the resolver's own
    ``isinstance(x, int) and not isinstance(x, bool)`` convention.
    """
    if config is None:
        return None
    value = config.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _adapter_config_lora_keys(config: Mapping[str, object] | None) -> tuple[str, ...] | None:
    """Return the explicit ``lora_parameters.keys`` list, or None when absent/malformed.

    ``None`` covers "no lora_parameters" and "keys is JSON null" (both
    trigger the resolver's own default-key discovery) as well as a
    structurally malformed value (not a list of strings), which
    ``AdapterConfigCheck`` flags separately as a config-shape problem.
    """
    if config is None:
        return None
    lora_parameters = config.get("lora_parameters")
    if not isinstance(lora_parameters, Mapping):
        return None
    raw_keys = lora_parameters.get("keys")
    if raw_keys is None:
        return None
    if not isinstance(raw_keys, list) or not all(isinstance(item, str) for item in raw_keys):
        return None
    return tuple(raw_keys)


def _tokenizer_fingerprint_for(target: ModelTarget, ctx: CheckContext) -> TokenizerFingerprint:
    """Compute a tokenizer fingerprint for one target from its repository files."""
    return tokenizer_fingerprint(
        tokenizer_config=ctx.tokenizer_config_json(),
        tokenizer_json=_read_tokenizer_json(target),
        chat_template=_template_string(ctx),
    )


@dataclass(slots=True, kw_only=True)
class ParityContext:
    """Shared state for cross-target adapter-parity checks.

    Composes one :class:`~mlx_model_doctor.context.CheckContext` per target
    (``base``/``adapter``/``fused``) so parity checks reuse the same cached,
    guarded-read machinery single-target checks use, and eagerly parses the
    adapter's ``adapter_config.json`` once (:attr:`adapter_config`, plus the
    :attr:`num_layers`/:attr:`fine_tune_type`/:attr:`lora_parameter_keys`
    convenience properties over its parity-relevant top-level fields).
    """

    targets: ParityTargets
    options: CheckOptions
    base: CheckContext = field(init=False)
    adapter: CheckContext = field(init=False)
    fused: CheckContext = field(init=False)
    adapter_config: Mapping[str, object] | None = field(init=False)

    def __post_init__(self) -> None:
        """Build the per-target check contexts and parse adapter_config.json once."""
        self.base = CheckContext(target=self.targets.base, options=self.options)
        self.adapter = CheckContext(target=self.targets.adapter, options=self.options)
        self.fused = CheckContext(target=self.targets.fused, options=self.options)
        self.adapter_config = _read_adapter_config(self.targets.adapter)

    @property
    def num_layers(self) -> int | None:
        """Return adapter_config's top-level num_layers, or None when absent/malformed."""
        return _adapter_config_int_field(self.adapter_config, "num_layers")

    @property
    def fine_tune_type(self) -> str | None:
        """Return adapter_config's top-level fine_tune_type, or None when absent/malformed."""
        return _adapter_config_str_field(self.adapter_config, "fine_tune_type")

    @property
    def lora_parameter_keys(self) -> tuple[str, ...] | None:
        """Return adapter_config's explicit lora_parameters.keys, or None."""
        return _adapter_config_lora_keys(self.adapter_config)

    def base_model_type(self) -> str | None:
        """Return the base target's config.json model_type, or None when absent/malformed."""
        config = self.base.config_json()
        if config is None:
            return None
        value = config.get("model_type")
        return value if isinstance(value, str) else None

    def base_tokenizer_fingerprint(self) -> TokenizerFingerprint:
        """Compute the base target's tokenizer fingerprint from its repository files."""
        return _tokenizer_fingerprint_for(self.targets.base, self.base)

    def fused_tokenizer_fingerprint(self) -> TokenizerFingerprint:
        """Compute the fused target's tokenizer fingerprint from its repository files."""
        return _tokenizer_fingerprint_for(self.targets.fused, self.fused)
