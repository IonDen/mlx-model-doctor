"""Loader-accurate LoRA target resolver (F5).

Reproduces, as a pure function with no MLX/mlx-lm import, the module-selection
logic of ``mlx_lm.tuner.utils.linear_to_lora_layers`` (pinned mlx-lm==0.31.3;
source read directly 2026-09-07 from the installed package):

- default per-block key discovery when ``lora_parameters.keys`` is absent or
  JSON ``null`` (mirrors ``config.get("keys", None)``);
- the ``layers[-max(num_layers, 0):]`` slice, so ``num_layers`` in ``{-1, 0}``
  (and any other negative value) selects every block, while a positive
  ``num_layers`` selects only the last N;
- the loader's *second*, unsliced full-model pass, which lets an explicit
  root/full-path key (``lm_head``, ``model.embed_tokens``) resolve outside
  the sliced blocks;
- ``fine_tune_type``, read at the top level of ``adapter_config`` (not from
  ``lora_parameters``) — a ``"full"`` fine-tune bypasses
  ``linear_to_lora_layers`` entirely, per ``mlx_lm.tuner.utils.load_adapters``.

Only architectures with a verified static parameter-name mapping are
resolved; an architecture outside that map is reported ``unverified`` rather
than guessed by suffix-matching a block-relative key against the flat module
list.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Literal

# Verified against pinned mlx-lm==0.31.3 model sources (2026-09-07,
# `mlx_lm/models/llama.py` / `qwen2.py` / `gemma.py`): each of these
# architectures wraps its transformer blocks at `model.layers` (the outer
# `Model.layers` property proxies to it) and every block exposes exactly
# these four attention projections plus three MLP projections as
# `nn.Linear` submodules — the same set `linear_to_lora_layers`'s
# `get_keys_for_lora` discovers by default (it matches any
# `nn.Linear`/`nn.QuantizedLinear`/`nn.Embedding`-family submodule of a
# block; norms like `input_layernorm` are not matched). "mistral" shares the
# identical structure because mlx-lm's `MODEL_REMAPPING` maps it onto the
# "llama" model file, not a distinct one.
_DEFAULT_TRANSFORMER_KEYS = frozenset(
    {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _ArchTargetMap:
    """A verified architecture's transformer-block path prefix + default keys."""

    block_prefix: str
    default_keys: frozenset[str]


_ARCH_TARGET_MAPS: dict[str, _ArchTargetMap] = {
    "llama": _ArchTargetMap(block_prefix="model.layers.", default_keys=_DEFAULT_TRANSFORMER_KEYS),
    "mistral": _ArchTargetMap(block_prefix="model.layers.", default_keys=_DEFAULT_TRANSFORMER_KEYS),
    "qwen2": _ArchTargetMap(block_prefix="model.layers.", default_keys=_DEFAULT_TRANSFORMER_KEYS),
    "gemma": _ArchTargetMap(block_prefix="model.layers.", default_keys=_DEFAULT_TRANSFORMER_KEYS),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedTargets:
    """The set of base-model module paths ``linear_to_lora_layers`` would target.

    ``covered`` is the resolver's expected-target set for the given
    ``num_layers``/``keys`` — full paths that would actually receive a LoRA
    layer. ``uncovered`` is the subset of the effective ``keys`` that
    correspond to no real module anywhere in the base model (a genuine
    config typo), independent of the ``num_layers`` slice — a key that is a
    real per-block name but simply falls outside the selected slice is
    neither covered nor uncovered. ``verified`` is ``False`` when ``arch``
    has no static parameter-name mapping; in that case only the
    arch-independent full-path pass is attempted, and ``uncovered`` should
    not be treated as authoritative.
    """

    covered: frozenset[str]
    uncovered: frozenset[str]
    keys: frozenset[str]
    keys_source: Literal["explicit", "default"]
    selected_block_indices: tuple[int, ...]
    verified: bool
    reason: str | None = None


@cache
def _block_index_pattern(block_prefix: str) -> re.Pattern[str]:
    """Compile (and cache) the regex that extracts a block index from a full path."""
    return re.compile(rf"^{re.escape(block_prefix)}(\d+)\.")


def _discover_block_indices(block_prefix: str, base_module_names: Sequence[str]) -> tuple[int, ...]:
    """Return every distinct block index present in ``base_module_names``, ascending."""
    pattern = _block_index_pattern(block_prefix)
    indices: set[int] = set()
    for name in base_module_names:
        match = pattern.match(name)
        if match:
            indices.add(int(match.group(1)))
    return tuple(sorted(indices))


def _effective_explicit_keys(lora_parameters: Mapping[str, object]) -> list[str] | None:
    """Return the explicit ``keys`` list, or ``None`` when discovery should run.

    Mirrors mlx-lm's ``(keys := config.get("keys", None)) is None`` check: a
    missing key and a JSON ``null`` both decode to Python ``None`` via
    ``dict.get``, so both trigger default discovery. A present-but-malformed
    value (not a list of strings) is treated as an empty explicit key set
    rather than raised on, since this is a pure resolver, not a validator.
    """
    raw = lora_parameters.get("keys")
    if raw is None:
        return None
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str)]


def _unverified_arch_result(
    effective_keys: frozenset[str],
    keys_source: Literal["explicit", "default"],
    arch: str,
    base_names: frozenset[str],
) -> ResolvedTargets:
    """Build the ``unverified`` result for an architecture with no static mapping.

    Only the arch-independent full-model literal-path pass is attempted here
    (never a guessed block-relative match), so ``base_names`` may be empty
    when no attempt should be made at all (the omitted-keys case, where there
    is nothing to look up).
    """
    covered = {key for key in effective_keys if key in base_names}
    return ResolvedTargets(
        covered=frozenset(covered),
        uncovered=frozenset(effective_keys - covered),
        keys=effective_keys,
        keys_source=keys_source,
        selected_block_indices=(),
        verified=False,
        reason=f"no verified static LoRA parameter-name mapping for architecture {arch!r}",
    )


def resolve_targets(
    adapter_config: Mapping[str, object],
    base_module_names: Sequence[str],
    *,
    arch: str,
) -> ResolvedTargets:
    """Resolve which ``base_module_names`` entries ``linear_to_lora_layers`` would target.

    ``adapter_config`` is the parsed top-level ``adapter_config.json`` object
    (``fine_tune_type``, ``num_layers``, and ``lora_parameters`` — the last
    optionally carrying ``keys`` — are all read from it). ``base_module_names``
    is the flat set of full module paths the base model exposes, matching
    what ``model.named_modules()`` would yield (e.g. ``"model.embed_tokens"``,
    ``"model.layers.0.self_attn.q_proj"``, ``"lm_head"``). ``arch`` is the
    base model's ``model_type`` (e.g. ``"llama"``).
    """
    fine_tune_type = adapter_config.get("fine_tune_type", "lora")
    if fine_tune_type == "full":
        return ResolvedTargets(
            covered=frozenset(),
            uncovered=frozenset(),
            keys=frozenset(),
            keys_source="default",
            selected_block_indices=(),
            verified=True,
            reason="fine_tune_type is 'full'; linear_to_lora_layers is not applied",
        )

    raw_lora_parameters = adapter_config.get("lora_parameters")
    lora_parameters: Mapping[str, object] = (
        raw_lora_parameters if isinstance(raw_lora_parameters, Mapping) else {}
    )
    explicit_keys = _effective_explicit_keys(lora_parameters)
    arch_map = _ARCH_TARGET_MAPS.get(arch)
    base_names = frozenset(base_module_names)

    keys_source: Literal["explicit", "default"]
    if explicit_keys is None:
        keys_source = "default"
        if arch_map is None:
            # No default keys can be discovered without a verified mapping.
            return _unverified_arch_result(frozenset(), keys_source, arch, base_names)
        effective_keys = arch_map.default_keys
    else:
        keys_source = "explicit"
        effective_keys = frozenset(explicit_keys)
        if arch_map is None:
            # No verified block-container path: only the arch-independent
            # full-path pass can be attempted; per-block matching is never
            # guessed by suffix.
            return _unverified_arch_result(effective_keys, keys_source, arch, base_names)

    num_layers_raw = adapter_config.get("num_layers", 0)
    num_layers = (
        num_layers_raw
        if isinstance(num_layers_raw, int) and not isinstance(num_layers_raw, bool)
        else 0
    )

    all_block_indices = _discover_block_indices(arch_map.block_prefix, base_module_names)
    # Mirrors `model.layers[-max(num_layers, 0):]` verbatim: `-max(n, 0)` is
    # `0` for any `n <= 0` (and `list[-0:] == list[0:]`, the whole list, since
    # negative zero indexes the same as zero), so `num_layers` in `{-1, 0}`
    # (or any other negative value) selects every block; a positive
    # `num_layers` slices the last N, clamped to the full list if N exceeds
    # the block count.
    selected_indices = all_block_indices[-max(num_layers, 0) :]
    selected_set = set(selected_indices)

    # Second full-model pass (arch-independent, unsliced): a key that is a
    # literal full path resolves regardless of the block slice.
    covered: set[str] = {key for key in effective_keys if key in base_names}
    matched_anywhere: set[str] = set(covered)

    # Per-block pass: relative-to-block keys, restricted to the selected
    # slice for `covered`, but checked against every block (not just the
    # slice) to tell a genuine typo (`uncovered`) from a real per-block key
    # that is simply outside this run's `num_layers` window.
    for key in effective_keys:
        for index in all_block_indices:
            candidate = f"{arch_map.block_prefix}{index}.{key}"
            if candidate in base_names:
                matched_anywhere.add(key)
                if index in selected_set:
                    covered.add(candidate)

    uncovered = effective_keys - matched_anywhere

    return ResolvedTargets(
        covered=frozenset(covered),
        uncovered=frozenset(uncovered),
        keys=effective_keys,
        keys_source=keys_source,
        selected_block_indices=selected_indices,
        verified=True,
        reason=None,
    )
