"""Tests for the loader-accurate LoRA target resolver (F5).

Each test pins ``resolve_targets`` against a specific clause of mlx-lm's
``tuner.utils.linear_to_lora_layers`` (pinned mlx-lm==0.31.3, source read
2026-09-07): default key discovery, the ``layers[-max(num_layers, 0):]``
slice, the second full-model pass for root/full-path keys, uncovered-key
detection, and the unverified-architecture fallback.
"""

from mlx_model_doctor.parity.resolver import ResolvedTargets, resolve_targets

# A tiny synthetic 4-block llama-family base model's full module paths, shaped
# exactly like what ``model.named_modules()`` would yield for a real mlx-lm
# ``llama.py``-family ``Model``: an embedding, 4 transformer blocks (each with
# the 7 default-discoverable Linear projections plus 2 non-Linear norms that
# must NOT be discovered by default), a final norm, and an untied ``lm_head``.
_BLOCK_LINEAR_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
_BLOCK_NON_LINEAR_SUFFIXES = ("input_layernorm", "post_attention_layernorm")
_NUM_BLOCKS = 4


def _base_module_names() -> tuple[str, ...]:
    names = ["model.embed_tokens"]
    for i in range(_NUM_BLOCKS):
        names.extend(
            f"model.layers.{i}.{suffix}"
            for suffix in _BLOCK_LINEAR_SUFFIXES + _BLOCK_NON_LINEAR_SUFFIXES
        )
    names.append("model.norm")
    names.append("lm_head")
    return tuple(names)


BASE_MODULE_NAMES = _base_module_names()


def _config(*, num_layers: int, keys: list[str] | None = None) -> dict[str, object]:
    lora_parameters: dict[str, object] = {}
    if keys is not None:
        lora_parameters["keys"] = keys
    return {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": lora_parameters,
    }


class TestAllBlocksSlice:
    """``num_layers`` in {-1, 0} must select every block (``max(n, 0)`` + full slice)."""

    def test_num_layers_minus_one_selects_all_blocks(self) -> None:
        config = _config(num_layers=-1, keys=["self_attn.q_proj"])
        resolved = resolve_targets(config, BASE_MODULE_NAMES, arch="llama")
        expected = {f"model.layers.{i}.self_attn.q_proj" for i in range(_NUM_BLOCKS)}
        # A `-max(num_layers, 0)` regression (e.g. dropping the `max`) would
        # slice `layers[1:]` instead of the whole list, losing block 0.
        assert resolved.covered == expected

    def test_num_layers_zero_selects_all_blocks(self) -> None:
        config = _config(num_layers=0, keys=["self_attn.q_proj"])
        resolved = resolve_targets(config, BASE_MODULE_NAMES, arch="llama")
        expected = {f"model.layers.{i}.self_attn.q_proj" for i in range(_NUM_BLOCKS)}
        # A naive `layers[-num_layers:]` (no `max`) would slice `layers[-0:]`
        # correctly by luck in Python, but a `layers[:num_layers]` inversion
        # would select nothing (empty slice) instead of everything.
        assert resolved.covered == expected


class TestPositiveNumLayersLimitsToLastN:
    """A positive ``num_layers`` must select only the LAST N blocks, not the first N."""

    def test_positive_num_layers_selects_last_n_only(self) -> None:
        config = _config(num_layers=2, keys=["self_attn.q_proj"])
        resolved = resolve_targets(config, BASE_MODULE_NAMES, arch="llama")
        # Last 2 of 4 blocks are indices 2 and 3, not 0 and 1. A `layers[:n]`
        # (first-N) inversion would produce {0, 1} instead and this assertion
        # would fail exactly on the set contents.
        assert resolved.covered == {
            "model.layers.2.self_attn.q_proj",
            "model.layers.3.self_attn.q_proj",
        }

    def test_positive_num_layers_larger_than_block_count_selects_all(self) -> None:
        config = _config(num_layers=100, keys=["self_attn.q_proj"])
        resolved = resolve_targets(config, BASE_MODULE_NAMES, arch="llama")
        expected = {f"model.layers.{i}.self_attn.q_proj" for i in range(_NUM_BLOCKS)}
        assert resolved.covered == expected


class TestRootFullPathKeyResolvesOutsideSlice:
    """A root/full-path key (e.g. ``lm_head``) must resolve via the loader's
    second, unsliced full-model pass — independent of the block slice."""

    def test_root_key_resolves_even_when_no_blocks_selected_for_it(self) -> None:
        # num_layers=1 selects only block 3; lm_head is not block-scoped at
        # all, so it must still resolve. A resolver that only ran the
        # per-block pass (dropped the second full-model pass) would produce
        # an empty `covered` here.
        config = _config(num_layers=1, keys=["lm_head"])
        resolved = resolve_targets(config, BASE_MODULE_NAMES, arch="llama")
        assert resolved.covered == {"lm_head"}

    def test_root_key_and_block_key_combine_correctly(self) -> None:
        config = _config(num_layers=1, keys=["lm_head", "self_attn.q_proj"])
        resolved = resolve_targets(config, BASE_MODULE_NAMES, arch="llama")
        # lm_head resolves regardless of slice; self_attn.q_proj resolves
        # ONLY in the selected last-1 block (index 3), not blocks 0-2.
        assert resolved.covered == {"lm_head", "model.layers.3.self_attn.q_proj"}


class TestOmittedKeysUseDiscoveryDefaults:
    """Omitting ``lora_parameters.keys`` must fall back to the static default
    per-arch target set, matching every default-discoverable module across
    every selected block, and none of the non-Linear norm modules."""

    def test_omitted_keys_discovers_default_targets_on_all_blocks(self) -> None:
        config = _config(num_layers=-1)  # no "keys" entry at all
        resolved = resolve_targets(config, BASE_MODULE_NAMES, arch="llama")
        assert resolved.keys_source == "default"
        expected = {
            f"model.layers.{i}.{suffix}"
            for i in range(_NUM_BLOCKS)
            for suffix in _BLOCK_LINEAR_SUFFIXES
        }
        # A default map that (wrongly) included a norm suffix would inflate
        # this set; a default map missing a real Linear suffix would shrink it.
        assert resolved.covered == expected
        for suffix in _BLOCK_NON_LINEAR_SUFFIXES:
            assert f"model.layers.0.{suffix}" not in resolved.covered

    def test_null_keys_also_uses_discovery_defaults(self) -> None:
        # JSON `"keys": null` decodes to Python None, same as an absent key
        # (mirrors mlx-lm's `config.get("keys", None)`).
        config = _config(num_layers=-1, keys=None)
        config["lora_parameters"] = {"keys": None}
        resolved = resolve_targets(config, BASE_MODULE_NAMES, arch="llama")
        assert resolved.keys_source == "default"
        assert resolved.covered  # non-empty: discovery ran


class TestUnmatchedKeyIsUncovered:
    """A key that corresponds to no real module anywhere in the base model
    (not a valid full path, not a valid relative-to-any-block name) is a true
    typo and must surface as ``uncovered``, never silently dropped."""

    def test_genuinely_unmatched_key_is_uncovered_not_covered(self) -> None:
        config = _config(num_layers=-1, keys=["self_attn.nonexistent_proj"])
        resolved = resolve_targets(config, BASE_MODULE_NAMES, arch="llama")
        assert resolved.covered == frozenset()
        assert resolved.uncovered == frozenset({"self_attn.nonexistent_proj"})

    def test_mixed_matched_and_unmatched_keys_partition_correctly(self) -> None:
        config = _config(num_layers=-1, keys=["self_attn.q_proj", "totally.bogus.key"])
        resolved = resolve_targets(config, BASE_MODULE_NAMES, arch="llama")
        assert "totally.bogus.key" in resolved.uncovered
        assert "self_attn.q_proj" not in resolved.uncovered
        assert "model.layers.0.self_attn.q_proj" in resolved.covered


class TestUnknownArchIsUnverified:
    """An architecture with no verified static parameter-name mapping must be
    reported ``unverified`` — never resolved by guessing a suffix pattern."""

    def test_unknown_arch_with_omitted_keys_is_unverified_and_empty(self) -> None:
        config = _config(num_layers=-1)
        resolved = resolve_targets(config, BASE_MODULE_NAMES, arch="some-future-arch-xyz")
        assert resolved.verified is False
        assert resolved.reason is not None
        assert resolved.covered == frozenset()

    def test_unknown_arch_does_not_invent_block_relative_matches(self) -> None:
        # Even though "self_attn.q_proj" IS a real relative-to-block name in
        # BASE_MODULE_NAMES, an unverified arch must not guess the block
        # prefix by suffix-matching to find it.
        config = _config(num_layers=-1, keys=["self_attn.q_proj"])
        resolved = resolve_targets(config, BASE_MODULE_NAMES, arch="some-future-arch-xyz")
        assert resolved.verified is False
        assert "model.layers.0.self_attn.q_proj" not in resolved.covered


class TestFineTuneTypeReadAtTopLevel:
    """``fine_tune_type`` is a top-level ``adapter_config`` field; a "full"
    fine-tune bypasses ``linear_to_lora_layers`` entirely (mlx-lm's
    ``load_adapters`` only calls it when ``fine_tune_type != "full"``)."""

    def test_full_fine_tune_type_resolves_no_targets(self) -> None:
        config = {
            "fine_tune_type": "full",
            "num_layers": -1,
            "lora_parameters": {"keys": ["self_attn.q_proj"]},
        }
        resolved = resolve_targets(config, BASE_MODULE_NAMES, arch="llama")
        assert resolved.covered == frozenset()
        assert resolved.uncovered == frozenset()


class TestMalformedExplicitKeysDoNotCrash:
    """A malformed ``lora_parameters.keys`` (not a list) must not raise — this
    is a pure resolver, not a validator; validation is a separate check."""

    def test_non_list_keys_value_resolves_no_targets_without_raising(self) -> None:
        config = _config(num_layers=-1)
        config["lora_parameters"] = {"keys": "self_attn.q_proj"}  # malformed: a bare string
        resolved = resolve_targets(config, BASE_MODULE_NAMES, arch="llama")
        # A regression that iterated a bare string char-by-char (since Python
        # strings are themselves iterable) would silently produce a nonempty,
        # nonsensical key set instead of failing safe to empty.
        assert resolved.keys == frozenset()
        assert resolved.covered == frozenset()


class TestResolvedTargetsIsFrozen:
    """``ResolvedTargets`` is a plain immutable value; guards against an
    accidental mutable-default or a non-frozen regression."""

    def test_resolved_targets_fields_are_frozensets(self) -> None:
        config = _config(num_layers=-1, keys=["lm_head"])
        resolved = resolve_targets(config, BASE_MODULE_NAMES, arch="llama")
        assert isinstance(resolved, ResolvedTargets)
        assert isinstance(resolved.covered, frozenset)
        assert isinstance(resolved.uncovered, frozenset)
