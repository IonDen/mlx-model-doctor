"""The index-vs-shipped-shards classifier shared by the index check and both targets."""

import json

import pytest

from mlx_model_doctor.shard_layout import (
    ShardLayoutKind,
    classify_shard_layout,
    parse_index_weight_map,
    resolve_shard_path,
    root_index_path,
)

ROOT_INDEX = "model.safetensors.index.json"
OLD = {f"t{i}": f"model-0000{i}-of-00003.safetensors" for i in range(1, 4)}


@pytest.mark.parametrize(
    ("files", "weight_map", "index_path", "kind", "absent", "shipped"),
    [
        pytest.param(
            # Bug caught: treating a fully present index as stale or missing.
            [ROOT_INDEX, "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
            {
                "a": "model-00002-of-00002.safetensors",
                "b": "model-00001-of-00002.safetensors",
            },
            ROOT_INDEX,
            ShardLayoutKind.CONSISTENT,
            (),
            (),
            id="consistent",
        ),
        pytest.param(
            # Bug caught: a stale index whose shards are all gone while others ship
            # being classified as missing weights (the 0082 false fail).
            [
                ROOT_INDEX,
                "model-00002-of-00002.safetensors",
                "model-00001-of-00002.safetensors",
                "nested/model.safetensors",
            ],
            OLD,
            ROOT_INDEX,
            ShardLayoutKind.STALE_ALL_ABSENT_WITH_SHIPPED,
            (
                "model-00001-of-00003.safetensors",
                "model-00002-of-00003.safetensors",
                "model-00003-of-00003.safetensors",
            ),
            ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
            id="stale-all-absent-with-shipped",
        ),
        pytest.param(
            # Bug caught: calling a partly missing index stale because some other shard
            # ships; weights really are missing.
            [
                ROOT_INDEX,
                "model-00001-of-00003.safetensors",
                "model-00003-of-00003.safetensors",
                "model-extra.safetensors",
            ],
            OLD,
            ROOT_INDEX,
            ShardLayoutKind.PARTIAL_MISSING,
            ("model-00002-of-00003.safetensors",),
            ("model-extra.safetensors",),
            id="partial-missing",
        ),
        pytest.param(
            # Bug caught: an all-absent index with nothing else shipped classified stale.
            # Only a nested shard ships, which MLX loaders never glob.
            [ROOT_INDEX, "config.json", "vae/model.safetensors"],
            OLD,
            ROOT_INDEX,
            ShardLayoutKind.ALL_MISSING,
            (
                "model-00001-of-00003.safetensors",
                "model-00002-of-00003.safetensors",
                "model-00003-of-00003.safetensors",
            ),
            (),
            id="all-absent-nothing-shipped",
        ),
        pytest.param(
            # Bug caught: inventing an index when none exists; shipped must still list
            # the top-level shards (and only those).
            ["model.safetensors", "text_encoder/0.safetensors"],
            None,
            None,
            ShardLayoutKind.NO_INDEX,
            (),
            ("model.safetensors",),
            id="no-index",
        ),
        pytest.param(
            # Bug caught: an index whose weight_map names no shard treated as consistent.
            ["model.safetensors", ROOT_INDEX],
            {},
            ROOT_INDEX,
            ShardLayoutKind.NO_INDEX,
            (),
            ("model.safetensors",),
            id="empty-weight-map-is-no-usable-index",
        ),
    ],
)
def test_classify_shard_layout_table(
    files: list[str],
    weight_map: dict[str, str] | None,
    index_path: str | None,
    kind: ShardLayoutKind,
    absent: tuple[str, ...],
    shipped: tuple[str, ...],
) -> None:
    layout = classify_shard_layout(files, weight_map, index_path=index_path)

    assert layout.kind is kind
    assert layout.absent == absent
    assert layout.shipped == shipped


def test_classify_sorts_named_shards_whatever_the_weight_map_order() -> None:
    # Bug caught: absent shards reported in weight_map (or set) order, so the shard a
    # message names changes from run to run.
    weight_map = {
        "z": "model-00003-of-00003.safetensors",
        "a": "model-00001-of-00003.safetensors",
        "m": "model-00002-of-00003.safetensors",
    }

    layout = classify_shard_layout([ROOT_INDEX], weight_map, index_path=ROOT_INDEX)

    assert layout.named == (
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    )
    assert layout.absent == layout.named


def test_classify_nested_index_resolves_relative_to_its_directory() -> None:
    # Bug caught: comparing a nested index's raw weight_map value ("0.safetensors")
    # against the listing instead of its resolved path ("text_encoder/0.safetensors").
    layout = classify_shard_layout(
        ["model.safetensors", "text_encoder/0.safetensors", "text_encoder/" + ROOT_INDEX],
        {"enc.w": "0.safetensors"},
        index_path="text_encoder/" + ROOT_INDEX,
    )

    assert layout.kind is ShardLayoutKind.CONSISTENT
    assert layout.named == ("text_encoder/0.safetensors",)


def test_classify_nested_index_with_absent_shards_is_never_stale() -> None:
    # Bug caught: a nested component index whose shards are gone being excused as
    # stale because root shards ship; MLX loaders only glob the repo root.
    layout = classify_shard_layout(
        ["model.safetensors", "text_encoder/" + ROOT_INDEX],
        {"enc.w": "0.safetensors"},
        index_path="text_encoder/" + ROOT_INDEX,
    )

    assert layout.kind is ShardLayoutKind.ALL_MISSING
    assert layout.absent == ("text_encoder/0.safetensors",)


def test_missing_weights_is_true_only_for_the_missing_kinds() -> None:
    # Bug caught: the stale kind counted as missing weights (back to the false fail),
    # or the partial kind not counted (a real loss of weights excused).
    kinds = {kind: kind.missing_weights for kind in ShardLayoutKind}

    assert kinds == {
        ShardLayoutKind.NO_INDEX: False,
        ShardLayoutKind.CONSISTENT: False,
        ShardLayoutKind.STALE_ALL_ABSENT_WITH_SHIPPED: False,
        ShardLayoutKind.PARTIAL_MISSING: True,
        ShardLayoutKind.ALL_MISSING: True,
    }


def test_absent_message_names_index_and_sorted_shards() -> None:
    # Bug caught: a header error that names whichever shard failed first, or no shard.
    layout = classify_shard_layout(
        [ROOT_INDEX, "model-00001-of-00003.safetensors"], OLD, index_path=ROOT_INDEX
    )

    assert layout.absent_message() == (
        "safetensors index model.safetensors.index.json names 2 shards absent from the "
        "repository: model-00002-of-00003.safetensors, model-00003-of-00003.safetensors"
    )


def test_root_index_prefers_the_standard_name() -> None:
    # Bug caught: picking an alphabetically earlier non-standard root index over the
    # model.safetensors.index.json that loaders and huggingface_hub read.
    files = ["adapter.safetensors.index.json", ROOT_INDEX, "config.json"]

    assert root_index_path(files) == ROOT_INDEX


def test_root_index_falls_back_to_first_sorted_top_level_index() -> None:
    # Bug caught: a non-standard root index ignored, or the fallback order unstable.
    files = ["b.safetensors.index.json", "a.safetensors.index.json"]

    assert root_index_path(files) == "a.safetensors.index.json"


def test_root_index_ignores_nested_component_indexes() -> None:
    # Bug caught (0079): a nested component index treated as the repo's root index.
    files = ["model.safetensors", "text_encoder/" + ROOT_INDEX]

    assert root_index_path(files) is None


def test_parse_index_weight_map_keeps_string_values() -> None:
    # Bug caught: non-string weight_map values leaking into the shard set.
    text = json.dumps({"weight_map": {"a": "x.safetensors", "b": 3}})

    assert parse_index_weight_map(text) == {"a": "x.safetensors"}


@pytest.mark.parametrize(
    "text",
    ["{not json", "[]", json.dumps({"metadata": {}}), json.dumps({"weight_map": []})],
    ids=["invalid-json", "not-an-object", "no-weight-map", "weight-map-not-object"],
)
def test_parse_index_weight_map_returns_none_when_unusable(text: str) -> None:
    # Bug caught: an unusable index raising out of a target instead of reading as absent.
    assert parse_index_weight_map(text) is None


@pytest.mark.parametrize(
    "shipped_file",
    ["consolidated.safetensors", "weights.safetensors"],
    ids=["only-consolidated", "only-non-model-name"],
)
def test_all_absent_index_with_only_loader_ineligible_shards_is_all_missing(
    shipped_file: str,
) -> None:
    # Bug caught: counting any top-level *.safetensors as shipped, so a repo neither MLX
    # loader would load (mlx-lm globs model*.safetensors; mlx-vlm skips
    # consolidated.safetensors) is excused as stale instead of failing.
    layout = classify_shard_layout([ROOT_INDEX, shipped_file], OLD, index_path=ROOT_INDEX)

    assert layout.kind is ShardLayoutKind.ALL_MISSING
    assert layout.shipped == ()


def test_mistral_shaped_stale_index_counts_only_loader_eligible_shards() -> None:
    # Bug caught: the eligibility filter dropping real model-*-of-* shards (the
    # Mistral-Small layout turns back into a fail), or counting consolidated.safetensors
    # in the warn message and the HF per-file reads.
    named = {f"t{i}": f"model-{i:05d}-of-00010.safetensors" for i in range(1, 11)}
    shipped = [f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5)]

    layout = classify_shard_layout(
        [ROOT_INDEX, "consolidated.safetensors", *reversed(shipped)], named, index_path=ROOT_INDEX
    )

    assert layout.kind is ShardLayoutKind.STALE_ALL_ABSENT_WITH_SHIPPED
    assert layout.shipped == tuple(shipped)
    assert len(layout.absent) == 10


def test_resolve_shard_path_root_and_nested() -> None:
    # Bug caught: nested values left unresolved, or root values given a "./" prefix.
    assert resolve_shard_path(ROOT_INDEX, "a.safetensors") == "a.safetensors"
    assert resolve_shard_path("te/" + ROOT_INDEX, "a.safetensors") == "te/a.safetensors"


def test_resolve_shard_path_normalizes_dot_segments() -> None:
    # Bug caught (C1): "./x" kept verbatim, so the classifier calls a shard absent that
    # target.exists (which resolves the path) finds.
    assert resolve_shard_path(ROOT_INDEX, "./model-00002.safetensors") == "model-00002.safetensors"
    assert resolve_shard_path(ROOT_INDEX, "a/../b.safetensors") == "b.safetensors"
    assert resolve_shard_path("te/" + ROOT_INDEX, "./0.safetensors") == "te/0.safetensors"


def test_resolve_shard_path_keeps_a_leading_parent_segment() -> None:
    # Bug caught: normalizing "../outside" away, so an out-of-repo reference looks valid.
    assert resolve_shard_path(ROOT_INDEX, "../outside.safetensors") == "../outside.safetensors"


_CASE_FILES = [ROOT_INDEX, "model-00001-of-00001.safetensors"]
_CASE_MAP = {"a": "Model-00001-of-00001.safetensors"}


def _case_insensitive(path: str) -> bool:
    return path.lower() in {name.lower() for name in _CASE_FILES}


def test_classify_presence_predicate_decides_which_named_shards_are_absent() -> None:
    # Bug caught (C1): the classifier ignoring the target's presence check, so a name
    # that exists on a case-insensitive filesystem is still called absent.
    layout = classify_shard_layout(
        _CASE_FILES, _CASE_MAP, index_path=ROOT_INDEX, present=_case_insensitive
    )

    assert layout.kind is ShardLayoutKind.CONSISTENT
    assert layout.absent == ()


def test_classify_presence_defaults_to_listing_membership() -> None:
    # Bug caught: a default predicate that folds case (or resolves paths), which would
    # make the Hugging Face listing, which is exact, look case-insensitive.
    layout = classify_shard_layout(_CASE_FILES, _CASE_MAP, index_path=ROOT_INDEX)

    assert layout.absent == ("Model-00001-of-00001.safetensors",)
    assert layout.kind is ShardLayoutKind.STALE_ALL_ABSENT_WITH_SHIPPED


def test_classify_presence_predicate_replaces_listing_membership() -> None:
    # Bug caught: the predicate OR-ed with the listing, so a listed shard the target
    # cannot read (a traversal error treated as absent) still counts as present.
    layout = classify_shard_layout(
        [ROOT_INDEX, "model.safetensors"],
        {"a": "model.safetensors"},
        index_path=ROOT_INDEX,
        present=lambda _path: False,
    )

    assert layout.absent == ("model.safetensors",)
    assert layout.kind is ShardLayoutKind.ALL_MISSING


def _absent_layout(count: int) -> str:
    weight_map = {f"t{i}": f"m{i:02d}.safetensors" for i in range(count)}
    return classify_shard_layout([ROOT_INDEX], weight_map, index_path=ROOT_INDEX).absent_message()


def test_absent_message_lists_at_most_eight_names_then_a_count() -> None:
    # Bug caught (S2): every absent name echoed, so a huge stale index makes a
    # multi-megabyte message.
    assert _absent_layout(20) == (
        "safetensors index model.safetensors.index.json names 20 shards absent from the "
        "repository: m00.safetensors, m01.safetensors, m02.safetensors, m03.safetensors, "
        "m04.safetensors, m05.safetensors, m06.safetensors, m07.safetensors, and 12 more"
    )


def test_absent_message_cap_boundary_eight_and_nine() -> None:
    # Bug caught: an off-by-one in the cap (7 or 9 names shown, or "and 0 more").
    assert _absent_layout(8).endswith("m06.safetensors, m07.safetensors")
    assert _absent_layout(9).endswith("m07.safetensors, and 1 more")


def test_absent_message_replaces_control_characters_but_details_keep_raw_names() -> None:
    # Bug caught (S2): a "\n" in a repo-controlled shard name faking a line in the text
    # report, or the raw name rewritten in the layout that feeds details.
    weight_map = {"a": "evil\nFAIL x.safetensors", "b": "del\x7f\x00.safetensors"}
    layout = classify_shard_layout([ROOT_INDEX], weight_map, index_path=ROOT_INDEX)

    assert layout.absent_message() == (
        "safetensors index model.safetensors.index.json names 2 shards absent from the "
        "repository: del??.safetensors, evil?FAIL x.safetensors"
    )
    assert layout.absent == ("del\x7f\x00.safetensors", "evil\nFAIL x.safetensors")
