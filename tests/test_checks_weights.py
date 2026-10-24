import json
from pathlib import Path

from mlx_model_doctor.checks.safetensors import SafetensorsIndexCheck
from mlx_model_doctor.checks.weights import TiedEmbeddingCheck, WeightParamCountCheck
from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.safetensors_header import FileHeader, SafetensorsHeader, TensorEntry
from mlx_model_doctor.targets import LocalTarget
from tests.fakes import FakeTarget, check_options, shard_repo, write_repo


def _entry(count: int = 4) -> TensorEntry:
    return TensorEntry(
        dtype="BF16", shape=(count,), data_offsets=(0, count * 2), stored_element_count=count
    )


def _header(*, tensors: dict[str, TensorEntry], weight_map: dict[str, str]) -> SafetensorsHeader:
    fh = FileHeader(
        filename="model.safetensors", tensors=tensors, metadata={}, header_length=10, file_size=None
    )
    return SafetensorsHeader(
        files=(fh,),
        weight_map=weight_map,
        sharded=False,
        stored_count_by_dtype={"BF16": sum(e.stored_element_count for e in tensors.values())},
    )


def _run(header: SafetensorsHeader | None) -> object:
    return WeightParamCountCheck().run(
        CheckContext(
            target=FakeTarget(files={}, _safetensors_header=header), options=check_options()
        )
    )


def test_param_count_pass_when_map_complete_and_nonzero() -> None:
    header = _header(tensors={"w": _entry()}, weight_map={"w": "model.safetensors"})
    assert _run(header).status == "pass"


def test_weight_param_count_detail_key_is_stored_element_count() -> None:
    header = _header(tensors={"w": _entry()}, weight_map={"w": "model.safetensors"})
    result = _run(header)
    assert result.details["stored_element_count"] == 4
    assert "total_parameter_count" not in result.details


def test_param_count_warn_on_declared_but_missing_tensor() -> None:
    header = _header(
        tensors={"w": _entry()}, weight_map={"w": "model.safetensors", "ghost": "model.safetensors"}
    )
    result = _run(header)
    assert result.status == "warn"
    assert result.details["missing_tensors"] == ("ghost",)


def test_param_count_warn_on_zero_params() -> None:
    fh = FileHeader(
        filename="model.safetensors", tensors={}, metadata={}, header_length=10, file_size=None
    )
    header = SafetensorsHeader(files=(fh,), weight_map={}, sharded=False, stored_count_by_dtype={})
    result = _run(header)
    assert result.status == "warn"
    assert result.details["empty_or_zero_param_files"] == ("model.safetensors",)


def test_param_count_skip_without_header() -> None:
    assert _run(None).status == "skip"


def test_param_count_warn_on_nonempty_files_but_zero_total() -> None:
    # File HAS a tensor, but stored_count_by_dtype sums to 0 -> the total==0 clause must fire.
    base = _header(tensors={"w": _entry()}, weight_map={"w": "model.safetensors"})
    header = SafetensorsHeader(
        files=base.files,
        weight_map=base.weight_map,
        sharded=False,
        stored_count_by_dtype={"BF16": 0},
    )
    result = _run(header)
    assert result.status == "warn"
    assert result.details["stored_element_count"] == 0
    assert "total_parameter_count" not in result.details


def _ctx(header: SafetensorsHeader | None, config: object) -> CheckContext:
    files = {"config.json": json.dumps(config).encode()} if config is not None else {}
    target = FakeTarget(files=files, _safetensors_header=header)
    return CheckContext(target=target, options=check_options())


def _names_header(names: set[str]) -> SafetensorsHeader:
    tensors = {n: _entry() for n in names}
    fh = FileHeader(
        filename="model.safetensors", tensors=tensors, metadata={}, header_length=10, file_size=None
    )
    return SafetensorsHeader(
        files=(fh,),
        weight_map=dict.fromkeys(names, "model.safetensors"),
        sharded=False,
        stored_count_by_dtype={},
    )


def test_tied_pass_when_declared_tied_and_only_input_stored() -> None:
    header = _names_header({"model.embed_tokens.weight"})
    result = TiedEmbeddingCheck().run(_ctx(header, {"tie_word_embeddings": True}))
    assert result.status == "pass"


def test_tied_warn_when_declared_tied_but_both_stored() -> None:
    header = _names_header({"model.embed_tokens.weight", "lm_head.weight"})
    result = TiedEmbeddingCheck().run(_ctx(header, {"tie_word_embeddings": True}))
    assert result.status == "warn"
    assert result.details["stored_both_distinct"] is True


def test_tied_warn_when_untied_but_no_output_head() -> None:
    header = _names_header({"model.embed_tokens.weight"})
    result = TiedEmbeddingCheck().run(_ctx(header, {"tie_word_embeddings": False}))
    assert result.status == "warn"
    assert result.details["missing_output_head"] is True
    assert "not enabled" in result.message
    assert "not set" not in result.message


def test_tied_skip_without_config_or_header() -> None:
    assert TiedEmbeddingCheck().run(_ctx(None, {"tie_word_embeddings": True})).status == "skip"
    header = _names_header({"model.embed_tokens.weight"})
    assert TiedEmbeddingCheck().run(_ctx(header, None)).status == "skip"


def test_tied_skip_when_no_recognized_embedding_tensor() -> None:
    # A header with tensors but none matching the embedding/head alias lists -> skip.
    header = _names_header({"some.random.weight"})
    result = TiedEmbeddingCheck().run(_ctx(header, {"tie_word_embeddings": True}))
    assert result.status == "skip"


def test_tied_treats_nonbool_truthy_as_untied() -> None:
    # tie_word_embeddings: 1 (not the bool True) must be read as "untied", so a
    # missing output head warns instead of silently passing.
    header = _names_header({"model.embed_tokens.weight"})
    result = TiedEmbeddingCheck().run(_ctx(header, {"tie_word_embeddings": 1}))
    assert result.status == "warn"
    assert result.details["missing_output_head"] is True


def test_tied_no_warn_when_flag_absent_and_head_missing() -> None:
    # An absent tie_word_embeddings key with an input embedding and no output head is a
    # tie-by-default model (e.g. Gemma): consistent, not the "declared untied, no head"
    # inconsistency. The check must not warn merely because the key is missing -- that
    # false-positive fired on valid, popular repos like mlx-community/gemma-2-9b-it-4bit.
    header = _names_header({"model.embed_tokens.weight"})
    result = TiedEmbeddingCheck().run(_ctx(header, {"model_type": "gemma2"}))
    assert result.status == "pass"


# --- LocalTarget headers for stale / partial / nested indexes (0079, 0082) ---------------


def _local_ctx(tmp_path: Path, layout: str) -> CheckContext:
    files, shards = shard_repo(layout)
    return CheckContext(
        target=LocalTarget(write_repo(tmp_path / layout, files, shards)), options=check_options()
    )


def test_local_nested_component_index_does_not_warn_param_count(tmp_path: Path) -> None:
    # Bug caught (0079): merging the nested text_encoder index's raw weight_map into the
    # root header, so param_count warns "enc.w" is absent from every shard header.
    result = WeightParamCountCheck().run(_local_ctx(tmp_path, "nested"))

    assert result.status == "pass"
    assert result.details["stored_element_count"] == 6  # root model.safetensors only: 2*3


def test_local_stale_index_param_count_judges_the_shipped_tensors(tmp_path: Path) -> None:
    # Bug caught (0082): keeping the stale weight_map, so param_count warns about tensors
    # old.x/old.y that no shipped shard holds instead of judging the real tensors.
    result = WeightParamCountCheck().run(_local_ctx(tmp_path, "stale"))

    assert result.status == "pass"
    assert result.details["stored_element_count"] == 18  # 2*3 + 5 + 7


def test_local_partial_index_reports_a_header_error_naming_absent_shards(
    tmp_path: Path,
) -> None:
    # Bug caught: the local target silently building a header from the shards that are
    # left, so local and HF disagree on the same partly missing repo.
    ctx = _local_ctx(tmp_path, "partial")

    assert ctx.safetensors_header() is None
    assert ctx.safetensors_header_error() == (
        "safetensors index model.safetensors.index.json names 2 shards absent from the "
        "repository: model-00001-of-00003.safetensors, model-00003-of-00003.safetensors"
    )
    assert WeightParamCountCheck().run(ctx).status == "skip"


def test_local_stale_index_beside_consolidated_counts_only_the_model_shards(
    tmp_path: Path,
) -> None:
    # Bug caught (T1): the local stale path reading every top-level shard, so the 100
    # elements in consolidated.safetensors join the count.
    ctx = _local_ctx(tmp_path, "stale_with_consolidated")

    header = ctx.safetensors_header()
    assert header is not None
    assert [fh.filename for fh in header.files] == [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]
    assert WeightParamCountCheck().run(ctx).details["stored_element_count"] == 18


def _many_shards_ctx(tmp_path: Path, count: int) -> CheckContext:
    files = {
        "config.json": b"{}",
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {"x": "model-gone.safetensors"}}
        ).encode(),
    }
    shards = {
        f"model-{i:05d}-of-{count:05d}.safetensors": {f"t{i}": ("F32", (1,))} for i in range(count)
    }
    return CheckContext(
        target=LocalTarget(write_repo(tmp_path / str(count), files, shards)),
        options=check_options(),
    )


def test_local_stale_index_with_513_shipped_shards_is_a_header_error(tmp_path: Path) -> None:
    # Bug caught (S1): the local stale path reading any number of shard headers, unlike
    # the capped Hugging Face path.
    ctx = _many_shards_ctx(tmp_path, 513)

    assert ctx.safetensors_header() is None
    assert ctx.safetensors_header_error() == (
        "safetensors index model.safetensors.index.json is stale and the repository ships "
        "513 shards, more than the 512 whose headers are read"
    )


def test_local_stale_index_with_512_shipped_shards_reads_them(tmp_path: Path) -> None:
    # Bug caught: the local cap off by one (512 shards refused).
    header = _many_shards_ctx(tmp_path, 512).safetensors_header()

    assert header is not None
    assert len(header.files) == 512


def test_local_dot_prefixed_index_value_reads_both_shards(tmp_path: Path) -> None:
    # Bug caught (C1): "./model-00002..." read as absent, so the local header raises a
    # partial-missing error on a repo whose index check passes.
    ctx = _local_ctx(tmp_path, "dot_prefixed")

    header = ctx.safetensors_header()
    assert header is not None
    assert [fh.filename for fh in header.files] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert header.weight_map["b"] == "model-00002-of-00002.safetensors"


def test_local_index_value_escaping_the_root_stays_invalid(tmp_path: Path) -> None:
    # Bug caught (C1): the local presence predicate letting the traversal TargetError
    # escape (no header at all) or following "../outside.safetensors" to a real file
    # beside the model, instead of treating it as absent and invalid.
    write_repo(tmp_path, {}, {"outside.safetensors": {"o": ("F32", (3,))}})
    files = {
        "config.json": b"{}",
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {"w": "../outside.safetensors"}}
        ).encode(),
    }
    root = write_repo(tmp_path / "repo", files, {"model.safetensors": {"a": ("F32", (2,))}})
    ctx = CheckContext(target=LocalTarget(root), options=check_options())

    index = SafetensorsIndexCheck().run(ctx)
    header = ctx.safetensors_header()

    assert index.status == "fail"
    assert index.details["invalid_shards"] == ("../outside.safetensors",)
    assert index.details["shard_layout"] == "all_missing"
    assert header is not None
    assert [fh.filename for fh in header.files] == ["model.safetensors"]


def test_local_header_agrees_with_the_index_check_on_a_path_only_exists_resolves(
    tmp_path: Path,
) -> None:
    # Bug caught (C1): LocalTarget classifying by listing membership instead of its own
    # exists(). The listing does not descend a directory symlink, but exists() resolves
    # it (as it resolves case on APFS), so the index check passes while the header
    # raises "absent" and the offsets check fails.
    files = {
        "config.json": b"{}",
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {"a": "alias/model-00001-of-00001.safetensors"}}
        ).encode(),
    }
    root = write_repo(
        tmp_path / "repo", files, {"model-00001-of-00001.safetensors": {"a": ("F32", (4,))}}
    )
    (root / "alias").symlink_to(root, target_is_directory=True)
    target = LocalTarget(root)
    assert "alias/model-00001-of-00001.safetensors" not in target.list_files()
    ctx = CheckContext(target=target, options=check_options())

    assert SafetensorsIndexCheck().run(ctx).status == "pass"
    header = ctx.safetensors_header()
    assert header is not None
    assert [fh.filename for fh in header.files] == ["alias/model-00001-of-00001.safetensors"]
