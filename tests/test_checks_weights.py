import json

from mlx_model_doctor.checks.weights import TiedEmbeddingCheck, WeightParamCountCheck
from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.safetensors_header import FileHeader, SafetensorsHeader, TensorEntry
from tests.fakes import FakeTarget, check_options


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
    assert result.details["total_parameter_count"] == 0


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
