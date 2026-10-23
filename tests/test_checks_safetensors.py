import json

import pytest

from mlx_model_doctor.checks.safetensors import SafetensorsIndexCheck, SafetensorsOffsetScanCheck
from mlx_model_doctor.context import _MAX_METADATA_BYTES, CheckContext
from mlx_model_doctor.errors import TargetError
from mlx_model_doctor.safetensors_header import FileHeader, SafetensorsHeader, TensorEntry
from tests.fakes import FakeTarget, check_options, context_for_files


def test_safetensors_index_check_skips_when_index_missing() -> None:
    result = SafetensorsIndexCheck().run(context_for_files({"model.safetensors": b"weights"}))

    assert result.status == "skip"
    assert result.severity == "info"
    assert "index" in result.message


def test_safetensors_index_check_fails_when_referenced_shard_is_missing() -> None:
    index = {"weight_map": {"model.embed_tokens.weight": "missing-00001.safetensors"}}

    result = SafetensorsIndexCheck().run(
        context_for_files({"model.safetensors.index.json": json.dumps(index).encode()})
    )

    assert result.status == "fail"
    assert result.severity == "high"
    assert "missing-00001.safetensors" in result.message
    assert result.remediation is not None


def test_safetensors_index_check_passes_when_referenced_shard_exists() -> None:
    index = {"weight_map": {"model.embed_tokens.weight": "model-00001-of-00001.safetensors"}}
    target = NoShardReadTarget(
        files={
            "model.safetensors.index.json": json.dumps(index).encode(),
            "model-00001-of-00001.safetensors": b"do-not-read",
        }
    )

    result = SafetensorsIndexCheck().run(CheckContext(target=target, options=check_options()))

    assert result.status == "pass"
    assert result.severity == "info"
    assert result.details["shards"] == ("model-00001-of-00001.safetensors",)


def test_safetensors_index_check_resolves_nested_index_shard_paths() -> None:
    # A component index in a subdirectory (e.g. a diffusion text_encoder) names its shards
    # relative to the index file's own directory: `0.safetensors` here means
    # `text_encoder/0.safetensors`, not a repo-root file. Resolving it against the root instead
    # is the bug this catches -- it makes a valid repo FAIL as "missing shard 0.safetensors".
    index = {"weight_map": {"encoder.layer.weight": "0.safetensors"}}

    result = SafetensorsIndexCheck().run(
        context_for_files(
            {
                "text_encoder/model.safetensors.index.json": json.dumps(index).encode(),
                "text_encoder/0.safetensors": b"weights",
            }
        )
    )

    assert result.status == "pass"
    assert result.details["shards"] == ("text_encoder/0.safetensors",)


def test_safetensors_index_check_fails_when_nested_shard_missing_at_resolved_path() -> None:
    # A real miss must still fail: the referenced shard resolves to text_encoder/1.safetensors,
    # which is absent (only 0.safetensors exists). The failure is reported at the resolved path,
    # not the bare weight_map value -- so the fix cannot mask a genuinely broken nested index.
    index = {"weight_map": {"encoder.layer.weight": "1.safetensors"}}

    result = SafetensorsIndexCheck().run(
        context_for_files(
            {
                "text_encoder/model.safetensors.index.json": json.dumps(index).encode(),
                "text_encoder/0.safetensors": b"weights",
            }
        )
    )

    assert result.status == "fail"
    assert result.severity == "high"
    assert "text_encoder/1.safetensors" in result.message
    assert result.details["missing_shards"] == ("text_encoder/1.safetensors",)


def test_safetensors_index_check_sorts_discovered_indexes_and_shards() -> None:
    first_index = {"weight_map": {"a.weight": "z-shard.safetensors"}}
    second_index = {"weight_map": {"b.weight": "a-shard.safetensors"}}
    target = UnsortedListTarget(
        files={
            "z-model.safetensors.index.json": json.dumps(first_index).encode(),
            "a-model.safetensors.index.json": json.dumps(second_index).encode(),
            "z-shard.safetensors": b"z",
            "a-shard.safetensors": b"a",
        },
        listed_paths=(
            "z-model.safetensors.index.json",
            "z-shard.safetensors",
            "a-model.safetensors.index.json",
            "a-shard.safetensors",
        ),
    )

    result = SafetensorsIndexCheck().run(CheckContext(target=target, options=check_options()))

    assert result.status == "pass"
    assert result.details["index_paths"] == (
        "a-model.safetensors.index.json",
        "z-model.safetensors.index.json",
    )
    assert result.details["shards"] == ("a-shard.safetensors", "z-shard.safetensors")


def test_safetensors_index_check_warns_when_listed_index_disappears() -> None:
    target = DisappearingIndexTarget(files={"model.safetensors.index.json": b'{"weight_map":{}}'})

    result = SafetensorsIndexCheck().run(CheckContext(target=target, options=check_options()))

    assert result.status == "warn"
    assert result.severity == "medium"
    assert "model.safetensors.index.json" in result.message
    assert result.details["index_path"] == "model.safetensors.index.json"


def test_safetensors_index_check_propagates_hf_index_read_error() -> None:
    target = HfIndexReadErrorTarget(
        files={"model.safetensors.index.json": b'{"weight_map":{}}'},
        _source="hf",
    )

    with pytest.raises(TargetError, match="index download failed") as exc_info:
        SafetensorsIndexCheck().run(CheckContext(target=target, options=check_options()))

    assert exc_info.value.source == "hf"


def test_safetensors_index_check_discovers_non_default_index_name() -> None:
    index = {"weight_map": {"adapter.layer.weight": "adapter_model.safetensors"}}

    result = SafetensorsIndexCheck().run(
        context_for_files(
            {
                "adapter_model.safetensors.index.json": json.dumps(index).encode(),
                "adapter_model.safetensors": b"adapter weights",
            }
        )
    )

    assert result.status == "pass"
    assert result.severity == "info"
    assert result.details["index_path"] == "adapter_model.safetensors.index.json"
    assert result.details["shards"] == ("adapter_model.safetensors",)


def test_safetensors_index_check_fails_when_shard_reference_is_invalid() -> None:
    index = {"weight_map": {"model.embed_tokens.weight": "../outside.safetensors"}}
    target = InvalidShardPathTarget(
        files={"model.safetensors.index.json": json.dumps(index).encode()}
    )

    result = SafetensorsIndexCheck().run(CheckContext(target=target, options=check_options()))

    assert result.status == "fail"
    assert result.severity == "high"
    assert "../outside.safetensors" in result.message
    assert result.remediation is not None
    assert result.details["missing_shards"] == ("../outside.safetensors",)


def test_safetensors_index_check_fails_on_invalid_json() -> None:
    result = SafetensorsIndexCheck().run(
        context_for_files({"model.safetensors.index.json": b"{not-json"})
    )

    assert result.status == "fail"
    assert result.severity == "high"
    assert "invalid JSON" in result.message
    assert result.remediation is not None


def test_safetensors_index_check_warns_when_index_is_not_an_object() -> None:
    result = SafetensorsIndexCheck().run(context_for_files({"model.safetensors.index.json": b"[]"}))

    assert result.status == "warn"
    assert result.severity == "medium"
    assert "object" in result.message
    assert result.remediation is not None


def test_safetensors_index_check_warns_when_weight_map_is_missing() -> None:
    result = SafetensorsIndexCheck().run(
        context_for_files({"model.safetensors.index.json": b'{"metadata":{}}'})
    )

    assert result.status == "warn"
    assert result.severity == "medium"
    assert "weight_map" in result.message
    assert result.remediation is not None


class NoShardReadTarget(FakeTarget):
    def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        if path.endswith(".safetensors"):
            raise AssertionError("SafetensorsIndexCheck must not read shard contents")
        return super().read_bytes(path, max_bytes=max_bytes)


class UnsortedListTarget(FakeTarget):
    listed_paths: tuple[str, ...]

    def __init__(self, *, files: dict[str, bytes], listed_paths: tuple[str, ...]) -> None:
        super().__init__(files=files)
        self.listed_paths = listed_paths

    def list_files(self) -> tuple[str, ...]:
        return self.listed_paths


class DisappearingIndexTarget(FakeTarget):
    def read_text(self, path: str, *, max_bytes: int | None = None) -> str:
        if path == "model.safetensors.index.json":
            raise FileNotFoundError(path)
        return super().read_text(path, max_bytes=max_bytes)


class HfIndexReadErrorTarget(FakeTarget):
    def read_text(self, path: str, *, max_bytes: int | None = None) -> str:
        if path == "model.safetensors.index.json":
            raise TargetError("index download failed", target=path, source=self.source)
        return super().read_text(path, max_bytes=max_bytes)


class InvalidShardPathTarget(FakeTarget):
    def exists(self, path: str) -> bool:
        if path == "../outside.safetensors":
            raise TargetError("outside model root", target=path, source=self.source)
        return super().exists(path)


def test_safetensors_index_check_handles_oversized_index_without_reading() -> None:
    oversized = b"{}" + b" " * (_MAX_METADATA_BYTES + 1)
    target = OversizedReadAssertTarget(files={"model.safetensors.index.json": oversized})
    result = SafetensorsIndexCheck().run(CheckContext(target=target, options=check_options()))
    assert result.status == "warn"
    assert "too large" in result.message


class OversizedReadAssertTarget(FakeTarget):
    """Raises if read_text is called on an oversized file (guard must fire before read)."""

    def read_text(self, path: str, *, max_bytes: int | None = None) -> str:
        if self.size(path) is not None and self.size(path) > _MAX_METADATA_BYTES:
            raise AssertionError(f"SafetensorsIndexCheck must not read oversized file {path!r}")
        return super().read_text(path, max_bytes=max_bytes)


# ---------------------------------------------------------------------------
# SafetensorsOffsetScanCheck tests
# ---------------------------------------------------------------------------


def _entry(dtype: str, begin: int, end: int) -> TensorEntry:
    return TensorEntry(
        dtype=dtype,
        shape=(end - begin,),
        data_offsets=(begin, end),
        stored_element_count=end - begin,
    )


def _offsets_header(
    tensors: dict[str, TensorEntry], *, file_size: int | None, header_length: int | None = 10
) -> SafetensorsHeader:
    fh = FileHeader(
        filename="model.safetensors",
        tensors=tensors,
        metadata={},
        header_length=header_length,
        file_size=file_size,
    )
    return SafetensorsHeader(
        files=(fh,),
        weight_map=dict.fromkeys(tensors, "model.safetensors"),
        sharded=False,
        stored_count_by_dtype={},
    )


def _run_offsets(header: SafetensorsHeader | None):
    return SafetensorsOffsetScanCheck().run(
        CheckContext(
            target=FakeTarget(files={}, _safetensors_header=header), options=check_options()
        )
    )


def test_offsets_pass_for_contiguous_in_bounds_tensors() -> None:
    # file_size = 8 + header_length(10) + data(8) = 26
    header = _offsets_header({"a": _entry("F32", 0, 4), "b": _entry("F32", 4, 8)}, file_size=26)
    assert _run_offsets(header).status == "pass"


def test_offsets_fail_on_overlap() -> None:
    header = _offsets_header({"a": _entry("F32", 0, 8), "b": _entry("F32", 4, 12)}, file_size=30)
    result = _run_offsets(header)
    assert result.status == "fail"
    assert result.severity == "high"
    assert result.details["overlapping"]


def test_offsets_fail_on_out_of_bounds_when_size_known() -> None:
    # data_section_length = 26 - 8 - 10 = 8, but tensor ends at 12
    header = _offsets_header({"a": _entry("F32", 0, 12)}, file_size=26)
    result = _run_offsets(header)
    assert result.status == "fail"
    assert result.severity == "high"
    assert result.details["out_of_bounds"]


def test_offsets_pass_when_only_size_unavailable() -> None:
    # On the HF path header_length is unknown -> data_section_length is None.
    # The upper bound can't be checked, but overlap/ordering are clean -> pass, not warn.
    header = _offsets_header({"a": _entry("F32", 0, 12)}, file_size=None, header_length=None)
    result = _run_offsets(header)
    assert result.status == "pass"
    assert result.details["upper_bound_checked"] is False


def test_offsets_warn_on_unknown_dtype() -> None:
    header = _offsets_header({"a": _entry("WEIRD8", 0, 4)}, file_size=22)
    result = _run_offsets(header)
    assert result.status == "warn"
    assert result.details["unknown_dtypes"]


def test_offsets_skip_without_header() -> None:
    assert _run_offsets(None).status == "skip"


def test_offsets_allow_zero_length_tensor() -> None:
    # begin == end is a legitimate empty tensor, must not be flagged out-of-bounds.
    header = _offsets_header({"a": _entry("F32", 0, 0)}, file_size=18)
    assert _run_offsets(header).status == "pass"


def test_offsets_fail_on_inverted_offset() -> None:
    # end < begin is a corrupt offset, independent of the data-section bound.
    header = _offsets_header({"a": _entry("F32", 8, 4)}, file_size=40)
    result = _run_offsets(header)
    assert result.status == "fail"
    assert result.severity == "high"
    assert result.details["out_of_bounds"]


def test_offsets_warn_on_non_contiguous_gap() -> None:
    # a ends at 4, b begins at 8 -> a benign gap; clean dtypes/bounds -> warn, not fail.
    header = _offsets_header({"a": _entry("F32", 0, 4), "b": _entry("F32", 8, 12)}, file_size=30)
    result = _run_offsets(header)
    assert result.status == "warn"
    assert result.details["gaps"]


def test_offsets_fail_when_header_present_but_unparseable() -> None:
    from mlx_model_doctor.safetensors_header import SafetensorsHeaderError

    class RaisingHeaderTarget(FakeTarget):
        def safetensors_header(self):
            raise SafetensorsHeaderError("model.safetensors: header is truncated")

    ctx = CheckContext(target=RaisingHeaderTarget(files={}), options=check_options())
    result = SafetensorsOffsetScanCheck().run(ctx)
    assert result.status == "fail"
    assert result.severity == "high"
    assert "could not be read" in result.message
    assert result.details["error"] == "model.safetensors: header is truncated"


def test_offsets_pass_with_empty_tensor_after_real_weight() -> None:
    # An empty tensor serializes as data_offsets=(0,0). Sorted by begin it sits beside
    # the real weight; it must NOT be read as an overlap. Weight-first insertion order
    # is the common real case (big weight before small/empty buffers).
    header = _offsets_header(
        {"model.weight": _entry("F32", 0, 400), "model.empty_buffer": _entry("F32", 0, 0)},
        file_size=8 + 10 + 400,  # header_length=10 default -> data_section_length=400
    )
    result = _run_offsets(header)
    assert result.status == "pass"


def test_offset_scan_reports_leading_gap() -> None:
    # Single tensor begins at offset 8, not 0 -> a leading gap before the data section.
    header = _offsets_header({"w": _entry("F32", 8, 24)}, file_size=None, header_length=None)
    result = _run_offsets(header)
    assert result.status == "warn"
    assert "model.safetensors:w" in result.details["gaps"]


# ---------------------------------------------------------------------------
# Stale vs missing root index (0082)
# ---------------------------------------------------------------------------

_STALE_INDEX = {
    "weight_map": {
        "old.y": "model-00002-of-00002.safetensors",
        "old.x": "model-00001-of-00002.safetensors",
    }
}
_SHIPPED = (
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
)


def _index_ctx(index: dict[str, object], shards: tuple[str, ...]) -> CheckContext:
    files = {"model.safetensors.index.json": json.dumps(index).encode()}
    files.update(dict.fromkeys(shards, b"shard"))
    return context_for_files(files)


def test_stale_index_with_shipped_shards_warns_medium_and_names_the_cause() -> None:
    # Bug caught: the 0082 false fail (FAIL high on a repo MLX loaders load), or a warn
    # that does not say why (no counts, no "stale").
    result = SafetensorsIndexCheck().run(_index_ctx(_STALE_INDEX, _SHIPPED))

    assert result.status == "warn"
    assert result.severity == "medium"
    assert result.message == (
        "Safetensors index model.safetensors.index.json is stale: it names 2 shards that are "
        "absent from the repository, while 3 other shards ship. mlx-lm never reads the index "
        "and loads model*.safetensors; mlx-vlm falls back to the shipped shards when none of "
        "the named ones exist. Loaders that follow the index, and huggingface_hub's "
        "get_safetensors_metadata, fail."
    )
    assert result.remediation == (
        "Regenerate model.safetensors.index.json from the shipped shards, or remove the "
        "stale index."
    )


def test_stale_index_details_list_sorted_absent_and_shipped_shards() -> None:
    # Bug caught: details missing the classification that tells a stale index from
    # missing weights, or the absent and shipped lists swapped.
    result = SafetensorsIndexCheck().run(_index_ctx(_STALE_INDEX, tuple(reversed(_SHIPPED))))

    assert result.details["shard_layout"] == "stale_all_absent_with_shipped"
    assert result.details["missing_shards"] == (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    assert result.details["shipped_shards"] == _SHIPPED
    assert result.details["index_path"] == "model.safetensors.index.json"


def test_partly_missing_index_still_fails_with_the_current_message() -> None:
    # Bug caught: excusing a partly missing index as stale because another shard ships.
    index = {
        "weight_map": {
            "a": "model-00001-of-00003.safetensors",
            "b": "model-00002-of-00003.safetensors",
            "c": "model-00003-of-00003.safetensors",
        }
    }

    result = SafetensorsIndexCheck().run(
        _index_ctx(index, ("model-00002-of-00003.safetensors", "extra.safetensors"))
    )

    assert result.status == "fail"
    assert result.severity == "high"
    assert result.message == (
        "Safetensors index model.safetensors.index.json references invalid or missing shard "
        "model-00001-of-00003.safetensors."
    )
    assert result.details["shard_layout"] == "partial_missing"
    assert result.details["missing_shards"] == (
        "model-00001-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    )


def test_all_absent_index_with_only_nested_shards_fails() -> None:
    # Bug caught: a nested component shard counted as "shipped", excusing an index whose
    # weights are all gone.
    result = SafetensorsIndexCheck().run(_index_ctx(_STALE_INDEX, ("vae/model.safetensors",)))

    assert result.status == "fail"
    assert result.details["shard_layout"] == "all_missing"


def test_all_absent_index_with_only_a_non_model_shard_fails() -> None:
    # Bug caught: a top-level shard mlx-lm would not glob (not model*.safetensors)
    # excusing an index whose named weights are all gone.
    result = SafetensorsIndexCheck().run(_index_ctx(_STALE_INDEX, ("weights.safetensors",)))

    assert result.status == "fail"
    assert result.details["shard_layout"] == "all_missing"


def test_nested_index_with_absent_shards_fails_even_when_root_shards_ship() -> None:
    # Bug caught: the stale excuse applied to a nested component index, which MLX loaders
    # never fall back for.
    files = {
        "model.safetensors": b"root",
        "text_encoder/model.safetensors.index.json": json.dumps(
            {"weight_map": {"enc.w": "0.safetensors"}}
        ).encode(),
    }

    result = SafetensorsIndexCheck().run(context_for_files(files))

    assert result.status == "fail"
    assert result.details["missing_shards"] == ("text_encoder/0.safetensors",)


def test_invalid_shard_reference_is_not_excused_as_stale() -> None:
    # Bug caught: an index naming a path outside the repo treated as merely stale; the
    # reference is invalid, and a loader that follows it may read outside the model.
    index = {"weight_map": {"w": "../outside.safetensors"}}
    files = {"model.safetensors.index.json": json.dumps(index).encode(), "model.safetensors": b"w"}
    target = InvalidShardPathTarget(files=files)

    result = SafetensorsIndexCheck().run(CheckContext(target=target, options=check_options()))

    assert result.status == "fail"
    assert result.details["invalid_shards"] == ("../outside.safetensors",)
    # Bug caught (C2): the fail tagged "stale_all_absent_with_shipped" for a details reader.
    assert result.details["shard_layout"] == "all_missing"


def test_stale_root_warning_does_not_hide_a_later_failing_index() -> None:
    # Bug caught: returning the root index's stale warn at once, so a nested component
    # index (sorted after it) with genuinely missing shards is never reported.
    files = {
        "model.safetensors.index.json": json.dumps(_STALE_INDEX).encode(),
        "model-00001-of-00003.safetensors": b"shard",
        "text_encoder/model.safetensors.index.json": json.dumps(
            {"weight_map": {"enc.w": "0.safetensors"}}
        ).encode(),
    }

    result = SafetensorsIndexCheck().run(context_for_files(files))

    assert result.status == "fail"
    assert result.details["index_path"] == "text_encoder/model.safetensors.index.json"


def test_stale_root_warning_survives_a_passing_nested_index() -> None:
    # Bug caught: a later passing index overwriting the stale warn with a pass.
    files = {
        "model.safetensors.index.json": json.dumps(_STALE_INDEX).encode(),
        "model-00001-of-00003.safetensors": b"shard",
        "text_encoder/model.safetensors.index.json": json.dumps(
            {"weight_map": {"enc.w": "0.safetensors"}}
        ).encode(),
        "text_encoder/0.safetensors": b"enc",
    }

    result = SafetensorsIndexCheck().run(context_for_files(files))

    assert result.status == "warn"
    assert result.details["shard_layout"] == "stale_all_absent_with_shipped"


def test_stale_index_does_not_hide_a_later_top_level_index_with_an_invalid_reference() -> None:
    # Bug caught (C2): run() keeping the first stale-tagged result, so an earlier stale
    # warn swallows a later index that fails on an out-of-repo reference.
    files = {
        "model.safetensors.index.json": json.dumps(_STALE_INDEX).encode(),
        "z.safetensors.index.json": json.dumps(
            {"weight_map": {"w": "../outside.safetensors"}}
        ).encode(),
        "model-00001-of-00003.safetensors": b"shard",
    }
    target = InvalidShardPathTarget(files=files)

    result = SafetensorsIndexCheck().run(CheckContext(target=target, options=check_options()))

    assert result.status == "fail"
    assert result.details["index_path"] == "z.safetensors.index.json"


def test_stale_index_message_uses_singular_forms_for_one_shard() -> None:
    # Bug caught (C3): "names 1 shard that are absent ... while 1 other shard ship".
    index = {"weight_map": {"old": "model-00001-of-00002.safetensors"}}

    result = SafetensorsIndexCheck().run(_index_ctx(index, ("model-00001-of-00001.safetensors",)))

    assert result.message == (
        "Safetensors index model.safetensors.index.json is stale: it names 1 shard that is "
        "absent from the repository, while 1 other shard ships. mlx-lm never reads the index "
        "and loads model*.safetensors; mlx-vlm falls back to the shipped shards when none of "
        "the named ones exist. Loaders that follow the index, and huggingface_hub's "
        "get_safetensors_metadata, fail."
    )


def test_index_check_accepts_a_dot_prefixed_shard_value() -> None:
    # Bug caught (C1): "./model-00002..." checked verbatim, so a present shard reads
    # as missing.
    index = {
        "weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": "./model-00002-of-00002.safetensors",
        }
    }
    shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")

    result = SafetensorsIndexCheck().run(_index_ctx(index, shards))

    assert result.status == "pass"
    assert result.details["shards"] == shards


class CaseInsensitiveTarget(FakeTarget):
    def exists(self, path: str) -> bool:
        return path.lower() in {name.lower() for name in self.files}


def test_case_mismatched_shard_counts_as_present_in_the_classification() -> None:
    # Bug caught (C1): the classification using listing membership while the check uses
    # target.exists, so a case-mismatched present shard plus one absent shard is called
    # stale (warn) instead of partly missing (fail), unlike the local header.
    index = {
        "weight_map": {
            "a": "Model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        }
    }
    files = {"model.safetensors.index.json": json.dumps(index).encode()}
    files.update(
        dict.fromkeys(
            ("model-00001-of-00002.safetensors", "model-00003-of-00003.safetensors"), b"s"
        )
    )
    target = CaseInsensitiveTarget(files=files)

    result = SafetensorsIndexCheck().run(CheckContext(target=target, options=check_options()))

    assert result.status == "fail"
    assert result.details["shard_layout"] == "partial_missing"
    assert result.details["missing_shards"] == ("model-00002-of-00002.safetensors",)


def test_missing_shard_message_replaces_control_characters() -> None:
    # Bug caught (S2): a "\n" in a shard name faking a line in the text report; the raw
    # name must still reach details.
    index = {"weight_map": {"w": "evil\nFAIL x.safetensors"}}

    result = SafetensorsIndexCheck().run(_index_ctx(index, ()))

    assert result.status == "fail"
    assert result.message == (
        "Safetensors index model.safetensors.index.json references invalid or missing shard "
        "evil?FAIL x.safetensors."
    )
    assert result.details["missing_shards"] == ("evil\nFAIL x.safetensors",)


@pytest.mark.parametrize(
    "value",
    ["../outside.safetensors", "sub/../../outside.safetensors", "/abs/outside.safetensors"],
    ids=["dotdot", "dotdot-after-normalizing", "absolute"],
)
def test_escaping_shard_value_is_invalid_whatever_the_target_says(value: str) -> None:
    # Bug caught (RC3): a value outside the repository judged by target.exists alone, so a
    # target that just answers False (HF) calls it stale while the local target fails it.
    index = {"weight_map": {"w": value}}

    result = SafetensorsIndexCheck().run(_index_ctx(index, ("model.safetensors",)))

    assert result.status == "fail"
    assert result.details["invalid_shards"] == (value.replace("sub/../", ""),)
    assert result.details["shard_layout"] == "all_missing"


def test_dotdot_named_file_inside_the_repo_is_not_an_escape() -> None:
    # Bug caught (RC3 over-reach): "..x.safetensors" is an ordinary file name inside the
    # repository, not a parent-directory reference.
    index = {"weight_map": {"w": "..x.safetensors"}}

    result = SafetensorsIndexCheck().run(_index_ctx(index, ("..x.safetensors",)))

    assert result.status == "pass"


def _named_index_ctx(index_path: str, index: dict[str, object], shards: tuple[str, ...]):
    files = {index_path: json.dumps(index).encode()}
    files.update(dict.fromkeys(shards, b"shard"))
    return context_for_files(files)


def test_stale_message_replaces_control_characters_in_the_index_path() -> None:
    # Bug caught (RC4): a "\n" in the index file name faking a report line in the stale
    # warning; details keep the raw path.
    index_path = "evil\nFAIL.safetensors.index.json"
    index = {"weight_map": {"old": "model-00001-of-00002.safetensors"}}

    result = SafetensorsIndexCheck().run(
        _named_index_ctx(index_path, index, ("model-00001-of-00001.safetensors",))
    )

    assert result.status == "warn"
    assert result.message.startswith(
        "Safetensors index evil?FAIL.safetensors.index.json is stale: it names 1 shard"
    )
    assert "\n" not in result.message
    assert result.remediation == (
        "Regenerate evil?FAIL.safetensors.index.json from the shipped shards, or remove the "
        "stale index."
    )
    assert result.details["index_path"] == index_path


def test_fail_message_replaces_control_characters_in_the_index_path() -> None:
    # Bug caught (RC4): the same raw index path in the missing-shard failure.
    index_path = "evil\nFAIL.safetensors.index.json"
    index = {"weight_map": {"w": "model-00001-of-00001.safetensors"}}

    result = SafetensorsIndexCheck().run(_named_index_ctx(index_path, index, ()))

    assert result.status == "fail"
    assert result.message == (
        "Safetensors index evil?FAIL.safetensors.index.json references invalid or missing "
        "shard model-00001-of-00001.safetensors."
    )
    assert result.details["index_path"] == index_path


def test_stale_index_message_agrees_with_mixed_counts() -> None:
    # Bug caught (RT2): the verb for the shipped count taken from the absent count
    # ("while 2 other shards ships"), or the noun swapped between the two counts.
    index = {"weight_map": {"old": "model-00001-of-00002.safetensors"}}

    result = SafetensorsIndexCheck().run(
        _index_ctx(index, ("model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors"))
    )

    assert result.message == (
        "Safetensors index model.safetensors.index.json is stale: it names 1 shard that is "
        "absent from the repository, while 2 other shards ship. mlx-lm never reads the index "
        "and loads model*.safetensors; mlx-vlm falls back to the shipped shards when none of "
        "the named ones exist. Loaders that follow the index, and huggingface_hub's "
        "get_safetensors_metadata, fail."
    )


class RaisingExistsTarget(FakeTarget):
    error: Exception | None = None

    def exists(self, path: str) -> bool:
        if path == "link.safetensors" and self.error is not None:
            raise self.error
        return super().exists(path)


@pytest.mark.parametrize(
    "error",
    [TargetError("outside model root", target="link.safetensors", source="local"), UnicodeError()],
    ids=["target-error", "unicode-error"],
)
def test_target_rejected_in_repo_name_is_invalid_not_stale(error: Exception) -> None:
    # Bug caught: a name the target itself rejects (a symlink leaving the root, an
    # unencodable name) counted as merely absent, so the shipped shard excuses it as stale.
    index = {"weight_map": {"w": "link.safetensors"}}
    files = {
        "model.safetensors.index.json": json.dumps(index).encode(),
        "model-00001-of-00001.safetensors": b"w",
    }
    target = RaisingExistsTarget(files=files)
    target.error = error

    result = SafetensorsIndexCheck().run(CheckContext(target=target, options=check_options()))

    assert result.status == "fail"
    assert result.details["invalid_shards"] == ("link.safetensors",)
