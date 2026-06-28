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
