import json
import struct

import pytest

from mlx_model_doctor.safetensors_header import (
    SafetensorsHeaderError,
    TensorEntry,
    parse_file_header,
)


def _header_bytes(header: dict[str, object], data_len: int = 0) -> bytes:
    raw = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw + (b"\x00" * data_len)


def test_parse_file_header_reads_one_tensor() -> None:
    raw = _header_bytes(
        {"w": {"dtype": "BF16", "shape": [4, 8], "data_offsets": [0, 64]}},
        data_len=64,
    )
    header = parse_file_header("model.safetensors", raw, file_size=len(raw))
    assert header.tensors["w"] == TensorEntry(
        dtype="BF16", shape=(4, 8), data_offsets=(0, 64), parameter_count=32
    )
    assert header.header_length == len(
        json.dumps({"w": {"dtype": "BF16", "shape": [4, 8], "data_offsets": [0, 64]}}).encode()
    )
    assert header.data_section_length == 64


def test_parse_file_header_lifts_metadata_out_of_tensors() -> None:
    raw = _header_bytes(
        {
            "__metadata__": {"format": "mlx"},
            "w": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]},
        }
    )
    header = parse_file_header("m.safetensors", raw, file_size=None)
    assert header.metadata == {"format": "mlx"}
    assert set(header.tensors) == {"w"}
    assert header.data_section_length is None  # file_size unknown


def test_parse_file_header_rejects_truncated_prefix() -> None:
    with pytest.raises(SafetensorsHeaderError, match="prefix"):
        parse_file_header("m.safetensors", b"\x01\x02", file_size=2)


def test_parse_file_header_rejects_oversized_header_length() -> None:
    raw = struct.pack("<Q", 25_000_001) + b"{}"
    with pytest.raises(SafetensorsHeaderError, match="too large"):
        parse_file_header("m.safetensors", raw, file_size=len(raw))


def test_parse_file_header_rejects_invalid_json() -> None:
    raw = struct.pack("<Q", 3) + b"{ x"
    with pytest.raises(SafetensorsHeaderError, match="JSON"):
        parse_file_header("m.safetensors", raw, file_size=len(raw))


def test_parse_file_header_rejects_non_object_header() -> None:
    arr = b"[1, 2, 3]"
    raw = struct.pack("<Q", len(arr)) + arr
    with pytest.raises(SafetensorsHeaderError, match="object"):
        parse_file_header("m.safetensors", raw, file_size=len(raw))


def test_parse_file_header_rejects_bad_data_offsets() -> None:
    raw = _header_bytes({"w": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4, 8]}})
    with pytest.raises(SafetensorsHeaderError, match="data_offsets"):
        parse_file_header("m.safetensors", raw, file_size=len(raw))


def test_parse_file_header_rejects_missing_dtype() -> None:
    raw = _header_bytes({"w": {"shape": [2], "data_offsets": [0, 4]}})
    with pytest.raises(SafetensorsHeaderError, match="dtype"):
        parse_file_header("m.safetensors", raw, file_size=len(raw))


from mlx_model_doctor.safetensors_header import (  # noqa: E402
    FileHeader,
    build_local_header,
    map_hf_repo_metadata,
)


def _fh(filename: str, tensors: dict[str, TensorEntry], file_size: int | None = None) -> FileHeader:
    return FileHeader(
        filename=filename, tensors=tensors, metadata={}, header_length=10, file_size=file_size
    )


def test_build_local_header_single_file_synthesizes_weight_map() -> None:
    entry = TensorEntry(dtype="BF16", shape=(2, 2), data_offsets=(0, 16), parameter_count=4)
    header = build_local_header([_fh("model.safetensors", {"w": entry})], weight_map=None)
    assert header.sharded is False
    assert header.weight_map == {"w": "model.safetensors"}
    assert header.param_count_by_dtype == {"BF16": 4}
    assert header.total_parameter_count() == 4


def test_build_local_header_uses_index_weight_map_when_present() -> None:
    entry = TensorEntry(dtype="F16", shape=(2,), data_offsets=(0, 4), parameter_count=2)
    header = build_local_header(
        [_fh("model-00001-of-00002.safetensors", {"w": entry})],
        weight_map={"w": "model-00001-of-00002.safetensors"},
    )
    assert header.sharded is True
    assert header.weight_map == {"w": "model-00001-of-00002.safetensors"}


class _FakeTensorInfo:
    def __init__(self, dtype: str, shape: list[int], offsets: tuple[int, int], count: int) -> None:
        self.dtype = dtype
        self.shape = shape
        self.data_offsets = offsets
        self.parameter_count = count


class _FakeFileMeta:
    def __init__(self, tensors: dict[str, _FakeTensorInfo]) -> None:
        self.tensors = tensors
        self.metadata = {"format": "mlx"}


class _FakeRepoMeta:
    def __init__(self) -> None:
        self.weight_map = {"w": "model.safetensors"}
        self.sharded = False
        self.files_metadata = {
            "model.safetensors": _FakeFileMeta({"w": _FakeTensorInfo("U32", [4, 16], (0, 256), 64)})
        }


def test_map_hf_repo_metadata_threads_file_sizes_and_maps_tensors() -> None:
    header = map_hf_repo_metadata(_FakeRepoMeta(), file_sizes={"model.safetensors": 9000})
    assert header.weight_map == {"w": "model.safetensors"}
    assert header.sharded is False
    file_header = header.files[0]
    assert file_header.file_size == 9000
    assert file_header.header_length is None  # the hub does not expose it -> no data_section_length
    assert file_header.data_section_length is None
    assert header.tensor("w") == TensorEntry(
        dtype="U32", shape=(4, 16), data_offsets=(0, 256), parameter_count=64
    )


def test_build_local_header_multi_shard_synthesizes_and_aggregates_two_dtypes() -> None:
    a = TensorEntry(dtype="BF16", shape=(2, 2), data_offsets=(0, 16), parameter_count=4)
    b = TensorEntry(dtype="F16", shape=(3,), data_offsets=(0, 6), parameter_count=3)
    header = build_local_header(
        [_fh("s1.safetensors", {"a": a}), _fh("s2.safetensors", {"b": b})],
        weight_map=None,
    )
    assert header.sharded is True
    assert header.weight_map == {"a": "s1.safetensors", "b": "s2.safetensors"}
    assert header.param_count_by_dtype == {"BF16": 4, "F16": 3}
    assert header.total_parameter_count() == 7
