"""Safetensors header parsing and aggregation (no weight download)."""

import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

# Match huggingface_hub's own SAFETENSORS_MAX_HEADER_LENGTH so the local and HF
# paths enforce the same upper bound on a single header.
_MAX_HEADER_BYTES = 25_000_000


class SafetensorsHeaderError(ValueError):
    """A safetensors header could not be parsed."""


@dataclass(frozen=True, slots=True, kw_only=True)
class TensorEntry:
    """A single tensor's header metadata."""

    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    parameter_count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class FileHeader:
    """Parsed header for one safetensors file."""

    filename: str
    tensors: Mapping[str, TensorEntry]
    metadata: Mapping[str, str]
    header_length: int | None  # local: parsed JSON length; hf: None (not exposed by the hub)
    file_size: int | None  # local: stat; hf: sibling size

    @property
    def data_section_length(self) -> int | None:
        """Bytes available for tensor data, or None when it can't be computed."""
        if self.file_size is None or self.header_length is None:
            return None
        return self.file_size - 8 - self.header_length


@dataclass(frozen=True, slots=True, kw_only=True)
class SafetensorsHeader:
    """Aggregated header across all safetensors files of a repository."""

    files: tuple[FileHeader, ...]
    weight_map: Mapping[str, str]
    sharded: bool
    param_count_by_dtype: Mapping[str, int]

    def tensor(self, name: str) -> TensorEntry | None:
        """Return a tensor entry by name across files, or None when absent."""
        for file_header in self.files:
            entry = file_header.tensors.get(name)
            if entry is not None:
                return entry
        return None

    def total_parameter_count(self) -> int:
        """Return the total stored-element count across all dtypes."""
        return sum(self.param_count_by_dtype.values())


def parse_file_header(filename: str, raw: bytes, *, file_size: int | None) -> FileHeader:
    """Parse the safetensors 8-byte-length + JSON header from leading file bytes."""
    if len(raw) < 8:
        raise SafetensorsHeaderError(f"{filename}: truncated safetensors header prefix")
    header_length = int(struct.unpack("<Q", raw[:8])[0])
    if header_length > _MAX_HEADER_BYTES:
        raise SafetensorsHeaderError(
            f"{filename}: safetensors header too large ({header_length} bytes)"
        )
    header_bytes = raw[8 : 8 + header_length]
    if len(header_bytes) < header_length:
        raise SafetensorsHeaderError(f"{filename}: safetensors header is truncated")
    try:
        parsed: object = json.loads(header_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SafetensorsHeaderError(f"{filename}: invalid safetensors header JSON") from exc
    if not isinstance(parsed, dict):
        raise SafetensorsHeaderError(f"{filename}: safetensors header must be a JSON object")
    raw_header = cast("dict[str, object]", parsed)
    metadata = _parse_metadata(raw_header.get("__metadata__"))
    tensors = {
        name: _parse_tensor(filename, name, value)
        for name, value in raw_header.items()
        if name != "__metadata__"
    }
    return FileHeader(
        filename=filename,
        tensors=tensors,
        metadata=metadata,
        header_length=header_length,
        file_size=file_size,
    )


def _parse_metadata(value: object) -> Mapping[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _parse_tensor(filename: str, name: str, value: object) -> TensorEntry:
    if not isinstance(value, dict):
        raise SafetensorsHeaderError(f"{filename}: tensor {name} must be a JSON object")
    entry = cast("dict[str, object]", value)
    dtype = entry.get("dtype")
    shape = entry.get("shape")
    offsets = entry.get("data_offsets")
    if not isinstance(dtype, str):
        raise SafetensorsHeaderError(f"{filename}: tensor {name} is missing a string dtype")
    if not isinstance(shape, list) or not all(_is_int(dim) for dim in shape):
        raise SafetensorsHeaderError(f"{filename}: tensor {name} has an invalid shape")
    if not isinstance(offsets, list) or len(offsets) != 2 or not all(_is_int(o) for o in offsets):
        raise SafetensorsHeaderError(f"{filename}: tensor {name} has invalid data_offsets")
    shape_tuple = tuple(cast("list[int]", shape))
    begin, end = cast("list[int]", offsets)
    return TensorEntry(
        dtype=dtype,
        shape=shape_tuple,
        data_offsets=(begin, end),
        parameter_count=math.prod(shape_tuple),
    )


def _is_int(value: object) -> bool:
    return type(value) is int
