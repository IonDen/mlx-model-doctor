import json
import math
import posixpath
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from mlx_model_doctor.context import CheckContext, CheckOptions
from mlx_model_doctor.safetensors_header import SafetensorsHeader, SafetensorsHeaderError

# A shard spec: tensor name -> (dtype, shape). Data bytes are zeros.
ShardSpec = Mapping[str, tuple[str, tuple[int, ...]]]
_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4}


def check_options() -> CheckOptions:
    return CheckOptions(
        max_memory_bytes=None,
        context_length=4096,
        include_weights=False,
        smoke=False,
        verbosity="normal",
    )


def context_for_files(
    files: dict[str, bytes],
    *,
    source: Literal["local", "hf"] = "local",
    name: str = "fake",
    tags: frozenset[str] = frozenset(),
    library_name: str | None = None,
    options: CheckOptions | None = None,
) -> CheckContext:
    return CheckContext(
        target=FakeTarget(
            files=files, name=name, _source=source, tags=tags, library_name=library_name
        ),
        options=options if options is not None else check_options(),
    )


@dataclass(slots=True)
class FakeTarget:
    files: dict[str, bytes]
    name: str = "fake"
    _source: Literal["local", "hf"] = "local"
    _safetensors_header: SafetensorsHeader | None = None
    tags: frozenset[str] = frozenset()
    library_name: str | None = None
    # Declared sizes independent of a path's actual byte length -- lets a test pin
    # a file's reported size (e.g. to exercise a byte-size cap) without writing
    # that many real bytes.
    size_overrides: dict[str, int] | None = None

    @property
    def source(self) -> Literal["local", "hf"]:
        return self._source

    def safetensors_header(self) -> SafetensorsHeader | None:
        return self._safetensors_header

    def exists(self, path: str) -> bool:
        return path in self.files

    def list_files(self) -> Sequence[str]:
        return tuple(sorted(self.files))

    def size(self, path: str) -> int | None:
        if self.size_overrides is not None and path in self.size_overrides:
            return self.size_overrides[path]
        data = self.files.get(path)
        return None if data is None else len(data)

    def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        data = self.files[path]
        return data if max_bytes is None else data[:max_bytes]

    def read_text(self, path: str, *, max_bytes: int | None = None) -> str:
        return self.read_bytes(path, max_bytes=max_bytes).decode("utf-8")


@dataclass(frozen=True, slots=True)
class FakeTensorInfo:
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    parameter_count: int


@dataclass(frozen=True, slots=True)
class FakeFileMetadata:
    tensors: dict[str, FakeTensorInfo]
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class FakeRepoMetadata:
    weight_map: dict[str, str]
    sharded: bool
    files_metadata: dict[str, FakeFileMetadata]


@dataclass(frozen=True, slots=True)
class FakeHfSibling:
    rfilename: str
    size: int | None


@dataclass(frozen=True, slots=True)
class FakeHfModelInfo:
    siblings: tuple[FakeHfSibling, ...]
    tags: tuple[str, ...] | None = None
    library_name: str | None = None


def _tensor_layout(spec: ShardSpec) -> dict[str, FakeTensorInfo]:
    infos: dict[str, FakeTensorInfo] = {}
    offset = 0
    for name, (dtype, shape) in spec.items():
        count = math.prod(shape)
        end = offset + count * _DTYPE_BYTES[dtype]
        infos[name] = FakeTensorInfo(
            dtype=dtype, shape=shape, data_offsets=(offset, end), parameter_count=count
        )
        offset = end
    return infos


def safetensors_bytes(spec: ShardSpec) -> bytes:
    """Real safetensors bytes (8-byte length, JSON header, zero data) for a shard spec."""
    infos = _tensor_layout(spec)
    header = {
        name: {
            "dtype": info.dtype,
            "shape": list(info.shape),
            "data_offsets": list(info.data_offsets),
        }
        for name, info in infos.items()
    }
    raw = json.dumps(header).encode("utf-8")
    data_length = max((info.data_offsets[1] for info in infos.values()), default=0)
    return struct.pack("<Q", len(raw)) + raw + b"\x00" * data_length


def file_metadata(spec: ShardSpec) -> FakeFileMetadata:
    """The huggingface_hub-shaped per-file metadata for a shard spec."""
    return FakeFileMetadata(tensors=_tensor_layout(spec), metadata={"format": "mlx"})


def write_repo(root: Path, files: Mapping[str, bytes], shards: Mapping[str, ShardSpec]) -> Path:
    """Write the same file set a FakeShardHub serves into a local directory."""
    root.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    for name, spec in shards.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(safetensors_bytes(spec))
    return root


@dataclass(slots=True)
class FakeShardHub:
    """An HfHubProtocol fake serving metadata files plus safetensors shard headers.

    ``safetensors_metadata`` follows huggingface_hub 1.19.0 ``get_safetensors_metadata``:
    ``model.safetensors`` first, else every shard ``model.safetensors.index.json`` names,
    else not a safetensors repo. A named shard that is absent raises the header error
    the real adapter maps a 404 to. Index values are fetched dot-normalized, as the real
    request is: ``hf_hub_url`` keeps ``/`` (``file_download.py:277``) and httpx drops
    ``.``/``..`` URL segments (``httpx/_urlparse.py:447`` ``normalize_path``); the
    returned ``files_metadata`` is keyed by the raw value, as huggingface_hub keys it.
    """

    files: dict[str, bytes]
    shards: dict[str, ShardSpec]
    metadata_error: Exception | None = None
    file_metadata_error: Exception | None = None
    unknown_sizes: frozenset[str] = frozenset()  # listed with size None
    safetensors_metadata_calls: list[str] = field(default_factory=list)
    file_metadata_calls: list[str] = field(default_factory=list)
    download_calls: list[str] = field(default_factory=list)

    def model_info(self, repo_id: str, *, files_metadata: bool) -> FakeHfModelInfo:
        siblings = [
            FakeHfSibling(rfilename=n, size=None if n in self.unknown_sizes else len(d))
            for n, d in self.files.items()
        ]
        siblings += [
            FakeHfSibling(rfilename=n, size=len(safetensors_bytes(s)))
            for n, s in self.shards.items()
        ]
        return FakeHfModelInfo(siblings=tuple(siblings))

    def download_bytes(self, repo_id: str, filename: str) -> bytes:
        self.download_calls.append(filename)
        return self.files[filename]

    def safetensors_metadata(self, repo_id: str) -> FakeRepoMetadata | None:
        self.safetensors_metadata_calls.append(repo_id)
        if self.metadata_error is not None:
            raise self.metadata_error
        if "model.safetensors" in self.shards:
            meta = file_metadata(self.shards["model.safetensors"])
            return FakeRepoMetadata(
                weight_map=dict.fromkeys(meta.tensors, "model.safetensors"),
                sharded=False,
                files_metadata={"model.safetensors": meta},
            )
        index = self.files.get("model.safetensors.index.json")
        if index is None:
            return None
        weight_map: dict[str, str] = json.loads(index)["weight_map"]
        files_meta: dict[str, FakeFileMetadata] = {}
        for name in set(weight_map.values()):
            fetched = posixpath.normpath(name)
            if fetched not in self.shards:
                raise SafetensorsHeaderError(f"404 Entry Not Found: {fetched}")
            files_meta[name] = file_metadata(self.shards[fetched])
        return FakeRepoMetadata(weight_map=weight_map, sharded=True, files_metadata=files_meta)

    def safetensors_file_metadata(self, repo_id: str, filename: str) -> FakeFileMetadata:
        self.file_metadata_calls.append(filename)
        if self.file_metadata_error is not None:
            raise self.file_metadata_error
        if filename not in self.shards:
            raise SafetensorsHeaderError(f"404 Entry Not Found: {filename}")
        return file_metadata(self.shards[filename])


STANDARD_INDEX = "model.safetensors.index.json"
_REPO_CONFIG = {
    "model_type": "llama",
    "hidden_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "vocab_size": 256,
    "intermediate_size": 512,
    "pad_token_id": 0,
    "eos_token_id": 1,
}


def shard_repo(layout: str) -> tuple[dict[str, bytes], dict[str, ShardSpec]]:
    """A small repo (metadata files, shard specs) for one index-vs-shards layout.

    - ``stale``: the root index names two shards, both absent; three others ship.
    - ``partial``: the root index names three shards; only the middle one ships.
    - ``nested``: a root ``model.safetensors`` plus a nested component index (0079).
    - ``consolidated_only``: the root index's one shard is absent; only
      ``consolidated.safetensors`` ships, which neither MLX loader picks up.
    - ``consistent``: the root index names two shards and both ship.
    - ``dot_prefixed``: ``consistent`` with one value written ``./model-00002-...``.
    - ``stale_with_consolidated``: ``stale`` plus a ``consolidated.safetensors`` holding a
      tensor of its own, which neither MLX loader reads beside ``model*`` shards.
    - ``escaping``: the root index names one ``../`` shard and one absolute shard (both
      outside the repository); ``model-00001-of-00001.safetensors`` ships.
    """
    files: dict[str, bytes] = {
        "config.json": json.dumps(_REPO_CONFIG).encode(),
        "tokenizer.json": b"{}",
    }
    shards: dict[str, ShardSpec]
    if layout in ("stale", "stale_with_consolidated"):
        files[STANDARD_INDEX] = json.dumps(
            {
                "weight_map": {
                    "old.x": "model-00001-of-00002.safetensors",
                    "old.y": "model-00002-of-00002.safetensors",
                }
            }
        ).encode()
        shards = {
            "model-00001-of-00003.safetensors": {"a": ("BF16", (2, 3))},
            "model-00002-of-00003.safetensors": {"b": ("BF16", (5,))},
            "model-00003-of-00003.safetensors": {"c": ("F32", (7,))},
        }
        if layout == "stale_with_consolidated":
            shards["consolidated.safetensors"] = {"z": ("F32", (100,))}
    elif layout == "partial":
        files[STANDARD_INDEX] = json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00003.safetensors",
                    "b": "model-00002-of-00003.safetensors",
                    "c": "model-00003-of-00003.safetensors",
                }
            }
        ).encode()
        shards = {"model-00002-of-00003.safetensors": {"b": ("BF16", (5,))}}
    elif layout == "nested":
        files["text_encoder/" + STANDARD_INDEX] = json.dumps(
            {"weight_map": {"enc.w": "0.safetensors"}}
        ).encode()
        shards = {
            "model.safetensors": {"a": ("BF16", (2, 3))},
            "text_encoder/0.safetensors": {"enc.w": ("F32", (11,))},
        }
    elif layout == "consolidated_only":
        files[STANDARD_INDEX] = json.dumps(
            {"weight_map": {"old.x": "model-00001-of-00002.safetensors"}}
        ).encode()
        shards = {"consolidated.safetensors": {"a": ("BF16", (2, 3))}}
    elif layout in ("consistent", "dot_prefixed"):
        prefix = "./" if layout == "dot_prefixed" else ""
        files[STANDARD_INDEX] = json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": prefix + "model-00002-of-00002.safetensors",
                }
            }
        ).encode()
        shards = {
            "model-00001-of-00002.safetensors": {"a": ("BF16", (2, 3))},
            "model-00002-of-00002.safetensors": {"b": ("BF16", (5,))},
        }
    elif layout == "escaping":
        files[STANDARD_INDEX] = json.dumps(
            {
                "weight_map": {
                    "a": "../model-00001-of-00002.safetensors",
                    "b": "/abs/model-00002-of-00002.safetensors",
                }
            }
        ).encode()
        shards = {"model-00001-of-00001.safetensors": {"a": ("BF16", (2, 3))}}
    else:
        raise ValueError(layout)
    return files, shards
