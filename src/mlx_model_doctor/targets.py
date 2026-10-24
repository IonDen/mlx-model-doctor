"""Model target abstractions."""

import posixpath
import struct
import sys
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG
from typing import Literal, Protocol, cast, runtime_checkable
from urllib.parse import urlparse

from mlx_model_doctor.errors import TargetError
from mlx_model_doctor.safetensors_header import (
    _MAX_HEADER_BYTES,
    FileHeader,
    HfFileMetadata,
    HfSafetensorsRepoMetadata,
    SafetensorsHeader,
    SafetensorsHeaderError,
    build_local_header,
    map_hf_repo_metadata,
    parse_file_header,
)
from mlx_model_doctor.shard_layout import (
    ShardLayout,
    ShardLayoutKind,
    classify_shard_layout,
    parse_index_weight_map,
    resolve_shard_path,
    root_index_path,
)

# Same bound as context._MAX_METADATA_BYTES (context imports this module, so the value
# is repeated rather than imported): an index larger than this is not read.
_MAX_INDEX_BYTES = 16 * 1024 * 1024


def _canonical_shard_paths(
    files: "Sequence[str]", weight_map: "Mapping[str, str] | None"
) -> list[str]:
    """Select the canonical safetensors shards, matching the HF metadata set.

    Prefer the shards named by a safetensors index ``weight_map``; otherwise fall back to
    top-level ``*.safetensors`` files, excluding nested component weights (e.g.
    ``vae/*.safetensors``) that ``get_safetensors_metadata`` does not aggregate.
    """
    all_shards = [path for path in files if path.endswith(".safetensors")]
    if weight_map:
        named = set(weight_map.values())
        selected = [path for path in all_shards if path in named]
        if selected:
            return sorted(selected)
    return sorted(path for path in all_shards if "/" not in path)


@dataclass(slots=True, kw_only=True)
class _ShippedShardsMetadata:
    """Repo metadata assembled from per-file headers, shaped like huggingface_hub's."""

    weight_map: Mapping[str, str]
    sharded: bool
    files_metadata: Mapping[str, HfFileMetadata]


class ModelTarget(Protocol):
    """Readable model repository target."""

    @property
    def name(self) -> str:
        """Return a human-readable target name."""

    @property
    def source(self) -> Literal["local", "hf"]:
        """Return the target source kind."""

    def exists(self, path: str) -> bool:
        """Return whether a file exists at a target-relative path."""

    def list_files(self) -> Sequence[str]:
        """Return target-relative POSIX file paths."""

    def size(self, path: str) -> int | None:
        """Return file size in bytes, or None when the file is absent."""

    def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        """Read file bytes from a target-relative path."""

    def read_text(self, path: str, *, max_bytes: int | None = None) -> str:
        """Read UTF-8 text from a target-relative path."""

    def safetensors_header(self) -> SafetensorsHeader | None:
        """Return the aggregated safetensors header, or None when unavailable."""


@runtime_checkable
class MlxListingMetadata(Protocol):
    """Optional Hub listing metadata used for MLX-compatibility signals."""

    @property
    def tags(self) -> frozenset[str]:
        """Return repository tags."""

    @property
    def library_name(self) -> str | None:
        """Return the declared library name, if any."""


class HfSiblingProtocol(Protocol):
    """Hugging Face repository sibling metadata."""

    rfilename: str
    size: int | None


class HfModelInfoProtocol(Protocol):
    """Hugging Face model info metadata used by the target."""

    siblings: Sequence[HfSiblingProtocol]
    tags: Sequence[str] | None
    library_name: str | None


class HfHubProtocol(Protocol):
    """Small Hugging Face Hub adapter boundary."""

    def model_info(self, repo_id: str, *, files_metadata: bool) -> HfModelInfoProtocol:
        """Return model repository metadata."""

    def download_bytes(self, repo_id: str, filename: str) -> bytes:
        """Download a single repository file as bytes."""

    def safetensors_metadata(self, repo_id: str) -> HfSafetensorsRepoMetadata | None:
        """Return safetensors header metadata via Range requests, or None when absent."""

    def safetensors_file_metadata(self, repo_id: str, filename: str) -> HfFileMetadata:
        """Return one safetensors file's header metadata via a Range request."""


@contextmanager
def _hub_progress_bars_disabled() -> Iterator[None]:
    """Silence huggingface_hub progress bars for the block, then restore the prior state.

    ``get_safetensors_metadata`` draws a "Parse safetensors files" bar on stderr
    (huggingface_hub 1.19.0 ``hf_api.py:6688``). Helpers verified in
    ``huggingface_hub/utils/tqdm.py``: ``progress_bar_states`` (:105, the per-group
    mapping that both helpers clear), ``disable_progress_bars`` (:108),
    ``enable_progress_bars`` (:158), ``are_progress_bars_disabled`` (:190). The mapping is
    snapshotted and restored, so a caller's per-group settings survive.
    """
    from huggingface_hub.utils.tqdm import (
        are_progress_bars_disabled,
        disable_progress_bars,
        enable_progress_bars,
    )

    # The module itself: ``huggingface_hub.utils.tqdm`` as an attribute is the tqdm class.
    states = cast(
        "dict[str, bool]", sys.modules[disable_progress_bars.__module__].progress_bar_states
    )
    snapshot = dict(states)
    was_disabled = are_progress_bars_disabled()
    if not was_disabled:
        with warnings.catch_warnings():
            # HF_HUB_DISABLE_PROGRESS_BARS=0 has priority and makes this a warning no-op.
            warnings.simplefilter("ignore")
            disable_progress_bars()
    try:
        yield
    finally:
        if not was_disabled:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                enable_progress_bars()
            states.clear()
            states.update(snapshot)


def _missing_safetensors_entry(exc: BaseException) -> str | None:
    """Return the missing ``.safetensors`` filename when ``exc`` is the Hub's entry 404.

    Only ``RemoteEntryNotFoundError`` (huggingface_hub 1.19.0 ``errors.py:367``, raised by
    ``hf_raise_for_status`` for ``X-Error-Code: EntryNotFound``) on a ``.safetensors``
    URL qualifies. ``LocalEntryNotFoundError`` (``errors.py:394``) is a connection
    problem and every other error stays a tool error.
    """
    from huggingface_hub.errors import RemoteEntryNotFoundError

    if not isinstance(exc, RemoteEntryNotFoundError):
        return None
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        url = response.url
    except RuntimeError:  # httpx: a response with no request has no URL
        return None
    path = urlparse(str(url)).path
    if not path.endswith(".safetensors"):
        return None
    return path.rsplit("/", 1)[-1]


class DefaultHfHub:
    """Hugging Face Hub adapter using targeted metadata and file reads."""

    def model_info(self, repo_id: str, *, files_metadata: bool) -> HfModelInfoProtocol:
        """Return model repository metadata from huggingface_hub."""
        from huggingface_hub import model_info

        return cast(
            "HfModelInfoProtocol",
            model_info(repo_id=repo_id, files_metadata=files_metadata),
        )

    def download_bytes(self, repo_id: str, filename: str) -> bytes:
        """Download a single file from huggingface_hub and read it as bytes."""
        from huggingface_hub import hf_hub_download

        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return Path(downloaded_path).read_bytes()

    def safetensors_metadata(self, repo_id: str) -> HfSafetensorsRepoMetadata | None:
        """Read safetensors header metadata from huggingface_hub (Range, no download)."""
        from huggingface_hub import HfApi
        from huggingface_hub.errors import NotASafetensorsRepoError, SafetensorsParsingError

        try:
            with _hub_progress_bars_disabled():
                return cast("HfSafetensorsRepoMetadata", HfApi().get_safetensors_metadata(repo_id))
        except (NotASafetensorsRepoError, SafetensorsParsingError):
            return None
        except Exception as exc:
            missing = _missing_safetensors_entry(exc)
            if missing is not None:
                raise SafetensorsHeaderError(
                    f"safetensors shard {missing} is absent from the repository"
                ) from exc
            raise TargetError(
                f"Could not read safetensors metadata for {repo_id}: {exc}",
                target=repo_id,
                source="hf",
            ) from exc

    def safetensors_file_metadata(self, repo_id: str, filename: str) -> HfFileMetadata:
        """Read one file's safetensors header from huggingface_hub (Range, no download)."""
        from huggingface_hub import HfApi
        from huggingface_hub.errors import SafetensorsParsingError

        try:
            with _hub_progress_bars_disabled():
                # huggingface_hub 1.19.0 hf_api.py:6704 parse_safetensors_file_metadata.
                return cast(
                    "HfFileMetadata", HfApi().parse_safetensors_file_metadata(repo_id, filename)
                )
        except SafetensorsParsingError as exc:
            raise SafetensorsHeaderError(f"{filename}: {exc}") from exc
        except Exception as exc:
            if _missing_safetensors_entry(exc) is not None:
                # Only listed shards are read one by one, so this is a Hub inconsistency.
                raise SafetensorsHeaderError(
                    f"safetensors shard {filename} returned 404 although the repository "
                    "listing includes it"
                ) from exc
            raise TargetError(
                f"Could not read safetensors metadata for {repo_id}:{filename}: {exc}",
                target=f"{repo_id}:{filename}",
                source="hf",
            ) from exc


class LocalTarget:
    """Readable target backed by a local model directory."""

    __slots__ = ("_root",)

    def __init__(self, root: str | Path) -> None:
        """Initialize a local target rooted at an existing directory."""
        requested_root = Path(root).expanduser()
        try:
            resolved_root = requested_root.resolve(strict=True)
        except OSError as exc:
            raise TargetError(
                f"Local target must be an existing directory: {requested_root}",
                target=str(requested_root),
                source="local",
            ) from exc
        if not resolved_root.is_dir():
            raise TargetError(
                f"Local target must be an existing directory: {requested_root}",
                target=str(requested_root),
                source="local",
            )
        self._root = resolved_root

    @property
    def name(self) -> str:
        """Return the resolved local model directory path."""
        return str(self._root)

    @property
    def source(self) -> Literal["local"]:
        """Return the local source kind."""
        return "local"

    def exists(self, path: str) -> bool:
        """Return whether a file exists at a model-relative path.

        A name the filesystem cannot stat (too long, no permission) is absent: on Python
        3.11-3.13 ``Path.is_file`` raises ``OSError`` for it instead of returning False.
        """
        file_path = self._path(path)
        try:
            return file_path.is_file()
        except OSError:
            return False

    def list_files(self) -> Sequence[str]:
        """Return sorted relative POSIX paths for files under the model root."""
        files: list[str] = []
        for path in self._root.rglob("*"):
            if not path.is_file():
                continue
            relative_path = path.relative_to(self._root).as_posix()
            try:
                self._path(relative_path)
            except TargetError:
                continue
            files.append(relative_path)
        return tuple(sorted(files))

    def size(self, path: str) -> int | None:
        """Return file size in bytes without reading file contents."""
        file_path = self._path(path)
        try:
            file_stat = file_path.stat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise self._target_error(f"Could not stat local target file: {path}", path) from exc
        if not S_ISREG(file_stat.st_mode):
            return None
        return file_stat.st_size

    def read_bytes(
        self,
        path: str,
        *,
        max_bytes: int | None = None,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes:
        """Read bytes from a model-relative path.

        With ``length`` given, this is a bounded ranged read: seek to
        ``offset`` and read at most ``length`` bytes (fewer at EOF). This is
        the primitive the parity delta map uses to byte-compare large tensor
        payloads in bounded chunks, without ever materializing a whole
        tensor's bytes in memory at once. With ``length`` omitted (the
        default), behavior is unchanged from before ``offset``/``length``
        existed: the whole file is read, or a leading ``max_bytes`` prefix
        when given; ``offset`` is not consulted in that case.
        """
        file_path = self._path(path)
        try:
            if length is not None:
                if offset < 0 or length < 0:
                    msg = f"offset and length must be non-negative, got {offset=}, {length=}"
                    raise ValueError(msg)
                with file_path.open("rb") as file:
                    file.seek(offset)
                    return file.read(length)
            if max_bytes is None:
                return file_path.read_bytes()
            with file_path.open("rb") as file:
                return file.read(max_bytes)
        except OSError as exc:
            raise self._target_error(f"Could not read local target file: {path}", path) from exc

    def read_text(self, path: str, *, max_bytes: int | None = None) -> str:
        """Read UTF-8 text from a model-relative path."""
        return self.read_bytes(path, max_bytes=max_bytes).decode("utf-8")

    def safetensors_header(self) -> SafetensorsHeader | None:
        """Read and aggregate safetensors headers from disk without reading weights.

        Only the root index's weight map is used, as on the Hugging Face path, so a
        nested component index never feeds tensors into the root header. A stale root
        index is ignored (the shipped top-level shards are read and their weight map
        synthesized); a root index with missing shards raises ``SafetensorsHeaderError``.
        """
        files = self.list_files()
        index_path = root_index_path(files)
        weight_map = self._index_weight_map(index_path) if index_path is not None else None
        layout = classify_shard_layout(
            files, weight_map, index_path=index_path, present=self._shard_present
        )
        if layout.kind.missing_weights:
            raise SafetensorsHeaderError(layout.absent_message())
        if layout.kind is ShardLayoutKind.CONSISTENT and weight_map and index_path:
            # The shards the index names, at the paths the presence check resolved.
            weight_map = {
                key: resolve_shard_path(index_path, value) for key, value in weight_map.items()
            }
            shard_paths = list(layout.named)
        elif layout.kind is ShardLayoutKind.STALE_ALL_ABSENT_WITH_SHIPPED:
            limit = layout.shipped_limit_message()
            if limit is not None:
                raise SafetensorsHeaderError(limit)
            # Same shard set the HF path reads: only the loader-eligible shipped shards.
            weight_map = None
            shard_paths = list(layout.shipped)
        else:
            weight_map = None
            shard_paths = _canonical_shard_paths(files, weight_map)
        if not shard_paths:
            return None
        file_headers = [self._read_local_file_header(path) for path in shard_paths]
        return build_local_header(file_headers, weight_map=weight_map)

    def _read_local_file_header(self, path: str) -> FileHeader:
        prefix = self.read_bytes(path, max_bytes=8)
        if len(prefix) < 8:
            raise SafetensorsHeaderError(f"{path}: truncated safetensors header prefix")
        header_length = int(struct.unpack("<Q", prefix)[0])
        if header_length > _MAX_HEADER_BYTES:
            raise SafetensorsHeaderError(f"{path}: safetensors header too large")
        raw = self.read_bytes(path, max_bytes=8 + header_length)
        return parse_file_header(path, raw, file_size=self.size(path))

    def _shard_present(self, path: str) -> bool:
        """Presence as ``text/safetensors.index`` sees it; an escaping path is absent."""
        try:
            return self.exists(path)
        except (TargetError, UnicodeError):
            return False

    def _index_weight_map(self, index_path: str) -> dict[str, str] | None:
        try:
            size = self.size(index_path)
            if size is None or size > _MAX_INDEX_BYTES:
                return None
            return parse_index_weight_map(self.read_text(index_path))
        except (UnicodeError, TargetError):
            return None

    def _path(self, path: str) -> Path:
        try:
            candidate = (self._root / path).resolve(strict=False)
        except (OSError, ValueError) as exc:
            raise self._target_error(f"Invalid local target path: {path}", path) from exc
        try:
            candidate.relative_to(self._root)
        except ValueError as exc:
            raise self._target_error(
                f"Local target path is outside the model root: {path}",
                path,
            ) from exc
        return candidate

    def _target_error(self, message: str, path: str | Path) -> TargetError:
        return TargetError(message, target=str(path), source=self.source)


class HfTarget:
    """Readable target backed by Hugging Face Hub model metadata."""

    __slots__ = ("_hub", "_library_name", "_metadata", "_repo_id", "_tags")

    def __init__(self, repo_id: str, *, hub: HfHubProtocol | None = None) -> None:
        """Initialize a Hugging Face target from model repository metadata."""
        self._repo_id = repo_id
        self._hub = hub if hub is not None else DefaultHfHub()
        try:
            info = self._hub.model_info(repo_id, files_metadata=True)
        except TargetError:
            raise
        except Exception as exc:
            raise TargetError(
                f"Could not inspect Hugging Face model {repo_id}: {exc}",
                target=repo_id,
                source="hf",
            ) from exc
        self._metadata = {sibling.rfilename: sibling.size for sibling in info.siblings}
        self._tags = frozenset(info.tags or ())
        self._library_name = info.library_name

    @property
    def name(self) -> str:
        """Return the Hugging Face repository ID."""
        return self._repo_id

    @property
    def source(self) -> Literal["hf"]:
        """Return the Hugging Face source kind."""
        return "hf"

    @property
    def tags(self) -> frozenset[str]:
        """Return repository tags from listing metadata."""
        return self._tags

    @property
    def library_name(self) -> str | None:
        """Return the declared library name, if any."""
        return self._library_name

    def exists(self, path: str) -> bool:
        """Return whether a file exists in repository metadata."""
        return path in self._metadata

    def list_files(self) -> Sequence[str]:
        """Return sorted repository file paths from metadata."""
        return tuple(sorted(self._metadata))

    def size(self, path: str) -> int | None:
        """Return file size from repository metadata."""
        return self._metadata.get(path)

    def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        """Download and read bytes for one metadata-listed repository file.

        ``max_bytes`` truncates the returned bytes *after* the whole file is
        downloaded; on this Hugging Face path it is not a bandwidth bound (unlike
        ``LocalTarget``, which streams only ``max_bytes`` from disk). To avoid
        fetching a large file, pre-check :meth:`size` or read the safetensors header
        via :meth:`safetensors_header`, which uses HTTP Range requests.
        """
        if path not in self._metadata:
            raise self._target_error(
                f"Hugging Face target path is not listed in repo metadata: {path}",
                path,
            )
        try:
            data = self._hub.download_bytes(self._repo_id, path)
        except TargetError:
            raise
        except Exception as exc:
            raise self._target_error(
                f"Could not read Hugging Face model file {path}: {exc}",
                path,
            ) from exc
        return data if max_bytes is None else data[:max_bytes]

    def read_text(self, path: str, *, max_bytes: int | None = None) -> str:
        """Read UTF-8 text from one repository file."""
        return self.read_bytes(path, max_bytes=max_bytes).decode("utf-8")

    def safetensors_header(self) -> SafetensorsHeader | None:
        """Map the Hugging Face safetensors header metadata, no weight download.

        A stale root index (every named shard absent, other top-level shards shipped)
        reads each shipped shard's own header; a root index with missing shards raises
        ``SafetensorsHeaderError`` naming them, without asking the Hub to walk the index.
        """
        layout = self._root_shard_layout()
        if layout.kind.missing_weights:
            raise SafetensorsHeaderError(layout.absent_message())
        if layout.kind is ShardLayoutKind.STALE_ALL_ABSENT_WITH_SHIPPED:
            limit = layout.shipped_limit_message()
            if limit is not None:
                raise SafetensorsHeaderError(limit)
            return self._shipped_shards_header(layout.shipped)
        repo_meta = self._hub.safetensors_metadata(self._repo_id)
        if repo_meta is None:
            return None
        # Root index values, normalized as the local target resolves them ("./x" is "x"),
        # so the file names and listing sizes match the local header.
        normalized = _ShippedShardsMetadata(
            weight_map={key: posixpath.normpath(v) for key, v in repo_meta.weight_map.items()},
            sharded=repo_meta.sharded,
            files_metadata={
                posixpath.normpath(name): meta for name, meta in repo_meta.files_metadata.items()
            },
        )
        file_sizes = {name: self._metadata.get(name) for name in normalized.files_metadata}
        return map_hf_repo_metadata(normalized, file_sizes=file_sizes)

    def _root_shard_layout(self) -> ShardLayout:
        files = self.list_files()
        index_path = root_index_path(files)
        weight_map: dict[str, str] | None = None
        if index_path is not None:
            size = self.size(index_path)
            if size is not None and size <= _MAX_INDEX_BYTES:
                try:
                    weight_map = parse_index_weight_map(self.read_text(index_path))
                except UnicodeError:
                    weight_map = None
        return classify_shard_layout(files, weight_map, index_path=index_path)

    def _shipped_shards_header(self, shipped: Sequence[str]) -> SafetensorsHeader:
        files_metadata = {
            name: self._hub.safetensors_file_metadata(self._repo_id, name) for name in shipped
        }
        weight_map: dict[str, str] = {}
        for name in shipped:
            for tensor in files_metadata[name].tensors:
                weight_map[tensor] = name
        repo_meta = _ShippedShardsMetadata(
            weight_map=weight_map, sharded=len(shipped) > 1, files_metadata=files_metadata
        )
        file_sizes = {name: self._metadata.get(name) for name in shipped}
        return map_hf_repo_metadata(repo_meta, file_sizes=file_sizes)

    def _target_error(self, message: str, path: str) -> TargetError:
        return TargetError(message, target=f"{self._repo_id}:{path}", source=self.source)
