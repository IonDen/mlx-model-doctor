"""Model target abstractions."""

import json
import struct
from collections.abc import Sequence
from pathlib import Path
from stat import S_ISREG
from typing import Literal, Protocol, cast

from mlx_model_doctor.errors import TargetError
from mlx_model_doctor.safetensors_header import (
    _MAX_HEADER_BYTES,
    FileHeader,
    HfSafetensorsRepoMetadata,
    SafetensorsHeader,
    SafetensorsHeaderError,
    build_local_header,
    map_hf_repo_metadata,
    parse_file_header,
)


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


class HfSiblingProtocol(Protocol):
    """Hugging Face repository sibling metadata."""

    rfilename: str
    size: int | None


class HfModelInfoProtocol(Protocol):
    """Hugging Face model info metadata used by the target."""

    siblings: Sequence[HfSiblingProtocol]


class HfHubProtocol(Protocol):
    """Small Hugging Face Hub adapter boundary."""

    def model_info(self, repo_id: str, *, files_metadata: bool) -> HfModelInfoProtocol:
        """Return model repository metadata."""

    def download_bytes(self, repo_id: str, filename: str) -> bytes:
        """Download a single repository file as bytes."""

    def safetensors_metadata(self, repo_id: str) -> HfSafetensorsRepoMetadata | None:
        """Return safetensors header metadata via Range requests, or None when absent."""


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
            return cast("HfSafetensorsRepoMetadata", HfApi().get_safetensors_metadata(repo_id))
        except (NotASafetensorsRepoError, SafetensorsParsingError):
            return None
        except Exception as exc:
            raise TargetError(
                f"Could not read safetensors metadata for {repo_id}: {exc}",
                target=repo_id,
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
        """Return whether a file exists at a model-relative path."""
        return self._path(path).is_file()

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

    def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        """Read bytes from a model-relative path."""
        file_path = self._path(path)
        try:
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
        """Read and aggregate safetensors headers from disk without reading weights."""
        shard_paths = sorted(p for p in self.list_files() if p.endswith(".safetensors"))
        if not shard_paths:
            return None
        weight_map = self._safetensors_index_weight_map()
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

    def _safetensors_index_weight_map(self) -> dict[str, str] | None:
        index_paths = [p for p in self.list_files() if p.endswith(".safetensors.index.json")]
        if not index_paths:
            return None
        merged: dict[str, str] = {}
        for index_path in index_paths:
            try:
                parsed = json.loads(self.read_text(index_path))
            except (json.JSONDecodeError, UnicodeError, TargetError):
                continue
            weight_map = parsed.get("weight_map") if isinstance(parsed, dict) else None
            if isinstance(weight_map, dict):
                merged.update(
                    {key: value for key, value in weight_map.items() if isinstance(value, str)}
                )
        return merged or None

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

    __slots__ = ("_hub", "_metadata", "_repo_id")

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

    @property
    def name(self) -> str:
        """Return the Hugging Face repository ID."""
        return self._repo_id

    @property
    def source(self) -> Literal["hf"]:
        """Return the Hugging Face source kind."""
        return "hf"

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
        """Download and read bytes for one metadata-listed repository file."""
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
        """Map the Hugging Face safetensors header metadata, no weight download."""
        repo_meta = self._hub.safetensors_metadata(self._repo_id)
        if repo_meta is None:
            return None
        file_sizes = {name: self._metadata.get(name) for name in repo_meta.files_metadata}
        return map_hf_repo_metadata(repo_meta, file_sizes=file_sizes)

    def _target_error(self, message: str, path: str) -> TargetError:
        return TargetError(message, target=f"{self._repo_id}:{path}", source=self.source)
