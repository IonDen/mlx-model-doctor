"""Model target abstractions."""

from collections.abc import Sequence
from pathlib import Path
from stat import S_ISREG
from typing import Literal, Protocol

from mlx_model_doctor.errors import TargetError


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
