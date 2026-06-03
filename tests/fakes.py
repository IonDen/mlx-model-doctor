from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from mlx_model_doctor.context import CheckContext, CheckOptions


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
) -> CheckContext:
    return CheckContext(target=FakeTarget(files=files, _source=source), options=check_options())


@dataclass(slots=True)
class FakeTarget:
    files: dict[str, bytes]
    name: str = "fake"
    _source: Literal["local", "hf"] = "local"

    @property
    def source(self) -> Literal["local", "hf"]:
        return self._source

    def exists(self, path: str) -> bool:
        return path in self.files

    def list_files(self) -> Sequence[str]:
        return tuple(sorted(self.files))

    def size(self, path: str) -> int | None:
        data = self.files.get(path)
        return None if data is None else len(data)

    def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        data = self.files[path]
        return data if max_bytes is None else data[:max_bytes]

    def read_text(self, path: str, *, max_bytes: int | None = None) -> str:
        return self.read_bytes(path, max_bytes=max_bytes).decode("utf-8")
