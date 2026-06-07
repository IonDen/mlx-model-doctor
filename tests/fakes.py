from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from mlx_model_doctor.context import CheckContext, CheckOptions
from mlx_model_doctor.safetensors_header import SafetensorsHeader


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
        data = self.files.get(path)
        return None if data is None else len(data)

    def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        data = self.files[path]
        return data if max_bytes is None else data[:max_bytes]

    def read_text(self, path: str, *, max_bytes: int | None = None) -> str:
        return self.read_bytes(path, max_bytes=max_bytes).decode("utf-8")
