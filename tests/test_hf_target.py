from dataclasses import dataclass

import pytest

from mlx_model_doctor.errors import TargetError
from mlx_model_doctor.safetensors_header import SafetensorsHeader as _STHeader
from mlx_model_doctor.targets import HfTarget, MlxListingMetadata
from mlx_model_doctor.targets import HfTarget as _HfTargetForST


@dataclass(frozen=True, slots=True)
class FakeSibling:
    rfilename: str
    size: int | None


@dataclass(frozen=True, slots=True)
class FakeModelInfo:
    siblings: tuple[FakeSibling, ...]
    tags: tuple[str, ...] | None = None
    library_name: str | None = None


class FakeHub:
    def __init__(
        self,
        *,
        files: dict[str, bytes] | None = None,
        sizes: dict[str, int | None] | None = None,
        model_info_error: Exception | None = None,
        download_error: Exception | None = None,
        safetensors: object | None = None,
        tags: tuple[str, ...] | None = None,
        library_name: str | None = None,
    ) -> None:
        self.files = files if files is not None else {}
        self.sizes = (
            sizes if sizes is not None else {path: len(data) for path, data in self.files.items()}
        )
        self.model_info_error = model_info_error
        self.download_error = download_error
        self.safetensors = safetensors
        self.tags = tags
        self.library_name = library_name
        self.model_info_calls: list[tuple[str, bool]] = []
        self.download_calls: list[tuple[str, str]] = []

    def model_info(self, repo_id: str, *, files_metadata: bool) -> FakeModelInfo:
        self.model_info_calls.append((repo_id, files_metadata))
        if self.model_info_error is not None:
            raise self.model_info_error
        return FakeModelInfo(
            siblings=tuple(
                FakeSibling(rfilename=path, size=size) for path, size in self.sizes.items()
            ),
            tags=self.tags,
            library_name=self.library_name,
        )

    def download_bytes(self, repo_id: str, filename: str) -> bytes:
        self.download_calls.append((repo_id, filename))
        if self.download_error is not None:
            raise self.download_error
        return self.files[filename]

    def safetensors_metadata(self, repo_id: str) -> object | None:
        return self.safetensors


def test_hf_target_loads_model_info_once_with_file_metadata() -> None:
    hub = FakeHub(files={"config.json": b"{}"})

    target = HfTarget("org/model", hub=hub)

    assert target.name == "org/model"
    assert target.source == "hf"
    assert hub.model_info_calls == [("org/model", True)]


def test_hf_target_metadata_methods_do_not_download_and_list_files_sorted() -> None:
    hub = FakeHub(
        files={
            "z-tokenizer.json": b"{}",
            "config.json": b"{}",
            "weights.safetensors": b"1234",
        },
        sizes={"z-tokenizer.json": 2, "config.json": 2, "weights.safetensors": None},
    )
    target = HfTarget("org/model", hub=hub)

    assert target.exists("config.json")
    assert not target.exists("missing.json")
    assert target.size("weights.safetensors") is None
    assert target.size("missing.json") is None
    assert target.list_files() == ("config.json", "weights.safetensors", "z-tokenizer.json")
    assert hub.download_calls == []


def test_hf_target_read_text_downloads_only_requested_file() -> None:
    hub = FakeHub(files={"config.json": b'{"model_type":"llama"}', "tokenizer.json": b"{}"})
    target = HfTarget("org/model", hub=hub)

    assert target.read_text("config.json") == '{"model_type":"llama"}'

    assert hub.download_calls == [("org/model", "config.json")]


def test_hf_target_read_bytes_max_bytes_slices_bytes_not_characters() -> None:
    hub = FakeHub(files={"tokenizer.json": "éabc".encode()})
    target = HfTarget("org/model", hub=hub)

    assert target.read_bytes("tokenizer.json", max_bytes=3) == b"\xc3\xa9a"


def test_hf_target_model_info_external_error_wraps_to_target_error() -> None:
    hub = FakeHub(model_info_error=RuntimeError("private repo"))

    with pytest.raises(TargetError, match="Could not inspect Hugging Face model") as exc_info:
        HfTarget("org/private", hub=hub)

    assert exc_info.value.source == "hf"
    assert exc_info.value.target == "org/private"


def test_hf_target_download_external_error_wraps_to_target_error() -> None:
    hub = FakeHub(files={"config.json": b"{}"}, download_error=RuntimeError("rate limited"))
    target = HfTarget("org/model", hub=hub)

    with pytest.raises(TargetError, match="Could not read Hugging Face model file") as exc_info:
        target.read_bytes("config.json")

    assert exc_info.value.source == "hf"
    assert exc_info.value.target == "org/model:config.json"


def test_hf_target_unknown_path_fails_cleanly_without_download() -> None:
    hub = FakeHub(files={"config.json": b"{}"})
    target = HfTarget("org/model", hub=hub)

    with pytest.raises(TargetError, match="not listed"):
        target.read_text("missing.json")

    assert hub.download_calls == []


class _STInfo:
    def __init__(
        self,
        dtype: str,
        shape: list[int],
        offsets: tuple[int, int],
        count: int,
    ) -> None:
        self.dtype = dtype
        self.shape = shape
        self.data_offsets = offsets
        self.parameter_count = count


class _STFile:
    def __init__(self, tensors: dict[str, _STInfo]) -> None:
        self.tensors = tensors
        self.metadata = {"format": "mlx"}


class _STRepo:
    def __init__(self) -> None:
        self.weight_map = {"w": "model.safetensors"}
        self.sharded = False
        self.files_metadata = {
            "model.safetensors": _STFile({"w": _STInfo("U32", [4, 16], (0, 256), 64)})
        }


def test_hf_target_maps_safetensors_header_with_sibling_file_size() -> None:
    hub = FakeHub(
        files={"config.json": b"{}", "model.safetensors": b"x" * 9000}, safetensors=_STRepo()
    )
    header = _HfTargetForST("org/repo", hub=hub).safetensors_header()
    assert isinstance(header, _STHeader)
    assert header.files[0].file_size == 9000  # from sibling metadata, not the hub object
    entry = header.tensor("w")
    assert entry is not None
    assert entry.dtype == "U32"


def test_hf_target_returns_none_for_non_safetensors_repo() -> None:
    hub = FakeHub(files={"config.json": b"{}"}, safetensors=None)
    assert _HfTargetForST("org/repo", hub=hub).safetensors_header() is None


def test_hf_target_retains_tags_and_library_name() -> None:
    hub = FakeHub(
        files={"config.json": b"{}"},
        tags=("mlx", "text-generation"),
        library_name="mlx-lm",
    )
    target = HfTarget("mlx-community/Foo", hub=hub)
    assert isinstance(target, MlxListingMetadata)
    assert target.tags == frozenset({"mlx", "text-generation"})
    assert target.library_name == "mlx-lm"


def test_hf_target_handles_missing_tags_library() -> None:
    hub = FakeHub(files={"config.json": b"{}"}, tags=None, library_name=None)
    target = HfTarget("org/Bar", hub=hub)
    assert target.tags == frozenset()
    assert target.library_name is None
