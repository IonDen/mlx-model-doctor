import json
import re
from dataclasses import dataclass
from typing import ClassVar

import httpx
import huggingface_hub
import pytest
from huggingface_hub.errors import (
    GatedRepoError,
    RemoteEntryNotFoundError,
    SafetensorsParsingError,
)
from huggingface_hub.utils import _tqdm as hf_tqdm

from mlx_model_doctor.errors import TargetError
from mlx_model_doctor.safetensors_header import SafetensorsHeader as _STHeader
from mlx_model_doctor.safetensors_header import SafetensorsHeaderError
from mlx_model_doctor.targets import DefaultHfHub, HfTarget, MlxListingMetadata
from mlx_model_doctor.targets import HfTarget as _HfTargetForST
from tests.fakes import FakeShardHub, safetensors_bytes, shard_repo


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
        safetensors_error: Exception | None = None,
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
        self.safetensors_error = safetensors_error
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
        if self.safetensors_error is not None:
            raise self.safetensors_error
        return self.safetensors

    def safetensors_file_metadata(self, repo_id: str, filename: str) -> object:
        raise AssertionError(f"unexpected per-file header read: {filename}")


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


def test_hf_target_safetensors_header_propagates_target_error() -> None:
    # A Hub/network failure while reading the safetensors header must surface as an
    # hf TargetError, never be swallowed into a "no safetensors" None (which would
    # hide a network/auth fault as a clean absence). safetensors_header() has no
    # try/except, so adding a bare `except: return None` there is what this catches.
    hub = FakeHub(
        files={"config.json": b"{}"},
        safetensors_error=TargetError("hub header read failed", target="org/repo", source="hf"),
    )
    target = HfTarget("org/repo", hub=hub)

    with pytest.raises(TargetError, match="hub header read failed") as exc_info:
        target.safetensors_header()
    assert exc_info.value.source == "hf"


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


# --- stale / partial root index (0080, 0082) -------------------------------------------

_INDEX = "model.safetensors.index.json"
_OLD_SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
_NEW_SHARDS = {
    "model-00001-of-00003.safetensors": {"a": ("BF16", (2, 3))},
    "model-00002-of-00003.safetensors": {"b": ("BF16", (5,))},
    "model-00003-of-00003.safetensors": {"c": ("F32", (7,))},
}


def _index_bytes(weight_map: dict[str, str]) -> bytes:
    return json.dumps({"weight_map": weight_map}).encode()


def _stale_hub() -> FakeShardHub:
    return FakeShardHub(
        files={
            "config.json": b"{}",
            _INDEX: _index_bytes({"x": _OLD_SHARDS[0], "y": _OLD_SHARDS[1]}),
        },
        shards=dict(_NEW_SHARDS),
    )


def test_stale_index_header_reads_shipped_shards_not_the_index() -> None:
    # Bug caught: the stale path still calling get_safetensors_metadata (the bare 404),
    # or reading a shard the index names instead of the shipped ones.
    hub = _stale_hub()

    header = HfTarget("org/stale", hub=hub).safetensors_header()

    assert header is not None
    assert hub.safetensors_metadata_calls == []
    assert hub.file_metadata_calls == sorted(_NEW_SHARDS)
    assert [fh.filename for fh in header.files] == sorted(_NEW_SHARDS)


def test_stale_index_header_ignores_the_stale_weight_map() -> None:
    # Bug caught: keeping the stale weight_map, so param_count warns about tensors x/y
    # that no shipped shard holds.
    header = HfTarget("org/stale", hub=_stale_hub()).safetensors_header()

    assert header is not None
    assert dict(header.weight_map) == {
        "a": "model-00001-of-00003.safetensors",
        "b": "model-00002-of-00003.safetensors",
        "c": "model-00003-of-00003.safetensors",
    }
    assert header.sharded is True
    assert header.total_stored_element_count() == 18  # 2*3 + 5 + 7


def test_stale_index_header_threads_sibling_sizes() -> None:
    # Bug caught: dropping the sibling size, so the offsets upper bound can never be checked.
    hub = _stale_hub()

    header = HfTarget("org/stale", hub=hub).safetensors_header()

    assert header is not None
    sizes = {fh.filename: fh.file_size for fh in header.files}
    assert sizes == {name: len(safetensors_bytes(spec)) for name, spec in _NEW_SHARDS.items()}


def test_partial_index_raises_header_error_naming_sorted_absent_shards() -> None:
    # Bug caught: a partly missing index raising a TargetError (exit 2, the 0080 bug), or
    # calling get_safetensors_metadata, which 404s on a random missing shard.
    hub = FakeShardHub(
        files={
            "config.json": b"{}",
            _INDEX: _index_bytes(
                {
                    "z": "model-00003-of-00003.safetensors",
                    "a": "model-00001-of-00003.safetensors",
                    "m": "model-00002-of-00003.safetensors",
                }
            ),
        },
        shards={"model-00002-of-00003.safetensors": {"m": ("BF16", (4,))}},
    )

    with pytest.raises(SafetensorsHeaderError) as exc_info:
        HfTarget("org/partial", hub=hub).safetensors_header()

    assert not isinstance(exc_info.value, TargetError)
    assert str(exc_info.value) == (
        "safetensors index model.safetensors.index.json names 2 shards absent from the "
        "repository: model-00001-of-00003.safetensors, model-00003-of-00003.safetensors"
    )
    assert hub.safetensors_metadata_calls == []
    assert hub.file_metadata_calls == []


def test_all_absent_index_with_nothing_shipped_raises_header_error() -> None:
    # Bug caught: an index whose shards are all gone, with no other shard, read as "no
    # safetensors" (None) instead of missing weights.
    hub = FakeShardHub(
        files={"config.json": b"{}", _INDEX: _index_bytes({"x": _OLD_SHARDS[0]})},
        shards={},
    )

    with pytest.raises(SafetensorsHeaderError, match="names 1 shard absent"):
        HfTarget("org/gone", hub=hub).safetensors_header()
    assert hub.safetensors_metadata_calls == []


def test_consistent_index_keeps_using_repo_metadata() -> None:
    # Bug caught: the consistent path switched to per-file reads (one request per shard
    # where one aggregated call was enough), or skipped the repo metadata.
    hub = FakeShardHub(
        files={
            "config.json": b"{}",
            _INDEX: _index_bytes({"a": "model-00001-of-00001.safetensors"}),
        },
        shards={"model-00001-of-00001.safetensors": {"a": ("BF16", (4,))}},
    )

    header = HfTarget("org/ok", hub=hub).safetensors_header()

    assert header is not None
    assert hub.safetensors_metadata_calls == ["org/ok"]
    assert hub.file_metadata_calls == []


def _many_shards_hub(count: int) -> FakeShardHub:
    shards = {
        f"model-{i:05d}-of-{count:05d}.safetensors": {f"t{i}": ("F32", (1,))} for i in range(count)
    }
    return FakeShardHub(
        files={"config.json": b"{}", _INDEX: _index_bytes({"x": _OLD_SHARDS[0]})},
        shards=shards,
    )


def test_stale_index_with_513_shipped_shards_raises_without_any_request() -> None:
    # Bug caught (S1): one range request per shipped shard with no cap, so a repo
    # listing thousands of model*.safetensors files makes check hf crawl.
    hub = _many_shards_hub(513)

    with pytest.raises(SafetensorsHeaderError) as exc_info:
        HfTarget("org/many", hub=hub).safetensors_header()

    assert str(exc_info.value) == (
        "safetensors index model.safetensors.index.json is stale and the repository ships "
        "513 shards, more than the 512 whose headers are read"
    )
    assert hub.file_metadata_calls == []
    assert hub.safetensors_metadata_calls == []


def test_stale_index_with_512_shipped_shards_reads_them_all() -> None:
    # Bug caught: the cap off by one (512 shards refused).
    hub = _many_shards_hub(512)

    header = HfTarget("org/many", hub=hub).safetensors_header()

    assert header is not None
    assert len(hub.file_metadata_calls) == 512


def test_root_index_of_unknown_size_falls_back_to_repo_metadata() -> None:
    # Bug caught (T6): an index whose listing size is unknown downloaded anyway (an
    # unbounded read) instead of treated as no usable index.
    files, shards = shard_repo("consistent")
    hub = FakeShardHub(files=files, shards=shards, unknown_sizes=frozenset({_INDEX}))

    header = HfTarget("org/ok", hub=hub).safetensors_header()

    assert header is not None
    assert hub.download_calls == []
    assert hub.safetensors_metadata_calls == ["org/ok"]
    assert hub.file_metadata_calls == []


def test_index_download_failure_stays_a_target_error() -> None:
    # Bug caught: a network failure reading the index swallowed into "no index", which
    # would hide a Hub fault behind a clean report.
    hub = FakeHub(
        files={"config.json": b"{}", _INDEX: _index_bytes({"a": "m.safetensors"})},
        download_error=RuntimeError("connection reset"),
    )

    with pytest.raises(TargetError, match="connection reset") as exc_info:
        HfTarget("org/repo", hub=hub).safetensors_header()
    assert exc_info.value.source == "hf"


def test_stale_shard_metadata_network_failure_stays_a_target_error() -> None:
    # Bug caught: a range-read network failure on a shipped shard turned into a soft
    # header error (a Hub fault masquerading as a broken file).
    hub = _stale_hub()
    hub.file_metadata_error = TargetError("range read failed", target="org/stale", source="hf")

    with pytest.raises(TargetError, match="range read failed"):
        HfTarget("org/stale", hub=hub).safetensors_header()


# --- DefaultHfHub: backstop, per-file metadata, progress bars ---------------------------


def _http_error(cls: type[Exception], status: int, url: str) -> Exception:
    response = httpx.Response(status, request=httpx.Request("GET", url))
    return cls(f"{status} Client Error.\n\nfor url: {url}.", response=response)


_SHARD_URL = "https://huggingface.co/org/r/resolve/main/model-00001-of-00002.safetensors"


@pytest.fixture
def isolated_progress_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    # Progress-bar state is process-global in huggingface_hub; give each test its own,
    # independent of any HF_HUB_DISABLE_PROGRESS_BARS in the environment.
    monkeypatch.setattr(hf_tqdm, "progress_bar_states", {})
    monkeypatch.setattr(hf_tqdm, "HF_HUB_DISABLE_PROGRESS_BARS", None)


class _RaisingApi:
    error: ClassVar[Exception | None] = None
    bars_disabled_during_call: ClassVar[list[bool]] = []

    def get_safetensors_metadata(self, repo_id: str) -> object:
        _RaisingApi.bars_disabled_during_call.append(hf_tqdm.are_progress_bars_disabled())
        if _RaisingApi.error is not None:
            raise _RaisingApi.error
        return "repo-metadata"

    def parse_safetensors_file_metadata(self, repo_id: str, filename: str) -> object:
        _RaisingApi.bars_disabled_during_call.append(hf_tqdm.are_progress_bars_disabled())
        if _RaisingApi.error is not None:
            raise _RaisingApi.error
        return f"file-metadata:{filename}"


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch, isolated_progress_bars: None) -> type[_RaisingApi]:
    monkeypatch.setattr(huggingface_hub, "HfApi", _RaisingApi)
    monkeypatch.setattr(_RaisingApi, "error", None)
    monkeypatch.setattr(_RaisingApi, "bars_disabled_during_call", [])
    return _RaisingApi


def test_backstop_maps_missing_shard_entry_to_header_error(fake_api: type[_RaisingApi]) -> None:
    # Bug caught: a 404 on an index-named shard escaping as a run-killing TargetError.
    fake_api.error = _http_error(RemoteEntryNotFoundError, 404, _SHARD_URL)

    with pytest.raises(
        SafetensorsHeaderError, match=re.escape("model-00001-of-00002.safetensors")
    ) as exc:
        DefaultHfHub().safetensors_metadata("org/r")
    assert not isinstance(exc.value, TargetError)


@pytest.mark.parametrize(
    "error",
    [
        _http_error(
            RemoteEntryNotFoundError,
            404,
            "https://huggingface.co/org/r/resolve/main/model.safetensors.index.json",
        ),
        _http_error(GatedRepoError, 401, _SHARD_URL),
        ConnectionError("connection refused"),
    ],
    ids=["missing-index-file", "gated-auth", "connection"],
)
def test_backstop_keeps_other_hub_failures_as_target_errors(
    fake_api: type[_RaisingApi], error: Exception
) -> None:
    # Bug caught: the backstop widened past "a .safetensors entry is missing", so an
    # auth or network failure masquerades as a clean header verdict.
    fake_api.error = error

    with pytest.raises(TargetError) as exc_info:
        DefaultHfHub().safetensors_metadata("org/r")
    assert exc_info.value.source == "hf"


def test_file_metadata_returns_the_parsed_header(fake_api: type[_RaisingApi]) -> None:
    # Bug caught: the adapter calling the wrong file or dropping the result.
    assert DefaultHfHub().safetensors_file_metadata("org/r", "a.safetensors") == (
        "file-metadata:a.safetensors"
    )


@pytest.mark.parametrize(
    "error",
    [
        _http_error(RemoteEntryNotFoundError, 404, _SHARD_URL),
        SafetensorsParsingError("bad header"),
    ],
    ids=["entry-not-found", "unparseable"],
)
def test_file_metadata_header_problems_are_header_errors(
    fake_api: type[_RaisingApi], error: Exception
) -> None:
    # Bug caught: a missing or corrupt shard header killing the run instead of being
    # reported by the offsets check.
    fake_api.error = error

    with pytest.raises(SafetensorsHeaderError):
        DefaultHfHub().safetensors_file_metadata("org/r", "model-00001-of-00002.safetensors")


def test_file_metadata_network_failure_is_a_target_error(fake_api: type[_RaisingApi]) -> None:
    # Bug caught: a network failure on a range read hidden as a header problem.
    fake_api.error = ConnectionError("connection refused")

    with pytest.raises(TargetError, match="connection refused"):
        DefaultHfHub().safetensors_file_metadata("org/r", "a.safetensors")


def test_metadata_calls_run_with_progress_bars_off_and_restore_them(
    fake_api: type[_RaisingApi],
) -> None:
    # Bug caught: the "Parse safetensors files" bar leaking to stderr, or the bars left
    # disabled for the caller after the call.
    DefaultHfHub().safetensors_metadata("org/r")
    DefaultHfHub().safetensors_file_metadata("org/r", "a.safetensors")

    assert fake_api.bars_disabled_during_call == [True, True]
    assert hf_tqdm.are_progress_bars_disabled() is False


def test_progress_bars_are_restored_after_an_exception(fake_api: type[_RaisingApi]) -> None:
    # Bug caught: restoring the bars only on success (no finally), leaving them off.
    fake_api.error = ConnectionError("down")

    with pytest.raises(TargetError):
        DefaultHfHub().safetensors_metadata("org/r")

    assert fake_api.bars_disabled_during_call == [True]
    assert hf_tqdm.are_progress_bars_disabled() is False


def test_progress_bars_already_disabled_stay_disabled(fake_api: type[_RaisingApi]) -> None:
    # Bug caught: unconditionally re-enabling bars the caller had turned off.
    hf_tqdm.disable_progress_bars()

    DefaultHfHub().safetensors_metadata("org/r")

    assert hf_tqdm.are_progress_bars_disabled() is True


def test_missing_entry_error_without_a_response_stays_a_target_error(
    fake_api: type[_RaisingApi],
) -> None:
    # Bug caught (F4): reading exc.response.url unguarded, so an entry error without a
    # response raises AttributeError out of the adapter instead of a tool error.
    error = _http_error(RemoteEntryNotFoundError, 404, _SHARD_URL)
    error.response = None  # type: ignore[assignment]
    fake_api.error = error

    with pytest.raises(TargetError) as exc_info:
        DefaultHfHub().safetensors_metadata("org/r")
    assert exc_info.value.source == "hf"


def test_missing_entry_error_whose_response_has_no_url_stays_a_target_error(
    fake_api: type[_RaisingApi],
) -> None:
    # Bug caught (F4): a response with no request (httpx raises RuntimeError on .url)
    # escaping the adapter.
    error = _http_error(RemoteEntryNotFoundError, 404, _SHARD_URL)
    error.response = httpx.Response(404)
    fake_api.error = error

    with pytest.raises(TargetError):
        DefaultHfHub().safetensors_metadata("org/r")


def test_file_metadata_404_on_a_listed_shard_says_the_listing_includes_it(
    fake_api: type[_RaisingApi],
) -> None:
    # Bug caught (R4): calling a shard "absent" that the repository listing shows; the
    # per-file reads only ever ask for listed shards, so a 404 is a Hub inconsistency.
    fake_api.error = _http_error(RemoteEntryNotFoundError, 404, _SHARD_URL)

    with pytest.raises(SafetensorsHeaderError) as exc_info:
        DefaultHfHub().safetensors_file_metadata("org/r", "model-00001-of-00002.safetensors")

    assert str(exc_info.value) == (
        "safetensors shard model-00001-of-00002.safetensors returned 404 although the "
        "repository listing includes it"
    )


@pytest.mark.parametrize("fails", [False, True], ids=["success", "exception"])
def test_group_progress_bar_state_survives_the_block(
    fake_api: type[_RaisingApi], fails: bool
) -> None:
    # Bug caught (C4): disable/enable_progress_bars() clearing progress_bar_states, so a
    # library caller's per-group setting is lost after check_hf_model.
    hf_tqdm.disable_progress_bars("huggingface_hub.http_get")
    before = dict(hf_tqdm.progress_bar_states)
    if fails:
        fake_api.error = ConnectionError("down")

    try:
        DefaultHfHub().safetensors_metadata("org/r")
    except TargetError:
        assert fails

    assert fake_api.bars_disabled_during_call == [True]
    assert hf_tqdm.progress_bar_states == before == {"huggingface_hub.http_get": False}
    assert hf_tqdm.are_progress_bars_disabled("huggingface_hub.http_get") is True
    assert hf_tqdm.are_progress_bars_disabled() is False
