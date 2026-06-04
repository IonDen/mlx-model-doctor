import json

import pytest

from mlx_model_doctor.checks.safetensors import SafetensorsIndexCheck
from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.errors import TargetError
from tests.fakes import FakeTarget, check_options, context_for_files


def test_safetensors_index_check_skips_when_index_missing() -> None:
    result = SafetensorsIndexCheck().run(context_for_files({"model.safetensors": b"weights"}))

    assert result.status == "skip"
    assert result.severity == "info"
    assert "index" in result.message


def test_safetensors_index_check_fails_when_referenced_shard_is_missing() -> None:
    index = {"weight_map": {"model.embed_tokens.weight": "missing-00001.safetensors"}}

    result = SafetensorsIndexCheck().run(
        context_for_files({"model.safetensors.index.json": json.dumps(index).encode()})
    )

    assert result.status == "fail"
    assert result.severity == "high"
    assert "missing-00001.safetensors" in result.message
    assert result.remediation is not None


def test_safetensors_index_check_passes_when_referenced_shard_exists() -> None:
    index = {"weight_map": {"model.embed_tokens.weight": "model-00001-of-00001.safetensors"}}
    target = NoShardReadTarget(
        files={
            "model.safetensors.index.json": json.dumps(index).encode(),
            "model-00001-of-00001.safetensors": b"do-not-read",
        }
    )

    result = SafetensorsIndexCheck().run(CheckContext(target=target, options=check_options()))

    assert result.status == "pass"
    assert result.severity == "info"
    assert result.details["shards"] == ("model-00001-of-00001.safetensors",)


def test_safetensors_index_check_sorts_discovered_indexes_and_shards() -> None:
    first_index = {"weight_map": {"a.weight": "z-shard.safetensors"}}
    second_index = {"weight_map": {"b.weight": "a-shard.safetensors"}}
    target = UnsortedListTarget(
        files={
            "z-model.safetensors.index.json": json.dumps(first_index).encode(),
            "a-model.safetensors.index.json": json.dumps(second_index).encode(),
            "z-shard.safetensors": b"z",
            "a-shard.safetensors": b"a",
        },
        listed_paths=(
            "z-model.safetensors.index.json",
            "z-shard.safetensors",
            "a-model.safetensors.index.json",
            "a-shard.safetensors",
        ),
    )

    result = SafetensorsIndexCheck().run(CheckContext(target=target, options=check_options()))

    assert result.status == "pass"
    assert result.details["index_paths"] == (
        "a-model.safetensors.index.json",
        "z-model.safetensors.index.json",
    )
    assert result.details["shards"] == ("a-shard.safetensors", "z-shard.safetensors")


def test_safetensors_index_check_warns_when_listed_index_disappears() -> None:
    target = DisappearingIndexTarget(files={"model.safetensors.index.json": b'{"weight_map":{}}'})

    result = SafetensorsIndexCheck().run(CheckContext(target=target, options=check_options()))

    assert result.status == "warn"
    assert result.severity == "medium"
    assert "model.safetensors.index.json" in result.message
    assert result.details["index_path"] == "model.safetensors.index.json"


def test_safetensors_index_check_propagates_hf_index_read_error() -> None:
    target = HfIndexReadErrorTarget(
        files={"model.safetensors.index.json": b'{"weight_map":{}}'},
        _source="hf",
    )

    with pytest.raises(TargetError, match="index download failed") as exc_info:
        SafetensorsIndexCheck().run(CheckContext(target=target, options=check_options()))

    assert exc_info.value.source == "hf"


def test_safetensors_index_check_discovers_non_default_index_name() -> None:
    index = {"weight_map": {"adapter.layer.weight": "adapter_model.safetensors"}}

    result = SafetensorsIndexCheck().run(
        context_for_files(
            {
                "adapter_model.safetensors.index.json": json.dumps(index).encode(),
                "adapter_model.safetensors": b"adapter weights",
            }
        )
    )

    assert result.status == "pass"
    assert result.severity == "info"
    assert result.details["index_path"] == "adapter_model.safetensors.index.json"
    assert result.details["shards"] == ("adapter_model.safetensors",)


def test_safetensors_index_check_fails_when_shard_reference_is_invalid() -> None:
    index = {"weight_map": {"model.embed_tokens.weight": "../outside.safetensors"}}
    target = InvalidShardPathTarget(
        files={"model.safetensors.index.json": json.dumps(index).encode()}
    )

    result = SafetensorsIndexCheck().run(CheckContext(target=target, options=check_options()))

    assert result.status == "fail"
    assert result.severity == "high"
    assert "../outside.safetensors" in result.message
    assert result.remediation is not None
    assert result.details["missing_shards"] == ("../outside.safetensors",)


def test_safetensors_index_check_fails_on_invalid_json() -> None:
    result = SafetensorsIndexCheck().run(
        context_for_files({"model.safetensors.index.json": b"{not-json"})
    )

    assert result.status == "fail"
    assert result.severity == "high"
    assert "invalid JSON" in result.message
    assert result.remediation is not None


def test_safetensors_index_check_warns_when_index_is_not_an_object() -> None:
    result = SafetensorsIndexCheck().run(context_for_files({"model.safetensors.index.json": b"[]"}))

    assert result.status == "warn"
    assert result.severity == "medium"
    assert "object" in result.message
    assert result.remediation is not None


def test_safetensors_index_check_warns_when_weight_map_is_missing() -> None:
    result = SafetensorsIndexCheck().run(
        context_for_files({"model.safetensors.index.json": b'{"metadata":{}}'})
    )

    assert result.status == "warn"
    assert result.severity == "medium"
    assert "weight_map" in result.message
    assert result.remediation is not None


class NoShardReadTarget(FakeTarget):
    def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        if path.endswith(".safetensors"):
            raise AssertionError("SafetensorsIndexCheck must not read shard contents")
        return super().read_bytes(path, max_bytes=max_bytes)


class UnsortedListTarget(FakeTarget):
    listed_paths: tuple[str, ...]

    def __init__(self, *, files: dict[str, bytes], listed_paths: tuple[str, ...]) -> None:
        super().__init__(files=files)
        self.listed_paths = listed_paths

    def list_files(self) -> tuple[str, ...]:
        return self.listed_paths


class DisappearingIndexTarget(FakeTarget):
    def read_text(self, path: str, *, max_bytes: int | None = None) -> str:
        if path == "model.safetensors.index.json":
            raise FileNotFoundError(path)
        return super().read_text(path, max_bytes=max_bytes)


class HfIndexReadErrorTarget(FakeTarget):
    def read_text(self, path: str, *, max_bytes: int | None = None) -> str:
        if path == "model.safetensors.index.json":
            raise TargetError("index download failed", target=path, source=self.source)
        return super().read_text(path, max_bytes=max_bytes)


class InvalidShardPathTarget(FakeTarget):
    def exists(self, path: str) -> bool:
        if path == "../outside.safetensors":
            raise TargetError("outside model root", target=path, source=self.source)
        return super().exists(path)
