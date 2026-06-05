from mlx_model_doctor.checks.weights import WeightParamCountCheck
from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.safetensors_header import FileHeader, SafetensorsHeader, TensorEntry
from tests.fakes import FakeTarget, check_options


def _entry(count: int = 4) -> TensorEntry:
    return TensorEntry(
        dtype="BF16", shape=(count,), data_offsets=(0, count * 2), parameter_count=count
    )


def _header(*, tensors: dict[str, TensorEntry], weight_map: dict[str, str]) -> SafetensorsHeader:
    fh = FileHeader(
        filename="model.safetensors", tensors=tensors, metadata={}, header_length=10, file_size=None
    )
    return SafetensorsHeader(
        files=(fh,),
        weight_map=weight_map,
        sharded=False,
        param_count_by_dtype={"BF16": sum(e.parameter_count for e in tensors.values())},
    )


def _run(header: SafetensorsHeader | None) -> object:
    return WeightParamCountCheck().run(
        CheckContext(
            target=FakeTarget(files={}, _safetensors_header=header), options=check_options()
        )
    )


def test_param_count_pass_when_map_complete_and_nonzero() -> None:
    header = _header(tensors={"w": _entry()}, weight_map={"w": "model.safetensors"})
    assert _run(header).status == "pass"


def test_param_count_warn_on_declared_but_missing_tensor() -> None:
    header = _header(
        tensors={"w": _entry()}, weight_map={"w": "model.safetensors", "ghost": "model.safetensors"}
    )
    result = _run(header)
    assert result.status == "warn"
    assert result.details["missing_tensors"] == ("ghost",)


def test_param_count_warn_on_zero_params() -> None:
    fh = FileHeader(
        filename="model.safetensors", tensors={}, metadata={}, header_length=10, file_size=None
    )
    header = SafetensorsHeader(files=(fh,), weight_map={}, sharded=False, param_count_by_dtype={})
    result = _run(header)
    assert result.status == "warn"
    assert result.details["empty_or_zero_param_files"] == ("model.safetensors",)


def test_param_count_skip_without_header() -> None:
    assert _run(None).status == "skip"
