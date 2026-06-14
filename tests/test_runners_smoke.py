import json
from dataclasses import dataclass, replace

import pytest

import mlx_model_doctor.runners.smoke as smoke_runner
from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.errors import TargetError
from mlx_model_doctor.report import CheckResult
from mlx_model_doctor.runners.smoke import run_smoke_checks
from tests.fakes import FakeTarget, check_options, context_for_files

GIB = 1024**3


def test_smoke_runner_skips_when_smoke_option_is_false() -> None:
    check = RecordingCheck()
    ctx = CheckContext(target=FakeTarget(files={}), options=check_options())

    results = run_smoke_checks(ctx, (check,), ())

    assert results == []
    assert check.calls == 0


def test_smoke_runner_refuses_over_budget_estimate_without_calling_backend() -> None:
    check = RecordingCheck()
    ctx = CheckContext(
        target=FakeTarget(files={}),
        options=replace(check_options(), smoke=True, max_memory_bytes=10 * GIB),
    )
    memory_result = memory_estimate_result(lower_bound_bytes=11 * GIB)

    results = run_smoke_checks(ctx, (check,), (memory_result,))

    assert check.calls == 0
    assert len(results) == 1
    assert results[0].check_id == "text/smoke.memory_budget"
    assert results[0].status == "fail"
    assert results[0].severity == "high"
    assert results[0].details["memory_lower_bound_bytes"] == 11 * GIB
    assert results[0].details["smoke_budget_bytes"] == 10 * GIB


def test_smoke_runner_refuses_marked_over_budget_estimate_with_custom_check_id() -> None:
    check = RecordingCheck()
    ctx = CheckContext(
        target=FakeTarget(files={}),
        options=replace(check_options(), smoke=True, max_memory_bytes=10 * GIB),
    )
    memory_result = memory_estimate_result(
        check_id="custom/runtime.memory_floor",
        lower_bound_bytes=11 * GIB,
    )

    results = run_smoke_checks(ctx, (check,), (memory_result,))

    assert check.calls == 0
    assert len(results) == 1
    assert results[0].check_id == "custom/smoke.memory_budget"
    assert results[0].status == "fail"
    assert results[0].details["memory_lower_bound_bytes"] == 11 * GIB


def test_smoke_runner_ignores_unmarked_lower_bound_details() -> None:
    check = RecordingCheck()
    ctx = CheckContext(
        target=FakeTarget(files={}),
        options=replace(check_options(), smoke=True, max_memory_bytes=10 * GIB),
    )
    memory_result = memory_estimate_result(lower_bound_bytes=11 * GIB, marked=False)

    results = run_smoke_checks(ctx, (check,), (memory_result,))

    assert check.calls == 1
    assert [result.check_id for result in results] == ["text/recording"]


def test_smoke_runner_calls_check_when_estimate_is_under_budget() -> None:
    check = RecordingCheck()
    ctx = CheckContext(
        target=FakeTarget(files={}),
        options=replace(check_options(), smoke=True, max_memory_bytes=10 * GIB),
    )
    memory_result = memory_estimate_result(lower_bound_bytes=8 * GIB)

    results = run_smoke_checks(ctx, (check,), (memory_result,))

    assert check.calls == 1
    assert [result.check_id for result in results] == ["text/recording"]


def test_smoke_runner_uses_mlx_recommended_working_set_budget(monkeypatch) -> None:
    check = RecordingCheck()
    ctx = CheckContext(
        target=FakeTarget(files={}),
        options=replace(check_options(), smoke=True),
    )

    monkeypatch.setattr(
        smoke_runner.importlib,
        "import_module",
        lambda _name: FakeMx({"max_recommended_working_set_size": 10 * GIB}),
    )

    results = run_smoke_checks(ctx, (check,), (memory_estimate_result(lower_bound_bytes=9 * GIB),))

    assert check.calls == 0
    assert results[0].status == "fail"
    assert results[0].details["smoke_budget_bytes"] == 8 * GIB
    assert results[0].details["smoke_budget_source"] == "mlx.max_recommended_working_set_size"


def test_smoke_runner_converts_unexpected_check_crash_to_failure() -> None:
    ctx = CheckContext(
        target=FakeTarget(files={}),
        options=replace(check_options(), smoke=True),
    )

    results = run_smoke_checks(ctx, (CrashingCheck(),), ())

    assert len(results) == 1
    assert results[0].check_id == "text/crashing-smoke"
    assert results[0].status == "fail"
    assert results[0].severity == "high"
    assert "check crashed: boom" in results[0].message


def test_smoke_runner_propagates_model_doctor_errors() -> None:
    ctx = CheckContext(
        target=FakeTarget(files={}),
        options=replace(check_options(), smoke=True),
    )

    with pytest.raises(TargetError, match="bad target"):
        run_smoke_checks(ctx, (TargetErrorCheck(),), ())


def memory_estimate_result(
    *,
    lower_bound_bytes: int,
    check_id: str = "text/memory.estimate",
    marked: bool = True,
) -> CheckResult:
    details: dict[str, object] = {
        "estimate_source": "config",
        "context_length": 4096,
        "lower_bound_bytes": lower_bound_bytes,
    }
    if marked:
        details["memory_lower_bound_kind"] = "model_runtime"
    return CheckResult(
        check_id=check_id,
        title="Memory estimate",
        status="warn",
        severity="low",
        message="estimate",
        details=details,
    )


@dataclass
class RecordingCheck:
    check_id: str = "text/recording"
    title: str = "Recording"
    calls: int = 0

    def run(self, _ctx: CheckContext) -> CheckResult:
        self.calls += 1
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message="ok",
        )


class CrashingCheck:
    check_id = "text/crashing-smoke"
    title = "Crashing smoke"

    def run(self, _ctx: CheckContext) -> CheckResult:
        raise RuntimeError("boom")


class TargetErrorCheck:
    check_id = "text/target-error-smoke"
    title = "Target error smoke"

    def run(self, _ctx: CheckContext) -> CheckResult:
        raise TargetError("bad target", target="fake", source="local")


class FakeMx:
    def __init__(self, device_info: dict[str, object]) -> None:
        self._device_info = device_info

    def device_info(self) -> dict[str, object]:
        return self._device_info


def test_smoke_runner_refuses_mixed_precision_corrected_bound_end_to_end() -> None:
    from mlx_model_doctor.checks.memory import MemoryEstimateCheck

    mixed_config = {
        "hidden_size": 4096,
        "num_hidden_layers": 2,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "quantization": {"bits": 4, "group_size": 64, "model.layers.0.mlp": {"bits": 8}},
    }
    files = {
        "config.json": json.dumps(mixed_config).encode(),
        "model.safetensors": b"a" * 1024,
    }
    options = replace(check_options(), smoke=True, max_memory_bytes=1024)
    ctx = context_for_files(files, options=options)

    memory_result = MemoryEstimateCheck().run(ctx)
    assert memory_result.details["estimate_source"] == "file_sizes"  # corrected path engaged

    check = RecordingCheck()
    results = run_smoke_checks(ctx, (check,), (memory_result,))

    assert check.calls == 0  # the gate refused; the smoke check never ran
    assert len(results) == 1
    assert results[0].check_id == "text/smoke.memory_budget"
    assert results[0].status == "fail"
