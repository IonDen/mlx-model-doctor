from dataclasses import replace

import pytest

from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.errors import TargetError
from mlx_model_doctor.report import CheckResult
from mlx_model_doctor.runners.static import run_static_checks
from tests.fakes import FakeTarget, check_options


def test_static_runner_collects_passing_check_result() -> None:
    ctx = CheckContext(target=FakeTarget(files={}), options=check_options())

    results = run_static_checks(ctx, (PassingCheck(),))

    assert [result.status for result in results] == ["pass"]
    assert results[0].check_id == "text/passing"


def test_static_runner_converts_unexpected_check_crash_to_failure() -> None:
    ctx = CheckContext(target=FakeTarget(files={}), options=check_options())

    results = run_static_checks(ctx, (CrashingCheck(), PassingCheck()))

    assert [result.status for result in results] == ["fail", "pass"]
    assert results[0].check_id == "text/crashing"
    assert results[0].severity == "high"
    assert "check crashed: boom" in results[0].message
    assert "traceback" not in results[0].details


def test_static_runner_includes_traceback_for_verbose_crashes_only() -> None:
    normal_ctx = CheckContext(target=FakeTarget(files={}), options=check_options())
    verbose_ctx = CheckContext(
        target=FakeTarget(files={}),
        options=replace(check_options(), verbosity="verbose"),
    )

    normal_result = run_static_checks(normal_ctx, (CrashingCheck(),))[0]
    verbose_result = run_static_checks(verbose_ctx, (CrashingCheck(),))[0]

    assert "traceback" not in normal_result.details
    assert "traceback" in verbose_result.details
    assert "RuntimeError: boom" in str(verbose_result.details["traceback"])


def test_static_runner_propagates_model_doctor_errors() -> None:
    ctx = CheckContext(target=FakeTarget(files={}), options=check_options())

    with pytest.raises(TargetError, match="bad target"):
        run_static_checks(ctx, (TargetErrorCheck(),))


class PassingCheck:
    check_id = "text/passing"
    title = "Passing"

    def run(self, _ctx: CheckContext) -> CheckResult:
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message="ok",
        )


class CrashingCheck:
    check_id = "text/crashing"
    title = "Crashing"

    def run(self, _ctx: CheckContext) -> CheckResult:
        raise RuntimeError("boom")


class TargetErrorCheck:
    check_id = "text/target-error"
    title = "Target error"

    def run(self, _ctx: CheckContext) -> CheckResult:
        raise TargetError("bad target", target="fake", source="local")
