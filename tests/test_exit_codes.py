import pytest

from mlx_model_doctor.errors import DependencyError, TargetError
from mlx_model_doctor.exit_codes import exit_code_for, exit_code_for_error
from mlx_model_doctor.report import CheckResult, DoctorReport


def report_with(status: str) -> DoctorReport:
    severity = "info" if status in {"pass", "skip"} else "medium"
    return DoctorReport(
        target="fixture",
        source="local",
        plugin="text",
        results=[
            CheckResult(
                check_id=f"text/{status}",
                title=status,
                status=status,
                severity=severity,
                message=status,
            )
        ],
    )


def test_exit_code_for_failures_and_warn_strictness() -> None:
    assert exit_code_for(report_with("pass"), fail_on="error") == 0
    assert exit_code_for(report_with("warn"), fail_on="error") == 0
    assert exit_code_for(report_with("warn"), fail_on="warn") == 1
    assert exit_code_for(report_with("fail"), fail_on="error") == 1
    assert exit_code_for(report_with("fail"), fail_on="never") == 0


def test_exit_code_for_zero_checks_is_tool_error() -> None:
    report = DoctorReport(target="fixture", source="local", plugin="text", results=[])
    assert exit_code_for(report, fail_on="error") == 2


def test_exit_code_for_rejects_unknown_fail_on_policy() -> None:
    with pytest.raises(ValueError, match="fail_on"):
        exit_code_for(report_with("pass"), fail_on="strict")


def test_exit_code_for_tool_errors_is_two() -> None:
    assert exit_code_for_error(TargetError("bad target", target="missing", source="local")) == 2
    assert (
        exit_code_for_error(
            DependencyError(
                missing_package="mlx_lm",
                extra_name="mlx-lm",
                executable="/tmp/.venv/bin/python",
            )
        )
        == 2
    )
