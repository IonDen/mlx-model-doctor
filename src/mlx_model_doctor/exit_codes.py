"""Pure exit-code decisions."""

from typing import Literal

from mlx_model_doctor.errors import ModelDoctorError
from mlx_model_doctor.report import DoctorReport

FailOn = Literal["error", "warn", "never"]
VALID_FAIL_ON = frozenset(("error", "warn", "never"))


def exit_code_for(report: DoctorReport, *, fail_on: FailOn) -> int:
    """Return the process exit code for a completed report."""
    if fail_on not in VALID_FAIL_ON:
        msg = f"fail_on must be one of {sorted(VALID_FAIL_ON)}"
        raise ValueError(msg)
    if not report.results:
        return 2
    if fail_on == "never":
        return 0
    if report.summary["fail"]:
        return 1
    if fail_on == "warn" and report.summary["warn"]:
        return 1
    return 0


def exit_code_for_error(error: ModelDoctorError) -> int:
    """Return the process exit code for a tool-level error."""
    return 2
