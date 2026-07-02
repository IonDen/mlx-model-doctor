"""Shared check runner primitives."""

import traceback
from collections.abc import Sequence

from mlx_model_doctor.checks.base import ModelCheck
from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.errors import ModelDoctorError
from mlx_model_doctor.report import CheckResult


def _safe_check_id(check: ModelCheck) -> str:
    """Return a plugin-namespaced check_id, sanitizing a malformed one."""
    raw = getattr(check, "check_id", "") or ""
    namespace, separator, name = raw.partition("/")
    if namespace and separator and name:
        return raw
    return f"unknown/{name or namespace or 'check'}"


def run_checks(ctx: CheckContext, checks: Sequence[ModelCheck]) -> list[CheckResult]:
    """Run checks, isolating unexpected crashes as failed check results."""
    results: list[CheckResult] = []
    for check in checks:
        try:
            results.append(check.run(ctx))
        except ModelDoctorError:
            raise
        except Exception as exc:
            details: dict[str, object] = {}
            if ctx.options.verbosity == "verbose":
                details["traceback"] = traceback.format_exc()
            results.append(
                CheckResult(
                    check_id=_safe_check_id(check),
                    title=check.title,
                    status="fail",
                    severity="high",
                    message=f"check crashed: {exc}",
                    details=details,
                )
            )
    return results
