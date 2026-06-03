"""Static check runner."""

import traceback
from collections.abc import Sequence

from mlx_model_doctor.checks.base import ModelCheck
from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.errors import ModelDoctorError
from mlx_model_doctor.report import CheckResult


def run_static_checks(ctx: CheckContext, checks: Sequence[ModelCheck]) -> list[CheckResult]:
    """Run static checks, isolating unexpected check crashes."""
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
                    check_id=check.check_id,
                    title=check.title,
                    status="fail",
                    severity="high",
                    message=f"check crashed: {exc}",
                    details=details,
                )
            )
    return results
