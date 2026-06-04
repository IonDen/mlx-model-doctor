"""Static check runner."""

from collections.abc import Sequence

from mlx_model_doctor.checks.base import ModelCheck
from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.report import CheckResult
from mlx_model_doctor.runners.core import run_checks


def run_static_checks(ctx: CheckContext, checks: Sequence[ModelCheck]) -> list[CheckResult]:
    """Run static checks, isolating unexpected check crashes."""
    return run_checks(ctx, checks)
