"""Base protocol for model checks."""

from typing import Protocol

from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.report import CheckResult


class ModelCheck(Protocol):
    """Protocol implemented by model repository checks."""

    check_id: str
    title: str

    def run(self, ctx: CheckContext) -> CheckResult:
        """Run the check against a model target context."""
