"""Checks for required model repository files."""

from dataclasses import dataclass

from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.errors import TargetError, raise_for_hf_target_error
from mlx_model_doctor.report import CheckResult


@dataclass(frozen=True, slots=True)
class RequiredConfigCheck:
    """Check that the model repository contains config.json."""

    check_id: str = "text/files.required"
    title: str = "Required config file"

    def run(self, ctx: CheckContext) -> CheckResult:
        """Return whether config.json is present."""
        try:
            has_config = ctx.target.exists("config.json")
        except TargetError as exc:
            raise_for_hf_target_error(exc)
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message=f"Could not inspect config.json: {exc}",
                remediation="Ensure the model target can be read and includes config.json.",
            )

        if not has_config:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message="Missing required config.json.",
                remediation="Add config.json to the model repository.",
            )

        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message="config.json is present.",
        )
