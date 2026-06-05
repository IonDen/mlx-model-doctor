"""Checks over the safetensors tensor map."""

from dataclasses import dataclass

from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.report import CheckResult


@dataclass(frozen=True, slots=True)
class WeightParamCountCheck:
    """Check the weight map against the parsed headers (internal consistency)."""

    check_id: str = "text/weights.param_count"
    title: str = "Weight parameter count"

    def run(self, ctx: CheckContext) -> CheckResult:
        """Return whether every mapped tensor exists and params are non-zero."""
        header = ctx.safetensors_header()
        if header is None:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message="No safetensors header to check parameter counts.",
            )
        missing = tuple(sorted(name for name in header.weight_map if header.tensor(name) is None))
        if missing:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message="The weight map references tensors absent from every shard header.",
                remediation="Fix the safetensors index weight_map or add the missing shard tensors.",
                details={"missing_tensors": missing},
            )
        empty_files = tuple(sorted(fh.filename for fh in header.files if not fh.tensors))
        total = header.total_parameter_count()
        if empty_files or total == 0:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message="The safetensors header reports no tensor parameters.",
                remediation="Confirm the safetensors shards actually contain weights.",
                details={"empty_or_zero_param_files": empty_files, "total_parameter_count": total},
            )
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message="The weight map resolves to present tensors with non-zero parameters.",
            details={"total_parameter_count": total},
        )
