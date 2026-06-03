"""Checks for MLX quantization metadata."""

from collections.abc import Mapping
from dataclasses import dataclass

from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.report import CheckResult


@dataclass(frozen=True, slots=True)
class QuantizationMetadataCheck:
    """Check that config.json exposes MLX quantization metadata."""

    check_id: str = "text/quantization.metadata"
    title: str = "Quantization metadata"

    def run(self, ctx: CheckContext) -> CheckResult:
        """Return whether MLX-compatible quantization metadata is available."""
        config = ctx.config_json()
        if config is None:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message="config.json is unavailable, so quantization metadata cannot be checked.",
            )

        mlx_quantization = config.get("quantization")
        if isinstance(mlx_quantization, Mapping):
            bits = _positive_number(mlx_quantization.get("bits"))
            if bits is None:
                return CheckResult(
                    check_id=self.check_id,
                    title=self.title,
                    status="warn",
                    severity="medium",
                    message="config.json contains incomplete MLX quantization metadata.",
                    remediation="Add a positive numeric bits value to the MLX quantization object.",
                    details={"quantization": dict(mlx_quantization)},
                )
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="pass",
                severity="info",
                message="config.json contains MLX quantization metadata.",
                details={
                    "bits": mlx_quantization.get("bits"),
                    "group_size": mlx_quantization.get("group_size"),
                },
            )
        if mlx_quantization is not None:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message="config.json quantization metadata must be an object for MLX.",
                remediation="Use an MLX quantization object such as {'bits': 4, 'group_size': 64}.",
                details={"quantization": mlx_quantization},
            )

        if "quantization_config" in config:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message="config.json contains non-MLX quantization_config metadata.",
                remediation="Convert quantization_config to the MLX top-level quantization object.",
                details={"quantization_config": config["quantization_config"]},
            )

        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="warn",
            severity="low",
            message="config.json does not contain quantization metadata.",
            remediation="Add MLX top-level quantization metadata when the model is quantized.",
        )


def _positive_number(value: object) -> float | None:
    if (type(value) is int or type(value) is float) and value > 0:
        return float(value)
    return None
