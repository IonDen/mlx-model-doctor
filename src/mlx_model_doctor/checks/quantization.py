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


_AFFINE_GROUP_SIZES = frozenset({32, 64, 128})
_AFFINE_BITS = frozenset({2, 3, 4, 5, 6, 8})
_FIXED_MODES: dict[str, tuple[int, int]] = {"mxfp4": (32, 4), "mxfp8": (32, 8), "nvfp4": (16, 4)}
_VALID_MODES = frozenset({"affine", *_FIXED_MODES})


@dataclass(frozen=True, slots=True)
class MlxQuantizationModeCheck:
    """Validate MLX quantization mode / group_size / bits against the MLX table."""

    check_id: str = "text/quantization.mode"
    title: str = "Quantization mode"

    def run(self, ctx: CheckContext) -> CheckResult:
        """Return whether the quantization (mode, group_size, bits) is MLX-valid."""
        config = ctx.config_json()
        quant = config.get("quantization") if config is not None else None
        if not isinstance(quant, Mapping):
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message="No MLX quantization object to validate.",
            )
        mode = quant.get("mode", "affine")
        if not isinstance(mode, str):
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message="Quantization mode must be a string.",
                remediation="Set mode to one of: affine, mxfp4, mxfp8, nvfp4.",
                details={"mode": mode},
            )
        if mode not in _VALID_MODES:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message=f"Unknown MLX quantization mode {mode!r}; MLX rejects it at convert/load.",
                remediation="Use one of: affine, mxfp4, mxfp8, nvfp4.",
                details={"mode": mode},
            )
        if mode == "affine":
            bad: list[tuple[str, object]] = []
            if "group_size" in quant and quant["group_size"] not in _AFFINE_GROUP_SIZES:
                bad.append(("group_size", quant["group_size"]))
            if "bits" in quant and quant["bits"] not in _AFFINE_BITS:
                bad.append(("bits", quant["bits"]))
            if bad:
                return CheckResult(
                    check_id=self.check_id,
                    title=self.title,
                    status="warn",
                    severity="medium",
                    message=f"affine quantization has off-table values {dict(bad)} (valid as of MLX 0.31.x).",
                    remediation="Use affine group_size in {32,64,128} and bits in {2,3,4,5,6,8}.",
                    details={"mode": mode, **dict(bad)},
                )
        else:
            expected_group_size, expected_bits = _FIXED_MODES[mode]
            group_size = quant.get("group_size", expected_group_size)
            bits = quant.get("bits", expected_bits)
            if group_size != expected_group_size or bits != expected_bits:
                return CheckResult(
                    check_id=self.check_id,
                    title=self.title,
                    status="warn",
                    severity="medium",
                    message=(
                        f"{mode} is a fixed format expecting group_size={expected_group_size}, "
                        f"bits={expected_bits}; got group_size={group_size}, bits={bits}."
                    ),
                    remediation=f"Use group_size={expected_group_size}, bits={expected_bits} for {mode}.",
                    details={"mode": mode, "group_size": group_size, "bits": bits},
                )
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message=f"MLX quantization mode {mode!r} has valid group_size/bits.",
            details={
                "mode": mode,
                "group_size": quant.get("group_size"),
                "bits": quant.get("bits"),
            },
        )
