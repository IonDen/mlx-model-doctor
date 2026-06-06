"""Checks for MLX quantization metadata."""

from collections.abc import Mapping
from dataclasses import dataclass

from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.report import CheckResult
from mlx_model_doctor.safetensors_header import SafetensorsHeader


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


_SCALES_SUFFIX = ".scales"


@dataclass(frozen=True, slots=True)
class MlxQuantShapeCheck:
    """Validate MLX quantized tensor shapes against config bits/group_size."""

    check_id: str = "text/quantization.shape"
    title: str = "Quantization shape"

    def run(self, ctx: CheckContext) -> CheckResult:
        """Return whether quantized layer shapes are internally consistent."""
        config = ctx.config_json()
        quant = config.get("quantization") if config is not None else None
        header = ctx.safetensors_header()
        if not isinstance(quant, Mapping) or header is None:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message="No MLX quantization metadata or safetensors header to check.",
            )
        bits = _positive_int(quant.get("bits"))
        group_size = _positive_int(quant.get("group_size"))
        if bits is None or group_size is None:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message="Quantization bits/group_size are incomplete; shape check skipped.",
            )
        if bits not in _AFFINE_BITS:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message=f"Quantization bits={bits} is outside the known set (valid as of MLX 0.31.x).",
                remediation="Confirm the MLX version supports this bit width.",
                details={"unknown_bits": bits},
            )
        scales = [name for name in self._all_names(header) if name.endswith(_SCALES_SUFFIX)]
        if not scales:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message="config declares quantization but no quantized tensors are stored.",
                remediation="Re-convert the model so the declared quantization is applied to its weights.",
                details={"no_quantized_tensors": True},
            )
        inconsistent: list[str] = []
        for scales_name in scales:
            prefix = scales_name[: -len(_SCALES_SUFFIX)]
            weight = header.tensor(prefix + ".weight")
            scales_entry = header.tensor(scales_name)
            if weight is None or scales_entry is None or weight.dtype != "U32":
                continue
            packed_last = weight.shape[-1]
            scales_last = scales_entry.shape[-1]
            in_s = scales_last * group_size
            if (packed_last * 32) % bits != 0 or packed_last * 32 // bits != in_s:
                inconsistent.append(prefix)
        if inconsistent:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message="Quantized tensor shapes are inconsistent with their scales (won't load).",
                remediation="Re-quantize/re-convert the model; the packed weight and scales disagree.",
                details={"inconsistent_layers": tuple(sorted(inconsistent))},
            )
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message="Quantized tensor shapes are consistent with config bits/group_size.",
        )

    def _all_names(self, header: SafetensorsHeader) -> list[str]:
        return [name for file_header in header.files for name in file_header.tensors]


def _positive_int(value: object) -> int | None:
    return value if type(value) is int and value > 0 else None
