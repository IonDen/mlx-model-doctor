"""Checks for MLX quantization metadata."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

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
class _ModeVerdict:
    """Result of classifying one (mode, group_size, bits) triple against the MLX table."""

    kind: str  # ok | non_string_mode | unknown_mode | off_table_affine | fixed_mismatch
    mode: object = None
    group_size: object = None
    bits: object = None
    bad: tuple[tuple[str, object], ...] = ()


def _off_table(value: object, allowed: "frozenset[int]") -> bool:
    """Membership that treats unhashable/wrong-type values as off-table instead of raising."""
    try:
        return value not in allowed
    except TypeError:
        return True


def _classify_quant(mode: object, group_size: object, bits: object) -> _ModeVerdict:
    """Classify a quantization triple. ``None`` group_size/bits mean 'field absent -> do not flag'."""
    if not isinstance(mode, str):
        return _ModeVerdict("non_string_mode", mode)
    if mode not in _VALID_MODES:
        return _ModeVerdict("unknown_mode", mode)
    if mode == "affine":
        bad: list[tuple[str, object]] = []
        if group_size is not None and _off_table(group_size, _AFFINE_GROUP_SIZES):
            bad.append(("group_size", group_size))
        if bits is not None and _off_table(bits, _AFFINE_BITS):
            bad.append(("bits", bits))
        if bad:
            return _ModeVerdict("off_table_affine", mode, bad=tuple(bad))
    else:
        expected_group_size, expected_bits = _FIXED_MODES[mode]
        if (group_size is not None and group_size != expected_group_size) or (
            bits is not None and bits != expected_bits
        ):
            return _ModeVerdict("fixed_mismatch", mode, group_size=group_size, bits=bits)
    return _ModeVerdict("ok", mode, group_size=group_size, bits=bits)


_PER_LAYER_FIELDS = frozenset({"mode", "bits", "group_size"})

_FAIL_KINDS = frozenset({"unknown_mode"})
_WARN_KINDS = frozenset({"non_string_mode", "off_table_affine", "fixed_mismatch"})


def _is_per_layer_override(value: object) -> bool:
    """A per-layer override is a mapping whose keys are a subset of the to_quantized fields.

    The subset filter skips a stray nested non-override mapping (e.g. a metadata block) that
    would otherwise be misread as a layer override — the config-only analogue of the shape
    check's tensor-prefix scoping.
    """
    return isinstance(value, Mapping) and set(value) <= _PER_LAYER_FIELDS


def _effective_mode_params(override: Mapping[str, object]) -> tuple[object, object, object]:
    """Resolve a per-layer override's effective (mode, group_size, bits).

    MLX passes the override verbatim to ``to_quantized``, so an absent ``mode`` defaults to
    ``affine`` (never the scalar mode) and absent ``bits``/``group_size`` are left as ``None``
    (``_classify_quant`` treats ``None`` as 'not present, do not flag'). Values are raw — no
    coercion — so a wrong-typed value is classified, not silently dropped.
    """
    return (override.get("mode", "affine"), override.get("group_size"), override.get("bits"))


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
        scalar = _classify_quant(
            quant.get("mode", "affine"), quant.get("group_size"), quant.get("bits")
        )
        overrides: list[tuple[str, Mapping[str, object]]] = []
        for name, value in quant.items():
            if isinstance(value, Mapping) and _is_per_layer_override(value):
                overrides.append((name, value))
        if not overrides:
            return self._render_scalar(scalar)
        return self._render_aggregate(scalar, overrides)

    def _render_scalar(self, verdict: _ModeVerdict) -> CheckResult:
        """Render today's exact scalar messages/details from a classified verdict."""
        if verdict.kind == "non_string_mode":
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message="Quantization mode must be a string.",
                remediation="Set mode to one of: affine, mxfp4, mxfp8, nvfp4.",
                details={"mode": verdict.mode},
            )
        if verdict.kind == "unknown_mode":
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message=f"Unknown MLX quantization mode {verdict.mode!r}; MLX rejects it at convert/load.",
                remediation="Use one of: affine, mxfp4, mxfp8, nvfp4.",
                details={"mode": verdict.mode},
            )
        if verdict.kind == "off_table_affine":
            bad = dict(verdict.bad)
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message=f"affine quantization has off-table values {bad} (valid as of MLX 0.31.x).",
                remediation="Use affine group_size in {32,64,128} and bits in {2,3,4,5,6,8}.",
                details={"mode": verdict.mode, **bad},
            )
        if verdict.kind == "fixed_mismatch":
            mode = cast("str", verdict.mode)
            expected_group_size, expected_bits = _FIXED_MODES[mode]
            group_size = (
                verdict.group_size if verdict.group_size is not None else expected_group_size
            )
            bits = verdict.bits if verdict.bits is not None else expected_bits
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
        if verdict.kind != "ok":  # pragma: no cover - exhaustiveness guard
            raise AssertionError(f"unhandled verdict kind: {verdict.kind!r}")
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message=f"MLX quantization mode {verdict.mode!r} has valid group_size/bits.",
            details={
                "mode": verdict.mode,
                "group_size": verdict.group_size,
                "bits": verdict.bits,
            },
        )

    def _render_aggregate(
        self, scalar: _ModeVerdict, overrides: list[tuple[str, "Mapping[str, object]"]]
    ) -> CheckResult:
        """Validate scalar default + each per-layer override; report worst severity."""
        fail_layers: list[str] = []
        warn_layers: list[str] = []
        for name, override in overrides:
            verdict = _classify_quant(*_effective_mode_params(override))
            if verdict.kind in _FAIL_KINDS:
                fail_layers.append(name)
            elif verdict.kind in _WARN_KINDS:
                warn_layers.append(name)
            elif verdict.kind != "ok":  # pragma: no cover - exhaustiveness guard
                raise AssertionError(f"unhandled verdict kind: {verdict.kind!r}")

        details: dict[str, object] = {}
        if fail_layers:
            details["invalid_mode_layers"] = tuple(sorted(fail_layers))
        if warn_layers:
            details["off_table_layers"] = tuple(sorted(warn_layers))
        if scalar.kind in _FAIL_KINDS or scalar.kind in _WARN_KINDS:
            details["scalar_default_invalid"] = True

        if fail_layers or scalar.kind in _FAIL_KINDS:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message="Quantization mode is invalid for one or more layers; MLX rejects unknown modes at load.",
                remediation="Use one of: affine, mxfp4, mxfp8, nvfp4 for every layer and the model default.",
                details=details,
            )
        if warn_layers or scalar.kind in _WARN_KINDS:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message="Quantization mode has off-table or non-canonical values for one or more layers (valid as of MLX 0.31.x).",
                remediation="Use affine group_size in {32,64,128} and bits in {2,3,4,5,6,8}, or a fixed mode's canonical pair.",
                details=details,
            )
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message=f"MLX quantization mode is valid across the default and {len(overrides)} per-layer override(s).",
            details={"per_layer_overrides": len(overrides)},
        )


_SCALES_SUFFIX = ".scales"


def _resolve_quant_field(
    override: Mapping[str, object], key: str, default: int | None
) -> int | None:
    """Resolve one per-layer quantization field.

    An *absent* field inherits the scalar default. A *present* field must be a
    positive int; an explicit invalid value (zero, negative, non-int) returns None
    so the layer is reported unverified rather than silently validated with the
    model default.
    """
    if key not in override:
        return default
    return _positive_int(override[key])


def _effective_quant(
    quant: Mapping[str, object], prefix: str, default_bits: int | None, default_gs: int | None
) -> tuple[int | None, int | None]:
    """Resolve a layer's effective (bits, group_size).

    Uses the per-layer override at ``quant[prefix]`` when it is a mapping, falling
    back to the scalar defaults for absent fields only. Mode does not affect the
    shape arithmetic, so it is not consulted here.
    """
    override = quant.get(prefix)
    if isinstance(override, Mapping):
        return (
            _resolve_quant_field(override, "bits", default_bits),
            _resolve_quant_field(override, "group_size", default_gs),
        )
    return default_bits, default_gs


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
        default_bits = _positive_int(quant.get("bits"))
        default_gs = _positive_int(quant.get("group_size"))

        scales = [name for name in header.tensor_names() if name.endswith(_SCALES_SUFFIX)]
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
        # "Is this a mixed-precision config?" — scoped to the layers we actually scan, so a stray
        # non-override mapping elsewhere in the block cannot suppress the incomplete-metadata skip.
        has_per_layer = any(
            isinstance(quant.get(name[: -len(_SCALES_SUFFIX)]), Mapping) for name in scales
        )

        inconsistent: list[str] = []
        unverified: list[str] = []
        consistent_count = 0
        for scales_name in scales:
            prefix = scales_name[: -len(_SCALES_SUFFIX)]
            weight = header.tensor(prefix + ".weight")
            scales_entry = header.tensor(scales_name)
            if weight is None or scales_entry is None or weight.dtype != "U32":
                continue
            bits, group_size = _effective_quant(quant, prefix, default_bits, default_gs)
            if bits is None or group_size is None or bits not in _AFFINE_BITS:
                unverified.append(prefix)
                continue
            packed_last = weight.shape[-1]
            scales_last = scales_entry.shape[-1]
            in_s = scales_last * group_size
            if (packed_last * 32) % bits != 0 or packed_last * 32 // bits != in_s:
                inconsistent.append(prefix)
            else:
                consistent_count += 1

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
        if (
            consistent_count == 0
            and not has_per_layer
            and (default_bits is None or default_gs is None)
        ):
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message="Quantization bits/group_size are incomplete; shape check skipped.",
            )
        if unverified:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message=(
                    "Shape unverified for layer(s) with an unrecognized or undeterminable "
                    "bit width (valid as of MLX 0.31.x)."
                ),
                remediation="Confirm the MLX version supports the quantization bit width(s) for these layers.",
                details={"unverified_layers": tuple(sorted(unverified))},
            )
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message="Quantized tensor shapes are consistent with config bits/group_size.",
        )


def _positive_int(value: object) -> int | None:
    return value if type(value) is int and value > 0 else None
