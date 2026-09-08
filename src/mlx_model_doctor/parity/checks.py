"""Cross-target static checks for adapter parity (F3/F4/F5/F6).

Each :class:`ParityCheck` reads a :class:`~mlx_model_doctor.parity.context.ParityContext`
(the base/adapter/fused targets) and answers a question the loader-accurate
resolver (:mod:`mlx_model_doctor.parity.resolver`) and tokenizer fingerprint
(:mod:`mlx_model_doctor.parity.context`) make answerable *before* any model
is loaded: is the adapter config well-formed, does every configured LoRA
target key actually name a base-model tensor, does the fused checkpoint
expose what a real fuse should produce, and do the base and fused
tokenizers still agree. :func:`run_parity_checks` runs a list of these,
mirroring :func:`~mlx_model_doctor.runners.core.run_checks`'s crash
isolation with one addition: it also reports whether any check crashed, so
a crash can be told apart from an ordinary, confidently-reached ``fail``
(F4).
"""

import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from mlx_model_doctor.errors import ModelDoctorError, TargetError, raise_for_hf_target_error
from mlx_model_doctor.parity.context import ParityContext, tokenizers_match
from mlx_model_doctor.parity.resolver import ResolvedTargets, resolve_targets
from mlx_model_doctor.report import CheckResult
from mlx_model_doctor.safetensors_header import SafetensorsHeader

# Suffixes stripped from a safetensors tensor name to recover its owning
# module's path, matching MLX's nn.QuantizedLinear/nn.Linear parameter
# naming (mirrors mlx_model_doctor.parity.deltamap's own suffix list).
_TENSOR_PAYLOAD_SUFFIXES = (".weight", ".scales", ".biases")

# Standalone LoRA/DoRA factor tensor suffixes a genuinely fused checkpoint
# must no longer carry for a target it has baked in (mlx_lm.tuner writes
# these under `{target}.lora_a` / `{target}.lora_b` / DoRA's `{target}.m`).
# Mirrors mlx-lm's current naming (pinned mlx-lm==0.31.3, same version the
# resolver and worker modules verify against); a future mlx-lm rename of
# these attributes would need a matching update here.
_LORA_FACTOR_SUFFIXES = (".lora_a", ".lora_b", ".m")


class HasCheckIdTitle(Protocol):
    """Minimal check_id/title shape shared by ModelCheck and ParityCheck.

    ``_safe_parity_check_id`` only needs these two attributes. Typing it
    against this narrower Protocol -- rather than reusing
    ``mlx_model_doctor.checks.base.ModelCheck``, whose ``run`` takes a
    single-target ``CheckContext`` -- is what lets it accept a
    :class:`ParityCheck` too: passing a ``ParityCheck`` to a parameter typed
    ``ModelCheck`` fails mypy --strict, since ``ParityCheck.run``'s
    signature is not a subtype of ``ModelCheck.run``'s.
    """

    check_id: str
    title: str


class ParityCheck(Protocol):
    """Protocol implemented by cross-target adapter-parity checks."""

    check_id: str
    title: str

    def run(self, pctx: ParityContext) -> CheckResult:
        """Run the check against a parity context (base/adapter/fused targets)."""


def _safe_parity_check_id(check: HasCheckIdTitle) -> str:
    """Return a plugin-namespaced check_id, sanitizing a malformed one.

    Mirrors ``mlx_model_doctor.runners.core._safe_check_id`` exactly, typed
    against :class:`HasCheckIdTitle` instead of ``ModelCheck`` so it accepts
    a :class:`ParityCheck` too.
    """
    raw = str(getattr(check, "check_id", "") or "")
    namespace, separator, name = raw.partition("/")
    if namespace and separator and name:
        return raw
    return f"unknown/{name or namespace or 'check'}"


def run_parity_checks(
    pctx: ParityContext, checks: Sequence[ParityCheck]
) -> tuple[list[CheckResult], bool]:
    """Run parity checks, isolating unexpected crashes as failed results.

    Mirrors ``mlx_model_doctor.runners.core.run_checks``'s crash isolation --
    a ``ModelDoctorError`` propagates unchanged; any other exception becomes
    a ``fail`` ``CheckResult`` via :func:`_safe_parity_check_id` -- plus one
    addition: the returned ``bool`` is ``True`` when at least one check
    crashed. A crash is a distinguishable outcome from an ordinary ``fail``:
    it means the check could not even determine an answer, so downstream
    exit-code logic can treat the run as cannot-determine rather than
    folding it into an authoritative fail (F4).
    """
    results: list[CheckResult] = []
    crashed = False
    for check in checks:
        try:
            results.append(check.run(pctx))
        except ModelDoctorError:
            raise
        except Exception as exc:
            crashed = True
            details: dict[str, object] = {}
            if pctx.options.verbosity == "verbose":
                details["traceback"] = traceback.format_exc()
            results.append(
                CheckResult(
                    check_id=_safe_parity_check_id(check),
                    title=check.title,
                    status="fail",
                    severity="high",
                    message=f"check crashed: {exc}",
                    details=details,
                )
            )
    return results, crashed


def _strip_tensor_payload_suffix(tensor_name: str) -> str:
    """Strip a trailing weight/quantization-payload suffix, returning the module path."""
    for suffix in _TENSOR_PAYLOAD_SUFFIXES:
        if tensor_name.endswith(suffix):
            return tensor_name[: -len(suffix)]
    return tensor_name


def _module_names_from_header(header: SafetensorsHeader) -> tuple[str, ...]:
    """Return the module paths a safetensors header's tensor names imply.

    The result matches what ``model.named_modules()`` would yield -- the
    shape ``resolve_targets`` expects for its ``base_module_names`` argument.
    """
    return tuple(sorted({_strip_tensor_payload_suffix(name) for name in header.tensor_names()}))


def _resolve_against_base(pctx: ParityContext) -> ResolvedTargets | str:
    """Resolve LoRA targets against the base target's safetensors header.

    Returns the resolver's :class:`ResolvedTargets`, or a human-readable
    reason string when resolution could not even be attempted (missing or
    malformed ``adapter_config.json``, or no readable base safetensors
    header). This is distinct from the resolver's own
    ``verified=False``/full-fine-tune outcomes, which are themselves valid
    resolutions a caller inspects via ``ResolvedTargets.reason``.
    """
    if pctx.adapter_config is None:
        return "adapter_config.json is missing or malformed; cannot resolve LoRA targets."
    header = pctx.base.safetensors_header()
    if header is None:
        error = pctx.base.safetensors_header_error()
        return error or "the base target has no safetensors header; cannot resolve LoRA targets."
    base_module_names = _module_names_from_header(header)
    arch = pctx.base_model_type() or "unknown"
    return resolve_targets(pctx.adapter_config, base_module_names, arch=arch)


def resolve_targets_against_base(pctx: ParityContext) -> ResolvedTargets | str:
    """Public wrapper over :func:`_resolve_against_base` for reuse by the API layer.

    Returns the resolver's :class:`ResolvedTargets` (whose ``covered`` set the
    delta map consumes as its expected-change targets), or a human-readable
    reason string when resolution could not be attempted at all.
    """
    return _resolve_against_base(pctx)


def _adapter_config_shape_problems(config: Mapping[str, object]) -> list[str]:
    """Return human-readable shape problems in a parsed adapter_config, if any."""
    problems: list[str] = []
    if "num_layers" in config:
        value = config["num_layers"]
        if not isinstance(value, int) or isinstance(value, bool):
            problems.append(f"num_layers must be an integer, got {value!r}")
    if "fine_tune_type" in config:
        value = config["fine_tune_type"]
        if not isinstance(value, str):
            problems.append(f"fine_tune_type must be a string, got {value!r}")
    lora_parameters = config.get("lora_parameters")
    if lora_parameters is not None and not isinstance(lora_parameters, Mapping):
        problems.append(f"lora_parameters must be an object, got {lora_parameters!r}")
    elif isinstance(lora_parameters, Mapping) and "keys" in lora_parameters:
        raw_keys = lora_parameters["keys"]
        if raw_keys is not None and (
            not isinstance(raw_keys, list) or not all(isinstance(k, str) for k in raw_keys)
        ):
            problems.append(f"lora_parameters.keys must be a list of strings, got {raw_keys!r}")
    return problems


@dataclass(frozen=True, slots=True)
class AdapterConfigCheck:
    """Check that the adapter target's adapter_config.json is present and well-formed."""

    check_id: str = "parity/adapter.config_well_formed"
    title: str = "Adapter config well-formed"

    def run(self, pctx: ParityContext) -> CheckResult:
        """Validate presence and shape of adapter_config.json's parity-relevant fields."""
        try:
            exists = pctx.targets.adapter.exists("adapter_config.json")
        except TargetError as exc:
            raise_for_hf_target_error(exc)
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message=f"Could not inspect adapter_config.json: {exc}",
            )
        if not exists:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message="Missing required adapter_config.json in the adapter target.",
                remediation="Ensure the adapter repository includes adapter_config.json.",
            )

        config = pctx.adapter_config
        if config is None:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message=(
                    "adapter_config.json is present but could not be parsed as a JSON object."
                ),
            )

        problems = _adapter_config_shape_problems(config)
        if problems:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message=f"adapter_config.json is malformed: {'; '.join(problems)}.",
                details={"problems": tuple(problems)},
                remediation="Fix the malformed field(s) in adapter_config.json.",
            )

        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message="adapter_config.json is present and well-formed.",
        )


@dataclass(frozen=True, slots=True)
class TargetCoverageCheck:
    """Check that every resolved LoRA target key maps to a real base-model tensor."""

    check_id: str = "parity/target.coverage"
    title: str = "LoRA target coverage"

    def run(self, pctx: ParityContext) -> CheckResult:
        """Resolve targets against the base model and flag any key with no matching module."""
        outcome = _resolve_against_base(pctx)
        if isinstance(outcome, str):
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message=outcome,
            )
        if outcome.reason is not None:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message=outcome.reason,
            )
        if outcome.uncovered:
            uncovered = sorted(outcome.uncovered)
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message=(
                    "Adapter config names LoRA target key(s) with no matching base-model "
                    f"tensor: {', '.join(uncovered)}."
                ),
                details={"uncovered_keys": tuple(uncovered)},
                remediation=(
                    "Fix the typo'd key(s) in adapter_config.json's lora_parameters.keys."
                ),
            )
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message=f"All {len(outcome.covered)} resolved LoRA target(s) map to a base tensor.",
        )


@dataclass(frozen=True, slots=True)
class FusedTargetConsistencyCheck:
    """Check that the fused target has every targeted weight, with no leftover LoRA factors."""

    check_id: str = "parity/fused.target_consistency"
    title: str = "Fused target consistency"

    def run(self, pctx: ParityContext) -> CheckResult:
        """Confirm the fused header covers every resolved target with no leftover factors."""
        outcome = _resolve_against_base(pctx)
        if isinstance(outcome, str):
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message=outcome,
            )
        if outcome.reason is not None:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message=outcome.reason,
            )

        fused_header = pctx.fused.safetensors_header()
        if fused_header is None:
            error = pctx.fused.safetensors_header_error()
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message=(
                    error or "the fused target has no safetensors header; cannot check consistency."
                ),
            )

        fused_names = frozenset(fused_header.tensor_names())
        omitted = sorted(
            target for target in outcome.covered if f"{target}.weight" not in fused_names
        )
        leftover = sorted(
            f"{target}{suffix}"
            for target in outcome.covered
            for suffix in _LORA_FACTOR_SUFFIXES
            if f"{target}{suffix}" in fused_names
        )
        if omitted or leftover:
            problems = []
            if omitted:
                problems.append(f"omitted target weight(s): {', '.join(omitted)}")
            if leftover:
                problems.append(f"leftover unfused LoRA factor(s): {', '.join(leftover)}")
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message=f"Fused target is inconsistent: {'; '.join(problems)}.",
                details={
                    "omitted_targets": tuple(omitted),
                    "leftover_lora_factors": tuple(leftover),
                },
                remediation=(
                    "Re-run the fuse step; the fused checkpoint must bake in every targeted "
                    "weight and drop the standalone LoRA factor tensors."
                ),
            )
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message=(
                f"Fused target exposes all {len(outcome.covered)} targeted weight(s) with no "
                "leftover LoRA factors."
            ),
        )


@dataclass(frozen=True, slots=True)
class TokenizerIdentityCheck:
    """Check that the base and fused targets' tokenizers are identical (F6)."""

    check_id: str = "parity/tokenizer.identity"
    title: str = "Tokenizer identity"

    def run(self, pctx: ParityContext) -> CheckResult:
        """Compare base vs fused tokenizer fingerprints; a mismatch voids the oracle."""
        match = tokenizers_match(
            pctx.base_tokenizer_fingerprint(), pctx.fused_tokenizer_fingerprint()
        )
        if not match.matched:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message=(
                    "The base and fused targets' tokenizers do not match "
                    f"({match.reason}); the parity oracle's outputs would not be comparable."
                ),
                details={"match_reason": match.reason, "void_oracle": True},
                remediation="Ensure the fuse step carries the base tokenizer through unchanged.",
            )
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message="The base and fused targets' tokenizers match.",
        )
