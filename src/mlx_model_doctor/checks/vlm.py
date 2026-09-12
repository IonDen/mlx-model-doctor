"""VLM image-processor presence check (gated; non-VLM repos skip)."""

from collections.abc import Mapping
from dataclasses import dataclass

from mlx_model_doctor.checks.memory import _estimate_details, _file_size_estimate
from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.report import CheckResult

_IMAGE_PREPROCESSOR_KEYS: tuple[str, ...] = (
    "image_processor_type",
    "image_mean",
    "image_std",
    "crop_size",
)
_IMAGE_AUTO_MAP_KEYS: tuple[str, ...] = ("AutoImageProcessor", "AutoFeatureExtractor")
_VLM_CONFIG_KEYS: tuple[str, ...] = (
    "vision_config",
    "mm_vision_tower",
    "vision_tower",
    "mm_projector_type",
    "image_token_id",
    "image_token_index",
    "image_token",
)
# Real image-placeholder tokens across model families. `[IMG]` is Pixtral / Mistral3's placeholder.
_ACTUAL_IMAGE_TOKENS = frozenset(
    ("<image>", "<|image|>", "<|image_pad|>", "<IMG_CONTEXT>", "[IMG]")
)
_WRAPPER_IMAGE_TOKENS = frozenset(("<|vision_start|>", "<|vision_end|>"))
_MAX_SPECIAL_TOKEN_SCAN_DEPTH = 256
_MAX_SPECIAL_TOKEN_SCAN_NODES = 4096


# Image-processor resolution heuristic verified against mlx-vlm 0.6.x and transformers v4.x.
@dataclass(frozen=True, slots=True)
class VlmImageProcessorCheck:
    """Report whether a vision-language repo can resolve an image processor at load."""

    check_id: str = "text/vlm.image_processor"
    title: str = "VLM image processor"

    def run(self, ctx: CheckContext) -> CheckResult:
        """Validate a vision-language repo's image_processor_type; skip non-VLM repos."""
        config = ctx.config_json()
        preproc = ctx.preprocessor_config_json()

        if not is_vlm_metadata(config, preproc):
            return self._result("skip", "info", "Not a vision-language repo; skipped.")

        found, value, source = _resolve_image_processor_type(preproc, config)
        if found:
            if isinstance(value, str) and value.strip():
                return self._result(
                    "pass",
                    "info",
                    f"image_processor_type '{value}' declared in {source}.",
                    details={"image_processor_type": value, "source": source},
                )
            return self._result(
                "fail",
                "high",
                "image_processor_type is present but empty or not a string.",
                remediation="Set image_processor_type to a valid image-processor class name.",
            )

        signal = _resolution_signal(preproc, config)
        if signal is not None:
            return self._result(
                "pass",
                "info",
                f"No image_processor_type, but the repo resolves an image processor via {signal}.",
                details={"resolution": signal},
            )

        return self._result(
            "warn",
            "medium",
            "No image_processor_type or other image-processor resolution signal (custom auto_map, "
            "feature_extractor_type, or processor_class). As of transformers 5.x the standard "
            "AutoImageProcessor path may be unable to resolve it, depending on the installed "
            "version's model_type mapping; verify the model loads before relying on it.",
            remediation="Add image_processor_type to preprocessor_config.json "
            '(for example "Qwen2VLImageProcessor"), or confirm the model loads via its processor.',
        )

    def _result(
        self,
        status: str,
        severity: str,
        message: str,
        *,
        remediation: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> CheckResult:
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status=status,  # type: ignore[arg-type]
            severity=severity,  # type: ignore[arg-type]
            message=message,
            remediation=remediation,
            details=details or {},
        )


@dataclass(frozen=True, slots=True)
class VlmMemoryEstimateCheck:
    """Estimate VLM memory using measured model file sizes only."""

    check_id: str = "vlm/memory.estimate"
    title: str = "VLM memory estimate"

    def run(self, ctx: CheckContext) -> CheckResult:
        """Return a measured weight-file lower bound for VLM memory use."""
        estimate = _file_size_estimate(ctx)
        if estimate is None:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message="VLM memory estimate skipped because weight file sizes are unavailable.",
                details={
                    "estimate_source": "unknown",
                    "context_length": ctx.options.context_length,
                },
            )
        if estimate.estimate_source == "unknown":
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message="VLM memory estimate skipped because weight file sizes are unavailable.",
                details=_estimate_details(
                    estimate, ctx.options.context_length, ctx.options.max_memory_bytes
                ),
            )

        details = _estimate_details(
            estimate, ctx.options.context_length, ctx.options.max_memory_bytes
        )
        max_memory_bytes = ctx.options.max_memory_bytes
        if max_memory_bytes is not None and estimate.lower_bound_bytes > max_memory_bytes:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message="Estimated VLM file-size lower bound exceeds the configured budget.",
                remediation=(
                    "Use a smaller model, stronger quantization, or a higher memory budget "
                    "before loading."
                ),
                details=details,
            )
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message=(
                "Estimated VLM file-size lower bound is advisory and below the configured budget."
                if max_memory_bytes is not None
                else "Estimated VLM file-size lower bound is advisory; no memory budget was configured."
            ),
            details=details,
        )


def is_vlm_metadata(
    config: Mapping[str, object] | None,
    preproc: Mapping[str, object] | None,
) -> bool:
    """Return whether config/preprocessor metadata has VLM structural signals."""
    if isinstance(config, Mapping) and any(key in config for key in _VLM_CONFIG_KEYS):
        return True
    return isinstance(preproc, Mapping) and any(key in preproc for key in _IMAGE_PREPROCESSOR_KEYS)


def _has_custom_processor_signal(cfg: Mapping[str, object]) -> bool:
    processor_class = cfg.get("processor_class")
    if isinstance(processor_class, str) and processor_class.strip():
        return True
    auto_map = cfg.get("auto_map")
    return isinstance(auto_map, Mapping) and any(key in auto_map for key in _IMAGE_AUTO_MAP_KEYS)


def _resolve_image_processor_type(
    preproc: Mapping[str, object] | None,
    config: Mapping[str, object] | None,
) -> tuple[bool, object, str | None]:
    for cfg, fname in ((preproc, "preprocessor_config.json"), (config, "config.json")):
        if isinstance(cfg, Mapping) and "image_processor_type" in cfg:
            return True, cfg["image_processor_type"], fname
    return False, None, None


def _resolution_signal(
    preproc: Mapping[str, object] | None,
    config: Mapping[str, object] | None,
) -> str | None:
    """Return how the repo resolves an image processor without image_processor_type, or None.

    transformers' AutoImageProcessor loads the preprocessor config first, so a custom
    `auto_map` is only a reliable signal when it lives in preprocessor_config.json — a
    config-only `auto_map` is not enough on its own. A `processor_class` (the AutoProcessor
    path mlx-vlm uses) in either file also resolves the processor.
    """
    if isinstance(preproc, Mapping):
        auto_map = preproc.get("auto_map")
        if isinstance(auto_map, Mapping) and any(key in auto_map for key in _IMAGE_AUTO_MAP_KEYS):
            return "preprocessor_config.json auto_map"
        feature_extractor = preproc.get("feature_extractor_type")
        if isinstance(feature_extractor, str) and feature_extractor.strip():
            return "feature_extractor_type"
    for cfg in (preproc, config):
        if isinstance(cfg, Mapping):
            processor_class = cfg.get("processor_class")
            if isinstance(processor_class, str) and processor_class.strip():
                return "processor_class"
    return None


def _token_from_decoder_entry(entry: object) -> str | None:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, Mapping):
        value = entry.get("content")
        return value if isinstance(value, str) else None
    return None


def _added_token_by_id(tokenizer_config: Mapping[str, object], token_id: int) -> str | None:
    decoder = tokenizer_config.get("added_tokens_decoder")
    if not isinstance(decoder, Mapping):
        return None
    return _token_from_decoder_entry(decoder.get(str(token_id)))


def _iter_special_token_values(value: object) -> tuple[str, ...]:
    tokens: list[str] = []
    stack: list[tuple[object, int]] = [(value, 0)]
    scanned = 0
    while stack and scanned < _MAX_SPECIAL_TOKEN_SCAN_NODES:
        current, depth = stack.pop()
        scanned += 1
        if isinstance(current, str):
            tokens.append(current)
            continue
        if depth >= _MAX_SPECIAL_TOKEN_SCAN_DEPTH:
            continue
        if isinstance(current, Mapping):
            content = current.get("content")
            if isinstance(content, str):
                tokens.append(content)
            stack.extend((item, depth + 1) for item in current.values())
            continue
        if isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
    return tuple(tokens)


def _known_token_strings(ctx: CheckContext) -> set[str]:
    tokens: set[str] = set()
    for source in (ctx.tokenizer_config_json() or {}, ctx.special_tokens_map_json() or {}):
        for value in source.values():
            tokens.update(_iter_special_token_values(value))
        extra = source.get("extra_special_tokens")
        if isinstance(extra, Mapping):
            for value in extra.values():
                tokens.update(_iter_special_token_values(value))
    template = _template_text(ctx)
    if template:
        for token in (*_ACTUAL_IMAGE_TOKENS, *_WRAPPER_IMAGE_TOKENS):
            if token in template:
                tokens.add(token)
    return tokens


def _template_text(ctx: CheckContext) -> str | None:
    jinja = ctx.chat_template_text()
    if jinja is not None and jinja.strip():
        return jinja
    tokenizer_config = ctx.tokenizer_config_json() or {}
    template = tokenizer_config.get("chat_template")
    if isinstance(template, str) and template.strip():
        return template
    if isinstance(template, list):
        parts = [
            entry["template"]
            for entry in template
            if isinstance(entry, Mapping) and isinstance(entry.get("template"), str)
        ]
        return "\n".join(parts) if parts else None
    return None


def _token_visible_in_metadata(ctx: CheckContext, token: str, known_tokens: set[str]) -> bool:
    if token in known_tokens:
        return True
    template = _template_text(ctx)
    return template is not None and token in template


@dataclass(frozen=True, slots=True)
class VlmImageTokenWiringCheck:
    """Check that VLM image-token config agrees with tokenizer/template metadata."""

    check_id: str = "vlm/image_token.wiring"
    title: str = "VLM image token wiring"

    def run(self, ctx: CheckContext) -> CheckResult:
        """Validate config, tokenizer, and template image-token metadata coherence."""
        config = ctx.config_json()
        preproc = ctx.preprocessor_config_json()
        if not is_vlm_metadata(config, preproc):
            return self._result("skip", "info", "Not a vision-language repo; skipped.")
        if not isinstance(config, Mapping):
            return self._result(
                "skip",
                "info",
                "config.json is unavailable, so image-token wiring cannot be checked.",
            )

        ids, malformed = _config_image_token_ids(config)
        if malformed:
            return self._result(
                "fail",
                "high",
                f"{malformed} must be a non-bool integer.",
                remediation=(
                    f"Set {malformed} to the tokenizer ID of the actual image placeholder token."
                ),
            )
        if len(set(ids.values())) > 1:
            return self._result(
                "warn",
                "medium",
                "image_token_id and image_token_index conflict.",
                remediation="Use one consistent image token ID field.",
                details={"image_token_ids": ids},
            )

        image_token = config.get("image_token")
        if image_token is not None and (
            not isinstance(image_token, str) or not image_token.strip()
        ):
            return self._result(
                "fail",
                "high",
                "image_token must be a non-empty string.",
                remediation="Set image_token to the actual image placeholder token string.",
            )

        tokenizer_config = ctx.tokenizer_config_json() or {}
        known_tokens = _known_token_strings(ctx)
        actual = known_tokens & _ACTUAL_IMAGE_TOKENS
        wrappers = known_tokens & _WRAPPER_IMAGE_TOKENS

        if ids:
            token_id = next(iter(ids.values()))
            mapped = _added_token_by_id(tokenizer_config, token_id)
            configured_token = image_token.strip() if isinstance(image_token, str) else None
            if configured_token is not None and mapped is not None and mapped != configured_token:
                return self._result(
                    "warn",
                    "medium",
                    "Image token ID mapping conflicts with image_token.",
                    remediation="Use one consistent image token string and token ID mapping.",
                    details={
                        "image_token_id": token_id,
                        "mapped_token": mapped,
                        "image_token": configured_token,
                    },
                )
            if mapped in _ACTUAL_IMAGE_TOKENS or (
                mapped is not None
                and mapped == configured_token
                and _token_visible_in_metadata(ctx, mapped, known_tokens)
            ):
                return self._result(
                    "pass",
                    "info",
                    f"Image token ID {token_id} maps to {mapped}.",
                    details={"image_token_id": token_id, "image_token": mapped},
                )
            if mapped in _WRAPPER_IMAGE_TOKENS:
                return self._result(
                    "warn",
                    "medium",
                    f"Image token ID {token_id} maps to wrapper token {mapped}, "
                    "not the actual image token.",
                    remediation=(
                        "Point image_token_id/image_token_index at the actual image "
                        "placeholder token."
                    ),
                    details={"image_token_id": token_id, "mapped_token": mapped},
                )
            return self._result(
                "warn",
                "medium",
                "Config declares an image token ID, but tokenizer metadata does not map "
                "it to an actual image placeholder.",
                remediation=(
                    "Ensure tokenizer_config.json added_tokens_decoder maps the image "
                    "token ID to <image>, <|image|>, <|image_pad|>, <IMG_CONTEXT>, or [IMG]."
                ),
                details={"image_token_id": token_id, "mapped_token": mapped or ""},
            )

        if isinstance(image_token, str) and image_token.strip():
            token = image_token.strip()
            if _token_visible_in_metadata(ctx, token, known_tokens):
                return self._result(
                    "pass",
                    "info",
                    f"image_token {token} is present in VLM token metadata.",
                    details={"image_token": token},
                )
            return self._result(
                "warn",
                "medium",
                f"image_token {token} is not visible in tokenizer/template metadata.",
                remediation=(
                    "Register the image token in tokenizer metadata and use it in the "
                    "chat template."
                ),
                details={"image_token": token},
            )

        if actual and (
            _has_custom_processor_signal(config)
            or (isinstance(preproc, Mapping) and _has_custom_processor_signal(preproc))
        ):
            return self._result(
                "pass",
                "info",
                "Custom processor path exposes runtime image-token wiring.",
                details={
                    "resolution": "custom_processor",
                    "image_tokens": tuple(sorted(actual)),
                },
            )

        if actual:
            return self._result(
                "warn",
                "medium",
                "Tokenizer/template metadata contains an image placeholder but config "
                "has no image-token field.",
                remediation=(
                    "Add image_token_id/image_token_index when this architecture expects "
                    "config-level image-token wiring, or rely on a documented custom "
                    "processor path."
                ),
                details={"image_tokens": tuple(sorted(actual))},
            )
        if wrappers:
            return self._result(
                "warn",
                "medium",
                "Tokenizer/template metadata contains only wrapper vision tokens, not an "
                "actual image placeholder.",
                remediation="Ensure the actual image placeholder token is registered and used.",
                details={"wrapper_tokens": tuple(sorted(wrappers))},
            )

        return self._result("skip", "info", "No image-token metadata to cross-check.")

    def _result(
        self,
        status: str,
        severity: str,
        message: str,
        *,
        remediation: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> CheckResult:
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status=status,  # type: ignore[arg-type]
            severity=severity,  # type: ignore[arg-type]
            message=message,
            remediation=remediation,
            details=details or {},
        )


def _config_image_token_ids(config: Mapping[str, object]) -> tuple[dict[str, int], str | None]:
    out: dict[str, int] = {}
    for key in ("image_token_id", "image_token_index"):
        if key not in config:
            continue
        value = config[key]
        if type(value) is not int:
            return {}, key
        out[key] = value
    return out, None
