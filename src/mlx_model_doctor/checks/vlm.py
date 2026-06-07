"""VLM image-processor presence check (gated; non-VLM repos skip)."""

from collections.abc import Mapping
from dataclasses import dataclass

from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.report import CheckResult

_IMAGE_PREPROCESSOR_KEYS: tuple[str, ...] = (
    "image_processor_type",
    "image_mean",
    "image_std",
    "crop_size",
)
_IMAGE_AUTO_MAP_KEYS: tuple[str, ...] = ("AutoImageProcessor", "AutoFeatureExtractor")


@dataclass(frozen=True, slots=True)
class VlmImageProcessorCheck:
    """Report whether a vision-language repo can resolve an image processor at load."""

    check_id: str = "text/vlm.image_processor"
    title: str = "VLM image processor"

    def run(self, ctx: CheckContext) -> CheckResult:
        """Validate a vision-language repo's image_processor_type; skip non-VLM repos."""
        config = ctx.config_json()
        preproc = ctx.preprocessor_config_json()

        if not _is_vlm(config, preproc):
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


def _is_vlm(config: Mapping[str, object] | None, preproc: Mapping[str, object] | None) -> bool:
    if isinstance(config, Mapping) and isinstance(config.get("vision_config"), Mapping):
        return True
    if isinstance(preproc, Mapping):
        return any(key in preproc for key in _IMAGE_PREPROCESSOR_KEYS)
    return False


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
