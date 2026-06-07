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


@dataclass(frozen=True, slots=True)
class VlmImageProcessorCheck:
    """Flag a VLM whose missing image_processor_type would break standard load."""

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

        if _has_image_processor_automap(preproc, config) or _has_feature_extractor_type(preproc):
            return self._result(
                "pass",
                "info",
                "No image_processor_type, but a feature-extractor / custom-processor path is "
                "declared; standard AutoImageProcessor is not required.",
            )

        return self._result(
            "fail",
            "high",
            "Vision-language repo has no image_processor_type; standard AutoImageProcessor load "
            "raises ValueError.",
            remediation="Add image_processor_type to preprocessor_config.json "
            '(for example "Qwen2VLImageProcessor").',
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


def _has_image_processor_automap(
    preproc: Mapping[str, object] | None,
    config: Mapping[str, object] | None,
) -> bool:
    for cfg in (preproc, config):
        if isinstance(cfg, Mapping):
            auto_map = cfg.get("auto_map")
            if isinstance(auto_map, Mapping) and (
                "AutoImageProcessor" in auto_map or "AutoFeatureExtractor" in auto_map
            ):
                return True
    return False


def _has_feature_extractor_type(preproc: Mapping[str, object] | None) -> bool:
    if not isinstance(preproc, Mapping):
        return False
    value = preproc.get("feature_extractor_type")
    return isinstance(value, str) and bool(value.strip())
