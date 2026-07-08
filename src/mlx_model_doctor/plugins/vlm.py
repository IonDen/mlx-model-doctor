"""Built-in vision-language model plugin."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from mlx_model_doctor.checks.base import ModelCheck
from mlx_model_doctor.checks.chat_template import (
    ChatTemplatePresenceCheck,
    ChatTemplateSpecialTokensCheck,
)
from mlx_model_doctor.checks.compat import MlxCompatSignalCheck
from mlx_model_doctor.checks.config import ConfigJsonCheck, ModelTypeCheck
from mlx_model_doctor.checks.files import RequiredConfigCheck
from mlx_model_doctor.checks.generation_config import GenerationConfigTokensCheck
from mlx_model_doctor.checks.quantization import (
    MlxQuantizationModeCheck,
    MlxQuantShapeCheck,
    QuantizationMetadataCheck,
)
from mlx_model_doctor.checks.safetensors import SafetensorsIndexCheck, SafetensorsOffsetScanCheck
from mlx_model_doctor.checks.tokenizer import SpecialTokensCheck, TokenizerFilesCheck
from mlx_model_doctor.checks.vlm import (
    VlmImageProcessorCheck,
    VlmImageTokenWiringCheck,
    VlmMemoryEstimateCheck,
)
from mlx_model_doctor.checks.weights import TiedEmbeddingCheck, WeightParamCountCheck


@dataclass(frozen=True, slots=True)
class VlmModelPlugin:
    """Plugin for vision-language model repositories."""

    name: str = "vlm"

    def static_checks(self) -> Sequence[ModelCheck]:
        """Return static VLM checks in stable execution order."""
        return cast(
            "Sequence[ModelCheck]",
            (
                RequiredConfigCheck(check_id="vlm/files.required"),
                ConfigJsonCheck(check_id="vlm/config.json"),
                ModelTypeCheck(check_id="vlm/config.model_type"),
                MlxCompatSignalCheck(check_id="vlm/compat.mlx_signal"),
                TokenizerFilesCheck(check_id="vlm/tokenizer.files"),
                SpecialTokensCheck(check_id="vlm/tokenizer.special_tokens"),
                ChatTemplatePresenceCheck(check_id="vlm/chat_template.presence"),
                ChatTemplateSpecialTokensCheck(check_id="vlm/chat_template.special_tokens"),
                SafetensorsIndexCheck(check_id="vlm/safetensors.index"),
                QuantizationMetadataCheck(check_id="vlm/quantization.metadata"),
                MlxQuantizationModeCheck(check_id="vlm/quantization.mode"),
                VlmImageProcessorCheck(check_id="vlm/image_processor"),
                VlmImageTokenWiringCheck(),
                GenerationConfigTokensCheck(check_id="vlm/generation_config.tokens"),
                VlmMemoryEstimateCheck(),
            ),
        )

    def weight_checks(self) -> Sequence[ModelCheck]:
        """Return VLM safetensors-header checks in stable execution order."""
        return cast(
            "Sequence[ModelCheck]",
            (
                SafetensorsOffsetScanCheck(check_id="vlm/safetensors.offsets"),
                WeightParamCountCheck(check_id="vlm/weights.param_count"),
                TiedEmbeddingCheck(check_id="vlm/weights.tied_embedding"),
                MlxQuantShapeCheck(check_id="vlm/quantization.shape"),
            ),
        )

    def smoke_checks(self) -> Sequence[ModelCheck]:
        """Return runtime smoke checks for this plugin."""
        return ()
