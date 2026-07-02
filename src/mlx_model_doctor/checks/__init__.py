"""Static, weight-header, and smoke checks for MLX model repositories."""

from mlx_model_doctor.checks.chat_template import (
    ChatTemplatePresenceCheck,
    ChatTemplateSpecialTokensCheck,
)
from mlx_model_doctor.checks.compat import MlxCompatSignalCheck
from mlx_model_doctor.checks.config import ConfigJsonCheck, ModelTypeCheck
from mlx_model_doctor.checks.files import RequiredConfigCheck
from mlx_model_doctor.checks.generation_config import GenerationConfigTokensCheck
from mlx_model_doctor.checks.memory import MemoryEstimateCheck
from mlx_model_doctor.checks.quantization import (
    MlxQuantizationModeCheck,
    MlxQuantShapeCheck,
    QuantizationMetadataCheck,
)
from mlx_model_doctor.checks.safetensors import SafetensorsIndexCheck, SafetensorsOffsetScanCheck
from mlx_model_doctor.checks.smoke import MlxLmSmokeCheck
from mlx_model_doctor.checks.tokenizer import SpecialTokensCheck, TokenizerFilesCheck
from mlx_model_doctor.checks.vlm import VlmImageProcessorCheck
from mlx_model_doctor.checks.weights import TiedEmbeddingCheck, WeightParamCountCheck

__all__ = [
    "ChatTemplatePresenceCheck",
    "ChatTemplateSpecialTokensCheck",
    "ConfigJsonCheck",
    "GenerationConfigTokensCheck",
    "MemoryEstimateCheck",
    "MlxCompatSignalCheck",
    "MlxLmSmokeCheck",
    "MlxQuantShapeCheck",
    "MlxQuantizationModeCheck",
    "ModelTypeCheck",
    "QuantizationMetadataCheck",
    "RequiredConfigCheck",
    "SafetensorsIndexCheck",
    "SafetensorsOffsetScanCheck",
    "SpecialTokensCheck",
    "TiedEmbeddingCheck",
    "TokenizerFilesCheck",
    "VlmImageProcessorCheck",
    "WeightParamCountCheck",
]
