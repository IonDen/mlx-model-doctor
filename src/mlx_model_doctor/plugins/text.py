"""Built-in text model plugin."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from mlx_model_doctor.checks.base import ModelCheck
from mlx_model_doctor.checks.config import ConfigJsonCheck, ModelTypeCheck
from mlx_model_doctor.checks.files import RequiredConfigCheck
from mlx_model_doctor.checks.memory import MemoryEstimateCheck
from mlx_model_doctor.checks.quantization import QuantizationMetadataCheck
from mlx_model_doctor.checks.safetensors import SafetensorsIndexCheck
from mlx_model_doctor.checks.smoke import MlxLmSmokeCheck
from mlx_model_doctor.checks.tokenizer import SpecialTokensCheck, TokenizerFilesCheck


@dataclass(frozen=True, slots=True)
class TextModelPlugin:
    """Plugin for text-generation model repositories."""

    name: str = "text"

    def static_checks(self) -> Sequence[ModelCheck]:
        """Return static checks in stable execution order."""
        return cast(
            "Sequence[ModelCheck]",
            (
                RequiredConfigCheck(),
                ConfigJsonCheck(),
                ModelTypeCheck(),
                TokenizerFilesCheck(),
                SpecialTokensCheck(),
                SafetensorsIndexCheck(),
                QuantizationMetadataCheck(),
                MemoryEstimateCheck(),
            ),
        )

    def smoke_checks(self) -> Sequence[ModelCheck]:
        """Return smoke checks for this plugin."""
        return cast("Sequence[ModelCheck]", (MlxLmSmokeCheck(),))
