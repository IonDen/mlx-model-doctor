"""Plugin protocol for model doctor checks."""

from collections.abc import Sequence
from typing import Protocol

from mlx_model_doctor.checks.base import ModelCheck


class DoctorPlugin(Protocol):
    """Plugin that supplies checks for a model family."""

    @property
    def name(self) -> str:
        """Return the plugin name."""

    def static_checks(self) -> Sequence[ModelCheck]:
        """Return static checks for this plugin."""

    def weight_checks(self) -> Sequence[ModelCheck]:
        """Return checks that read the safetensors tensor header."""

    def smoke_checks(self) -> Sequence[ModelCheck]:
        """Return smoke checks for this plugin."""
