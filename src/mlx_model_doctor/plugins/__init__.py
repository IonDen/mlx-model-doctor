"""Plugin registry."""

from collections.abc import Mapping
from types import MappingProxyType

from mlx_model_doctor.errors import ModelDoctorError
from mlx_model_doctor.plugins.base import DoctorPlugin
from mlx_model_doctor.plugins.text import TextModelPlugin

TEXT_PLUGIN: DoctorPlugin = TextModelPlugin()
BUILTIN_PLUGINS: Mapping[str, DoctorPlugin] = MappingProxyType(
    {"text": TEXT_PLUGIN}
)


def get_plugin(name: str) -> DoctorPlugin:
    """Return a built-in plugin by name."""
    plugin = BUILTIN_PLUGINS.get(name)
    if plugin is None:
        raise ModelDoctorError(f"Unknown plugin: {name}")
    return plugin


__all__ = ["BUILTIN_PLUGINS", "DoctorPlugin", "TextModelPlugin", "get_plugin"]
