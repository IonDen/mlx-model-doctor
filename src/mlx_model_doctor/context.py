"""Shared check context."""

import json
from dataclasses import dataclass, field
from typing import Literal, cast

from mlx_model_doctor.errors import TargetError
from mlx_model_doctor.targets import ModelTarget

_UNSET = object()


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckOptions:
    """Options controlling check execution."""

    max_memory_bytes: int | None
    context_length: int
    include_weights: bool
    smoke: bool
    verbosity: Literal["quiet", "normal", "verbose"]


@dataclass(slots=True, kw_only=True)
class CheckContext:
    """Shared state for checks against a model target."""

    target: ModelTarget
    options: CheckOptions
    _config_json: object = field(default=_UNSET, init=False, repr=False)

    def config_json(self) -> dict[str, object] | None:
        """Return parsed config JSON, or None when it is absent or unusable."""
        if self._config_json is not _UNSET:
            if self._config_json is None:
                return None
            return cast("dict[str, object]", self._config_json)

        try:
            if not self.target.exists("config.json"):
                self._config_json = None
                return None
            raw_config = self.target.read_text("config.json")
            parsed_config: object = json.loads(raw_config)
        except (FileNotFoundError, TargetError, json.JSONDecodeError, UnicodeError):
            self._config_json = None
            return None

        if not isinstance(parsed_config, dict):
            self._config_json = None
            return None

        config = cast("dict[str, object]", parsed_config)
        self._config_json = config
        return config
