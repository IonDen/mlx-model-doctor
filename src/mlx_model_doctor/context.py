"""Shared check context."""

import json
from dataclasses import dataclass, field
from typing import Literal, cast

from mlx_model_doctor.errors import TargetError, raise_for_hf_target_error
from mlx_model_doctor.safetensors_header import SafetensorsHeader, SafetensorsHeaderError
from mlx_model_doctor.targets import ModelTarget

_MAX_METADATA_BYTES = 16 * 1024 * 1024


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
    _file_cache: dict[str, object] = field(default_factory=dict, init=False, repr=False)

    def config_json(self) -> dict[str, object] | None:
        """Return parsed config JSON, or None when it is absent or unusable."""
        return self._read_json_file("config.json")

    def tokenizer_config_json(self) -> dict[str, object] | None:
        """Return parsed tokenizer_config.json, or None when absent/unusable."""
        return self._read_json_file("tokenizer_config.json")

    def special_tokens_map_json(self) -> dict[str, object] | None:
        """Return parsed special_tokens_map.json, or None when absent/unusable."""
        return self._read_json_file("special_tokens_map.json")

    def generation_config_json(self) -> dict[str, object] | None:
        """Return parsed generation_config.json, or None when absent/unusable."""
        return self._read_json_file("generation_config.json")

    def chat_template_text(self) -> str | None:
        """Return the sibling chat_template.jinja text, or None when absent/unusable."""
        key = "text:chat_template.jinja"
        if key in self._file_cache:
            return cast("str | None", self._file_cache[key])
        text = self._read_text_guarded("chat_template.jinja")
        self._file_cache[key] = text
        return text

    def safetensors_header(self) -> SafetensorsHeader | None:
        """Return the cached safetensors header (one fetch shared across checks)."""
        self._ensure_safetensors_header_loaded()
        return cast("SafetensorsHeader | None", self._file_cache["safetensors:header"])

    def safetensors_header_error(self) -> str | None:
        """Return why a present safetensors header could not be read, or None when absent/ok."""
        self._ensure_safetensors_header_loaded()
        return cast("str | None", self._file_cache["safetensors:header_error"])

    def _ensure_safetensors_header_loaded(self) -> None:
        if "safetensors:header" in self._file_cache:
            return
        header: SafetensorsHeader | None = None
        error: str | None = None
        try:
            header = self.target.safetensors_header()
        except SafetensorsHeaderError as exc:
            error = str(exc)
        except TargetError as exc:
            raise_for_hf_target_error(exc)
            error = str(exc)
        self._file_cache["safetensors:header"] = header
        self._file_cache["safetensors:header_error"] = error

    def _read_json_file(self, name: str) -> dict[str, object] | None:
        if name in self._file_cache:
            return cast("dict[str, object] | None", self._file_cache[name])
        parsed = self._load_json(name)
        self._file_cache[name] = parsed
        return parsed

    def _load_json(self, name: str) -> dict[str, object] | None:
        text = self._read_text_guarded(name)
        if text is None:
            return None
        try:
            value: object = json.loads(text)
        except (json.JSONDecodeError, UnicodeError):
            return None
        return cast("dict[str, object]", value) if isinstance(value, dict) else None

    def _read_text_guarded(self, name: str) -> str | None:
        try:
            if not self.target.exists(name):
                return None
            size = self.target.size(name)
            if size is not None and size > _MAX_METADATA_BYTES:
                return None
            return self.target.read_text(name)
        except TargetError as exc:
            raise_for_hf_target_error(exc)
            return None
        except (FileNotFoundError, UnicodeError):
            return None
