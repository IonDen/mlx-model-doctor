"""Shared fakes for the ``parity`` subsystem's offline (MLX/numpy-free) tests."""

import json
import math
import struct
from collections.abc import Callable, Sequence
from pathlib import Path


def safetensors_header_bytes(tensors: dict[str, dict[str, object]]) -> bytes:
    """Build the 8-byte-length + JSON header of a safetensors file, with no tensor data."""
    raw = json.dumps(tensors).encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def lora_tensor(shape: Sequence[int], *, dtype: str = "F32") -> dict[str, object]:
    """Return a safetensors header tensor entry for a shape, with zero-length dummy data."""
    return {"dtype": dtype, "shape": list(shape), "data_offsets": [0, 0]}


def write_adapter_dir(
    directory: Path,
    *,
    fine_tune_type: str = "lora",
    scale: float = 20.0,
    tensors: dict[str, dict[str, object]] | None = None,
    config_overrides: dict[str, object] | None = None,
    omit_config: bool = False,
    omit_weights: bool = False,
) -> Path:
    """Write a minimal ``adapter_config.json`` + ``adapters.safetensors`` (header-only) pair."""
    directory.mkdir(parents=True, exist_ok=True)
    if not omit_config:
        config: dict[str, object] = {
            "num_layers": 1,
            "fine_tune_type": fine_tune_type,
            "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": scale, "keys": ["target"]},
        }
        if config_overrides:
            config.update(config_overrides)
        (directory / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    if not omit_weights:
        header = tensors if tensors is not None else {}
        (directory / "adapters.safetensors").write_bytes(safetensors_header_bytes(header))
    return directory


def make_import_module(mapping: dict[str, object]) -> Callable[[str], object]:
    """Return an ``importlib.import_module`` stand-in resolving names from a mapping."""

    def _import_module(name: str) -> object:
        if name in mapping:
            return mapping[name]
        raise ImportError(name)

    return _import_module


class OrderAwareCaps:
    """Caps installer that records whether ``install`` ran before the backend was called."""

    def __init__(self, caps: tuple[int, int] = (20, 22)) -> None:
        self._caps = caps
        self.installed_before_load = False

    def install(self) -> tuple[int, int]:
        self.installed_before_load = True
        return self._caps


class OrderCheckingBackend:
    """Backend that asserts caps were installed before ``load_argmax`` runs (F-review H4)."""

    def __init__(self, caps: OrderAwareCaps, result: object) -> None:
        self._caps = caps
        self._result = result

    def load_argmax(self, spec: object) -> object:
        assert self._caps.installed_before_load, (
            "caps must be installed before backend.load_argmax is called"
        )
        return self._result


class NeverCalledBackend:
    """Backend whose ``load_argmax`` must never run (proves refuse-if-uncapped)."""

    def load_argmax(self, spec: object) -> object:
        raise AssertionError("load_argmax must not be called when memory caps are uncapped")


class StubBackend:
    """Backend that returns a fixed ``WorkerResult`` regardless of the spec."""

    def __init__(self, result: object) -> None:
        self._result = result

    def load_argmax(self, spec: object) -> object:
        return self._result


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


class FakeIntVector:
    """Fake stand-in for the mx array returned by ``mx.argmax``."""

    def __init__(self, values: list[int]) -> None:
        self._values = values

    def tolist(self) -> list[int]:
        return list(self._values)


class FakeMxCore:
    """Fake ``mlx.core`` surface for offline parity-worker tests."""

    def __init__(
        self,
        *,
        logits: list[list[float]] | None = None,
        peak_memory: int = 4096,
        memory_size: int = 34_359_738_368,
        active_memory: int = 0,
        cache_memory: int = 0,
        cap_failure: str | None = None,
    ) -> None:
        self._logits = logits if logits is not None else [[0.1, 5.0, 0.2], [3.0, 0.1, 0.2]]
        self._peak_memory = peak_memory
        self._memory_size = memory_size
        self._active_memory = active_memory
        self._cache_memory = cache_memory
        self._cap_failure = cap_failure
        self.wired_limit: int | None = None
        self.memory_limit: int | None = None
        self.cache_limit: int | None = None
        self.eval_calls = 0

    def device_info(self) -> dict[str, object]:
        if self._cap_failure == "device_info":
            raise RuntimeError("device unavailable")
        return {
            "max_recommended_working_set_size": 25 * 1024**3,
            "memory_size": self._memory_size,
        }

    def set_wired_limit(self, value: int) -> None:
        if self._cap_failure == "set_wired_limit":
            raise RuntimeError("wired limit unavailable")
        self.wired_limit = value

    def set_memory_limit(self, value: int) -> None:
        if self._cap_failure == "set_memory_limit":
            raise RuntimeError("memory limit unavailable")
        self.memory_limit = value

    def set_cache_limit(self, value: int) -> None:
        self.cache_limit = value

    def get_active_memory(self) -> int:
        return self._active_memory

    def get_cache_memory(self) -> int:
        return self._cache_memory

    def get_peak_memory(self) -> int:
        return self._peak_memory

    def array(self, value: object) -> object:
        return value

    def isfinite(self, value: list[list[float]]) -> list[list[bool]]:
        return [[_is_finite(v) for v in row] for row in value]

    def all(self, value: list[list[bool]]) -> bool:
        return all(all(row) for row in value)

    def abs(self, value: Sequence[float]) -> list[float]:
        return [abs(v) for v in value]

    def max(self, value: Sequence[float]) -> float:
        return max(value)

    def argmax(self, value: list[list[float]], axis: int) -> FakeIntVector:
        return FakeIntVector([row.index(max(row)) for row in value])

    def eval(self, *values: object) -> None:
        self.eval_calls += 1


class FakeLoraModule:
    """Fake structural stand-in for a LoRA/DoRA-family module (duck-typed)."""

    def __init__(
        self,
        *,
        lora_a: list[float] | None = None,
        lora_b: list[float] | None = None,
        m: list[float] | None = None,
    ) -> None:
        if lora_a is not None:
            self.lora_a = lora_a
        if lora_b is not None:
            self.lora_b = lora_b
        if m is not None:
            self.m = m


class FakeMlxLmModel:
    """Fake ``nn.Module``-shaped model: callable + ``named_modules``."""

    def __init__(
        self,
        logits: list[list[float]],
        modules: list[tuple[str, object]] | None = None,
    ) -> None:
        self._logits = logits
        self._modules = modules if modules is not None else []
        self.called_with: object | None = None

    def __call__(self, ids: object) -> list[list[list[float]]]:
        self.called_with = ids
        return [self._logits]

    def named_modules(self) -> list[tuple[str, object]]:
        return list(self._modules)


class FakeMlxLmModule:
    """Fake top-level ``mlx_lm`` module exposing ``load``."""

    def __init__(
        self,
        model: object,
        tokenizer: object = "tok",
        *,
        raise_error: Exception | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._raise_error = raise_error
        self.load_calls: list[dict[str, object]] = []

    def load(self, path_or_repo: str, *, adapter_path: str | None = None) -> tuple[object, object]:
        self.load_calls.append({"path": path_or_repo, "adapter_path": adapter_path})
        if self._raise_error is not None:
            raise self._raise_error
        return (self._model, self._tokenizer)


class LoadForbiddenMlxLmModule:
    """Fake ``mlx_lm`` whose ``load`` must never be called (proves refuse-if-uncapped)."""

    def __init__(self) -> None:
        self.load_calls = 0

    def load(self, path_or_repo: str, *, adapter_path: str | None = None) -> tuple[object, object]:
        self.load_calls += 1
        raise AssertionError("mlx_lm.load() must not be called without MLX memory caps")
