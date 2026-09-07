"""Shared fakes for the ``parity`` subsystem's offline (MLX/numpy-free) tests."""

import json
import math
import struct
import time
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


# --- Orchestrator fakes (offline, pure Python) --------------------------------------


class TrackingLauncher:
    """Fake ``Launcher`` recording call order + in-flight count (proves F2 strict-serial run).

    Sleeps ``delay_s`` between recording entry and exit so an overlapping call
    would actually be observed as ``active > 1``. Without that delay, CPython's
    GIL makes the increment/decrement window too narrow to ever catch a
    concurrency regression: a mutant ``run_parity_workers`` that ran specs via a
    4-worker ``ThreadPoolExecutor`` scored ``max_concurrent == 1`` in 200/200
    trials against the undelayed fake (confirmed by review), and only the
    delayed version can fail on that mutant.
    """

    def __init__(self, outcomes: Sequence[object], *, delay_s: float = 0.01) -> None:
        self._outcomes = list(outcomes)
        self._delay_s = delay_s
        self.calls: list[object] = []
        self.active = 0
        self.max_concurrent = 0

    def run_one(self, spec: object) -> object:
        self.calls.append(spec)
        self.active += 1
        self.max_concurrent = max(self.max_concurrent, self.active)
        try:
            time.sleep(self._delay_s)
            return self._outcomes[len(self.calls) - 1]
        finally:
            self.active -= 1


class RaisingLauncher:
    """Fake ``Launcher`` whose ``run_one`` raises for specs with a role in ``raising_roles``."""

    def __init__(self, outcomes_by_role: dict[str, object], raising_roles: Sequence[str]) -> None:
        self._outcomes_by_role = outcomes_by_role
        self._raising_roles = set(raising_roles)
        self.calls: list[str] = []

    def run_one(self, spec: object) -> object:
        role = spec.role  # type: ignore[attr-defined]
        self.calls.append(role)
        if role in self._raising_roles:
            raise RuntimeError(f"boom for role {role}")
        return self._outcomes_by_role[role]


# --- SubprocessLauncher stub scripts (offline, no mlx) ------------------------------
#
# Each stub is a trivial standalone Python script spawned as a real subprocess by
# ``SubprocessLauncher`` tests, so they intentionally do their own minimal argv
# parsing rather than importing anything from this package.

_STUB_SUCCESS_SOURCE = """
import json
import sys


def _arg(flag):
    return sys.argv[sys.argv.index(flag) + 1]


out_path = _arg("--out")
fixture_id = _arg("--fixture-id")
role = _arg("--role")
token_ids = [t for t in _arg("--token-ids").split(",") if t]
payload = {
    "fixture_id": fixture_id,
    "role": role,
    "argmax": [0] * len(token_ids),
    "peak_bytes": 123,
    "adapter_applied": None,
}
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle)
print(f"::PARITY_WORKER::ok role={role} fixture={fixture_id}")
sys.exit(0)
"""

_STUB_WRONG_LENGTH_SOURCE = """
import json
import sys


def _arg(flag):
    return sys.argv[sys.argv.index(flag) + 1]


out_path = _arg("--out")
fixture_id = _arg("--fixture-id")
role = _arg("--role")
payload = {
    "fixture_id": fixture_id,
    "role": role,
    "argmax": [0],
    "peak_bytes": 123,
    "adapter_applied": None,
}
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle)
print(f"::PARITY_WORKER::ok role={role} fixture={fixture_id}")
sys.exit(0)
"""

_STUB_EXIT_NONZERO_SOURCE = """
import sys

sys.exit(7)
"""

_STUB_MISSING_OUTPUT_SOURCE = """
import sys

fixture_id = sys.argv[sys.argv.index("--fixture-id") + 1]
role = sys.argv[sys.argv.index("--role") + 1]
print(f"::PARITY_WORKER::ok role={role} fixture={fixture_id}")
sys.exit(0)
"""

_STUB_MALFORMED_OUTPUT_SOURCE = """
import sys


def _arg(flag):
    return sys.argv[sys.argv.index(flag) + 1]


out_path = _arg("--out")
fixture_id = _arg("--fixture-id")
role = _arg("--role")
with open(out_path, "w", encoding="utf-8") as handle:
    handle.write("{not valid json")
print(f"::PARITY_WORKER::ok role={role} fixture={fixture_id}")
sys.exit(0)
"""

_STUB_NONRESPONSIVE_SOURCE = """
import signal
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(30)
"""

_STUB_SLOW_COOPERATIVE_SOURCE = """
import time

time.sleep(30)
"""


def write_stub_script(path: Path, source: str) -> Path:
    """Write a trivial worker-stub script (no mlx) for offline ``SubprocessLauncher`` tests."""
    path.write_text(source, encoding="utf-8")
    return path


def write_success_stub(path: Path) -> Path:
    """Stub that writes a valid worker artifact and prints the success sentinel."""
    return write_stub_script(path, _STUB_SUCCESS_SOURCE)


def write_wrong_length_stub(path: Path) -> Path:
    """Stub that writes a valid-shaped but always length-1 ``argmax`` artifact."""
    return write_stub_script(path, _STUB_WRONG_LENGTH_SOURCE)


def write_exit_nonzero_stub(path: Path) -> Path:
    """Stub that exits nonzero without writing an artifact."""
    return write_stub_script(path, _STUB_EXIT_NONZERO_SOURCE)


def write_missing_output_stub(path: Path) -> Path:
    """Stub that claims success (sentinel + exit 0) but never writes its output file."""
    return write_stub_script(path, _STUB_MISSING_OUTPUT_SOURCE)


def write_malformed_output_stub(path: Path) -> Path:
    """Stub that claims success but writes an invalid-JSON output file."""
    return write_stub_script(path, _STUB_MALFORMED_OUTPUT_SOURCE)


def write_nonresponsive_stub(path: Path) -> Path:
    """Stub that ignores SIGTERM and sleeps, to exercise the parent's TERM->KILL escalation."""
    return write_stub_script(path, _STUB_NONRESPONSIVE_SOURCE)


def write_slow_cooperative_stub(path: Path) -> Path:
    """Stub that sleeps past ``timeout_s`` but honors default SIGTERM handling (no KILL needed)."""
    return write_stub_script(path, _STUB_SLOW_COOPERATIVE_SOURCE)
