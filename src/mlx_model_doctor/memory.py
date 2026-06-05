"""Memory parsing helpers."""

import importlib
import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Protocol, cast

_MEMORY_PATTERN = re.compile(
    r"^\s*(?P<amount>(?:\d+(?:\.\d+)?)|(?:\.\d+))\s*(?P<unit>[kmgt]?i?b)\s*$", re.IGNORECASE
)
_UNIT_MULTIPLIERS = {
    "b": 1,
    "kb": 1024,
    "kib": 1024,
    "mb": 1024**2,
    "mib": 1024**2,
    "gb": 1024**3,
    "gib": 1024**3,
    "tb": 1024**4,
    "tib": 1024**4,
}
GIB = 1024**3
DESIRED_WIRED_GIB = 20
DESIRED_MEMORY_GIB = 22
WORKING_SET_HEADROOM_GIB = 2
DEFAULT_SMOKE_BUDGET_FRACTION = 0.8


class MlxMemoryModule(Protocol):
    """MLX memory API surface used by production and tests."""

    def device_info(self) -> Mapping[str, object]:
        """Return MLX device metadata."""

    def set_wired_limit(self, value: int) -> None:
        """Set MLX wired-memory limit in bytes."""

    def set_memory_limit(self, value: int) -> None:
        """Set MLX memory limit in bytes."""


def parse_memory(value: str) -> int:
    """Parse a memory size string into bytes."""
    match = _MEMORY_PATTERN.match(value)
    if match is None:
        raise ValueError(f"Invalid memory value: {value!r}")

    try:
        amount = Decimal(match.group("amount"))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid memory value: {value!r}") from exc
    unit = match.group("unit").lower()
    multiplier = _UNIT_MULTIPLIERS.get(unit)
    if multiplier is None:
        raise ValueError(f"Invalid memory value: {value!r}")

    # Bare decimal-looking units intentionally map to binary powers. In MLX/macOS
    # workflows, "32gb" is normally used as ergonomic shorthand for 32 GiB.
    return int(amount * multiplier)


def caps_from_device_info(info: Mapping[str, object]) -> tuple[int, int]:
    """Return conservative MLX caps in GiB from recommended working-set metadata."""
    recommended_gib = _gib_from_device_value(info.get("max_recommended_working_set_size"))
    if recommended_gib < 2:
        return (0, 0)

    memory_gib = min(DESIRED_MEMORY_GIB, recommended_gib)
    wired_gib = min(
        DESIRED_WIRED_GIB,
        max(1, recommended_gib - WORKING_SET_HEADROOM_GIB),
    )
    if wired_gib >= memory_gib:
        wired_gib = memory_gib - 1
    return (wired_gib, memory_gib)


def install_mlx_memory_caps(mx_module: MlxMemoryModule | None = None) -> tuple[int, int]:
    """Install hard MLX memory caps, returning the configured GiB limits."""
    mx = _import_mlx_core() if mx_module is None else mx_module
    if mx is None:
        return (0, 0)

    try:
        info = mx.device_info()
    except Exception:
        return (0, 0)

    wired_gib, memory_gib = caps_from_device_info(info)
    if wired_gib == 0:
        return (0, 0)

    try:
        mx.set_wired_limit(wired_gib * GIB)
        mx.set_memory_limit(memory_gib * GIB)
    except Exception:
        return (0, 0)
    return (wired_gib, memory_gib)


def smoke_budget_from_device_info(
    info: Mapping[str, object],
    *,
    fraction: float = DEFAULT_SMOKE_BUDGET_FRACTION,
) -> int | None:
    """Return a safe smoke-check budget in bytes from MLX recommended working set."""
    recommended_bytes = _positive_device_bytes(info.get("max_recommended_working_set_size"))
    if recommended_bytes <= 0 or fraction <= 0:
        return None
    safe_fraction = min(fraction, 1.0)
    return int(recommended_bytes * safe_fraction)


def _import_mlx_core() -> MlxMemoryModule | None:
    try:
        return cast("MlxMemoryModule", importlib.import_module("mlx.core"))
    except ImportError:
        return None


def _gib_from_device_value(value: object) -> int:
    return _positive_device_bytes(value) // GIB


def _positive_device_bytes(value: object) -> int:
    if type(value) is int:
        return max(0, value)
    if type(value) is float:
        return max(0, int(value))
    if not isinstance(value, str):
        return 0
    try:
        bytes_value = int(value)
    except ValueError:
        return 0
    return max(0, bytes_value)
