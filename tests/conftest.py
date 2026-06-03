"""Pytest gates and MLX memory safety guard."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping

GIB = 1024**3
DESIRED_WIRED_GB = 20
DESIRED_MEMORY_GB = 22
DEFAULT_SMALL_MAC_HEADROOM_GB = 8
DEFAULT_LARGE_MAC_HEADROOM_GB = 12
WORKING_SET_HEADROOM_GB = 2

GATED_MARKERS: tuple[tuple[str, str, str], ...] = (
    ("network", "--run-network", "real network I/O"),
    ("smoke", "--run-smoke", "runtime model smoke checks"),
    ("benchmark", "--run-benchmark", "performance timings"),
)


def _gib_from_device_value(value: object) -> int:
    try:
        return int(value) // GIB
    except (TypeError, ValueError):
        return 0


def _caps_from_device_info(info: Mapping[str, object]) -> tuple[int, int]:
    recommended_gb = _gib_from_device_value(info.get("max_recommended_working_set_size"))
    if recommended_gb <= 0:
        total_gb = _gib_from_device_value(info.get("memory_size"))
        if total_gb <= 0:
            return (0, 0)
        headroom_gb = (
            DEFAULT_SMALL_MAC_HEADROOM_GB if total_gb <= 36 else DEFAULT_LARGE_MAC_HEADROOM_GB
        )
        recommended_gb = max(1, total_gb - headroom_gb)

    if recommended_gb < 2:
        return (0, 0)

    memory_gb = min(DESIRED_MEMORY_GB, recommended_gb)
    wired_gb = min(DESIRED_WIRED_GB, max(1, recommended_gb - WORKING_SET_HEADROOM_GB))
    if wired_gb >= memory_gb:
        wired_gb = memory_gb - 1
    return (wired_gb, memory_gb)


def _install_mlx_memory_caps() -> tuple[int, int]:
    try:
        import mlx.core as mx
    except ImportError:
        return (0, 0)

    try:
        info = mx.device_info()
    except RuntimeError:
        return (0, 0)

    wired_gb, memory_gb = _caps_from_device_info(info)
    if wired_gb == 0:
        return (0, 0)

    try:
        mx.set_wired_limit(wired_gb * GIB)
        mx.set_memory_limit(memory_gb * GIB)
    except Exception:
        return (0, 0)
    return (wired_gb, memory_gb)


INSTALLED_CAPS_GB = _install_mlx_memory_caps()


def _markers_to_skip(enabled_flags: set[str]) -> list[tuple[str, str]]:
    return [
        (marker, f"requires {flag} ({description})")
        for marker, flag, description in GATED_MARKERS
        if flag not in enabled_flags
    ]


def pytest_addoption(parser: pytest.Parser) -> None:
    for marker, flag, description in GATED_MARKERS:
        parser.addoption(
            flag,
            action="store_true",
            default=False,
            help=f"run `{marker}` tests ({description}); skipped by default",
        )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    enabled = {flag for _marker, flag, _description in GATED_MARKERS if config.getoption(flag)}
    for marker, reason in _markers_to_skip(enabled):
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
