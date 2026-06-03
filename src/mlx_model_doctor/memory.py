"""Memory parsing helpers."""

import re
from decimal import Decimal, InvalidOperation

_MEMORY_PATTERN = re.compile(r"^\s*(?P<amount>(?:\d+(?:\.\d+)?)|(?:\.\d+))\s*(?P<unit>[kmgt]?i?b)\s*$", re.IGNORECASE)
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
