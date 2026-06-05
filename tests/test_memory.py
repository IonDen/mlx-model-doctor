import pytest

from mlx_model_doctor.memory import _positive_device_bytes, parse_memory


def test_parse_memory_accepts_binary_gib() -> None:
    assert parse_memory("1gib") == 1024**3


def test_parse_memory_treats_bare_gb_as_gib_for_mlx_workflows() -> None:
    # Bare gb intentionally means GiB in this project.
    assert parse_memory("1gb") == 1024**3


def test_parse_memory_accepts_binary_mb() -> None:
    assert parse_memory("512mb") == 512 * 1024**2


def test_parse_memory_accepts_decimal_values() -> None:
    assert parse_memory("1.5gib") == int(1.5 * 1024**3)


def test_parse_memory_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="Invalid memory"):
        parse_memory("not-memory")


def test_positive_device_bytes_rejects_bool() -> None:
    assert _positive_device_bytes(True) == 0
    assert _positive_device_bytes(False) == 0


def test_positive_device_bytes_accepts_real_int_and_float() -> None:
    assert _positive_device_bytes(2048) == 2048
    assert _positive_device_bytes(2048.0) == 2048
