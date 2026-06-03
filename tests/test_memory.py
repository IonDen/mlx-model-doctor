import pytest

from mlx_model_doctor.memory import parse_memory


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
