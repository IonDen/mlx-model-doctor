from mlx_model_doctor.memory import (
    GIB,
    caps_from_device_info,
    install_mlx_memory_caps,
    smoke_budget_from_device_info,
)


def test_caps_keep_32gb_m_series_targets_when_recommended_memory_allows() -> None:
    caps = caps_from_device_info(
        {
            "memory_size": 32 * GIB,
            "max_recommended_working_set_size": 25 * GIB,
        }
    )

    assert caps == (20, 22)


def test_caps_clamp_below_smaller_recommended_working_set() -> None:
    wired_gb, memory_gb = caps_from_device_info(
        {
            "memory_size": 16 * GIB,
            "max_recommended_working_set_size": 10 * GIB,
        }
    )

    assert (wired_gb, memory_gb) == (8, 10)
    assert wired_gb < memory_gb <= 10


def test_caps_are_noop_when_device_memory_is_unknown() -> None:
    assert caps_from_device_info({}) == (0, 0)


def test_caps_are_noop_when_recommended_working_set_is_missing() -> None:
    assert caps_from_device_info({"memory_size": 32 * GIB}) == (0, 0)


def test_smoke_budget_uses_safe_fraction_of_recommended_working_set() -> None:
    budget = smoke_budget_from_device_info(
        {"max_recommended_working_set_size": 10 * GIB},
        fraction=0.8,
    )

    assert budget == 8 * GIB


def test_smoke_budget_is_unknown_without_recommended_working_set() -> None:
    assert smoke_budget_from_device_info({"memory_size": 32 * GIB}) is None


def test_install_mlx_memory_caps_sets_byte_limits_on_fake_mx() -> None:
    mx = FakeMx({"max_recommended_working_set_size": 25 * GIB})

    assert install_mlx_memory_caps(mx) == (20, 22)
    assert mx.wired_limit == 20 * GIB
    assert mx.memory_limit == 22 * GIB


def test_install_mlx_memory_caps_is_noop_below_minimum_recommended_working_set() -> None:
    # A recommended working set under the 2-GiB floor yields no usable wired cap.
    # install must bail out at the `wired_gib == 0` guard and never call
    # set_wired_limit — pinning a 0-byte wired cap is worse than installing none.
    # Deleting that guard would call set_wired_limit(0); this catches that.
    mx = FakeMx({"max_recommended_working_set_size": 1 * GIB})

    assert install_mlx_memory_caps(mx) == (0, 0)
    assert mx.wired_limit is None
    assert mx.memory_limit is None


class FakeMx:
    def __init__(self, device_info: dict[str, object]) -> None:
        self._device_info = device_info
        self.wired_limit: int | None = None
        self.memory_limit: int | None = None

    def device_info(self) -> dict[str, object]:
        return self._device_info

    def set_wired_limit(self, value: int) -> None:
        self.wired_limit = value

    def set_memory_limit(self, value: int) -> None:
        self.memory_limit = value
