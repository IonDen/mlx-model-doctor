import conftest

GIB = 1024**3


def test_caps_keep_32gb_m_series_targets_when_recommended_memory_allows() -> None:
    caps = conftest._caps_from_device_info(
        {
            "memory_size": 32 * GIB,
            "max_recommended_working_set_size": 25 * GIB,
        }
    )

    assert caps == (20, 22)


def test_caps_clamp_below_smaller_recommended_working_set() -> None:
    wired_gb, memory_gb = conftest._caps_from_device_info(
        {
            "memory_size": 16 * GIB,
            "max_recommended_working_set_size": 10 * GIB,
        }
    )

    assert (wired_gb, memory_gb) == (8, 10)
    assert wired_gb < memory_gb <= 10


def test_caps_fall_back_to_total_memory_when_recommended_set_is_missing() -> None:
    assert conftest._caps_from_device_info({"memory_size": 32 * GIB}) == (20, 22)


def test_caps_are_noop_when_device_memory_is_unknown() -> None:
    assert conftest._caps_from_device_info({}) == (0, 0)
