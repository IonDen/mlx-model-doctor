from tests.conftest import _markers_to_skip


def test_default_run_skips_network_smoke_and_benchmark() -> None:
    skipped = dict(_markers_to_skip(set()))

    assert "network" in skipped
    assert "smoke" in skipped
    assert "benchmark" in skipped


def test_gpu_is_not_an_opt_in_gate() -> None:
    skipped = dict(_markers_to_skip(set()))
    assert "gpu" not in skipped


def test_enabled_flag_unskips_only_its_own_marker() -> None:
    # Exercise the flag-present branch: --run-network must drop *only* network
    # from the skip list, leaving the other gates intact. The all-absent case
    # alone never runs the `flag not in enabled_flags` filter against a present flag.
    skipped = dict(_markers_to_skip({"--run-network"}))

    assert "network" not in skipped
    assert "smoke" in skipped
    assert "benchmark" in skipped
