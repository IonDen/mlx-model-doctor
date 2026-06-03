from tests.conftest import _markers_to_skip


def test_default_run_skips_network_smoke_and_benchmark() -> None:
    skipped = dict(_markers_to_skip(set()))

    assert "network" in skipped
    assert "smoke" in skipped
    assert "benchmark" in skipped


def test_gpu_is_not_an_opt_in_gate() -> None:
    skipped = dict(_markers_to_skip(set()))
    assert "gpu" not in skipped
