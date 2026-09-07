"""Tests for memory watchdog and flush-safe abort."""

from unittest import mock

from mlx_model_doctor.parity.watchdog import (
    MemoryProbe,
    _do_abort,
    default_ceiling_bytes,
    should_abort,
)


class TestShouldAbort:
    """Test the abort decision function."""

    def test_below_ceiling_returns_false(self) -> None:
        """should_abort returns False when active + cache < ceiling."""
        assert should_abort(active=100, cache=50, ceiling=200) is False

    def test_at_ceiling_returns_true(self) -> None:
        """should_abort returns True when active + cache == ceiling."""
        assert should_abort(active=100, cache=100, ceiling=200) is True

    def test_above_ceiling_returns_true(self) -> None:
        """should_abort returns True when active + cache > ceiling."""
        assert should_abort(active=150, cache=100, ceiling=200) is True


class TestDefaultCeilingBytes:
    """Test the default ceiling calculation."""

    def test_32gb_device_returns_28gb(self) -> None:
        """default_ceiling_bytes(32 GB) should return 32 GB - 4 GB."""
        device_memory = 34359738368  # 32 GiB in bytes
        expected = 30064771072  # 28 GiB in bytes
        assert default_ceiling_bytes(device_memory) == expected

    def test_ceiling_is_4gb_less_than_device(self) -> None:
        """Ceiling is always device_memory - 4 GB."""
        four_gib = 4 * 1024**3
        device_memory = 100 * 1024**3
        result = default_ceiling_bytes(device_memory)
        assert result == device_memory - four_gib


class TestFlushSafeAbort:
    """Test that _do_abort unconditionally calls os._exit even if flush raises."""

    def test_abort_calls_os_exit_unconditionally(self, tmp_path: any) -> None:
        """_do_abort must call os._exit(3) even if marker write or flush raises."""
        out_dir = str(tmp_path)
        reason = "test abort"

        with (
            mock.patch("os._exit") as mock_exit,
            mock.patch("sys.stdout.flush", side_effect=BrokenPipeError),
        ):
            _do_abort(out_dir, reason)

            # Must have called os._exit(3) even though flush raised
            mock_exit.assert_called_once_with(3)

    def test_abort_writes_marker_on_success(self, tmp_path: any) -> None:
        """_do_abort should write marker file if no exception occurs."""
        out_dir = str(tmp_path)
        reason = "test reason"

        with mock.patch("os._exit"):
            _do_abort(out_dir, reason)

        marker_file = tmp_path / "parity_worker_abort.txt"
        assert marker_file.exists()
        assert marker_file.read_text(encoding="utf-8") == reason

    def test_abort_survives_marker_write_error(self, tmp_path: any) -> None:
        """_do_abort must call os._exit(3) even if marker write fails."""
        out_dir = "/dev/null/no/such/path"  # Will fail to write

        with mock.patch("os._exit") as mock_exit:
            _do_abort(out_dir, "test")
            mock_exit.assert_called_once_with(3)


class TestMemoryProbeProtocol:
    """Test that MemoryProbe is a valid Protocol."""

    def test_memory_probe_has_required_methods(self) -> None:
        """MemoryProbe should be a Protocol with required methods."""

        # This is a structural test: can instantiate a class meeting the protocol
        class ConcreteProbe:
            def active_bytes(self) -> int:
                return 100

            def cache_bytes(self) -> int:
                return 50

        probe: MemoryProbe = ConcreteProbe()
        assert probe.active_bytes() == 100
        assert probe.cache_bytes() == 50


class TestRunWatchdogMemoryTriggered:
    """Test run_watchdog with memory-triggered abort."""

    def test_watchdog_aborts_on_memory_exceeded(self) -> None:
        """run_watchdog should invoke on_abort when memory exceeds ceiling."""
        from mlx_model_doctor.parity.watchdog import run_watchdog

        class RampingProbe:
            def __init__(self) -> None:
                self.step = 0

            def active_bytes(self) -> int:
                return 100 * self.step

            def cache_bytes(self) -> int:
                self.step += 1
                return 50

        probe = RampingProbe()
        ceiling = 100
        abort_called: list[str] = []

        def on_abort(reason: str) -> None:
            abort_called.append(reason)

        # Run watchdog with tight poll interval
        thread = run_watchdog(probe, ceiling, on_abort=on_abort, poll_s=0.01, deadline_s=2.0)
        thread.join(timeout=3.0)

        # Should have called abort when memory exceeded
        assert len(abort_called) > 0
        assert "memory" in abort_called[0].lower()

    def test_watchdog_stays_active_below_ceiling(self) -> None:
        """run_watchdog should not abort when memory stays below ceiling."""
        from mlx_model_doctor.parity.watchdog import run_watchdog

        class ConstantProbe:
            def active_bytes(self) -> int:
                return 50

            def cache_bytes(self) -> int:
                return 30

        probe = ConstantProbe()
        ceiling = 100
        abort_called: list[str] = []

        def on_abort(reason: str) -> None:
            abort_called.append(reason)

        # Run watchdog with short deadline
        thread = run_watchdog(probe, ceiling, on_abort=on_abort, poll_s=0.01, deadline_s=0.1)
        thread.join(timeout=1.0)

        # Should timeout naturally without calling abort
        # (abort may be called with deadline reason, but NOT memory reason)
        for reason in abort_called:
            assert "memory" not in reason.lower()


class TestRunWatchdogDeadlineTriggered:
    """Test run_watchdog with deadline-triggered abort."""

    def test_watchdog_aborts_on_deadline(self) -> None:
        """run_watchdog should invoke on_abort when deadline elapses."""
        from mlx_model_doctor.parity.watchdog import run_watchdog

        class ConstantProbe:
            def active_bytes(self) -> int:
                return 10

            def cache_bytes(self) -> int:
                return 10

        probe = ConstantProbe()
        ceiling = 1000  # Very high ceiling
        abort_called: list[str] = []

        def on_abort(reason: str) -> None:
            abort_called.append(reason)

        # Run watchdog with short deadline
        thread = run_watchdog(probe, ceiling, on_abort=on_abort, poll_s=0.01, deadline_s=0.1)
        thread.join(timeout=1.0)

        # Should have called abort due to deadline
        assert len(abort_called) > 0
        assert "deadline" in abort_called[0].lower() or "time" in abort_called[0].lower()
