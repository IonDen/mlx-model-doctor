"""Memory watchdog and flush-safe abort utilities."""

import os
import sys
from collections.abc import Callable
from threading import Thread
from typing import Protocol


class MemoryProbe(Protocol):
    """Protocol for a memory status probe."""

    def active_bytes(self) -> int:
        """Return active memory in bytes."""
        ...

    def cache_bytes(self) -> int:
        """Return cache memory in bytes."""
        ...


def should_abort(active: int, cache: int, ceiling: int) -> bool:
    """Determine if memory usage exceeds ceiling.

    Args:
        active: Active memory in bytes.
        cache: Cache memory in bytes.
        ceiling: Ceiling in bytes.

    Returns:
        True if active + cache >= ceiling, False otherwise.
    """
    return active + cache >= ceiling


def default_ceiling_bytes(device_memory: int) -> int:
    """Calculate default memory ceiling.

    Args:
        device_memory: Total device memory in bytes.

    Returns:
        device_memory - 4 GiB.
    """
    four_gib = 4 * 1024**3
    return device_memory - four_gib


def _do_abort(out_dir: str, reason: str) -> None:
    """Abort unconditionally, writing a marker and flushing.

    This function must call os._exit(3) regardless of whether the marker
    write or flush raises an exception. Each operation is wrapped in its
    own try/except to ensure termination always happens.

    Args:
        out_dir: Directory for the abort marker file.
        reason: Reason for the abort.
    """
    try:
        from pathlib import Path

        Path(out_dir, "parity_worker_abort.txt").write_text(reason, encoding="utf-8")
    except Exception:
        # Abort must not depend on the marker write succeeding
        pass
    try:  # noqa: SIM105
        sys.stdout.flush()
    except Exception:
        # A broken pipe must not block termination
        pass
    os._exit(3)  # pragma: no cover


def run_watchdog(
    probe: MemoryProbe,
    ceiling: int,
    *,
    on_abort: Callable[[str], None],
    poll_s: float = 0.05,
    deadline_s: float | None = None,
    stop: object | None = None,
) -> Thread:
    """Run a memory watchdog in a background thread.

    Args:
        probe: Memory probe implementing MemoryProbe protocol.
        ceiling: Memory ceiling in bytes.
        on_abort: Callback to invoke on abort (receives reason string).
        poll_s: Poll interval in seconds.
        deadline_s: Optional deadline in seconds.
        stop: Optional stop signal (threading.Event or similar).

    Returns:
        The watchdog thread.
    """
    import time

    def _run() -> None:
        start_time = time.time()
        while True:
            # Check memory
            active = probe.active_bytes()
            cache = probe.cache_bytes()
            if should_abort(active, cache, ceiling):
                on_abort(f"memory exceeded: {active + cache} >= {ceiling}")
                return

            # Check deadline
            if deadline_s is not None:
                elapsed = time.time() - start_time
                if elapsed >= deadline_s:
                    on_abort(f"deadline exceeded: {elapsed}s >= {deadline_s}s")
                    return

            # Check stop signal
            if stop is not None and hasattr(stop, "is_set") and stop.is_set():
                return

            time.sleep(poll_s)

    thread = Thread(target=_run, daemon=True)
    thread.start()
    return thread
