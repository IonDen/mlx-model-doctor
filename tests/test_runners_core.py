from dataclasses import dataclass
from pathlib import Path

import pytest

from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.runners.core import run_checks
from tests.fakes import FakeTarget, check_options, use_fake_home


@dataclass(frozen=True, slots=True)
class BadIdCrashingCheck:
    check_id: str = "bogus"  # not plugin/name shaped
    title: str = "Bad id crash"

    def run(self, ctx: CheckContext) -> object:
        raise RuntimeError("boom")


@dataclass(frozen=True, slots=True)
class NonStringIdCrashingCheck:
    check_id: int = 123  # a non-string check_id, e.g. from a malformed plugin
    title: str = "Non-string id crash"

    def run(self, ctx: CheckContext) -> object:
        raise RuntimeError("boom")


def test_run_checks_isolates_crash_even_with_malformed_check_id() -> None:
    ctx = CheckContext(target=FakeTarget(files={}), options=check_options())

    results = run_checks(ctx, [BadIdCrashingCheck()])

    assert len(results) == 1
    assert results[0].status == "fail"
    assert "/" in results[0].check_id  # sanitized to a valid plugin/name id
    assert "boom" in results[0].message


def test_run_checks_isolates_crash_with_non_string_check_id() -> None:
    ctx = CheckContext(target=FakeTarget(files={}), options=check_options())

    results = run_checks(ctx, [NonStringIdCrashingCheck()])

    assert len(results) == 1
    assert results[0].status == "fail"
    assert "/" in results[0].check_id  # sanitized to a valid plugin/name id
    assert "boom" in results[0].message


@dataclass(frozen=True, slots=True)
class PathCrashingCheck:
    error: str
    check_id: str = "text/path-crash"
    title: str = "Crash naming local paths"

    def run(self, ctx: CheckContext) -> object:
        raise RuntimeError(self.error)


def test_run_checks_crash_message_keeps_no_local_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: the crash text is copied raw, so the message carries the home; or the
    # target's model folder is not passed and stays "~/my models/m".
    home = tmp_path / "Users" / "John Smith"
    model = home / "my models" / "m"
    use_fake_home(monkeypatch, home=str(home))
    ctx = CheckContext(target=FakeTarget(files={}, name=str(model)), options=check_options())

    results = run_checks(ctx, [PathCrashingCheck(error=f"bad {home}/x/y.py; cannot open {model}")])

    assert results[0].message == "check crashed: bad y.py; cannot open m"
