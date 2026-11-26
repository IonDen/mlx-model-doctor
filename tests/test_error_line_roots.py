"""The ``Error:`` line shows the home as ``~`` and keeps the rest of the path.

A missing optional dependency names the Python executable, and a report or GitHub
file that cannot be written names its path. Under the home that path starts with
``~``; outside it the full path stays, never only its last part. ``--verbose``
changes neither the ``Error:`` line nor the raw traceback in ``details``.
"""

import sys
from dataclasses import replace
from pathlib import Path

import pytest

import mlx_model_doctor.checks.smoke as smoke_module
from mlx_model_doctor import cli
from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.memory import GIB
from mlx_model_doctor.redaction import redact_roots
from mlx_model_doctor.report import CheckResult
from mlx_model_doctor.runners.core import run_checks
from tests.fakes import FakeTarget, check_options, use_fake_home
from tests.parity_fakes import write_parity_model_repo


def _real_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "Users" / "you"
    home.mkdir(parents=True)
    use_fake_home(monkeypatch, home=str(home))
    return home


class FakeMx:
    def device_info(self) -> dict[str, object]:
        return {"max_recommended_working_set_size": 25 * GIB}

    def set_wired_limit(self, value: int) -> None:
        pass

    def set_memory_limit(self, value: int) -> None:
        pass

    def reset_peak_memory(self) -> None:
        pass

    def get_peak_memory(self) -> int:
        return 1


def _install_mlx_without_mlx_lm(monkeypatch: pytest.MonkeyPatch) -> None:
    mx = FakeMx()

    def import_module(name: str) -> object:
        if name == "mlx.core":
            return mx
        raise ImportError(name)

    monkeypatch.setattr(smoke_module.importlib, "import_module", import_module)


def _missing_mlx_lm_error(
    executable: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra_args: tuple[str, ...] = (),
) -> tuple[int, str]:
    home = _real_home(tmp_path, monkeypatch)
    model = write_parity_model_repo(home / "models" / "tiny-model")
    _install_mlx_without_mlx_lm(monkeypatch)
    monkeypatch.setattr(sys, "executable", executable)
    argv = ["check", "local", str(model), "--smoke", "--max-memory", "8gb", *extra_args]
    code = cli.main(argv)
    return code, capsys.readouterr().err


def test_executable_under_home_starts_with_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    executable = str(tmp_path / "Users" / "you" / "proj" / ".venv" / "bin" / "python")

    code, err = _missing_mlx_lm_error(executable, tmp_path, monkeypatch, capsys)

    assert code == 2
    assert err.splitlines()[0] == "Error: mlx_lm is missing for executable: ~/proj/.venv/bin/python"


def test_verbose_executable_under_home_still_starts_with_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Bug caught: under --verbose the Error line prints the raw exception text,
    # so the full home path appears instead of ~.
    home = tmp_path / "Users" / "you"
    executable = str(home / "proj" / ".venv" / "bin" / "python")

    code, err = _missing_mlx_lm_error(
        executable, tmp_path, monkeypatch, capsys, extra_args=("--verbose",)
    )

    assert code == 2
    assert err.splitlines()[0] == "Error: mlx_lm is missing for executable: ~/proj/.venv/bin/python"
    assert str(home) not in err


def test_executable_outside_home_keeps_full_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, err = _missing_mlx_lm_error("/opt/py/bin/python", tmp_path, monkeypatch, capsys)

    assert code == 2
    assert err.splitlines()[0] == "Error: mlx_lm is missing for executable: /opt/py/bin/python"


def test_executable_in_spaced_volume_keeps_full_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    executable = "/Volumes/Alice SSD/venv/bin/python"

    code, err = _missing_mlx_lm_error(executable, tmp_path, monkeypatch, capsys)

    assert code == 2
    assert err.splitlines()[0].endswith("executable: /Volumes/Alice SSD/venv/bin/python")


def test_output_folder_under_home_starts_with_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = _real_home(tmp_path, monkeypatch)
    model = write_parity_model_repo(home / "models" / "tiny-model")
    (home / "reports").mkdir()

    code = cli.main(["check", "local", str(model), "--output", str(home / "reports")])
    err = capsys.readouterr().err

    assert code == 2
    assert err.startswith("Error: Could not write report to ~/reports: ")
    assert "'~/reports'" in err
    assert str(home) not in err


def test_output_folder_outside_home_keeps_full_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Bug caught: the Error line runs the full path redaction instead of the
    # roots-only one, so an output path outside the home loses its leading folders.
    home = _real_home(tmp_path, monkeypatch)
    model = write_parity_model_repo(home / "models" / "tiny-model")
    output = tmp_path / "Volumes" / "Alice SSD" / "reports"
    output.mkdir(parents=True)

    code = cli.main(["check", "local", str(model), "--output", str(output)])
    err = capsys.readouterr().err

    assert code == 2
    assert err.startswith(f"Error: Could not write report to {output}: ")


def test_github_summary_folder_under_home_starts_with_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = _real_home(tmp_path, monkeypatch)
    model = write_parity_model_repo(home / "models" / "tiny-model")
    (home / "summary-dir").mkdir()
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(home / "summary-dir"))
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

    code = cli.main(["check", "local", str(model), "--format", "github"])
    err = capsys.readouterr().err

    assert code == 2
    assert err.startswith("Error: Could not write to ~/summary-dir: ")
    assert str(home) not in err


class _CrashingCheck:
    check_id = "text/crashing"
    title = "Crashing"

    def run(self, _ctx: CheckContext) -> CheckResult:
        raise RuntimeError("boom at /Users/you/x/a.py")


def test_verbose_crash_message_is_cut_and_traceback_stays_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_home(monkeypatch, home="/Users/you")
    ctx = CheckContext(
        target=FakeTarget(files={}),
        options=replace(check_options(), verbosity="verbose"),
    )

    [result] = run_checks(ctx, (_CrashingCheck(),))

    assert result.message == "check crashed: boom at a.py"
    assert "/Users/you/x/a.py" in str(result.details["traceback"])


def test_verbose_leaves_error_line_unchanged(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Bug caught: a verbose branch in the API re-raises with the raw target.
    use_fake_home(monkeypatch, home="/Users/you")

    code = cli.main(["check", "local", "/Users/you/models/typo", "--verbose"])

    assert code == 2
    assert capsys.readouterr().err == "Error: Local target must be an existing directory: typo\n"


def test_folder_beside_the_home_keeps_its_whole_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # Bug caught: the home and the "+old" run of a sibling folder are cut to "/", which
    # prints a root path that does not exist ("/venv/bin/python").
    use_fake_home(monkeypatch, home="/Users/you")
    text = "mlx_lm is missing for executable: /Users/you+old/venv/bin/python"

    assert redact_roots(text) == text


def test_folder_beside_the_hub_cache_falls_back_to_the_hf_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_home(monkeypatch, home="/Users/you")

    assert redact_roots("/Users/you/.cache/huggingface/hub#2/out.md") == "$HF_HOME/hub#2/out.md"
