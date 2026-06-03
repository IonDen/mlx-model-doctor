import argparse

from mlx_model_doctor import cli
from mlx_model_doctor.errors import TargetError


def test_version_command_prints_executable(capsys) -> None:
    code = cli.main(["version"])
    output = capsys.readouterr().out

    assert code == 0
    assert "mlx-model-doctor" in output
    assert "Executable:" in output


def test_version_command_reports_active_virtualenv(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli.os, "environ", {"VIRTUAL_ENV": "/repo/.venv"})

    code = cli.main(["version"])
    output = capsys.readouterr().out

    assert code == 0
    assert "Virtualenv: /repo/.venv" in output


def test_man_command_prints_exit_codes(capsys) -> None:
    code = cli.main(["man"])
    output = capsys.readouterr().out

    assert code == 0
    assert "Exit codes" in output
    assert "check local" in output


def test_plugins_command_lists_text_plugin(capsys) -> None:
    code = cli.main(["plugins"])
    output = capsys.readouterr().out

    assert code == 0
    assert "text" in output


def test_main_maps_model_doctor_errors_to_exit_code_two(monkeypatch, capsys) -> None:
    parser = cli.build_parser()
    subparsers_action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    boom = subparsers_action.add_parser("boom")

    def raise_target_error(_args: argparse.Namespace) -> int:
        raise TargetError("bad target", target="missing", source="local")

    boom.set_defaults(func=raise_target_error)
    monkeypatch.setattr(cli, "build_parser", lambda: parser)

    assert cli.main(["boom"]) == 2
    assert "bad target" in capsys.readouterr().err
