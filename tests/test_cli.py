import argparse
import json
from pathlib import Path

import pytest

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


def test_check_without_leaf_subcommand_is_argparse_error(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["check"])

    assert exc_info.value.code == 2
    assert "required" in capsys.readouterr().err


def test_check_local_returns_zero_for_valid_fixture(tmp_path: Path, capsys) -> None:
    model = write_local_model(tmp_path)

    code = cli.main(["check", "local", str(model)])
    output = capsys.readouterr().out

    assert code == 0
    assert "MLX Model Doctor" in output


def test_check_local_missing_config_fails_by_default_and_can_be_non_strict(
    tmp_path: Path,
    capsys,
) -> None:
    model = tmp_path / "missing-config"
    model.mkdir()

    assert cli.main(["check", "local", str(model)]) == 1
    capsys.readouterr()

    assert cli.main(["check", "local", str(model), "--fail-on", "never"]) == 0


def test_check_local_fail_on_warn_fails_for_warn_only_model(tmp_path: Path, capsys) -> None:
    model = write_local_model(tmp_path)

    default_code = cli.main(["check", "local", str(model)])
    capsys.readouterr()
    warn_code = cli.main(["check", "local", str(model), "--fail-on", "warn"])

    assert default_code == 0
    assert warn_code == 1


def test_check_local_json_format_emits_schema_and_target(tmp_path: Path, capsys) -> None:
    model = write_local_model(tmp_path)

    code = cli.main(["check", "local", str(model), "--format", "json"])
    data = json.loads(capsys.readouterr().out)

    assert code == 0
    assert data["schema_version"] == "1.0"
    assert data["target"] == str(model.resolve())
    assert data["plugin"] == "text"


def test_check_local_markdown_output_writes_file_without_stdout(
    tmp_path: Path,
    capsys,
) -> None:
    model = write_local_model(tmp_path)
    output_path = tmp_path / "report.md"

    code = cli.main(
        ["check", "local", str(model), "--format", "markdown", "--output", str(output_path)]
    )
    captured = capsys.readouterr()

    assert code == 0
    assert captured.out == ""
    assert output_path.read_text(encoding="utf-8").startswith("# MLX Model Doctor")


def test_check_local_output_directory_returns_tool_error_without_traceback(
    tmp_path: Path,
    capsys,
) -> None:
    model = write_local_model(tmp_path)
    output_path = tmp_path / "reports"
    output_path.mkdir()

    code = cli.main(["check", "local", str(model), "--output", str(output_path)])
    captured = capsys.readouterr()

    assert code == 2
    assert "Could not write report" in captured.err
    assert "Traceback" not in captured.err


def test_check_local_memory_options_are_reflected_in_json_report(
    tmp_path: Path,
    capsys,
) -> None:
    model = write_local_model(tmp_path)

    code = cli.main(
        [
            "check",
            "local",
            str(model),
            "--format",
            "json",
            "--max-memory",
            "1b",
            "--context-length",
            "8",
        ]
    )
    data = json.loads(capsys.readouterr().out)
    memory = next(result for result in data["results"] if result["check_id"] == "text/memory.estimate")

    assert code == 0
    assert memory["status"] == "warn"
    assert memory["severity"] == "high"
    assert memory["details"]["max_memory_bytes"] == 1
    assert memory["details"]["context_length"] == 8


def test_check_local_invalid_max_memory_returns_tool_error(tmp_path: Path, capsys) -> None:
    model = write_local_model(tmp_path)

    code = cli.main(["check", "local", str(model), "--max-memory", "ten"])
    captured = capsys.readouterr()

    assert code == 2
    assert "Invalid memory value" in captured.err


def test_check_local_non_positive_context_length_returns_tool_error(
    tmp_path: Path,
    capsys,
) -> None:
    model = write_local_model(tmp_path)

    code = cli.main(["check", "local", str(model), "--context-length", "0"])
    captured = capsys.readouterr()

    assert code == 2
    assert "context-length must be positive" in captured.err


def write_local_model(root: Path) -> Path:
    model = root / "model"
    model.mkdir()
    config = {
        "model_type": "llama",
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": 256,
        "intermediate_size": 512,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "quantization": {"bits": 4, "group_size": 64},
    }
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    return model
