import argparse
import json
from pathlib import Path

import pytest

import mlx_model_doctor.targets as targets
from mlx_model_doctor import cli
from mlx_model_doctor.context import CheckOptions
from mlx_model_doctor.errors import TargetError
from mlx_model_doctor.report import CheckResult, DoctorReport
from mlx_model_doctor.targets import HfSiblingProtocol


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


def test_parser_accepts_check_hf_and_no_bare_hf_command(capsys) -> None:
    args = cli.build_parser().parse_args(["check", "hf", "org/model"])

    assert args.check_command == "hf"
    assert args.repo_id == "org/model"

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["hf", "org/model"])

    assert exc_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


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


def test_check_hf_json_format_dispatches_options_and_plugin(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_check_hf_model(
        repo_id: str,
        *,
        options: CheckOptions | None = None,
        plugin_name: str = "text",
    ) -> DoctorReport:
        captured["repo_id"] = repo_id
        captured["options"] = options
        captured["plugin_name"] = plugin_name
        return hf_report(
            results=(
                CheckResult(
                    check_id="text/memory.estimate",
                    title="Memory estimate",
                    status="warn",
                    severity="low",
                    message="memory warning",
                ),
            )
        )

    monkeypatch.setattr(cli, "check_hf_model", fake_check_hf_model)

    code = cli.main(
        [
            "check",
            "hf",
            "org/model",
            "--format",
            "json",
            "--plugin",
            "text",
            "--max-memory",
            "1b",
            "--context-length",
            "8",
            "--include-weights",
            "--quiet",
        ]
    )
    data = json.loads(capsys.readouterr().out)
    options = captured["options"]

    assert code == 0
    assert data["source"] == "hf"
    assert data["target"] == "org/model"
    assert captured["repo_id"] == "org/model"
    assert captured["plugin_name"] == "text"
    assert isinstance(options, CheckOptions)
    assert options.max_memory_bytes == 1
    assert options.context_length == 8
    assert options.include_weights is True
    assert options.verbosity == "quiet"


def test_check_hf_markdown_output_and_fail_on_warn(monkeypatch, tmp_path: Path, capsys) -> None:
    def fake_check_hf_model(
        repo_id: str,
        *,
        options: CheckOptions | None = None,
        plugin_name: str = "text",
    ) -> DoctorReport:
        return hf_report(
            results=(
                CheckResult(
                    check_id="text/tokenizer.files",
                    title="Tokenizer files",
                    status="warn",
                    severity="medium",
                    message="tokenizer warning",
                ),
            )
        )

    monkeypatch.setattr(cli, "check_hf_model", fake_check_hf_model)
    output_path = tmp_path / "hf-report.md"

    code = cli.main(
        [
            "check",
            "hf",
            "org/model",
            "--format",
            "markdown",
            "--output",
            str(output_path),
            "--fail-on",
            "warn",
        ]
    )
    captured = capsys.readouterr()

    assert code == 1
    assert captured.out == ""
    assert output_path.read_text(encoding="utf-8").startswith("# MLX Model Doctor")


def test_check_hf_target_error_returns_exit_two_without_traceback(monkeypatch, capsys) -> None:
    def fake_check_hf_model(
        repo_id: str,
        *,
        options: CheckOptions | None = None,
        plugin_name: str = "text",
    ) -> DoctorReport:
        raise TargetError("private Hugging Face repo", target=repo_id, source="hf")

    monkeypatch.setattr(cli, "check_hf_model", fake_check_hf_model)

    code = cli.main(["check", "hf", "org/private"])
    captured = capsys.readouterr()

    assert code == 2
    assert "private Hugging Face repo" in captured.err
    assert "Traceback" not in captured.err


def test_check_hf_download_error_returns_exit_two_without_traceback(monkeypatch, capsys) -> None:
    monkeypatch.setattr(targets, "DefaultHfHub", DownloadErrorHub)

    code = cli.main(["check", "hf", "org/model"])
    captured = capsys.readouterr()

    assert code == 2
    assert "Could not read Hugging Face model file config.json" in captured.err
    assert "network unavailable" in captured.err
    assert "Traceback" not in captured.err


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


def hf_report(*, results: tuple[CheckResult, ...]) -> DoctorReport:
    return DoctorReport(target="org/model", source="hf", plugin="text", results=results)


class FakeHfSibling:
    def __init__(self, *, rfilename: str, size: int | None) -> None:
        self.rfilename = rfilename
        self.size = size


class FakeHfModelInfo:
    def __init__(self, *, siblings: tuple[HfSiblingProtocol, ...]) -> None:
        self.siblings = siblings


class DownloadErrorHub:
    def model_info(self, repo_id: str, *, files_metadata: bool) -> FakeHfModelInfo:
        return FakeHfModelInfo(
            siblings=(
                FakeHfSibling(rfilename="config.json", size=2),
                FakeHfSibling(rfilename="tokenizer.json", size=2),
            )
        )

    def download_bytes(self, repo_id: str, filename: str) -> bytes:
        if filename == "config.json":
            raise RuntimeError("network unavailable")
        return b"{}"
