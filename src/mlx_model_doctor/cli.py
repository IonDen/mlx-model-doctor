"""Command-line interface for mlx-model-doctor."""

import argparse
import os
import platform
import sys
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Literal, cast

from mlx_model_doctor.api import check_local_model
from mlx_model_doctor.context import CheckOptions
from mlx_model_doctor.environment import detect_venv, package_status
from mlx_model_doctor.errors import ModelDoctorError
from mlx_model_doctor.exit_codes import exit_code_for, exit_code_for_error
from mlx_model_doctor.memory import parse_memory
from mlx_model_doctor.report import DoctorReport, render_json, render_markdown, render_text

Command = Callable[[argparse.Namespace], int]
Verbosity = Literal["quiet", "normal", "verbose"]
DEPENDENCIES: tuple[str, ...] = ("huggingface-hub", "safetensors", "mlx", "mlx-lm")


def _package_version() -> str:
    try:
        return metadata.version("mlx-model-doctor")
    except metadata.PackageNotFoundError:
        return "unknown"


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"mlx-model-doctor {_package_version()}")
    print(f"Python: {platform.python_version()}")
    print(f"Executable: {sys.executable}")
    virtualenv = detect_venv(prefix=sys.prefix, base_prefix=sys.base_prefix, environ=os.environ)
    print(f"Virtualenv: {virtualenv or 'none'}")
    print("Dependencies:")
    for dependency in DEPENDENCIES:
        status = package_status(dependency)
        version = status.version if status.installed else "not installed"
        print(f"  {status.name}: {version}")
    return 0


def _cmd_man(_args: argparse.Namespace) -> int:
    print(
        "\n".join(
            (
                "mlx-model-doctor manual",
                "",
                "Examples:",
                "  mlx-model-doctor version",
                "  mlx-model-doctor plugins",
                "  mlx-model-doctor check local ./model",
                "",
                "Exit codes:",
                "  0: checks passed or informational command completed",
                "  1: checks found failures under the selected fail policy",
                "  2: tool error, bad target, missing dependency, or zero checks",
            )
        )
    )
    return 0


def _cmd_plugins(_args: argparse.Namespace) -> int:
    print("text")
    return 0


def _cmd_check_local(args: argparse.Namespace) -> int:
    max_memory_bytes = _parse_optional_memory(args.max_memory)
    context_length = _positive_context_length(args.context_length)
    options = CheckOptions(
        max_memory_bytes=max_memory_bytes,
        context_length=context_length,
        include_weights=args.include_weights,
        smoke=False,
        verbosity=_verbosity(args),
    )
    report = check_local_model(args.path, options=options, plugin_name=args.plugin)
    rendered = _render_report(report, args.format)
    if args.output is None:
        print(rendered)
    else:
        output_path = Path(args.output)
        try:
            output_path.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            raise ModelDoctorError(f"Could not write report to {output_path}: {exc}") from exc
    return exit_code_for(report, fail_on=cast("Literal['error', 'warn', 'never']", args.fail_on))


def _parse_optional_memory(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return parse_memory(value)
    except ValueError as exc:
        raise ModelDoctorError(str(exc)) from exc


def _positive_context_length(value: int) -> int:
    if value <= 0:
        raise ModelDoctorError("context-length must be positive")
    return value


def _verbosity(args: argparse.Namespace) -> Verbosity:
    if args.verbose:
        return "verbose"
    if args.quiet:
        return "quiet"
    return "normal"


def _render_report(report: DoctorReport, output_format: str) -> str:
    if output_format == "json":
        return render_json(report)
    if output_format == "markdown":
        return render_markdown(report)
    return render_text(report)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="mlx-model-doctor",
        description="Validate MLX model repositories before loading them.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        dest="show_version",
        help="print environment-aware version information",
    )

    subparsers = parser.add_subparsers(dest="command")
    version = subparsers.add_parser("version", help="print environment-aware version information")
    version.set_defaults(func=_cmd_version)

    manual = subparsers.add_parser("man", help="print command examples and exit codes")
    manual.set_defaults(func=_cmd_man)

    plugins = subparsers.add_parser("plugins", help="list registered plugins")
    plugins.set_defaults(func=_cmd_plugins)

    check = subparsers.add_parser("check", help="run model checks")
    check_subparsers = check.add_subparsers(dest="check_command", required=True)
    local = check_subparsers.add_parser("local", help="check a local model repository")
    local.add_argument("path", help="path to a local model repository")
    local.add_argument("--plugin", default="text", help="plugin name to run")
    local.add_argument(
        "--format",
        choices=("text", "json", "markdown"),
        default="text",
        help="report output format",
    )
    local.add_argument("--output", help="write rendered report to a file")
    local.add_argument("--max-memory", help="memory budget such as 32gb or 512mb")
    local.add_argument(
        "--context-length",
        type=int,
        default=4096,
        help="context length used by memory estimates",
    )
    local.add_argument(
        "--fail-on",
        choices=("error", "warn", "never"),
        default="error",
        help="exit-code strictness",
    )
    local.add_argument(
        "--include-weights",
        action="store_true",
        help="include weight content checks when available",
    )
    local.add_argument("--quiet", action="store_true", help="reduce diagnostic verbosity")
    local.add_argument("--verbose", action="store_true", help="include verbose diagnostics")
    local.set_defaults(func=_cmd_check_local)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "show_version", False):
        return _cmd_version(args)

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0

    try:
        return cast("Command", func)(args)
    except ModelDoctorError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return exit_code_for_error(exc)
