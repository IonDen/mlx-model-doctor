"""Command-line interface for mlx-model-doctor."""

import argparse
import os
import platform
import sys
from collections.abc import Callable
from importlib import metadata
from typing import cast

from mlx_model_doctor.environment import detect_venv, package_status
from mlx_model_doctor.errors import ModelDoctorError
from mlx_model_doctor.exit_codes import exit_code_for_error

Command = Callable[[argparse.Namespace], int]
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
