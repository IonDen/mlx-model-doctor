"""Command-line interface for mlx-model-doctor."""

import argparse
import os
import platform
import sys
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Literal, cast

from mlx_model_doctor.api import (
    ParityOptions,
    check_adapter_parity,
    check_hf_model,
    check_local_model,
)
from mlx_model_doctor.cache import ListingCache
from mlx_model_doctor.compat import LISTING_VISIBLE_SIGNALS
from mlx_model_doctor.context import CheckOptions
from mlx_model_doctor.environment import detect_venv, package_status
from mlx_model_doctor.errors import ModelDoctorError
from mlx_model_doctor.exit_codes import exit_code_for, exit_code_for_error
from mlx_model_doctor.memory import parse_memory
from mlx_model_doctor.parity.exit_codes import parity_exit_code
from mlx_model_doctor.parity.fixtures import DEFAULT_FIXTURE_ID
from mlx_model_doctor.parity.prompts import build_prompts_fixture
from mlx_model_doctor.parity.report import (
    ParityReport,
    render_parity_json,
    render_parity_markdown,
    render_parity_text,
)
from mlx_model_doctor.plugins import BUILTIN_PLUGINS
from mlx_model_doctor.report import (
    DoctorReport,
    github_output_lines,
    render_github,
    render_json,
    render_markdown,
    render_text,
)
from mlx_model_doctor.sampling import (
    DefaultHfModelLister,
    SampleBatchReport,
    render_sample_batch_json,
    render_sample_batch_markdown,
    render_sample_batch_text,
    run_hf_sample,
    sample_batch_exit_code,
)

Command = Callable[[argparse.Namespace], int]
Verbosity = Literal["quiet", "normal", "verbose"]
DEPENDENCIES: tuple[str, ...] = ("huggingface-hub", "safetensors", "mlx", "mlx-lm", "mlx-vlm")


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
                "  mlx-model-doctor check hf mlx-community/Llama-3.2-3B-Instruct-4bit",
                "  mlx-model-doctor sample hf --author mlx-community --limit 5",
                "  mlx-model-doctor parity mlx --base <B> --adapter <A> --fused <F>",
                "  mlx-model-doctor parity mlx --base <B> --adapter <A> --fused <F>"
                " --prompts <examples.json>",
                "",
                "Exit codes:",
                "  0: checks passed or informational command completed",
                "  1: checks found failures under the selected fail policy",
                "  2: tool error, bad target, missing dependency, or zero checks",
                "",
                "parity mlx exit codes:",
                "  0: PASS -- the fused model tracks the adapter-applied reference",
                "  1: a determined defect (tokenizer mismatch, uncovered LoRA target,",
                "     or a confirmed regression/gross-divergence verdict)",
                "  2: a tool/setup error or a runtime cannot-determine outcome",
            )
        )
    )
    return 0


def _cmd_plugins(_args: argparse.Namespace) -> int:
    for name in BUILTIN_PLUGINS:
        print(name)
    return 0


def _cmd_check_local(args: argparse.Namespace) -> int:
    options = _options_from_args(args)
    report = check_local_model(args.path, options=options, plugin_name=args.plugin)
    return _finish_check(report, args)


def _cmd_check_hf(args: argparse.Namespace) -> int:
    options = _options_from_args(args)
    report = check_hf_model(args.repo_id, options=options, plugin_name=args.plugin)
    return _finish_check(report, args)


def _finish_check(report: DoctorReport, args: argparse.Namespace) -> int:
    code = exit_code_for(report, fail_on=cast("Literal['error', 'warn', 'never']", args.fail_on))
    if args.format == "github":
        if args.output is not None:
            raise ModelDoctorError(
                "--output is not supported with --format github"
                " (the report is written to GitHub's annotations, step summary, and step outputs)."
            )
        _emit_github(report, code)
        return code
    rendered = _render_report(report, args.format, quiet=_verbosity(args) == "quiet")
    _emit_report(rendered, args.output)
    return code


def _emit_github(report: DoctorReport, exit_code: int) -> None:
    print(render_github(report))
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        _append_github_file(Path(summary_path), render_markdown(report) + "\n")
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        _append_github_file(
            Path(output_path), github_output_lines(report, exit_code=exit_code) + "\n"
        )


def _append_github_file(path: Path, content: str) -> None:
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(content)
    except OSError as exc:
        raise ModelDoctorError(f"Could not write to {path}: {exc}") from exc


def _cmd_parity_mlx(args: argparse.Namespace) -> int:
    if args.prompts is not None:
        # --prompts wins over --fixture when both are given.
        options = ParityOptions(fixture=build_prompts_fixture(args.base, args.prompts))
    else:
        options = ParityOptions(fixture_id=args.fixture)
    report = check_adapter_parity(
        base=args.base, adapter=args.adapter, fused=args.fused, options=options
    )
    print(_render_parity_report(report, args.format))
    return parity_exit_code(report)


def _render_parity_report(report: ParityReport, output_format: str) -> str:
    if output_format == "json":
        return render_parity_json(report)
    if output_format == "markdown":
        return render_parity_markdown(report)
    return render_parity_text(report)


def _cmd_sample_hf(args: argparse.Namespace) -> int:
    signal_filter = _parse_signal_filter(args.signal_filter)
    lister = _build_sample_lister(args)
    batch = run_hf_sample(
        author=args.author,
        task=args.task,
        limit=args.limit,
        plugin_name=args.plugin,
        max_candidates=args.max_candidates,
        signal_filter=signal_filter,
        lister=lister,
    )
    print(_render_sample_batch(batch, args.format))
    return sample_batch_exit_code(batch)


def _parse_signal_filter(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    signals = tuple(s.strip() for s in raw.split(",") if s.strip())
    if not signals:
        return None
    invalid = [s for s in signals if s not in LISTING_VISIBLE_SIGNALS]
    if invalid:
        raise ModelDoctorError(
            f"Invalid signal filter values: {invalid}. Allowed: {sorted(LISTING_VISIBLE_SIGNALS)}"
        )
    return signals


def _build_sample_lister(args: argparse.Namespace) -> DefaultHfModelLister | None:
    if args.no_cache:
        return None
    cache = ListingCache(ttl_seconds=args.cache_ttl)
    return DefaultHfModelLister(cache=cache)


def _options_from_args(args: argparse.Namespace) -> CheckOptions:
    max_memory_bytes = _parse_optional_memory(args.max_memory)
    context_length = _positive_context_length(args.context_length)
    return CheckOptions(
        max_memory_bytes=max_memory_bytes,
        context_length=context_length,
        include_weights=not args.skip_weights,
        smoke=args.smoke,
        verbosity=_verbosity(args),
    )


def _emit_report(rendered: str, output: str | None) -> None:
    if output is None:
        print(rendered)
        return
    output_path = Path(output)
    try:
        output_path.write_text(rendered, encoding="utf-8")
    except OSError as exc:
        raise ModelDoctorError(f"Could not write report to {output_path}: {exc}") from exc


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


def _render_report(report: DoctorReport, output_format: str, *, quiet: bool = False) -> str:
    if output_format == "json":
        return render_json(report)
    if output_format == "markdown":
        return render_markdown(report)
    return render_text(report, quiet=quiet)


def _render_sample_batch(batch: SampleBatchReport, output_format: str) -> str:
    if output_format == "json":
        return render_sample_batch_json(batch)
    if output_format == "markdown":
        return render_sample_batch_markdown(batch)
    return render_sample_batch_text(batch)


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

    sample = subparsers.add_parser("sample", help="sample live model repositories")
    sample_subparsers = sample.add_subparsers(dest="sample_command", required=True)
    sample_hf = sample_subparsers.add_parser("hf", help="sample Hugging Face model repositories")
    sample_hf.add_argument("--author", default="mlx-community", help="Hugging Face author filter")
    sample_hf.add_argument(
        "--task",
        default=None,
        help="Hugging Face pipeline task filter, mapped to pipeline_tag",
    )
    sample_hf.add_argument(
        "--limit",
        type=int,
        default=10,
        help="number of MLX candidates to check (use --max-candidates to control scan depth)",
    )
    sample_hf.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="maximum models to scan from the Hub before filtering; overrides the default depth",
    )
    sample_hf.add_argument(
        "--signal-filter",
        default=None,
        help=(
            "comma-separated MLX signal names to keep; filters on the highest-priority "
            "signal per candidate (e.g. tag:mlx,library:mlx-lm)"
        ),
    )
    sample_hf.add_argument(
        "--no-cache",
        action="store_true",
        help="bypass the local listing cache and always fetch from the Hub",
    )
    sample_hf.add_argument(
        "--cache-ttl",
        type=int,
        default=3600,
        help="listing cache TTL in seconds (default: 3600)",
    )
    sample_hf.add_argument("--plugin", default="text", help="plugin name to run")
    sample_hf.add_argument(
        "--format",
        choices=("text", "json", "markdown"),
        default="text",
        help="batch report output format",
    )
    sample_hf.set_defaults(func=_cmd_sample_hf)

    check = subparsers.add_parser("check", help="run model checks")
    check_subparsers = check.add_subparsers(dest="check_command", required=True)
    local = check_subparsers.add_parser("local", help="check a local model repository")
    local.add_argument("path", help="path to a local model repository")
    _add_check_options(local)
    local.set_defaults(func=_cmd_check_local)

    hf = check_subparsers.add_parser("hf", help="check a Hugging Face model repository")
    hf.add_argument("repo_id", help="Hugging Face model repository ID")
    _add_check_options(hf)
    hf.set_defaults(func=_cmd_check_hf)

    parity = subparsers.add_parser("parity", help="verify adapter/fused-model parity")
    parity_subparsers = parity.add_subparsers(dest="parity_command", required=True)
    parity_mlx = parity_subparsers.add_parser(
        "mlx", help="verify a fused model preserved a LoRA adapter's behavior"
    )
    parity_mlx.add_argument(
        "--base", required=True, help="base model local path or Hugging Face repo id"
    )
    parity_mlx.add_argument(
        "--adapter", required=True, help="LoRA adapter local path or Hugging Face repo id"
    )
    parity_mlx.add_argument(
        "--fused", required=True, help="fused model local path or Hugging Face repo id"
    )
    parity_mlx.add_argument(
        "--format",
        choices=("text", "json", "markdown"),
        default="text",
        help="report output format",
    )
    parity_mlx.add_argument(
        "--fixture",
        default=DEFAULT_FIXTURE_ID,
        help="pinned token-id fixture id used for the teacher-forced worker loads",
    )
    parity_mlx.add_argument(
        "--prompts",
        default=None,
        help=(
            "JSON file of [{prompt, completion}] examples; builds a fixture bound to "
            "the base model's own tokenizer instead of the built-in default (overrides "
            "--fixture when set). Requires --base to be an existing local directory; "
            "this flag never downloads a Hugging Face repository on its own."
        ),
    )
    parity_mlx.set_defaults(func=_cmd_parity_mlx)
    return parser


def _add_check_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plugin", default="text", help="plugin name to run")
    parser.add_argument(
        "--format",
        choices=("text", "json", "markdown", "github"),
        default="text",
        help="report output format (github emits GitHub Actions annotations)",
    )
    parser.add_argument("--output", help="write rendered report to a file")
    parser.add_argument("--max-memory", help="memory budget such as 32gb or 512mb")
    parser.add_argument(
        "--context-length",
        type=int,
        default=4096,
        help="context length used by memory estimates",
    )
    parser.add_argument(
        "--fail-on",
        choices=("error", "warn", "never"),
        default="error",
        help="exit-code strictness",
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="skip safetensors tensor-header checks (faster, config-only)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run optional runtime smoke checks",
    )
    parser.add_argument("--quiet", action="store_true", help="reduce diagnostic verbosity")
    parser.add_argument("--verbose", action="store_true", help="include verbose diagnostics")


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
