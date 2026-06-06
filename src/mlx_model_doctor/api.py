"""Public Python API."""

from pathlib import Path

from mlx_model_doctor.context import CheckContext, CheckOptions
from mlx_model_doctor.plugins import get_plugin
from mlx_model_doctor.report import DoctorReport
from mlx_model_doctor.runners.smoke import run_smoke_checks
from mlx_model_doctor.runners.static import run_static_checks
from mlx_model_doctor.targets import HfHubProtocol, HfTarget, LocalTarget


def check_local_model(
    path: str | Path,
    *,
    options: CheckOptions | None = None,
    plugin_name: str = "text",
) -> DoctorReport:
    """Check a local model repository."""
    target = LocalTarget(path)
    plugin = get_plugin(plugin_name)
    ctx = CheckContext(
        target=target,
        options=options if options is not None else _default_options(),
    )
    static_results = run_static_checks(ctx, plugin.static_checks())
    weight_checks = plugin.weight_checks() if ctx.options.include_weights else ()
    weight_results = run_static_checks(ctx, weight_checks)
    smoke_checks = plugin.smoke_checks() if ctx.options.smoke else ()
    results = [
        *static_results,
        *weight_results,
        *run_smoke_checks(ctx, smoke_checks, static_results),
    ]
    return DoctorReport(
        target=target.name,
        source=target.source,
        plugin=plugin.name,
        results=results,
    )


def check_hf_model(
    repo_id: str,
    *,
    options: CheckOptions | None = None,
    plugin_name: str = "text",
    hub: HfHubProtocol | None = None,
) -> DoctorReport:
    """Check a Hugging Face model repository."""
    target = HfTarget(repo_id, hub=hub)
    plugin = get_plugin(plugin_name)
    ctx = CheckContext(
        target=target,
        options=options if options is not None else _default_options(),
    )
    static_results = run_static_checks(ctx, plugin.static_checks())
    weight_checks = plugin.weight_checks() if ctx.options.include_weights else ()
    weight_results = run_static_checks(ctx, weight_checks)
    smoke_checks = plugin.smoke_checks() if ctx.options.smoke else ()
    results = [
        *static_results,
        *weight_results,
        *run_smoke_checks(ctx, smoke_checks, static_results),
    ]
    return DoctorReport(
        target=target.name,
        source=target.source,
        plugin=plugin.name,
        results=results,
    )


def _default_options() -> CheckOptions:
    return CheckOptions(
        max_memory_bytes=None,
        context_length=4096,
        include_weights=True,
        smoke=False,
        verbosity="normal",
    )
