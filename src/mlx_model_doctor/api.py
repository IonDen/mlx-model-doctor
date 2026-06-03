"""Public Python API."""

from pathlib import Path

from mlx_model_doctor.context import CheckContext, CheckOptions
from mlx_model_doctor.plugins import get_plugin
from mlx_model_doctor.report import DoctorReport
from mlx_model_doctor.runners.static import run_static_checks
from mlx_model_doctor.targets import LocalTarget


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
    results = run_static_checks(ctx, plugin.static_checks())
    return DoctorReport(
        target=target.name,
        source=target.source,
        plugin=plugin.name,
        results=results,
    )


def check_hf_model(repo_id: str) -> DoctorReport:
    """Check a Hugging Face model repository."""
    raise NotImplementedError("check_hf_model requires the Hugging Face target")


def _default_options() -> CheckOptions:
    return CheckOptions(
        max_memory_bytes=None,
        context_length=4096,
        include_weights=False,
        smoke=False,
        verbosity="normal",
    )
