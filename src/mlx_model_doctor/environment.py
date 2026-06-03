"""Environment and dependency inspection helpers."""

import os
from collections.abc import Container, Mapping
from dataclasses import dataclass
from importlib import metadata, util


@dataclass(frozen=True, slots=True)
class DependencyStatus:
    """Installed status for a Python package."""

    name: str
    version: str | None

    @property
    def installed(self) -> bool:
        """Return whether the dependency is installed."""
        return self.version is not None


def detect_venv(*, prefix: str, base_prefix: str, environ: Mapping[str, str]) -> str | None:
    """Return the active virtual environment path, if one is detectable."""
    virtual_env = environ.get("VIRTUAL_ENV")
    if virtual_env:
        return virtual_env
    if prefix != base_prefix:
        return prefix
    return None


def package_status(name: str) -> DependencyStatus:
    """Return installed package status without importing the package."""
    import_name = name.replace("-", "_")
    if util.find_spec(import_name) is None:
        return DependencyStatus(name=name, version=None)

    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        version = None
    return DependencyStatus(name=name, version=version)


def has_uv_context(
    *,
    cwd_files: Container[str],
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether the current context looks like an active uv project."""
    current_environ = os.environ if environ is None else environ
    return bool(current_environ.get("VIRTUAL_ENV")) and (
        "uv.lock" in cwd_files or "pyproject.toml" in cwd_files
    )


def format_install_hint(
    *,
    missing_package: str,
    extra_name: str,
    executable: str,
    has_uv_context: bool,
) -> str:
    """Return a user-facing installation hint for an optional dependency."""
    extra = f"mlx-model-doctor[{extra_name}]"
    uv_command = f'uv add "{extra}"'
    pip_command = f'python -m pip install "{extra}"'
    commands = [uv_command, pip_command] if has_uv_context else [pip_command, uv_command]
    return "\n".join(
        (
            f"{missing_package} is missing for executable: {executable}",
            "Install it with:",
            f"  {commands[0]}",
            f"  {commands[1]}",
        )
    )
