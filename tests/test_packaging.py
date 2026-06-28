"""Package-surface guard: a local build must not leak working-tree state.

Builds the sdist + wheel from the real working tree, offline (``--no-isolation``,
so no backend is pip-installed and no network is touched), with a pretend version
so the build does not depend on git tags. The build's vcs hook rewrites
``src/mlx_model_doctor/_version.py`` in place, so the fixture saves and restores it.
"""

import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_PREFIXES = ("src/", "tests/")
ALLOWED_ROOT_FILES = frozenset(
    {
        "README.md",
        "LICENSE",
        "NOTICE",
        "CHANGELOG.md",
        "ROADMAP.md",
        "EXAMPLES.md",
        "pyproject.toml",
        "PKG-INFO",  # generated into the sdist by the build backend
        ".gitignore",  # hatchling force-includes VCS exclusion files unconditionally
    }
)
FORBIDDEN_SUBSTRINGS = (
    ".claude",
    ".codegraph",
    ".DS_Store",
    "docs/",
    ".coverage",
    ".hypothesis",
    "uv.lock",
)


@pytest.fixture(scope="session")
def built_artifacts(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    out = tmp_path_factory.mktemp("dist")
    version_file = REPO_ROOT / "src" / "mlx_model_doctor" / "_version.py"
    original = version_file.read_bytes() if version_file.exists() else None
    env = {**os.environ, "SETUPTOOLS_SCM_PRETEND_VERSION": "0.0.0"}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(out), str(REPO_ROOT)],
            capture_output=True,
            text=True,
            env=env,
        )
    finally:
        if original is not None:
            version_file.write_bytes(original)
        elif version_file.exists():
            version_file.unlink()  # the build hook created it; leave no trace
    if proc.returncode != 0:
        pytest.fail(f"build failed (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    sdists = list(out.glob("*.tar.gz"))
    wheels = list(out.glob("*.whl"))
    assert len(sdists) == 1, f"expected exactly one sdist, got {sdists}"
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    return sdists[0], wheels[0]


def test_sdist_contains_only_package_surface(built_artifacts: tuple[Path, Path]) -> None:
    sdist, _wheel = built_artifacts
    with tarfile.open(sdist) as tf:
        # sdist members are prefixed with "<name>-<version>/"; strip it.
        members = [name.split("/", 1)[1] for name in tf.getnames() if "/" in name]
    members = [m for m in members if m]

    assert "src/mlx_model_doctor/__init__.py" in members
    assert "pyproject.toml" in members

    for member in members:
        assert member.startswith(ALLOWED_PREFIXES) or member in ALLOWED_ROOT_FILES, (
            f"stray sdist member outside the allowlist: {member}"
        )
    for bad in FORBIDDEN_SUBSTRINGS:
        assert not any(bad in member for member in members), f"forbidden path in sdist: {bad}"


def test_wheel_contains_only_the_package(built_artifacts: tuple[Path, Path]) -> None:
    _sdist, wheel = built_artifacts
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()

    assert "mlx_model_doctor/__init__.py" in names
    assert "mlx_model_doctor/py.typed" in names
    for name in names:
        top_level = name.split("/", 1)[0]
        assert name.startswith("mlx_model_doctor/") or top_level.endswith(".dist-info"), (
            f"stray wheel member outside the package: {name}"
        )


def test_wheel_ships_the_report_schema(built_artifacts: tuple[Path, Path]) -> None:
    _sdist, wheel = built_artifacts
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    assert "mlx_model_doctor/schema/report.v1.schema.json" in names, (
        f"report schema missing from wheel: {names}"
    )


def test_sdist_ships_the_report_schema(built_artifacts: tuple[Path, Path]) -> None:
    sdist, _wheel = built_artifacts
    with tarfile.open(sdist) as tf:
        # sdist members are prefixed with "<name>-<version>/"; strip it.
        members = [name.split("/", 1)[1] for name in tf.getnames() if "/" in name]
    assert "src/mlx_model_doctor/schema/report.v1.schema.json" in members, (
        f"report schema missing from sdist: {members}"
    )


def test_wheel_ships_the_sample_batch_schema(built_artifacts: tuple[Path, Path]) -> None:
    _sdist, wheel = built_artifacts
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    assert "mlx_model_doctor/schema/sample-batch.v1.schema.json" in names, (
        f"sample-batch schema missing from wheel: {names}"
    )


def test_sdist_ships_the_sample_batch_schema(built_artifacts: tuple[Path, Path]) -> None:
    sdist, _wheel = built_artifacts
    with tarfile.open(sdist) as tf:
        # sdist members are prefixed with "<name>-<version>/"; strip it.
        members = [name.split("/", 1)[1] for name in tf.getnames() if "/" in name]
    assert "src/mlx_model_doctor/schema/sample-batch.v1.schema.json" in members, (
        f"sample-batch schema missing from sdist: {members}"
    )
