from mlx_model_doctor.environment import (
    DependencyStatus,
    detect_venv,
    format_install_hint,
    has_uv_context,
    metadata,
    package_status,
    util,
)


def test_detect_venv_uses_virtual_env_first() -> None:
    env = {"VIRTUAL_ENV": "/repo/.venv"}
    assert detect_venv(prefix="/repo/.venv", base_prefix="/usr", environ=env) == "/repo/.venv"


def test_detect_venv_falls_back_to_prefix_difference() -> None:
    assert detect_venv(prefix="/repo/.venv", base_prefix="/usr", environ={}) == "/repo/.venv"
    assert detect_venv(prefix="/usr", base_prefix="/usr", environ={}) is None


def test_has_uv_context_requires_virtualenv_and_project_file() -> None:
    assert has_uv_context(cwd_files={"pyproject.toml"}, environ={"VIRTUAL_ENV": "/repo/.venv"})
    assert not has_uv_context(cwd_files={"pyproject.toml"}, environ={})
    assert not has_uv_context(cwd_files=set(), environ={"VIRTUAL_ENV": "/repo/.venv"})


def test_has_uv_context_defaults_to_process_environment(monkeypatch) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", "/repo/.venv")

    assert has_uv_context(cwd_files={"uv.lock"})


def test_format_install_hint_contains_executable_and_uv_and_pip() -> None:
    hint = format_install_hint(
        missing_package="mlx_lm",
        extra_name="mlx-lm",
        executable="/repo/.venv/bin/python",
        has_uv_context=True,
    )

    assert "/repo/.venv/bin/python" in hint
    assert 'uv add "mlx-model-doctor[mlx-lm]"' in hint
    assert 'python -m pip install "mlx-model-doctor[mlx-lm]"' in hint


def test_package_status_reports_missing_package_without_importing() -> None:
    status = package_status("definitely_missing_package_for_model_doctor")
    assert status == DependencyStatus(
        name="definitely_missing_package_for_model_doctor",
        version=None,
    )


def test_package_status_normalizes_hyphenated_names(monkeypatch) -> None:
    requested_specs = []
    requested_versions = []

    def find_spec(name: str) -> object:
        requested_specs.append(name)
        return object()

    def version(name: str) -> str:
        requested_versions.append(name)
        return "1.2.3"

    monkeypatch.setattr(util, "find_spec", find_spec)
    monkeypatch.setattr(metadata, "version", version)

    status = package_status("huggingface-hub")

    assert status == DependencyStatus(name="huggingface-hub", version="1.2.3")
    assert requested_specs == ["huggingface_hub"]
    assert requested_versions == ["huggingface-hub"]
