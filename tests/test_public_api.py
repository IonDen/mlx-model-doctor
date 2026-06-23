"""The frozen public API surface must be importable from the package root."""

import mlx_model_doctor as mmd

FROZEN_SURFACE = frozenset(
    {
        "check_local_model",
        "check_hf_model",
        "CheckOptions",
        "DoctorReport",
        "CheckResult",
        "render_json",
        "render_text",
        "render_markdown",
        "render_github",
        "exit_code_for",
        "FailOn",
        "ModelDoctorError",
        "TargetError",
        "DependencyError",
        "MemorySafetyError",
    }
)


def test_frozen_surface_is_exported() -> None:
    # Superset (not ==) is intentional: the stable surface may grow; accidental
    # exports of internals are guarded by test_internal_helpers_are_not_exported.
    assert set(mmd.__all__) >= FROZEN_SURFACE


def test_frozen_surface_is_importable() -> None:
    for name in FROZEN_SURFACE:
        assert getattr(mmd, name) is not None


def test_internal_helpers_are_not_exported() -> None:
    # CI-internal / unstable extension layer must NOT be in the frozen surface.
    for name in ("github_output_lines", "exit_code_for_error", "CheckContext", "get_plugin"):
        assert name not in mmd.__all__
