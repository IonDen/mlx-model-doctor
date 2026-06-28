def test_public_package_exports_core_names() -> None:
    import mlx_model_doctor

    # `X is X` after `from pkg import X` is a tautology (both sides are the same
    # attribute access). Assert instead that each core name is BOTH declared in the
    # public surface (__all__) AND actually present on the module: this catches a
    # name dropped from __all__ and an __all__ entry that no longer resolves.
    for name in (
        "CheckResult",
        "DependencyError",
        "DoctorReport",
        "ModelDoctorError",
        "TargetError",
        "check_local_model",
        "check_hf_model",
    ):
        assert name in mlx_model_doctor.__all__, f"{name} missing from __all__"
        assert hasattr(mlx_model_doctor, name), f"{name} declared in __all__ but not exported"
