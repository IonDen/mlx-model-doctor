def test_public_package_exports_core_names() -> None:
    import mlx_model_doctor
    from mlx_model_doctor import (
        CheckResult,
        DependencyError,
        DoctorReport,
        ModelDoctorError,
        TargetError,
        check_hf_model,
        check_local_model,
    )

    assert "CheckResult" in mlx_model_doctor.__all__
    assert "DoctorReport" in mlx_model_doctor.__all__
    assert "check_local_model" in mlx_model_doctor.__all__
    assert "check_hf_model" in mlx_model_doctor.__all__
    assert mlx_model_doctor.CheckResult is CheckResult
    assert mlx_model_doctor.DependencyError is DependencyError
    assert mlx_model_doctor.DoctorReport is DoctorReport
    assert mlx_model_doctor.ModelDoctorError is ModelDoctorError
    assert mlx_model_doctor.TargetError is TargetError
    assert mlx_model_doctor.check_local_model is check_local_model
    assert mlx_model_doctor.check_hf_model is check_hf_model
