from mlx_model_doctor.checks.files import RequiredConfigCheck
from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.errors import TargetError
from tests.fakes import FakeTarget, check_options, context_for_files


def test_required_config_check_fails_when_config_missing() -> None:
    result = RequiredConfigCheck().run(context_for_files({}))

    assert result.check_id == "text/files.required"
    assert result.status == "fail"
    assert result.severity == "high"
    assert "config.json" in result.message
    assert result.remediation is not None


def test_required_config_check_passes_when_config_exists() -> None:
    result = RequiredConfigCheck().run(context_for_files({"config.json": b"{}"}))

    assert result.check_id == "text/files.required"
    assert result.status == "pass"
    assert result.severity == "info"
    assert "config.json" in result.message


def test_required_config_check_fails_cleanly_on_target_error() -> None:
    result = RequiredConfigCheck().run(
        CheckContext(target=ExistsErrorTarget(files={}), options=check_options())
    )

    assert result.status == "fail"
    assert result.severity == "high"
    assert "Could not inspect" in result.message
    assert result.remediation is not None


class ExistsErrorTarget(FakeTarget):
    def exists(self, path: str) -> bool:
        raise TargetError("exists failed", target=path, source=self.source)
