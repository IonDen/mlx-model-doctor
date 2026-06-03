"""Package-rooted errors."""


class ModelDoctorError(Exception):
    """Base error for tool-level model doctor failures."""


class TargetError(ModelDoctorError, ValueError):
    """A target could not be resolved or read."""

    def __init__(self, message: str, *, target: str, source: str) -> None:
        """Initialize a target resolution error."""
        super().__init__(message)
        self.target = target
        self.source = source


class DependencyError(ModelDoctorError, ImportError):
    """An explicitly requested optional dependency is missing."""

    def __init__(self, *, missing_package: str, extra_name: str, executable: str) -> None:
        """Initialize a missing optional dependency error."""
        self.missing_package = missing_package
        self.extra_name = extra_name
        self.executable = executable
        super().__init__(
            f"{missing_package} is not installed in this Python environment: {executable}"
        )
