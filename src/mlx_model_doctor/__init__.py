"""MLX model repository validation."""

import logging
from typing import Final

from mlx_model_doctor.api import check_hf_model, check_local_model
from mlx_model_doctor.errors import (
    DependencyError,
    MemorySafetyError,
    ModelDoctorError,
    TargetError,
)
from mlx_model_doctor.report import CheckResult, DoctorReport

__all__: Final = [
    "CheckResult",
    "DependencyError",
    "DoctorReport",
    "MemorySafetyError",
    "ModelDoctorError",
    "TargetError",
    "check_hf_model",
    "check_local_model",
]

logging.getLogger("mlx_model_doctor").addHandler(logging.NullHandler())
