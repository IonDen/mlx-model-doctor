"""MLX model repository validation."""

import logging
from typing import Final

from mlx_model_doctor.api import (
    ParityOptions,
    check_adapter_parity,
    check_hf_model,
    check_local_model,
)
from mlx_model_doctor.context import CheckOptions
from mlx_model_doctor.errors import (
    DependencyError,
    MemorySafetyError,
    ModelDoctorError,
    TargetError,
)
from mlx_model_doctor.exit_codes import FailOn, exit_code_for
from mlx_model_doctor.parity.exit_codes import parity_exit_code
from mlx_model_doctor.parity.oracle import ParityVerdict
from mlx_model_doctor.parity.report import (
    ParityReport,
    render_parity_json,
    render_parity_markdown,
    render_parity_text,
)
from mlx_model_doctor.report import (
    CheckResult,
    DoctorReport,
    render_github,
    render_json,
    render_markdown,
    render_text,
)

__all__: Final = [
    "CheckOptions",
    "CheckResult",
    "DependencyError",
    "DoctorReport",
    "FailOn",
    "MemorySafetyError",
    "ModelDoctorError",
    "ParityOptions",
    "ParityReport",
    "ParityVerdict",
    "TargetError",
    "check_adapter_parity",
    "check_hf_model",
    "check_local_model",
    "exit_code_for",
    "parity_exit_code",
    "render_github",
    "render_json",
    "render_markdown",
    "render_parity_json",
    "render_parity_markdown",
    "render_parity_text",
    "render_text",
]

logging.getLogger("mlx_model_doctor").addHandler(logging.NullHandler())
