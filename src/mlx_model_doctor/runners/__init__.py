"""Check runners."""

from mlx_model_doctor.runners.smoke import run_smoke_checks
from mlx_model_doctor.runners.static import run_static_checks

__all__ = ["run_smoke_checks", "run_static_checks"]
