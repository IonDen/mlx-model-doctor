"""Public Python API."""

from pathlib import Path

from mlx_model_doctor.report import DoctorReport


def check_local_model(path: str | Path) -> DoctorReport:
    """Check a local model repository."""
    raise NotImplementedError("check_local_model requires the local static runner")


def check_hf_model(repo_id: str) -> DoctorReport:
    """Check a Hugging Face model repository."""
    raise NotImplementedError("check_hf_model requires the Hugging Face target")
