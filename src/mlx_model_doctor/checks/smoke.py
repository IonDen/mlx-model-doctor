"""Runtime smoke checks for MLX text-generation models."""

import importlib
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.environment import format_install_hint, has_uv_context
from mlx_model_doctor.errors import DependencyError, ModelDoctorError
from mlx_model_doctor.memory import install_mlx_memory_caps
from mlx_model_doctor.report import CheckResult


@dataclass(frozen=True, slots=True)
class SmokeGeneration:
    """Observed output from a runtime smoke generation."""

    text: str
    peak_memory_bytes: int | None = None
    memory_caps_gib: tuple[int, int] | None = None


class SmokeBackend(Protocol):
    """Backend boundary for expensive runtime generation."""

    def generate(self, ctx: CheckContext) -> SmokeGeneration:
        """Run a short generation and return observable smoke details."""


class MlxCoreSmokeModule(Protocol):
    """MLX core API surface used by the smoke backend."""

    def device_info(self) -> Mapping[str, object]:
        """Return device metadata."""

    def set_wired_limit(self, value: int) -> None:
        """Set the wired-memory limit."""

    def set_memory_limit(self, value: int) -> None:
        """Set the memory limit."""

    def reset_peak_memory(self) -> None:
        """Reset the top-level peak-memory counter."""

    def get_peak_memory(self) -> int:
        """Return the top-level peak-memory counter."""


class MlxLmModule(Protocol):
    """mlx-lm API surface used by the smoke backend."""

    def load(self, path_or_repo: str) -> tuple[object, object]:
        """Load a model and tokenizer."""

    def generate(
        self,
        model: object,
        tokenizer: object,
        *,
        prompt: str,
        max_tokens: int,
        verbose: bool,
    ) -> str:
        """Generate short text."""


@dataclass(frozen=True, slots=True)
class MlxLmBackend:
    """Smoke backend backed by optional mlx-lm runtime dependencies."""

    prompt: str = "Hello"

    def generate(self, ctx: CheckContext) -> SmokeGeneration:
        """Load the target with mlx-lm and generate a tiny completion."""
        mx, mlx_lm = _import_optional_dependencies()
        caps_gib = install_mlx_memory_caps(mx)
        mx.reset_peak_memory()
        model, tokenizer = mlx_lm.load(ctx.target.name)
        text = mlx_lm.generate(
            model,
            tokenizer,
            prompt=self.prompt,
            max_tokens=8,
            verbose=False,
        )
        return SmokeGeneration(
            text=text,
            peak_memory_bytes=mx.get_peak_memory(),
            memory_caps_gib=caps_gib,
        )


@dataclass(frozen=True, slots=True)
class MlxLmSmokeCheck:
    """Smoke check that asserts mlx-lm can produce non-empty text."""

    check_id: str = "text/smoke.mlx_lm"
    title: str = "MLX-LM smoke check"
    backend: SmokeBackend = field(default_factory=MlxLmBackend)

    def run(self, ctx: CheckContext) -> CheckResult:
        """Run a tiny generation through the configured backend."""
        try:
            generation = self.backend.generate(ctx)
        except ModelDoctorError:
            raise
        except Exception as exc:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message=f"Smoke generation failed: {exc}",
                remediation="Inspect the model with mlx-lm directly before using it.",
            )

        details = _generation_details(generation)
        if not generation.text.strip():
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message="Smoke generation returned empty text.",
                remediation="Verify the tokenizer and generation backend can produce text.",
                details=details,
            )

        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message="Smoke generation produced non-empty text.",
            details=details,
        )


def _generation_details(generation: SmokeGeneration) -> dict[str, object]:
    details: dict[str, object] = {
        "generated_text_chars": len(generation.text),
    }
    if generation.peak_memory_bytes is not None:
        details["peak_memory_bytes"] = generation.peak_memory_bytes
    if generation.memory_caps_gib is not None:
        wired_gib, memory_gib = generation.memory_caps_gib
        details["wired_limit_gib"] = wired_gib
        details["memory_limit_gib"] = memory_gib
    return details


def _import_optional_dependencies() -> tuple[MlxCoreSmokeModule, MlxLmModule]:
    try:
        mx = importlib.import_module("mlx.core")
    except ImportError as exc:
        raise _dependency_error("mlx") from exc

    try:
        mlx_lm = importlib.import_module("mlx_lm")
    except ImportError as exc:
        raise _dependency_error("mlx_lm") from exc

    return cast("MlxCoreSmokeModule", mx), cast("MlxLmModule", mlx_lm)


def _dependency_error(missing_package: str) -> DependencyError:
    hint = format_install_hint(
        missing_package=missing_package,
        extra_name="mlx-lm",
        executable=sys.executable,
        has_uv_context=has_uv_context(
            cwd_files=_cwd_file_names(),
            environ=os.environ,
        ),
    )
    return DependencyError(
        missing_package=missing_package,
        extra_name="mlx-lm",
        executable=sys.executable,
        message=hint,
    )


def _cwd_file_names() -> set[str]:
    try:
        return {path.name for path in Path.cwd().iterdir()}
    except OSError:
        return set()
