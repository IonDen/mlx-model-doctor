"""Runtime smoke checks for MLX text-generation models."""

import importlib
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from mlx_model_doctor.checks.remote_code import (
    KEY_REMEDIATION,
    MLX_LM_KEYS,
    MLX_VLM_KEYS,
    AutoMapLoadError,
    RemoteCodeKeys,
    RemoteCodeRefusedError,
    TemplateRendererRefusedError,
    describe_auto_map_load_failure,
    describe_refusal,
    describe_template_renderer_refusal,
    find_auto_map,
    find_remote_code,
    find_template_renderer,
    reduce_absolute_paths,
    refusal_remediation,
    template_renderer_remediation,
)
from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.environment import format_install_hint, has_uv_context
from mlx_model_doctor.errors import DependencyError, MemorySafetyError, ModelDoctorError
from mlx_model_doctor.memory import install_mlx_memory_caps
from mlx_model_doctor.report import CheckResult
from mlx_model_doctor.targets import HfTarget

if TYPE_CHECKING:
    from mlx_model_doctor.targets import ModelTarget

# The files each loader would download for a Hub repo id, minus "*.py": mlx-lm 0.31.3
# utils._download and mlx-vlm 0.6.12 utils.get_model_path (read 2026-11-16). The smoke
# load runs from this local snapshot. The download does not fetch .py files; the folder
# may still hold some from an earlier full download, which is why the guard reads it.
_MLX_LM_SNAPSHOT_PATTERNS = (
    "*.json",
    "model*.safetensors",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "*.txt",
    "*.jsonl",
    "*.jinja",
)
_MLX_VLM_SNAPSHOT_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.model",
    "*.tiktoken",
    "*.txt",
    "*.jinja",
)


@dataclass(frozen=True, slots=True)
class SmokeGeneration:
    """Observed output from a runtime smoke generation."""

    text: str
    peak_memory_bytes: int | None = None
    memory_caps_gib: tuple[int, int] | None = None
    snapshot_revision: str | None = None


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

    def load(
        self, path_or_hf_repo: str, tokenizer_config: dict[str, object] | None = None
    ) -> tuple[object, object]:
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
        if caps_gib[0] <= 0 or caps_gib[1] <= 0:
            raise MemorySafetyError(
                "MLX memory caps could not be installed; refusing to load model uncapped."
            )
        model_dir, revision = _checked_model_dir(
            ctx.target, allow_patterns=_MLX_LM_SNAPSHOT_PATTERNS, keys=MLX_LM_KEYS
        )
        mx.reset_peak_memory()
        try:
            model, tokenizer = mlx_lm.load(
                str(model_dir), tokenizer_config={"trust_remote_code": False}
            )
        except ModelDoctorError:
            raise
        except Exception as exc:
            # transformers' own text advises trust_remote_code=True; replace it.
            auto_map_file = find_auto_map(model_dir)
            if auto_map_file is None:
                raise
            raise AutoMapLoadError(
                auto_map_file,
                upstream_error=reduce_absolute_paths(str(exc), model_dir=model_dir),
                revision=revision,
            ) from exc
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
            snapshot_revision=revision,
        )


def _checked_model_dir(
    target: "ModelTarget", *, allow_patterns: Sequence[str], keys: RemoteCodeKeys
) -> tuple[Path, str | None]:
    """Return the local directory to load and its snapshot revision, after the guard.

    A Hugging Face target is downloaded first, so the guard and the loader see the same
    files. Raises ``RemoteCodeRefusedError`` when the directory names loader code from
    ``keys``, or a guarded file cannot be read.
    """
    revision: str | None = None
    if target.source == "hf":
        if not isinstance(target, HfTarget):
            raise TypeError(
                f"the smoke load needs a local snapshot of {target.name}, "
                "and this target cannot provide one"
            )
        snapshot = target.snapshot(allow_patterns=allow_patterns)
        root, revision = snapshot.path, snapshot.revision
    else:
        root = Path(target.name)
    try:
        model_dir: Path | None = root.resolve(strict=True)
    except (OSError, RuntimeError):
        model_dir = None
    if model_dir is None or not model_dir.is_dir():
        raise NotADirectoryError(f"model path is not an existing directory: {root}")
    refusal = find_remote_code(model_dir, keys=keys)
    if refusal is not None:
        raise RemoteCodeRefusedError(refusal, revision=revision)
    return model_dir, revision


def _remote_code_result(check_id: str, title: str, exc: RemoteCodeRefusedError) -> CheckResult:
    refusal = exc.refusal
    details: dict[str, object] = {"remote_code_file": refusal.file}
    if refusal.key is None:
        details["remote_code_unreadable"] = True
        if refusal.reason == "dangling_link":
            details["remote_code_dangling_link"] = True
        outcome = "The load was refused, so repository code was not run."
    else:
        details["remote_code_key"] = refusal.key
        outcome = "The load was refused, so that code was not run."
    message = f"{describe_refusal(refusal)} {outcome} The static checks above still ran."
    remediation = refusal_remediation(refusal)
    if exc.revision is not None:
        details["snapshot_revision"] = exc.revision
    return CheckResult(
        check_id=check_id,
        title=title,
        status="fail",
        severity="high",
        message=message,
        remediation=remediation,
        details=details,
    )


def _template_renderer_result(
    check_id: str, title: str, exc: TemplateRendererRefusedError
) -> CheckResult:
    refusal = exc.refusal
    details: dict[str, object] = {}
    if refusal.renderer is not None:
        details["remote_code_renderer"] = refusal.renderer
    if refusal.reason == "unreadable":
        details["remote_code_file"] = refusal.file
        details["remote_code_unreadable"] = True
    else:
        details["remote_code_template_source"] = refusal.file
    if exc.revision is not None:
        details["snapshot_revision"] = exc.revision
    return CheckResult(
        check_id=check_id,
        title=title,
        status="fail",
        severity="high",
        message=(
            f"{describe_template_renderer_refusal(refusal)} The static checks above still ran."
        ),
        remediation=template_renderer_remediation(refusal),
        details=details,
    )


def _auto_map_load_result(check_id: str, title: str, exc: AutoMapLoadError) -> CheckResult:
    details: dict[str, object] = {
        "remote_code_file": exc.file,
        "remote_code_key": "auto_map",
        "upstream_error": exc.upstream_error,
    }
    if exc.revision is not None:
        details["snapshot_revision"] = exc.revision
    return CheckResult(
        check_id=check_id,
        title=title,
        status="fail",
        severity="high",
        message=f"{describe_auto_map_load_failure(exc.file)} The static checks above still ran.",
        remediation=KEY_REMEDIATION,
        details=details,
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
        except RemoteCodeRefusedError as exc:
            return _remote_code_result(self.check_id, self.title, exc)
        except AutoMapLoadError as exc:
            return _auto_map_load_result(self.check_id, self.title, exc)
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


# Checked against mlx-vlm 0.6.12 with ``inspect.signature`` on the installed package, and
# against the 0.7.10 source on 2026-10-24 (the same signatures).
class MlxVlmModule(Protocol):
    """mlx-vlm API surface used by the VLM smoke backend.

    Signatures verified against mlx-vlm 0.6.12 and 0.7.10: ``load`` and ``generate``
    accept many more parameters via ``**kwargs`` than declared here; only the ones this
    backend passes are part of this Protocol. ``apply_chat_template`` takes a
    model-config argument (``processor, config, prompt, ...``), not just a processor
    and a prompt.
    """

    def load(
        self, path_or_hf_repo: str, *, trust_remote_code: bool = False
    ) -> tuple[object, object]:
        """Load a VLM model and processor."""

    def apply_chat_template(
        self, processor: object, config: object, prompt: str, *, num_images: int = 1
    ) -> str:
        """Format a prompt with image tokens via the model's chat template."""

    def generate(
        self,
        model: object,
        processor: object,
        prompt: str,
        image: object,
        *,
        max_tokens: int = 8,
        verbose: bool = False,
    ) -> object:
        """Generate text from a VLM. Returns an object with a .text attribute."""


@dataclass(frozen=True, slots=True)
class MlxVlmBackend:
    """Smoke backend backed by optional mlx-vlm runtime dependencies."""

    prompt: str = "Describe this image."
    vlm_module: MlxVlmModule | None = None
    mx_module: MlxCoreSmokeModule | None = None

    def generate(self, ctx: CheckContext) -> SmokeGeneration:
        """Load a VLM target and generate a tiny completion from a dummy image."""
        vlm = self.vlm_module if self.vlm_module is not None else _import_vlm_module()
        mx = self.mx_module if self.mx_module is not None else _import_mlx_for_vlm()
        caps_gib = install_mlx_memory_caps(mx)
        if caps_gib[0] <= 0 or caps_gib[1] <= 0:
            raise MemorySafetyError(
                "MLX memory caps could not be installed; refusing to load VLM uncapped."
            )
        model_dir, revision = _checked_model_dir(
            ctx.target, allow_patterns=_MLX_VLM_SNAPSHOT_PATTERNS, keys=MLX_VLM_KEYS
        )
        renderer = find_template_renderer(model_dir)
        if renderer is not None:
            raise TemplateRendererRefusedError(renderer, revision=revision)
        mx.reset_peak_memory()
        model, processor = vlm.load(str(model_dir), trust_remote_code=False)
        # Re-install caps after load (mlx-vlm may override wired limit during load).
        caps_gib = install_mlx_memory_caps(mx)
        if caps_gib[0] <= 0 or caps_gib[1] <= 0:
            raise MemorySafetyError(
                "MLX memory caps could not be reinstalled after VLM load; refusing to generate uncapped."
            )
        config = getattr(model, "config", None)
        formatted_prompt = vlm.apply_chat_template(processor, config, self.prompt, num_images=1)
        image = _dummy_image()
        result = vlm.generate(
            model,
            processor,
            formatted_prompt,
            image,
            max_tokens=8,
            verbose=False,
        )
        raw_text = getattr(result, "text", None)
        if isinstance(raw_text, str):
            text = raw_text
        elif isinstance(result, str):
            text = result
        else:
            text = ""
        return SmokeGeneration(
            text=text,
            peak_memory_bytes=mx.get_peak_memory(),
            memory_caps_gib=caps_gib,
            snapshot_revision=revision,
        )


def _dummy_image() -> object:
    """Create a 64x64 solid-color PIL Image for smoke testing."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise _dependency_error_vlm("Pillow") from exc
    return Image.new("RGB", (64, 64), color=(128, 128, 128))


@dataclass(frozen=True, slots=True)
class MlxVlmSmokeCheck:
    """Smoke check that asserts mlx-vlm can produce non-empty text from a dummy image."""

    check_id: str = "vlm/smoke.mlx_vlm"
    title: str = "MLX-VLM smoke check"
    backend: SmokeBackend = field(default_factory=MlxVlmBackend)

    def run(self, ctx: CheckContext) -> CheckResult:
        """Run a tiny VLM generation through the configured backend."""
        try:
            generation = self.backend.generate(ctx)
        except ModelDoctorError:
            raise
        except RemoteCodeRefusedError as exc:
            return _remote_code_result(self.check_id, self.title, exc)
        except TemplateRendererRefusedError as exc:
            return _template_renderer_result(self.check_id, self.title, exc)
        except Exception as exc:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message=f"VLM smoke generation failed: {exc}",
                remediation="Inspect the model with mlx-vlm directly before using it.",
            )

        details = _generation_details(generation)
        if not generation.text.strip():
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message="VLM smoke generation returned empty text.",
                remediation="Verify the VLM model and processor can produce text from an image.",
                details=details,
            )

        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message="VLM smoke generation produced non-empty text.",
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
    if generation.snapshot_revision is not None:
        details["snapshot_revision"] = generation.snapshot_revision
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


def _import_vlm_module() -> MlxVlmModule:
    try:
        vlm = importlib.import_module("mlx_vlm")
    except ImportError as exc:
        raise _dependency_error_vlm("mlx-vlm") from exc
    return cast("MlxVlmModule", vlm)


def _import_mlx_for_vlm() -> MlxCoreSmokeModule:
    try:
        mx = importlib.import_module("mlx.core")
    except ImportError as exc:
        raise _dependency_error_vlm("mlx") from exc
    return cast("MlxCoreSmokeModule", mx)


def _dependency_error_vlm(missing_package: str) -> DependencyError:
    hint = format_install_hint(
        missing_package=missing_package,
        extra_name="mlx-vlm",
        executable=sys.executable,
        has_uv_context=has_uv_context(
            cwd_files=_cwd_file_names(),
            environ=os.environ,
        ),
    )
    return DependencyError(
        missing_package=missing_package,
        extra_name="mlx-vlm",
        executable=sys.executable,
        message=hint,
    )
