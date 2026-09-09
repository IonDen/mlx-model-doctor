"""Memory-safe parity worker.

Defines the worker's spec/result types, the ``ParityWorkerBackend`` seam,
adapter-manifest validation (F3), the validated numpy-free JSON IPC contract
(F9), the real ``mlx-lm``-backed backend, and the subprocess ``main()`` entry
point invoked as ``python -m mlx_model_doctor.parity.worker``.
"""

import argparse
import importlib
import json
import os
import struct
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Literal, Protocol, cast

from mlx_model_doctor.environment import format_install_hint, has_uv_context
from mlx_model_doctor.errors import (
    DependencyError,
    MemorySafetyError,
    ModelDoctorError,
    WorkerArtifactError,
)
from mlx_model_doctor.memory import MlxMemoryModule, install_mlx_memory_caps
from mlx_model_doctor.parity.watchdog import _do_abort, default_ceiling_bytes, run_watchdog
from mlx_model_doctor.safetensors_header import (
    _MAX_HEADER_BYTES,
    FileHeader,
    SafetensorsHeaderError,
    parse_file_header,
)

# The mlx-lm LoRA family's fine_tune_type values this worker knows how to validate
# and score a reference load against. An unrecognized value is "unsupported", never
# a silent "not applied".
SUPPORTED_FINE_TUNE_TYPES = frozenset({"lora", "dora", "lora-switch", "lora-embed"})

# mlx-lm's own CONFIG_DEFAULTS lora scale (mlx_lm.lora.CONFIG_DEFAULTS["lora_parameters"]).
_DEFAULT_LORA_SCALE = 20.0

# Below this magnitude a LoRA factor's contribution is treated as the untrained
# (zero-initialized) state, not a genuine learned effect.
_NONNEGLIGIBLE_DELTA_EPS = 1e-6

_WORKER_CACHE_LIMIT_BYTES = 4 * 1024**3
_DEFAULT_WALL_DEADLINE_S = 300.0
_DEFAULT_POLL_S = 0.05


# --- Spec / result --------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkerSpec:
    """A single worker invocation: which model/adapter to load and what to score.

    ``token_ids`` is a tuple of one or more token-id sequences. The backend
    loads the model ONCE and runs one forward per sequence, concatenating the
    per-sequence argmax, in sequence order, into ``WorkerResult.argmax``.
    """

    model_path: str
    adapter_path: str | None = None
    token_ids: tuple[tuple[int, ...], ...]
    fixture_id: str
    role: str


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkerResult:
    """The observable output of one worker run."""

    argmax: list[int]
    adapter_applied: bool | None
    peak_bytes: int
    role: str
    fixture_id: str


class ParityWorkerBackend(Protocol):
    """Backend boundary for producing a worker's teacher-forced argmax result."""

    def load_argmax(self, spec: WorkerSpec) -> WorkerResult:
        """Load the target (and optional adapter) and score the spec's token ids."""


def run_worker_body(
    spec: WorkerSpec,
    backend: ParityWorkerBackend,
    *,
    caps_fn: Callable[[], tuple[int, int]],
) -> WorkerResult:
    """Refuse to run uncapped, then load and score through ``backend`` (F2).

    ``caps_fn`` is called (and its result checked) strictly before
    ``backend.load_argmax`` so a worker can never load a model with unbounded
    MLX memory. The expected argmax length is the TOTAL token count across
    every sequence in ``spec.token_ids`` (the flat per-sequence-argmax
    concatenation a backend produces), not the sequence count.
    """
    wired_gib, memory_gib = caps_fn()
    if wired_gib <= 0 or memory_gib <= 0:
        raise MemorySafetyError(
            "MLX memory caps could not be installed; refusing to run the parity worker uncapped."
        )
    result = backend.load_argmax(spec)
    expected_length = sum(len(sequence) for sequence in spec.token_ids)
    if len(result.argmax) != expected_length:
        raise ModelDoctorError(
            f"worker backend returned {len(result.argmax)} argmax ids for "
            f"{expected_length} total token ids across {len(spec.token_ids)} "
            f"sequence(s) (fixture={spec.fixture_id!r}, role={spec.role!r})"
        )
    return result


# --- Adapter-manifest validation (F3) -------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class AdapterValidation:
    """Result of validating an adapter directory against an expected target list."""

    status: Literal["ok", "incomplete", "unsupported"]
    missing: list[str] = field(default_factory=list)
    reason: str | None = None


def validate_adapter_manifest(
    adapter_dir: Path | str, resolver_targets: Sequence[str]
) -> AdapterValidation:
    """Validate an mlx-lm adapter directory before it is trusted as a reference (F3).

    Reads ``adapter_config.json`` and the ``adapters.safetensors`` header (never
    weight data) and checks: the ``fine_tune_type`` is a supported family; every
    expected target in ``resolver_targets`` has its required factor pair (``lora_a``
    and ``lora_b``; DoRA also its magnitude ``m``) with matching rank dimensions.
    """
    directory = Path(adapter_dir)
    config_path = directory / "adapter_config.json"
    try:
        raw_config = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        return AdapterValidation(
            status="incomplete",
            missing=list(resolver_targets),
            reason=f"could not read adapter_config.json: {exc}",
        )
    try:
        config = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        return AdapterValidation(
            status="incomplete",
            missing=list(resolver_targets),
            reason=f"adapter_config.json is not valid JSON: {exc}",
        )
    if not isinstance(config, dict):
        return AdapterValidation(
            status="incomplete",
            missing=list(resolver_targets),
            reason="adapter_config.json must be a JSON object",
        )

    fine_tune_type = config.get("fine_tune_type", "lora")
    if not isinstance(fine_tune_type, str) or fine_tune_type not in SUPPORTED_FINE_TUNE_TYPES:
        return AdapterValidation(
            status="unsupported",
            reason=f"unsupported fine_tune_type: {fine_tune_type!r}",
        )

    lora_parameters = config.get("lora_parameters", {})
    scale = _DEFAULT_LORA_SCALE
    if isinstance(lora_parameters, dict) and "scale" in lora_parameters:
        raw_scale = lora_parameters["scale"]
        try:
            scale = float(raw_scale)
        except (TypeError, ValueError):
            # An absent scale falls back to mlx-lm's own default (below); a
            # *present but garbled* scale is a manifest we can't trust (F3).
            return AdapterValidation(
                status="incomplete",
                missing=list(resolver_targets),
                reason=f"lora_parameters.scale is not numeric: {raw_scale!r}",
            )
    if scale == 0:
        return AdapterValidation(
            status="incomplete",
            missing=list(resolver_targets),
            reason="lora_parameters.scale is zero; the adapter has no effective contribution",
        )

    weights_path = directory / "adapters.safetensors"
    try:
        header = _read_local_safetensors_header(weights_path)
    except (OSError, SafetensorsHeaderError) as exc:
        return AdapterValidation(
            status="incomplete",
            missing=list(resolver_targets),
            reason=f"could not read adapters.safetensors: {exc}",
        )

    requires_magnitude = fine_tune_type == "dora"
    missing: list[str] = []
    reason: str | None = None
    for target in resolver_targets:
        a_entry = header.tensors.get(f"{target}.lora_a")
        b_entry = header.tensors.get(f"{target}.lora_b")
        if a_entry is None:
            missing.append(target)
            reason = reason or f"target {target!r} is missing its lora_a tensor"
            continue
        if b_entry is None:
            missing.append(target)
            reason = reason or f"target {target!r} is missing its lora_b tensor"
            continue
        if a_entry.shape[-1] != b_entry.shape[0]:
            missing.append(target)
            reason = reason or (
                f"target {target!r} has mismatched lora_a/lora_b rank dimensions "
                f"({a_entry.shape} vs {b_entry.shape})"
            )
            continue
        if requires_magnitude:
            m_entry = header.tensors.get(f"{target}.m")
            if m_entry is None:
                missing.append(target)
                reason = reason or f"target {target!r} is missing its DoRA magnitude tensor"
                continue
            if not _dora_magnitude_shape_matches(m_entry.shape, a_entry.shape, b_entry.shape):
                missing.append(target)
                reason = reason or (
                    f"target {target!r} has a DoRA magnitude tensor with an unexpected "
                    f"shape {m_entry.shape} (expected rank-1, matching lora_a.shape[0] "
                    f"or lora_b.shape[-1])"
                )
                continue

    if missing:
        return AdapterValidation(status="incomplete", missing=missing, reason=reason)
    return AdapterValidation(status="ok")


def _dora_magnitude_shape_matches(
    m_shape: tuple[int, ...], a_shape: tuple[int, ...], b_shape: tuple[int, ...]
) -> bool:
    """Check a DoRA magnitude tensor's shape against both legitimate conventions.

    The safetensors header alone can't disambiguate ``DoRALinear`` (``m.shape ==
    (output_dims,)``, matching ``lora_b.shape[-1]``) from ``DoRAEmbedding``
    (``m.shape == (num_embeddings,)``, matching ``lora_a.shape[0]``) — see
    ``mlx_lm/tuner/dora.py`` — so either convention is accepted.
    """
    if len(m_shape) != 1:
        return False
    dim = m_shape[0]
    return dim == a_shape[0] or dim == b_shape[-1]


def _read_local_safetensors_header(path: Path) -> FileHeader:
    """Read one local safetensors file's header without loading tensor data."""
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) < 8:
            raise SafetensorsHeaderError(f"{path}: truncated safetensors header prefix")
        header_length = int(struct.unpack("<Q", prefix)[0])
        if header_length > _MAX_HEADER_BYTES:
            raise SafetensorsHeaderError(f"{path}: safetensors header too large")
        header_bytes = handle.read(header_length)
    raw = prefix + header_bytes
    return parse_file_header(str(path), raw, file_size=path.stat().st_size)


# --- Validated JSON IPC (F9) -----------------------------------------------------


def write_worker_json(path: Path | str, result: WorkerResult) -> None:
    """Write a worker result as a plain JSON object (no ``.npy``, no numpy)."""
    payload: dict[str, object] = {
        "fixture_id": result.fixture_id,
        "role": result.role,
        "argmax": list(result.argmax),
        "peak_bytes": result.peak_bytes,
        "adapter_applied": result.adapter_applied,
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload), encoding="utf-8")


def read_worker_json(
    path: Path | str,
    *,
    expect_fixture: str,
    expect_role: str,
    vocab_size: int,
    length: int,
) -> WorkerResult:
    """Read and fully validate a worker JSON artifact before it is trusted (F9).

    Rejects a missing file, malformed JSON, a wrong rank/dtype ``argmax``, a
    length that does not equal ``length``, an out-of-vocab id, and a wrong
    ``fixture_id``/``role`` — each with a clear ``WorkerArtifactError``.
    """
    if length <= 0:
        raise ValueError("length must be positive")

    in_path = Path(path)
    try:
        raw = in_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkerArtifactError(f"worker artifact missing or unreadable: {path}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WorkerArtifactError(f"worker artifact is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise WorkerArtifactError(f"worker artifact must be a JSON object: {path}")

    for key in ("fixture_id", "role", "argmax", "peak_bytes", "adapter_applied"):
        if key not in payload:
            raise WorkerArtifactError(f"worker artifact is missing {key!r}: {path}")

    fixture_id = payload["fixture_id"]
    role = payload["role"]
    argmax = payload["argmax"]
    peak_bytes = payload["peak_bytes"]
    adapter_applied = payload["adapter_applied"]

    if not isinstance(fixture_id, str):
        raise WorkerArtifactError(f"worker artifact fixture_id must be a string: {path}")
    if not isinstance(role, str):
        raise WorkerArtifactError(f"worker artifact role must be a string: {path}")
    if not isinstance(peak_bytes, int) or isinstance(peak_bytes, bool):
        raise WorkerArtifactError(f"worker artifact peak_bytes must be an integer: {path}")
    if adapter_applied is not None and not isinstance(adapter_applied, bool):
        raise WorkerArtifactError(f"worker artifact adapter_applied must be a bool or null: {path}")
    if not isinstance(argmax, list) or not all(_is_plain_int(x) for x in argmax):
        raise WorkerArtifactError(f"worker artifact argmax must be a flat list of integers: {path}")
    if len(argmax) != length:
        raise WorkerArtifactError(
            f"worker artifact argmax length {len(argmax)} != expected fixture length "
            f"{length}: {path}"
        )
    if any(item < 0 or item >= vocab_size for item in argmax):
        raise WorkerArtifactError(f"worker artifact argmax contains an out-of-vocab id: {path}")
    if fixture_id != expect_fixture:
        raise WorkerArtifactError(
            f"worker artifact fixture_id {fixture_id!r} != expected {expect_fixture!r}: {path}"
        )
    if role != expect_role:
        raise WorkerArtifactError(
            f"worker artifact role {role!r} != expected {expect_role!r}: {path}"
        )

    return WorkerResult(
        argmax=[int(item) for item in argmax],
        adapter_applied=adapter_applied,
        peak_bytes=peak_bytes,
        role=role,
        fixture_id=fixture_id,
    )


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# --- Real mlx-lm-backed worker backend -------------------------------------------


class MxArrayLike(Protocol):
    """Minimal MLX array surface used by the parity worker."""

    def tolist(self) -> list[int]:
        """Convert a 1-D array to a flat list of ints."""

    def __bool__(self) -> bool:
        """Support truthiness checks on a reduction's result."""

    def __float__(self) -> float:
        """Support ``float()`` on a scalar reduction's result."""

    def __getitem__(self, index: int) -> "MxArrayLike":
        """Index into the leading axis."""


class MlxCoreParityModule(Protocol):
    """MLX core API surface used by the parity worker."""

    def device_info(self) -> dict[str, object]:
        """Return device metadata."""

    def set_wired_limit(self, value: int) -> None:
        """Set the wired-memory limit."""

    def set_memory_limit(self, value: int) -> None:
        """Set the memory limit."""

    def set_cache_limit(self, value: int) -> None:
        """Set the retained allocator cache limit."""

    def get_active_memory(self) -> int:
        """Return current active (in-use) memory in bytes."""

    def get_cache_memory(self) -> int:
        """Return current retained cache memory in bytes."""

    def get_peak_memory(self) -> int:
        """Return the top-level peak-memory counter."""

    def array(self, value: object) -> MxArrayLike:
        """Construct an MLX array."""

    def argmax(self, value: object, axis: int) -> MxArrayLike:
        """Return the index of the maximum value along an axis."""

    def isfinite(self, value: object) -> MxArrayLike:
        """Elementwise finiteness check."""

    def all(self, value: object) -> MxArrayLike:
        """Reduce with logical AND."""

    def abs(self, value: object) -> MxArrayLike:
        """Elementwise absolute value."""

    def max(self, value: object) -> MxArrayLike:
        """Reduce with maximum."""

    def eval(self, *values: object) -> None:
        """Force evaluation of lazily-constructed arrays."""


class ParityModel(Protocol):
    """Minimal ``nn.Module``-shaped surface used by the parity worker."""

    def __call__(self, ids: object) -> MxArrayLike:
        """Run a forward pass and return logits."""

    def named_modules(self) -> Iterable[tuple[str, object]]:
        """Yield ``(name, module)`` pairs across the full module tree."""


class MlxLmParityModule(Protocol):
    """mlx-lm API surface used by the parity worker."""

    def load(
        self, path_or_repo: str, *, adapter_path: str | None = None
    ) -> tuple[ParityModel, object]:
        """Load a model and tokenizer, optionally applying LoRA/DoRA adapters."""


@dataclass(frozen=True, slots=True)
class MlxLmWorkerBackend:
    """Parity worker backend backed by the real ``mlx-lm`` runtime.

    Caps installation is the caller's responsibility (``run_worker_body``); this
    backend assumes MLX memory is already bounded by the time it runs.
    """

    def load_argmax(self, spec: WorkerSpec) -> WorkerResult:
        """Load the target ONCE (with optional adapter) and score every sequence.

        Runs one forward pass per sequence in ``spec.token_ids`` against the
        single loaded model, concatenating each sequence's per-position argmax,
        in sequence order, into one flat result vector.
        """
        mx = _import_mlx_core()
        mlx_lm_module = _import_mlx_lm()
        model, _tokenizer = mlx_lm_module.load(spec.model_path, adapter_path=spec.adapter_path)
        applied = _adapter_applied_from_reference(model, spec) if spec.adapter_path else None
        flat_argmax: list[int] = []
        for sequence in spec.token_ids:
            ids = mx.array([list(sequence)])
            logits = model(ids)[0]
            if not bool(mx.all(mx.isfinite(logits))):
                raise ModelDoctorError("non-finite logits in parity worker")
            am = mx.argmax(logits, axis=-1)
            mx.eval(am)
            flat_argmax.extend(int(x) for x in am.tolist())
        return WorkerResult(
            argmax=flat_argmax,
            adapter_applied=applied,
            peak_bytes=int(mx.get_peak_memory()),
            role=spec.role,
            fixture_id=spec.fixture_id,
        )


def _adapter_applied_from_reference(model: ParityModel, spec: WorkerSpec) -> bool | None:
    """Decide ``adapter_applied`` from the base+adapter reference only (never fused bytes, F1)."""
    assert spec.adapter_path is not None, "an adapter reference requires an adapter_path"
    targets = _discover_lora_family_targets(model)
    validation = validate_adapter_manifest(Path(spec.adapter_path), targets)
    if validation.status == "unsupported":
        return None
    if validation.status == "incomplete":
        return False
    if not targets:
        return False
    return _has_nonnegligible_delta(model, targets)


def _discover_lora_family_targets(model: ParityModel) -> list[str]:
    """Walk the model's module tree for LoRA/DoRA-family modules (duck-typed).

    A module is treated as LoRA-family (``LoRALinear``/``DoRALinear``/
    ``LoRASwitchLinear``/``LoRAEmbedding``) when it exposes both ``lora_a`` and
    ``lora_b``, which all four mlx-lm classes do.
    """
    return [
        name
        for name, module in model.named_modules()
        if hasattr(module, "lora_a") and hasattr(module, "lora_b")
    ]


def _has_nonnegligible_delta(model: ParityModel, targets: Sequence[str]) -> bool:
    """Check whether any discovered target's ``lora_b`` factor is non-negligible.

    Checks the zero-initialized factor's magnitude directly rather than
    multiplying out the full dense delta.
    """
    mx = _import_mlx_core()
    modules = dict(model.named_modules())
    for name in targets:
        module = modules.get(name)
        if module is None:
            continue
        lora_b = getattr(module, "lora_b", None)
        if lora_b is None:
            continue
        magnitude = float(mx.max(mx.abs(lora_b)))
        if magnitude > _NONNEGLIGIBLE_DELTA_EPS:
            return True
    return False


def _import_mlx_core() -> MlxCoreParityModule:
    try:
        mx = importlib.import_module("mlx.core")
    except ImportError as exc:
        raise _dependency_error("mlx") from exc
    return cast("MlxCoreParityModule", mx)


def _import_mlx_lm() -> MlxLmParityModule:
    try:
        mlx_lm_module = importlib.import_module("mlx_lm")
    except ImportError as exc:
        raise _dependency_error("mlx_lm") from exc
    return cast("MlxLmParityModule", mlx_lm_module)


def _dependency_error(missing_package: str) -> DependencyError:
    hint = format_install_hint(
        missing_package=missing_package,
        extra_name="mlx-lm",
        executable=sys.executable,
        has_uv_context=has_uv_context(cwd_files=_cwd_file_names(), environ=os.environ),
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


# --- Subprocess entry point -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _MxMemoryProbe:
    """Adapts an ``MlxCoreParityModule`` to the watchdog's ``MemoryProbe`` protocol."""

    mx: MlxCoreParityModule

    def active_bytes(self) -> int:
        """Return current active memory in bytes."""
        return self.mx.get_active_memory()

    def cache_bytes(self) -> int:
        """Return current retained cache memory in bytes."""
        return self.mx.get_cache_memory()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m mlx_model_doctor.parity.worker")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument(
        "--token-ids",
        required=True,
        help=(
            "token-id sequences: one or more comma-separated integer sequences, "
            "separated by ';' (e.g. '1,2,3;4,5' for two sequences)"
        ),
    )
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--out", required=True, help="path to write the worker result JSON")
    parser.add_argument("--wall-deadline-s", type=float, default=_DEFAULT_WALL_DEADLINE_S)
    parser.add_argument("--poll-s", type=float, default=_DEFAULT_POLL_S)
    return parser


def _parse_token_id_sequences(raw: str) -> tuple[tuple[int, ...], ...]:
    """Parse ``;``-separated sequences of ``,``-separated integer token ids.

    A fixture requires at least one nonempty sequence: both an empty overall
    input and an empty individual sequence (e.g. two consecutive ``;``, or a
    trailing ``;``) raise ``ValueError``.
    """
    stripped = raw.strip()
    if not stripped:
        raise ValueError(
            "token id sequences must not be empty; a fixture requires at least one sequence"
        )
    sequences: list[tuple[int, ...]] = []
    for chunk in stripped.split(";"):
        chunk = chunk.strip()
        if not chunk:
            raise ValueError(f"token id sequences must not contain an empty sequence: {raw!r}")
        sequences.append(tuple(int(item) for item in chunk.split(",")))
    return tuple(sequences)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the parity worker subprocess body end-to-end.

    Installs MLX memory caps, starts the memory+wall watchdog, runs the worker
    body through the real ``mlx-lm`` backend, and writes the validated JSON
    result artifact.
    """
    args = _build_arg_parser().parse_args(argv)
    spec = WorkerSpec(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        token_ids=_parse_token_id_sequences(args.token_ids),
        fixture_id=args.fixture_id,
        role=args.role,
    )
    out_path = Path(args.out)
    # Create the output directory before the watchdog can possibly fire, so an
    # early abort's marker write (best-effort, F2) doesn't silently fail on a
    # not-yet-created directory.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = str(out_path.parent)

    mx = _import_mlx_core()
    wired_gib, memory_gib = install_mlx_memory_caps(cast("MlxMemoryModule", mx))
    if wired_gib <= 0 or memory_gib <= 0:
        raise MemorySafetyError(
            "MLX memory caps could not be installed; refusing to run the parity worker uncapped."
        )
    mx.set_cache_limit(_WORKER_CACHE_LIMIT_BYTES)

    device_info = mx.device_info()
    memory_size = device_info.get("memory_size")
    if not isinstance(memory_size, int):
        raise MemorySafetyError("MLX device_info() did not report an integer memory_size")
    ceiling = default_ceiling_bytes(memory_size)

    stop = Event()
    thread = run_watchdog(
        _MxMemoryProbe(mx),
        ceiling,
        on_abort=lambda reason: _do_abort(out_dir, reason),
        poll_s=args.poll_s,
        deadline_s=args.wall_deadline_s,
        stop=stop,
    )
    try:
        result = run_worker_body(
            spec,
            MlxLmWorkerBackend(),
            caps_fn=lambda: install_mlx_memory_caps(cast("MlxMemoryModule", mx)),
        )
        write_worker_json(out_path, result)
        print(f"::PARITY_WORKER::ok role={spec.role} fixture={spec.fixture_id}")
        return 0
    finally:
        stop.set()
        thread.join(timeout=2.0)


if __name__ == "__main__":
    sys.exit(main())
