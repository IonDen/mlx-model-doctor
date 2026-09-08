"""Quant-config-gated byte-compare delta map for adapter-fuse parity (F11).

For every tensor the base repository's safetensors header lists, this module
decides whether its fused-repository counterpart is comparable at all before
ever comparing a single byte, then classifies the pair into one of five
kinds (:data:`TensorDeltaKlass`).

**Comparability gate, first.** A tensor's effective quantization shape (MLX
terms: mode/bits/group_size) is fully determined, for byte-compare purposes,
by the dtype and shape of its ``.weight`` tensor plus which of ``.scales``/
``.biases`` exist alongside it: any real change to bits or group_size
changes at least one of those (MLX's ``to_quantized``/``QuantizedLinear``).
So comparing dtype+shape per tensor -- anchored on the owning module's
``.weight`` -- detects a representation change without needing
``config.json``'s quantization block at all. A ``--dequantize`` fuse, which
turns quantized ``nn.QuantizedLinear`` weights into plain float
``nn.Linear`` weights (dropping ``.scales``/``.biases`` and changing the
``.weight`` dtype/shape) for the WHOLE model -- targets and non-targets
alike -- is reported ``non_comparable`` with a reason, never misclassified
as ``missing`` or ``unexpected``. Conversely, when the ``.weight`` anchor is
byte-for-byte unchanged but a required ``.scales``/``.biases`` sibling
vanished, that is a genuine failure (``missing``), not a benign
representation change.

**Then classify byte-comparable tensors.** A target (its module path is in
the resolver's expected-change set) that differs is the positive proof a
fuse applied (``changed``); a target with identical bytes is ``unchanged``;
a byte-comparable tensor outside the target set that differs anyway is
``unexpected``.

Local repositories only in v1 (Hugging Face targets are out of scope for
this module); comparison is byte-only -- no MLX/numpy import -- and reads
each tensor's payload in bounded chunks via
``LocalTarget.read_bytes(..., offset=..., length=...)`` so a single very
large tensor is never materialized in memory at once.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from mlx_model_doctor.safetensors_header import FileHeader, SafetensorsHeader, TensorEntry
from mlx_model_doctor.targets import LocalTarget

TensorDeltaKlass = Literal["changed", "unchanged", "missing", "unexpected", "non_comparable"]

_ALL_KLASSES: tuple[TensorDeltaKlass, ...] = (
    "changed",
    "unchanged",
    "missing",
    "unexpected",
    "non_comparable",
)

# Bound a single read from either file to this many bytes, so byte-comparing a
# multi-gigabyte tensor never materializes more than this much memory at once.
_COMPARE_CHUNK_BYTES = 1024 * 1024  # 1 MiB

# Suffixes stripped to recover a module's path, matching MLX's
# nn.QuantizedLinear / nn.Linear parameter names.
_QUANT_PAYLOAD_SUFFIXES = (".weight", ".scales", ".biases")


@dataclass(frozen=True, slots=True, kw_only=True)
class TensorDelta:
    """One tensor's classification between a base repo and its fused counterpart."""

    tensor: str
    klass: TensorDeltaKlass
    reason: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class DeltaSummary:
    """Diagnostic aggregate over a delta map.

    Diagnostic only: this does NOT decide whether the adapter was applied to
    the base model -- that verdict comes from the reference-load logit
    comparison (F1), independently of this static byte map.
    """

    counts_by_klass: Mapping[TensorDeltaKlass, int]
    tensors_by_klass: Mapping[TensorDeltaKlass, tuple[str, ...]]


def _prefix_of(tensor_name: str) -> str:
    """Strip a trailing quantization-payload suffix, returning the module path."""
    for suffix in _QUANT_PAYLOAD_SUFFIXES:
        if tensor_name.endswith(suffix):
            return tensor_name[: -len(suffix)]
    return tensor_name


def _weight_representation_changed(
    tensor_name: str, base_hdr: SafetensorsHeader, fused_hdr: SafetensorsHeader
) -> bool:
    """Whether the tensor's module-level ``.weight`` anchor changed dtype/shape.

    This is what tells a benign, whole-module representation change (e.g. a
    dequantize fuse) apart from a genuinely missing quantization auxiliary:
    when the anchor ``.weight`` is unchanged, an absent ``.scales``/
    ``.biases`` is a real failure (``missing``); when the anchor itself
    changed, a sibling that vanished did so *because* the representation
    changed (``non_comparable``).
    """
    weight_name = _prefix_of(tensor_name) + ".weight"
    if weight_name == tensor_name:
        return False  # this tensor IS the anchor; there is no separate anchor to consult
    base_weight = base_hdr.tensor(weight_name)
    fused_weight = fused_hdr.tensor(weight_name)
    if base_weight is None or fused_weight is None:
        return False
    return base_weight.dtype != fused_weight.dtype or base_weight.shape != fused_weight.shape


def _locate(header: SafetensorsHeader, tensor_name: str) -> tuple[FileHeader, TensorEntry] | None:
    """Return the ``(file_header, tensor_entry)`` owning a tensor name, or ``None``."""
    for file_header in header.files:
        entry = file_header.tensors.get(tensor_name)
        if entry is not None:
            return file_header, entry
    return None


def _absolute_range(file_header: FileHeader, entry: TensorEntry) -> tuple[int, int]:
    """Resolve a tensor's absolute, validated ``[start, end)`` byte range in its file."""
    if file_header.header_length is None:
        msg = f"{file_header.filename}: header_length is unknown; cannot locate tensor payload"
        raise ValueError(msg)
    base = 8 + file_header.header_length
    start = base + entry.data_offsets[0]
    end = base + entry.data_offsets[1]
    if start < 0 or end < start:
        msg = f"{file_header.filename}: invalid data_offsets {entry.data_offsets}"
        raise ValueError(msg)
    if file_header.file_size is not None and end > file_header.file_size:
        msg = f"{file_header.filename}: data_offsets {entry.data_offsets} exceed file size"
        raise ValueError(msg)
    return start, end


def _bytes_differ(
    tensor_name: str,
    base: LocalTarget,
    fused: LocalTarget,
    base_hdr: SafetensorsHeader,
    fused_hdr: SafetensorsHeader,
) -> bool:
    """Bounded, chunked byte compare of one tensor's payload across two repos.

    Reads at most ``_COMPARE_CHUNK_BYTES`` from each side per iteration and
    stops at the first differing chunk, so neither a whole large tensor nor
    even a whole chunk pair beyond what is needed is ever retained together.
    """
    base_location = _locate(base_hdr, tensor_name)
    fused_location = _locate(fused_hdr, tensor_name)
    if base_location is None or fused_location is None:  # pragma: no cover - defensive
        # Unreachable via delta_map: _classify only calls this once
        # SafetensorsHeader.tensor() already found the entry, and _locate
        # searches the same `.files` list `.tensor()` does. Guards against a
        # future caller passing a header whose `.tensor()` view and `.files`
        # listing have drifted out of sync with each other.
        msg = f"{tensor_name}: tensor header entry could not be located in its own file listing"
        raise ValueError(msg)
    base_file, base_entry = base_location
    fused_file, fused_entry = fused_location
    base_start, base_end = _absolute_range(base_file, base_entry)
    fused_start, fused_end = _absolute_range(fused_file, fused_entry)
    total_length = base_end - base_start
    if total_length != fused_end - fused_start:
        # Comparability was already gated on identical shape/dtype, so equal
        # stored length is expected; a mismatch here means corrupt offsets.
        msg = f"{tensor_name}: payload length mismatch despite identical shape/dtype"
        raise ValueError(msg)

    offset = 0
    while offset < total_length:
        chunk_length = min(_COMPARE_CHUNK_BYTES, total_length - offset)
        base_chunk = base.read_bytes(
            base_file.filename, offset=base_start + offset, length=chunk_length
        )
        fused_chunk = fused.read_bytes(
            fused_file.filename, offset=fused_start + offset, length=chunk_length
        )
        if base_chunk != fused_chunk:
            return True
        offset += chunk_length
    return False


def _classify(
    tensor_name: str,
    base: LocalTarget,
    fused: LocalTarget,
    base_hdr: SafetensorsHeader,
    fused_hdr: SafetensorsHeader,
    target_prefixes: frozenset[str],
) -> TensorDelta:
    """Classify one base-repository tensor against its fused counterpart."""
    fused_entry = fused_hdr.tensor(tensor_name)
    base_entry = base_hdr.tensor(tensor_name)
    if base_entry is None:  # pragma: no cover - defensive; callers pass only base tensor names
        msg = f"{tensor_name}: not present in the base header"
        raise ValueError(msg)
    is_target = _prefix_of(tensor_name) in target_prefixes

    if fused_entry is None:
        if _weight_representation_changed(tensor_name, base_hdr, fused_hdr):
            return TensorDelta(
                tensor=tensor_name,
                klass="non_comparable",
                reason=(
                    f"{tensor_name} is absent from the fused repository, and its module's "
                    "weight representation changed (quantization is no longer comparable)"
                ),
            )
        return TensorDelta(
            tensor=tensor_name,
            klass="missing",
            reason=(
                f"{tensor_name} is present in the base repository but absent from the fused one"
            ),
        )

    if base_entry.dtype != fused_entry.dtype or base_entry.shape != fused_entry.shape:
        return TensorDelta(
            tensor=tensor_name,
            klass="non_comparable",
            reason=(
                f"representation changed: {base_entry.dtype}{list(base_entry.shape)} in base "
                f"vs {fused_entry.dtype}{list(fused_entry.shape)} in fused"
            ),
        )

    if _bytes_differ(tensor_name, base, fused, base_hdr, fused_hdr):
        return TensorDelta(tensor=tensor_name, klass="changed" if is_target else "unexpected")
    return TensorDelta(tensor=tensor_name, klass="unchanged")


def delta_map(
    base: LocalTarget,
    fused: LocalTarget,
    *,
    base_hdr: SafetensorsHeader,
    fused_hdr: SafetensorsHeader,
    targets: Sequence[str],
) -> list[TensorDelta]:
    """Classify every base-repository tensor against its fused-repository counterpart.

    ``targets`` is the resolver's expected-change set (base module paths,
    e.g. ``"model.layers.0.self_attn.q_proj"`` -- see
    :mod:`mlx_model_doctor.parity.resolver`): a tensor under one of these
    prefixes that byte-differs is the positive proof a fuse applied
    (``changed``); a byte-difference anywhere else is ``unexpected``. Every
    tensor is comparability-gated first (see the module docstring), so a
    uniform representation change (e.g. ``--dequantize``) is reported as
    ``non_comparable`` for targets and non-targets alike, never misread as a
    real ``missing``/``unexpected`` failure.

    Iterates over the tensors the base header lists; a tensor that exists
    only in the fused repository (not present in base at all) is out of
    scope for this map.
    """
    target_prefixes = frozenset(targets)
    return [
        _classify(name, base, fused, base_hdr, fused_hdr, target_prefixes)
        for name in sorted(base_hdr.tensor_names())
    ]


def aggregate_delta_summary(deltas: Sequence[TensorDelta]) -> DeltaSummary:
    """Aggregate a delta map into per-klass counts and tensor names.

    Diagnostic only: this does NOT decide whether the adapter was applied --
    that determination comes from the reference-load logit comparison (F1),
    independently of this static byte map.
    """
    tensors_by_klass: dict[TensorDeltaKlass, list[str]] = {klass: [] for klass in _ALL_KLASSES}
    for delta in deltas:
        tensors_by_klass[delta.klass].append(delta.tensor)
    frozen_tensors = {klass: tuple(names) for klass, names in tensors_by_klass.items()}
    counts = {klass: len(names) for klass, names in frozen_tensors.items()}
    return DeltaSummary(counts_by_klass=counts, tensors_by_klass=frozen_tensors)
