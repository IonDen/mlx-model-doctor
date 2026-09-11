"""Tests for the quant-config-gated byte-compare delta map (F11).

Each test builds a tiny, REAL two-repo (base + fused) safetensors pair via
``tiny_repos(kind=...)`` and asserts the exact tensor name + classification
``delta_map`` produces for it — never just "a delta exists somewhere". The
comparability-gate cases (``non_comparable`` vs the ``missing``-quant-
auxiliary case) are the crux of finding F11: a whole-module representation
change (e.g. a ``--dequantize`` fuse) must never be misread as a genuine
``missing``/``unexpected`` failure, and a genuinely dropped quantization
auxiliary (with its ``.weight`` anchor unchanged) must never be waved away as
a benign representation change.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from mlx_model_doctor.parity.deltamap import (
    TensorDelta,
    _absolute_range,
    _bytes_differ,
    _locate,
    aggregate_delta_summary,
    delta_map,
)
from mlx_model_doctor.safetensors_header import FileHeader, SafetensorsHeader, TensorEntry
from mlx_model_doctor.targets import LocalTarget
from tests.parity_fakes import write_safetensors_file

_TARGET_PREFIX = "model.layers.0.self_attn.q_proj"
_OTHER_PREFIX = "model.layers.0.mlp.gate_proj"

# Real, fixed-length dummy payloads. Shapes/dtypes below are chosen only to be
# internally consistent (byte length == prod(shape) * dtype width); their
# quantization semantics are not exercised, only dtype/shape/byte equality.
_BASE_WEIGHT = bytes(range(16))  # U32, shape (2, 2): 4 * 4 bytes
_CHANGED_WEIGHT = bytes(range(16, 32))  # same dtype/shape, different bytes
_BASE_SCALES = bytes([1, 2, 3, 4])  # F16, shape (2, 1): 2 * 2 bytes
_BASE_BIASES = bytes([5, 6, 7, 8])  # F16, shape (2, 1): 2 * 2 bytes
_DEQUANT_WEIGHT = bytes([9] * 8)  # F16, shape (2, 2): 4 * 2 bytes -- unpacked float

TensorSpec = tuple[str, tuple[int, ...], bytes]


def _module_tensors(
    prefix: str,
    *,
    weight: bytes,
    scales: bytes | None,
    biases: bytes | None,
    weight_dtype: str = "U32",
    weight_shape: tuple[int, ...] = (2, 2),
) -> dict[str, TensorSpec]:
    """Build the ``.weight``(+``.scales``+``.biases``) tensor specs for one module."""
    tensors: dict[str, TensorSpec] = {f"{prefix}.weight": (weight_dtype, weight_shape, weight)}
    if scales is not None:
        tensors[f"{prefix}.scales"] = ("F16", (2, 1), scales)
    if biases is not None:
        tensors[f"{prefix}.biases"] = ("F16", (2, 1), biases)
    return tensors


def _quantized_module(prefix: str, *, weight: bytes = _BASE_WEIGHT) -> dict[str, TensorSpec]:
    """A "fully quantized, unchanged" module: weight + scales + biases."""
    return _module_tensors(prefix, weight=weight, scales=_BASE_SCALES, biases=_BASE_BIASES)


@dataclass(frozen=True, slots=True, kw_only=True)
class TinyRepoPair:
    """Pinned return shape for ``tiny_repos``: a ready-to-compare base/fused pair."""

    base: LocalTarget
    fused: LocalTarget
    base_hdr: SafetensorsHeader
    fused_hdr: SafetensorsHeader
    targets: list[str]


def _build_kind(tmp_path: Path, kind: str) -> TinyRepoPair:
    base_dir = tmp_path / f"{kind}-base"
    fused_dir = tmp_path / f"{kind}-fused"
    base_dir.mkdir()
    fused_dir.mkdir()

    base_tensors: dict[str, TensorSpec] = {
        **_quantized_module(_TARGET_PREFIX),
        **_quantized_module(_OTHER_PREFIX),
    }

    fused_tensors: dict[str, TensorSpec]
    if kind == "changed":
        # Correct fuse: the target is re-quantized (same dtype/shape, new
        # bytes); everything else is byte-identical.
        fused_tensors = {
            **_quantized_module(_TARGET_PREFIX, weight=_CHANGED_WEIGHT),
            **_quantized_module(_OTHER_PREFIX),
        }
    elif kind == "unchanged":
        fused_tensors = dict(base_tensors)
    elif kind == "missing":
        # The target module vanished from the fused repo entirely.
        fused_tensors = {**_quantized_module(_OTHER_PREFIX)}
    elif kind == "unexpected":
        # A non-target module changed bytes it should not have.
        fused_tensors = {
            **_quantized_module(_TARGET_PREFIX),
            **_quantized_module(_OTHER_PREFIX, weight=_CHANGED_WEIGHT),
        }
    elif kind == "non_comparable":
        # A --dequantize-shaped fuse: EVERY module (target and non-target
        # alike) becomes a plain float weight with no scales/biases.
        fused_tensors = {
            **_module_tensors(
                _TARGET_PREFIX,
                weight=_DEQUANT_WEIGHT,
                scales=None,
                biases=None,
                weight_dtype="F16",
                weight_shape=(2, 2),
            ),
            **_module_tensors(
                _OTHER_PREFIX,
                weight=_DEQUANT_WEIGHT,
                scales=None,
                biases=None,
                weight_dtype="F16",
                weight_shape=(2, 2),
            ),
        }
    elif kind == "missing_aux":
        # The target's .weight is untouched (still quantized, identical
        # bytes) but its .scales auxiliary was dropped -- a real bug, not a
        # representation change.
        fused_tensors = {
            **_module_tensors(
                _TARGET_PREFIX, weight=_BASE_WEIGHT, scales=None, biases=_BASE_BIASES
            ),
            **_quantized_module(_OTHER_PREFIX),
        }
    elif kind == "suffixless":
        # A tensor whose name carries none of the .weight/.scales/.biases
        # suffixes (e.g. a raw buffer) must still classify correctly: its
        # own full name is its "prefix", so target-membership and the
        # byte-compare gate both still apply normally.
        base_tensors = {**base_tensors, "some_buffer": ("F32", (1,), bytes([1, 2, 3, 4]))}
        fused_tensors = {
            **_quantized_module(_TARGET_PREFIX),
            **_quantized_module(_OTHER_PREFIX),
            "some_buffer": ("F32", (1,), bytes([9, 9, 9, 9])),
        }
    else:
        raise ValueError(f"unknown tiny_repos kind: {kind!r}")

    write_safetensors_file(base_dir / "model.safetensors", base_tensors)
    write_safetensors_file(fused_dir / "model.safetensors", fused_tensors)

    base = LocalTarget(base_dir)
    fused = LocalTarget(fused_dir)
    base_hdr = base.safetensors_header()
    fused_hdr = fused.safetensors_header()
    assert base_hdr is not None
    assert fused_hdr is not None
    return TinyRepoPair(
        base=base, fused=fused, base_hdr=base_hdr, fused_hdr=fused_hdr, targets=[_TARGET_PREFIX]
    )


@pytest.fixture
def tiny_repos(tmp_path: Path) -> Callable[[str], TinyRepoPair]:
    """Return a factory building a tiny, real base/fused safetensors repo pair by ``kind``."""

    def _build(kind: str) -> TinyRepoPair:
        return _build_kind(tmp_path, kind)

    return _build


def _find(deltas: list[TensorDelta], tensor: str) -> TensorDelta:
    """Look up one tensor's delta by exact name, failing loudly if it's absent."""
    for delta in deltas:
        if delta.tensor == tensor:
            return delta
    raise AssertionError(
        f"no delta produced for tensor {tensor!r}; got {[d.tensor for d in deltas]}"
    )


class TestChangedIsThePositiveProof:
    """A correctly re-quantized target (same dtype/shape, new bytes) is ``changed``."""

    def test_target_weight_with_different_bytes_is_changed(
        self, tiny_repos: Callable[[str], TinyRepoPair]
    ) -> None:
        pair = tiny_repos("changed")
        deltas = delta_map(
            pair.base,
            pair.fused,
            base_hdr=pair.base_hdr,
            fused_hdr=pair.fused_hdr,
            targets=pair.targets,
        )
        delta = _find(deltas, f"{_TARGET_PREFIX}.weight")
        assert delta.tensor == f"{_TARGET_PREFIX}.weight"
        assert delta.klass == "changed"
        assert delta.is_target is True

    def test_non_target_tensor_records_is_target_false(
        self, tiny_repos: Callable[[str], TinyRepoPair]
    ) -> None:
        pair = tiny_repos("changed")
        deltas = delta_map(
            pair.base,
            pair.fused,
            base_hdr=pair.base_hdr,
            fused_hdr=pair.fused_hdr,
            targets=pair.targets,
        )
        assert _find(deltas, f"{_OTHER_PREFIX}.weight").is_target is False

    def test_untouched_target_auxiliaries_stay_unchanged_alongside_the_change(
        self, tiny_repos: Callable[[str], TinyRepoPair]
    ) -> None:
        pair = tiny_repos("changed")
        deltas = delta_map(
            pair.base,
            pair.fused,
            base_hdr=pair.base_hdr,
            fused_hdr=pair.fused_hdr,
            targets=pair.targets,
        )
        assert _find(deltas, f"{_TARGET_PREFIX}.scales").klass == "unchanged"
        assert _find(deltas, f"{_OTHER_PREFIX}.weight").klass == "unchanged"


class TestUnchangedTargetVerbatim:
    """A target with byte-identical weights in both repos is ``unchanged``."""

    def test_target_weight_with_identical_bytes_is_unchanged(
        self, tiny_repos: Callable[[str], TinyRepoPair]
    ) -> None:
        pair = tiny_repos("unchanged")
        deltas = delta_map(
            pair.base,
            pair.fused,
            base_hdr=pair.base_hdr,
            fused_hdr=pair.fused_hdr,
            targets=pair.targets,
        )
        delta = _find(deltas, f"{_TARGET_PREFIX}.weight")
        assert delta.tensor == f"{_TARGET_PREFIX}.weight"
        assert delta.klass == "unchanged"


class TestMissingTargetAbsent:
    """A target module dropped entirely from the fused repo is ``missing``."""

    def test_target_weight_absent_from_fused_is_missing(
        self, tiny_repos: Callable[[str], TinyRepoPair]
    ) -> None:
        pair = tiny_repos("missing")
        deltas = delta_map(
            pair.base,
            pair.fused,
            base_hdr=pair.base_hdr,
            fused_hdr=pair.fused_hdr,
            targets=pair.targets,
        )
        delta = _find(deltas, f"{_TARGET_PREFIX}.weight")
        assert delta.tensor == f"{_TARGET_PREFIX}.weight"
        assert delta.klass == "missing"
        assert delta.reason is not None
        assert delta.is_target is True


class TestUnexpectedNonTargetDiffers:
    """A byte-comparable non-target tensor that changed is ``unexpected``, not ``changed``."""

    def test_non_target_weight_with_different_bytes_is_unexpected(
        self, tiny_repos: Callable[[str], TinyRepoPair]
    ) -> None:
        pair = tiny_repos("unexpected")
        deltas = delta_map(
            pair.base,
            pair.fused,
            base_hdr=pair.base_hdr,
            fused_hdr=pair.fused_hdr,
            targets=pair.targets,
        )
        delta = _find(deltas, f"{_OTHER_PREFIX}.weight")
        assert delta.tensor == f"{_OTHER_PREFIX}.weight"
        assert delta.klass == "unexpected"

    def test_target_itself_stays_unchanged_in_this_scenario(
        self, tiny_repos: Callable[[str], TinyRepoPair]
    ) -> None:
        pair = tiny_repos("unexpected")
        deltas = delta_map(
            pair.base,
            pair.fused,
            base_hdr=pair.base_hdr,
            fused_hdr=pair.fused_hdr,
            targets=pair.targets,
        )
        assert _find(deltas, f"{_TARGET_PREFIX}.weight").klass == "unchanged"


class TestNonComparableWholeModelDequantize:
    """A --dequantize-shaped fuse (targets AND non-targets lose .scales/.biases,
    weight dtype changes) is ``non_comparable`` everywhere -- never ``missing``
    or ``unexpected`` (F11's central distinction)."""

    def test_target_weight_dtype_change_is_non_comparable(
        self, tiny_repos: Callable[[str], TinyRepoPair]
    ) -> None:
        pair = tiny_repos("non_comparable")
        deltas = delta_map(
            pair.base,
            pair.fused,
            base_hdr=pair.base_hdr,
            fused_hdr=pair.fused_hdr,
            targets=pair.targets,
        )
        delta = _find(deltas, f"{_TARGET_PREFIX}.weight")
        assert delta.tensor == f"{_TARGET_PREFIX}.weight"
        assert delta.klass == "non_comparable"
        assert delta.reason is not None
        # An on-target non_comparable tensor is the case the text renderer's
        # notable-tensor filter must still surface (a --dequantize fuse that
        # hit a target is still worth a human's attention).
        assert delta.is_target is True

    def test_non_target_weight_dtype_change_is_also_non_comparable(
        self, tiny_repos: Callable[[str], TinyRepoPair]
    ) -> None:
        pair = tiny_repos("non_comparable")
        deltas = delta_map(
            pair.base,
            pair.fused,
            base_hdr=pair.base_hdr,
            fused_hdr=pair.fused_hdr,
            targets=pair.targets,
        )
        # This is the crux of F11: a non-target whose representation changed
        # must NOT be reported as "unexpected" just because a naive
        # implementation tried a raw byte compare against a differently
        # shaped/typed payload.
        delta = _find(deltas, f"{_OTHER_PREFIX}.weight")
        assert delta.tensor == f"{_OTHER_PREFIX}.weight"
        assert delta.klass == "non_comparable"
        assert delta.is_target is False

    def test_vanished_scales_are_non_comparable_not_missing(
        self, tiny_repos: Callable[[str], TinyRepoPair]
    ) -> None:
        pair = tiny_repos("non_comparable")
        deltas = delta_map(
            pair.base,
            pair.fused,
            base_hdr=pair.base_hdr,
            fused_hdr=pair.fused_hdr,
            targets=pair.targets,
        )
        # The .scales tensor is present in base and absent from fused -- the
        # SAME raw signal as the missing-auxiliary case below. Only the
        # comparability gate (the weight anchor's dtype/shape changed) tells
        # them apart: here it must be non_comparable, never missing.
        delta = _find(deltas, f"{_TARGET_PREFIX}.scales")
        assert delta.tensor == f"{_TARGET_PREFIX}.scales"
        assert delta.klass == "non_comparable"


class TestMissingQuantAuxiliaryIsARealFailure:
    """A dropped .scales with an UNCHANGED .weight anchor is a real failure
    (``missing``), never waved away as ``non_comparable`` (F11)."""

    def test_dropped_scales_with_unchanged_weight_is_missing(
        self, tiny_repos: Callable[[str], TinyRepoPair]
    ) -> None:
        pair = tiny_repos("missing_aux")
        deltas = delta_map(
            pair.base,
            pair.fused,
            base_hdr=pair.base_hdr,
            fused_hdr=pair.fused_hdr,
            targets=pair.targets,
        )
        delta = _find(deltas, f"{_TARGET_PREFIX}.scales")
        assert delta.tensor == f"{_TARGET_PREFIX}.scales"
        assert delta.klass == "missing"
        assert delta.reason is not None

    def test_the_still_present_weight_and_biases_classify_normally(
        self, tiny_repos: Callable[[str], TinyRepoPair]
    ) -> None:
        pair = tiny_repos("missing_aux")
        deltas = delta_map(
            pair.base,
            pair.fused,
            base_hdr=pair.base_hdr,
            fused_hdr=pair.fused_hdr,
            targets=pair.targets,
        )
        # Proves the weight anchor itself is unaffected by its sibling's
        # disappearance: same dtype/shape/bytes -> unchanged, not swept into
        # non_comparable alongside .scales.
        assert _find(deltas, f"{_TARGET_PREFIX}.weight").klass == "unchanged"
        assert _find(deltas, f"{_TARGET_PREFIX}.biases").klass == "unchanged"


class TestAggregateDeltaSummaryIsDiagnosticOnly:
    """``aggregate_delta_summary`` rolls up counts/names per klass; it never
    decides adapter_applied (that verdict is the reference-load's, F1)."""

    def test_counts_and_tensor_names_are_grouped_by_klass(
        self, tiny_repos: Callable[[str], TinyRepoPair]
    ) -> None:
        pair = tiny_repos("changed")
        deltas = delta_map(
            pair.base,
            pair.fused,
            base_hdr=pair.base_hdr,
            fused_hdr=pair.fused_hdr,
            targets=pair.targets,
        )
        summary = aggregate_delta_summary(deltas)
        assert summary.counts_by_klass["changed"] == 1
        assert f"{_TARGET_PREFIX}.weight" in summary.tensors_by_klass["changed"]
        assert summary.counts_by_klass["unchanged"] == len(deltas) - 1
        # No verdict field exists on the summary at all -- it is a pure count/
        # name rollup, not an adapter_applied decision.
        assert not hasattr(summary, "adapter_applied")


class TestSuffixlessTensorClassifiesByItsOwnFullName:
    """A tensor with none of the .weight/.scales/.biases suffixes uses its own
    full name as its "prefix" -- target-membership and the byte-compare gate
    both still apply normally (exercises ``_prefix_of``'s no-suffix path)."""

    def test_suffixless_non_target_buffer_that_differs_is_unexpected(
        self, tiny_repos: Callable[[str], TinyRepoPair]
    ) -> None:
        pair = tiny_repos("suffixless")
        deltas = delta_map(
            pair.base,
            pair.fused,
            base_hdr=pair.base_hdr,
            fused_hdr=pair.fused_hdr,
            targets=pair.targets,
        )
        delta = _find(deltas, "some_buffer")
        assert delta.tensor == "some_buffer"
        assert delta.klass == "unexpected"


class TestLocateReturnsNoneForAnAbsentTensor:
    """``_locate`` is a plain lookup: a name absent from every file returns
    ``None`` rather than raising, so callers can distinguish "absent" from
    "found in this file" without a try/except."""

    def test_absent_tensor_name_returns_none(self) -> None:
        file_header = FileHeader(
            filename="m.safetensors",
            tensors={
                "w": TensorEntry(
                    dtype="F32", shape=(1,), data_offsets=(0, 4), stored_element_count=1
                )
            },
            metadata={},
            header_length=0,
            file_size=None,
        )
        header = SafetensorsHeader(
            files=(file_header,),
            weight_map={"w": "m.safetensors"},
            sharded=False,
            stored_count_by_dtype={"F32": 1},
        )
        assert _locate(header, "does.not.exist") is None


class TestCorruptOffsetsFailSafeRatherThanMisread:
    """``_absolute_range``/``_bytes_differ`` validate offsets so corrupt header
    data raises loudly rather than silently reading garbage or misreporting a
    tensor pair as comparable when it is not."""

    def test_inverted_data_offsets_raise(self) -> None:
        file_header = FileHeader(
            filename="m.safetensors", tensors={}, metadata={}, header_length=10, file_size=1000
        )
        bad_entry = TensorEntry(
            dtype="F32", shape=(1,), data_offsets=(5, 2), stored_element_count=1
        )
        with pytest.raises(ValueError, match="invalid data_offsets"):
            _absolute_range(file_header, bad_entry)

    def test_offsets_exceeding_file_size_raise(self) -> None:
        file_header = FileHeader(
            filename="m.safetensors", tensors={}, metadata={}, header_length=10, file_size=20
        )
        bad_entry = TensorEntry(
            dtype="F32", shape=(100,), data_offsets=(0, 1000), stored_element_count=100
        )
        with pytest.raises(ValueError, match="exceed file size"):
            _absolute_range(file_header, bad_entry)

    def test_missing_header_length_raises(self) -> None:
        file_header = FileHeader(
            filename="m.safetensors", tensors={}, metadata={}, header_length=None, file_size=20
        )
        entry = TensorEntry(dtype="F32", shape=(1,), data_offsets=(0, 4), stored_element_count=1)
        with pytest.raises(ValueError, match="header_length is unknown"):
            _absolute_range(file_header, entry)

    def test_payload_length_mismatch_despite_matching_shape_dtype_raises(
        self, tmp_path: Path
    ) -> None:
        # Same declared dtype/shape (so the comparability gate would pass)
        # but internally-inconsistent data_offsets spans -- a corrupt header,
        # not a legitimate representation change.
        base_dir = tmp_path / "corrupt-base"
        fused_dir = tmp_path / "corrupt-fused"
        base_dir.mkdir()
        fused_dir.mkdir()
        base = LocalTarget(base_dir)
        fused = LocalTarget(fused_dir)
        base_file = FileHeader(
            filename="m.safetensors",
            tensors={
                "w": TensorEntry(
                    dtype="F32", shape=(1,), data_offsets=(0, 4), stored_element_count=1
                )
            },
            metadata={},
            header_length=0,
            file_size=None,
        )
        fused_file = FileHeader(
            filename="m.safetensors",
            tensors={
                "w": TensorEntry(
                    dtype="F32", shape=(1,), data_offsets=(0, 8), stored_element_count=1
                )
            },
            metadata={},
            header_length=0,
            file_size=None,
        )
        base_hdr = SafetensorsHeader(
            files=(base_file,),
            weight_map={"w": "m.safetensors"},
            sharded=False,
            stored_count_by_dtype={"F32": 1},
        )
        fused_hdr = SafetensorsHeader(
            files=(fused_file,),
            weight_map={"w": "m.safetensors"},
            sharded=False,
            stored_count_by_dtype={"F32": 1},
        )
        with pytest.raises(ValueError, match="payload length mismatch"):
            _bytes_differ("w", base, fused, base_hdr, fused_hdr)
