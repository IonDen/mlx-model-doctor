"""Guard: version-bound checks must warn (not fail) for unknown-but-well-typed values."""

import json

import pytest

from mlx_model_doctor.checks.chat_template import ChatTemplatePresenceCheck
from mlx_model_doctor.checks.quantization import MlxQuantizationModeCheck, MlxQuantShapeCheck
from mlx_model_doctor.checks.safetensors import SafetensorsOffsetScanCheck
from mlx_model_doctor.checks.vlm import VlmImageProcessorCheck
from mlx_model_doctor.checks.weights import TiedEmbeddingCheck
from mlx_model_doctor.safetensors_header import FileHeader, SafetensorsHeader, TensorEntry
from tests.fakes import context_for_files

# --- Fixture builders ---


def _quant_mode_unknown_fixture():
    """Config with a well-typed but unknown quantization mode string."""
    config = {"quantization": {"mode": "future_mode_v2", "bits": 4, "group_size": 64}}
    return context_for_files({"config.json": json.dumps(config).encode()})


def _quant_shape_unknown_fixture():
    """Config with well-typed but off-table affine bits."""
    config = {"quantization": {"mode": "affine", "bits": 99, "group_size": 64}}
    files = {"config.json": json.dumps(config).encode()}
    header = SafetensorsHeader(
        files=(
            FileHeader(
                filename="model.safetensors",
                tensors={
                    "model.layers.0.self_attn.q_proj.scales": TensorEntry(
                        dtype="F16",
                        shape=(1, 32),
                        data_offsets=(0, 64),
                        stored_element_count=32,
                    ),
                    "model.layers.0.self_attn.q_proj.weight": TensorEntry(
                        dtype="U32",
                        shape=(32, 2),
                        data_offsets=(64, 320),
                        stored_element_count=64,
                    ),
                },
                metadata={},
                header_length=200,
                file_size=1000,
            ),
        ),
        weight_map={},
        sharded=False,
        stored_count_by_dtype={"F16": 32, "U32": 64},
    )
    ctx = context_for_files(files)
    ctx.target._safetensors_header = header
    return ctx


def _safetensors_dtype_unknown_fixture():
    """Safetensors header with an unknown but well-typed dtype string."""
    header = SafetensorsHeader(
        files=(
            FileHeader(
                filename="model.safetensors",
                tensors={
                    "weight": TensorEntry(
                        dtype="FUTURE_FP6",
                        shape=(10,),
                        data_offsets=(0, 60),
                        stored_element_count=10,
                    ),
                },
                metadata={},
                header_length=100,
                file_size=200,
            ),
        ),
        weight_map={},
        sharded=False,
        stored_count_by_dtype={"FUTURE_FP6": 10},
    )
    ctx = context_for_files({"config.json": b'{"model_type": "llama"}'})
    ctx.target._safetensors_header = header
    return ctx


def _vlm_image_processor_unknown_fixture():
    """VLM-gated repo exposing none of the check's recognized resolution signals.

    ``VlmImageProcessorCheck`` accepts any well-typed, non-empty ``image_processor_type``
    string as-is — it has no closed allow-list of processor class names, so an "unrecognized"
    processor name is not actually a warn case for this check. Its version-bound table is
    instead the *set of recognized resolution signals* (``image_processor_type``, a
    preprocessor-side ``auto_map``, ``feature_extractor_type``, or ``processor_class``),
    verified against mlx-vlm 0.6.x / transformers v4.x. A repo that is VLM-gated (via
    ``vision_config``) but exposes none of those signals is this check's actual
    unknown-but-well-typed case, and is exactly what it warns on.
    """
    config = {"model_type": "some_vlm", "vision_config": {}}
    return context_for_files(
        {"config.json": json.dumps(config).encode()},
        source="hf",
        name="mlx-community/future-vlm",
    )


def _chat_template_absent_fixture():
    """Tokenizer config present but no chat template anywhere."""
    return context_for_files(
        {
            "config.json": b'{"model_type": "llama"}',
            "tokenizer_config.json": b'{"model_max_length": 4096}',
        }
    )


def _tied_embedding_mismatch_fixture():
    """tie_word_embeddings true but both embedding and lm-head stored."""
    config = {"model_type": "llama", "tie_word_embeddings": True}
    header = SafetensorsHeader(
        files=(
            FileHeader(
                filename="model.safetensors",
                tensors={
                    "model.embed_tokens.weight": TensorEntry(
                        dtype="F16",
                        shape=(32000, 4096),
                        data_offsets=(0, 100),
                        stored_element_count=131072000,
                    ),
                    "lm_head.weight": TensorEntry(
                        dtype="F16",
                        shape=(32000, 4096),
                        data_offsets=(100, 200),
                        stored_element_count=131072000,
                    ),
                },
                metadata={},
                header_length=200,
                file_size=1000,
            ),
        ),
        weight_map={},
        sharded=False,
        stored_count_by_dtype={"F16": 262144000},
    )
    ctx = context_for_files({"config.json": json.dumps(config).encode()})
    ctx.target._safetensors_header = header
    return ctx


# --- Registry ---
# Every check with a version-bound allow-list MUST be registered here.
# The fixture must produce a context where the check encounters an unknown-but-well-typed
# value (not a wrong type or missing field) — or, for a heuristic-based check with no closed
# allow-list of values (VlmImageProcessorCheck), a context matching none of its recognized
# resolution signals.

VERSION_BOUND_REGISTRY: list[tuple[object, object, str]] = [
    (MlxQuantizationModeCheck(), _quant_mode_unknown_fixture, "unknown quant mode"),
    (MlxQuantShapeCheck(), _quant_shape_unknown_fixture, "off-table affine bits"),
    (
        SafetensorsOffsetScanCheck(),
        _safetensors_dtype_unknown_fixture,
        "unknown safetensors dtype",
    ),
    (
        VlmImageProcessorCheck(),
        _vlm_image_processor_unknown_fixture,
        "no recognized resolution signal",
    ),
    (ChatTemplatePresenceCheck(), _chat_template_absent_fixture, "absent chat template"),
    (TiedEmbeddingCheck(), _tied_embedding_mismatch_fixture, "tied embedding mismatch"),
]

# The expected set of check classes that have version-bound tables (from the spec inventory).
# A new check with a version-bound table that is not registered here will fail the
# completeness assertion.
EXPECTED_VERSION_BOUND_CLASSES = frozenset(
    {
        "MlxQuantizationModeCheck",
        "MlxQuantShapeCheck",
        "SafetensorsOffsetScanCheck",
        "VlmImageProcessorCheck",
        "ChatTemplatePresenceCheck",
        "TiedEmbeddingCheck",
    }
)


@pytest.mark.parametrize(
    ("check", "fixture_builder", "desc"),
    VERSION_BOUND_REGISTRY,
    ids=[desc for _, _, desc in VERSION_BOUND_REGISTRY],
)
def test_version_bound_checks_warn_not_fail(check, fixture_builder, desc) -> None:
    ctx = fixture_builder()
    result = check.run(ctx)
    assert result.status == "warn", (
        f"{check.__class__.__name__} returned status={result.status!r} for {desc}; "
        f"version-bound checks must warn, not fail, for unknown-but-well-typed values."
    )


def test_version_bound_registry_is_complete() -> None:
    """Every check class in the spec inventory must be registered in the guard."""
    registered_classes = frozenset(
        check.__class__.__name__ for check, _, _ in VERSION_BOUND_REGISTRY
    )
    assert registered_classes == EXPECTED_VERSION_BOUND_CLASSES, (
        f"Registry drift: registered={sorted(registered_classes)}, "
        f"expected={sorted(EXPECTED_VERSION_BOUND_CLASSES)}"
    )
