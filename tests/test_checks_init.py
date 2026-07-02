"""Assert `checks.__all__` matches the check classes the `text` plugin actually uses."""

import mlx_model_doctor.checks as checks_pkg


def test_checks_all_matches_exported_check_classes() -> None:
    exported = set(checks_pkg.__all__)
    # every name in __all__ is importable from the package
    for name in exported:
        assert hasattr(checks_pkg, name), name
    # the check classes actually used by the text plugin (plugins/text.py's
    # static_checks() / weight_checks() / smoke_checks() tuples) are all exported
    for name in (
        "ChatTemplatePresenceCheck",
        "ChatTemplateSpecialTokensCheck",
        "ConfigJsonCheck",
        "GenerationConfigTokensCheck",
        "MemoryEstimateCheck",
        "MlxCompatSignalCheck",
        "MlxLmSmokeCheck",
        "MlxQuantizationModeCheck",
        "MlxQuantShapeCheck",
        "ModelTypeCheck",
        "QuantizationMetadataCheck",
        "RequiredConfigCheck",
        "SafetensorsIndexCheck",
        "SafetensorsOffsetScanCheck",
        "SpecialTokensCheck",
        "TiedEmbeddingCheck",
        "TokenizerFilesCheck",
        "VlmImageProcessorCheck",
        "WeightParamCountCheck",
    ):
        assert name in exported, name
