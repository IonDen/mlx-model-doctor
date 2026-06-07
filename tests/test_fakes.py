from mlx_model_doctor.targets import MlxListingMetadata
from tests.fakes import context_for_files


def test_context_for_files_sets_name_and_source():
    ctx = context_for_files({}, source="hf", name="mlx-community/Foo")
    assert ctx.target.name == "mlx-community/Foo"
    assert ctx.target.source == "hf"


def test_fake_target_exposes_listing_metadata():
    ctx = context_for_files(
        {}, source="hf", name="org/m", tags=frozenset({"mlx"}), library_name="mlx-lm"
    )
    assert isinstance(ctx.target, MlxListingMetadata)
    assert ctx.target.tags == frozenset({"mlx"})
    assert ctx.target.library_name == "mlx-lm"


def test_fake_target_defaults_have_empty_listing_metadata():
    ctx = context_for_files({})
    assert ctx.target.tags == frozenset()
    assert ctx.target.library_name is None
