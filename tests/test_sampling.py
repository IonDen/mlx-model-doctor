"""Tests for sampling helpers."""

import pytest

from mlx_model_doctor.sampling import candidate_signal


class _Cand:
    def __init__(self, id, tags=(), library_name=None):
        self.id = id
        self.tags = tags
        self.library_name = library_name


@pytest.mark.parametrize(
    ("cand", "expected"),
    [
        (_Cand("org/m", tags=["mlx"]), "tag:mlx"),
        (_Cand("org/m", library_name="mlx-lm"), "library:mlx-lm"),
        (_Cand("org/m", library_name="mlx"), "library:mlx"),
        (_Cand("mlx-community/Foo"), "author:mlx-community"),
        (_Cand("someorg/Llama-4bit"), "repo-name"),
        (_Cand("someorg/plain"), None),
        (_Cand("mlx-community/Foo-4bit", tags=["mlx"]), "tag:mlx"),
    ],
)
def test_candidate_signal_preserves_outputs(cand, expected):
    assert candidate_signal(cand) == expected
