"""Tests for ``tokenizer_fingerprint_for_path`` (F9 CLI wiring).

The whole point of this helper is that its result is IDENTICAL to what a real
parity run computes for a base target via
``ParityContext.base_tokenizer_fingerprint`` -- that identity is what lets a
fixture built against a local path (see
:mod:`mlx_model_doctor.parity.prompts`) pass the fixture-vs-base tokenizer
gate for a genuinely matching repository, instead of gating a valid run by
accident.
"""

import json
from pathlib import Path

from mlx_model_doctor.parity.context import (
    ParityContext,
    ParityTargets,
    tokenizer_fingerprint_for_path,
)
from mlx_model_doctor.parity.fixtures import reference_tokenizer_files
from mlx_model_doctor.targets import LocalTarget
from tests.fakes import check_options


def _write_tokenizer_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    tokenizer_config, tokenizer_json, _ = reference_tokenizer_files()
    (root / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    (root / "tokenizer.json").write_text(json.dumps(tokenizer_json), encoding="utf-8")
    return root


def test_matches_parity_context_base_tokenizer_fingerprint_for_the_same_files(
    tmp_path: Path,
) -> None:
    root = _write_tokenizer_repo(tmp_path / "repo")

    direct = tokenizer_fingerprint_for_path(str(root))

    pctx = ParityContext(
        targets=ParityTargets(
            base=LocalTarget(root), adapter=LocalTarget(root), fused=LocalTarget(root)
        ),
        options=check_options(),
    )
    assert direct == pctx.base_tokenizer_fingerprint()


def test_produces_a_complete_fingerprint_for_a_well_formed_tokenizer(tmp_path: Path) -> None:
    root = _write_tokenizer_repo(tmp_path / "repo")

    fp = tokenizer_fingerprint_for_path(str(root))

    assert fp.vocab_size is not None
    assert fp.token_id_map_digest is not None
    assert fp.special_tokens_digest is not None
    assert fp.chat_template_digest is not None


def test_missing_tokenizer_files_yield_a_fingerprint_with_unavailable_components(
    tmp_path: Path,
) -> None:
    root = tmp_path / "empty_repo"
    root.mkdir()

    fp = tokenizer_fingerprint_for_path(str(root))

    assert fp.vocab_size is None
    assert fp.token_id_map_digest is None
    assert fp.special_tokens_digest is None
    assert fp.chat_template_digest is None
