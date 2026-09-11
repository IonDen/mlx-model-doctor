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
from tests.fakes import FakeTarget, check_options


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


def _fake_target_with_declared_tokenizer_json_size(declared_size: int) -> FakeTarget:
    """A fake target whose tokenizer.json REPORTS ``declared_size`` regardless of its
    actual (small) byte length -- lets a test pin the read-size-cap boundary (B1a)
    without writing a genuinely multi-megabyte file to disk.
    """
    tokenizer_config, tokenizer_json, _ = reference_tokenizer_files()
    return FakeTarget(
        files={
            "tokenizer_config.json": json.dumps(tokenizer_config).encode("utf-8"),
            "tokenizer.json": json.dumps(tokenizer_json).encode("utf-8"),
        },
        size_overrides={"tokenizer.json": declared_size},
    )


def test_tokenizer_json_between_metadata_cap_and_tokenizer_cap_is_still_read() -> None:
    """B1a: a tokenizer.json reported between 16 MiB (the shared metadata cap used
    for config.json/tokenizer_config.json/adapter_config.json) and 64 MiB (the
    tokenizer.json-specific cap) must still be read. This is the real shape of, for
    example, Llama-3.2's ~17 MB tokenizer.json: the old shared 16 MiB cap silently
    treated its vocabulary as unavailable, voiding the adapter-parity oracle for a
    correct fuse. One-line bug this catches: reusing the shared metadata cap for
    ``_read_tokenizer_json`` instead of a larger, tokenizer.json-specific one.
    """
    declared_size = 17 * 1024 * 1024  # bigger than the 16 MiB shared metadata cap
    target = _fake_target_with_declared_tokenizer_json_size(declared_size)
    pctx = ParityContext(
        targets=ParityTargets(base=target, adapter=target, fused=target),
        options=check_options(),
    )

    fp = pctx.base_tokenizer_fingerprint()

    assert fp.vocab_size is not None
    assert fp.token_id_map_digest is not None


def test_tokenizer_json_over_tokenizer_cap_is_still_unavailable() -> None:
    """B1a: the cap increase is bounded, not removed -- a tokenizer.json genuinely
    larger than the new 64 MiB tokenizer.json-specific cap must still read as
    unavailable rather than being read without limit.
    """
    declared_size = 65 * 1024 * 1024  # bigger than the new 64 MiB tokenizer.json cap
    target = _fake_target_with_declared_tokenizer_json_size(declared_size)
    pctx = ParityContext(
        targets=ParityTargets(base=target, adapter=target, fused=target),
        options=check_options(),
    )

    fp = pctx.base_tokenizer_fingerprint()

    assert fp.vocab_size is None
    assert fp.token_id_map_digest is None
