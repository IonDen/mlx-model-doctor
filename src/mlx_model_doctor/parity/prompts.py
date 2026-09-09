"""Build a tokenizer-bound parity fixture from a user-supplied prompts file (F9).

``mlx-model-doctor parity mlx --prompts <file>`` lets a user verify a REAL
model end-to-end instead of trusting the small built-in reference fixture
(:data:`~mlx_model_doctor.parity.fixtures.DEFAULT_FIXTURE_ID`): supply real
``(prompt, completion)`` pairs, and :func:`build_prompts_fixture` resolves the
base model to a local snapshot, loads its own tokenizer, and delegates to
:func:`~mlx_model_doctor.parity.fixtures.build_fixture_from_prompts` -- so the
fixture's ``tokenizer_fingerprint`` is computed from that SAME tokenizer via
:func:`~mlx_model_doctor.parity.context.tokenizer_fingerprint_for_path`, which
is what lets it pass the fixture-vs-base tokenizer gate in
:func:`~mlx_model_doctor.api.check_adapter_parity` for a genuinely matching
repository, rather than being gated as a mismatch.

Every collaborator that isn't pure/offline (the real tokenizer load, the
Hugging Face snapshot resolution) is an injectable seam
(``tokenizer_loader``/``resolver``/``fingerprint_fn``/``downloader``), so this
module is fully testable with fakes and never needs MLX, ``transformers``, or
network access to exercise its own logic. The default ``tokenizer_loader``
lazily imports ``transformers`` -- pulled in transitively by the optional
``mlx-lm`` extra -- so importing this module, or calling
:func:`build_prompts_fixture` with an injected loader, never requires
``transformers`` to be installed.

Resolving the base is deliberately network-gated (``allow_network=False`` by
default, matching the rest of the ``parity mlx`` CLI, which exposes no
network opt-in today): a base that is not an existing local directory raises a
clear :class:`~mlx_model_doctor.errors.ModelDoctorError` instead of silently
downloading a repository.
"""

import importlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from mlx_model_doctor.environment import format_install_hint, has_uv_context
from mlx_model_doctor.errors import DependencyError, ModelDoctorError
from mlx_model_doctor.parity.context import TokenizerFingerprint, tokenizer_fingerprint_for_path
from mlx_model_doctor.parity.fixtures import FixtureRef, build_fixture_from_prompts
from mlx_model_doctor.parity.sources import ResolvedSources, SnapshotDownloader, resolve_sources


class _ChatTemplateTokenizer(Protocol):
    """The minimal tokenizer surface ``build_prompts_fixture`` needs."""

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, add_generation_prompt: bool, tokenize: bool
    ) -> list[int]:
        """Render ``messages`` to token ids, matching the ``transformers`` tokenizer API."""


class _AutoTokenizerClass(Protocol):
    """The ``transformers.AutoTokenizer`` surface ``_default_tokenizer_loader`` needs."""

    @staticmethod
    def from_pretrained(path: str, *, trust_remote_code: bool) -> _ChatTemplateTokenizer:
        """Load a tokenizer from a local directory."""


def _default_tokenizer_loader(path: str) -> _ChatTemplateTokenizer:
    """Lazily load a tokenizer via ``transformers.AutoTokenizer.from_pretrained``.

    The import happens here, not at module top level, so this module -- and
    calling :func:`build_prompts_fixture` with an injected ``tokenizer_loader``
    -- never requires ``transformers`` to be installed. ``transformers`` is
    pulled in transitively by the optional ``mlx-lm`` extra.

    Pins ``trust_remote_code=False`` explicitly: ``path`` may point at an
    unvetted base repo (the whole reason ``--prompts`` exists is to test a
    real, user-supplied model), and the ``transformers`` default for a repo
    that ships custom tokenizer code is an interactive prompt -- never
    trust-on-load.
    """
    try:
        transformers_module = importlib.import_module("transformers")
    except ImportError as exc:
        raise _dependency_error("transformers") from exc
    auto_tokenizer = cast("_AutoTokenizerClass", transformers_module.AutoTokenizer)
    return auto_tokenizer.from_pretrained(path, trust_remote_code=False)


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


def build_prompts_fixture(
    base_ref: str,
    prompts_path: str,
    *,
    allow_network: bool = False,
    downloader: SnapshotDownloader | None = None,
    tokenizer_loader: Callable[[str], _ChatTemplateTokenizer] = _default_tokenizer_loader,
    fingerprint_fn: Callable[[str], TokenizerFingerprint] = tokenizer_fingerprint_for_path,
    resolver: Callable[..., ResolvedSources] = resolve_sources,
) -> tuple[FixtureRef, tuple[tuple[int, ...], ...]]:
    """Build a tokenizer-bound fixture from real ``(prompt, completion)`` examples (F9).

    Reads ``prompts_path`` as a JSON array of ``{"prompt": str, "completion":
    str}`` objects, resolves ``base_ref`` to a local snapshot (a local
    directory is used as-is; a Hugging Face repo id requires
    ``allow_network=True``), loads that snapshot's own tokenizer via
    ``tokenizer_loader``, and delegates to
    :func:`~mlx_model_doctor.parity.fixtures.build_fixture_from_prompts` with a
    fingerprint computed from that SAME local path via ``fingerprint_fn``.

    Raises:
        ModelDoctorError: ``prompts_path`` is missing, not valid JSON, not a
            non-empty array of well-formed pairs, or ``base_ref`` cannot be
            resolved to a local directory without network access.
        ValueError: a pair's completion renders no additional tokens beyond
            the prompt-only render (see ``build_fixture_from_prompts``).
    """
    pairs = _read_prompt_pairs(prompts_path)
    sources = resolver(
        base_ref, base_ref, base_ref, allow_network=allow_network, downloader=downloader
    )
    local_base_path = sources.base.path
    tokenizer = tokenizer_loader(local_base_path)

    def apply_chat_template(
        messages: list[dict[str, str]], *, add_generation_prompt: bool
    ) -> list[int]:
        return list(
            tokenizer.apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt, tokenize=True
            )
        )

    fingerprint = fingerprint_fn(local_base_path)
    return build_fixture_from_prompts(
        apply_chat_template=apply_chat_template,
        pairs=pairs,
        tokenizer_fingerprint=fingerprint,
    )


def _read_prompt_pairs(prompts_path: str) -> list[tuple[str, str]]:
    """Read and validate ``prompts_path`` into a list of ``(prompt, completion)`` pairs."""
    path = Path(prompts_path)
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelDoctorError(f"could not read prompts file {prompts_path!r}: {exc}") from exc
    try:
        parsed: object = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ModelDoctorError(f"prompts file {prompts_path!r} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, list) or not parsed:
        raise ModelDoctorError(
            f"prompts file {prompts_path!r} must be a non-empty JSON array of "
            '{"prompt": <str>, "completion": <str>} objects'
        )
    pairs: list[tuple[str, str]] = []
    for index, item in enumerate(parsed):
        prompt = item.get("prompt") if isinstance(item, dict) else None
        completion = item.get("completion") if isinstance(item, dict) else None
        if not isinstance(prompt, str) or not isinstance(completion, str):
            raise ModelDoctorError(
                f"prompts file {prompts_path!r}: item {index} must be an object with "
                'string "prompt" and "completion" fields'
            )
        pairs.append((prompt, completion))
    return pairs
