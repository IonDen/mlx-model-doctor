"""Offline tests for ``build_prompts_fixture`` (F9 CLI wiring).

Exercises the CLI-testable seam behind ``parity mlx --prompts``: reading and
validating a user-supplied JSON prompts file, resolving the base model to a
local path, and delegating to the REAL ``build_fixture_from_prompts`` core --
with only the tokenizer loader and (where noted) the fingerprint function
faked, so no transformers/network/MLX is ever touched.
"""

import json
from pathlib import Path

import pytest

from mlx_model_doctor.errors import DependencyError, ModelDoctorError
from mlx_model_doctor.parity import prompts as prompts_module
from mlx_model_doctor.parity.context import TokenizerFingerprint, tokenizer_fingerprint
from mlx_model_doctor.parity.prompts import build_prompts_fixture
from mlx_model_doctor.parity.report import ResolvedIdentity
from mlx_model_doctor.parity.sources import ResolvedSources
from tests.fakes import use_fake_home
from tests.parity_fakes import make_import_module


class _FakeTokenizer:
    """Fake ``AutoTokenizer``-shaped object: only ``apply_chat_template`` is needed."""

    def __init__(self, responses: dict[tuple[str, bool], list[int]]) -> None:
        self._responses = responses
        self.calls: list[tuple[list[dict[str, str]], bool, bool]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
        return_dict: bool,
    ) -> list[int]:
        self.calls.append((messages, add_generation_prompt, tokenize))
        key = (messages[0]["content"], add_generation_prompt)
        return list(self._responses[key])


class _TransformersFiveTokenizer:
    """Models the real ``transformers`` 5.x ``apply_chat_template`` default.

    Verified against the installed transformers 5.15.0: with ``tokenize=True``
    and no explicit ``return_dict`` argument, ``apply_chat_template`` returns a
    dict/``BatchEncoding`` (``{"input_ids": [...], "attention_mask": [...]}``),
    not a flat list of token ids. Only an explicit ``return_dict=False`` yields
    the flat ``list[int]`` ``build_prompts_fixture`` needs.
    """

    def __init__(self, responses: dict[tuple[str, bool], list[int]]) -> None:
        self._responses = responses

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
        return_dict: bool = True,
    ) -> list[int] | dict[str, list[int]]:
        input_ids = list(self._responses[(messages[0]["content"], add_generation_prompt)])
        if return_dict is False:
            return input_ids
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


def _fp() -> TokenizerFingerprint:
    return tokenizer_fingerprint(tokenizer_config=None, tokenizer_json=None, chat_template=None)


def _write_prompts(path: Path, pairs: list[dict[str, str]]) -> None:
    path.write_text(json.dumps(pairs), encoding="utf-8")


def _make_base_dir(tmp_path: Path) -> Path:
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    return base_dir


class TestBuildPromptsFixture:
    def test_valid_prompts_file_builds_the_expected_fixture(self, tmp_path: Path) -> None:
        base_dir = _make_base_dir(tmp_path)
        prompts_path = tmp_path / "prompts.json"
        _write_prompts(prompts_path, [{"prompt": "hi", "completion": "there"}])
        tokenizer = _FakeTokenizer({("hi", True): [1, 2], ("hi", False): [1, 2, 3, 4]})
        fingerprint = _fp()

        ref, sequences = build_prompts_fixture(
            str(base_dir),
            str(prompts_path),
            tokenizer_loader=lambda _path: tokenizer,
            fingerprint_fn=lambda _path: fingerprint,
        )

        assert sequences == ((1, 2, 3, 4),)
        assert ref.id == "user-prompts"
        assert ref.tokenizer_fingerprint is fingerprint

    def test_tokenizer_loader_and_fingerprint_fn_receive_the_resolved_base_path(
        self, tmp_path: Path
    ) -> None:
        base_dir = _make_base_dir(tmp_path)
        prompts_path = tmp_path / "prompts.json"
        _write_prompts(prompts_path, [{"prompt": "hi", "completion": "there"}])
        received: list[tuple[str, str]] = []

        def tokenizer_loader(path: str) -> _FakeTokenizer:
            received.append(("tokenizer", path))
            return _FakeTokenizer({("hi", True): [1], ("hi", False): [1, 2]})

        def fingerprint_fn(path: str) -> TokenizerFingerprint:
            received.append(("fingerprint", path))
            return _fp()

        build_prompts_fixture(
            str(base_dir),
            str(prompts_path),
            tokenizer_loader=tokenizer_loader,
            fingerprint_fn=fingerprint_fn,
        )

        resolved = str(base_dir.resolve())
        assert ("tokenizer", resolved) in received
        assert ("fingerprint", resolved) in received

    def test_missing_prompts_file_raises_model_doctor_error(self, tmp_path: Path) -> None:
        base_dir = _make_base_dir(tmp_path)
        missing_path = tmp_path / "does-not-exist.json"

        with pytest.raises(ModelDoctorError, match="could not read prompts file"):
            build_prompts_fixture(str(base_dir), str(missing_path))

    def test_malformed_json_raises_model_doctor_error(self, tmp_path: Path) -> None:
        base_dir = _make_base_dir(tmp_path)
        prompts_path = tmp_path / "prompts.json"
        prompts_path.write_text("{not valid json", encoding="utf-8")

        with pytest.raises(ModelDoctorError, match="not valid JSON"):
            build_prompts_fixture(str(base_dir), str(prompts_path))

    def test_non_list_json_raises_model_doctor_error(self, tmp_path: Path) -> None:
        base_dir = _make_base_dir(tmp_path)
        prompts_path = tmp_path / "prompts.json"
        prompts_path.write_text(
            json.dumps({"prompt": "hi", "completion": "there"}), encoding="utf-8"
        )

        with pytest.raises(ModelDoctorError, match="non-empty"):
            build_prompts_fixture(str(base_dir), str(prompts_path))

    def test_empty_list_raises_model_doctor_error(self, tmp_path: Path) -> None:
        base_dir = _make_base_dir(tmp_path)
        prompts_path = tmp_path / "prompts.json"
        prompts_path.write_text("[]", encoding="utf-8")

        with pytest.raises(ModelDoctorError, match="non-empty"):
            build_prompts_fixture(str(base_dir), str(prompts_path))

    def test_pair_missing_completion_raises_model_doctor_error(self, tmp_path: Path) -> None:
        base_dir = _make_base_dir(tmp_path)
        prompts_path = tmp_path / "prompts.json"
        _write_prompts(prompts_path, [{"prompt": "hi"}])

        with pytest.raises(ModelDoctorError, match="completion"):
            build_prompts_fixture(str(base_dir), str(prompts_path))

    def test_pair_missing_prompt_raises_model_doctor_error(self, tmp_path: Path) -> None:
        base_dir = _make_base_dir(tmp_path)
        prompts_path = tmp_path / "prompts.json"
        _write_prompts(prompts_path, [{"completion": "there"}])

        with pytest.raises(ModelDoctorError, match="prompt"):
            build_prompts_fixture(str(base_dir), str(prompts_path))

    def test_empty_completion_surfaces_build_fixture_from_prompts_value_error(
        self, tmp_path: Path
    ) -> None:
        base_dir = _make_base_dir(tmp_path)
        prompts_path = tmp_path / "prompts.json"
        _write_prompts(prompts_path, [{"prompt": "hi", "completion": ""}])
        # The completion renders no additional tokens beyond the prompt-only render --
        # build_fixture_from_prompts must reject this, not build_prompts_fixture itself.
        tokenizer = _FakeTokenizer({("hi", True): [1, 2], ("hi", False): [1, 2]})

        with pytest.raises(ValueError, match="completion produced no additional tokens"):
            build_prompts_fixture(
                str(base_dir),
                str(prompts_path),
                tokenizer_loader=lambda _path: tokenizer,
                fingerprint_fn=lambda _path: _fp(),
            )

    def test_base_ref_that_is_not_a_local_directory_raises_a_clear_error(
        self, tmp_path: Path
    ) -> None:
        prompts_path = tmp_path / "prompts.json"
        _write_prompts(prompts_path, [{"prompt": "hi", "completion": "there"}])

        with pytest.raises(ModelDoctorError, match="it is not a local directory"):
            build_prompts_fixture("definitely/not-a-real-directory", str(prompts_path))

    def test_builds_a_real_fixture_against_a_transformers_five_tokenizer(
        self, tmp_path: Path
    ) -> None:
        """A transformers 5.x tokenizer returns a dict by default, not a flat list.

        Catches: ``build_prompts_fixture`` asking for the default
        ``apply_chat_template(..., tokenize=True)`` shape instead of the flat
        ``list[int]`` it needs. Against a real transformers 5.x tokenizer that
        bug renders every prompt as the dict's two keys
        (``["input_ids", "attention_mask"]``) instead of real token ids, so the
        fixture always looks like a 2-token, zero-additional-token completion
        and ``build_fixture_from_prompts`` rejects it as empty -- even though
        the supplied completion is real and non-empty.
        """
        base_dir = _make_base_dir(tmp_path)
        prompts_path = tmp_path / "prompts.json"
        _write_prompts(prompts_path, [{"prompt": "hi", "completion": "there"}])
        tokenizer = _TransformersFiveTokenizer({("hi", True): [1, 2], ("hi", False): [1, 2, 3, 4]})

        ref, sequences = build_prompts_fixture(
            str(base_dir),
            str(prompts_path),
            tokenizer_loader=lambda _path: tokenizer,
            fingerprint_fn=lambda _path: _fp(),
        )

        assert sequences == ((1, 2, 3, 4),)
        assert ref.id == "user-prompts"


class TestDefaultTokenizerLoader:
    def test_raises_dependency_error_when_transformers_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def import_module(_name: str) -> object:
            raise ImportError("missing")

        monkeypatch.setattr(prompts_module.importlib, "import_module", import_module)

        with pytest.raises(DependencyError, match="Install it with") as exc_info:
            prompts_module._default_tokenizer_loader("/some/path")
        assert exc_info.value.missing_package == "transformers"

    def test_pins_trust_remote_code_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The ``--prompts`` tokenizer load must never enable ``trust_remote_code``.

        Catches: the ``--prompts`` tokenizer load enabling remote code execution
        from an untrusted base repo (an omitted or ``True`` pin here reaches
        ``transformers.AutoTokenizer.from_pretrained`` for a repo the user only
        pointed at by path/id, not one they've vetted).
        """
        calls: list[dict[str, object]] = []

        class _FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(path: str, **kwargs: object) -> _FakeTokenizer:
                calls.append({"path": path, **kwargs})
                return _FakeTokenizer({})

        class _FakeTransformers:
            AutoTokenizer = _FakeAutoTokenizer

        monkeypatch.setattr(
            prompts_module.importlib,
            "import_module",
            make_import_module({"transformers": _FakeTransformers()}),
        )

        prompts_module._default_tokenizer_loader("/some/path")

        assert calls == [{"path": "/some/path", "trust_remote_code": False}]


class _RecordingLoaderStopError(Exception):
    """Raised by the recording tokenizer loader once it has seen its call."""


def test_a_base_that_names_loader_code_is_refused_before_the_tokenizer_load(
    tmp_path: Path,
) -> None:
    # Bug: --prompts loads the base tokenizer before any guard runs, so the user gets
    # transformers' error (or a load) instead of the refusal parity gives.
    base_dir = _make_base_dir(tmp_path)
    (base_dir / "config.json").write_text(
        json.dumps({"model_type": "llama", "text_config": {"model_file": "m.py"}}),
        encoding="utf-8",
    )
    prompts_path = tmp_path / "prompts.json"
    _write_prompts(prompts_path, [{"prompt": "hi", "completion": "there"}])
    calls: list[str] = []

    def loader(path: str) -> object:
        calls.append(path)
        raise _RecordingLoaderStopError

    with pytest.raises(ModelDoctorError) as exc_info:
        build_prompts_fixture(str(base_dir), str(prompts_path), tokenizer_loader=loader)  # type: ignore[arg-type]

    assert str(exc_info.value) == (
        "The base repository names loader code (model_file in config.json) that mlx-lm "
        "could import. The load was refused, so that code was not run. Do not load this "
        "repository with mlx-lm or mlx-vlm unless you trust its author. The static checks "
        "read it without running its code."
    )
    assert calls == []


def test_a_base_with_only_an_auto_map_still_loads_its_tokenizer(tmp_path: Path) -> None:
    # Bug: the --prompts pre-check uses the VLM key table and refuses an auto_map that
    # the tokenizer load, with trust_remote_code pinned off, never imports.
    base_dir = _make_base_dir(tmp_path)
    (base_dir / "tokenizer_config.json").write_text(
        json.dumps({"auto_map": {"AutoTokenizer": ["tokenization_x.XTokenizer", None]}}),
        encoding="utf-8",
    )
    prompts_path = tmp_path / "prompts.json"
    _write_prompts(prompts_path, [{"prompt": "hi", "completion": "there"}])
    calls: list[str] = []

    def loader(path: str) -> object:
        calls.append(path)
        raise _RecordingLoaderStopError

    with pytest.raises(_RecordingLoaderStopError):
        build_prompts_fixture(str(base_dir), str(prompts_path), tokenizer_loader=loader)  # type: ignore[arg-type]

    assert calls == [str(base_dir.resolve())]


def test_a_base_that_resolves_to_a_file_is_refused_before_the_tokenizer_load(
    tmp_path: Path,
) -> None:
    # Bug: the --prompts guard treats a resolved base that is not a directory as clean
    # (or words it as a named key, "that code was not run"), so AutoTokenizer receives a
    # path it would read as a Hub repo id.
    base_file = tmp_path / "base.safetensors"
    base_file.write_bytes(b"not a directory")
    prompts_path = tmp_path / "prompts.json"
    _write_prompts(prompts_path, [{"prompt": "hi", "completion": "there"}])
    calls: list[str] = []

    def loader(path: str) -> object:
        calls.append(path)
        raise _RecordingLoaderStopError

    def resolver(*refs: str, **_kwargs: object) -> ResolvedSources:
        identity = ResolvedIdentity(path=str(base_file), original_ref=refs[0], source="local")
        return ResolvedSources(base=identity, adapter=identity, fused=identity)

    with pytest.raises(ModelDoctorError) as exc_info:
        build_prompts_fixture(
            "org/base",
            str(prompts_path),
            tokenizer_loader=loader,  # type: ignore[arg-type]
            resolver=resolver,
        )

    assert str(exc_info.value) == (
        f"{base_file} is not an existing directory, so it is unknown whether the base "
        "repository names loader code. The load was refused, so repository code was not "
        "run. Pass an existing model directory, then run again."
    )
    assert calls == []


def test_a_tokenizer_load_error_names_only_the_base_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: the --prompts tool error (exit 2) prints the base path with the user's home;
    # or the reduced error is no longer a ValueError, which the CLI turns into exit 2.
    home = tmp_path / "Users" / "John Smith"
    base_dir = home / "my models" / "base"
    base_dir.mkdir(parents=True)
    use_fake_home(monkeypatch, home=str(home))
    prompts_path = tmp_path / "prompts.json"
    _write_prompts(prompts_path, [{"prompt": "hi", "completion": "there"}])

    def loader(path: str) -> _FakeTokenizer:
        raise ValueError(f"Unrecognized model in {path}. Check {path}/config.json")

    with pytest.raises(ValueError, match=r"^Unrecognized model in ") as exc_info:
        build_prompts_fixture(
            str(base_dir),
            str(prompts_path),
            tokenizer_loader=loader,
            fingerprint_fn=lambda _path: _fp(),
        )

    assert exc_info.type is ValueError
    assert str(exc_info.value) == "Unrecognized model in base. Check config.json"
