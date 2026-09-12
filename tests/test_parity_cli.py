"""Offline CLI-wiring tests for ``mlx-model-doctor parity mlx`` (F1/F4/F10).

Exercises the real ``check_adapter_parity`` pipeline (source resolution, the
embedded ``text`` reports, the cross-target static checks, the byte-compare
delta map, and the three-way oracle) through the CLI entry point, with only
the worker ``Launcher`` faked -- so the run is fully offline (no MLX, no
subprocess, no real model load) while every other stage runs for real. This
mirrors ``tests/test_parity_api.py``'s injection style (``ParityOptions.launcher``),
applied at the CLI layer by wrapping the real ``check_adapter_parity`` the CLI
module imports.
"""

import json
from pathlib import Path

import pytest

from mlx_model_doctor import cli
from mlx_model_doctor.api import ParityOptions
from mlx_model_doctor.api import check_adapter_parity as real_check_adapter_parity
from mlx_model_doctor.parity.oracle import ROLE_BASE, ROLE_FUSED, ROLE_NOISE, ROLE_REFERENCE
from tests.test_parity_api import (
    FakeLauncher,
    RaisingLauncher,
    _pass_launcher,
    _Repos,
    _tracks_base_launcher,
    tiny_local_repos,
)

__all__ = ["tiny_local_repos"]  # re-exported fixture; keep the import "used" for linters


class _FakeCliTokenizer:
    """Fake tokenizer used to inject a real, but fake-backed, ``--prompts`` fixture."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
        return_dict: bool,
    ) -> list[int]:
        return [100, 101, 102] if add_generation_prompt else [100, 101, 102, 103, 104]


def _inject_prompts_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrap the real ``build_prompts_fixture`` so it always runs with a fake tokenizer.

    Everything else (reading the prompts file, resolving the base to a local
    path, computing the base's own tokenizer fingerprint) runs for real; only
    the tokenizer load is faked, so no transformers/network is ever touched.
    """
    real_build_prompts_fixture = cli.build_prompts_fixture

    def fake_build_prompts_fixture(base_ref: str, prompts_path: str) -> object:
        return real_build_prompts_fixture(
            base_ref, prompts_path, tokenizer_loader=lambda _path: _FakeCliTokenizer()
        )

    monkeypatch.setattr(cli, "build_prompts_fixture", fake_build_prompts_fixture)


def _inject_launcher(
    monkeypatch: pytest.MonkeyPatch, launcher: FakeLauncher | RaisingLauncher
) -> None:
    """Wrap the real ``check_adapter_parity`` so it always runs with a fake launcher.

    The CLI's ``_cmd_parity_mlx`` builds its own ``ParityOptions`` from parsed
    args; this replaces only the ``launcher`` field before delegating to the
    real function, so source resolution/checks/delta-map/oracle all run for
    real and only the worker loads are faked. Both ``fixture_id`` and
    ``fixture`` are forwarded unchanged, so this also covers the ``--prompts``
    path (which sets ``fixture``, not ``fixture_id``).
    """

    def fake_check_adapter_parity(
        *,
        base: str,
        adapter: str,
        fused: str,
        options: ParityOptions | None = None,
    ) -> object:
        opts = options if options is not None else ParityOptions()
        return real_check_adapter_parity(
            base=base,
            adapter=adapter,
            fused=fused,
            options=ParityOptions(
                launcher=launcher, fixture_id=opts.fixture_id, fixture=opts.fixture
            ),
        )

    monkeypatch.setattr(cli, "check_adapter_parity", fake_check_adapter_parity)


def test_parity_mlx_pass_prints_valid_parity_json_and_exits_zero(
    tiny_local_repos: _Repos, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _inject_launcher(monkeypatch, _pass_launcher())

    code = cli.main(
        [
            "parity",
            "mlx",
            "--base",
            tiny_local_repos.base,
            "--adapter",
            tiny_local_repos.adapter,
            "--fused",
            tiny_local_repos.fused_diff,
            "--format",
            "json",
        ]
    )
    data = json.loads(capsys.readouterr().out)

    assert code == 0
    assert data["schema_version"] == "parity/1.0"
    assert data["verdict"] == "pass"


def test_parity_mlx_default_format_is_text(
    tiny_local_repos: _Repos, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _inject_launcher(monkeypatch, _pass_launcher())

    code = cli.main(
        [
            "parity",
            "mlx",
            "--base",
            tiny_local_repos.base,
            "--adapter",
            tiny_local_repos.adapter,
            "--fused",
            tiny_local_repos.fused_diff,
        ]
    )
    output = capsys.readouterr().out

    assert code == 0
    assert "MLX Model Doctor (parity):" in output
    assert "Verdict: pass" in output


def test_parity_mlx_markdown_format(
    tiny_local_repos: _Repos, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _inject_launcher(monkeypatch, _pass_launcher())

    code = cli.main(
        [
            "parity",
            "mlx",
            "--base",
            tiny_local_repos.base,
            "--adapter",
            tiny_local_repos.adapter,
            "--fused",
            tiny_local_repos.fused_diff,
            "--format",
            "markdown",
        ]
    )
    output = capsys.readouterr().out

    assert code == 0
    assert output.startswith("# MLX Model Doctor (parity):")


def test_parity_mlx_determined_bad_exits_one(
    tiny_local_repos: _Repos, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # a copied-base fuse with a genuinely-applied adapter reference is a
    # determined regression (FAIL_TRACKS_BASE) -> exit 1, not 0 or 2.
    _inject_launcher(monkeypatch, _tracks_base_launcher())

    code = cli.main(
        [
            "parity",
            "mlx",
            "--base",
            tiny_local_repos.base,
            "--adapter",
            tiny_local_repos.adapter,
            "--fused",
            tiny_local_repos.fused_same,
            "--format",
            "json",
        ]
    )
    data = json.loads(capsys.readouterr().out)

    assert code == 1
    assert data["verdict"] == "fail_tracks_base"


def test_parity_mlx_never_launches_a_real_worker_when_statically_gated(
    tiny_local_repos: _Repos, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A genuine tokenizer VOCABULARY mismatch gates the oracle before any worker
    # runs; a launcher whose run_one() raises proves the CLI path never attempts
    # a real load. (A chat-template-only mismatch no longer gates -- see
    # tests/test_parity_api.py::test_tokenizer_metadata_only_mismatch_warns_but_runs_the_oracle.)
    launcher = RaisingLauncher()
    _inject_launcher(monkeypatch, launcher)

    code = cli.main(
        [
            "parity",
            "mlx",
            "--base",
            tiny_local_repos.base,
            "--adapter",
            tiny_local_repos.adapter,
            "--fused",
            tiny_local_repos.fused_vocab_mismatch,
            "--format",
            "json",
        ]
    )
    data = json.loads(capsys.readouterr().out)

    assert code == 1
    assert data["verdict"] is None
    assert launcher.model_paths_by_role == {}


def test_parity_mlx_fixture_flag_is_wired(
    tiny_local_repos: _Repos, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def fake_check_adapter_parity(
        *,
        base: str,
        adapter: str,
        fused: str,
        options: ParityOptions | None = None,
    ) -> object:
        captured["fixture_id"] = options.fixture_id if options is not None else None
        return real_check_adapter_parity(
            base=base,
            adapter=adapter,
            fused=fused,
            options=ParityOptions(
                launcher=_pass_launcher(),
                fixture_id=options.fixture_id if options is not None else "default-v1",
            ),
        )

    monkeypatch.setattr(cli, "check_adapter_parity", fake_check_adapter_parity)

    code = cli.main(
        [
            "parity",
            "mlx",
            "--base",
            tiny_local_repos.base,
            "--adapter",
            tiny_local_repos.adapter,
            "--fused",
            tiny_local_repos.fused_diff,
            "--fixture",
            "default-v1",
            "--format",
            "json",
        ]
    )
    capsys.readouterr()

    assert code == 0
    assert captured["fixture_id"] == "default-v1"


def test_parity_mlx_unknown_fixture_exits_cleanly_with_a_clear_message(
    tiny_local_repos: _Repos, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown ``--fixture`` id must surface a clear message, not ``KeyError``'s
    quoted repr (``Error: 'nonexistent'``) and never an uncaught traceback.
    """
    code = cli.main(
        [
            "parity",
            "mlx",
            "--base",
            tiny_local_repos.base,
            "--adapter",
            tiny_local_repos.adapter,
            "--fused",
            tiny_local_repos.fused_diff,
            "--fixture",
            "nonexistent",
        ]
    )
    captured = capsys.readouterr()

    assert code == 2
    assert captured.err == (
        "Error: unknown parity fixture 'nonexistent'; the only built-in fixture is "
        "'default-v1' — use --prompts to build a fixture for your own model\n"
    )


def test_parity_mlx_prompts_flag_builds_a_tokenizer_bound_fixture_and_reaches_workers(
    tiny_local_repos: _Repos,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """--prompts must build a REAL fixture (via the real build_prompts_fixture, with
    only the tokenizer loader faked) whose token ids reach every worker spec -- not
    silently fall back to the default fixture.
    """
    prompts_path = tmp_path / "prompts.json"
    prompts_path.write_text(json.dumps([{"prompt": "hi", "completion": "there"}]), encoding="utf-8")
    _inject_prompts_tokenizer(monkeypatch)
    launcher = _pass_launcher()
    _inject_launcher(monkeypatch, launcher)

    code = cli.main(
        [
            "parity",
            "mlx",
            "--base",
            tiny_local_repos.base,
            "--adapter",
            tiny_local_repos.adapter,
            "--fused",
            tiny_local_repos.fused_diff,
            "--prompts",
            str(prompts_path),
            "--format",
            "json",
        ]
    )
    data = json.loads(capsys.readouterr().out)

    assert code == 0
    assert data["fixture"]["id"] == "user-prompts"
    for role in (ROLE_BASE, ROLE_NOISE, ROLE_REFERENCE, ROLE_FUSED):
        assert launcher.specs_by_role[role].token_ids == ((100, 101, 102, 103, 104),)


def test_parity_mlx_prompts_flag_wins_when_fixture_flag_is_also_given(
    tiny_local_repos: _Repos,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    prompts_path = tmp_path / "prompts.json"
    prompts_path.write_text(json.dumps([{"prompt": "hi", "completion": "there"}]), encoding="utf-8")
    _inject_prompts_tokenizer(monkeypatch)
    _inject_launcher(monkeypatch, _pass_launcher())

    code = cli.main(
        [
            "parity",
            "mlx",
            "--base",
            tiny_local_repos.base,
            "--adapter",
            tiny_local_repos.adapter,
            "--fused",
            tiny_local_repos.fused_diff,
            "--prompts",
            str(prompts_path),
            "--fixture",
            "default-v1",
            "--format",
            "json",
        ]
    )
    data = json.loads(capsys.readouterr().out)

    assert code == 0
    assert data["fixture"]["id"] == "user-prompts"


class _EmptyCompletionTokenizer:
    """Fake tokenizer whose full render never adds tokens beyond the prompt-only
    render -- the real ``build_prompts_fixture``/``build_fixture_from_prompts``
    path raises a bare ``ValueError`` for this shape (an empty completion).
    """

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
        return_dict: bool,
    ) -> list[int]:
        return [100, 101, 102]


def _inject_empty_completion_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    real_build_prompts_fixture = cli.build_prompts_fixture

    def fake_build_prompts_fixture(base_ref: str, prompts_path: str) -> object:
        return real_build_prompts_fixture(
            base_ref, prompts_path, tokenizer_loader=lambda _path: _EmptyCompletionTokenizer()
        )

    monkeypatch.setattr(cli, "build_prompts_fixture", fake_build_prompts_fixture)


def test_parity_mlx_prompts_empty_completion_exits_cleanly_not_a_traceback(
    tiny_local_repos: _Repos,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """A realistic bad ``--prompts`` input (a pair whose completion renders no
    additional tokens) must surface through the tool's clean error path -- a
    printed ``Error: ...`` message and a tool-error exit code -- never an
    uncaught ``ValueError`` traceback from the pure fixture builder.
    """
    prompts_path = tmp_path / "prompts.json"
    prompts_path.write_text(json.dumps([{"prompt": "hi", "completion": ""}]), encoding="utf-8")
    _inject_empty_completion_tokenizer(monkeypatch)

    code = cli.main(
        [
            "parity",
            "mlx",
            "--base",
            tiny_local_repos.base,
            "--adapter",
            tiny_local_repos.adapter,
            "--fused",
            tiny_local_repos.fused_diff,
            "--prompts",
            str(prompts_path),
        ]
    )
    captured = capsys.readouterr()

    assert code == 2
    assert captured.err.startswith("Error:")


def test_parity_command_requires_leaf_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["parity"])

    assert exc_info.value.code == 2
    assert "required" in capsys.readouterr().err


def test_parity_mlx_help_does_not_promise_a_hugging_face_repo_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--base``/``--adapter``/``--fused`` accept only a local directory today (the CLI
    has no network opt-in); the help text must not claim an HF repo id also works.
    """
    with pytest.raises(SystemExit):
        cli.main(["parity", "mlx", "--help"])
    help_text = capsys.readouterr().out

    assert "Hugging Face repo id" not in help_text
    assert "local base model directory" in help_text
    assert "local LoRA adapter directory" in help_text
    assert "local fused model directory" in help_text


def test_man_command_mentions_parity_mlx(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["man"])
    output = capsys.readouterr().out

    assert code == 0
    assert "parity mlx" in output


def test_man_command_mentions_prompts_flag(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["man"])
    output = capsys.readouterr().out

    assert code == 0
    assert "--prompts" in output
