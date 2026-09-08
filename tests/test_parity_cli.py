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

import pytest

from mlx_model_doctor import cli
from mlx_model_doctor.api import ParityOptions
from mlx_model_doctor.api import check_adapter_parity as real_check_adapter_parity
from tests.test_parity_api import (
    FakeLauncher,
    RaisingLauncher,
    _pass_launcher,
    _Repos,
    _tracks_base_launcher,
    tiny_local_repos,
)

__all__ = ["tiny_local_repos"]  # re-exported fixture; keep the import "used" for linters


def _inject_launcher(
    monkeypatch: pytest.MonkeyPatch, launcher: FakeLauncher | RaisingLauncher
) -> None:
    """Wrap the real ``check_adapter_parity`` so it always runs with a fake launcher.

    The CLI's ``_cmd_parity_mlx`` builds its own ``ParityOptions`` from parsed
    args; this replaces only the ``launcher`` field before delegating to the
    real function, so source resolution/checks/delta-map/oracle all run for
    real and only the worker loads are faked.
    """

    def fake_check_adapter_parity(
        *,
        base: str,
        adapter: str,
        fused: str,
        options: ParityOptions | None = None,
    ) -> object:
        fixture_id = options.fixture_id if options is not None else ParityOptions().fixture_id
        return real_check_adapter_parity(
            base=base,
            adapter=adapter,
            fused=fused,
            options=ParityOptions(launcher=launcher, fixture_id=fixture_id),
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
    # A tokenizer mismatch gates the oracle before any worker runs; a launcher
    # whose run_one() raises proves the CLI path never attempts a real load.
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
            tiny_local_repos.fused_bad_tokenizer,
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


def test_parity_command_requires_leaf_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["parity"])

    assert exc_info.value.code == 2
    assert "required" in capsys.readouterr().err


def test_man_command_mentions_parity_mlx(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["man"])
    output = capsys.readouterr().out

    assert code == 0
    assert "parity mlx" in output
