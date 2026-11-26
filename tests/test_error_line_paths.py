"""The ``Error:`` line names the folder or file, not the full path.

A missing ``check local`` folder, a ``parity mlx`` reference that is not a folder, and
a ``--prompts`` file that cannot be used are reported by name. ``TargetError.target``
keeps the path the caller passed.
"""

import json
from pathlib import Path

import pytest

from mlx_model_doctor import cli
from mlx_model_doctor.api import check_local_model
from mlx_model_doctor.errors import ModelDoctorError, TargetError
from mlx_model_doctor.parity.prompts import build_prompts_fixture
from mlx_model_doctor.parity.report import ResolvedIdentity
from mlx_model_doctor.parity.sources import ResolvedSources
from tests.fakes import use_fake_home
from tests.parity_fakes import write_parity_adapter, write_parity_model_repo

HOME_YOU = "/Users/you"


def _real_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "Users" / "you"
    home.mkdir(parents=True)
    use_fake_home(monkeypatch, home=str(home))
    return home


def _parity_args(tmp_path: Path, *, base: str | None = None) -> list[str]:
    real_base = write_parity_model_repo(tmp_path / "base")
    adapter = write_parity_adapter(tmp_path / "adapter")
    fused = write_parity_model_repo(tmp_path / "fused", fused=True)
    return [
        "parity",
        "mlx",
        "--base",
        base if base is not None else str(real_base),
        "--adapter",
        str(adapter),
        "--fused",
        str(fused),
    ]


# --- check local (targets.py) ---------------------------------------------------------


def test_missing_local_folder_is_named_by_its_folder(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_fake_home(monkeypatch, home=HOME_YOU)

    code = cli.main(["check", "local", "/Users/you/models/typo"])

    assert code == 2
    assert capsys.readouterr().err == "Error: Local target must be an existing directory: typo\n"


def test_missing_local_folder_with_a_space_is_named_by_its_folder(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_fake_home(monkeypatch, home=HOME_YOU)

    code = cli.main(["check", "local", "/Users/you/my models/typo"])

    assert code == 2
    assert capsys.readouterr().err == "Error: Local target must be an existing directory: typo\n"


def test_local_target_that_is_a_file_is_named_by_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = _real_home(tmp_path, monkeypatch)
    config = home / "m" / "config.json"
    config.parent.mkdir()
    config.write_text("{}", encoding="utf-8")

    code = cli.main(["check", "local", str(config)])

    assert code == 2
    assert (
        capsys.readouterr().err
        == "Error: Local target must be an existing directory: config.json\n"
    )


def test_relative_local_target_stays_as_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_fake_home(monkeypatch, home=HOME_YOU)
    monkeypatch.chdir(tmp_path)

    code = cli.main(["check", "local", "models/typo-rel"])

    assert code == 2
    assert capsys.readouterr().err == (
        "Error: Local target must be an existing directory: models/typo-rel\n"
    )


def test_missing_cache_snapshot_keeps_its_revision(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Bug caught: a missing snapshot is named only by its org/name, so the line no
    # longer says which revision was asked for.
    use_fake_home(monkeypatch, home=HOME_YOU)

    code = cli.main(
        ["check", "local", "~/.cache/huggingface/hub/models--org--name/snapshots/deadbeef0"]
    )

    assert code == 2
    assert capsys.readouterr().err == (
        "Error: Local target must be an existing directory: org/name (cache snapshot deadbee)\n"
    )


def test_unknown_tilde_user_local_target_is_an_error_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Bug caught: the home lookup for an unknown user raises out of the target as a
    # raw RuntimeError with a traceback.
    use_fake_home(monkeypatch, home=HOME_YOU)

    code = cli.main(["check", "local", "~nosuchuser/m"])

    assert code == 2
    err = capsys.readouterr().err
    assert err == "Error: Local target must be an existing directory: ~nosuchuser/m\n"
    assert "Traceback" not in err


def test_str_target_error_message_names_the_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_home(monkeypatch, home=HOME_YOU)

    with pytest.raises(TargetError) as caught:
        check_local_model("/Users/you/models/typo")

    assert str(caught.value) == "Local target must be an existing directory: typo"


def test_target_error_keeps_the_path_for_python_callers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_home(monkeypatch, home=HOME_YOU)

    with pytest.raises(TargetError) as caught:
        check_local_model("/Users/you/models/typo")

    assert caught.value.target == "/Users/you/models/typo"


def test_target_error_keeps_an_unknown_tilde_user_path_as_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Bug caught: the unknown "~user" branch stores an empty or cut target, so a Python
    # caller loses the path it passed.
    use_fake_home(monkeypatch, home=HOME_YOU)

    with pytest.raises(TargetError) as caught:
        check_local_model("~nosuchuser/m")

    assert caught.value.target == "~nosuchuser/m"


# --- parity mlx references (parity/sources.py) ------------------------------------------


def test_parity_base_that_is_not_a_folder_is_named_by_its_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_fake_home(monkeypatch, home=HOME_YOU)

    code = cli.main(_parity_args(tmp_path, base="/Users/you/models/typo"))

    assert code == 2
    assert capsys.readouterr().err == PATH_TYPO_LINE


PATH_TYPO_LINE = (
    "Error: cannot use 'typo': it is not a local directory. Check the path for a typo.\n"
)


def test_parity_base_folder_with_a_space_is_named_by_its_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_fake_home(monkeypatch, home=HOME_YOU)

    code = cli.main(_parity_args(tmp_path, base="/Users/you/my models/typo"))

    assert code == 2
    assert capsys.readouterr().err == PATH_TYPO_LINE


def test_parity_base_with_a_trailing_slash_is_named_by_its_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Bug caught: the trailing slash stays in the reference, so the line says 'typo/'.
    use_fake_home(monkeypatch, home=HOME_YOU)

    code = cli.main(_parity_args(tmp_path, base="/Users/you/models/typo/"))

    assert code == 2
    assert capsys.readouterr().err == PATH_TYPO_LINE


def test_parity_base_one_folder_below_the_root_is_named_by_its_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Bug caught: the reference is not passed as a known model folder, so a path with
    # a single folder keeps its leading "/" (the general rule needs two parts).
    use_fake_home(monkeypatch, home=HOME_YOU)

    code = cli.main(_parity_args(tmp_path, base="/typo"))

    assert code == 2
    assert capsys.readouterr().err == PATH_TYPO_LINE


def test_parity_base_cache_snapshot_keeps_its_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_fake_home(monkeypatch, home=HOME_YOU)

    code = cli.main(
        _parity_args(
            tmp_path, base="~/.cache/huggingface/hub/models--org--name/snapshots/deadbeef0"
        )
    )

    assert code == 2
    assert capsys.readouterr().err == (
        "Error: cannot use 'org/name (cache snapshot deadbee)': it is not a local "
        "directory. Check the path for a typo.\n"
    )


def test_parity_base_with_an_unknown_tilde_user_is_an_error_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Bug caught: the home lookup for an unknown user raises out of the resolver as a
    # raw RuntimeError with a traceback.
    use_fake_home(monkeypatch, home=HOME_YOU)

    code = cli.main(_parity_args(tmp_path, base="~nosuchuser/m"))

    assert code == 2
    err = capsys.readouterr().err
    assert err == (
        "Error: cannot use '~nosuchuser/m': it is not a local directory. "
        "Check the path for a typo.\n"
    )
    assert "Traceback" not in err


def test_parity_base_with_an_unknown_tilde_user_drops_the_trailing_slash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Bug caught: the unknown "~user" branch names the reference as passed, so the
    # line says '~nosuchuser/m/' where every other branch drops the trailing slash.
    use_fake_home(monkeypatch, home=HOME_YOU)

    code = cli.main(_parity_args(tmp_path, base="~nosuchuser/m/"))

    assert code == 2
    assert capsys.readouterr().err == (
        "Error: cannot use '~nosuchuser/m': it is not a local directory. "
        "Check the path for a typo.\n"
    )


def test_parity_base_repo_id_is_kept_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_fake_home(monkeypatch, home=HOME_YOU)

    code = cli.main(_parity_args(tmp_path, base="org/name"))

    assert code == 2
    assert capsys.readouterr().err == (
        "Error: cannot use 'org/name': it is not a local directory. Download it first (e.g. "
        "`hf download org/name --local-dir ./name`) and pass that path. "
        "If this was meant to be a local path, check it for a typo.\n"
    )


# --- parity mlx --prompts (parity/prompts.py) -------------------------------------------


def test_missing_prompts_file_is_named_by_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_fake_home(monkeypatch, home=HOME_YOU)

    code = cli.main([*_parity_args(tmp_path), "--prompts", "/Users/you/p/missing.json"])

    assert code == 2
    assert capsys.readouterr().err == (
        "Error: could not read prompts file 'missing.json': "
        "[Errno 2] No such file or directory: 'missing.json'\n"
    )


def test_prompts_file_that_is_not_json_is_named_by_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = _real_home(tmp_path, monkeypatch)
    bad = home / "p" / "bad.json"
    bad.parent.mkdir()
    bad.write_text("not json", encoding="utf-8")

    code = cli.main([*_parity_args(tmp_path), "--prompts", str(bad)])

    assert code == 2
    assert capsys.readouterr().err == (
        "Error: prompts file 'bad.json' is not valid JSON: "
        "Expecting value: line 1 column 1 (char 0)\n"
    )


def test_prompts_file_that_is_not_an_array_is_named_by_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = _real_home(tmp_path, monkeypatch)
    obj = home / "p" / "obj.json"
    obj.parent.mkdir()
    obj.write_text("{}", encoding="utf-8")

    code = cli.main([*_parity_args(tmp_path), "--prompts", str(obj)])

    assert code == 2
    assert capsys.readouterr().err == (
        "Error: prompts file 'obj.json' must be a non-empty JSON array of "
        '{"prompt": <str>, "completion": <str>} objects\n'
    )


def test_prompts_file_with_a_bad_item_is_named_by_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = _real_home(tmp_path, monkeypatch)
    items = home / "p" / "items.json"
    items.parent.mkdir()
    items.write_text("[1]", encoding="utf-8")

    code = cli.main([*_parity_args(tmp_path), "--prompts", str(items)])

    assert code == 2
    assert capsys.readouterr().err == (
        "Error: prompts file 'items.json': item 0 must be an object with "
        'string "prompt" and "completion" fields\n'
    )


def test_refused_base_that_is_not_a_folder_is_named_by_its_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_fake_home(monkeypatch, home=HOME_YOU)
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps([{"prompt": "a", "completion": "b"}]), encoding="utf-8")
    gone = ResolvedIdentity(
        path="/Users/you/models/gone", original_ref="/Users/you/models/gone", source="local"
    )

    def resolver(*_refs: str, **_kwargs: object) -> ResolvedSources:
        return ResolvedSources(base=gone, adapter=gone, fused=gone)

    with pytest.raises(ModelDoctorError) as caught:
        build_prompts_fixture("/Users/you/models/gone", str(prompts), resolver=resolver)

    assert str(caught.value) == (
        "gone is not an existing directory, so it is unknown whether the base repository "
        "names loader code. The load was refused, so repository code was not run. "
        "Pass an existing model directory, then run again."
    )


def test_refused_base_folder_with_a_space_is_named_by_its_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_fake_home(monkeypatch, home=HOME_YOU)
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps([{"prompt": "a", "completion": "b"}]), encoding="utf-8")
    gone = ResolvedIdentity(
        path="/Users/you/my models/gone",
        original_ref="/Users/you/my models/gone",
        source="local",
    )

    def resolver(*_refs: str, **_kwargs: object) -> ResolvedSources:
        return ResolvedSources(base=gone, adapter=gone, fused=gone)

    with pytest.raises(ModelDoctorError) as caught:
        build_prompts_fixture("/Users/you/my models/gone", str(prompts), resolver=resolver)

    assert str(caught.value).startswith("gone is not an existing directory,")
