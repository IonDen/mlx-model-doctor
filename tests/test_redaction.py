"""Failure text loses every local root and keeps the name of what failed.

Each row's id names the bug it catches. Every test sets the home through
``use_fake_home``, so the machine's own home, passwd entry and Hugging Face variables
never join the roots the reduction knows.
"""

import pwd
import re
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import mlx_model_doctor.redaction as redaction
from mlx_model_doctor.redaction import (
    LocalIdentity,
    describe_local_path,
    identity_label,
    redact_local_paths,
    target_model_dirs,
)
from mlx_model_doctor.targets import HfTarget, LocalTarget
from tests.fakes import FakeShardHub, use_fake_home

REAL_PWD_HOME = redaction._pwd_home

HOME = "/Users/John Smith"
NO_HOME = "/nonexistent/home"
SHA = "0123456789abcdef0123456789abcdef01234567"
SNAP = (
    "/Users/John Smith/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit"
    "/snapshots/" + SHA
)
GPT2_SNAP = "/Users/John Smith/.cache/huggingface/hub/models--gpt2/snapshots/" + SHA
REPO = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
VOL = "/Volumes/Alice SSD/models/m"

HOME_YOU = "/Users/you"
IL_SHA = "da28692b5f139cb0ec58a356b437486b7dac7462"

JOHN = ("John", "Smith")
ALICE_SSD = ("Alice", "SSD")
CACHE_LAYOUT = (*JOHN, "models--", "snapshots", SHA, ".cache")


def no_home() -> Path:
    raise RuntimeError("Could not determine home directory.")


def no_passwd_entry(uid: int) -> pwd.struct_passwd:
    raise KeyError(f"getpwuid(): uid not found: {uid}")


def assert_none_of(names: tuple[str, ...], result: str) -> None:
    for name in names:
        assert name not in result


MODEL_DIR_ROWS = [
    pytest.param(
        (VOL,),
        NO_HOME,
        "No safetensors found in /Volumes/Alice SSD/models/m",
        "No safetensors found in m",
        ALICE_SSD,
        id="R1-model-dir-outside-home-is-not-cut",
    ),
    pytest.param(
        (VOL,),
        NO_HOME,
        "No such file or directory: '/Volumes/Alice SSD/models/m/tokenizer.json'",
        "No such file or directory: 'tokenizer.json'",
        ALICE_SSD,
        id="R2-file-below-a-spaced-model-dir-keeps-its-folders",
    ),
    pytest.param(
        (VOL,),
        NO_HOME,
        "Checkout 'https://huggingface.co//Volumes/Alice SSD/models/m/tree/main'",
        "Checkout 'https://huggingface.co/m/tree/main'",
        ALICE_SSD,
        id="R3-model-dir-followed-by-a-slash-inside-a-url-survives",
    ),
    pytest.param(
        (VOL,),
        NO_HOME,
        "Checkout 'https://huggingface.co//Volumes/Alice%20SSD/models/m/tree/main'",
        "Checkout 'https://huggingface.co/m/tree/main'",
        ALICE_SSD,
        id="R4-percent-encoded-model-dir-is-not-cut",
    ),
    pytest.param(
        ("/data/m",),
        HOME,
        "see /data/m2/x/y.py and /data/m",
        "see y.py and m",
        JOHN,
        id="R5-a-sibling-m2-is-cut-as-the-model-dir",
    ),
    pytest.param(
        ("/",),
        HOME,
        "see /opt/x/y.py",
        "see y.py",
        JOHN,
        id="R7-a-one-part-model-dir-rewrites-every-slash",
    ),
    pytest.param(
        ("/",),
        HOME,
        "cwd is / now",
        "cwd is / now",
        JOHN,
        id="R7-a-one-part-model-dir-rewrites-a-lone-slash",
    ),
    pytest.param(
        ("/a/m", "/a/m-fused"),
        HOME,
        "/a/m-fused/x.json; /a/m/y.json; /a/m-fused",
        "x.json; y.json; m-fused",
        JOHN,
        id="R24-the-slash-placement-breaks-for-two-dirs-with-a-shared-prefix",
    ),
    pytest.param(
        ("/Volumes/Alice SSD//models/m",),
        NO_HOME,
        "No safetensors found in /Volumes/Alice SSD//models/m",
        "No safetensors found in m",
        ALICE_SSD,
        id="R26-a-model-dir-with-a-double-slash-is-matched-only-normalised",
    ),
    pytest.param(
        (HOME_YOU,),
        HOME_YOU,
        "No safetensors found in /Users/you",
        "No safetensors found in ~",
        ("you",),
        id="N1-27-a-model-dir-that-is-the-home-becomes-a-tilde-not-the-user-name",
    ),
    pytest.param(
        (HOME_YOU,),
        HOME_YOU,
        "cannot read /Users/you/config.json",
        "cannot read config.json",
        ("you",),
        id="N1-27b-a-file-below-a-home-model-dir-keeps-no-tilde-or-slash",
    ),
]


@pytest.mark.parametrize(("dirs", "home", "text", "expected", "names"), MODEL_DIR_ROWS)
def test_a_model_dir_becomes_its_folder_name(
    dirs: tuple[str, ...],
    home: str,
    text: str,
    expected: str,
    names: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_home(monkeypatch, home=home)

    result = redact_local_paths(text, model_dirs=dirs)

    assert result == expected
    assert_none_of(names, result)


def test_r6_a_relative_model_dir_is_never_replaced_as_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    use_fake_home(monkeypatch, home=HOME)

    # The second text can tell a kept "m" from a replaced one: a replaced "m" before
    # "/" becomes "/m", and the general rule then cuts "/m/tokenizer.json".
    for text in ("m is missing; got m", "m/tokenizer.json is missing"):
        assert redact_local_paths(text, model_dirs=("m",)) == text


IDENTITY_ROWS = [
    pytest.param(
        "/Users/you/models/qwen-0.5b-4bit",
        ("qwen-0.5b-4bit", "local folder", None),
        HOME_YOU,
        id="IL1-a-model-folder-is-named-by-its-own-name",
    ),
    pytest.param(
        "/Users/you/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-4bit"
        "/snapshots/" + IL_SHA,
        ("mlx-community/Qwen3.5-0.8B-4bit", "cache snapshot", IL_SHA),
        HOME_YOU,
        id="IL2-a-cache-snapshot-is-its-repo-id-with-the-full-revision",
    ),
    pytest.param(
        "/Volumes/Alice SSD/models/m",
        ("m", "local folder", None),
        HOME_YOU,
        id="IL3-a-spaced-volume-keeps-only-the-model-folder-name",
    ),
    pytest.param(
        "/Users/you",
        ("~", "local folder", None),
        HOME_YOU,
        id="IL4-the-home-as-the-model-folder-is-a-tilde-not-the-user-name",
    ),
    pytest.param(
        "/",
        ("/", "local folder", None),
        HOME_YOU,
        id="IL5-the-root-folder-is-named-by-its-path-not-an-empty-label",
    ),
    pytest.param(
        "/Volumes/Alice SSD/hf/models--gpt2/snapshots/" + IL_SHA,
        ("gpt2", "cache snapshot", IL_SHA),
        HOME_YOU,
        id="IL6-a-snapshot-outside-the-hub-cache-is-still-a-cache-snapshot",
    ),
    pytest.param(
        "/Users/you/.cache/huggingface/hub",
        ("$HF_HUB_CACHE", "local folder", None),
        HOME_YOU,
        id="IL7-the-hub-cache-as-the-model-folder-is-its-variable-name",
    ),
    pytest.param(
        "/Users/you/.cache/huggingface",
        ("$HF_HOME", "local folder", None),
        HOME_YOU,
        id="IL7b-the-hf-home-as-the-model-folder-is-its-variable-name",
    ),
    pytest.param(
        "/Volumes/Alice SSD",
        ("Alice SSD", "local folder", None),
        HOME_YOU,
        id="IL8-a-volume-as-the-model-folder-keeps-the-volume-name",
    ),
    pytest.param(
        "/Users/you/models/m/",
        ("m", "local folder", None),
        HOME_YOU,
        id="IL10-a-trailing-slash-does-not-empty-the-label",
    ),
    pytest.param(
        "/x/models--/snapshots/abc",
        ("abc", "local folder", None),
        HOME_YOU,
        id="a-snapshot-shape-with-an-empty-repo-name-is-a-plain-folder-not-an-empty-label",
    ),
    pytest.param(
        "/Users/you/models--mine/exports/v1",
        ("v1", "local folder", None),
        HOME_YOU,
        id="a-models-folder-without-a-snapshots-parent-is-a-plain-folder",
    ),
    pytest.param(
        "/runs/exp1/snapshots/step-1000",
        ("step-1000", "local folder", None),
        HOME_YOU,
        id="a-snapshots-parent-without-a-models-folder-is-a-plain-folder",
    ),
    pytest.param(
        "/root",
        ("root", "local folder", None),
        "/root",
        id="a-one-component-home-as-the-model-folder-keeps-its-name-not-a-tilde",
    ),
]


@pytest.mark.parametrize(("path", "expected", "home"), IDENTITY_ROWS)
def test_a_local_path_gets_one_label_kind_and_revision(
    path: str,
    expected: tuple[str, str, str | None],
    home: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_home(monkeypatch, home=home)

    identity = describe_local_path(path)

    assert isinstance(identity, LocalIdentity)
    assert (identity.label, identity.kind, identity.revision) == expected
    assert identity_label(path) == expected[0]


SNAPSHOT_ROWS = [
    pytest.param(
        (),
        f"No safetensors found in {SNAP}",
        f"No safetensors found in {REPO}",
        CACHE_LAYOUT,
        id="R8-snapshot-shows-as-a-commit-id-or-keeps-the-cache-layout",
    ),
    pytest.param(
        (SNAP,),
        f"{SNAP}/config.json not found",
        "config.json not found",
        CACHE_LAYOUT,
        id="R9-a-snapshot-file-keeps-its-folder",
    ),
    # A snapshot passed as a model folder is labelled by its commit id, not org/name.
    pytest.param(
        (SNAP,),
        f"No safetensors found in {SNAP}",
        f"No safetensors found in {REPO}",
        CACHE_LAYOUT,
        id="R9-a-snapshot-model-dir-shows-its-commit-id",
    ),
    pytest.param(
        (),
        f"{SNAP}/tokenizer.json",
        "tokenizer.json",
        CACHE_LAYOUT,
        id="R10-the-shape-rule-needs-the-dir-to-be-passed",
    ),
    pytest.param(
        (),
        f"No safetensors found in {GPT2_SNAP}",
        "No safetensors found in gpt2",
        CACHE_LAYOUT,
        id="R11-an-org-less-legacy-repo-prints-its-cache-folder",
    ),
    pytest.param(
        (),
        f"Checkout 'https://huggingface.co/{SNAP}/tree/main'",
        f"Checkout 'https://huggingface.co/{REPO}/tree/main'",
        CACHE_LAYOUT,
        id="R12-the-glued-hub-url-loses-its-host-or-keeps-a-double-slash",
    ),
    pytest.param(
        (),
        "https://huggingface.co/docs/hub and /opt/x/y.py",
        "https://huggingface.co/docs/hub and y.py",
        JOHN,
        id="R13-the-shape-or-url-logic-eats-a-plain-url",
    ),
]


@pytest.mark.parametrize(("dirs", "text", "expected", "names"), SNAPSHOT_ROWS)
def test_a_cache_snapshot_becomes_its_repo_id(
    dirs: tuple[str, ...],
    text: str,
    expected: str,
    names: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_home(monkeypatch, home=HOME)

    result = redact_local_paths(text, model_dirs=dirs)

    assert result == expected
    assert_none_of(names, result)


CACHE_ROOT_ROWS = [
    pytest.param(
        {},
        "cannot write /Users/John Smith/.cache/huggingface/hub",
        "cannot write $HF_HUB_CACHE",
        JOHN,
        id="R14-the-hub-cache-under-home-shows-as-a-home-path",
    ),
    pytest.param(
        {"HF_HUB_CACHE": "/Volumes/Alice SSD/hf"},
        "lock at /Volumes/Alice SSD/hf/.locks/models--o--n/abc.lock; "
        f"snapshot /Volumes/Alice SSD/hf/models--o--n/snapshots/{SHA}",
        "lock at abc.lock; snapshot o/n",
        (*ALICE_SSD, *JOHN),
        id="R15-a-custom-cache-outside-home-keeps-the-volume-name",
    ),
    pytest.param(
        {"HF_HOME": "/Volumes/Alice SSD/hfh"},
        "xet log at /Volumes/Alice SSD/hfh/xet/logs/a.log; home /Volumes/Alice SSD/hfh",
        "xet log at a.log; home $HF_HOME",
        (*ALICE_SSD, *JOHN),
        id="R16-hf-home-is-not-a-root",
    ),
    pytest.param(
        {"HF_HOME": "/Volumes/Alice SSD//hfh"},
        "log /Volumes/Alice SSD//hfh/xet/a.log; home /Volumes/Alice SSD//hfh",
        "log a.log; home $HF_HOME",
        (*ALICE_SSD, *JOHN),
        id="R26-an-hf-home-with-a-double-slash-is-matched-only-normalised",
    ),
    # XDG_CACHE_HOME is ignored, so the hub cache under it keeps the volume name.
    pytest.param(
        {"XDG_CACHE_HOME": "/Volumes/Alice SSD/xdg"},
        "lock at /Volumes/Alice SSD/xdg/huggingface/hub/.locks/a.lock; "
        "root /Volumes/Alice SSD/xdg/huggingface/hub",
        "lock at a.lock; root $HF_HUB_CACHE",
        (*ALICE_SSD, *JOHN),
        id="R15-xdg-cache-home-does-not-move-the-hub-cache",
    ),
    # The legacy HUGGINGFACE_HUB_CACHE is ignored, so its volume name stays.
    pytest.param(
        {"HUGGINGFACE_HUB_CACHE": "/Volumes/Alice SSD/legacy"},
        "lock at /Volumes/Alice SSD/legacy/.locks/a.lock; root /Volumes/Alice SSD/legacy",
        "lock at a.lock; root $HF_HUB_CACHE",
        (*ALICE_SSD, *JOHN),
        id="R15-the-legacy-hub-cache-variable-is-not-a-root",
    ),
    # A variable inside HF_HOME is not expanded, so the HF home is never matched.
    pytest.param(
        {"ROOTVOL": "/Volumes/Alice SSD", "HF_HOME": "$ROOTVOL/hfh"},
        "log /Volumes/Alice SSD/hfh/xet/a.log; home /Volumes/Alice SSD/hfh",
        "log a.log; home $HF_HOME",
        (*ALICE_SSD, *JOHN),
        id="R16-a-variable-in-hf-home-is-not-expanded",
    ),
    # A variable inside HF_HUB_CACHE is not expanded, so the hub cache is never matched.
    pytest.param(
        {"ROOTVOL": "/Volumes/Alice SSD", "HF_HUB_CACHE": "$ROOTVOL/hub"},
        "lock at /Volumes/Alice SSD/hub/.locks/a.lock; root /Volumes/Alice SSD/hub",
        "lock at a.lock; root $HF_HUB_CACHE",
        (*ALICE_SSD, *JOHN),
        id="R15-a-variable-in-hf-hub-cache-is-not-expanded",
    ),
]


@pytest.mark.parametrize(("env", "text", "expected", "names"), CACHE_ROOT_ROWS)
def test_a_hugging_face_cache_root_becomes_its_variable_name(
    env: dict[str, str],
    text: str,
    expected: str,
    names: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_home(monkeypatch, home=HOME)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    result = redact_local_paths(text)

    assert result == expected
    assert_none_of(names, result)


HOME_ROWS = [
    pytest.param(
        "/tmp/sandbox/home",
        "/Users/Alice Smith",
        "bad '/Users/Alice Smith/Library/Caches/x.json'; cwd /Users/Alice Smith",
        "bad 'x.json'; cwd ~",
        ("Alice", "Smith"),
        id="R17-only-HOME-is-cut-not-the-passwd-home",
    ),
    pytest.param(
        "/Users/alice",
        None,
        "/Users/alice@work/projects/client/model/tok.json",
        "tok.json",
        ("alice", "work"),
        id="R19-at-sibling-keeps-the-tail-after-the-home",
    ),
    pytest.param(
        "/Users/alice",
        None,
        "/Users/alice+bob/a/b/c.json",
        "c.json",
        ("alice", "bob"),
        id="R19-plus-sibling-keeps-the-tail-after-the-home",
    ),
    pytest.param(
        "/Users/alice",
        None,
        "/Users/alice%20b/x/y.py",
        "y.py",
        ("alice",),
        id="R19-percent-sibling-keeps-the-tail-after-the-home",
    ),
    pytest.param(
        "/Users/alice",
        None,
        "/Users/alice=x/p/q.txt",
        "q.txt",
        ("alice",),
        id="R19-equals-sibling-keeps-the-tail-after-the-home",
    ),
    pytest.param(
        "/Users/alice",
        None,
        "/Users/alice#x/p/q.txt",
        "q.txt",
        ("alice",),
        id="R19-hash-sibling-keeps-the-tail-after-the-home",
    ),
    pytest.param(
        "/Users/alice",
        None,
        "/Users/alice$x/p/q.txt",
        "q.txt",
        ("alice",),
        id="R19-dollar-sibling-keeps-the-tail-after-the-home",
    ),
    # A "~" sibling is not a sibling, so its folder names stay after a "~".
    pytest.param(
        "/Users/alice",
        None,
        "/Users/alice~x/p/q.txt",
        "q.txt",
        ("alice",),
        id="R19-tilde-sibling-keeps-the-tail-after-the-home",
    ),
    pytest.param(
        NO_HOME,
        None,
        "~alice/models/m/tok.json",
        "tok.json",
        ("alice",),
        id="R20-a-tilde-user-path-is-kept",
    ),
    pytest.param(
        NO_HOME,
        None,
        "approx ~5/10 GB",
        "approx ~5/10 GB",
        (),
        id="R20-prose-tilde-5-slash-10-is-rewritten",
    ),
    pytest.param(
        NO_HOME,
        "/root",
        "/root/x/y.py is in /root",
        "y.py is in /root",
        (),
        id="R21-a-passwd-home-under-three-parts-is-cut",
    ),
    pytest.param(
        "/Users/./John Smith",
        None,
        "bad '/Users/./John Smith/Library/x.json'; cwd /Users/./John Smith",
        "bad 'x.json'; cwd ~",
        JOHN,
        id="R26-a-home-with-a-dot-segment-is-matched-only-normalised",
    ),
]


@pytest.mark.parametrize(("home", "pwd_home", "text", "expected", "names"), HOME_ROWS)
def test_a_home_becomes_a_tilde(
    home: str,
    pwd_home: str | None,
    text: str,
    expected: str,
    names: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_home(monkeypatch, home=home, pwd_home=pwd_home)

    result = redact_local_paths(text)

    assert result == expected
    assert_none_of(names, result)


SHARED_FOLDERS = ("Client", "Work", "Shared")

# A quoted absolute path runs to its close quote, spaces included, and becomes its last
# part when it holds a space and at least two parts.
QUOTED_ROWS = [
    pytest.param(
        NO_HOME,
        "No such file or directory: '/Users/Shared/Client Work/app/x.py'",
        "No such file or directory: 'x.py'",
        SHARED_FOLDERS,
        id="Q1-a-quoted-spaced-folder-under-users-is-cut-to-the-file",
    ),
    pytest.param(
        NO_HOME,
        "[Errno 2] No such file or directory: '/opt/My Models/m/a.json'",
        "[Errno 2] No such file or directory: 'a.json'",
        ("My", "Models"),
        id="Q2-a-quoted-spaced-folder-anywhere-is-cut-to-the-file",
    ),
    pytest.param(
        NO_HOME,
        "cannot open '/Volumes/Alice SSD/venvs/x.py'",
        "cannot open 'x.py'",
        ALICE_SSD,
        id="Q3-a-quoted-spaced-volume-is-cut-to-the-file",
    ),
    pytest.param(
        HOME_YOU,
        "No such file or directory: '/Users/you/Client Work/app/x.py'",
        "No such file or directory: 'x.py'",
        ("you", "Client"),
        id="Q4-a-quoted-spaced-folder-below-the-home-loses-the-tilde-too",
    ),
    pytest.param(
        NO_HOME,
        "'/Volumes/Alice SSD/venvs/x.py' is not a file; check the path",
        "'x.py' is not a file; check the path",
        ALICE_SSD,
        id="Q5-prose-after-the-close-quote-is-kept",
    ),
    pytest.param(
        NO_HOME,
        '"/Users/Shared/Bob\'s Files/x.py" not found',
        '"x.py" not found',
        ("Bob", "Files"),
        id="Q6-a-double-quoted-path-holding-an-apostrophe-closes-on-the-double-quote",
    ),
    pytest.param(
        NO_HOME,
        "'/Users/Shared/Bob's Files/x.py'",
        "'x.py'",
        ("Bob", "Files"),
        id="Q7-a-quote-followed-by-a-word-character-is-not-the-close",
    ),
    pytest.param(
        NO_HOME,
        "bad value '/opt/a and b/c'",
        "bad value 'c'",
        (),
        id="Q8-accepted-false-cut-of-quoted-prose-with-slashes",
    ),
    pytest.param(
        NO_HOME,
        "cannot open '/tmp'",
        "cannot open '/tmp'",
        (),
        id="Q9-pin-a-quoted-one-part-path-with-no-space-stays",
    ),
    pytest.param(
        NO_HOME,
        "cannot open '/My Drive'",
        "cannot open '/My Drive'",
        (),
        id="Q10-pin-a-quoted-one-part-spaced-path-stays",
    ),
    pytest.param(
        NO_HOME,
        "No safetensors found in '/Volumes/Alice SSD'",
        "No safetensors found in 'Alice SSD'",
        (),
        id="L3-pin-a-quoted-path-ending-at-a-volume-shows-the-volume-name",
    ),
    pytest.param(
        NO_HOME,
        "'name:/Volumes/Alice SSD/m/a.json'",
        "'name:/Volumes/Alice SSD/m/a.json'",
        (),
        id="L6-pin-a-path-glued-after-a-colon-inside-quotes-is-not-a-start",
    ),
    pytest.param(
        NO_HOME,
        "cannot open '~/Client Work'",
        "cannot open '~/Client Work'",
        (),
        id="pin-a-tilde-prefix-is-not-a-part-of-a-quoted-path",
    ),
    # Bug: the quoted pass is skipped, so the spaced pass sees no file tail after the
    # closing quote and the general rule leaves the folder name "My Models/".
    pytest.param(
        NO_HOME,
        "No safetensors in '/opt/My Models/m'",
        "No safetensors in 'm'",
        ("My", "Models"),
        id="pin-a-quoted-spaced-path-with-no-file-tail-is-cut-by-the-quoted-pass",
    ),
    # Bug: the close quote is searched past the line end, so the quote on the next
    # line closes this one and the text between is cut away.
    pytest.param(
        NO_HOME,
        "bad '/opt/a b\nnext 'c/d' end",
        "bad 'a b\nnext 'c/d' end",
        ("opt",),
        id="pin-a-quoted-path-never-closes-on-a-later-line",
    ),
]


@pytest.mark.parametrize(("home", "text", "expected", "names"), QUOTED_ROWS)
def test_a_quoted_path_with_a_space_becomes_its_last_part(
    home: str,
    text: str,
    expected: str,
    names: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_home(monkeypatch, home=home)

    result = redact_local_paths(text)

    assert result == expected
    assert_none_of(names, result)


# Unquoted, a run of up to four single-spaced words closed by "/" is one folder name
# when no word before a space ends a sentence, and it sits directly under "/Users/" or
# "/Volumes/" or a folder and a file follow it. Anything else ends the path at the space.
SPACED_ROWS = [
    pytest.param(
        NO_HOME,
        "failed to import /Users/Shared/Client Work/app/x.py",
        "failed to import x.py",
        ("Client", "Work"),
        id="U1-a-spaced-folder-with-a-folder-and-a-file-below-is-cut-to-the-file",
    ),
    pytest.param(
        NO_HOME,
        "see /opt/My Models/m/a.json",
        "see a.json",
        ("My", "Models"),
        id="U2-a-spaced-folder-anywhere-with-a-file-tail-is-cut",
    ),
    pytest.param(
        NO_HOME,
        "load /Volumes/Alice SSD/venvs/x.py",
        "load x.py",
        ALICE_SSD,
        id="U3-a-spaced-volume-name-is-cut",
    ),
    pytest.param(
        HOME_YOU,
        "import /Users/you/Client Work/app/x.py",
        "import x.py",
        ("you", "Client"),
        id="U4-a-spaced-folder-below-the-home-loses-the-tilde-too",
    ),
    pytest.param(
        NO_HOME,
        "could not read /Volumes/Alice SSD/venvs/x.py because it is locked",
        "could not read x.py because it is locked",
        ALICE_SSD,
        id="U5-prose-after-the-file-is-kept",
    ),
    pytest.param(
        NO_HOME,
        "/Volumes/My Big Backup Drive/m/a.json",
        "a.json",
        ("Backup", "Drive"),
        id="U7-a-four-word-volume-name-is-cut",
    ),
    pytest.param(
        NO_HOME,
        "on /Volumes/SSD and/or the cache",
        "on or the cache",
        (),
        id="U11-accepted-false-cut-of-prose-after-a-volume",
    ),
    pytest.param(
        NO_HOME,
        "copied /tmp/out to models/org/name.json",
        "copied name.json",
        (),
        id="U12-accepted-false-cut-of-a-two-part-file-tail-after-prose",
    ),
    pytest.param(
        NO_HOME,
        "in /opt/x for mlx-community/Qwen/config.json",
        "in config.json",
        (),
        id="D1-accepted-false-cut-of-a-repo-file-after-prose",
    ),
    # Accepted false cut: a full stop after a relative file ends the sentence, yet the
    # file still reads as the tail of the path before it, so the words between go too.
    # Bug: "." is dropped from the marks allowed after the extension.
    pytest.param(
        NO_HOME,
        "Copied /opt/x to models/m/config.json.",
        "Copied config.json.",
        (),
        id="accepted-false-cut-of-a-relative-file-that-ends-a-sentence",
    ),
    pytest.param(
        NO_HOME,
        "in /opt/x for mlx-community/Qwen2.5-0.5B",
        "in x for mlx-community/Qwen2.5-0.5B",
        (),
        id="U9-pin-a-repo-id-after-prose-is-not-swallowed",
    ),
    pytest.param(
        NO_HOME,
        "in /opt/m and/or the cache",
        "in m and/or the cache",
        (),
        id="U10-pin-prose-after-a-folder-outside-the-anchors-is-kept",
    ),
    pytest.param(
        HOME_YOU,
        "No safetensors found in /Users/you/models for mlx-community/Qwen",
        "No safetensors found in ~/models for mlx-community/Qwen",
        (),
        id="U19-pin-the-home-is-not-an-anchor-so-a-repo-id-after-it-is-kept",
    ),
    # Bug: a word ending in "." before a space joins a spaced folder, so "x. See docs/"
    # runs on to the file and the sentence is lost.
    pytest.param(
        NO_HOME,
        "in /opt/x. See docs/a/b.md",
        "in x. See docs/a/b.md",
        (),
        id="U13-pin-a-word-that-ends-a-sentence-does-not-join-a-folder",
    ),
    # Bug: the home is cut after the spaced pass, so "John Smith and/" joins under
    # "/Users/" and the prose up to "or" is lost.
    pytest.param(
        HOME,
        "cwd is /Users/John Smith and/or /tmp",
        "cwd is ~ and/or /tmp",
        JOHN,
        id="U18-pin-the-home-is-cut-before-the-spaced-pass",
    ),
    # Bug: the file-extension check is dropped, so the repo's "main" reads as a file.
    pytest.param(
        NO_HOME,
        "in /opt/x for mlx-community/Qwen/main",
        "in x for mlx-community/Qwen/main",
        (),
        id="pin-a-two-part-tail-without-an-extension-is-not-a-file",
    ),
    # Bug: an extension may start with a digit, so "Qwen2.5-0.5B" reads as a file. The
    # tail has two parts, so only the letter rule decides this row.
    pytest.param(
        NO_HOME,
        "in /opt/x for mlx-community/q4/Qwen2.5-0.5B",
        "in x for mlx-community/q4/Qwen2.5-0.5B",
        (),
        id="pin-an-extension-starts-with-a-letter",
    ),
    # Bug: "/Users/" is not an anchor, so an account name with a space ends the path
    # at its space and the general rule leaves "John Smithers/models/m".
    pytest.param(
        NO_HOME,
        "No safetensors found in /Users/John Smithers/models/m",
        "No safetensors found in m",
        ("John", "Smithers"),
        id="pin-a-spaced-account-name-under-users-is-cut",
    ),
    # Bug: the file tail may have no folder, so the prose "x for org/" reads as a
    # spaced folder over "name.json" and the path and prose are cut to the file.
    pytest.param(
        NO_HOME,
        "in /opt/x for org/name.json",
        "in x for org/name.json",
        ("opt",),
        id="pin-a-file-tail-needs-a-folder-before-the-file",
    ),
    # Bug: only "." ends a sentence, so "m: see docs/" reads as a spaced folder and
    # the path and the advice are cut to "setup.md".
    pytest.param(
        NO_HOME,
        "Could not load /opt/models/m: see docs/guide/setup.md",
        "Could not load m: see docs/guide/setup.md",
        ("opt", "models"),
        id="pin-a-colon-before-a-space-ends-a-sentence-too",
    ),
    # Bug: the extension must end the name, so punctuation after the file hides it and
    # the spaced folder stays.
    pytest.param(
        NO_HOME,
        "cannot open /opt/My Models/m/a.json: Permission denied",
        "cannot open a.json: Permission denied",
        ("My", "Models"),
        id="a-colon-after-the-file-keeps-the-cut-and-the-reason",
    ),
    pytest.param(
        NO_HOME,
        "see /opt/My Models/m/a.json.",
        "see a.json.",
        ("My", "Models"),
        id="a-full-stop-after-the-file-keeps-the-cut",
    ),
    pytest.param(
        NO_HOME,
        "(see /opt/My Models/m/a.json)",
        "(see a.json)",
        ("My", "Models"),
        id="a-closing-bracket-after-the-file-keeps-the-cut",
    ),
    pytest.param(
        NO_HOME,
        "see /opt/My Models/m/a.json, then retry",
        "see a.json, then retry",
        ("My", "Models"),
        id="a-comma-after-the-file-keeps-the-cut",
    ),
    # Bug: the mark is dropped from the marks allowed after the extension, so the file
    # no longer reads as a file and the spaced folder stays.
    pytest.param(
        NO_HOME,
        "see /opt/My Models/m/a.json; then retry",
        "see a.json; then retry",
        ("My", "Models"),
        id="a-semicolon-after-the-file-keeps-the-cut",
    ),
    pytest.param(
        NO_HOME,
        "see /opt/My Models/m/a.json! then retry",
        "see a.json! then retry",
        ("My", "Models"),
        id="an-exclamation-mark-after-the-file-keeps-the-cut",
    ),
    pytest.param(
        NO_HOME,
        "see /opt/My Models/m/a.json? then retry",
        "see a.json? then retry",
        ("My", "Models"),
        id="a-question-mark-after-the-file-keeps-the-cut",
    ),
    pytest.param(
        NO_HOME,
        "see /opt/My Models/m/a.json] then retry",
        "see a.json] then retry",
        ("My", "Models"),
        id="a-closing-square-bracket-after-the-file-keeps-the-cut",
    ),
    # Bug: an anchored path that ends in "/" has no last part, so nothing is cut and
    # the volume or user name shows.
    pytest.param(
        NO_HOME,
        "see /Volumes/Alice SSD/models/",
        "see models/",
        ALICE_SSD,
        id="an-anchored-volume-path-ending-in-a-slash-is-cut-to-its-last-folder",
    ),
    pytest.param(
        NO_HOME,
        "see /Users/John Smith/models/",
        "see models/",
        JOHN,
        id="an-anchored-user-path-ending-in-a-slash-is-cut-to-its-last-folder",
    ),
    # The general rule keeps the trailing slash of a folder path; the anchored rows
    # above keep it the same way.
    pytest.param(
        NO_HOME,
        "see /opt/models/",
        "see models/",
        ("opt",),
        id="pin-a-plain-path-ending-in-a-slash-keeps-the-slash",
    ),
    # Bug: a URL scheme reads as prose inside a spaced folder, so the path joins the
    # URL and both are cut to the URL's file.
    pytest.param(
        NO_HOME,
        "from /opt/x to https://huggingface.co/org/name/resolve/main/config.json",
        "from x to https://huggingface.co/org/name/resolve/main/config.json",
        ("opt",),
        id="a-url-after-a-path-is-kept-whole",
    ),
    # Bug: any ":/" reads as a URL scheme, so a colon that starts a second path (a
    # search path or a mount) ends the volume name and the name shows.
    pytest.param(
        NO_HOME,
        "/Volumes/Alice SSD:/usr/bin",
        "bin",
        ALICE_SSD,
        id="a-colon-and-slash-that-is-no-url-scheme-keeps-the-cut",
    ),
    # The limits below are pinned at today's output; a change to any is a decision.
    # Bug: the word bound is raised to five, so a 5-word volume name is cut.
    pytest.param(
        NO_HOME,
        "/Volumes/My Very Big Backup Drive/m/a.json",
        "My Very Big Backup Drive/m/a.json",
        (),
        id="L1-limit-a-five-word-volume-name-stays",
    ),
    # Bug: the word bound is raised to six, so a 6-word volume name is cut. With L1
    # this shows the limit is five or more words, not five alone.
    pytest.param(
        NO_HOME,
        "/Volumes/My Very Big Old Backup Drive/m/a.json",
        "My Very Big Old Backup Drive/m/a.json",
        (),
        id="limit-a-six-word-volume-name-stays",
    ),
    # Limit: a file whose name has no extension starting with a letter is not a file
    # tail, so the spaced folder stays. Bug: the extension check is dropped.
    pytest.param(
        NO_HOME,
        "see /opt/My Models/m/LICENSE",
        "see My Models/m/LICENSE",
        (),
        id="limit-a-spaced-folder-over-a-file-with-no-extension-stays",
    ),
    # Bug: an extension may start with a digit, so "weights.7z" reads as a file.
    pytest.param(
        NO_HOME,
        "see /opt/My Models/m/weights.7z",
        "see My Models/m/weights.7z",
        (),
        id="limit-a-spaced-folder-over-a-file-with-a-digit-led-extension-stays",
    ),
    # Limit: a name holding ". " reads as the end of a sentence even under an anchor,
    # so the path ends at the space. Bug: "." no longer ends a sentence.
    pytest.param(
        NO_HOME,
        "see /Volumes/Dr. Smith/venvs/x.py",
        "see Dr. Smith/venvs/x.py",
        (),
        id="limit-an-anchored-name-holding-a-full-stop-and-a-space-stays",
    ),
    # Limit: the same holds under "/Users/", so an account name holding ". " stays.
    pytest.param(
        NO_HOME,
        "see /Users/J. Smith/models/m/config.json",
        "see J. Smith/models/m/config.json",
        (),
        id="limit-an-account-name-holding-a-full-stop-and-a-space-stays",
    ),
    # Limit: a file directly in a spaced folder outside the anchors has no folder
    # between them, so the spaced folder stays. Bug: the file tail needs no folder.
    pytest.param(
        NO_HOME,
        "see /opt/My Models/a.json",
        "see My Models/a.json",
        (),
        id="limit-a-file-directly-in-an-unanchored-spaced-folder-stays",
    ),
    # Bug: every folder counts as an anchor, so a spaced folder with no file tail joins.
    pytest.param(
        NO_HOME,
        "No safetensors found in /opt/My Models/m and the cache is empty",
        "No safetensors found in My Models/m and the cache is empty",
        (),
        id="L2-limit-an-unanchored-spaced-folder-with-no-file-tail-stays",
    ),
    # Bug: a spaced run may end the path, so the volume name itself is cut away.
    pytest.param(
        NO_HOME,
        "No safetensors found in /Volumes/Alice SSD",
        "No safetensors found in Alice SSD",
        (),
        id="L3-limit-an-unquoted-path-ending-at-a-volume-shows-the-volume-name",
    ),
    # Bug: the last folder read before the volume name counts as the path's last
    # folder, so "Volumes/" replaces the path.
    pytest.param(
        NO_HOME,
        "see /Volumes/Alice SSD/ now",
        "see Alice SSD/ now",
        ("Volumes",),
        id="limit-a-path-ending-at-a-volume-folder-shows-the-volume-name",
    ),
    # Bug: a double space joins the words, so the volume name is cut.
    pytest.param(
        NO_HOME,
        "/Volumes/Alice  SSD/x.py",
        "Alice  SSD/x.py",
        (),
        id="L5-limit-a-double-spaced-name-stays",
    ),
    # Bug: the file tail may cross a second spaced folder, so the nested path is cut.
    pytest.param(
        NO_HOME,
        "see /opt/My Models/Sub Dir/a.json",
        "see My Models/Sub Dir/a.json",
        (),
        id="L7-limit-a-spaced-folder-nested-in-another-outside-the-anchors-stays",
    ),
    # Limit: a backtick is not one of the marks allowed after the extension, so the
    # file does not read as a file and the spaced folder stays. Bug: the backtick is
    # allowed.
    pytest.param(
        NO_HOME,
        "see `/opt/My Models/m/a.json`",
        "see `My Models/m/a.json`",
        (),
        id="limit-a-backtick-after-the-file-keeps-the-spaced-folder",
    ),
    # Limit: a mark followed by more text in the same word (a line and a column) is
    # not allowed after the extension, so the spaced folder stays. Bug: digits are
    # allowed among the marks.
    pytest.param(
        NO_HOME,
        "/opt/My Models/m/a.py:12:5: error",
        "My Models/m/a.py:12:5: error",
        (),
        id="limit-a-line-and-column-after-the-file-keeps-the-spaced-folder",
    ),
    # Limit: as with a volume (L3), a path that ends at another account's spaced name
    # shows the name. Bug: a spaced name may end the path and is cut away.
    pytest.param(
        NO_HOME,
        "see /Users/Other Name",
        "see Other Name",
        (),
        id="limit-a-path-ending-at-another-accounts-spaced-name-shows-the-name",
    ),
]


@pytest.mark.parametrize(("home", "text", "expected", "names"), SPACED_ROWS)
def test_an_unquoted_spaced_folder_is_cut_only_on_evidence(
    home: str,
    text: str,
    expected: str,
    names: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_home(monkeypatch, home=home)

    result = redact_local_paths(text)

    assert result == expected
    assert_none_of(names, result)


def test_r18_the_realpath_of_home_is_a_home_spelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "Real Alice"
    real.mkdir()
    real = real.resolve()
    link = tmp_path / "link home"
    link.symlink_to(real)
    use_fake_home(monkeypatch, home=str(link))

    result = redact_local_paths(f"{real}/a/b.json and {real}")

    assert result == "b.json and ~"
    assert_none_of(("Real", "Alice"), result)


def test_r22_a_missing_passwd_entry_gives_no_passwd_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_home(monkeypatch, home=NO_HOME)
    monkeypatch.setattr(redaction, "_pwd_home", REAL_PWD_HOME)
    monkeypatch.setattr(pwd, "getpwuid", no_passwd_entry)

    assert redaction._pwd_home() is None
    assert redact_local_paths("/opt/x/y.py") == "y.py"


def test_r23_a_failing_path_home_still_cuts_the_passwd_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_home(monkeypatch, home=HOME, pwd_home="/Users/Alice Smith")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: no_home()))

    result = redact_local_paths("cwd /Users/Alice Smith")

    assert result == "cwd ~"


def test_r25_a_failing_lookup_skips_that_candidate_and_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # os.path.realpath raises ValueError on the NUL; Path.home and the passwd
    # lookup fail too. The reduction runs inside crash handlers, so it must not
    # replace the real failure with its own traceback.
    use_fake_home(monkeypatch, home=NO_HOME)
    monkeypatch.setattr(redaction, "_pwd_home", REAL_PWD_HOME)
    monkeypatch.setattr(pwd, "getpwuid", no_passwd_entry)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: no_home()))

    result = redact_local_paths("see /opt/x/y.py", model_dirs=("/a/b\x00c",))

    assert result == "see y.py"


# A lone surrogate is how Python decodes a non-UTF-8 byte in an environment variable or
# a file name. Percent-encoding such a path raises UnicodeEncodeError.
SURROGATE_HOME = "/Users/John\udcff Smith"
SURROGATE_HF_HOME = "/Volumes/Alice\udcff SSD/hfh"
SURROGATE_MODEL_DIR = "/Volumes/Alice\udcff SSD/models/m"

SURROGATE_ROWS = [
    pytest.param(
        SURROGATE_HOME,
        {},
        (),
        f"cwd {SURROGATE_HOME}; bad '{SURROGATE_HOME}/Library/x.json'",
        "cwd ~; bad 'x.json'",
        (*JOHN, "\udcff"),
        id="R25-a-surrogate-home-raises-or-loses-its-plain-spelling",
    ),
    pytest.param(
        HOME,
        {"HF_HOME": SURROGATE_HF_HOME},
        (),
        f"home {SURROGATE_HF_HOME}",
        "home $HF_HOME",
        (*ALICE_SSD, "\udcff"),
        id="R25-a-surrogate-hf-home-raises-or-loses-its-plain-spelling",
    ),
    pytest.param(
        HOME,
        {},
        (SURROGATE_MODEL_DIR,),
        f"No safetensors found in {SURROGATE_MODEL_DIR}",
        "No safetensors found in m",
        (*ALICE_SSD, "\udcff"),
        id="R25-a-surrogate-model-dir-raises-or-loses-its-plain-spelling",
    ),
]


@pytest.mark.parametrize(("home", "env", "dirs", "text", "expected", "names"), SURROGATE_ROWS)
def test_r25_a_surrogate_path_skips_only_its_encoded_spelling(
    home: str,
    env: dict[str, str],
    dirs: tuple[str, ...],
    text: str,
    expected: str,
    names: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_home(monkeypatch, home=home)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    result = redact_local_paths(text, model_dirs=dirs)

    assert result == expected
    assert_none_of(names, result)


# The snapshot pattern as it stood before the linear pass: the reference for which
# spans a snapshot cut covers. Quadratic on long whitespace-free text, so it only
# ever sees the short texts below.
QUADRATIC_SNAPSHOT = re.compile(
    r"(?:~|\$HF_HUB_CACHE|\$HF_HOME|(?<![\w.~$:-])(?<!:/)(?=/))[^\s'\"]*?"
    r"/models--(?P<repo>[\w.-]+)/snapshots/\w[\w.-]*?(?![\w-]|\.\w)"
)
SNAPSHOT_FRAGMENTS = (
    "~",
    "$",
    "$HF_HOME",
    "$HF_HUB_CACHE",
    "/",
    ":",
    "://",
    ".",
    "-",
    "x",
    "é",
    " ",
    "　",
    "'",
    '"',
    "/models--a",
    "models--a",
    "/models--o--n",
    "/snapshots/",
    "b",
    "/models--a/snapshots/b",
    "/models--o--n/snapshots/x",
)


@settings(max_examples=3000, deadline=None, derandomize=True)
@given(st.lists(st.sampled_from(SNAPSHOT_FRAGMENTS), max_size=14).map("".join))
def test_the_snapshot_cut_covers_the_same_spans_as_the_reference_pattern(text: str) -> None:
    # Bug: the linear pass picks a later prefix start, crosses a space or quote, or
    # reads a lookbehind from the rewritten text instead of the original.
    expected = QUADRATIC_SNAPSHOT.sub(
        lambda match: f"<{match.start()}:{match.end()}:{match['repo']}>", text
    )

    assert redaction._cut_snapshots(text, _span_marker) == expected


def _span_marker(text: str, start: int, end: int, repo: str) -> str:
    return f"<{start}:{end}:{repo}>"


def _repeated(unit: str) -> Callable[[int], str]:
    return lambda chars: unit * (chars // len(unit))


def _tilde_run_then_snapshot(chars: int) -> str:
    tail = " /models--a/snapshots/b"
    return "~" * (chars - len(tail)) + tail


# Each shape gives the old pattern a prefix start at almost every character, or a
# snapshot literal at almost every 23rd one. The last shape also defeats a guard that
# skips the pass when the text holds no "/models--".
SCALING_SHAPES = [
    pytest.param(_repeated("~"), id="tilde-run"),
    pytest.param(_repeated(",/"), id="comma-slash-run"),
    pytest.param(_repeated("$HF_HOME"), id="hf-home-token-run"),
    pytest.param(_repeated("x/models--a/snapshots/b"), id="snapshot-literal-with-no-start"),
    pytest.param(_tilde_run_then_snapshot, id="tilde-run-then-one-snapshot"),
    # Bug: a later same-quote opener no longer closes an earlier start, so every start
    # reads to the line end.
    pytest.param(_repeated("'/a "), id="unclosed-quote-start-run"),
    pytest.param(_repeated("'/a b/"), id="unclosed-quoted-spaced-start-run"),
    # Guard for the spaced pass: an anchored folder that never closes. No known mutant
    # turns it red; it is kept as a guard.
    pytest.param(_repeated("/Volumes/a b "), id="anchored-spaced-run-that-never-closes"),
    # Spaced folders of the most words the pass accepts. Each unit starts with a space,
    # so every "/o/" is a path start and the spaced pass reads every unit. Bug: the
    # file tail crosses whitespace, so every start reads on to the end of the text.
    pytest.param(_repeated(" /o/a b c d/"), id="four-word-spaced-folder-run"),
]

# A linear pass doubles its time when the text doubles (ratio about 2); a quadratic
# one quadruples it (about 4). 3 sits between them. Each size takes the best of five
# runs, so a busy machine slows a sample without moving the ratio; a run already
# over the budget is quadratic, and stops the sampling so the failure stays quick.
RATIO_LIMIT = 3.0
SAMPLE_BUDGET_SECONDS = 0.25


def _best_seconds(text: str) -> float:
    best = float("inf")
    for _ in range(5):
        began = time.perf_counter()
        redact_local_paths(text)
        best = min(best, time.perf_counter() - began)
        if best > SAMPLE_BUDGET_SECONDS:
            break
    return best


@pytest.mark.parametrize("shape", SCALING_SHAPES)
def test_the_reduction_time_grows_linearly_with_the_text(
    shape: Callable[[int], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Bug: a prefix start, or the search for one, rescans the rest of its run.
    use_fake_home(monkeypatch, home=HOME)

    small = _best_seconds(shape(20_000))
    large = _best_seconds(shape(40_000))

    assert large / small < RATIO_LIMIT, f"20k chars {small:.4f} s, 40k chars {large:.4f} s"


@pytest.mark.parametrize("shape", SCALING_SHAPES)
def test_100k_characters_reduce_in_under_half_a_second(
    shape: Callable[[int], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Bug: the snapshot pass is quadratic (100k chars of "~" took 55 s).
    use_fake_home(monkeypatch, home=HOME)
    text = shape(100_000)

    began = time.perf_counter()
    redact_local_paths(text)
    seconds = time.perf_counter() - began

    assert seconds < 0.5, f"100k chars {seconds:.4f} s"


def test_target_model_dirs_names_a_local_target_by_its_resolved_root(tmp_path: Path) -> None:
    assert target_model_dirs(LocalTarget(tmp_path)) == (Path(str(tmp_path.resolve())),)


def test_target_model_dirs_passes_no_dir_for_a_hub_repo_id() -> None:
    # Bug: the repo id is passed as a relative dir and rewrites "o/n" text.
    hub = FakeShardHub(files={"config.json": b"{}"}, shards={})

    assert target_model_dirs(HfTarget("o/n", hub=hub)) == ()
