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
from mlx_model_doctor.redaction import redact_local_paths, target_model_dirs
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
