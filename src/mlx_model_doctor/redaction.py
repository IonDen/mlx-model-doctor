"""Shorten the local paths in failure text to the names a user can act on.

A model folder becomes its name, a Hugging Face cache snapshot becomes its ``org/name``,
a file becomes its own name, the home folder becomes ``~``, and the Hugging Face cache
roots become ``$HF_HUB_CACHE`` and ``$HF_HOME``.
"""

import os
import re
import urllib.parse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mlx_model_doctor.targets import ModelTarget

HF_HUB_CACHE_TOKEN = "$HF_HUB_CACHE"
HF_HOME_TOKEN = "$HF_HOME"

LocalKind = Literal["local folder", "cache snapshot"]


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalIdentity:
    """How a local model path is named: its label, its kind and its revision."""

    label: str
    kind: LocalKind
    revision: str | None


def describe_local_path(path: str | PurePath) -> LocalIdentity:
    """Return the label, kind and revision of a local model path.

    The home is ``~``, the Hugging Face hub cache and home are ``$HF_HUB_CACHE`` and
    ``$HF_HOME``, a cache snapshot is its ``org/name`` with its revision, and any other
    folder is its own name (``/`` stays ``/``). The roots are the ones the free-text
    reduction cuts, and a failing lookup skips its root, so this never raises.
    """
    model_dir = PurePath(path)
    written = str(model_dir)
    for candidates, token in (
        (_home_candidates(), "~"),
        (_hub_cache_candidates(), HF_HUB_CACHE_TOKEN),
        (_hf_home_candidates(), HF_HOME_TOKEN),
    ):
        if written in _spellings(candidates, min_parts=3):
            return LocalIdentity(label=token, kind="local folder", revision=None)
    repo_folder = model_dir.parent.parent.name
    repo = _repo_label(repo_folder.removeprefix("models--"))
    if model_dir.parent.name == "snapshots" and repo_folder.startswith("models--") and repo:
        return LocalIdentity(label=repo, kind="cache snapshot", revision=model_dir.name)
    return LocalIdentity(label=model_dir.name or written, kind="local folder", revision=None)


def identity_label(path: str | PurePath) -> str:
    """Return the label of a local model path (see ``describe_local_path``)."""
    return describe_local_path(path).label


# The lookups that can fail on an odd environment (a NUL in a path, no home, no passwd
# entry). A failing lookup skips its candidate: the reduction runs inside crash
# handlers and must never replace the real failure with its own.
_LOOKUP_ERRORS = (OSError, ValueError, RuntimeError, KeyError)

# What blocks a root or model-dir match: a word character, a "-", or a "." followed by
# a word character, which continue the name into another folder ("/Users/alice2",
# "/data/m-fused", "/Users/alice.smith"). Anything else, or the end of the text, cuts.
_NAME_END = r"(?![\w-]|\.\w)"

# A sibling folder that starts with a root's name: one of "@ + % = # $ ~", more
# characters, then "/" ("/Users/alice@work/..."). The root and that run become "/".
_SIBLING_TAIL = r"(?P<sibling>[@+%=#$~][^\s/]+/)?"
# The same sibling, refused: the root does not match there, so a shorter root may.
_NOT_SIBLING = r"(?![@+%=#$~][^\s/]+/)"

# A Hugging Face cache snapshot, wherever it appears: a prefix ("~", a root token, or a
# "/" that does not continue a word or a "://"), any run of characters with no
# whitespace or quote, then "/models--<repo>/snapshots/<revision>". The pass finds the
# "/models--" part first and then the earliest prefix in the same run, so the time
# stays linear in the text (see ``_cut_snapshots``).
_SNAPSHOT_CORE = re.compile(r"/models--(?P<repo>[\w.-]+)/snapshots/\w[\w.-]*?" + _NAME_END)
_SNAPSHOT_START = re.compile(r"~|\$HF_HUB_CACHE|\$HF_HOME|(?<![\w.~$:-])(?<!:/)/")
_RUN_BREAK = re.compile(r"[\s'\"]+")

# "~user" directly before "/" ("~alice/models"). A bare "~alice" stays: it cannot be
# told apart from prose, and neither can "~5/10".
_TILDE_USER = re.compile(r"(?<![\w~])~[A-Za-z_][\w.-]*(?=/)")

# An absolute path: an optional "~" or root token, then a "/" that does not continue a
# word, a URL, another "/" or a "~", then at least two components. The match is
# replaced by its last component.
_ABSOLUTE_PATH = re.compile(
    r"(?<![\w:/.~])(?:~|\$HF_HUB_CACHE|\$HF_HOME)?/+(?:[^\s/'\"]+/+)+([^\s/'\"]+)"
)


def redact_local_paths(text: str, *, model_dirs: Sequence[Path | str] = ()) -> str:
    """Return ``text`` with every local path it holds cut to the name that matters.

    ``model_dirs`` are the model folders the caller knows; each is cut to its name (or
    ``org/name`` for a cache snapshot), plain, resolved or percent-encoded. The home,
    its resolved path and the passwd home become ``~``, the Hugging Face hub cache and
    home become ``$HF_HUB_CACHE`` and ``$HF_HOME``, and any other absolute path becomes
    its last component.
    """
    reduced = _cut_model_dirs(text, model_dirs)
    reduced = _cut_roots(reduced)
    reduced = _cut_snapshots(reduced)
    reduced = _TILDE_USER.sub("~", reduced)
    return _ABSOLUTE_PATH.sub(r"\1", reduced)


def target_model_dirs(target: "ModelTarget") -> tuple[Path, ...]:
    """Return the model folder of a local target; a Hub target passes none."""
    if target.source == "local":
        return (Path(target.name),)
    return ()


def redact_roots(text: str) -> str:
    """Return ``text`` with the home as ``~`` and the Hugging Face cache roots as tokens.

    Nothing else changes: a path outside these roots keeps every component, and a
    folder beside a root ("/Users/alice+old/...") is left whole.
    """
    return _substitute(text, _root_tokens(), lambda _match, token: token, tail=_NOT_SIBLING)


def _pwd_home() -> str | None:
    """Return the passwd home of the current user, or ``None`` when there is none."""
    try:
        import pwd  # absent on some platforms, so imported here

        return pwd.getpwuid(os.getuid()).pw_dir
    except (ImportError, KeyError, OSError):
        return None


def _cut_model_dirs(text: str, model_dirs: Sequence[Path | str]) -> str:
    labels: dict[str, str] = {}
    for model_dir in model_dirs:
        label = identity_label(model_dir)
        for spelling in _spellings((os.fspath(model_dir),), min_parts=2):
            labels.setdefault(spelling, label)
    return _substitute(text, labels, _placed)


def _root_tokens() -> dict[str, str]:
    tokens: dict[str, str] = {}
    for candidates, token in (
        (_hub_cache_candidates(), HF_HUB_CACHE_TOKEN),
        (_hf_home_candidates(), HF_HOME_TOKEN),
        (_home_candidates(), "~"),
    ):
        for spelling in _spellings(candidates, min_parts=3):
            tokens.setdefault(spelling, token)
    return tokens


def _cut_roots(text: str) -> str:
    return _substitute(
        text,
        _root_tokens(),
        lambda match, token: "/" if match["sibling"] else token,
        tail=_SIBLING_TAIL,
    )


def _substitute(
    text: str,
    replacements: dict[str, str],
    replace: Callable[[re.Match[str], str], str],
    *,
    tail: str = "",
) -> str:
    """Replace every spelling in one pass, the longest first, each with its callback."""
    if not replacements:
        return text
    ordered = sorted(replacements, key=len, reverse=True)
    pattern = re.compile(
        "(?P<spelling>" + "|".join(map(re.escape, ordered)) + ")" + _NAME_END + tail
    )
    return pattern.sub(lambda match: replace(match, replacements[match["spelling"]]), text)


def _placed(match: re.Match[str], label: str) -> str:
    """Return ``label`` for a matched folder, as its position in the text needs it."""
    return _placed_at(match.string, *match.span(), label)


def _placed_at(text: str, start: int, end: int, label: str) -> str:
    """Return ``label`` for the folder at ``text[start:end]``, as its position needs it.

    Glued to a URL (after a "/") or at the end of a path, the folder becomes its label.
    Followed by "/", it becomes "/" and the label, so the general rule then cuts the
    file below it to its own name.
    """
    preceded_by_slash = start > 0 and text[start - 1] == "/"
    followed_by_slash = text[end : end + 1] == "/"
    if followed_by_slash and not preceded_by_slash:
        return "/" + label
    return label


def _placed_snapshot(text: str, start: int, end: int, repo: str) -> str:
    return _placed_at(text, start, end, _repo_label(repo))


def _cut_snapshots(
    text: str, replace: Callable[[str, int, int, str], str] = _placed_snapshot
) -> str:
    """Replace each cache snapshot path with ``replace(text, start, end, repo)``.

    For each ``/models--…/snapshots/…`` hit, the path starts at the earliest prefix
    (see ``_SNAPSHOT_START``) in the same whitespace- and quote-free run. ``scanned``
    marks where that search resumes: no prefix sits in the run before it, and the
    text before a replaced path is never searched again. Each character is read a
    constant number of times, and every lookbehind reads the original text.
    """
    pieces: list[str] = []
    kept = 0
    scanned = 0
    core = _SNAPSHOT_CORE.search(text)
    while core is not None:
        at = core.start()
        run_start = scanned
        for run_break in _RUN_BREAK.finditer(text, scanned, at):
            run_start = run_break.end()
        start = _SNAPSHOT_START.search(text, run_start, at + 1)
        if start is None:
            scanned = at + 1
            core = _SNAPSHOT_CORE.search(text, at + 1)
            continue
        pieces.append(text[kept : start.start()])
        pieces.append(replace(text, start.start(), core.end(), core["repo"]))
        kept = scanned = core.end()
        core = _SNAPSHOT_CORE.search(text, kept)
    pieces.append(text[kept:])
    return "".join(pieces)


def _repo_label(repo: str) -> str:
    """Turn a cache folder's ``org--name`` into ``org/name`` (``gpt2`` stays)."""
    return repo.replace("--", "/", 1)


def _spellings(candidates: Iterable[str], *, min_parts: int) -> list[str]:
    """Return each absolute candidate and its realpath: normalised, as given, encoded.

    The spelling as given keeps a "//" or "/./" that the Hugging Face cache and a
    loader print unchanged. A path that cannot be percent-encoded (a lone surrogate
    from a non-UTF-8 byte) skips that spelling only.
    """
    spellings: list[str] = []
    for candidate in candidates:
        for spelling in (candidate, _guarded(os.path.realpath, candidate)):
            if spelling is None:
                continue
            path = PurePath(spelling)
            if not path.is_absolute() or len(path.parts) < min_parts:
                continue
            encoded = _guarded(urllib.parse.quote, str(path))
            for written in (str(path), spelling.rstrip("/"), encoded):
                if written is not None and written not in spellings:
                    spellings.append(written)
    return spellings


def _home_candidates() -> list[str]:
    # Path.home() normalises "/Users/./name"; the HOME string as given is a spelling too.
    homes = [
        _guarded(lambda: str(Path.home())),
        _guarded(os.path.expanduser, "~"),
        _guarded(_pwd_home),
    ]
    return [home for home in homes if home is not None]


def _hf_home_candidates() -> list[str]:
    return [home] if (home := _guarded(_hf_home)) is not None else []


def _hub_cache_candidates() -> list[str]:
    return [cache] if (cache := _guarded(_hub_cache)) is not None else []


def _hf_home() -> str:
    """Return the HF home as huggingface_hub 1.x ``constants.HF_HOME`` derives it."""
    default = os.path.join(  # noqa: PTH118 - mirrors huggingface_hub/constants.py
        os.getenv("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")),  # noqa: PTH111, PTH118
        "huggingface",
    )
    return os.path.expandvars(os.path.expanduser(os.getenv("HF_HOME", default)))  # noqa: PTH111


def _hub_cache() -> str:
    """Return the hub cache as huggingface_hub 1.x ``constants.HF_HUB_CACHE`` derives it."""
    legacy = os.getenv("HUGGINGFACE_HUB_CACHE", os.path.join(_hf_home(), "hub"))  # noqa: PTH118
    return os.path.expandvars(os.path.expanduser(os.getenv("HF_HUB_CACHE", legacy)))  # noqa: PTH111


def _guarded(lookup: Callable[..., str | None], *args: str) -> str | None:
    """Return the lookup's value, or ``None`` when it fails (see ``_LOOKUP_ERRORS``)."""
    try:
        return lookup(*args)
    except _LOOKUP_ERRORS:
        return None
