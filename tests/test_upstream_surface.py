"""Upstream surface guard for the installed mlx-lm and mlx-vlm.

``checks/smoke.py`` and ``parity/worker.py`` reach ``mlx_lm`` and ``mlx_vlm`` only
through ``importlib``, so neither mypy nor an import-time error catches an upstream
rename: the call site would fail only at runtime, inside a model run. These tests pin
the parameters those call sites pass, and that importing ``mlx_vlm`` stays free of
audio I/O and of import-time warnings. The tests that import MLX are marked ``gpu``
and stay out of the pure-Python CI suite; the MLX imports happen inside those tests
so collection never needs MLX. The checker's own self-tests are pure Python and run
in the default suite.
"""

import ast
import importlib
import importlib.util
import inspect
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

POK = inspect.Parameter.POSITIONAL_OR_KEYWORD
VAR_KEYWORD = inspect.Parameter.VAR_KEYWORD

# sentencepiece 0.2.1's SWIG bindings raise "builtin type SwigPyPacked / SwigPyObject /
# swigvarlink has no __module__ attribute" (DeprecationWarning, Python 3.12+), the first
# two at import and the last at exit. Turned into errors they leave the extension
# half-initialized and the interpreter segfaults at exit, so the two tests that import
# mlx_lm / mlx_vlm ignore this one message family and nothing else. Warning filters
# match the message case-insensitively from its start.
SWIG_WARNING_PREFIX = "builtin type swig"


@dataclass(frozen=True, slots=True, kw_only=True)
class SurveyedCall:
    """One upstream call our code makes, with the parameters it relies on."""

    module: str
    attr: str
    positional: tuple[str, ...]
    keywords: tuple[str, ...]
    var_keyword: bool
    call_args: tuple[object, ...]
    call_kwargs: dict[str, object]


# Each entry mirrors one call site; the positional prefix is what we pass by position.
SURVEYED_CALLS = (
    # parity/worker.py: load(path, adapter_path=, tokenizer_config={...});
    # checks/smoke.py: load(local_dir, tokenizer_config={"trust_remote_code": False}).
    SurveyedCall(
        module="mlx_lm",
        attr="load",
        positional=("path_or_hf_repo",),
        keywords=("adapter_path", "tokenizer_config"),
        var_keyword=False,
        call_args=("repo",),
        call_kwargs={"adapter_path": None, "tokenizer_config": {"trust_remote_code": False}},
    ),
    # checks/smoke.py: generate(model, tokenizer, prompt=, max_tokens=8, verbose=False).
    SurveyedCall(
        module="mlx_lm",
        attr="generate",
        positional=("model", "tokenizer"),
        keywords=("prompt", "verbose"),
        var_keyword=True,
        call_args=("model", "tokenizer"),
        call_kwargs={"prompt": "p", "max_tokens": 8, "verbose": False},
    ),
    # checks/smoke.py: load(local_dir, trust_remote_code=False); the flag rides **kwargs.
    SurveyedCall(
        module="mlx_vlm",
        attr="load",
        positional=("path_or_hf_repo",),
        keywords=(),
        var_keyword=True,
        call_args=("repo",),
        call_kwargs={"trust_remote_code": False},
    ),
    # checks/smoke.py: apply_chat_template(processor, config, prompt, num_images=1).
    SurveyedCall(
        module="mlx_vlm",
        attr="apply_chat_template",
        positional=("processor", "config", "prompt"),
        keywords=("num_images",),
        var_keyword=True,
        call_args=("processor", None, "p"),
        call_kwargs={"num_images": 1},
    ),
    # checks/smoke.py: generate(model, processor, prompt, image, max_tokens=8, verbose=False).
    SurveyedCall(
        module="mlx_vlm",
        attr="generate",
        positional=("model", "processor", "prompt", "image"),
        keywords=("verbose",),
        var_keyword=True,
        call_args=("model", "processor", "p", "image"),
        call_kwargs={"max_tokens": 8, "verbose": False},
    ),
)


def _surface_problems(call: SurveyedCall, sig: inspect.Signature) -> list[str]:
    params = list(sig.parameters.values())
    by_name = sig.parameters
    problems: list[str] = []
    for index, name in enumerate(call.positional):
        if index >= len(params) or params[index].name != name or params[index].kind != POK:
            problems.append(f"positional #{index} is not {name!r} (positional-or-keyword)")
    problems.extend(
        f"{name!r} is not a positional-or-keyword parameter"
        for name in call.keywords
        if name not in by_name or by_name[name].kind != POK
    )
    has_var_keyword = any(p.kind == VAR_KEYWORD for p in params)
    if has_var_keyword != call.var_keyword:
        problems.append(f"**kwargs present is {has_var_keyword}, recorded {call.var_keyword}")
    try:
        sig.bind(*call.call_args, **call.call_kwargs)
    except TypeError as exc:
        problems.append(f"our call shape does not bind: {exc}")
    return problems


def _skip_unless_installed(module: str) -> None:
    """Skip, without importing it, when an optional extra's package is absent."""
    if importlib.util.find_spec(module) is None:
        extra = module.replace("_", "-")
        pytest.skip(f"{module} is not installed; install the [{extra}] extra to run this test")


@pytest.mark.gpu
@pytest.mark.filterwarnings(
    f"ignore:{SWIG_WARNING_PREFIX}.* has no __module__ attribute:DeprecationWarning"
)
@pytest.mark.parametrize("call", SURVEYED_CALLS, ids=lambda c: f"{c.module}.{c.attr}")
def test_upstream_signatures_match_the_surveyed_calls(call: SurveyedCall) -> None:
    _skip_unless_installed(call.module)
    func = getattr(importlib.import_module(call.module), call.attr)
    sig = inspect.signature(func)

    problems = _surface_problems(call, sig)

    assert not problems, f"{call.module}.{call.attr}{sig}: {problems}"


@pytest.mark.gpu
def test_importing_mlx_vlm_does_not_load_audio_io() -> None:
    _skip_unless_installed("mlx_vlm")
    probe = (
        "import sys, mlx_vlm; "
        "bad = [m for m in ('sounddevice', '_sounddevice', 'miniaudio') if m in sys.modules]; "
        "assert not bad, bad"
    )

    swig_ignore = f"ignore:{SWIG_WARNING_PREFIX}:DeprecationWarning"

    proc = subprocess.run(
        [sys.executable, "-W", "error", "-W", swig_ignore, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert proc.returncode == 0, proc.stderr[-4000:]


# --- Self-tests of the checker (pure Python, no MLX import) ---------------------

_GENERATE_CALL = next(c for c in SURVEYED_CALLS if (c.module, c.attr) == ("mlx_lm", "generate"))


def _generate_as_surveyed(
    model: object, tokenizer: object, prompt: str, verbose: bool = False, **kwargs: object
) -> None:
    """Stand-in with the parameters recorded for ``mlx_lm.generate``."""


def _generate_with_verbose_renamed(
    model: object, tokenizer: object, prompt: str, quiet: bool = False, **kwargs: object
) -> None:
    """Stand-in where an upstream release renamed ``verbose``; our call still binds."""


def test_surface_checker_finds_no_problem_in_a_matching_signature() -> None:
    # Bug caught: the checker reports a problem for a signature that matches the record,
    # so the gpu guard fails on every run and its failures stop meaning anything.
    assert _surface_problems(_GENERATE_CALL, inspect.signature(_generate_as_surveyed)) == []


def test_surface_checker_flags_a_renamed_keyword() -> None:
    # Bug caught: the keyword check is lost; a renamed ``verbose`` would then be swallowed
    # by ``**kwargs`` at runtime, and binding our call alone does not notice.
    problems = _surface_problems(_GENERATE_CALL, inspect.signature(_generate_with_verbose_renamed))

    assert problems == ["'verbose' is not a positional-or-keyword parameter"]


# --- the trust_remote_code pin reaches AutoTokenizer --------------------------------


class _TokenizerReachedError(Exception):
    """Raised by the stand-in AutoTokenizer once it has recorded its kwargs."""


def _from_pretrained_kwargs(
    utils: object, tokenizer_utils: object, monkeypatch: pytest.MonkeyPatch, model_dir: str
) -> dict[str, object]:
    """Call ``utils.load`` with the pin and return what AutoTokenizer received.

    The download and the model build are replaced, so nothing is fetched or loaded.
    """
    seen: dict[str, object] = {}

    class RecordingAutoTokenizer:
        @staticmethod
        def from_pretrained(path: object, **kwargs: object) -> object:
            seen.update(kwargs)
            raise _TokenizerReachedError

    monkeypatch.setattr(utils, "_download", lambda path, **_kw: __import__("pathlib").Path(path))
    monkeypatch.setattr(utils, "load_model", lambda *_a, **_kw: (object(), {}))
    monkeypatch.setattr(tokenizer_utils, "AutoTokenizer", RecordingAutoTokenizer)
    with pytest.raises(_TokenizerReachedError):
        utils.load(model_dir, tokenizer_config={"trust_remote_code": False})  # type: ignore[attr-defined]
    return seen


@pytest.mark.gpu
@pytest.mark.filterwarnings(
    f"ignore:{SWIG_WARNING_PREFIX}.* has no __module__ attribute:DeprecationWarning"
)
def test_mlx_lm_load_forwards_tokenizer_config_to_auto_tokenizer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    # Bug caught: an mlx-lm release stops forwarding load(tokenizer_config=...) to
    # AutoTokenizer.from_pretrained, so the trust_remote_code=False pin our smoke and
    # parity loads rely on (it is what keeps auto_map inert) silently drops.
    _skip_unless_installed("mlx_lm")
    utils = importlib.import_module("mlx_lm.utils")
    tokenizer_utils = importlib.import_module("mlx_lm.tokenizer_utils")

    seen = _from_pretrained_kwargs(utils, tokenizer_utils, monkeypatch, str(tmp_path))

    assert seen == {"trust_remote_code": False}


@pytest.mark.gpu
@pytest.mark.filterwarnings(
    f"ignore:{SWIG_WARNING_PREFIX}.* has no __module__ attribute:DeprecationWarning"
)
def test_the_forwarding_probe_notices_a_load_that_drops_tokenizer_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    # Bug caught: the probe reads its own stand-in instead of what load forwarded, so the
    # test above passes whatever mlx-lm does.
    _skip_unless_installed("mlx_lm")
    utils = importlib.import_module("mlx_lm.utils")
    tokenizer_utils = importlib.import_module("mlx_lm.tokenizer_utils")
    real_load_tokenizer = utils.load_tokenizer

    def dropping_load_tokenizer(model_path: object, *_a: object, **kw: object) -> object:
        return real_load_tokenizer(model_path, None, **kw)

    monkeypatch.setattr(utils, "load_tokenizer", dropping_load_tokenizer)

    seen = _from_pretrained_kwargs(utils, tokenizer_utils, monkeypatch, str(tmp_path))

    assert seen == {}


# --- the snapshot lists match each loader's own download list -----------------------


@pytest.mark.gpu
@pytest.mark.filterwarnings(
    f"ignore:{SWIG_WARNING_PREFIX}.* has no __module__ attribute:DeprecationWarning"
)
def test_text_snapshot_patterns_are_mlx_lm_download_patterns_without_py() -> None:
    # Bug caught: an mlx-lm release adds a download pattern (a new tokenizer file type)
    # that the `check hf --smoke` snapshot does not fetch, so the smoke load fails where
    # plain mlx-lm would succeed; or our list drops one of mlx-lm's patterns.
    _skip_unless_installed("mlx_lm")
    from mlx_model_doctor.checks.smoke import _MLX_LM_SNAPSHOT_PATTERNS

    upstream = importlib.import_module("mlx_lm.utils").DEFAULT_ALLOW_PATTERNS

    assert sorted(_MLX_LM_SNAPSHOT_PATTERNS) == sorted(set(upstream) - {"*.py"})


class _DownloadReachedError(Exception):
    """Raised by the stand-in snapshot_download once it has recorded its patterns."""


@pytest.mark.gpu
@pytest.mark.filterwarnings(
    f"ignore:{SWIG_WARNING_PREFIX}.* has no __module__ attribute:DeprecationWarning"
)
def test_vlm_snapshot_patterns_are_mlx_vlm_download_patterns_without_py(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Bug caught: an mlx-vlm release adds a download pattern (0.7.x added "*.jsonl" for
    # plamo2vl's tokenizer.jsonl) that the `check hf --smoke --plugin vlm` snapshot does
    # not fetch, so the smoke load misses a file plain mlx-vlm would have.
    _skip_unless_installed("mlx_vlm")
    from mlx_model_doctor.checks.smoke import _MLX_VLM_SNAPSHOT_PATTERNS

    utils = importlib.import_module("mlx_vlm.utils")
    seen: list[str] = []

    def recording_snapshot_download(**kwargs: object) -> str:
        patterns = kwargs["allow_patterns"]
        assert isinstance(patterns, list)
        seen.extend(patterns)
        raise _DownloadReachedError

    monkeypatch.setattr(utils, "snapshot_download", recording_snapshot_download)
    with pytest.raises(_DownloadReachedError):
        utils.get_model_path(str(tmp_path / "not-a-local-folder"))

    assert sorted(_MLX_VLM_SNAPSHOT_PATTERNS) == sorted(set(seen) - {"*.py"})


# --- no unsurveyed jinja2 renderer in mlx-vlm ------------------------------------

# The mlx_vlm modules that call jinja2 Template(, Environment( or .from_string(,
# surveyed on 0.6.12, 0.7.10 and 0.7.12 (2026-11-16): each renderer module, with the
# VLM_TEMPLATE_RENDERERS family it backs.
SURVEYED_RENDERER_MODULES = {
    "models/molmo/processing_molmo.py": "molmo",
    "models/molmo2/processing.py": "molmo2",
    "models/ernie4_5_moe_vl/processing_ernie4_5_moe_vl.py": "ernie4_5_moe_vl",
    "models/phi3_v/processing_phi3_v.py": "phi3_v",
    "models/kimi_vl/processing_kimi_vl.py": "kimi_vl",
    "models/locateanything/processing_locateanything.py": "locateanything",
}
# Surveyed in the same pass and not a renderer: 0.7.12 server code that only calls
# Environment(...).parse(template) (it parses, never renders) and that load() does not
# import.
SURVEYED_NON_RENDERING_MODULES = frozenset({"server/model_discovery.py"})

_UNSANDBOXED_JINJA = re.compile(
    r"\bTemplate\(|\bEnvironment\(|\.from_string\(|\bimport jinja2\b|\bfrom jinja2\b"
)


def _unsandboxed_jinja_modules(root: Path) -> list[str]:
    """Return the .py files under ``root`` that import jinja2 or call a plain renderer.

    An import counts, so a renderer reached through an alias (``import jinja2 as j``,
    ``from jinja2 import Template as T``) is not missed. ``SandboxedEnvironment(`` does
    not match: ``Environment`` is not a word there. A module that imports jinja2 only to
    sandbox it would match; none does in the surveyed releases.
    """
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if _UNSANDBOXED_JINJA.search(path.read_text(encoding="utf-8"))
    )


def test_the_surveyed_renderer_modules_back_exactly_the_renderer_table() -> None:
    # Bug caught: a family is added to (or dropped from) VLM_TEMPLATE_RENDERERS without
    # the survey below, so the scan's expected set no longer describes the table.
    from mlx_model_doctor.checks.remote_code import VLM_TEMPLATE_RENDERERS

    assert set(SURVEYED_RENDERER_MODULES.values()) == set(VLM_TEMPLATE_RENDERERS.values())


def test_the_jinja_scan_finds_plain_renderers_and_skips_the_sandbox(tmp_path: Path) -> None:
    # Bug caught: the scan misses a Template(, Environment( or from_string( call, or
    # counts transformers' SandboxedEnvironment, so the gpu scan below proves nothing.
    files = {
        "a/template.py": "from jinja2 import Template\nTemplate(t).render()\n",
        "b/env.py": "env = jinja2.Environment(extensions=[])\n",
        "c/from_string.py": "tpl = env.from_string(chat_template)\n",
        "d/sandbox.py": "env = SandboxedEnvironment(trim_blocks=True)\n",
        "e/plain.py": "x = 1\n",
        "f/import_alias.py": "import jinja2 as j2\nT = j2.Template\nT(t).render()\n",
        "g/from_alias.py": "from jinja2 import Template as T\nT(t).render()\n",
    }
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True)
        (tmp_path / name).write_text(text, encoding="utf-8")

    found = _unsandboxed_jinja_modules(tmp_path)

    assert found == [
        "a/template.py",
        "b/env.py",
        "c/from_string.py",
        "f/import_alias.py",
        "g/from_alias.py",
    ]


def test_mlx_vlm_has_no_jinja_renderer_outside_the_surveyed_modules() -> None:
    # Bug caught: an mlx-vlm release inside the admitted range adds a processor that
    # renders the repository's chat template with a plain jinja2 Template or
    # Environment; VLM_TEMPLATE_RENDERERS does not list it, so `--smoke` loads it.
    spec = importlib.util.find_spec("mlx_vlm")
    if spec is None or spec.origin is None:
        pytest.skip("mlx_vlm is not installed; install the [mlx-vlm] extra to run this test")
    root = Path(spec.origin).parent
    surveyed = set(SURVEYED_RENDERER_MODULES) | SURVEYED_NON_RENDERING_MODULES

    unsurveyed = [m for m in _unsandboxed_jinja_modules(root) if m not in surveyed]

    assert not unsurveyed, (
        f"mlx_vlm modules call a plain jinja2 renderer outside the surveyed set: "
        f"{unsurveyed}; the renderer table (VLM_TEMPLATE_RENDERERS) needs a fresh survey"
    )


# --- every model_type that selects a surveyed renderer is in the table -----------

# The processor class behind each VLM_TEMPLATE_RENDERERS family, defined in the module
# SURVEYED_RENDERER_MODULES names for it (same survey). Molmo2Processor is selected by
# molmo2's own AutoProcessor patch, not by install_auto_processor_patch.
RENDERER_PROCESSOR_CLASSES = {
    "MolmoProcessor": "molmo",
    "Molmo2Processor": "molmo2",
    "Ernie4_5_VLProcessor": "ernie4_5_moe_vl",
    "Phi3VProcessor": "phi3_v",
    "KimiVLProcessor": "kimi_vl",
    "LocateAnythingProcessor": "locateanything",
}


_PATCH_FUNCTION = "install_auto_processor_patch"
_FAMILY_CLASS = {family: name for name, family in RENDERER_PROCESSOR_CLASSES.items()}


def _imported_names(tree: ast.Module) -> dict[str, str]:
    """Map each name a module imports to what it stands for upstream.

    ``as`` aliases resolve to the imported name, and a renderer family package's
    ``Processor`` export (``from ..kimi_vl import Processor``) to that family's class.
    """
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            package = (node.module or "").rsplit(".", 1)[-1]
            for alias in node.names:
                original = alias.name
                if original == "Processor" and package in _FAMILY_CLASS:
                    original = _FAMILY_CLASS[package]
                names[alias.asname or alias.name] = original
    return names


def _last_name(node: ast.expr | None) -> str | None:
    """Return the name a ``Name`` or the attribute an ``Attribute`` ends with."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _argument(call: ast.Call, index: int, keyword: str) -> ast.expr | None:
    """Return the call's argument at ``index``, or the one passed as ``keyword``."""
    if len(call.args) > index:
        return call.args[index]
    return next((kw.value for kw in call.keywords if kw.arg == keyword), None)


def _model_types(node: ast.expr | None) -> list[str]:
    """Return the lowercased model types a literal names; a non-literal is reported as is."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value.lower()]
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return [model_type for element in node.elts for model_type in _model_types(element)]
    return [f"<not a literal: {ast.unparse(node) if node is not None else 'missing'}>"]


def _renderer_registrations(root: Path) -> list[tuple[str, str, str]]:
    """Return (file, model_type, family) for each renderer registration under ``root``.

    A registration is an ``install_auto_processor_patch(<types>, <class>)`` call whose
    class resolves to one of RENDERER_PROCESSOR_CLASSES; sorted.
    """
    found: list[tuple[str, str, str]] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = _imported_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = _last_name(node.func)
            if function is None or names.get(function, function) != _PATCH_FUNCTION:
                continue
            cls = _last_name(_argument(node, 1, "processor_cls"))
            family = RENDERER_PROCESSOR_CLASSES.get(names.get(cls, cls)) if cls else None
            if family is None:
                continue
            file = path.relative_to(root).as_posix()
            types = _model_types(_argument(node, 0, "target_model_types"))
            found.extend((file, model_type, family) for model_type in types)
    return sorted(found)


def _unlisted_registrations(
    registrations: list[tuple[str, str, str]], table: Mapping[str, str]
) -> list[tuple[str, str, str]]:
    """Return the registrations whose model_type does not select their family in ``table``."""
    return [entry for entry in registrations if table.get(entry[1]) != entry[2]]


def test_the_registration_scan_follows_reuse_aliases_and_package_exports(
    tmp_path: Path,
) -> None:
    # Bug caught: the scan reads only the class name written at the call, so a surveyed
    # processor imported under another name, through its package's "Processor" export or
    # by keyword, is registered under a new model_type the renderer table does not list.
    files = {
        "models/kimi_vl/processing_kimi_vl.py": (
            "class KimiVLProcessor: ...\ninstall_auto_processor_patch('kimi_vl', KimiVLProcessor)\n"
        ),
        "models/new_vl/processing_new_vl.py": (
            "from ..kimi_vl.processing_kimi_vl import KimiVLProcessor as NewProcessor\n"
            "from ..base import install_auto_processor_patch\n"
            "install_auto_processor_patch(['new_vl', 'New-VL'], NewProcessor)\n"
        ),
        "models/la2/__init__.py": (
            "from ..locateanything import Processor\n"
            "from .. import base\n"
            "base.install_auto_processor_patch(processor_cls=Processor, target_model_types='la2')\n"
        ),
        "models/phi_alias/p.py": (
            "from ..base import install_auto_processor_patch as patch\n"
            "from ..phi3_v import processing_phi3_v as pp\n"
            "patch(\n    ('phi_alias',),\n    pp.Phi3VProcessor,\n)\n"
        ),
        "models/other/p.py": "install_auto_processor_patch('other', OtherProcessor)\n",
    }
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True)
        (tmp_path / name).write_text(text, encoding="utf-8")

    found = _renderer_registrations(tmp_path)

    assert found == [
        ("models/kimi_vl/processing_kimi_vl.py", "kimi_vl", "kimi_vl"),
        ("models/la2/__init__.py", "la2", "locateanything"),
        ("models/new_vl/processing_new_vl.py", "new-vl", "kimi_vl"),
        ("models/new_vl/processing_new_vl.py", "new_vl", "kimi_vl"),
        ("models/phi_alias/p.py", "phi_alias", "phi3_v"),
    ]


def test_a_reuse_registration_under_a_new_type_is_flagged_against_the_table(
    tmp_path: Path,
) -> None:
    # Bug caught: the gap check compares against the wrong table or never flags, so the
    # installed-tree check below passes whatever mlx-vlm registers.
    from mlx_model_doctor.checks.remote_code import VLM_TEMPLATE_RENDERERS

    module = tmp_path / "models" / "reuse" / "processing_reuse.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "from ..molmo.processing_molmo import MolmoProcessor\n"
        "install_auto_processor_patch(['molmo', 'molmo_reuse'], MolmoProcessor)\n",
        encoding="utf-8",
    )

    gaps = _unlisted_registrations(_renderer_registrations(tmp_path), VLM_TEMPLATE_RENDERERS)

    assert gaps == [("models/reuse/processing_reuse.py", "molmo_reuse", "molmo")]


def test_every_mlx_vlm_registration_of_a_renderer_processor_is_in_the_table() -> None:
    # Bug caught: an mlx-vlm release registers a surveyed renderer processor under a new
    # model_type (a reuse, or an alias); VLM_TEMPLATE_RENDERERS does not list that type,
    # so `--smoke` loads a repository whose chat template it renders unsandboxed.
    from mlx_model_doctor.checks.remote_code import VLM_TEMPLATE_RENDERERS

    spec = importlib.util.find_spec("mlx_vlm")
    if spec is None or spec.origin is None:
        pytest.skip("mlx_vlm is not installed; install the [mlx-vlm] extra to run this test")
    root = Path(spec.origin).parent
    undefined = [
        name
        for name, family in RENDERER_PROCESSOR_CLASSES.items()
        if not re.search(
            rf"^class {name}\(",
            (root / _module_of_family(family)).read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    ]
    registrations = _renderer_registrations(root)

    assert not undefined, f"renderer classes not defined where surveyed: {undefined}"
    assert registrations, "no renderer registration found; the scan reads nothing"
    assert not _unlisted_registrations(registrations, VLM_TEMPLATE_RENDERERS), (
        f"mlx_vlm registers a renderer processor under a model_type the renderer table "
        f"(VLM_TEMPLATE_RENDERERS) does not list: "
        f"{_unlisted_registrations(registrations, VLM_TEMPLATE_RENDERERS)}"
    )


def _module_of_family(family: str) -> str:
    """Return the surveyed module that defines ``family``'s processor class."""
    return next(path for path, fam in SURVEYED_RENDERER_MODULES.items() if fam == family)


@pytest.mark.parametrize(
    "test",
    [
        test_mlx_vlm_has_no_jinja_renderer_outside_the_surveyed_modules,
        test_every_mlx_vlm_registration_of_a_renderer_processor_is_in_the_table,
    ],
    ids=lambda t: t.__name__,
)
def test_the_source_only_renderer_scans_run_without_the_gpu_marker(test: object) -> None:
    # Bug caught: a scan that reads mlx_vlm source text and imports nothing carries the
    # gpu marker, so the default suite (-m "not gpu") never runs it and an mlx-vlm
    # upgrade that adds a renderer goes unnoticed.
    marks = [mark.name for mark in getattr(test, "pytestmark", [])]

    assert "gpu" not in marks


# --- no call to mlx-lm's sharded loader ------------------------------------------

_SHARDED_LOADER = "sharded" + "_load"


def _files_naming(root: "Path", word: str) -> list[str]:
    """Return the .py files under ``root`` whose text contains ``word``, sorted."""
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if word in path.read_text(encoding="utf-8")
    )


def test_the_package_never_names_mlx_lm_sharded_load() -> None:
    # Bug caught: a call to mlx_lm.sharded_load is added; it defaults tokenizer_config
    # to {"trust_remote_code": True} (mlx-lm 0.31.3 utils.py:576), which would make an
    # auto_map in the repository importable on a path the guard treats as mlx-lm.
    src = Path(__file__).resolve().parent.parent / "src" / "mlx_model_doctor"

    assert _files_naming(src, _SHARDED_LOADER) == []


def test_the_sharded_load_scan_finds_a_file_that_names_it(tmp_path: "Path") -> None:
    # Bug caught: the scan looks at the wrong files or never reads them, so the test
    # above passes whatever the package calls.
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text(f"mlx_lm.{_SHARDED_LOADER}(repo)\n", encoding="utf-8")

    assert _files_naming(tmp_path, _SHARDED_LOADER) == ["pkg/b.py"]
