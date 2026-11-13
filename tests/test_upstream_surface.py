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

import importlib
import importlib.util
import inspect
import subprocess
import sys
from dataclasses import dataclass

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
    # checks/smoke.py: load(name).
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
    # checks/smoke.py: load(name, trust_remote_code=False); the flag rides **kwargs.
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
