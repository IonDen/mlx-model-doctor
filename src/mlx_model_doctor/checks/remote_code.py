"""Refuse a model directory whose metadata names repository code for the loader to run."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal


@dataclass(frozen=True, slots=True, kw_only=True)
class RemoteCodeKeys:
    """The (file, key) pairs that make one loader import repository code.

    A key counts at any depth of its file (dicts and lists), and its value is never
    inspected; adding a key is one row.
    """

    loader: str
    keys: tuple[tuple[str, str], ...]


# mlx-lm: config.json "model_file" is imported with no trust_remote_code gate (mlx-lm
# 0.31.3 utils.py:325); mlx-lm 0.32.0 and later gate it, and the key stays because the
# base install does not pin mlx-lm. "auto_map" is not listed: transformers imports what it names only
# with trust_remote_code on, and every mlx-lm load here pins it off (the smoke load and
# the parity worker pass tokenizer_config={"trust_remote_code": False}; the --prompts
# tokenizer passes trust_remote_code=False). mlx-lm's distributed sharded loader turns it
# on by default and is never called (a test greps the package for it).
MLX_LM_KEYS = RemoteCodeKeys(loader="mlx-lm", keys=(("config.json", "model_file"),))

# mlx-vlm: "model_file" as above (mlx-vlm 0.6.16-0.7.12 utils.py:254, no gate). Some
# mlx-vlm processors force trust_remote_code on (0.6.12 inkling processing_inkling.py:366,
# 0.7.12 models/base.py:697), so every "auto_map" transformers 5.15 reads counts: the
# config, tokenizer, processor (and its image, video, feature-extractor and audio
# sub-configs), image processor and video processor files. "configuration_files" makes
# transformers re-read the config from config.<version>.json, a file this guard does not
# open (configuration_utils.py:752).
MLX_VLM_KEYS = RemoteCodeKeys(
    loader="mlx-vlm",
    keys=(
        ("config.json", "model_file"),
        ("config.json", "configuration_files"),
        ("config.json", "auto_map"),
        ("tokenizer_config.json", "auto_map"),
        ("processor_config.json", "auto_map"),
        ("preprocessor_config.json", "auto_map"),
        ("video_preprocessor_config.json", "auto_map"),
    ),
)

RefusalReason = Literal["names_code", "unreadable", "dangling_link", "not_a_directory"]

KEY_REMEDIATION = (
    "Do not load this repository with mlx-lm or mlx-vlm unless you trust its author. "
    "The static checks read it without running its code."
)


@dataclass(frozen=True, slots=True, kw_only=True)
class RemoteCodeRefusal:
    """Why a model directory was refused before loading.

    ``key`` is the key found in ``file`` (reason ``names_code``). It is None when
    ``file`` exists but could not be read as a JSON object (``unreadable``, or
    ``dangling_link`` for a link whose target is missing), so whether it names code is
    unknown, and when ``file`` is the model path itself and is not an existing directory
    (``not_a_directory``). ``loader`` is the loader whose key table was applied.
    """

    file: str
    key: str | None
    loader: str
    reason: RefusalReason


class RemoteCodeRefusedError(Exception):
    """The smoke load was refused because the repository may run its own code."""

    def __init__(self, refusal: RemoteCodeRefusal, *, revision: str | None) -> None:
        """Record the refusal and the snapshot revision it applies to, if known."""
        super().__init__(f"{refusal.file}: {refusal.key}")
        self.refusal = refusal
        self.revision = revision


def find_remote_code(
    model_dir: Path, *, keys: RemoteCodeKeys = MLX_VLM_KEYS
) -> RemoteCodeRefusal | None:
    """Return the first refusal reason for ``model_dir`` under ``keys``, or None.

    A path that is not an existing directory is refused. A guarded file that is absent
    is not a refusal. One that exists (a dangling link included) but cannot be read and
    parsed as a JSON object is: the guard fails closed.
    """
    if not model_dir.is_dir():
        return RemoteCodeRefusal(
            file=str(model_dir), key=None, loader=keys.loader, reason="not_a_directory"
        )
    parsed: dict[str, object] = {}
    for file, key in keys.keys:
        if file not in parsed:
            path = model_dir / file
            if not path.is_symlink() and not path.exists():
                continue
            data = _json_object(path)
            if data is None:
                reason: RefusalReason = (
                    "dangling_link" if path.is_symlink() and not path.exists() else "unreadable"
                )
                return RemoteCodeRefusal(file=file, key=None, loader=keys.loader, reason=reason)
            parsed[file] = data
        if _has_key(parsed[file], key):
            return RemoteCodeRefusal(file=file, key=key, loader=keys.loader, reason="names_code")
    return None


def describe_refusal(refusal: RemoteCodeRefusal, *, subject: str = "this repository") -> str:
    """Return the sentence that says what the guard found, about ``subject``."""
    if refusal.reason == "names_code":
        return (
            f"{subject[:1].upper()}{subject[1:]} names loader code ({refusal.key} in "
            f"{refusal.file}) that {refusal.loader} could import."
        )
    if refusal.reason == "not_a_directory":
        return (
            f"{refusal.file} is not an existing directory, so it is unknown whether "
            f"{subject} names loader code."
        )
    return (
        f"{refusal.file} could not be read (missing target, unreadable, or not a JSON "
        f"object), so it is unknown whether {subject} names loader code."
    )


def refusal_remediation(refusal: RemoteCodeRefusal) -> str:
    """Return what the user can do about ``refusal``."""
    if refusal.reason == "names_code":
        return KEY_REMEDIATION
    if refusal.reason == "not_a_directory":
        return "Pass an existing model directory, then run again."
    return f"Re-download or repair {refusal.file}, then run again."


# The files where transformers reads an auto_map: the "auto_map" rows of the VLM table.
_AUTO_MAP_FILES = tuple(file for file, key in MLX_VLM_KEYS.keys if key == "auto_map")


def find_auto_map(model_dir: Path) -> str | None:
    """Return the first file in ``model_dir`` that names ``auto_map`` at any depth, or None.

    Used after a failed mlx-lm load, to tell a load that needed repository code from
    any other failure. A file that is absent or unreadable does not count.
    """
    for file in _AUTO_MAP_FILES:
        data = _json_object(model_dir / file)
        if data is not None and _has_key(data, "auto_map"):
            return file
    return None


def describe_auto_map_load_failure(file: str) -> str:
    """Return the sentences that replace a failed mlx-lm load's upstream message."""
    return (
        f"This repository names tokenizer or model code (auto_map in {file}). mlx-lm was "
        "called with remote code turned off, so that code was not run, and the load could "
        "not finish without it."
    )


class AutoMapLoadError(Exception):
    """An mlx-lm load failed on a directory that names ``auto_map`` code."""

    def __init__(self, file: str, *, upstream_error: str, revision: str | None) -> None:
        """Record the file that names the code, the reduced upstream text and revision."""
        super().__init__(describe_auto_map_load_failure(file))
        self.file = file
        self.upstream_error = upstream_error
        self.revision = revision


# mlx-vlm 0.6.12-0.7.12, surveyed 2026-11-16: the processors that render a chat template
# with a plain jinja2 Template or Environment instead of transformers' sandbox, keyed by
# every config.json value that selects them. Each processing module patches
# AutoProcessor.from_pretrained to return its processor when the raw config.json
# "model_type", lowercased and not remapped, is in its list (models/base.py
# install_auto_processor_patch; molmo2 processing.py has its own "molmo2" patch). The
# phi3_v patch also lists "phi3-v" and "phi3v". The model package itself is imported
# from (model_type or speculators_model_type).lower() through MODEL_REMAPPING, and no
# remapping value is one of these packages. A processor_class key cannot select them:
# none is registered with AutoProcessor.
VLM_TEMPLATE_RENDERERS: Mapping[str, str] = MappingProxyType(
    {
        "molmo": "molmo",
        "molmo2": "molmo2",
        "ernie4_5_moe_vl": "ernie4_5_moe_vl",
        "phi3_v": "phi3_v",
        "phi3-v": "phi3_v",
        "phi3v": "phi3_v",
        "kimi_vl": "kimi_vl",
        "locateanything": "locateanything",
    }
)

# Where those processors and the transformers 5.15 tokenizer they build read a chat
# template, in the order they are checked: a top-level non-null "chat_template" key in a
# JSON file, then the chat_template.jinja file, then any *.jinja file under
# additional_chat_templates/ (transformers tokenization_utils_base.py, processing_utils.py).
# special_tokens_map.json is a source because, when tokenizer_config.json has no
# added_tokens_decoder or is absent, every one of its keys is merged into the tokenizer's
# init kwargs (tokenization_utils_base.py:1861-1880), which then take chat_template (:1078).
_TEMPLATE_KEY_FILES = (
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.json",
    "processor_config.json",
    "preprocessor_config.json",
)
_TEMPLATE_FILE = "chat_template.jinja"
_TEMPLATE_DIR = "additional_chat_templates"

TemplateRendererReason = Literal["ships_template", "unreadable"]


@dataclass(frozen=True, slots=True, kw_only=True)
class TemplateRendererRefusal:
    """A VLM repository whose own chat template an unsandboxed renderer would run.

    ``renderer`` is the mlx-vlm processor family, None when ``config.json`` could not be
    read. ``file`` is where the template was found (reason ``ships_template``; a folder
    ends in ``/``), or the file that could not be read (reason ``unreadable``).
    """

    renderer: str | None
    file: str
    reason: TemplateRendererReason


class TemplateRendererRefusedError(Exception):
    """The VLM smoke load was refused because an unsandboxed renderer would run a template."""

    def __init__(self, refusal: TemplateRendererRefusal, *, revision: str | None) -> None:
        """Record the refusal and the snapshot revision it applies to, if known."""
        super().__init__(f"{refusal.renderer}: {refusal.file}")
        self.refusal = refusal
        self.revision = revision


def find_template_renderer(model_dir: Path) -> TemplateRendererRefusal | None:
    """Return why ``model_dir`` reaches an unsandboxed template renderer, or None.

    A repository is refused when its ``config.json`` selects a processor from
    ``VLM_TEMPLATE_RENDERERS`` and it ships a chat template of its own. An absent
    ``config.json`` selects nothing; one that exists but cannot be read is refused, as
    is a template source of a selected family that cannot be read.
    """
    config = _read_guarded(model_dir / "config.json")
    if config is None:
        return None
    if isinstance(config, _Unreadable):
        return TemplateRendererRefusal(renderer=None, file="config.json", reason="unreadable")
    renderer = _selected_renderer(config)
    if renderer is None:
        return None
    for file in _TEMPLATE_KEY_FILES:
        data = _read_guarded(model_dir / file)
        if isinstance(data, _Unreadable):
            return TemplateRendererRefusal(renderer=renderer, file=file, reason="unreadable")
        if data is not None and data.get("chat_template") is not None:
            return TemplateRendererRefusal(renderer=renderer, file=file, reason="ships_template")
    jinja = model_dir / _TEMPLATE_FILE
    if jinja.exists():
        return TemplateRendererRefusal(
            renderer=renderer, file=_TEMPLATE_FILE, reason="ships_template"
        )
    if jinja.is_symlink():
        return TemplateRendererRefusal(renderer=renderer, file=_TEMPLATE_FILE, reason="unreadable")
    return _extra_templates(model_dir / _TEMPLATE_DIR, renderer=renderer)


def describe_template_renderer_refusal(refusal: TemplateRendererRefusal) -> str:
    """Return the sentences that say what the renderer guard found and what it did."""
    if refusal.renderer is None:
        return (
            f"{refusal.file} could not be read (missing target, unreadable, or not a JSON "
            "object), so it is unknown whether this repository uses a processor that "
            "renders its chat template without a sandbox. The load was refused, so no "
            "template was rendered."
        )
    uses = (
        f"This repository uses a `{refusal.renderer}` processor that renders the "
        "repository's chat template without a sandbox, and"
    )
    if refusal.reason == "unreadable":
        return (
            f"{uses} {refusal.file} could not be read, so it is unknown whether the "
            "repository ships its own template. The load was refused, so no template was "
            "rendered."
        )
    return (
        f"{uses} the repository ships its own template (`{refusal.file}`). The load was "
        "refused, so that template was not rendered."
    )


def template_renderer_remediation(refusal: TemplateRendererRefusal) -> str:
    """Return what the user can do about ``refusal``."""
    if refusal.reason == "unreadable":
        return f"Re-download or repair {refusal.file}, then run again."
    return KEY_REMEDIATION


def _selected_renderer(config: dict[str, object]) -> str | None:
    """Return the renderer family ``config`` selects, the way mlx-vlm normalises it.

    The processor patch compares ``str(model_type).lower()``; the package import uses
    ``speculators_model_type`` when ``model_type`` is empty. Both count.
    """
    model_type = config.get("model_type")
    selectors: list[object] = [model_type]
    if not model_type:
        selectors.append(config.get("speculators_model_type"))
    for value in selectors:
        if value is not None:
            family = VLM_TEMPLATE_RENDERERS.get(str(value).lower())
            if family is not None:
                return family
    return None


def _extra_templates(folder: Path, *, renderer: str) -> TemplateRendererRefusal | None:
    """Return a refusal when ``folder`` holds a ``*.jinja`` template, or cannot be listed."""
    if not folder.is_dir():
        return None
    where = f"{_TEMPLATE_DIR}/"
    try:
        names = [entry.name for entry in folder.iterdir()]
    except OSError:
        return TemplateRendererRefusal(renderer=renderer, file=where, reason="unreadable")
    if any(name.endswith(".jinja") for name in names):
        return TemplateRendererRefusal(renderer=renderer, file=where, reason="ships_template")
    return None


class _Unreadable:
    """Marks a file that exists (a dangling link included) but is not a JSON object."""


def _read_guarded(path: Path) -> dict[str, object] | _Unreadable | None:
    """Return the JSON object in ``path``, None when absent, or ``_Unreadable``."""
    if not path.is_symlink() and not path.exists():
        return None
    data = _json_object(path)
    return _Unreadable() if data is None else data


def _has_key(data: object, key: str) -> bool:
    """Return whether ``key`` is a key of any object nested in ``data``.

    Walks with an explicit stack, so a file that parsed cannot overflow the walk.
    """
    stack = [data]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if key in node:
                return True
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return False


def _json_object(path: Path) -> dict[str, object] | None:
    try:
        data: object = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None
