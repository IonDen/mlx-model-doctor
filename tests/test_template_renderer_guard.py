"""VLM smoke refuses a repository whose own chat template an unsandboxed renderer would run.

Six mlx-vlm processors (molmo, molmo2, ernie4_5_moe_vl, phi3_v, kimi_vl, locateanything,
mlx-vlm 0.6.12 and 0.7.12) render the chat template with a plain jinja2 ``Template`` or
``Environment``, not transformers' sandbox. They are picked by the raw ``config.json``
``model_type``, lowercased (``models/base.py`` ``install_auto_processor_patch``). Every
test uses a real model directory and a fake loader that records its calls.
"""

import os
from pathlib import Path

import pytest

import mlx_model_doctor.checks.smoke as smoke_module
from mlx_model_doctor.checks.remote_code import (
    TemplateRendererRefusal,
    find_template_renderer,
)
from mlx_model_doctor.targets import HfTarget, LocalTarget
from tests.fakes import write_hf_cache_snapshot
from tests.test_remote_code_guard import KEY_REMEDIATION, FakeSnapshotHub, as_json, run_smoke

TEMPLATE = "{% for m in messages %}{{ m['content'] }}{% endfor %}"


def write_repo(root: Path, files: dict[str, bytes]) -> Path:
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


def config(model_type: str) -> bytes:
    return as_json({"model_type": model_type, "hidden_size": 64})


def with_tokenizer_template(model_type: str) -> dict[str, bytes]:
    return {
        "config.json": config(model_type),
        "tokenizer_config.json": as_json({"chat_template": TEMPLATE}),
    }


def refusal_message(family: str, where: str) -> str:
    return (
        f"This repository uses a `{family}` processor that renders the repository's chat "
        f"template without a sandbox, and the repository ships its own template "
        f"(`{where}`). The load was refused, so that template was not rendered. The static "
        "checks above still ran."
    )


# --- refused: each family, keyed on config.json model_type ---------------------------

FAMILIES = ["molmo", "molmo2", "ernie4_5_moe_vl", "phi3_v", "kimi_vl", "locateanything"]


@pytest.mark.parametrize("family", FAMILIES)
def test_vlm_smoke_refuses_a_renderer_family_that_ships_a_tokenizer_template(
    family: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = write_repo(tmp_path / "model", with_tokenizer_template(family))

    run = run_smoke("vlm", LocalTarget(model), monkeypatch)

    assert run.loader.load_paths == []
    assert run.result.status == "fail"
    assert run.result.severity == "high"
    assert run.result.check_id == "vlm/smoke.mlx_vlm"
    assert run.result.message == refusal_message(family, "tokenizer_config.json")
    assert run.result.remediation == KEY_REMEDIATION
    assert run.result.details["remote_code_renderer"] == family
    assert run.result.details["remote_code_template_source"] == "tokenizer_config.json"


# The phi3_v patch also intercepts "phi3-v" and "phi3v"; every patch lowercases the raw
# value; the package import falls back to speculators_model_type when model_type is empty.
ALIASES = [
    pytest.param({"model_type": "phi3-v"}, "phi3_v", id="phi3-v"),
    pytest.param({"model_type": "phi3v"}, "phi3_v", id="phi3v"),
    pytest.param({"model_type": "Phi3_V"}, "phi3_v", id="mixed-case"),
    pytest.param({"model_type": "MOLMO2"}, "molmo2", id="upper-case"),
    pytest.param({"speculators_model_type": "kimi_vl"}, "kimi_vl", id="speculators-only"),
    pytest.param(
        {"model_type": "", "speculators_model_type": "Molmo"}, "molmo", id="speculators-empty"
    ),
]


@pytest.mark.parametrize(("selector", "family"), ALIASES)
def test_vlm_smoke_refuses_every_name_that_selects_a_renderer(
    selector: dict[str, str], family: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {
        "config.json": as_json({**selector, "hidden_size": 64}),
        "tokenizer_config.json": as_json({"chat_template": TEMPLATE}),
    }
    model = write_repo(tmp_path / "model", files)

    run = run_smoke("vlm", LocalTarget(model), monkeypatch)

    assert run.loader.load_paths == []
    assert run.result.details["remote_code_renderer"] == family


def test_vlm_smoke_ignores_speculators_model_type_when_model_type_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: speculators_model_type is read even when model_type is set, so a qwen2_vl
    # repository is refused for a renderer mlx-vlm never selects (mlx-vlm imports
    # (model_type or speculators_model_type), and its processor patch reads model_type).
    files = {
        "config.json": as_json(
            {"model_type": "qwen2_vl", "speculators_model_type": "kimi_vl", "hidden_size": 64}
        ),
        "tokenizer_config.json": as_json({"chat_template": TEMPLATE}),
    }
    model = write_repo(tmp_path / "model", files)

    run = run_smoke("vlm", LocalTarget(model), monkeypatch)

    assert run.loader.load_paths == [str(model.resolve())]
    assert run.result.status == "pass"


# A model_type that is set but is not a string: mlx-vlm 0.7.12 get_model_type calls
# (model_type or speculators_model_type).lower() and fails before any processor is built,
# and the processor patch compares str(model_type).lower(), which names no family. The
# guard reads it the same way: no family, and speculators_model_type is not consulted.
NON_STRING_MODEL_TYPES = [
    pytest.param({"name": "molmo"}, id="nested-object"),
    pytest.param(["molmo"], id="list"),
    pytest.param(7, id="number"),
    pytest.param(True, id="true"),
]


@pytest.mark.parametrize("model_type", NON_STRING_MODEL_TYPES)
def test_a_non_string_model_type_selects_no_renderer(model_type: object, tmp_path: Path) -> None:
    # Bug: a non-string model_type is treated as empty, so speculators_model_type selects
    # a renderer for a repository mlx-vlm cannot load at all.
    model = write_repo(
        tmp_path / "model",
        {
            "config.json": as_json({"model_type": model_type, "speculators_model_type": "molmo"}),
            "tokenizer_config.json": as_json({"chat_template": TEMPLATE}),
        },
    )

    assert find_template_renderer(model) is None


# A model_type that is false, zero or empty falls through to speculators_model_type, as
# mlx-vlm's (model_type or speculators_model_type) does.
FALSY_MODEL_TYPES = [
    pytest.param(False, id="false"),
    pytest.param(0, id="zero"),
    pytest.param([], id="empty-list"),
    pytest.param({}, id="empty-object"),
]


@pytest.mark.parametrize("model_type", FALSY_MODEL_TYPES)
def test_a_falsy_model_type_falls_back_to_speculators_model_type(
    model_type: object, tmp_path: Path
) -> None:
    # Bug: only a missing or empty-string model_type falls back, so a repository with
    # "model_type": false and a renderer in speculators_model_type is not refused.
    model = write_repo(
        tmp_path / "model",
        {
            "config.json": as_json({"model_type": model_type, "speculators_model_type": "molmo"}),
            "tokenizer_config.json": as_json({"chat_template": TEMPLATE}),
        },
    )

    assert find_template_renderer(model) == TemplateRendererRefusal(
        renderer="molmo", file="tokenizer_config.json", reason="ships_template"
    )


# --- refused: each place a template is read from ---------------------------------------

SOURCES = [
    pytest.param(
        {"chat_template.json": as_json({"chat_template": TEMPLATE})},
        "chat_template.json",
        id="chat_template.json",
    ),
    pytest.param({"chat_template.jinja": TEMPLATE.encode()}, "chat_template.jinja", id="jinja"),
    pytest.param(
        {"processor_config.json": as_json({"chat_template": TEMPLATE})},
        "processor_config.json",
        id="processor_config.json",
    ),
    pytest.param(
        {"preprocessor_config.json": as_json({"chat_template": TEMPLATE})},
        "preprocessor_config.json",
        id="preprocessor_config.json",
    ),
    pytest.param(
        {"additional_chat_templates/tool_use.jinja": TEMPLATE.encode()},
        "additional_chat_templates/",
        id="additional_chat_templates",
    ),
    # transformers 5.15 merges every special_tokens_map.json key into the tokenizer's
    # init kwargs when tokenizer_config.json has no added_tokens_decoder.
    pytest.param(
        {"special_tokens_map.json": as_json({"chat_template": TEMPLATE})},
        "special_tokens_map.json",
        id="special_tokens_map.json",
    ),
]


# A list of named templates and an empty string are both non-null values the tokenizer
# takes as its chat template, so neither may pass as "no template of its own".
NON_STRING_OR_EMPTY = [
    pytest.param([{"name": "default", "template": TEMPLATE}], id="named-list"),
    pytest.param("", id="empty-string"),
]


@pytest.mark.parametrize("value", NON_STRING_OR_EMPTY)
def test_a_list_or_empty_tokenizer_template_of_a_renderer_family_is_refused(
    value: object, tmp_path: Path
) -> None:
    model = write_repo(
        tmp_path / "model",
        {
            "config.json": config("molmo2"),
            "tokenizer_config.json": as_json({"chat_template": value}),
        },
    )

    assert find_template_renderer(model) == TemplateRendererRefusal(
        renderer="molmo2", file="tokenizer_config.json", reason="ships_template"
    )


@pytest.mark.parametrize(("files", "where"), SOURCES)
def test_vlm_smoke_refuses_a_template_from_every_source(
    files: dict[str, bytes], where: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = write_repo(tmp_path / "model", {"config.json": config("locateanything"), **files})

    run = run_smoke("vlm", LocalTarget(model), monkeypatch)

    assert run.loader.load_paths == []
    assert run.result.message == refusal_message("locateanything", where)
    assert run.result.details["remote_code_template_source"] == where


# --- not refused -----------------------------------------------------------------------

NO_TEMPLATE = [
    pytest.param({}, id="no-template-files"),
    pytest.param({"tokenizer_config.json": as_json({"chat_template": None})}, id="null"),
    pytest.param(
        {"tokenizer_config.json": as_json({"model_max_length": 8})}, id="tokenizer-no-key"
    ),
    pytest.param({"chat_template.json": as_json({"other": TEMPLATE})}, id="json-no-key"),
    pytest.param({"additional_chat_templates/notes.txt": b"x"}, id="extra-dir-no-jinja"),
]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("files", NO_TEMPLATE)
def test_vlm_smoke_loads_a_renderer_family_with_no_template_of_its_own(
    family: str, files: dict[str, bytes], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = write_repo(tmp_path / "model", {"config.json": config(family), **files})

    run = run_smoke("vlm", LocalTarget(model), monkeypatch)

    assert run.loader.load_paths == [str(model.resolve())]
    assert run.result.status == "pass"


@pytest.mark.parametrize("model_type", ["qwen2_vl", "llava", "phi3", "kimi_k2", "molmo_point"])
@pytest.mark.parametrize(("files", "where"), SOURCES)
def test_vlm_smoke_loads_another_family_that_ships_a_template(
    model_type: str,
    files: dict[str, bytes],
    where: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = write_repo(tmp_path / "model", {"config.json": config(model_type), **files})

    run = run_smoke("vlm", LocalTarget(model), monkeypatch)

    assert run.loader.load_paths == [str(model.resolve())]
    assert run.result.status == "pass"


def test_text_smoke_does_not_apply_the_renderer_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = write_repo(tmp_path / "model", with_tokenizer_template("kimi_vl"))

    run = run_smoke("text", LocalTarget(model), monkeypatch)

    assert run.loader.load_paths == [str(model.resolve())]
    assert run.result.status == "pass"


# --- fail closed -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "data", [b"{not json", b"[1, 2]", b"\xff\xfe"], ids=["invalid", "list", "not-utf8"]
)
def test_an_unreadable_selector_file_is_refused(data: bytes, tmp_path: Path) -> None:
    model = write_repo(tmp_path / "model", {"config.json": data})

    assert find_template_renderer(model) == TemplateRendererRefusal(
        renderer=None, file="config.json", reason="unreadable"
    )


def test_a_dangling_selector_file_is_refused(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").symlink_to(tmp_path / "gone.json")

    assert find_template_renderer(model) == TemplateRendererRefusal(
        renderer=None, file="config.json", reason="unreadable"
    )


def test_an_absent_selector_file_is_not_refused(tmp_path: Path) -> None:
    model = write_repo(tmp_path / "model", {"chat_template.jinja": TEMPLATE.encode()})

    assert find_template_renderer(model) is None


def test_vlm_smoke_refuses_a_renderer_family_whose_template_file_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = write_repo(
        tmp_path / "model",
        {"config.json": config("molmo"), "chat_template.json": b"{not json"},
    )

    run = run_smoke("vlm", LocalTarget(model), monkeypatch)

    assert run.loader.load_paths == []
    assert run.result.status == "fail"
    assert run.result.message == (
        "This repository uses a `molmo` processor that renders the repository's chat "
        "template without a sandbox, and chat_template.json could not be read, so it is "
        "unknown whether the repository ships its own template. The load was refused, so "
        "no template was rendered. The static checks above still ran."
    )
    assert run.result.remediation == "Re-download or repair chat_template.json, then run again."
    assert run.result.details["remote_code_renderer"] == "molmo"
    assert run.result.details["remote_code_file"] == "chat_template.json"
    assert run.result.details["remote_code_unreadable"] is True


def test_an_unreadable_special_tokens_map_of_a_renderer_family_is_refused(
    tmp_path: Path,
) -> None:
    model = write_repo(
        tmp_path / "model",
        {"config.json": config("kimi_vl"), "special_tokens_map.json": b"{not json"},
    )

    assert find_template_renderer(model) == TemplateRendererRefusal(
        renderer="kimi_vl", file="special_tokens_map.json", reason="unreadable"
    )


def test_vlm_smoke_reports_a_selector_file_that_became_unreadable_after_the_key_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The key guard reads config.json first and refuses an unreadable one, so this
    # result is reached only when the file changes between the two reads.
    model = write_repo(tmp_path / "model", {"config.json": config("llava")})
    unreadable = TemplateRendererRefusal(renderer=None, file="config.json", reason="unreadable")
    monkeypatch.setattr(smoke_module, "find_template_renderer", lambda _dir: unreadable)

    run = run_smoke("vlm", LocalTarget(model), monkeypatch)

    assert run.loader.load_paths == []
    assert run.result.status == "fail"
    assert run.result.message == (
        "config.json could not be read (missing target, unreadable, or not a JSON object), "
        "so it is unknown whether this repository uses a processor that renders its chat "
        "template without a sandbox. The load was refused, so no template was rendered. "
        "The static checks above still ran."
    )
    assert run.result.remediation == "Re-download or repair config.json, then run again."
    assert dict(run.result.details) == {
        "remote_code_file": "config.json",
        "remote_code_unreadable": True,
    }


def test_a_dangling_template_file_of_a_renderer_family_is_refused(tmp_path: Path) -> None:
    model = write_repo(tmp_path / "model", {"config.json": config("phi3_v")})
    (model / "chat_template.jinja").symlink_to(tmp_path / "gone.jinja")

    assert find_template_renderer(model) == TemplateRendererRefusal(
        renderer="phi3_v", file="chat_template.jinja", reason="unreadable"
    )


@pytest.mark.skipif(os.geteuid() == 0, reason="root lists a folder whose mode is 0o000")
def test_vlm_smoke_refuses_a_renderer_family_whose_extra_templates_folder_cannot_be_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: an OSError from listing additional_chat_templates/ is read as "no template",
    # so a folder the guard could not read is loaded and its *.jinja rendered.
    model = write_repo(
        tmp_path / "model",
        {
            "config.json": config("molmo"),
            "additional_chat_templates/tool_use.jinja": TEMPLATE.encode(),
        },
    )
    folder = model / "additional_chat_templates"
    folder.chmod(0o000)
    try:
        run = run_smoke("vlm", LocalTarget(model), monkeypatch)
    finally:
        folder.chmod(0o755)

    assert run.loader.load_paths == []
    assert run.result.status == "fail"
    assert dict(run.result.details) == {
        "remote_code_renderer": "molmo",
        "remote_code_file": "additional_chat_templates/",
        "remote_code_unreadable": True,
    }
    assert run.result.remediation == (
        "Re-download or repair additional_chat_templates/, then run again."
    )


# --- Hugging Face targets: the refusal applies to the materialized snapshot -----------

SNAPSHOT_COMMIT = "fedcba9876543210fedcba9876543210fedcba98"


def test_vlm_smoke_refuses_a_renderer_family_in_an_hf_snapshot_and_names_its_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: the renderer refusal drops the snapshot revision, so the report does not say
    # which commit of the Hub repository was refused; or the HF path skips the guard.
    snapshot = write_hf_cache_snapshot(
        tmp_path / "hub", "org/vlm", with_tokenizer_template("kimi_vl"), commit=SNAPSHOT_COMMIT
    )
    hub = FakeSnapshotHub(snapshot=snapshot)

    run = run_smoke("vlm", HfTarget("org/vlm", hub=hub), monkeypatch)

    assert hub.snapshot_calls == ["org/vlm"]
    assert run.loader.load_paths == []
    assert run.result.status == "fail"
    assert run.result.message == refusal_message("kimi_vl", "tokenizer_config.json")
    assert dict(run.result.details) == {
        "remote_code_renderer": "kimi_vl",
        "remote_code_template_source": "tokenizer_config.json",
        "snapshot_revision": "fedcba9876543210fedcba9876543210fedcba98",
    }
