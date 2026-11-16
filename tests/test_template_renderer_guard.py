"""VLM smoke refuses a repository whose own chat template an unsandboxed renderer would run.

Six mlx-vlm processors (molmo, molmo2, ernie4_5_moe_vl, phi3_v, kimi_vl, locateanything,
mlx-vlm 0.6.12 and 0.7.12) render the chat template with a plain jinja2 ``Template`` or
``Environment``, not transformers' sandbox. They are picked by the raw ``config.json``
``model_type``, lowercased (``models/base.py`` ``install_auto_processor_patch``). Every
test uses a real model directory and a fake loader that records its calls.
"""

from pathlib import Path

import pytest

import mlx_model_doctor.checks.smoke as smoke_module
from mlx_model_doctor.checks.remote_code import (
    TemplateRendererRefusal,
    find_template_renderer,
)
from mlx_model_doctor.targets import LocalTarget
from tests.test_remote_code_guard import KEY_REMEDIATION, as_json, run_smoke

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
