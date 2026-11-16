"""The smoke-path guard: a repository that names its own loader code is never loaded.

Every test uses a real model directory on disk and a fake loader that records its calls,
so the guard reads real files and the assertions observe what the loader received.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

import pytest

import mlx_model_doctor.checks.smoke as smoke_module
from mlx_model_doctor.checks.smoke import (
    MlxLmBackend,
    MlxLmSmokeCheck,
    MlxVlmBackend,
    MlxVlmSmokeCheck,
)
from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.errors import TargetError
from mlx_model_doctor.memory import GIB
from mlx_model_doctor.report import CheckResult
from mlx_model_doctor.targets import HfTarget, LocalTarget, ModelTarget
from tests.fakes import FakeHfModelInfo, FakeTarget, check_options, write_hf_cache_snapshot

CLEAN_CONFIG = {"model_type": "llama", "hidden_size": 64}
COMMIT = "0123456789abcdef0123456789abcdef01234567"
REPO_ID = "org/model"
KINDS = ("text", "vlm")


# --- fakes -------------------------------------------------------------------------


class FakeMx:
    def __init__(self) -> None:
        self.peak_was_reset = False

    def device_info(self) -> dict[str, object]:
        return {"max_recommended_working_set_size": 25 * GIB}

    def set_wired_limit(self, value: int) -> None:
        pass

    def set_memory_limit(self, value: int) -> None:
        pass

    def reset_peak_memory(self) -> None:
        self.peak_was_reset = True

    def get_peak_memory(self) -> int:
        return 1


@dataclass
class RecordingMlxLm:
    load_paths: list[str] = field(default_factory=list)
    load_kwargs: list[dict[str, object]] = field(default_factory=list)

    def load(self, path_or_hf_repo: str, **kwargs: object) -> tuple[object, object]:
        self.load_paths.append(path_or_hf_repo)
        self.load_kwargs.append(kwargs)
        return (object(), object())

    def generate(
        self, model: object, tokenizer: object, *, prompt: str, max_tokens: int, verbose: bool
    ) -> str:
        return "generated"


@dataclass
class RecordingVlm:
    load_paths: list[str] = field(default_factory=list)
    load_kwargs: list[dict[str, object]] = field(default_factory=list)

    def load(self, path_or_hf_repo: str, **kwargs: object) -> tuple[object, object]:
        self.load_paths.append(path_or_hf_repo)
        self.load_kwargs.append(kwargs)
        return ("model", "processor")

    def apply_chat_template(
        self, processor: object, config: object, prompt: str, *, num_images: int = 1
    ) -> str:
        return prompt

    def generate(
        self,
        model: object,
        processor: object,
        prompt: str,
        image: object,
        *,
        max_tokens: int = 8,
        verbose: bool = False,
    ) -> str:
        return "a grey square"


@dataclass
class FakeSnapshotHub:
    """A hub whose snapshot download returns a prepared cache snapshot directory."""

    snapshot: Path | None = None
    error: Exception | None = None
    snapshot_calls: list[str] = field(default_factory=list)
    allow_patterns: list[list[str]] = field(default_factory=list)

    def model_info(self, repo_id: str, *, files_metadata: bool) -> FakeHfModelInfo:
        return FakeHfModelInfo(siblings=())

    def snapshot_download(self, repo_id: str, *, allow_patterns: Sequence[str]) -> str:
        self.snapshot_calls.append(repo_id)
        self.allow_patterns.append(list(allow_patterns))
        if self.error is not None:
            raise self.error
        assert self.snapshot is not None
        return str(self.snapshot)


@dataclass
class SmokeRun:
    result: CheckResult
    loader: RecordingMlxLm | RecordingVlm


def smoke_check(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> tuple[MlxLmSmokeCheck | MlxVlmSmokeCheck, RecordingMlxLm | RecordingVlm]:
    """The real smoke check and backend for ``kind``, wired to a recording fake loader."""
    mx = FakeMx()
    if kind == "text":
        mlx_lm = RecordingMlxLm()

        def import_module(name: str) -> object:
            if name == "mlx.core":
                return mx
            if name == "mlx_lm":
                return mlx_lm
            raise ImportError(name)

        monkeypatch.setattr(smoke_module.importlib, "import_module", import_module)
        return MlxLmSmokeCheck(backend=MlxLmBackend()), mlx_lm
    vlm = RecordingVlm()
    return MlxVlmSmokeCheck(backend=MlxVlmBackend(vlm_module=vlm, mx_module=mx)), vlm


def run_smoke(kind: str, target: ModelTarget, monkeypatch: pytest.MonkeyPatch) -> SmokeRun:
    check, loader = smoke_check(kind, monkeypatch)
    result = check.run(CheckContext(target=target, options=check_options()))
    return SmokeRun(result=result, loader=loader)


def write_model(root: Path, files: dict[str, bytes]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (root / name).write_bytes(data)
    return root


def as_json(value: object) -> bytes:
    return json.dumps(value).encode()


EXPECTED_CHECK_ID = {"text": "text/smoke.mlx_lm", "vlm": "vlm/smoke.mlx_vlm"}


# --- refused keys --------------------------------------------------------------------

LOADER = {"text": "mlx-lm", "vlm": "mlx-vlm"}
KEY_REMEDIATION = (
    "Do not load this repository with mlx-lm or mlx-vlm unless you trust its author. "
    "The static checks read it without running its code."
)
AUTO_MAP_CLASS = {"AutoImageProcessor": "image_processing_x.XImageProcessor"}

# Keys both loaders act on: mlx-lm 0.31.x and mlx-vlm 0.6.16+ import a config.json
# model_file with no trust_remote_code gate, so text and VLM smoke both refuse these.
# (case id, files on disk, file named in the result, key named in the result)
REFUSED_ON_BOTH = [
    # Bug: the guard only refuses a model_file that resolves inside the repo (or skips
    # the (config.json, model_file) row), so a plain relative module name is loaded.
    (
        "model_file_relative",
        {"config.json": as_json({**CLEAN_CONFIG, "model_file": "modeling_custom.py"})},
        "config.json",
        "model_file",
    ),
    # Bug: the guard normalizes the value and drops a "../" path instead of refusing it.
    (
        "model_file_parent",
        {"config.json": as_json({**CLEAN_CONFIG, "model_file": "../x.py"})},
        "config.json",
        "model_file",
    ),
    # Bug: the guard checks the value with a relative-path rule that an absolute path
    # passes.
    (
        "model_file_absolute",
        {"config.json": as_json({**CLEAN_CONFIG, "model_file": "/opt/code/x.py"})},
        "config.json",
        "model_file",
    ),
    # Bug: the guard tests the value for truthiness (`if config.get("model_file")`)
    # instead of the key's presence.
    (
        "model_file_empty",
        {"config.json": as_json({**CLEAN_CONFIG, "model_file": ""})},
        "config.json",
        "model_file",
    ),
    # Bug: the guard uses `config.get("model_file") is not None`, so a null slips through.
    (
        "model_file_null",
        {"config.json": as_json({**CLEAN_CONFIG, "model_file": None})},
        "config.json",
        "model_file",
    ),
    # Bug: the guard looks for model_file at the top level only, so one nested in a
    # sub-config (text_config) is never seen.
    (
        "model_file_nested_text_config",
        {"config.json": as_json({**CLEAN_CONFIG, "text_config": {"model_file": "m.py"}})},
        "config.json",
        "model_file",
    ),
]

# Keys only the VLM path refuses. On the mlx-lm paths transformers reads auto_map only
# with trust_remote_code on, and every mlx-lm load pins it off; some mlx-vlm processors
# force it on (mlx-vlm 0.6.12 inkling processing_inkling.py:366-373).
REFUSED_ON_VLM_ONLY = [
    # Bug: the (config.json, auto_map) row is missing from the VLM table.
    (
        "config_auto_map",
        {
            "config.json": as_json(
                {**CLEAN_CONFIG, "auto_map": {"AutoConfig": "configuration_x.XConfig"}}
            )
        },
        "config.json",
        "auto_map",
    ),
    # Bug: the (config.json, configuration_files) row is missing, so a config.json that
    # redirects transformers to config.<version>.json, whose auto_map the guard never
    # opens, passes.
    (
        "config_redirect",
        {"config.json": as_json({**CLEAN_CONFIG, "configuration_files": ["config.4.0.0.json"]})},
        "config.json",
        "configuration_files",
    ),
    # Bug: the guard reads only config.json, so the (tokenizer_config.json, auto_map)
    # row is never consulted.
    (
        "tokenizer_auto_map",
        {
            "config.json": as_json(CLEAN_CONFIG),
            "tokenizer_config.json": as_json(
                {"auto_map": {"AutoTokenizer": ["tokenization_x.XTokenizer", None]}}
            ),
        },
        "tokenizer_config.json",
        "auto_map",
    ),
    # Bug: the (processor_config.json, auto_map) row is missing, so a processor that
    # forces trust_remote_code imports the repository's AutoProcessor class.
    (
        "processor_auto_map",
        {
            "config.json": as_json(CLEAN_CONFIG),
            "processor_config.json": as_json(
                {"auto_map": {"AutoProcessor": "processing_x.XProcessor"}}
            ),
        },
        "processor_config.json",
        "auto_map",
    ),
    # Bug: the guard reads auto_map at the top level only; transformers 5.15 reads it
    # from the image_processor sub-config of processor_config.json.
    (
        "processor_nested_image_processor_auto_map",
        {
            "config.json": as_json(CLEAN_CONFIG),
            "processor_config.json": as_json({"image_processor": {"auto_map": AUTO_MAP_CLASS}}),
        },
        "processor_config.json",
        "auto_map",
    ),
    # Bug: the nested search walks dicts but not lists, so an auto_map inside a list
    # of sub-configs is missed.
    (
        "processor_auto_map_inside_a_list",
        {
            "config.json": as_json(CLEAN_CONFIG),
            "processor_config.json": as_json({"processors": [{"auto_map": AUTO_MAP_CLASS}]}),
        },
        "processor_config.json",
        "auto_map",
    ),
    # Bug: the (preprocessor_config.json, auto_map) row is missing, so the repository's
    # AutoImageProcessor class is imported.
    (
        "preprocessor_auto_map",
        {
            "config.json": as_json(CLEAN_CONFIG),
            "preprocessor_config.json": as_json({"auto_map": AUTO_MAP_CLASS}),
        },
        "preprocessor_config.json",
        "auto_map",
    ),
    # Bug: the (video_preprocessor_config.json, auto_map) row is missing, so the
    # repository's AutoVideoProcessor class is imported.
    (
        "video_preprocessor_auto_map",
        {
            "config.json": as_json(CLEAN_CONFIG),
            "video_preprocessor_config.json": as_json(
                {"auto_map": {"AutoVideoProcessor": "video_processing_x.XVideoProcessor"}}
            ),
        },
        "video_preprocessor_config.json",
        "auto_map",
    ),
]

REFUSED_CASES = [
    pytest.param(kind, files, ef, ek, id=f"{kind}-{case}")
    for case, files, ef, ek in REFUSED_ON_BOTH
    for kind in KINDS
] + [
    pytest.param("vlm", files, ef, ek, id=f"vlm-{case}")
    for case, files, ef, ek in REFUSED_ON_VLM_ONLY
]


@pytest.mark.parametrize(("kind", "files", "expected_file", "expected_key"), REFUSED_CASES)
def test_smoke_refuses_a_repo_that_names_loader_code(
    kind: str,
    files: dict[str, bytes],
    expected_file: str,
    expected_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = write_model(tmp_path / "model", files)

    run = run_smoke(kind, LocalTarget(model), monkeypatch)

    assert run.loader.load_paths == []
    assert run.result.status == "fail"
    assert run.result.severity == "high"
    assert run.result.check_id == EXPECTED_CHECK_ID[kind]
    assert run.result.details["remote_code_key"] == expected_key
    assert run.result.details["remote_code_file"] == expected_file
    assert run.result.message == (
        f"This repository names loader code ({expected_key} in {expected_file}) that "
        f"{LOADER[kind]} could import. The load was refused, so that code was not run. "
        "The static checks above still ran."
    )
    assert run.result.remediation == KEY_REMEDIATION


@pytest.mark.parametrize(
    "files", [pytest.param(files, id=case) for case, files, _, _ in REFUSED_ON_VLM_ONLY]
)
def test_text_smoke_loads_a_repo_whose_keys_only_mlx_vlm_acts_on(
    files: dict[str, bytes], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: the text smoke path uses the VLM key table, so a repository that kept an
    # auto_map from its source is refused although mlx-lm, with trust_remote_code
    # pinned off, never imports what it names.
    model = write_model(tmp_path / "model", files)

    run = run_smoke("text", LocalTarget(model), monkeypatch)

    assert run.result.status == "pass"
    assert run.loader.load_paths == [str(model.resolve())]
    assert run.loader.load_kwargs == [{"tokenizer_config": {"trust_remote_code": False}}]


# (case id, files on disk, the unreadable file named in the result)
UNREADABLE_ON_BOTH = [
    # Bug: a JSONDecodeError is swallowed as "no keys found", so a config the guard
    # cannot read is loaded anyway (fail open).
    ("config_invalid_json", {"config.json": b'{"model_file": "x.py",'}, "config.json"),
    # Bug: the guard calls `"model_file" in data` on a parsed list, which is False, and
    # loads a config that is not a JSON object at all.
    ("config_not_an_object", {"config.json": b'["model_file"]'}, "config.json"),
    # Bug: the guard decodes with errors="ignore" or catches only JSONDecodeError, so a
    # non-UTF-8 config.json is treated as clean or crashes the check.
    ("config_not_utf8", {"config.json": b"\xff\xfe{}"}, "config.json"),
]

UNREADABLE_ON_VLM_ONLY = [
    # Bug: as above, for tokenizer_config.json on the VLM path.
    (
        "tokenizer_config_not_utf8",
        {"config.json": as_json(CLEAN_CONFIG), "tokenizer_config.json": b"\xff\xfe{}"},
        "tokenizer_config.json",
    ),
    # Bug: the guard never opens processor_config.json (its row is missing), so an
    # unreadable one, which might name code, is loaded anyway.
    (
        "processor_config_invalid_json",
        {"config.json": as_json(CLEAN_CONFIG), "processor_config.json": b'{"auto_map":'},
        "processor_config.json",
    ),
    # Bug: as above, for preprocessor_config.json.
    (
        "preprocessor_config_invalid_json",
        {"config.json": as_json(CLEAN_CONFIG), "preprocessor_config.json": b"\xff\xfe{}"},
        "preprocessor_config.json",
    ),
    # Bug: as above, for video_preprocessor_config.json.
    (
        "video_preprocessor_config_not_an_object",
        {"config.json": as_json(CLEAN_CONFIG), "video_preprocessor_config.json": b'"auto_map"'},
        "video_preprocessor_config.json",
    ),
]

UNREADABLE_CASES = [
    pytest.param(kind, files, ef, id=f"{kind}-{case}")
    for case, files, ef in UNREADABLE_ON_BOTH
    for kind in KINDS
] + [pytest.param("vlm", files, ef, id=f"vlm-{case}") for case, files, ef in UNREADABLE_ON_VLM_ONLY]


@pytest.mark.parametrize(("kind", "files", "expected_file"), UNREADABLE_CASES)
def test_smoke_refuses_when_a_guarded_file_cannot_be_read_as_a_json_object(
    kind: str,
    files: dict[str, bytes],
    expected_file: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = write_model(tmp_path / "model", files)

    run = run_smoke(kind, LocalTarget(model), monkeypatch)

    assert run.loader.load_paths == []
    assert run.result.status == "fail"
    assert run.result.severity == "high"
    assert run.result.check_id == EXPECTED_CHECK_ID[kind]
    assert run.result.details["remote_code_file"] == expected_file
    assert run.result.details["remote_code_unreadable"] is True
    assert "remote_code_dangling_link" not in run.result.details
    assert run.result.message == (
        f"{expected_file} could not be read (missing target, unreadable, or not a JSON "
        "object), so it is unknown whether this repository names loader code. The load "
        "was refused, so repository code was not run. The static checks above still ran."
    )
    assert run.result.remediation == f"Re-download or repair {expected_file}, then run again."


@pytest.mark.parametrize(
    "files", [pytest.param(files, id=case) for case, files, _ in UNREADABLE_ON_VLM_ONLY]
)
def test_text_smoke_does_not_read_files_only_the_vlm_table_guards(
    files: dict[str, bytes], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: the text path reads the VLM table's files and refuses a repository whose
    # processor or tokenizer config it has no reason to read.
    model = write_model(tmp_path / "model", files)

    run = run_smoke("text", LocalTarget(model), monkeypatch)

    assert run.result.status == "pass"
    assert run.loader.load_paths == [str(model.resolve())]


@pytest.mark.parametrize("kind", KINDS)
def test_smoke_refuses_a_dangling_guarded_file(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: the guard tests presence with Path.exists(), which follows the link and reports
    # a dangling config.json as absent, so the guard passes a file it never read; or the
    # result drops the dangling-link detail that tells an incomplete download apart.
    model = write_model(tmp_path / "model", {})
    (model / "config.json").symlink_to(tmp_path / "gone.json")

    run = run_smoke(kind, LocalTarget(model), monkeypatch)

    assert run.loader.load_paths == []
    assert run.result.status == "fail"
    assert run.result.details["remote_code_file"] == "config.json"
    assert run.result.details["remote_code_unreadable"] is True
    assert run.result.details["remote_code_dangling_link"] is True
    assert "missing target" in run.result.message
    assert run.result.remediation == "Re-download or repair config.json, then run again."


# --- clean directories load exactly what was checked --------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_clean_repo_is_loaded_once_from_the_resolved_directory_the_guard_read(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: the loader receives ctx.target.name as given (here a link to the model), not
    # the resolved directory the guard read, so the two can name different trees.
    model = write_model(
        tmp_path / "model",
        {"config.json": as_json(CLEAN_CONFIG), "tokenizer_config.json": as_json({})},
    )
    link = tmp_path / "link"
    link.symlink_to(model, target_is_directory=True)

    run = run_smoke(kind, FakeTarget(files={}, name=str(link)), monkeypatch)

    assert run.result.status == "pass"
    assert run.loader.load_paths == [str(model.resolve())]


@pytest.mark.parametrize("kind", KINDS)
def test_the_guard_reads_the_directory_a_linked_target_resolves_to(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: the guard does not follow the link (it lstat()s or lists the link itself, so
    # config.json beneath it reads as absent) and misses the key the loader would see.
    model = write_model(
        tmp_path / "model", {"config.json": as_json({**CLEAN_CONFIG, "model_file": "m.py"})}
    )
    link = tmp_path / "link"
    link.symlink_to(model, target_is_directory=True)

    run = run_smoke(kind, FakeTarget(files={}, name=str(link)), monkeypatch)

    assert run.loader.load_paths == []
    assert run.result.details["remote_code_key"] == "model_file"


def test_mlx_lm_load_pins_tokenizer_trust_remote_code_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: the text smoke load drops tokenizer_config, so mlx-lm 0.31.3 falls back to
    # its own default of {"trust_remote_code": True} for the tokenizer.
    model = write_model(tmp_path / "model", {"config.json": as_json(CLEAN_CONFIG)})

    run = run_smoke("text", LocalTarget(model), monkeypatch)

    assert run.loader.load_kwargs == [{"tokenizer_config": {"trust_remote_code": False}}]


def test_mlx_vlm_load_keeps_trust_remote_code_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: the VLM smoke load drops trust_remote_code=False while switching to the
    # checked local path.
    model = write_model(tmp_path / "model", {"config.json": as_json(CLEAN_CONFIG)})

    run = run_smoke("vlm", LocalTarget(model), monkeypatch)

    assert run.loader.load_kwargs == [{"trust_remote_code": False}]


@pytest.mark.parametrize("kind", KINDS)
def test_a_local_target_that_is_not_a_directory_is_never_handed_to_the_loader(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: a local name that does not exist reaches the loader, which treats "org/model"
    # as a Hub repository id and downloads and runs whatever the Hub serves.
    monkeypatch.chdir(tmp_path)

    run = run_smoke(kind, FakeTarget(files={}, name=REPO_ID), monkeypatch)

    assert run.loader.load_paths == []
    assert run.result.status == "fail"
    assert "not an existing directory" in run.result.message


@pytest.mark.parametrize("kind", KINDS)
def test_a_local_target_that_is_a_file_is_never_handed_to_the_loader(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: the directory check is dropped, so a file path passes the guard (it finds no
    # config.json beneath a file) and reaches the loader unchecked.
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"\x00")

    run = run_smoke(kind, FakeTarget(files={}, name=str(weights)), monkeypatch)

    assert run.loader.load_paths == []
    assert run.result.status == "fail"
    assert "not an existing directory" in run.result.message


# --- Hugging Face targets: materialize, guard, then load the local snapshot ----------


def hf_snapshot(tmp_path: Path, files: dict[str, bytes]) -> Path:
    return write_hf_cache_snapshot(tmp_path / "hub", REPO_ID, files, commit=COMMIT)


@pytest.mark.parametrize("kind", KINDS)
def test_hf_target_loads_the_materialized_snapshot_not_the_repo_id(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: the HF smoke path still hands the repo id to the loader, which then downloads
    # and loads whatever the Hub serves at that moment, unchecked.
    snapshot = hf_snapshot(tmp_path, {"config.json": as_json(CLEAN_CONFIG)})
    hub = FakeSnapshotHub(snapshot=snapshot)

    run = run_smoke(kind, HfTarget(REPO_ID, hub=hub), monkeypatch)

    assert hub.snapshot_calls == [REPO_ID]
    assert run.loader.load_paths == [str(snapshot.resolve())]
    assert run.result.status == "pass"
    assert run.result.details["snapshot_revision"] == COMMIT


@pytest.mark.parametrize("kind", KINDS)
def test_hf_target_guard_reads_the_materialized_snapshot(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: the guard runs before (or without) materializing the snapshot, so it reads
    # nothing for an HF target and the key in the downloaded config.json goes unseen.
    snapshot = hf_snapshot(
        tmp_path, {"config.json": as_json({**CLEAN_CONFIG, "model_file": "modeling_x.py"})}
    )
    hub = FakeSnapshotHub(snapshot=snapshot)

    run = run_smoke(kind, HfTarget(REPO_ID, hub=hub), monkeypatch)

    assert run.loader.load_paths == []
    assert run.result.status == "fail"
    assert run.result.details["remote_code_key"] == "model_file"
    assert run.result.details["remote_code_file"] == "config.json"
    assert run.result.details["snapshot_revision"] == COMMIT


@pytest.mark.parametrize("kind", KINDS)
def test_hf_snapshot_never_fetches_repository_python_files(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: the snapshot download reuses the loaders' own patterns, "*.py" included, and
    # pulls repository code onto disk; or it drops the weights and configs the load needs.
    snapshot = hf_snapshot(tmp_path, {"config.json": as_json(CLEAN_CONFIG)})
    hub = FakeSnapshotHub(snapshot=snapshot)

    run_smoke(kind, HfTarget(REPO_ID, hub=hub), monkeypatch)

    (patterns,) = hub.allow_patterns
    for wanted in ("config.json", "tokenizer_config.json", "model.safetensors"):
        assert any(fnmatch(wanted, p) for p in patterns), wanted
    assert not any(fnmatch("modeling_x.py", p) for p in patterns)


# Each loader's own download patterns minus "*.py", copied from mlx-lm 0.31.3
# utils._download and mlx-vlm 0.6.12 utils.get_model_path.
LOADER_PATTERNS_WITHOUT_PY = {
    "text": [
        "*.json",
        "model*.safetensors",
        "tokenizer.model",
        "*.tiktoken",
        "tiktoken.model",
        "*.txt",
        "*.jsonl",
        "*.jinja",
    ],
    "vlm": ["*.json", "*.safetensors", "*.model", "*.tiktoken", "*.txt", "*.jinja"],
}


@pytest.mark.parametrize("kind", KINDS)
def test_hf_snapshot_fetches_every_file_the_loader_would_fetch_except_code(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: a pattern the loader needs ("tokenizer.model", "*.jinja", "*.txt",
    # "*.tiktoken") is dropped from the snapshot list, so a real `check hf --smoke`
    # fails to load a tokenizer or chat template that the plain loader would find.
    snapshot = hf_snapshot(tmp_path, {"config.json": as_json(CLEAN_CONFIG)})
    hub = FakeSnapshotHub(snapshot=snapshot)

    run_smoke(kind, HfTarget(REPO_ID, hub=hub), monkeypatch)

    assert hub.allow_patterns == [LOADER_PATTERNS_WITHOUT_PY[kind]]


@pytest.mark.parametrize("kind", KINDS)
def test_hf_snapshot_outside_a_cache_snapshots_folder_has_no_revision(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: the revision is always taken from the folder name, so a snapshot that is not
    # under a Hub cache snapshots/ directory reports its folder name as a commit.
    folder = write_model(tmp_path / "plain-folder", {"config.json": as_json(CLEAN_CONFIG)})
    hub = FakeSnapshotHub(snapshot=folder)

    run = run_smoke(kind, HfTarget(REPO_ID, hub=hub), monkeypatch)

    assert run.result.status == "pass"
    assert run.loader.load_paths == [str(folder.resolve())]
    assert "snapshot_revision" not in run.result.details


@pytest.mark.parametrize("kind", KINDS)
def test_hf_snapshot_failure_is_a_tool_error_and_nothing_is_loaded(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: a failed snapshot download is swallowed and the loader gets the repo id, or
    # a network failure is reported as an ordinary smoke failure instead of a tool error.
    hub = FakeSnapshotHub(error=OSError("connection reset"))
    check, loader = smoke_check(kind, monkeypatch)
    ctx = CheckContext(target=HfTarget(REPO_ID, hub=hub), options=check_options())

    with pytest.raises(TargetError, match="connection reset"):
        check.run(ctx)

    assert loader.load_paths == []
    assert hub.snapshot_calls == [REPO_ID]


@pytest.mark.parametrize("kind", KINDS)
def test_hf_source_target_without_a_snapshot_is_refused(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: a target that reports source "hf" but cannot materialize a snapshot falls
    # back to passing its name (a repo id) straight to the loader.
    run = run_smoke(kind, FakeTarget(files={}, name=REPO_ID, _source="hf"), monkeypatch)

    assert run.loader.load_paths == []
    assert run.result.status == "fail"
    assert "local snapshot" in run.result.message


# --- the guard and the hub adapter, directly -----------------------------------------


def test_guard_passes_a_directory_with_neither_guarded_file(tmp_path: Path) -> None:
    # Bug: a missing tokenizer_config.json (or config.json) counts as a refusal, which
    # refuses every repo that ships no tokenizer config.
    from mlx_model_doctor.checks.remote_code import find_remote_code

    assert find_remote_code(write_model(tmp_path / "model", {})) is None


def test_guard_looks_up_keys_not_text(tmp_path: Path) -> None:
    # Bug: the guard searches the raw text, so a value that mentions a key refuses a
    # clean repo.
    from mlx_model_doctor.checks.remote_code import find_remote_code

    model = write_model(
        tmp_path / "model",
        {
            "config.json": as_json({**CLEAN_CONFIG, "note": "no model_file or auto_map"}),
            "tokenizer_config.json": as_json({"chat_template": "auto_map"}),
        },
    )

    assert find_remote_code(model) is None


def test_default_hub_snapshot_download_forwards_repo_and_patterns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Bug: the adapter drops allow_patterns and downloads the whole repository (every
    # weight format and any code file it ships).
    import huggingface_hub

    from mlx_model_doctor.targets import DefaultHfHub

    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return "/cache/models--org--model/snapshots/" + COMMIT

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    path = DefaultHfHub().snapshot_download(REPO_ID, allow_patterns=("*.json",))

    assert path == "/cache/models--org--model/snapshots/" + COMMIT
    assert calls == [{"repo_id": REPO_ID, "allow_patterns": ["*.json"]}]


def test_guard_refuses_a_guarded_file_that_is_a_directory(tmp_path: Path) -> None:
    # Bug: the OSError from reading a directory escapes the guard (the smoke check
    # crashes instead of refusing) or is read as "file absent" (the guard passes).
    from mlx_model_doctor.checks.remote_code import MLX_LM_KEYS, find_remote_code

    model = write_model(tmp_path / "model", {})
    (model / "config.json").mkdir()

    refusal = find_remote_code(model, keys=MLX_LM_KEYS)

    assert refusal is not None
    assert (refusal.file, refusal.key, refusal.reason) == ("config.json", None, "unreadable")


def test_guard_refuses_a_config_nested_too_deep_to_parse(tmp_path: Path) -> None:
    # Bug: the RecursionError json raises on deep nesting escapes the guard, so the
    # check crashes instead of refusing a file it could not read.
    from mlx_model_doctor.checks.remote_code import MLX_LM_KEYS, find_remote_code

    # 10**6 levels: json raises RecursionError (checked on Python 3.14, which parses 10**5).
    deep = b'{"a":' * 1_000_000 + b"1" + b"}" * 1_000_000
    model = write_model(tmp_path / "model", {"config.json": deep})

    refusal = find_remote_code(model, keys=MLX_LM_KEYS)

    assert refusal is not None
    assert (refusal.file, refusal.key, refusal.reason) == ("config.json", None, "unreadable")


@pytest.mark.parametrize("make", ["missing", "file"])
def test_guard_refuses_a_path_that_is_not_an_existing_directory(make: str, tmp_path: Path) -> None:
    # Bug: a missing path (or a file) finds no guarded file beneath it and passes, so a
    # caller hands it to mlx-lm, which treats a missing path as a Hub repo id and
    # downloads its *.py files.
    from mlx_model_doctor.checks.remote_code import MLX_LM_KEYS, find_remote_code

    path = tmp_path / "org" / "model"
    if make == "file":
        path.parent.mkdir()
        path.write_bytes(b"{}")

    refusal = find_remote_code(path, keys=MLX_LM_KEYS)

    assert refusal is not None
    assert (refusal.file, refusal.key, refusal.reason) == (str(path), None, "not_a_directory")


# --- end to end: a refusal through the CLI, memory gate first ------------------------


def test_cli_check_local_smoke_refuses_a_model_file_repo_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Bug: the refusal is lost between the smoke backend and the report (the runner,
    # the memory gate or the CLI drops or rewrites it), so `check local --smoke` loads
    # the repository or exits as if the smoke check had passed.
    from mlx_model_doctor import cli
    from tests.test_parity_api import _F4, _write_model_repo

    model = _write_model_repo(tmp_path / "model", q_bytes=_F4 * 64)
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    (model / "config.json").write_text(
        json.dumps({**config, "model_file": "modeling_x.py"}), encoding="utf-8"
    )
    check, loader = smoke_check("text", monkeypatch)
    assert isinstance(check, MlxLmSmokeCheck)

    code = cli.main(["check", "local", str(model), "--smoke", "--format", "json"])

    results = json.loads(capsys.readouterr().out)["results"]
    failed = [r for r in results if r["status"] == "fail"]
    assert [r["check_id"] for r in failed] == ["text/smoke.mlx_lm"]
    assert failed[0]["details"]["remote_code_key"] == "model_file"
    assert failed[0]["details"]["remote_code_file"] == "config.json"
    assert "mlx-lm could import" in failed[0]["message"]
    assert code == 1
    assert loader.load_paths == []


# --- a text load that fails on an auto_map: our message, not transformers' advice ----


def transformers_remote_code_error(path: str) -> ValueError:
    """The ValueError transformers 5.15.0 raises for a local directory whose auto_map it
    would have to import with trust_remote_code off (dynamic_module_utils.py:742-792)."""
    error_message = (
        f"The repository {path} contains custom code which must be executed to correctly "
        f"load the model. You can inspect the repository content at {path} .\n"
    )
    return ValueError(
        f"{error_message} You can inspect the repository content at https://hf.co/{path}.\n"
        "Please pass the argument `trust_remote_code=True` to allow custom code to be run."
    )


@dataclass
class RaisingMlxLm(RecordingMlxLm):
    """An mlx-lm whose load fails the way transformers does on an auto_map."""

    def load(self, path_or_hf_repo: str, **kwargs: object) -> tuple[object, object]:
        super().load(path_or_hf_repo, **kwargs)
        raise transformers_remote_code_error(path_or_hf_repo)


def run_text_smoke_with(
    mlx_lm: RecordingMlxLm, target: ModelTarget, monkeypatch: pytest.MonkeyPatch
) -> CheckResult:
    mx = FakeMx()

    def import_module(name: str) -> object:
        if name == "mlx.core":
            return mx
        if name == "mlx_lm":
            return mlx_lm
        raise ImportError(name)

    monkeypatch.setattr(smoke_module.importlib, "import_module", import_module)
    check = MlxLmSmokeCheck(backend=MlxLmBackend())
    return check.run(CheckContext(target=target, options=check_options()))


AUTO_MAP = {"AutoTokenizer": ["tokenization_x.XTokenizer", None]}


@pytest.mark.parametrize("auto_map_file", ["tokenizer_config.json", "config.json"])
def test_text_smoke_load_failure_on_an_auto_map_gets_our_message_not_upstream_advice(
    auto_map_file: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: the smoke fail relays transformers' text, which tells the user to pass
    # trust_remote_code=True and prints a hf.co//<absolute path> link.
    files = {"config.json": as_json(CLEAN_CONFIG)}
    files[auto_map_file] = as_json({**CLEAN_CONFIG, "auto_map": AUTO_MAP})
    model = write_model(tmp_path / "model", files)
    mlx_lm = RaisingMlxLm()

    result = run_text_smoke_with(mlx_lm, LocalTarget(model), monkeypatch)

    assert mlx_lm.load_paths == [str(model.resolve())]
    assert result.status == "fail"
    assert result.message == (
        f"This repository names tokenizer or model code (auto_map in {auto_map_file}). "
        "mlx-lm was called with remote code turned off, so that code was not run, and the "
        "load could not finish without it. The static checks above still ran."
    )
    assert "trust_remote_code=True" not in result.message
    assert result.remediation == KEY_REMEDIATION
    assert result.details["upstream_error"] == (
        "The repository model contains custom code which must be executed to correctly "
        "load the model. You can inspect the repository content at model .\n You can "
        "inspect the repository content at https://hf.co/model.\n"
        "Please pass the argument `trust_remote_code=True` to allow custom code to be run."
    )


def test_text_smoke_load_failure_without_an_auto_map_keeps_the_upstream_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: every failed load is reworded as an auto_map refusal, so a repository with no
    # auto_map gets a message that blames code it does not name.
    model = write_model(tmp_path / "model", {"config.json": as_json(CLEAN_CONFIG)})

    result = run_text_smoke_with(RaisingMlxLm(), LocalTarget(model), monkeypatch)

    assert result.status == "fail"
    assert result.message.startswith("Smoke generation failed: The repository ")
    assert "auto_map" not in result.message


def test_upstream_error_paths_are_reduced_to_their_file_names() -> None:
    # Bug: only the model directory is shortened, so another absolute path in the
    # upstream text (a cached module, a parent folder) reaches the report in full.
    from mlx_model_doctor.checks.remote_code import reduce_absolute_paths

    text = (
        "No file at /Users/someone/models/m/tokenization_x.py or '/opt/cache/x.json'; "
        "see https://hf.co//Users/someone/models/m and https://hf.co/docs/hub."
    )

    assert reduce_absolute_paths(text, model_dir=Path("/Users/someone/models/m")) == (
        "No file at tokenization_x.py or 'x.json'; see https://hf.co/m and https://hf.co/docs/hub."
    )
