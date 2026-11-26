"""Failure text in every report format names files and folders, never the user's home.

Each test puts the model folders in a spaced ``my models`` folder under a fake
home whose name has spaces (``.../Users/Jane Q Doe/my models/<name>``), drives a
smoke or parity failure whose upstream text names absolute paths, and reads the
failure text back out of each renderer. The home must be gone from that text, and
the folder or file that failed must still be named. The ``my models`` folder is
what makes the folder name depend on the caller passing its model folders: without
them the home cut leaves ``~/my models/<name>``, which the exact-name asserts reject.

Identity fields are pinned here, so a change to them is a visible decision: the
``check`` text, Markdown and GitHub headers show the model's short label with its
kind (``tiny-model (local folder)``), JSON ``target`` keeps the resolved model
path, and parity ``base``/``adapter``/``fused`` ``path`` hold the
pinned path every worker loads. That is why the home check never runs over a
whole report, only over the failure text each test extracts.
"""

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import pytest
from jsonschema import Draft202012Validator

import mlx_model_doctor.api as api_module
import mlx_model_doctor.checks.smoke as smoke_module
from mlx_model_doctor import cli
from mlx_model_doctor.api import ParityOptions, check_adapter_parity, check_hf_model
from mlx_model_doctor.context import CheckOptions
from mlx_model_doctor.errors import TargetError
from mlx_model_doctor.memory import GIB
from mlx_model_doctor.parity.exit_codes import parity_exit_code
from mlx_model_doctor.parity.oracle import ROLE_BASE, ROLE_FUSED, ROLE_NOISE, ROLE_REFERENCE
from mlx_model_doctor.parity.orchestrator import WorkerOutcome
from mlx_model_doctor.parity.report import (
    ParityReport,
    render_parity_json,
    render_parity_markdown,
    render_parity_text,
)
from mlx_model_doctor.parity.worker import WorkerSpec
from mlx_model_doctor.report import DoctorReport, render_json, render_markdown, render_text
from tests.fakes import FakeShardHub, shard_repo, use_fake_home, write_hf_cache_snapshot
from tests.parity_fakes import write_parity_adapter, write_parity_model_repo
from tests.test_parity_schema_contract import PARITY_SCHEMA, REGISTRY
from tests.test_schema_contract import SCHEMA as REPORT_SCHEMA

SMOKE_ID = "text/smoke.mlx_lm"
REMOTE_CODE_ID = "parity/load.remote_code"
REPO_ID = "o/n"


# --- the fake home and the check on failure text ----------------------------------------


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A real home folder with a spaced, two-word user name, and the only home known."""
    folder = tmp_path / "Users" / "Jane Q Doe"
    folder.mkdir(parents=True)
    use_fake_home(monkeypatch, home=str(folder), pwd_home=None)
    # `--format github` appends to these files when they are set (as on a CI runner).
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    return folder


def assert_no_home(text: str, home: Path) -> None:
    """No spelling of the home, and no part of the user's name, is in ``text``."""
    for spelling in {str(home), os.path.realpath(home)}:
        assert spelling not in text, f"home {spelling!r} in: {text!r}"
        assert quote(spelling) not in text, f"quoted home {quote(spelling)!r} in: {text!r}"
    for part in ("Jane", "Doe"):
        assert re.search(rf"\b{part}\b", text) is None, f"{part!r} in: {text!r}"


# --- a fake mlx-lm whose load fails on the folder it was given --------------------------


class FakeMx:
    def device_info(self) -> dict[str, object]:
        return {"max_recommended_working_set_size": 25 * GIB}

    def set_wired_limit(self, value: int) -> None:
        pass

    def set_memory_limit(self, value: int) -> None:
        pass

    def reset_peak_memory(self) -> None:
        pass

    def get_peak_memory(self) -> int:
        return 1


class NoWeightsMlxLm:
    """An mlx-lm whose load fails the way it does on a folder with no weights."""

    def load(self, path_or_hf_repo: str, **kwargs: object) -> tuple[object, object]:
        raise OSError(f"No safetensors found in {path_or_hf_repo}")

    def generate(self, *args: object, **kwargs: object) -> str:
        raise AssertionError("generate must not run after a failed load")


def install_failing_mlx_lm(monkeypatch: pytest.MonkeyPatch) -> None:
    mx, mlx_lm = FakeMx(), NoWeightsMlxLm()

    def import_module(name: str) -> object:
        if name == "mlx.core":
            return mx
        if name == "mlx_lm":
            return mlx_lm
        raise ImportError(name)

    monkeypatch.setattr(smoke_module.importlib, "import_module", import_module)


def smoke_options() -> CheckOptions:
    return CheckOptions(
        max_memory_bytes=8 * GIB,
        context_length=4096,
        include_weights=False,
        smoke=True,
        verbosity="normal",
    )


def smoke_failure_text(output: str, fmt: str) -> str:
    """The smoke result's own text in one rendered ``check`` report."""
    if fmt == "text":
        lines = output.splitlines()
        start = lines.index(f"FAIL high {SMOKE_ID}")
        return "\n".join(lines[start : start + 2])
    if fmt == "markdown":
        start = output.index(f"### FAIL {SMOKE_ID}")
        end = output.find("\n### ", start + 1)
        return output[start:] if end == -1 else output[start:end]
    if fmt == "json":
        [result] = [r for r in json.loads(output)["results"] if r["check_id"] == SMOKE_ID]
        return json.dumps(
            {"message": result["message"], "details": result["details"]}, ensure_ascii=False
        )
    [line] = [x for x in output.splitlines() if x.startswith(f"::error title={SMOKE_ID}::")]
    return line


# --- check local --smoke through the CLI -----------------------------------------------


def run_check_local_smoke(
    home: Path,
    fmt: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> tuple[int, str, Path]:
    model = write_parity_model_repo(home / "my models" / "tiny-model")
    install_failing_mlx_lm(monkeypatch)
    code = cli.main(
        ["check", "local", str(model), "--smoke", "--max-memory", "8gb", "--format", fmt]
    )
    return code, capsys.readouterr().out, model


@pytest.mark.parametrize("fmt", ["text", "json", "markdown", "github"])
def test_check_smoke_failure_renders_short_paths_in_every_format(
    fmt: str,
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Bug: the smoke message copies the loader text raw, so this renderer prints the
    # user's home; or the smoke check stops passing the local model folder
    # (target_model_dirs), so the text says "~/my models/tiny-model"; or the CLI's
    # exit code for a failed smoke check moved off 1.
    code, output, model = run_check_local_smoke(home, fmt, monkeypatch, capsys)
    resolved = str(model.resolve())

    assert code == 1
    # S13 pin: the human headers show the label and its kind; JSON target keeps the
    # resolved path on purpose (it is not a leak).
    if fmt == "text":
        assert output.splitlines()[0] == "MLX Model Doctor: tiny-model (local folder)"
    elif fmt == "markdown":
        assert output.splitlines()[0] == "# MLX Model Doctor: tiny-model (local folder)"
    elif fmt == "json":
        assert json.loads(output)["target"] == resolved
    else:
        assert output.splitlines()[-1].startswith(
            "::notice title=mlx-model-doctor::tiny-model (local folder) "
        )
    failure = smoke_failure_text(output, fmt)
    assert "Smoke generation failed: No safetensors found in " in failure
    assert_no_home(failure, home)
    assert "Smoke generation failed: No safetensors found in tiny-model" in failure
    if fmt == "json":
        [result] = [r for r in json.loads(output)["results"] if r["check_id"] == SMOKE_ID]
        assert result["message"] == "Smoke generation failed: No safetensors found in tiny-model"


def test_check_smoke_json_still_validates_report_v1(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Bug: shortening the failure text changed a field's type or added a field, so the
    # JSON a failed smoke run prints no longer matches report.v1.
    _code, output, _model = run_check_local_smoke(home, "json", monkeypatch, capsys)
    payload = json.loads(output)

    assert any(r["check_id"] == SMOKE_ID for r in payload["results"])
    errors = sorted(Draft202012Validator(REPORT_SCHEMA).iter_errors(payload), key=str)
    assert not errors, [error.message for error in errors]


# --- check hf --smoke through the API (the CLI cannot take a fake hub) -------------------


@dataclass(slots=True)
class SnapshotShardHub(FakeShardHub):
    """A FakeShardHub that also hands out a cache snapshot, or fails to download one."""

    snapshot: Path | None = None
    snapshot_error: Exception | None = None

    def snapshot_download(self, repo_id: str, *, allow_patterns: object) -> str:
        if self.snapshot_error is not None:
            raise self.snapshot_error
        assert self.snapshot is not None
        return str(self.snapshot)


RENDERERS: dict[str, Callable[[DoctorReport], str]] = {
    "text": render_text,
    "markdown": render_markdown,
    "json": render_json,
}


@pytest.mark.parametrize("fmt", ["text", "markdown", "json"])
def test_check_hf_smoke_failure_shows_the_repo_id(
    fmt: str, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: the Hugging Face cache snapshot path the user never typed (under the home)
    # reaches the smoke failure text, instead of the repo id.
    files, shards = shard_repo("consistent")
    snapshot = write_hf_cache_snapshot(
        home / ".cache" / "huggingface" / "hub", REPO_ID, files, shards
    )
    hub = SnapshotShardHub(files=files, shards=shards, snapshot=snapshot)
    install_failing_mlx_lm(monkeypatch)

    report = check_hf_model(REPO_ID, options=smoke_options(), hub=hub)
    failure = smoke_failure_text(RENDERERS[fmt](report), fmt)

    assert report.target == REPO_ID
    assert "Smoke generation failed: No safetensors found in " in failure
    assert_no_home(failure, home)
    assert "Smoke generation failed: No safetensors found in o/n" in failure


def test_snapshot_download_tool_error_is_short(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Bug: the tool error for a failed snapshot download prints the cache lock path
    # under the home; or the download failure escapes unwrapped instead of as the
    # TargetError the CLI turns into exit code 2 (pytest.raises carries that pin).
    files, shards = shard_repo("consistent")
    lock = home / ".cache" / "huggingface" / "hub" / ".locks" / "models--o--n" / "abc.lock"
    hub = SnapshotShardHub(
        files=files,
        shards=shards,
        snapshot_error=OSError(f"No space left on device: '{lock}'"),
    )
    install_failing_mlx_lm(monkeypatch)

    with pytest.raises(TargetError) as exc_info:
        check_hf_model(REPO_ID, options=smoke_options(), hub=hub)
    text = str(exc_info.value)

    assert text.startswith("Could not download a snapshot of Hugging Face model o/n: ")
    assert_no_home(text, home)
    assert text == (
        "Could not download a snapshot of Hugging Face model o/n: "
        "No space left on device: 'abc.lock'"
    )


# --- parity mlx ----------------------------------------------------------------------------

_BASE_ARGMAX = [1, 1, 1, 1, 1, 1, 1]
_ADAPTER_ARGMAX = [2, 2, 2, 1, 1, 1, 1]


@dataclass(frozen=True)
class ParityRepos:
    base: Path
    adapter: Path
    fused: Path


@pytest.fixture
def repos(home: Path) -> ParityRepos:
    # The models sit in a spaced folder under the home, so a call site that drops its
    # model folders leaves "~/my models/<name>" in the failure text. The fused folder's
    # name starts with the base folder's name, as real layouts do; the cut order and
    # the name-end rule that keep it whole are pinned in test_redaction.py, not here.
    models = home / "my models"
    return ParityRepos(
        base=write_parity_model_repo(models / "qwen-base"),
        adapter=write_parity_adapter(models / "qwen-adapter"),
        fused=write_parity_model_repo(models / "qwen-base-fused", fused=True),
    )


def _ok(role: str, argmax: list[int], *, adapter_applied: bool | None = None) -> WorkerOutcome:
    return WorkerOutcome(
        role=role,
        argmax=argmax,
        adapter_applied=adapter_applied,
        peak_bytes=100,
        status="ok",
        error=None,
    )


def fused_worker_traceback(home: Path, model_path: str) -> str:
    """The stderr a fused worker leaves when mlx-lm cannot load the folder it was given."""
    return (
        "Traceback (most recent call last):\n"
        f'  File "{home}/.venv/lib/python3.13/site-packages/mlx_lm/utils.py", line 5, in load\n'
        "    raise FileNotFoundError(message)\n"
        f"FileNotFoundError: No safetensors found in {model_path}"
    )


class FusedLoadFailsLauncher:
    """Every worker succeeds except fused, whose error is its stderr traceback."""

    def __init__(self, home: Path) -> None:
        self._home = home

    def run_one(self, spec: WorkerSpec) -> WorkerOutcome:
        if spec.role == ROLE_FUSED:
            return WorkerOutcome(
                role=ROLE_FUSED,
                argmax=None,
                adapter_applied=None,
                peak_bytes=None,
                status="error",
                error=fused_worker_traceback(self._home, spec.model_path),
            )
        if spec.role == ROLE_REFERENCE:
            return _ok(ROLE_REFERENCE, _ADAPTER_ARGMAX, adapter_applied=True)
        assert spec.role in (ROLE_BASE, ROLE_NOISE)
        return _ok(spec.role, _BASE_ARGMAX)


class NoWorkerLauncher:
    def run_one(self, spec: WorkerSpec) -> WorkerOutcome:
        raise AssertionError(f"no worker may run after a crashed check: {spec.role}")


def run_parity_with_fused_failure(home: Path, repos: ParityRepos) -> ParityReport:
    return check_adapter_parity(
        base=str(repos.base),
        adapter=str(repos.adapter),
        fused=str(repos.fused),
        options=ParityOptions(launcher=FusedLoadFailsLauncher(home)),
    )


def _between(text: str, start: str, end: str | None) -> str:
    begin = text.index(start)
    if end is None:
        return text[begin:]
    return text[begin : text.index(end, begin)]


def parity_failure_text(report: ParityReport, fmt: str) -> str:
    """The reasons, the worker errors and the check messages of one rendered parity report.

    A worker error is a multi-line traceback, so whole sections are cut out, not lines
    picked by prefix: the path sits on the traceback's continuation lines.
    """
    if fmt == "text":
        output = render_parity_text(report)
        return _between(output, "  Reason:", "\nAgreement:") + _between(
            output, "Worker errors:", None
        )
    if fmt == "markdown":
        output = render_parity_markdown(report)
        return _between(output, "\n> ", "\n| Metric |") + _between(output, "## Worker errors", None)
    payload = json.loads(render_parity_json(report))
    return json.dumps(
        {
            "reasons": payload["reasons"],
            "worker_errors": payload["worker_errors"],
            "messages": [result["message"] for result in payload["results"]],
        },
        ensure_ascii=False,
    )


@pytest.mark.parametrize("fmt", ["text", "markdown", "json"])
def test_parity_worker_error_renders_short_paths_in_every_format(
    fmt: str, home: Path, repos: ParityRepos
) -> None:
    # Bug: the fused worker's stderr reaches worker_errors or reasons raw, so this
    # renderer prints the home; or the worker-error reduction stops passing the model
    # folders, so the text says "~/my models/qwen-base-fused".
    # The exit-2 assert is a pin only: with the fused worker failed the verdict is
    # null, which exits 2 whatever the worker-status rule says.
    report = run_parity_with_fused_failure(home, repos)
    failure = parity_failure_text(report, fmt)

    assert parity_exit_code(report) == 2
    # The human header shows the fused folder's name; JSON keeps the typed path.
    if fmt == "text":
        assert render_parity_text(report).splitlines()[0].endswith("-> qwen-base-fused")
    elif fmt == "markdown":
        assert render_parity_markdown(report).splitlines()[0].endswith("vs qwen-base-fused")
    else:
        assert json.loads(render_parity_json(report))["fused"]["original_ref"] == str(repos.fused)
    assert "utils.py" in failure
    assert "FileNotFoundError: No safetensors found in " in failure
    assert_no_home(failure, home)
    assert "No safetensors found in qwen-base-fused" in failure
    assert report.worker_errors[ROLE_FUSED] == (
        "Traceback (most recent call last):\n"
        '  File "utils.py", line 5, in load\n'
        "    raise FileNotFoundError(message)\n"
        "FileNotFoundError: No safetensors found in qwen-base-fused"
    )


def test_parity_json_still_validates_parity_v1(home: Path, repos: ParityRepos) -> None:
    # Bug: the reduction changed a parity.v1 type or field; or someone reduced the
    # pinned identity paths (D2) without that decision.
    report = run_parity_with_fused_failure(home, repos)
    payload = json.loads(render_parity_json(report))

    errors = sorted(
        Draft202012Validator(PARITY_SCHEMA, registry=REGISTRY).iter_errors(payload), key=str
    )
    assert not errors, [error.message for error in errors]
    # D2 pin: the identity paths keep the full resolved path every worker loads.
    assert report.base.path == str(repos.base.resolve())
    assert payload["base"]["path"] == str(repos.base.resolve())
    assert payload["adapter"]["path"] == str(repos.adapter.resolve())
    assert payload["fused"]["path"] == str(repos.fused.resolve())


@dataclass(frozen=True)
class CrashingRemoteCodeCheck:
    """A remote-code check that crashes with a message naming the base folder."""

    base: Path
    check_id: str = REMOTE_CODE_ID
    title: str = "Repository loader code"

    def run(self, pctx: object) -> object:
        raise RuntimeError(f"cannot parse {self.base}/config.json in {self.base}")


def test_parity_crashed_check_keeps_exit_code_2(
    home: Path, repos: ParityRepos, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: the crash text keeps the home; or the crash handler stops passing the model
    # folders, so the text says "~/my models/qwen-base"; or the reduction touched the
    # "check crashed:" prefix the parity exit code reads, so exit 2 became 1.
    # The adapter also names a LoRA target no base tensor has (a determined defect,
    # exit 1 on its own), so only a recognized crash keeps the run at exit 2.
    adapter = write_parity_adapter(
        home / "my models" / "qwen-adapter-bad-key", keys=("self_attn.nonexistent",)
    )
    base = repos.base.resolve()
    checks = tuple(
        CrashingRemoteCodeCheck(base) if check.check_id == REMOTE_CODE_ID else check
        for check in api_module._PARITY_CHECKS
    )
    monkeypatch.setattr(api_module, "_PARITY_CHECKS", checks)

    report = check_adapter_parity(
        base=str(repos.base),
        adapter=str(adapter),
        fused=str(repos.fused),
        options=ParityOptions(launcher=NoWorkerLauncher()),
    )
    [result] = [r for r in report.results if r.check_id == REMOTE_CODE_ID]

    assert any(
        r.check_id == "parity/target.coverage" and r.status == "fail" for r in report.results
    )
    assert result.message.startswith("check crashed:")
    assert parity_exit_code(report) == 2
    assert_no_home(result.message, home)
    assert result.message == "check crashed: cannot parse config.json in qwen-base"
