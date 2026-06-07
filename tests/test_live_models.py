import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from mlx_model_doctor.context import CheckOptions
from mlx_model_doctor.errors import ModelDoctorError
from mlx_model_doctor.report import CheckResult, DoctorReport
from mlx_model_doctor.sampling import (
    DefaultHfModelLister,
    SampleBatchReport,
    SampledModelResult,
    candidate_signal,
    deterministic_sample,
    render_sample_batch_json,
    render_sample_batch_markdown,
    render_sample_batch_text,
    run_hf_sample,
    sample_batch_exit_code,
)


@dataclass(frozen=True, slots=True)
class FakeModel:
    id: str
    tags: tuple[str, ...] = ()
    library_name: str | None = None


class FakeApi:
    def __init__(self, models: tuple[FakeModel, ...]) -> None:
        self.models = models
        self.calls: list[dict[str, object]] = []

    def list_models(
        self,
        *,
        author: str | None,
        pipeline_tag: str | None,
        limit: int,
        expand: list[str],
    ) -> tuple[FakeModel, ...]:
        self.calls.append(
            {
                "author": author,
                "pipeline_tag": pipeline_tag,
                "limit": limit,
                "expand": expand,
            }
        )
        return self.models


class FakeLister:
    def __init__(
        self,
        models: tuple[FakeModel, ...] = (),
        error: Exception | None = None,
    ) -> None:
        self.models = models
        self.error = error
        self.calls: list[dict[str, object]] = []

    def list_models(
        self,
        *,
        author: str,
        pipeline_tag: str | None,
        limit: int,
    ) -> tuple[FakeModel, ...]:
        self.calls.append({"author": author, "pipeline_tag": pipeline_tag, "limit": limit})
        if self.error is not None:
            raise self.error
        return self.models


def test_candidate_signal_uses_priority_order() -> None:
    assert (
        candidate_signal(
            FakeModel(
                id="mlx-community/Model-4bit",
                tags=("mlx",),
                library_name="mlx-lm",
            )
        )
        == "tag:mlx"
    )
    assert candidate_signal(FakeModel(id="org/model", library_name="mlx-lm")) == "library:mlx-lm"
    assert candidate_signal(FakeModel(id="mlx-community/model")) == "author:mlx-community"
    assert candidate_signal(FakeModel(id="org/Foo-8BIT")) == "repo-name"
    assert candidate_signal(FakeModel(id="org/plain")) is None


def test_deterministic_sample_filters_sorts_and_limits_candidates() -> None:
    models = (
        FakeModel(id="z/not-a-candidate"),
        FakeModel(id="c/model", tags=("mlx",)),
        FakeModel(id="a/model", library_name="mlx"),
        FakeModel(id="b/model-4bit"),
    )

    sampled = deterministic_sample(models, limit=2)

    assert [model.id for model in sampled] == ["a/model", "b/model-4bit"]


def test_deterministic_sample_zero_limit_returns_empty_list() -> None:
    assert deterministic_sample((FakeModel(id="a/model", tags=("mlx",)),), limit=0) == []


def test_deterministic_sample_rejects_negative_limit() -> None:
    with pytest.raises(ModelDoctorError, match="limit"):
        deterministic_sample((FakeModel(id="a/model", tags=("mlx",)),), limit=-1)


def test_default_hf_model_lister_maps_task_to_pipeline_tag_and_requests_metadata() -> None:
    api = FakeApi((FakeModel(id="a/model", tags=("mlx",)),))
    lister = DefaultHfModelLister(api_factory=lambda: api)

    models = tuple(
        lister.list_models(author="mlx-community", pipeline_tag="text-generation", limit=5)
    )

    assert models == (FakeModel(id="a/model", tags=("mlx",)),)
    assert api.calls == [
        {
            "author": "mlx-community",
            "pipeline_tag": "text-generation",
            "limit": 5,
            "expand": ["tags", "library_name"],
        }
    ]


def test_run_hf_sample_checks_only_sampled_repos_with_static_options() -> None:
    lister = FakeLister(
        (
            FakeModel(id="z/plain"),
            FakeModel(id="b/model", tags=("mlx",)),
            FakeModel(id="a/model", tags=("mlx",)),
        )
    )
    calls: list[tuple[str, CheckOptions | None, str]] = []

    def fake_check(
        repo_id: str,
        *,
        options: CheckOptions | None = None,
        plugin_name: str = "text",
    ) -> DoctorReport:
        calls.append((repo_id, options, plugin_name))
        return sample_report(repo_id, status="pass")

    batch = run_hf_sample(
        author="mlx-community",
        task="text-generation",
        limit=1,
        plugin_name="text",
        lister=lister,
        check_model=fake_check,
    )

    assert lister.calls[0]["author"] == "mlx-community"
    assert lister.calls[0]["pipeline_tag"] == "text-generation"
    # Over-fetch: the lister is called with a window larger than the user limit.
    assert lister.calls[0]["limit"] > 1
    assert [item.repo_id for item in batch.items] == ["a/model"]
    assert [item.signal for item in batch.items] == ["tag:mlx"]
    assert [item.status for item in batch.items] == ["checked"]
    assert [call[0] for call in calls] == ["a/model"]
    assert isinstance(calls[0][1], CheckOptions)
    assert calls[0][1].smoke is False
    assert calls[0][1].include_weights is False
    assert calls[0][2] == "text"


def test_run_hf_sample_captures_per_model_tool_error_and_continues() -> None:
    lister = FakeLister(
        (
            FakeModel(id="a/good", tags=("mlx",)),
            FakeModel(id="b/bad", tags=("mlx",)),
        )
    )
    calls: list[str] = []

    def fake_check(
        repo_id: str,
        *,
        options: CheckOptions | None = None,
        plugin_name: str = "text",
    ) -> DoctorReport:
        calls.append(repo_id)
        if repo_id == "b/bad":
            raise ModelDoctorError("model metadata is unavailable")
        return sample_report(repo_id, status="pass")

    batch = run_hf_sample(lister=lister, check_model=fake_check)

    assert calls == ["a/good", "b/bad"]
    assert [(item.repo_id, item.status, item.error) for item in batch.items] == [
        ("a/good", "checked", None),
        ("b/bad", "tool-error", "model metadata is unavailable"),
    ]


def test_run_hf_sample_listing_failure_is_tool_error() -> None:
    lister = FakeLister(error=RuntimeError("rate limited"))

    with pytest.raises(ModelDoctorError, match="Could not list Hugging Face models"):
        run_hf_sample(lister=lister, check_model=unused_check)


def test_sample_batch_renderers_include_repo_signals_statuses_reports_and_errors() -> None:
    batch = SampleBatchReport(
        author="mlx-community",
        task="text-generation",
        limit=2,
        plugin="text",
        items=(
            SampledModelResult(
                repo_id="a/good",
                signal="tag:mlx",
                status="checked",
                report=sample_report("a/good", status="pass"),
            ),
            SampledModelResult(
                repo_id="b/bad",
                signal="repo-name",
                status="tool-error",
                error="not found",
            ),
        ),
    )

    data = json.loads(render_sample_batch_json(batch))
    markdown = render_sample_batch_markdown(batch)
    text = render_sample_batch_text(batch)

    assert data["summary"] == {"checked": 1, "tool-error": 1}
    assert data["items"][0]["repo_id"] == "a/good"
    assert data["items"][0]["signal"] == "tag:mlx"
    assert data["items"][0]["report"]["summary"]["pass"] == 1
    assert data["items"][1]["status"] == "tool-error"
    assert data["items"][1]["error"] == "not found"
    assert "a/good" in markdown
    assert "tag:mlx" in markdown
    assert "tool-error" in markdown
    assert "b/bad" in text
    assert "not found" in text
    assert "checked: 1" in text
    assert "tool-error: 1" in text


def test_sample_batch_exit_code_is_two_when_no_models_could_be_checked() -> None:
    # Every attempted model errored, so the tool validated nothing: exit 2
    # (the documented "tool error" code). Catches _cmd_sample_hf hardcoding 0.
    batch = SampleBatchReport(
        author="mlx-community",
        task=None,
        limit=2,
        plugin="text",
        items=(
            SampledModelResult(repo_id="a/x", signal="tag:mlx", status="tool-error", error="boom"),
            SampledModelResult(repo_id="b/y", signal="tag:mlx", status="tool-error", error="boom"),
        ),
    )

    assert sample_batch_exit_code(batch) == 2


def test_sample_batch_exit_code_is_zero_when_any_model_was_checked() -> None:
    # A mixed batch is still a successful survey; per-model errors are data,
    # not a tool failure. Catches an over-strict "any tool-error -> 2" policy.
    batch = SampleBatchReport(
        author="mlx-community",
        task=None,
        limit=2,
        plugin="text",
        items=(
            SampledModelResult(
                repo_id="a/x",
                signal="tag:mlx",
                status="checked",
                report=sample_report("a/x", status="pass"),
            ),
            SampledModelResult(repo_id="b/y", signal="tag:mlx", status="tool-error", error="boom"),
        ),
    )

    assert sample_batch_exit_code(batch) == 0


def test_sample_batch_exit_code_is_zero_for_empty_batch() -> None:
    # No MLX candidates matched the filter is a valid empty survey, not an error.
    batch = SampleBatchReport(author="mlx-community", task=None, limit=2, plugin="text", items=())

    assert sample_batch_exit_code(batch) == 0


def test_run_hf_sample_rejects_negative_limit_before_listing() -> None:
    # The limit guard must fire before any listing call, so a negative limit
    # never reaches the (possibly networked) lister.
    lister = FakeLister((FakeModel(id="a/model", tags=("mlx",)),))

    with pytest.raises(ModelDoctorError, match="non-negative"):
        run_hf_sample(limit=-1, lister=lister, check_model=unused_check)

    assert lister.calls == []


def _live_records() -> tuple[dict[str, str], ...]:
    records: list[dict[str, str]] = []
    for path in sorted((Path(__file__).parent / "live").glob("known-*.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        records.extend(data["models"])
    return tuple(records)


@pytest.mark.network
@pytest.mark.parametrize("record", list(_live_records()))
def test_live_records_exercise_check_hf_tool_behavior(record: dict[str, str], capsys) -> None:
    from mlx_model_doctor import cli

    code = cli.main(["check", "hf", record["repo"], "--format", "json"])
    captured = capsys.readouterr()

    assert "Traceback" not in captured.err
    if code in {0, 1}:
        payload = json.loads(captured.out)
        assert payload["target"] == record["repo"]
        assert "summary" in payload
        return

    assert code == 2
    assert captured.err.startswith("Error: ")


def sample_report(
    target: str,
    *,
    status: Literal["pass", "warn", "fail", "skip"],
) -> DoctorReport:
    severity: Literal["info", "low", "medium", "high"] = (
        "info" if status in {"pass", "skip"} else "high"
    )
    return DoctorReport(
        target=target,
        source="hf",
        plugin="text",
        results=(
            CheckResult(
                check_id="text/files.required",
                title="Required files",
                status=status,
                severity=severity,
                message=f"{target} {status}",
            ),
        ),
    )


def unused_check(
    repo_id: str,
    *,
    options: CheckOptions | None = None,
    plugin_name: str = "text",
) -> DoctorReport:
    raise AssertionError(f"unexpected check call for {repo_id}")


@pytest.mark.network
def test_hf_safetensors_header_is_readable_without_download() -> None:
    from mlx_model_doctor.targets import HfTarget

    header = HfTarget("mlx-community/Qwen2.5-0.5B-Instruct-4bit").safetensors_header()
    assert header is not None
    entry = next(iter(header.files[0].tensors.values()))
    assert entry.dtype
    assert entry.shape
    assert entry.data_offsets[1] >= entry.data_offsets[0]


@pytest.mark.network
def test_hf_known_good_repo_has_no_failures_with_weight_checks() -> None:
    from mlx_model_doctor.api import check_hf_model

    report = check_hf_model("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    assert report.summary["fail"] == 0


def test_run_hf_sample_overfetches_so_limit_counts_mlx_candidates() -> None:
    # Non-MLX repos ("aaa/plain-1", "aab/plain-2") sort before the two MLX ones.
    # With the old code (list limit=2), both listed repos are non-MLX, so 0 MLX
    # candidates are checked. After the fix, run_hf_sample over-fetches (>2), sees
    # all four candidates, filters to the 2 MLX ones, and checks exactly 2.
    candidates = (
        FakeModel(id="aaa/plain-1"),
        FakeModel(id="aab/plain-2"),
        FakeModel(id="mlx-community/m1", tags=("mlx",)),
        FakeModel(id="mlx-community/m2", tags=("mlx",)),
    )
    lister = FakeLister(candidates)

    def ok_check(
        repo_id: str,
        *,
        options: CheckOptions | None = None,
        plugin_name: str = "text",
    ) -> DoctorReport:
        return sample_report(repo_id, status="pass")

    batch = run_hf_sample(limit=2, lister=lister, check_model=ok_check)

    assert batch.summary["checked"] == 2
    assert lister.calls[0]["limit"] > 2  # over-fetched beyond the user limit


def _vlm_result(report):
    return next(r for r in report.results if r.check_id == "text/vlm.image_processor")


@pytest.mark.network
def test_internvl3_is_not_false_failed():
    from mlx_model_doctor.api import check_hf_model

    report = check_hf_model("mlx-community/InternVL3-2B-4bit")
    assert _vlm_result(report).status in {"pass", "skip"}


@pytest.mark.network
def test_qwen2_audio_skips_as_non_vlm():
    from mlx_model_doctor.api import check_hf_model

    report = check_hf_model("mlx-community/Qwen2-Audio-7B-Instruct-4bit")
    assert _vlm_result(report).status == "skip"


@pytest.mark.network
def test_qwen2_5_vl_passes():
    from mlx_model_doctor.api import check_hf_model

    report = check_hf_model("mlx-community/Qwen2.5-VL-3B-Instruct-bf16")
    assert _vlm_result(report).status == "pass"
