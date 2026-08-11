"""Offline unit tests for the Hugging Face sampling helpers.

Live, networked sampling/check behavior is exercised in ``test_live_models.py``
(``@pytest.mark.network``); everything here runs against fakes with no network.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from mlx_model_doctor.cache import ListingCache
from mlx_model_doctor.compat import LISTING_VISIBLE_SIGNALS
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


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (FakeModel("org/m", tags=("mlx",)), "tag:mlx"),
        (FakeModel("org/m", library_name="mlx-lm"), "library:mlx-lm"),
        (FakeModel("org/m", library_name="mlx"), "library:mlx"),
        (FakeModel("mlx-community/Foo"), "author:mlx-community"),
        (FakeModel("someorg/Llama-4bit"), "repo-name"),
        (FakeModel("someorg/plain"), None),
        (FakeModel("mlx-community/Foo-4bit", tags=("mlx",)), "tag:mlx"),
        # org contains an MLX token but the repo name does not, and it isn't
        # mlx-community -> no signal (matching is on the repo-name segment only).
        (FakeModel("mlxorg/clean-model"), None),
    ],
)
def test_candidate_signal_preserves_outputs(model: FakeModel, expected: str | None) -> None:
    assert candidate_signal(model) == expected


def test_candidate_signal_uses_priority_order() -> None:
    assert (
        candidate_signal(
            FakeModel(id="mlx-community/Model-4bit", tags=("mlx",), library_name="mlx-lm")
        )
        == "tag:mlx"
    )
    assert candidate_signal(FakeModel(id="org/model", library_name="mlx-lm")) == "library:mlx-lm"
    assert candidate_signal(FakeModel(id="mlx-community/model")) == "author:mlx-community"
    assert candidate_signal(FakeModel(id="org/Foo-8BIT")) == "repo-name"
    assert candidate_signal(FakeModel(id="org/plain")) is None


def test_deterministic_sample_filters_sorts_and_limits_with_carried_signals() -> None:
    # ids and signals are deliberately in OPPOSITE alphabetical order so this pins
    # that the sort key is the repo id, not the carried signal. By id: a, b, c;
    # by signal: library:mlx (b), repo-name (c), tag:mlx (a) -> a different top-2.
    models = (
        FakeModel(id="z/not-a-candidate"),  # no signal -> filtered out
        FakeModel(id="a/model", tags=("mlx",)),  # signal tag:mlx      (id 1st, signal last)
        FakeModel(id="b/model", library_name="mlx"),  # signal library:mlx (id 2nd, signal 1st)
        FakeModel(id="c/model-4bit"),  # signal repo-name   (id 3rd, signal middle)
    )

    sampled = deterministic_sample(models, limit=2)

    # Sorted by id (a, b); sorting by signal instead would yield (b, c).
    assert [(model.id, signal) for model, signal in sampled] == [
        ("a/model", "tag:mlx"),
        ("b/model", "library:mlx"),
    ]


def test_deterministic_sample_zero_limit_returns_empty_list() -> None:
    assert deterministic_sample((FakeModel(id="a/model", tags=("mlx",)),), limit=0) == []


def test_deterministic_sample_rejects_negative_limit() -> None:
    with pytest.raises(ModelDoctorError, match="limit"):
        deterministic_sample((FakeModel(id="a/model", tags=("mlx",)),), limit=-1)


def test_deterministic_sample_with_signal_filter() -> None:
    models = [
        FakeModel("mlx-community/a", tags=("mlx",), library_name="mlx"),
        FakeModel("mlx-community/b", tags=(), library_name="mlx-lm"),
        FakeModel("mlx-community/c", tags=("mlx",)),
    ]
    result = deterministic_sample(models, limit=10, signal_filter=("library:mlx-lm",))
    assert len(result) == 1
    assert result[0][0].id == "mlx-community/b"
    assert result[0][1] == "library:mlx-lm"


def test_deterministic_sample_no_filter_keeps_all() -> None:
    models = [
        FakeModel("mlx-community/a", tags=("mlx",), library_name="mlx"),
        FakeModel("mlx-community/b", tags=(), library_name="mlx-lm"),
    ]
    result = deterministic_sample(models, limit=10, signal_filter=None)
    assert len(result) == 2


def test_signal_filter_applied_before_limit_slice() -> None:
    models = [
        FakeModel("mlx-community/b", tags=("mlx",)),
        FakeModel("mlx-community/a", tags=("mlx",)),
        FakeModel("mlx-community/c", tags=(), library_name="mlx-lm"),
    ]
    result = deterministic_sample(models, limit=1, signal_filter=("tag:mlx",))
    assert len(result) == 1
    assert result[0][0].id == "mlx-community/a"


def test_listing_visible_signals_is_correct_set() -> None:
    expected = frozenset(
        {
            "tag:mlx",
            "library:mlx",
            "library:mlx-lm",
            "author:mlx-community",
            "repo-name",
        }
    )
    assert expected == LISTING_VISIBLE_SIGNALS


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


def test_default_hf_model_lister_uses_cache(tmp_path: Path) -> None:
    """Cache hit skips the API; cache miss fetches then writes."""
    models = (FakeModel("mlx-community/a", tags=("mlx",)),)
    api = FakeApi(models)
    cache = ListingCache(cache_dir=tmp_path, ttl_seconds=3600)
    lister = DefaultHfModelLister(api_factory=lambda: api, cache=cache)

    # First call: cache miss -> API called, cache written
    result1 = list(lister.list_models(author="mlx-community", pipeline_tag=None, limit=200))
    assert len(api.calls) == 1
    assert len(result1) == 1

    # Second call: cache hit -> API NOT called
    result2 = list(lister.list_models(author="mlx-community", pipeline_tag=None, limit=200))
    assert len(api.calls) == 1  # still 1 -- cache hit
    assert len(result2) == 1
    assert result2[0].id == "mlx-community/a"


def test_default_hf_model_lister_without_cache_always_fetches(tmp_path: Path) -> None:
    """Without a cache, every call hits the API."""
    models = (FakeModel("mlx-community/a", tags=("mlx",)),)
    api = FakeApi(models)
    lister = DefaultHfModelLister(api_factory=lambda: api)

    list(lister.list_models(author="mlx-community", pipeline_tag=None, limit=200))
    list(lister.list_models(author="mlx-community", pipeline_tag=None, limit=200))
    assert len(api.calls) == 2


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


def test_run_hf_sample_rejects_negative_limit_before_listing() -> None:
    # The limit guard must fire before any listing call, so a negative limit
    # never reaches the (possibly networked) lister.
    lister = FakeLister((FakeModel(id="a/model", tags=("mlx",)),))

    with pytest.raises(ModelDoctorError, match="non-negative"):
        run_hf_sample(limit=-1, lister=lister, check_model=unused_check)

    assert lister.calls == []


def test_run_hf_sample_rejects_vlm_plugin_without_vlm_task_before_listing() -> None:
    lister = FakeLister((FakeModel(id="mlx-community/model", tags=("mlx",)),))

    with pytest.raises(ModelDoctorError, match="--plugin vlm requires --task"):
        run_hf_sample(plugin_name="vlm", lister=lister, check_model=unused_check)

    assert lister.calls == []


def test_run_hf_sample_rejects_vlm_plugin_with_non_vlm_task_before_listing() -> None:
    lister = FakeLister((FakeModel(id="mlx-community/model", tags=("mlx",)),))

    with pytest.raises(ModelDoctorError, match="image-text-to-text"):
        run_hf_sample(
            plugin_name="vlm",
            task="text-generation",
            lister=lister,
            check_model=unused_check,
        )

    assert lister.calls == []


def test_run_hf_sample_accepts_vlm_plugin_with_image_text_task() -> None:
    lister = FakeLister((FakeModel(id="mlx-community/model", tags=("mlx",)),))
    calls: list[tuple[str, CheckOptions | None, str]] = []

    def fake_check(
        repo_id: str,
        *,
        options: CheckOptions | None = None,
        plugin_name: str = "text",
    ) -> DoctorReport:
        calls.append((repo_id, options, plugin_name))
        return sample_report(repo_id, status="pass", plugin="vlm")

    batch = run_hf_sample(
        plugin_name="vlm",
        task="image-text-to-text",
        lister=lister,
        check_model=fake_check,
    )

    assert batch.plugin == "vlm"
    assert lister.calls[0]["pipeline_tag"] == "image-text-to-text"
    assert calls[0][2] == "vlm"
    assert calls[0][1] is not None
    assert calls[0][1].include_weights is False
    assert calls[0][1].smoke is False


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


def test_run_hf_sample_preserves_legacy_fetch_depth_when_max_candidates_is_none() -> None:
    """When max_candidates is None (default), the legacy overfetch formula is used."""
    models = tuple(FakeModel(f"mlx-community/m{i}", tags=("mlx",)) for i in range(50))
    lister = FakeLister(models)
    run_hf_sample(author="mlx-community", limit=5, lister=lister)
    # Legacy: min(max(5 * 5, 5), 200) = 25 — NOT 200
    assert lister.calls[0]["limit"] == 25


def test_run_hf_sample_uses_explicit_max_candidates_as_fetch_depth() -> None:
    models = tuple(FakeModel(f"mlx-community/m{i}", tags=("mlx",)) for i in range(3))
    lister = FakeLister(models)
    run_hf_sample(author="mlx-community", limit=5, lister=lister, max_candidates=500)
    assert lister.calls[0]["limit"] == 500


def test_run_hf_sample_passes_signal_filter_through_to_deterministic_sample() -> None:
    lister = FakeLister(
        (
            FakeModel(id="a/lib", library_name="mlx-lm"),
            FakeModel(id="b/tag", tags=("mlx",)),
        )
    )

    def ok_check(
        repo_id: str,
        *,
        options: CheckOptions | None = None,
        plugin_name: str = "text",
    ) -> DoctorReport:
        return sample_report(repo_id, status="pass")

    batch = run_hf_sample(lister=lister, check_model=ok_check, signal_filter=("library:mlx-lm",))

    assert [item.repo_id for item in batch.items] == ["a/lib"]


def test_batch_report_has_optional_fields() -> None:
    batch = run_hf_sample(
        author="mlx-community",
        limit=0,
        lister=FakeLister(()),
        max_candidates=500,
        signal_filter=("tag:mlx",),
    )
    assert batch.max_candidates == 500
    assert batch.signal_filter == ("tag:mlx",)


def test_batch_report_omits_optional_fields_when_none() -> None:
    batch = run_hf_sample(
        author="mlx-community",
        limit=0,
        lister=FakeLister(()),
    )
    assert batch.max_candidates is None
    assert batch.signal_filter is None


def test_sampled_model_result_requires_report_when_checked() -> None:
    with pytest.raises(ValueError, match="checked sample results must include a report"):
        SampledModelResult(repo_id="a/x", signal="tag:mlx", status="checked", report=None)


def test_sampled_model_result_requires_error_when_tool_error() -> None:
    with pytest.raises(ValueError, match="tool-error sample results must include an error"):
        SampledModelResult(repo_id="a/x", signal="tag:mlx", status="tool-error", error=None)


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
    # Per-row/per-line co-location: each item's fields must land together on its
    # own row/section, not merely appear somewhere in the document (which a loose
    # substring check, blind to table layout or swapped rows, would accept).
    md_rows = [line for line in markdown.splitlines() if line.startswith("| `")]
    a_row = next(row for row in md_rows if "a/good" in row)
    b_row = next(row for row in md_rows if "b/bad" in row)
    assert "tag:mlx" in a_row
    assert "checked" in a_row
    assert "tool-error" in b_row
    assert "not found" in b_row

    text_lines = text.splitlines()
    assert "CHECKED a/good" in text_lines
    assert "  Signal: tag:mlx" in text_lines
    assert "TOOL-ERROR b/bad" in text_lines
    assert "  Error: not found" in text_lines
    assert "  checked: 1" in text_lines
    assert "  tool-error: 1" in text_lines


def test_render_sample_batch_shows_max_candidates_and_signal_filter_when_present() -> None:
    batch = SampleBatchReport(
        author="mlx-community",
        task=None,
        limit=5,
        plugin="text",
        items=(),
        max_candidates=500,
        signal_filter=("tag:mlx", "library:mlx-lm"),
    )

    text = render_sample_batch_text(batch)
    markdown = render_sample_batch_markdown(batch)

    assert "Max candidates: 500" in text.splitlines()
    assert "Signal filter: tag:mlx, library:mlx-lm" in text.splitlines()
    assert "- Max candidates: `500`" in markdown.splitlines()
    assert "- Signal filter: `tag:mlx, library:mlx-lm`" in markdown.splitlines()


def test_render_sample_batch_omits_max_candidates_and_signal_filter_when_none() -> None:
    batch = SampleBatchReport(author="mlx-community", task=None, limit=5, plugin="text", items=())

    text = render_sample_batch_text(batch)
    markdown = render_sample_batch_markdown(batch)

    assert "Max candidates" not in text
    assert "Signal filter" not in text
    assert "Max candidates" not in markdown
    assert "Signal filter" not in markdown


def test_render_sample_batch_json_includes_new_fields_when_present() -> None:
    batch = SampleBatchReport(
        author="mlx-community",
        task=None,
        limit=5,
        plugin="text",
        items=(),
        max_candidates=500,
        signal_filter=("tag:mlx",),
    )
    data = json.loads(render_sample_batch_json(batch))
    assert data["max_candidates"] == 500
    assert data["signal_filter"] == ["tag:mlx"]


def test_render_sample_batch_json_omits_new_fields_when_none() -> None:
    batch = SampleBatchReport(author="mlx-community", task=None, limit=5, plugin="text", items=())
    data = json.loads(render_sample_batch_json(batch))
    assert "max_candidates" not in data
    assert "signal_filter" not in data


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
    # No MLX candidates matched the filter is a valid empty survey, not an error
    # (the documented empty-batch policy: a survey finding nothing is informational).
    batch = SampleBatchReport(author="mlx-community", task=None, limit=2, plugin="text", items=())

    assert sample_batch_exit_code(batch) == 0


def sample_report(
    target: str,
    *,
    status: Literal["pass", "warn", "fail", "skip"],
    plugin: str = "text",
) -> DoctorReport:
    severity: Literal["info", "low", "medium", "high"] = (
        "info" if status in {"pass", "skip"} else "high"
    )
    return DoctorReport(
        target=target,
        source="hf",
        plugin=plugin,
        results=(
            CheckResult(
                check_id=f"{plugin}/files.required",
                title="Required config file",
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
