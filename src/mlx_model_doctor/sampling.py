"""Hugging Face live sampling helpers."""

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from mlx_model_doctor.api import check_hf_model
from mlx_model_doctor.compat import mlx_signals
from mlx_model_doctor.context import CheckOptions
from mlx_model_doctor.errors import ModelDoctorError
from mlx_model_doctor.report import DoctorReport, render_json

ItemStatus = Literal["checked", "tool-error"]
_MODEL_METADATA_EXPAND: list[str] = ["tags", "library_name"]
_ITEM_STATUSES: tuple[ItemStatus, ...] = ("checked", "tool-error")
_OVERFETCH_FACTOR = 5
_OVERFETCH_CEILING = 200


class ModelCandidate(Protocol):
    """Hugging Face model metadata needed for sampling."""

    id: str
    tags: Sequence[str]
    library_name: str | None


class HfModelLister(Protocol):
    """Small Hugging Face model-listing boundary."""

    def list_models(
        self,
        *,
        author: str,
        pipeline_tag: str | None,
        limit: int,
    ) -> Iterable[ModelCandidate]:
        """List Hugging Face model candidates."""


class _HfApiProtocol(Protocol):
    def list_models(
        self,
        *,
        author: str | None,
        pipeline_tag: str | None,
        limit: int | None,
        expand: list[str],
    ) -> Iterable[ModelCandidate]:
        """List models through huggingface_hub.HfApi."""


class HfCheckFunction(Protocol):
    """Callable boundary for checking one Hugging Face model."""

    def __call__(
        self,
        repo_id: str,
        *,
        options: CheckOptions | None = None,
        plugin_name: str = "text",
    ) -> DoctorReport:
        """Run checks for one Hugging Face model."""


class DefaultHfModelLister:
    """Hugging Face model lister backed by huggingface_hub.HfApi."""

    def __init__(self, api_factory: Callable[[], _HfApiProtocol] | None = None) -> None:
        """Initialize the lister with an optional fakeable API factory."""
        self._api_factory = api_factory

    def list_models(
        self,
        *,
        author: str,
        pipeline_tag: str | None,
        limit: int,
    ) -> Iterable[ModelCandidate]:
        """List models and request the metadata needed for MLX candidate signals."""
        return self._api().list_models(
            author=author,
            pipeline_tag=pipeline_tag,
            limit=limit,
            expand=list(_MODEL_METADATA_EXPAND),
        )

    def _api(self) -> _HfApiProtocol:
        if self._api_factory is not None:
            return self._api_factory()
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ModelDoctorError("huggingface-hub is required for `sample hf`") from exc
        return cast("_HfApiProtocol", HfApi())


@dataclass(frozen=True, slots=True, kw_only=True)
class SampledModelResult:
    """Result for one sampled model in a batch report."""

    repo_id: str
    signal: str
    status: ItemStatus
    report: DoctorReport | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        """Validate item/report consistency."""
        if self.status == "checked" and self.report is None:
            raise ValueError("checked sample results must include a report")
        if self.status == "tool-error" and not self.error:
            raise ValueError("tool-error sample results must include an error")


@dataclass(frozen=True, slots=True, kw_only=True)
class SampleBatchReport:
    """Batch report for a deterministic Hugging Face sample."""

    author: str
    task: str | None
    limit: int
    plugin: str
    items: Sequence[SampledModelResult]
    schema_version: str = "sample-batch/1.0"

    @property
    def summary(self) -> dict[ItemStatus, int]:
        """Return item counts grouped by batch item status."""
        return {
            status: sum(1 for item in self.items if item.status == status)
            for status in _ITEM_STATUSES
        }

    def __post_init__(self) -> None:
        """Copy caller-owned collections so the batch report is stable."""
        object.__setattr__(self, "items", tuple(self.items))


def candidate_signal(model: ModelCandidate) -> str | None:
    """Return the highest-priority MLX compatibility signal for a model."""
    signals = mlx_signals(
        name=model.id,
        source="hf",
        tags=frozenset(model.tags),
        library_name=model.library_name,
        config=None,
        has_quantized_tensors=False,
    )
    return signals[0] if signals else None


def deterministic_sample(
    models: Iterable[ModelCandidate],
    *,
    limit: int,
) -> list[ModelCandidate]:
    """Filter likely MLX candidates, sort by repo ID, and take a deterministic sample."""
    if limit < 0:
        raise ModelDoctorError("sample limit must be non-negative")
    if limit == 0:
        return []
    eligible = [model for model in models if candidate_signal(model) is not None]
    return sorted(eligible, key=lambda model: model.id)[:limit]


def run_hf_sample(
    *,
    author: str = "mlx-community",
    task: str | None = None,
    limit: int = 10,
    plugin_name: str = "text",
    lister: HfModelLister | None = None,
    check_model: HfCheckFunction | None = None,
) -> SampleBatchReport:
    """List, sample, and statically check likely MLX Hugging Face models."""
    if limit < 0:
        raise ModelDoctorError("sample limit must be non-negative")

    model_lister = lister if lister is not None else DefaultHfModelLister()
    fetch_limit = min(max(limit * _OVERFETCH_FACTOR, limit), _OVERFETCH_CEILING)
    try:
        listed_models = tuple(
            model_lister.list_models(author=author, pipeline_tag=task, limit=fetch_limit)
        )
    except ModelDoctorError:
        raise
    except Exception as exc:
        raise ModelDoctorError(f"Could not list Hugging Face models: {exc}") from exc

    sampled_models = deterministic_sample(listed_models, limit=limit)
    checker = check_model if check_model is not None else check_hf_model
    options = CheckOptions(
        max_memory_bytes=None,
        context_length=4096,
        include_weights=False,
        smoke=False,
        verbosity="normal",
    )
    items: list[SampledModelResult] = []
    for model in sampled_models:
        signal = candidate_signal(model)
        # deterministic_sample only yields models with a signal; this guard is an
        # unreachable type-narrowing safeguard (candidate_signal returns str | None).
        if signal is None:  # pragma: no cover
            continue
        try:
            report = checker(model.id, options=options, plugin_name=plugin_name)
        except ModelDoctorError as exc:
            items.append(
                SampledModelResult(
                    repo_id=model.id,
                    signal=signal,
                    status="tool-error",
                    error=str(exc),
                )
            )
            continue
        items.append(
            SampledModelResult(
                repo_id=model.id,
                signal=signal,
                status="checked",
                report=report,
            )
        )

    return SampleBatchReport(
        author=author,
        task=task,
        limit=limit,
        plugin=plugin_name,
        items=tuple(items),
    )


def sample_batch_exit_code(batch: SampleBatchReport) -> int:
    """Return the process exit code for a completed Hugging Face sample batch.

    A sample is a survey: a per-model tool error is recorded as a batch item and
    does not, on its own, fail the run. The tool failed only if it attempted
    models but could check none of them — every item is a ``tool-error`` — which
    maps to exit code 2 (the documented "tool error" code). An empty batch (no
    MLX candidates matched the filter) is a valid informational result: exit 0.
    """
    if batch.items and batch.summary["checked"] == 0:
        return 2
    return 0


def render_sample_batch_json(batch: SampleBatchReport) -> str:
    """Render a sample batch report as stable JSON."""
    return json.dumps(_batch_to_dict(batch), indent=2, sort_keys=True)


def render_sample_batch_markdown(batch: SampleBatchReport) -> str:
    """Render a sample batch report as Markdown."""
    lines = [
        "# MLX Model Doctor HF Sample",
        "",
        f"- Author: `{batch.author}`",
        f"- Task: `{batch.task or 'any'}`",
        f"- Limit: `{batch.limit}`",
        f"- Plugin: `{batch.plugin}`",
        "",
        "| Repo | Signal | Status | Pass | Warn | Fail | Skip | Error |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for item in batch.items:
        summary = _item_summary(item)
        lines.append(
            "| "
            f"`{item.repo_id}` | `{item.signal}` | `{item.status}` | "
            f"{summary['pass']} | {summary['warn']} | {summary['fail']} | "
            f"{summary['skip']} | {item.error or ''} |"
        )
    return "\n".join(lines)


def render_sample_batch_text(batch: SampleBatchReport) -> str:
    """Render a sample batch report as plain text."""
    lines = [
        "MLX Model Doctor HF Sample",
        f"Author: {batch.author}",
        f"Task: {batch.task or 'any'}",
        f"Limit: {batch.limit}",
        f"Plugin: {batch.plugin}",
        "",
        "Summary:",
        *(f"  {key}: {value}" for key, value in batch.summary.items()),
    ]
    for item in batch.items:
        summary = _item_summary(item)
        lines.extend(
            [
                "",
                f"{item.status.upper()} {item.repo_id}",
                f"  Signal: {item.signal}",
                (
                    "  Results: "
                    f"pass={summary['pass']} warn={summary['warn']} "
                    f"fail={summary['fail']} skip={summary['skip']}"
                ),
            ]
        )
        if item.error:
            lines.append(f"  Error: {item.error}")
    return "\n".join(lines)


def _batch_to_dict(batch: SampleBatchReport) -> dict[str, object]:
    return {
        "schema_version": batch.schema_version,
        "source": "hf",
        "author": batch.author,
        "task": batch.task,
        "limit": batch.limit,
        "plugin": batch.plugin,
        "summary": dict(batch.summary),
        "items": [_item_to_dict(item) for item in batch.items],
    }


def _item_to_dict(item: SampledModelResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "repo_id": item.repo_id,
        "signal": item.signal,
        "status": item.status,
        "error": item.error,
    }
    if item.report is not None:
        payload["report"] = _report_to_dict(item.report)
    return payload


def _report_to_dict(report: DoctorReport) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(render_json(report)))


def _item_summary(item: SampledModelResult) -> dict[str, int]:
    if item.report is None:
        return {"pass": 0, "warn": 0, "fail": 0, "skip": 0}
    return item.report.summary
