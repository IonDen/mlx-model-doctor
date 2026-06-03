"""Report models and renderers."""

import json
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

Status = Literal["pass", "warn", "fail", "skip"]
Severity = Literal["info", "low", "medium", "high"]
VALID_STATUSES = frozenset(("pass", "warn", "fail", "skip"))
VALID_SEVERITIES = frozenset(("info", "low", "medium", "high"))


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckResult:
    """A single model-doctor check result."""

    check_id: str
    title: str
    status: Status
    severity: Severity
    message: str
    remediation: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)
    duration_s: float | None = None

    def __post_init__(self) -> None:
        """Validate result invariants."""
        if self.status not in VALID_STATUSES:
            msg = f"status must be one of {sorted(VALID_STATUSES)}"
            raise ValueError(msg)
        if self.severity not in VALID_SEVERITIES:
            msg = f"severity must be one of {sorted(VALID_SEVERITIES)}"
            raise ValueError(msg)
        if self.status in {"pass", "skip"} and self.severity != "info":
            msg = f"{self.status} results must use severity='info'"
            raise ValueError(msg)
        namespace, separator, name = self.check_id.partition("/")
        if not namespace or not separator or not name:
            msg = "check_id must be plugin-namespaced, for example 'text/files.required'"
            raise ValueError(msg)
        object.__setattr__(self, "details", _freeze_mapping(self.details))


@dataclass(frozen=True, slots=True, kw_only=True)
class DoctorReport:
    """A full model-doctor report."""

    target: str
    source: Literal["local", "hf"]
    plugin: str
    results: Sequence[CheckResult]
    environment: Mapping[str, object] = field(default_factory=dict)
    schema_version: str = "1.0"
    zero_check_reason: str | None = None

    @property
    def summary(self) -> dict[str, int]:
        """Return result counts grouped by status."""
        return {
            status: sum(1 for result in self.results if result.status == status)
            for status in ("pass", "warn", "fail", "skip")
        }

    def __post_init__(self) -> None:
        """Copy caller-owned collections so the report is stable."""
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "environment", _freeze_mapping(self.environment))


def _result_to_dict(result: CheckResult) -> dict[str, object]:
    return {
        "check_id": result.check_id,
        "title": result.title,
        "status": result.status,
        "severity": result.severity,
        "message": result.message,
        "remediation": result.remediation,
        "details": dict(result.details),
        "duration_s": result.duration_s,
    }


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, AbstractSet):
        return frozenset(_freeze_value(item) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(_freeze_value(item) for item in value)
    return value


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, AbstractSet):
        return sorted((_json_safe(item) for item in value), key=str)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_safe(item) for item in value]
    return str(value)


def render_json(report: DoctorReport) -> str:
    """Render a report as stable JSON."""
    payload = {
        "schema_version": report.schema_version,
        "target": report.target,
        "source": report.source,
        "plugin": report.plugin,
        "summary": report.summary,
        "environment": _json_safe(report.environment),
        "zero_check_reason": report.zero_check_reason,
        "results": [_json_safe(_result_to_dict(result)) for result in report.results],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def render_text(report: DoctorReport) -> str:
    """Render a report as plain text."""
    lines = [
        f"MLX Model Doctor: {report.target}",
        "",
        "Summary:",
        *(f"  {key}: {value}" for key, value in report.summary.items()),
    ]
    for result in report.results:
        lines.extend(
            [
                "",
                f"{result.status.upper()} {result.severity} {result.check_id}",
                f"  {result.title}: {result.message}",
            ]
        )
        if result.remediation:
            lines.append(f"  Fix: {result.remediation}")
    return "\n".join(lines)


def render_markdown(report: DoctorReport) -> str:
    """Render a report as Markdown."""
    lines = [
        f"# MLX Model Doctor: {report.target}",
        "",
        "| Status | Count |",
        "|---|---:|",
        *(f"| {key} | {value} |" for key, value in report.summary.items()),
        "",
        "## Results",
        "",
    ]
    for result in report.results:
        lines.extend(
            [
                f"### {result.status.upper()} {result.check_id}",
                "",
                f"**{result.title}.** {result.message}",
                "",
            ]
        )
        if result.remediation:
            lines.extend([f"Remediation: {result.remediation}", ""])
    return "\n".join(lines)
