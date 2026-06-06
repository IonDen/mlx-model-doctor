"""Checks for safetensors index metadata."""

import json
from dataclasses import dataclass
from typing import cast

from mlx_model_doctor.context import _MAX_METADATA_BYTES, CheckContext
from mlx_model_doctor.errors import TargetError, raise_for_hf_target_error
from mlx_model_doctor.report import CheckResult
from mlx_model_doctor.safetensors_header import FileHeader

SAFETENSORS_INDEX_SUFFIX = ".safetensors.index.json"


@dataclass(frozen=True, slots=True)
class SafetensorsIndexCheck:
    """Check that safetensors index shards are referenced consistently."""

    check_id: str = "text/safetensors.index"
    title: str = "Safetensors index"

    def run(self, ctx: CheckContext) -> CheckResult:
        """Return whether the safetensors index references present shard files."""
        try:
            index_paths = tuple(
                sorted(
                    path
                    for path in ctx.target.list_files()
                    if path.endswith(SAFETENSORS_INDEX_SUFFIX)
                )
            )
        except TargetError as exc:
            raise_for_hf_target_error(exc)
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message=f"Could not list target files for safetensors indexes: {exc}",
                remediation="Ensure the model target can be listed and safetensors indexes are readable.",
            )

        if not index_paths:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message="No safetensors index was found.",
            )

        all_shards: set[str] = set()
        for index_path in index_paths:
            result = _validate_index(ctx, self.check_id, self.title, index_path)
            if result.status != "pass":
                return result
            shards = result.details.get("shards")
            if isinstance(shards, tuple):
                all_shards.update(shard for shard in shards if isinstance(shard, str))

        details: dict[str, object] = {
            "index_paths": index_paths,
            "shards": tuple(sorted(all_shards)),
        }
        if len(index_paths) == 1:
            details["index_path"] = index_paths[0]

        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message="Safetensors indexes reference shard files that are present.",
            details=details,
        )


def _validate_index(
    ctx: CheckContext,
    check_id: str,
    title: str,
    index_path: str,
) -> CheckResult:
    try:
        size = ctx.target.size(index_path)
        if size is not None and size > _MAX_METADATA_BYTES:
            return CheckResult(
                check_id=check_id,
                title=title,
                status="warn",
                severity="medium",
                message=f"Safetensors index {index_path} is too large to validate ({size} bytes).",
                remediation=f"Ensure {index_path} is a normal safetensors index file.",
                details={"index_path": index_path},
            )
        raw_index = ctx.target.read_text(index_path)
    except FileNotFoundError:
        return CheckResult(
            check_id=check_id,
            title=title,
            status="warn",
            severity="medium",
            message=f"Safetensors index {index_path} was listed but could not be read.",
            remediation=f"Ensure {index_path} still exists and is readable UTF-8 JSON.",
            details={"index_path": index_path},
        )
    except TargetError as exc:
        raise_for_hf_target_error(exc)
        return CheckResult(
            check_id=check_id,
            title=title,
            status="warn",
            severity="medium",
            message=f"Could not read safetensors index {index_path}: {exc}",
            remediation=f"Ensure {index_path} is readable UTF-8 JSON.",
            details={"index_path": index_path},
        )
    except UnicodeError as exc:
        return CheckResult(
            check_id=check_id,
            title=title,
            status="warn",
            severity="medium",
            message=f"Could not read safetensors index {index_path}: {exc}",
            remediation=f"Ensure {index_path} is readable UTF-8 JSON.",
            details={"index_path": index_path},
        )

    try:
        parsed_index: object = json.loads(raw_index)
    except json.JSONDecodeError as exc:
        return CheckResult(
            check_id=check_id,
            title=title,
            status="fail",
            severity="high",
            message=f"Safetensors index {index_path} contains invalid JSON: {exc.msg}.",
            remediation=f"Fix {index_path} so it contains valid JSON.",
            details={"index_path": index_path},
        )

    if not isinstance(parsed_index, dict):
        return CheckResult(
            check_id=check_id,
            title=title,
            status="warn",
            severity="medium",
            message=f"Safetensors index {index_path} should contain a JSON object.",
            remediation=f"Replace {index_path} with an object containing weight_map.",
            details={"index_path": index_path},
        )

    index = cast("dict[str, object]", parsed_index)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        return CheckResult(
            check_id=check_id,
            title=title,
            status="warn",
            severity="medium",
            message=f"Safetensors index {index_path} does not contain a weight_map object.",
            remediation="Add a weight_map object mapping tensor names to safetensors shard files.",
            details={"index_path": index_path},
        )

    shard_names = tuple(sorted({value for value in weight_map.values() if isinstance(value, str)}))
    if not shard_names:
        return CheckResult(
            check_id=check_id,
            title=title,
            status="warn",
            severity="medium",
            message=f"Safetensors index {index_path} weight_map does not reference any shard files.",
            remediation="Populate weight_map with tensor-to-shard entries.",
            details={"index_path": index_path},
        )

    missing_shards: list[str] = []
    invalid_shards: list[str] = []
    for shard in shard_names:
        try:
            shard_exists = ctx.target.exists(shard)
        except TargetError as exc:
            raise_for_hf_target_error(exc)
            missing_shards.append(shard)
            invalid_shards.append(shard)
            continue
        except UnicodeError:
            missing_shards.append(shard)
            invalid_shards.append(shard)
            continue
        if not shard_exists:
            missing_shards.append(shard)

    if missing_shards:
        first_missing = missing_shards[0]
        return CheckResult(
            check_id=check_id,
            title=title,
            status="fail",
            severity="high",
            message=f"Safetensors index {index_path} references invalid or missing shard {first_missing}.",
            remediation="Add the missing safetensors shard or fix the index weight_map.",
            details={
                "index_path": index_path,
                "missing_shards": tuple(missing_shards),
                "invalid_shards": tuple(invalid_shards),
                "shards": shard_names,
            },
        )

    return CheckResult(
        check_id=check_id,
        title=title,
        status="pass",
        severity="info",
        message="Safetensors index references shard files that are present.",
        details={"index_path": index_path, "shards": shard_names},
    )


_KNOWN_ST_DTYPES = frozenset(
    {
        "BOOL",
        "U8",
        "I8",
        "F8_E5M2",
        "F8_E4M3",
        "I16",
        "U16",
        "F16",
        "BF16",
        "I32",
        "U32",
        "F32",
        "F64",
        "I64",
        "U64",
    }
)


@dataclass(frozen=True, slots=True)
class SafetensorsOffsetScanCheck:
    """Scan safetensors header offsets for corruption (no weight download)."""

    check_id: str = "text/safetensors.offsets"
    title: str = "Safetensors offsets"

    def run(self, ctx: CheckContext) -> CheckResult:
        """Return whether tensor data offsets are well-formed and in bounds."""
        header = ctx.safetensors_header()
        if header is None:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message="No safetensors header to scan.",
            )
        out_of_bounds: list[str] = []
        overlapping: list[str] = []
        unknown_dtypes: list[str] = []
        gaps: list[str] = []
        size_unknown = False
        for file_header in header.files:
            size_unknown = (
                self._scan_file(file_header, out_of_bounds, overlapping, unknown_dtypes, gaps)
                or size_unknown
            )
        if out_of_bounds or overlapping:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message="Safetensors header has corrupt tensor offsets (overlap or out of bounds).",
                remediation="Re-save the safetensors shard; its tensor offsets are inconsistent.",
                details={
                    "out_of_bounds": tuple(out_of_bounds),
                    "overlapping": tuple(overlapping),
                },
            )
        if unknown_dtypes or gaps:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message="Safetensors header offsets are usable but could not be fully validated.",
                remediation="Review unknown dtypes or non-contiguous offsets.",
                details={
                    "unknown_dtypes": tuple(unknown_dtypes),
                    "gaps": tuple(gaps),
                },
            )
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message=(
                "Safetensors header tensor offsets are well-formed and in bounds."
                if not size_unknown
                else "Safetensors tensor offsets are well-formed; the data-section upper "
                "bound was not checked (header length unavailable on this target)."
            ),
            details={"upper_bound_checked": not size_unknown},
        )

    def _scan_file(
        self,
        file_header: FileHeader,
        out_of_bounds: list[str],
        overlapping: list[str],
        unknown_dtypes: list[str],
        gaps: list[str],
    ) -> bool:
        data_section_length = file_header.data_section_length
        entries = sorted(file_header.tensors.items(), key=lambda kv: kv[1].data_offsets[0])
        prev_end = 0
        prev_name: str | None = None
        for name, entry in entries:
            begin, end = entry.data_offsets
            label = f"{file_header.filename}:{name}"
            if begin < 0 or end < begin:
                out_of_bounds.append(label)
                continue
            if data_section_length is not None and end > data_section_length:
                out_of_bounds.append(label)
            if prev_name is not None and begin < prev_end:
                overlapping.append(f"{file_header.filename}:{prev_name}->{name}")
            elif prev_name is not None and begin > prev_end:
                gaps.append(label)
            if entry.dtype not in _KNOWN_ST_DTYPES:
                unknown_dtypes.append(f"{label}({entry.dtype})")
            prev_end = max(prev_end, end)
            prev_name = name
        return data_section_length is None
