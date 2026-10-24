"""Classify a repository's safetensors index against the shards it actually ships.

One pure classifier shared by ``text/safetensors.index`` and both targets, so the
index check and the header the weight checks read use the same presence rule and the
same shipped-shard set. One case differs on purpose: an index that fails for an invalid
entry (a ``..`` or absolute value) still lets the header read the shipped shards.

Why a stale root index is not missing weights: MLX loaders do not follow it.
mlx-vlm v0.6.12 ``mlx_vlm/utils.py`` ``load_model`` keeps only the index-named shards
that exist (``if (model_path / shard).exists()``) and, when none remain, falls back to
a ``*.safetensors`` glob that skips ``consolidated.safetensors``. mlx-lm 0.31.3
``mlx_lm/utils.py:316`` ``load_model`` globs ``model*.safetensors`` at the repo root and
never reads the index. So a shard counts as shipped only when both globs pick it up.
Index-following loaders and ``huggingface_hub``'s ``get_safetensors_metadata`` fail.
"""

import fnmatch
import json
import posixpath
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

SAFETENSORS_SUFFIX = ".safetensors"
SAFETENSORS_INDEX_SUFFIX = ".safetensors.index.json"
STANDARD_ROOT_INDEX = "model.safetensors.index.json"
LOADER_SHARD_GLOB = "model*.safetensors"  # mlx-lm 0.31.3 load_model glob
CONSOLIDATED_SHARD = "consolidated.safetensors"  # skipped by mlx-vlm 0.6.12's fallback
# The most shipped shards whose headers a stale-index read fetches (one request each).
_MAX_SHIPPED_SHARDS = 512
# The most absent shard names a message lists before summarizing the rest.
_MAX_LISTED_NAMES = 8
_CONTROL_CHARACTERS = dict.fromkeys((*range(0x20), 0x7F), "?")


def printable_name(name: str) -> str:
    """Return a repo-controlled name with control characters replaced by ``?``.

    A name from an index can hold a newline and fake a line in the text report; messages
    use this form while ``details`` keep the raw name.
    """
    return name.translate(_CONTROL_CHARACTERS)


def listed_names(names: Sequence[str]) -> str:
    """Join the first few names for a message, then count the rest."""
    shown = [printable_name(name) for name in names[:_MAX_LISTED_NAMES]]
    rest = len(names) - len(shown)
    return ", ".join([*shown, f"and {rest} more"] if rest > 0 else shown)


class ShardLayoutKind(StrEnum):
    """How a safetensors index relates to the shards present in the repository."""

    NO_INDEX = "no_index"
    CONSISTENT = "consistent"
    STALE_ALL_ABSENT_WITH_SHIPPED = "stale_all_absent_with_shipped"
    PARTIAL_MISSING = "partial_missing"
    ALL_MISSING = "all_missing"

    @property
    def missing_weights(self) -> bool:
        """Return whether this layout means index-named weights are really missing."""
        return self in (ShardLayoutKind.PARTIAL_MISSING, ShardLayoutKind.ALL_MISSING)


@dataclass(frozen=True, slots=True, kw_only=True)
class ShardLayout:
    """The classification plus the sorted shard lists behind it."""

    kind: ShardLayoutKind
    index_path: str | None
    named: tuple[str, ...]  # resolved shard paths the index names
    absent: tuple[str, ...]  # named shards not in the listing
    shipped: tuple[str, ...]  # loader-eligible top-level shards the index does not name

    def absent_message(self) -> str:
        """Describe the absent index-named shards, sorted, for a header error."""
        count = len(self.absent)
        noun = "shard" if count == 1 else "shards"
        index = printable_name(self.index_path or "")
        return (
            f"safetensors index {index} names {count} {noun} absent from the "
            f"repository: {listed_names(self.absent)}"
        )

    def shipped_limit_message(self) -> str | None:
        """Return why the shipped shards are too many to read, or None within the cap."""
        count = len(self.shipped)
        if count <= _MAX_SHIPPED_SHARDS:
            return None
        return (
            f"safetensors index {printable_name(self.index_path or '')} is stale and the "
            f"repository ships {count} shards, more than the {_MAX_SHIPPED_SHARDS} whose "
            "headers are read"
        )


def resolve_shard_path(index_path: str, value: str) -> str:
    """Resolve a weight_map shard value relative to the index file's directory.

    Safetensors index ``weight_map`` values name shards relative to the directory that
    contains the index, so a component index at ``text_encoder/model.safetensors.index.json``
    that lists ``0.safetensors`` refers to ``text_encoder/0.safetensors``. A root-level index
    (empty directory) leaves the value unchanged. The result is normalized (``./x`` is
    ``x``, ``a/../b`` is ``b``) as a filesystem lookup or the Hub's URL resolution would
    read it; a leading ``..`` survives, so an escape stays visible to the target.
    """
    index_dir = posixpath.dirname(index_path)
    return posixpath.normpath(posixpath.join(index_dir, value) if index_dir else value)


def root_index_path(files: Sequence[str]) -> str | None:
    """Return the repository's root safetensors index path, or None.

    ``model.safetensors.index.json`` wins (it is the file loaders and huggingface_hub
    read); otherwise the first sorted top-level ``*.safetensors.index.json``. Nested
    component indexes are never the root index.
    """
    roots = sorted(
        path for path in files if "/" not in path and path.endswith(SAFETENSORS_INDEX_SUFFIX)
    )
    if STANDARD_ROOT_INDEX in roots:
        return STANDARD_ROOT_INDEX
    return roots[0] if roots else None


def parse_index_weight_map(text: str) -> dict[str, str] | None:
    """Return an index's string-valued weight_map, or None when unusable."""
    try:
        parsed: object = json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        return None
    if not isinstance(parsed, dict):
        return None
    weight_map = parsed.get("weight_map")
    if not isinstance(weight_map, dict):
        return None
    return {str(key): value for key, value in weight_map.items() if isinstance(value, str)}


def is_loader_eligible(path: str) -> bool:
    """Return whether both MLX loaders would load this top-level shard without an index.

    mlx-lm globs ``model*.safetensors``; mlx-vlm's fallback glob skips
    ``consolidated.safetensors``.
    """
    return (
        "/" not in path
        and fnmatch.fnmatchcase(path, LOADER_SHARD_GLOB)
        and path != CONSOLIDATED_SHARD
    )


def classify_shard_layout(
    files: Sequence[str],
    weight_map: Mapping[str, str] | None,
    *,
    index_path: str | None,
    present: Callable[[str], bool] | None = None,
) -> ShardLayout:
    """Classify an index's weight_map against the repository file listing.

    ``present`` decides whether a named shard exists. It defaults to exact listing
    membership; a target whose own lookup resolves paths (for example on a
    case-insensitive filesystem) passes that lookup, so this verdict agrees with
    ``text/safetensors.index``.

    ``stale_all_absent_with_shipped`` is recognised only for a root-level index, the
    layout MLX loaders glob: every named shard is absent and other loader-eligible
    top-level shards ship (see :func:`is_loader_eligible`). Every list in the result is
    sorted, so messages built from it are stable.
    """
    listed = set(files)
    top_level = sorted(
        path for path in listed if "/" not in path and path.endswith(SAFETENSORS_SUFFIX)
    )
    named: tuple[str, ...] = ()
    if weight_map and index_path is not None:
        named = tuple(sorted({resolve_shard_path(index_path, v) for v in weight_map.values()}))
    if not named:
        return ShardLayout(
            kind=ShardLayoutKind.NO_INDEX,
            index_path=index_path,
            named=(),
            absent=(),
            shipped=tuple(shard for shard in top_level if is_loader_eligible(shard)),
        )
    named_set = set(named)
    is_present = present if present is not None else listed.__contains__
    absent = tuple(shard for shard in named if not is_present(shard))
    shipped = tuple(
        shard for shard in top_level if shard not in named_set and is_loader_eligible(shard)
    )
    if not absent:
        kind = ShardLayoutKind.CONSISTENT
    elif len(absent) < len(named):
        kind = ShardLayoutKind.PARTIAL_MISSING
    elif shipped and index_path is not None and "/" not in index_path:
        kind = ShardLayoutKind.STALE_ALL_ABSENT_WITH_SHIPPED
    else:
        kind = ShardLayoutKind.ALL_MISSING
    return ShardLayout(
        kind=kind, index_path=index_path, named=named, absent=absent, shipped=shipped
    )
