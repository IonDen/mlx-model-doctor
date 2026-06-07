"""Pure MLX-compatibility signal core (no I/O)."""

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Literal

MLX_NAME_TOKENS: tuple[str, ...] = ("mlx", "4bit", "8bit")
MLX_COMMUNITY_PREFIX = "mlx-community/"
WEAK_SIGNALS: frozenset[str] = frozenset({"repo-name"})


def mlx_signals(
    *,
    name: str | None,
    source: Literal["local", "hf"] | None,
    tags: frozenset[str],
    library_name: str | None,
    config: Mapping[str, object] | None,
    has_quantized_tensors: bool,
) -> tuple[str, ...]:
    """Return MLX-compatibility signals, highest-priority first, deduped."""
    signals: list[str] = []

    if any(tag.lower() == "mlx" for tag in tags):
        signals.append("tag:mlx")

    library = library_name.lower() if library_name is not None else None
    if library in {"mlx", "mlx-lm"}:
        signals.append(f"library:{library}")

    if source == "hf" and name is not None and name.startswith(MLX_COMMUNITY_PREFIX):
        signals.append("author:mlx-community")

    if _has_mlx_quant_config(config):
        signals.append("config:quantization")

    if has_quantized_tensors:
        signals.append("weights:mlx-quant")

    match_name = _name_for_match(name, source)
    if match_name is not None and any(token in match_name for token in MLX_NAME_TOKENS):
        signals.append("repo-name")

    return tuple(signals)


def _name_for_match(name: str | None, source: Literal["local", "hf"] | None) -> str | None:
    """Return the name fragment to match MLX tokens against.

    Args:
        name: The model name or path.
        source: Whether the model source is ``"local"`` or ``"hf"``.

    Returns:
        The lowercased basename for local paths, the full lowercased name for HF
        repos, or ``None`` when ``name`` is ``None``.
    """
    if name is None:
        return None
    if source == "local":
        return PurePosixPath(name).name.lower()
    # For HF repos ("org/name"), match only the repo-name part so that
    # an org component like "mlx-community" does not trigger the weak signal.
    return name.split("/")[-1].lower()


def _has_mlx_quant_config(config: Mapping[str, object] | None) -> bool:
    """Return True if *config* contains an MLX-style ``quantization`` mapping with positive bits.

    Args:
        config: The parsed ``config.json`` content, or ``None``.

    Returns:
        ``True`` when a ``quantization`` key exists, is a mapping, and has a
        numeric ``bits`` value greater than zero.  ``False`` otherwise.
    """
    if config is None:
        return False
    quantization = config.get("quantization")
    if not isinstance(quantization, Mapping):
        return False
    bits = quantization.get("bits")
    return isinstance(bits, int | float) and not isinstance(bits, bool) and bits > 0
