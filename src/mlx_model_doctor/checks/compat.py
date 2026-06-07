"""MLX-compatibility signal check."""

from dataclasses import dataclass

from mlx_model_doctor.compat import WEAK_SIGNALS, mlx_signals
from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.report import CheckResult
from mlx_model_doctor.safetensors_header import SafetensorsHeader
from mlx_model_doctor.targets import MlxListingMetadata

_SCALES_SUFFIX = ".scales"


@dataclass(frozen=True, slots=True)
class MlxCompatSignalCheck:
    """Report whether a repository looks like an MLX/mlx-lm model, and why."""

    check_id: str = "text/compat.mlx_signal"
    title: str = "MLX compatibility signal"

    def run(self, ctx: CheckContext) -> CheckResult:
        """Compute and report MLX-compatibility signals (informational only).

        Args:
            ctx: The check context for the target model repository.

        Returns:
            A ``CheckResult`` with ``status="pass"`` and ``severity="info"``
            in all cases.  The ``details["signals"]`` tuple contains any
            detected MLX-compatibility signals.
        """
        signals = mlx_compat_signals(ctx)
        strong = tuple(s for s in signals if s not in WEAK_SIGNALS)
        if strong:
            message = f"MLX-compatibility signals: {', '.join(signals)}."
        elif signals:
            message = f"Weak MLX hint only ({', '.join(signals)}); no MLX-specific metadata found."
        else:
            message = "No MLX-compatibility signals found; this may not be an MLX/mlx-lm model."
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message=message,
            details={"signals": signals},
        )


def mlx_compat_signals(ctx: CheckContext) -> tuple[str, ...]:
    """Gather signal inputs from the context and delegate to the pure core.

    Args:
        ctx: The check context for the target model repository.

    Returns:
        A tuple of MLX-compatibility signal strings, highest-priority first.
    """
    target = ctx.target
    if isinstance(target, MlxListingMetadata):
        tags = target.tags
        library_name = target.library_name
    else:
        tags = frozenset()
        library_name = None

    has_quantized_tensors = False
    if ctx.options.include_weights:
        header = ctx.safetensors_header()
        has_quantized_tensors = header is not None and _header_has_quantized_tensors(header)

    return mlx_signals(
        name=target.name,
        source=target.source,
        tags=tags,
        library_name=library_name,
        config=ctx.config_json(),
        has_quantized_tensors=has_quantized_tensors,
    )


def _header_has_quantized_tensors(header: SafetensorsHeader) -> bool:
    """Return True when the header contains MLX-quantized weight tensors.

    Args:
        header: The aggregated safetensors header to inspect.

    Returns:
        ``True`` when a ``*.scales`` tensor exists alongside a ``*.weight``
        tensor with dtype ``U32`` (the MLX quantized-weight layout).
    """
    for name in header.tensor_names():
        if name.endswith(_SCALES_SUFFIX):
            weight = header.tensor(name[: -len(_SCALES_SUFFIX)] + ".weight")
            if weight is not None and weight.dtype == "U32":
                return True
    return False
