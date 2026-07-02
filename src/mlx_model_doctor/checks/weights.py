"""Checks over the safetensors tensor map."""

from dataclasses import dataclass

from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.report import CheckResult


@dataclass(frozen=True, slots=True)
class WeightParamCountCheck:
    """Check the weight map against the parsed headers (internal consistency)."""

    check_id: str = "text/weights.param_count"
    title: str = "Weight parameter count"

    def run(self, ctx: CheckContext) -> CheckResult:
        """Return whether every mapped tensor exists and params are non-zero."""
        header = ctx.safetensors_header()
        if header is None:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message="No safetensors header to check parameter counts.",
            )
        missing = tuple(sorted(name for name in header.weight_map if header.tensor(name) is None))
        if missing:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message="The weight map references tensors absent from every shard header.",
                remediation="Fix the safetensors index weight_map or add the missing shard tensors.",
                details={"missing_tensors": missing},
            )
        empty_files = tuple(sorted(fh.filename for fh in header.files if not fh.tensors))
        total = header.total_stored_element_count()
        if empty_files or total == 0:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message="The safetensors header reports no tensor parameters.",
                remediation="Confirm the safetensors shards actually contain weights.",
                details={"empty_or_zero_param_files": empty_files, "stored_element_count": total},
            )
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message="The weight map resolves to present tensors with non-zero parameters.",
            details={"stored_element_count": total},
        )


_INPUT_EMBED_NAMES = (
    "model.embed_tokens.weight",
    "transformer.wte.weight",
    "embed_tokens.weight",
    "tok_embeddings.weight",
)
_OUTPUT_HEAD_NAMES = (
    "lm_head.weight",
    "output.weight",
    "transformer.lm_head.weight",
)


@dataclass(frozen=True, slots=True)
class TiedEmbeddingCheck:
    """Cross-check tie_word_embeddings against stored embedding/head tensors."""

    check_id: str = "text/weights.tied_embedding"
    title: str = "Tied embeddings"

    def run(self, ctx: CheckContext) -> CheckResult:
        """Return whether declared tying matches the stored embedding tensors."""
        header = ctx.safetensors_header()
        config = ctx.config_json()
        if header is None or config is None:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message="No safetensors header or config to check embedding tying.",
            )
        has_input = any(header.tensor(name) is not None for name in _INPUT_EMBED_NAMES)
        has_output = any(header.tensor(name) is not None for name in _OUTPUT_HEAD_NAMES)
        if not has_input and not has_output:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message="No recognized embedding or output-head tensor; cannot check tying.",
            )
        tied = config.get("tie_word_embeddings") is True
        if tied and has_input and has_output:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message="tie_word_embeddings is true but both embedding and lm-head weights are stored.",
                remediation="Drop the duplicate lm_head weight or set tie_word_embeddings to false.",
                details={"stored_both_distinct": True},
            )
        if not tied and not has_output:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message="tie_word_embeddings is not enabled but no output-head weight is stored.",
                remediation="Store an lm_head weight or set tie_word_embeddings to true.",
                details={"missing_output_head": True},
            )
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message="Embedding tying is consistent with the stored tensors.",
        )
