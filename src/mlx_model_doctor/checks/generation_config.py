"""Check for cross-file generation token-id consistency."""

from dataclasses import dataclass

from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.report import CheckResult

_TOKEN_KEYS = ("eos_token_id", "pad_token_id", "bos_token_id")


def _id_set(value: object) -> frozenset[int]:
    if type(value) is int:
        return frozenset({value})
    if isinstance(value, list):
        return frozenset(item for item in value if type(item) is int)
    return frozenset()


@dataclass(frozen=True, slots=True)
class GenerationConfigTokensCheck:
    """Check generation/config token IDs are present and agree across files."""

    check_id: str = "text/generation_config.tokens"
    title: str = "Generation-config tokens"

    def run(self, ctx: CheckContext) -> CheckResult:
        """Return whether token IDs are consistent across config files."""
        if ctx.generation_config_json() is None and ctx.target.exists("generation_config.json"):
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message="generation_config.json is present but could not be parsed.",
                remediation="Ensure generation_config.json is valid JSON.",
            )

        sources = {
            "config.json": ctx.config_json() or {},
            "generation_config.json": ctx.generation_config_json() or {},
            "tokenizer_config.json": ctx.tokenizer_config_json() or {},
        }
        has_any = any(
            _id_set(source.get(key)) for source in sources.values() for key in _TOKEN_KEYS
        )
        if not has_any:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message="No generation token IDs declared, so consistency cannot be checked.",
            )

        gen = sources["generation_config.json"]
        if _id_set(gen.get("eos_token_id")) and not _id_set(gen.get("pad_token_id")):
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="low",
                message=(
                    "generation_config.json sets eos_token_id but no pad_token_id; "
                    "some generate() paths then crash. (Base/non-chat models legitimately omit "
                    "pad_token_id and generate fine — informational.)"
                ),
                remediation="Add a pad_token_id to generation_config.json when padding is used.",
            )

        for key in _TOKEN_KEYS:
            declared = {
                name: _id_set(source.get(key))
                for name, source in sources.items()
                if _id_set(source.get(key))
            }
            distinct = {frozenset(value) for value in declared.values()}
            if len(distinct) > 1:
                return CheckResult(
                    check_id=self.check_id,
                    title=self.title,
                    status="warn",
                    severity="medium",
                    message=f"{key} disagrees across config files: {declared}.",
                    remediation="Make the token IDs agree across config.json / generation_config.json / tokenizer_config.json.",
                    details={key: {name: sorted(value) for name, value in declared.items()}},
                )

        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message="Generation token IDs are present and consistent.",
        )
