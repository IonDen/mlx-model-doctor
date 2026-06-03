"""Checks for tokenizer metadata and special token configuration."""

from dataclasses import dataclass

from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.errors import TargetError
from mlx_model_doctor.report import CheckResult

TOKENIZER_ARTIFACTS = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
)


@dataclass(frozen=True, slots=True)
class TokenizerFilesCheck:
    """Check that at least one tokenizer artifact is present."""

    check_id: str = "text/tokenizer.files"
    title: str = "Tokenizer files"

    def run(self, ctx: CheckContext) -> CheckResult:
        """Return whether tokenizer artifacts are available."""
        try:
            present = tuple(path for path in TOKENIZER_ARTIFACTS if ctx.target.exists(path))
        except TargetError as exc:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message=f"Could not inspect tokenizer artifacts: {exc}",
                remediation="Ensure tokenizer files are readable from the model repository.",
            )

        if not present:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message="No tokenizer artifacts were found.",
                remediation="Add tokenizer.json or another tokenizer artifact to the model repository.",
                details={"expected_any": TOKENIZER_ARTIFACTS},
            )

        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message="Tokenizer artifacts are present.",
            details={"present": present},
        )


@dataclass(frozen=True, slots=True)
class SpecialTokensCheck:
    """Check for risky special token ID configuration."""

    check_id: str = "text/tokenizer.special_tokens"
    title: str = "Special token IDs"

    def run(self, ctx: CheckContext) -> CheckResult:
        """Return whether pad and eos token IDs are distinct when available."""
        config = ctx.config_json()
        if config is None:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message="config.json is unavailable, so special token IDs cannot be checked.",
            )

        pad_token_id = config.get("pad_token_id")
        eos_token_id = config.get("eos_token_id")
        if pad_token_id is None or eos_token_id is None:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message="pad_token_id or eos_token_id is unavailable in config.json.",
            )
        if type(pad_token_id) is not int or type(eos_token_id) is not int:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message="pad_token_id and eos_token_id should be integer token IDs.",
                remediation="Use integer token IDs in config.json special token metadata.",
                details={"pad_token_id": pad_token_id, "eos_token_id": eos_token_id},
            )

        details = {"pad_token_id": pad_token_id, "eos_token_id": eos_token_id}
        if pad_token_id == eos_token_id:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message="pad_token_id equals eos_token_id, which can make padding ambiguous.",
                remediation="Use distinct pad_token_id and eos_token_id values when the model supports it.",
                details=details,
            )

        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message="pad_token_id and eos_token_id are distinct.",
            details=details,
        )
