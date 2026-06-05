"""Checks for chat-template presence and special-token consistency."""

import re
from dataclasses import dataclass

from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.report import CheckResult

_TOKEN_RE = re.compile(r"<\|[^\s|>]+\|?>")


def _template_string(ctx: CheckContext) -> str | None:
    """Return the effective chat-template string from either location, if a string."""
    jinja = ctx.chat_template_text()
    if jinja is not None and jinja.strip():
        return jinja
    tokenizer_config = ctx.tokenizer_config_json()
    if tokenizer_config is not None:
        template = tokenizer_config.get("chat_template")
        if isinstance(template, str) and template.strip():
            return template
    return None


def _has_template(ctx: CheckContext) -> bool:
    if _template_string(ctx) is not None:
        return True
    tokenizer_config = ctx.tokenizer_config_json()
    if tokenizer_config is not None:
        template = tokenizer_config.get("chat_template")
        if isinstance(template, list) and template:
            return True
    return False


@dataclass(frozen=True, slots=True)
class ChatTemplatePresenceCheck:
    """Check that a chat template is present in either supported location."""

    check_id: str = "text/chat_template.presence"
    title: str = "Chat template"

    def run(self, ctx: CheckContext) -> CheckResult:
        """Return whether a chat template is present (or expected-but-absent)."""
        has_tokenizer_config = ctx.target.exists("tokenizer_config.json")
        has_jinja = ctx.target.exists("chat_template.jinja")
        if not has_tokenizer_config and not has_jinja:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message="No tokenizer metadata, so a chat template cannot be checked.",
            )
        if has_tokenizer_config and ctx.tokenizer_config_json() is None and not has_jinja:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message="tokenizer_config.json is present but could not be parsed.",
                remediation="Ensure tokenizer_config.json is valid JSON.",
            )
        if _has_template(ctx):
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="pass",
                severity="info",
                message="A chat template is present.",
                details={
                    "tokenizer_config": has_tokenizer_config,
                    "chat_template_jinja": has_jinja,
                },
            )
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="warn",
            severity="low",
            message=(
                "No chat template found in tokenizer_config.json or chat_template.jinja; "
                "apply_chat_template() will fail for a chat/instruct model "
                "(for a base/non-chat model this is expected)."
            ),
            remediation="Add a chat_template to tokenizer_config.json or a chat_template.jinja file.",
        )
