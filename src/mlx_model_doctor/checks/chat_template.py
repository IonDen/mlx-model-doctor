"""Checks for chat-template presence and special-token consistency."""

import re
from dataclasses import dataclass

from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.report import CheckResult

_TOKEN_RE = re.compile(r"<\|[^\s|>]+\|?>")


def _tokenizer_config_template(ctx: CheckContext) -> str | None:
    tokenizer_config = ctx.tokenizer_config_json()
    if tokenizer_config is None:
        return None
    template = tokenizer_config.get("chat_template")
    if isinstance(template, str) and template.strip():
        return template
    if isinstance(template, list):
        bodies = [
            entry["template"]
            for entry in template
            if isinstance(entry, dict) and isinstance(entry.get("template"), str)
        ]
        if bodies:
            return "\n".join(bodies)
    return None


def _chat_template_json_template(ctx: CheckContext) -> str | None:
    chat_template_json = ctx.chat_template_json()
    if chat_template_json is None:
        return None
    template = chat_template_json.get("chat_template")
    return template if isinstance(template, str) and template.strip() else None


def _jinja_template(ctx: CheckContext) -> str | None:
    jinja = ctx.chat_template_text()
    return jinja if jinja is not None and jinja.strip() else None


def _template_string(ctx: CheckContext, *, prefer_processor_template: bool = False) -> str | None:
    """Return the effective chat-template string from the source the runtime actually loads.

    Text profile: mlx-lm's tokenizer loads chat_template.jinja then tokenizer_config.chat_template
    (AutoTokenizer ignores chat_template.json), so scan jinja -> tokenizer_config, with
    chat_template.json last so presence still detects a template that lives only there. VLM
    profile: the processor renders chat_template.json when present and otherwise falls back to its
    tokenizer, so scan chat_template.json -> jinja -> tokenizer_config.
    """
    getters = (
        (_chat_template_json_template, _jinja_template, _tokenizer_config_template)
        if prefer_processor_template
        else (_jinja_template, _tokenizer_config_template, _chat_template_json_template)
    )
    for getter in getters:
        template = getter(ctx)
        if template is not None:
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


# Chat-template sources (tokenizer_config.json / chat_template.json / chat_template.jinja) and their
# load precedence verified against transformers 5.15, mlx-lm 0.31.3, and mlx-vlm 0.6.x.
@dataclass(frozen=True, slots=True)
class ChatTemplatePresenceCheck:
    """Check that a chat template is present in either supported location."""

    check_id: str = "text/chat_template.presence"
    title: str = "Chat template"

    def run(self, ctx: CheckContext) -> CheckResult:
        """Return whether a chat template is present (or expected-but-absent)."""
        has_tokenizer_config = ctx.target.exists("tokenizer_config.json")
        has_jinja = ctx.target.exists("chat_template.jinja")
        has_chat_template_json = ctx.target.exists("chat_template.json")
        if not has_tokenizer_config and not has_jinja and not has_chat_template_json:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message="No tokenizer metadata, so a chat template cannot be checked.",
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
                    "chat_template_json": has_chat_template_json,
                },
            )
        if has_tokenizer_config and ctx.tokenizer_config_json() is None:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message="tokenizer_config.json is present but could not be parsed.",
                remediation="Ensure tokenizer_config.json is valid JSON.",
            )
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="warn",
            severity="low",
            message=(
                "No chat template found in tokenizer_config.json, chat_template.json, or "
                "chat_template.jinja; apply_chat_template() will fail for a chat/instruct model "
                "(for a base/non-chat model this is expected)."
            ),
            remediation=(
                "Add a chat_template to tokenizer_config.json, a chat_template.json, or a "
                "chat_template.jinja file."
            ),
        )


def _registered_literals(ctx: CheckContext) -> set[str]:
    literals: set[str] = set()
    tokenizer_config = ctx.tokenizer_config_json() or {}
    decoder = tokenizer_config.get("added_tokens_decoder")
    if isinstance(decoder, dict):
        for entry in decoder.values():
            if isinstance(entry, dict) and isinstance(entry.get("content"), str):
                literals.add(entry["content"])
    for source in (tokenizer_config, ctx.special_tokens_map_json() or {}):
        for key in ("eos_token", "bos_token", "pad_token", "unk_token"):
            value = source.get(key)
            if isinstance(value, str):
                literals.add(value)
            elif isinstance(value, dict) and isinstance(value.get("content"), str):
                literals.add(value["content"])
        extra = source.get("additional_special_tokens")
        if isinstance(extra, list):
            literals.update(item for item in extra if isinstance(item, str))
    return literals


@dataclass(frozen=True, slots=True)
class ChatTemplateSpecialTokensCheck:
    """Check that chat-template end-of-turn literals are registered special tokens."""

    check_id: str = "text/chat_template.special_tokens"
    title: str = "Chat template tokens"
    prefer_processor_template: bool = False

    def run(self, ctx: CheckContext) -> CheckResult:
        """Return whether template-emitted literals match registered special tokens."""
        template = _template_string(ctx, prefer_processor_template=self.prefer_processor_template)
        if template is None:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message="No chat template string, so token consistency cannot be checked.",
            )
        registered = _registered_literals(ctx)
        if not registered:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message="No special-token metadata to cross-check against the template.",
            )
        unregistered = sorted(set(_TOKEN_RE.findall(template)) - registered)
        if unregistered:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message=(
                    f"Chat template uses token literal(s) {unregistered} not registered as "
                    "special tokens; a typo here (e.g. a missing delimiter) makes generation "
                    "never stop. (A typo that is itself registered, or sub-word fragmentation, "
                    "needs the full tokenizer.json — not checked here.)"
                ),
                remediation="Register the template's end-of-turn token, or fix the registered literal.",
                details={"unregistered": unregistered},
            )
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message="Chat-template token literals are registered special tokens.",
        )
