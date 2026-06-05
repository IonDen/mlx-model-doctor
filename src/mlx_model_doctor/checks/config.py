"""Checks for model config.json contents."""

import json
from dataclasses import dataclass
from typing import cast

from mlx_model_doctor.context import _MAX_METADATA_BYTES, CheckContext
from mlx_model_doctor.errors import TargetError, raise_for_hf_target_error
from mlx_model_doctor.report import CheckResult


@dataclass(frozen=True, slots=True)
class ConfigJsonCheck:
    """Check that config.json is readable as a JSON object."""

    check_id: str = "text/config.json"
    title: str = "Config JSON"

    def run(self, ctx: CheckContext) -> CheckResult:
        """Return whether config.json can be parsed as an object."""
        try:
            if not ctx.target.exists("config.json"):
                return self._missing_result()
            size = ctx.target.size("config.json")
            if size is not None and size > _MAX_METADATA_BYTES:
                return CheckResult(
                    check_id=self.check_id,
                    title=self.title,
                    status="fail",
                    severity="high",
                    message=f"config.json is too large to validate ({size} bytes).",
                    remediation="Ensure config.json is a normal model configuration file.",
                )
            raw_config = ctx.target.read_text("config.json")
        except FileNotFoundError:
            return self._missing_result()
        except TargetError as exc:
            raise_for_hf_target_error(exc)
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message=f"Could not read config.json: {exc}",
                remediation="Ensure config.json is readable from the model repository.",
            )
        except UnicodeError:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message="config.json is not valid UTF-8 text.",
                remediation="Write config.json as UTF-8 encoded JSON.",
            )

        try:
            parsed_config: object = json.loads(raw_config)
        except json.JSONDecodeError as exc:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message=f"config.json contains invalid JSON: {exc.msg}.",
                remediation="Fix config.json so it contains valid JSON.",
            )

        if not isinstance(parsed_config, dict):
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="fail",
                severity="high",
                message="config.json must contain a JSON object.",
                remediation="Replace config.json with an object containing model metadata.",
            )

        config = cast("dict[str, object]", parsed_config)
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message="config.json contains a valid JSON object.",
            details={"keys": tuple(sorted(config))},
        )

    def _missing_result(self) -> CheckResult:
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="fail",
            severity="high",
            message="Missing config.json; cannot validate model configuration.",
            remediation="Add a valid config.json file to the model repository.",
        )


@dataclass(frozen=True, slots=True)
class ModelTypeCheck:
    """Check that config.json declares a model_type."""

    check_id: str = "text/config.model_type"
    title: str = "Model type"

    def run(self, ctx: CheckContext) -> CheckResult:
        """Return whether model_type is available in parsed config."""
        config = ctx.config_json()
        if config is None:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="skip",
                severity="info",
                message="config.json is unavailable, so model_type cannot be checked.",
            )

        model_type = config.get("model_type")
        if not isinstance(model_type, str) or not model_type.strip():
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status="warn",
                severity="medium",
                message="config.json does not declare a non-empty string model_type.",
                remediation="Add a string model_type to config.json so tooling can identify the architecture.",
            )

        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message=f"config.json declares model_type={model_type!s}.",
            details={"model_type": model_type},
        )
