"""Base protocol for model checks."""

from typing import Protocol

from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.report import CheckResult

# Version-aware check policy
#
# Checks that contain version-bound allow-lists (quantization modes/bits,
# safetensors dtypes, etc.) follow a warn-not-fail convention: a value
# outside the known set gets status="warn" with a message naming the
# upstream version the list was verified against. Only structurally invalid
# values (wrong type, missing required field, contradictory metadata) may
# use status="fail". Checks key on observed metadata, not the installed
# toolchain — there is no runtime version detection.


class ModelCheck(Protocol):
    """Protocol implemented by model repository checks."""

    check_id: str
    title: str

    def run(self, ctx: CheckContext) -> CheckResult:
        """Run the check against a model target context."""
