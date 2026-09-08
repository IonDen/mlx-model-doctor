"""Offline end-to-end tests for ``check_adapter_parity`` (F1/F4/F10).

The worker backend is faked through ``ParityOptions.launcher`` (no MLX, no
subprocess), but source resolution, the embedded ``text`` reports, the
cross-target static checks, the loader-accurate resolver, the byte-compare
delta map, and the oracle all run for real against small on-disk repositories.
The launcher's argmax vectors are the only thing pinned by the test, so a wiring
regression (wrong role fed to the oracle, adapter-applied read from fused bytes,
a blocking failure that leaks to exit 0) surfaces as a wrong verdict/exit code.
"""

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import pytest

from mlx_model_doctor import (
    ParityOptions,
    ParityVerdict,
    check_adapter_parity,
    parity_exit_code,
)
from mlx_model_doctor.parity.oracle import ROLE_BASE, ROLE_FUSED, ROLE_NOISE, ROLE_REFERENCE
from mlx_model_doctor.parity.orchestrator import WorkerOutcome
from mlx_model_doctor.parity.worker import WorkerSpec
from tests.parity_fakes import lora_tensor, safetensors_header_bytes, write_safetensors_file

_F4 = struct.pack("<f", 1.0)
_F4_TWO = struct.pack("<f", 2.0)


# --- on-disk repository builders ------------------------------------------------


def _model_tensors(q_bytes: bytes) -> dict[str, tuple[str, list[int], bytes]]:
    return {
        "model.embed_tokens.weight": ("F32", [16, 8], _F4 * 128),
        "model.layers.0.self_attn.q_proj.weight": ("F32", [8, 8], q_bytes),
        "model.layers.0.self_attn.k_proj.weight": ("F32", [8, 8], _F4 * 64),
        "model.layers.0.self_attn.v_proj.weight": ("F32", [8, 8], _F4 * 64),
        "model.layers.0.self_attn.o_proj.weight": ("F32", [8, 8], _F4 * 64),
        "model.layers.0.mlp.gate_proj.weight": ("F32", [16, 8], _F4 * 128),
        "model.layers.0.mlp.up_proj.weight": ("F32", [16, 8], _F4 * 128),
        "model.layers.0.mlp.down_proj.weight": ("F32", [8, 16], _F4 * 128),
        "model.norm.weight": ("F32", [8], _F4 * 8),
        "lm_head.weight": ("F32", [16, 8], _F4 * 128),
    }


def _write_model_repo(root: Path, *, q_bytes: bytes, chat_template: str = "{{ x }}") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "llama",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "vocab_size": 16,
        "intermediate_size": 16,
        "pad_token_id": 0,
        "eos_token_id": 1,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "BPE", "vocab": {f"t{i}": i for i in range(16)}}}),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps({"bos_token": "t0", "eos_token": "t1", "chat_template": chat_template}),
        encoding="utf-8",
    )
    write_safetensors_file(root / "model.safetensors", _model_tensors(q_bytes))
    return root


def _write_adapter(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "num_layers": 1,
        "fine_tune_type": "lora",
        "lora_parameters": {"rank": 4, "scale": 20.0, "keys": ["self_attn.q_proj"]},
    }
    (root / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "adapters.safetensors").write_bytes(
        safetensors_header_bytes(
            {
                "model.layers.0.self_attn.q_proj.lora_a": lora_tensor([4, 8]),
                "model.layers.0.self_attn.q_proj.lora_b": lora_tensor([8, 4]),
            }
        )
    )
    return root


@dataclass(frozen=True)
class _Repos:
    base: str
    adapter: str
    fused_diff: str  # q_proj differs from base (a genuine fuse)
    fused_same: str  # q_proj byte-identical to base (a copied-base fuse)
    fused_bad_tokenizer: str  # a changed chat template (a real tokenizer incompatibility)


@pytest.fixture
def tiny_local_repos(tmp_path: Path) -> _Repos:
    base = _write_model_repo(tmp_path / "base", q_bytes=_F4 * 64)
    fused_diff = _write_model_repo(tmp_path / "fused_diff", q_bytes=_F4_TWO * 64)
    fused_same = _write_model_repo(tmp_path / "fused_same", q_bytes=_F4 * 64)
    fused_bad_tokenizer = _write_model_repo(
        tmp_path / "fused_bad_tokenizer", q_bytes=_F4_TWO * 64, chat_template="{{ y }}"
    )
    adapter = _write_adapter(tmp_path / "adapter")
    return _Repos(
        base=str(base),
        adapter=str(adapter),
        fused_diff=str(fused_diff),
        fused_same=str(fused_same),
        fused_bad_tokenizer=str(fused_bad_tokenizer),
    )


# --- fake worker backend --------------------------------------------------------

# argmax vectors match the default fixture's length (7).
_BASE_AM = [1, 1, 1, 1, 1, 1, 1]
_ADAPTER_AM = [2, 2, 2, 1, 1, 1, 1]  # differs from base at positions 0-2 (agree 4/7)


def _ok(role: str, argmax: list[int], *, adapter_applied: bool | None = None) -> WorkerOutcome:
    return WorkerOutcome(
        role=role,
        argmax=argmax,
        adapter_applied=adapter_applied,
        peak_bytes=100,
        status="ok",
        error=None,
    )


def _error(role: str) -> WorkerOutcome:
    return WorkerOutcome(
        role=role, argmax=None, adapter_applied=None, peak_bytes=None, status="error", error="boom"
    )


class FakeLauncher:
    """Fake ``Launcher`` returning a pinned outcome per role, recording model paths."""

    def __init__(self, by_role: dict[str, WorkerOutcome]) -> None:
        self._by_role = by_role
        self.model_paths_by_role: dict[str, str] = {}

    def run_one(self, spec: WorkerSpec) -> WorkerOutcome:
        self.model_paths_by_role[spec.role] = spec.model_path
        return self._by_role[spec.role]


def _pass_launcher() -> FakeLauncher:
    # fused tracks the base+adapter reference exactly.
    return FakeLauncher(
        {
            ROLE_BASE: _ok(ROLE_BASE, _BASE_AM),
            ROLE_NOISE: _ok(ROLE_NOISE, _BASE_AM),
            ROLE_REFERENCE: _ok(ROLE_REFERENCE, _ADAPTER_AM, adapter_applied=True),
            ROLE_FUSED: _ok(ROLE_FUSED, _ADAPTER_AM),
        }
    )


def _tracks_base_launcher() -> FakeLauncher:
    # adapter IS applied on the reference, but fused reproduces base-alone.
    return FakeLauncher(
        {
            ROLE_BASE: _ok(ROLE_BASE, _BASE_AM),
            ROLE_NOISE: _ok(ROLE_NOISE, _BASE_AM),
            ROLE_REFERENCE: _ok(ROLE_REFERENCE, _ADAPTER_AM, adapter_applied=True),
            ROLE_FUSED: _ok(ROLE_FUSED, _BASE_AM),
        }
    )


def _incomplete_reference_launcher() -> FakeLauncher:
    # the reference load could not validate the adapter -> adapter_applied is None.
    return FakeLauncher(
        {
            ROLE_BASE: _ok(ROLE_BASE, _BASE_AM),
            ROLE_NOISE: _ok(ROLE_NOISE, _BASE_AM),
            ROLE_REFERENCE: _ok(ROLE_REFERENCE, _BASE_AM, adapter_applied=None),
            ROLE_FUSED: _ok(ROLE_FUSED, _BASE_AM),
        }
    )


def _no_effect_reference_launcher() -> FakeLauncher:
    # the reference load applied a LoRA module with no effective contribution.
    return FakeLauncher(
        {
            ROLE_BASE: _ok(ROLE_BASE, _BASE_AM),
            ROLE_NOISE: _ok(ROLE_NOISE, _BASE_AM),
            ROLE_REFERENCE: _ok(ROLE_REFERENCE, _BASE_AM, adapter_applied=False),
            ROLE_FUSED: _ok(ROLE_FUSED, _BASE_AM),
        }
    )


class RaisingLauncher:
    """Launcher whose ``run_one`` must never be called (proves a gate skipped the workers)."""

    def __init__(self) -> None:
        self.model_paths_by_role: dict[str, str] = {}

    def run_one(self, spec: WorkerSpec) -> WorkerOutcome:
        raise AssertionError(
            f"run_one must not be called when the run is gated: role={spec.role!r}"
        )


def _fused_crash_launcher() -> FakeLauncher:
    return FakeLauncher(
        {
            ROLE_BASE: _ok(ROLE_BASE, _BASE_AM),
            ROLE_NOISE: _ok(ROLE_NOISE, _BASE_AM),
            ROLE_REFERENCE: _ok(ROLE_REFERENCE, _ADAPTER_AM, adapter_applied=True),
            ROLE_FUSED: _error(ROLE_FUSED),
        }
    )


def _opts(launcher: FakeLauncher) -> ParityOptions:
    return ParityOptions(launcher=launcher)


# --- tests ----------------------------------------------------------------------


def test_pass_yields_pass_verdict_and_exit_0(tiny_local_repos: _Repos) -> None:
    report = check_adapter_parity(
        base=tiny_local_repos.base,
        adapter=tiny_local_repos.adapter,
        fused=tiny_local_repos.fused_diff,
        options=_opts(_pass_launcher()),
    )
    assert report.adapter_applied is True
    assert report.verdict is ParityVerdict.PASS
    assert parity_exit_code(report) == 0


def test_copied_base_fuse_is_tracks_base_not_voided_exit_1(tiny_local_repos: _Repos) -> None:
    # F1: the adapter is genuinely applied on the reference, so the copied-base
    # fuse is a determined regression (FAIL_TRACKS_BASE), never a void.
    report = check_adapter_parity(
        base=tiny_local_repos.base,
        adapter=tiny_local_repos.adapter,
        fused=tiny_local_repos.fused_same,
        options=_opts(_tracks_base_launcher()),
    )
    assert report.adapter_applied is True
    assert report.verdict is ParityVerdict.FAIL_TRACKS_BASE
    assert parity_exit_code(report) == 1


def test_negative_names_reverted_tensor_and_first_divergence(tiny_local_repos: _Repos) -> None:
    # The copied-base fused repo leaves q_proj byte-identical to base, so the delta
    # map marks it `unchanged` AND the oracle's first_divergence is set -- both in
    # one report, from one run.
    report = check_adapter_parity(
        base=tiny_local_repos.base,
        adapter=tiny_local_repos.adapter,
        fused=tiny_local_repos.fused_same,
        options=_opts(_tracks_base_launcher()),
    )
    assert any(
        delta.tensor.endswith("self_attn.q_proj.weight") and delta.klass == "unchanged"
        for delta in report.delta_map
    )
    assert report.first_divergence is not None
    assert parity_exit_code(report) == 1


def test_incomplete_adapter_reference_is_indeterminate_exit_2(tiny_local_repos: _Repos) -> None:
    # F3: the reference load could not validate the adapter -> adapter_applied None,
    # verdict null, cannot-determine (exit 2).
    report = check_adapter_parity(
        base=tiny_local_repos.base,
        adapter=tiny_local_repos.adapter,
        fused=tiny_local_repos.fused_diff,
        options=_opts(_incomplete_reference_launcher()),
    )
    assert report.adapter_applied is None
    assert report.verdict is None
    assert parity_exit_code(report) == 2


def test_no_effect_adapter_reference_is_indeterminate_exit_2(tiny_local_repos: _Repos) -> None:
    # A reference whose adapter has no effective contribution has no valid
    # reference to score against: adapter_applied False, verdict null, exit 2.
    report = check_adapter_parity(
        base=tiny_local_repos.base,
        adapter=tiny_local_repos.adapter,
        fused=tiny_local_repos.fused_diff,
        options=_opts(_no_effect_reference_launcher()),
    )
    assert report.adapter_applied is False
    assert report.verdict is None
    assert parity_exit_code(report) == 2


def test_tokenizer_mismatch_gates_the_oracle_without_running_workers(
    tiny_local_repos: _Repos,
) -> None:
    # A changed chat template is a real incompatibility (F6): the tokenizer check
    # fails, the oracle is void-skipped (verdict null), and no worker is launched.
    launcher = RaisingLauncher()
    report = check_adapter_parity(
        base=tiny_local_repos.base,
        adapter=tiny_local_repos.adapter,
        fused=tiny_local_repos.fused_bad_tokenizer,
        options=ParityOptions(launcher=launcher),
    )
    assert any(
        result.check_id == "parity/tokenizer.identity" and result.status == "fail"
        for result in report.results
    )
    assert report.verdict is None
    assert launcher.model_paths_by_role == {}
    assert set(report.worker_status.values()) == {"skipped"}


def test_worker_crash_returns_report_with_error_status_exit_2(tiny_local_repos: _Repos) -> None:
    report = check_adapter_parity(
        base=tiny_local_repos.base,
        adapter=tiny_local_repos.adapter,
        fused=tiny_local_repos.fused_diff,
        options=_opts(_fused_crash_launcher()),
    )
    assert report.worker_status[ROLE_FUSED] == "error"
    assert report.verdict is None
    assert parity_exit_code(report) == 2


def test_all_base_derived_workers_share_the_one_pinned_base_path(
    tiny_local_repos: _Repos,
) -> None:
    # F10: base is resolved once; the base/base-repeat/reference loads all use that
    # same pinned path, and only the fused load uses the fused path.
    launcher = _pass_launcher()
    check_adapter_parity(
        base=tiny_local_repos.base,
        adapter=tiny_local_repos.adapter,
        fused=tiny_local_repos.fused_diff,
        options=_opts(launcher),
    )
    base_resolved = str(Path(tiny_local_repos.base).resolve())
    assert launcher.model_paths_by_role[ROLE_BASE] == base_resolved
    assert launcher.model_paths_by_role[ROLE_NOISE] == base_resolved
    assert launcher.model_paths_by_role[ROLE_REFERENCE] == base_resolved
    assert launcher.model_paths_by_role[ROLE_FUSED] == str(
        Path(tiny_local_repos.fused_diff).resolve()
    )
