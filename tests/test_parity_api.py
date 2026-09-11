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
from mlx_model_doctor.parity.context import tokenizer_fingerprint_for_path
from mlx_model_doctor.parity.fixtures import (
    DEFAULT_FIXTURE_ID,
    FixtureRef,
    compute_input_digest,
    get_fixture,
    reference_tokenizer_files,
)
from mlx_model_doctor.parity.oracle import ROLE_BASE, ROLE_FUSED, ROLE_NOISE, ROLE_REFERENCE
from mlx_model_doctor.parity.orchestrator import WorkerOutcome
from mlx_model_doctor.parity.worker import WorkerSpec
from tests.parity_fakes import lora_tensor, safetensors_header_bytes, write_safetensors_file

_F4 = struct.pack("<f", 1.0)
_F4_TWO = struct.pack("<f", 2.0)

# The default fixture's own reference chat template -- used as the DEFAULT chat
# template for every synthetic repo below, so a repo's own tokenizer fingerprint
# matches DEFAULT_FIXTURE_ID's by default (the new fixture-vs-base gate, F6/F9)
# and only an EXPLICIT override (see `fused_bad_tokenizer`) produces a mismatch.
_, _, _REFERENCE_CHAT_TEMPLATE = reference_tokenizer_files()


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


def _write_model_repo(
    root: Path, *, q_bytes: bytes, chat_template: str = _REFERENCE_CHAT_TEMPLATE
) -> Path:
    """Write a synthetic model repo whose tokenizer is the default fixture's OWN
    reference tokenizer (see ``reference_tokenizer_files``), so its base tokenizer
    fingerprint matches ``DEFAULT_FIXTURE_ID``'s and the fixture-vs-base gate never
    fires by accident -- only an explicit ``chat_template`` override (a real
    base-vs-fused incompatibility, see ``fused_bad_tokenizer``) produces a mismatch.
    """
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
    tokenizer_config, tokenizer_json, _ = reference_tokenizer_files()
    tokenizer_config["chat_template"] = chat_template
    (root / "tokenizer.json").write_text(json.dumps(tokenizer_json), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    write_safetensors_file(root / "model.safetensors", _model_tensors(q_bytes))
    return root


def _write_model_repo_with_mismatched_vocab(root: Path, *, q_bytes: bytes) -> Path:
    """Write a repo whose tokenizer VOCABULARY differs from the reference tokenizer.

    Base uses the reference tokenizer's full vocabulary; this repo uses a small,
    unrelated vocabulary instead -- a genuine incompatibility, distinct from
    ``fused_bad_tokenizer``'s metadata-only (chat-template) difference, which
    must still gate the oracle even under the metadata-tolerant tokenizer check
    (F6): the fixed-id oracle's token ids would not mean the same thing under a
    different token<->id map.
    """
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
        json.dumps(
            {"model": {"type": "BPE", "vocab": {f"different_tok_{i}": i for i in range(8)}}}
        ),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "different_tok_0",
                "eos_token": "different_tok_1",
                "chat_template": _REFERENCE_CHAT_TEMPLATE,
            }
        ),
        encoding="utf-8",
    )
    write_safetensors_file(root / "model.safetensors", _model_tensors(q_bytes))
    return root


def _write_adapter(root: Path, *, keys: tuple[str, ...] = ("self_attn.q_proj",)) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "num_layers": 1,
        "fine_tune_type": "lora",
        "lora_parameters": {"rank": 4, "scale": 20.0, "keys": list(keys)},
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
    adapter_bad_key: str  # names a LoRA target key that maps to no base tensor
    fused_diff: str  # q_proj differs from base (a genuine fuse)
    fused_same: str  # q_proj byte-identical to base (a copied-base fuse)
    fused_bad_tokenizer: str  # a changed chat template only (vocab unchanged: metadata, not a gate)
    fused_vocab_mismatch: str  # a different tokenizer vocabulary (a real tokenizer incompatibility)


@pytest.fixture
def tiny_local_repos(tmp_path: Path) -> _Repos:
    base = _write_model_repo(tmp_path / "base", q_bytes=_F4 * 64)
    fused_diff = _write_model_repo(tmp_path / "fused_diff", q_bytes=_F4_TWO * 64)
    fused_same = _write_model_repo(tmp_path / "fused_same", q_bytes=_F4 * 64)
    fused_bad_tokenizer = _write_model_repo(
        tmp_path / "fused_bad_tokenizer", q_bytes=_F4_TWO * 64, chat_template="{{ y }}"
    )
    fused_vocab_mismatch = _write_model_repo_with_mismatched_vocab(
        tmp_path / "fused_vocab_mismatch", q_bytes=_F4_TWO * 64
    )
    adapter = _write_adapter(tmp_path / "adapter")
    adapter_bad_key = _write_adapter(tmp_path / "adapter_bad_key", keys=("self_attn.nonexistent",))
    return _Repos(
        base=str(base),
        adapter=str(adapter),
        adapter_bad_key=str(adapter_bad_key),
        fused_diff=str(fused_diff),
        fused_same=str(fused_same),
        fused_bad_tokenizer=str(fused_bad_tokenizer),
        fused_vocab_mismatch=str(fused_vocab_mismatch),
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
        self.specs_by_role: dict[str, WorkerSpec] = {}

    def run_one(self, spec: WorkerSpec) -> WorkerOutcome:
        self.model_paths_by_role[spec.role] = spec.model_path
        self.specs_by_role[spec.role] = spec
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


def _inconclusive_launcher() -> FakeLauncher:
    # Engineered to a real INCONCLUSIVE verdict under the production constants
    # (PARITY_K=0.3, PARITY_PASS_FLOOR=0.9, PARITY_GROSS_FLOOR=0.5): base_repeat
    # equals base exactly (noise=0), and fused lands equidistant from both base
    # and the base+adapter reference (agree_fa == agree_fb == 4/7), so neither a
    # PASS nor a FAIL_TRACKS_BASE preference clears the noise floor -- an honest
    # "the fuse partially degraded the adapter" case, not a crash or a gate.
    base_am = [0, 0, 0, 0, 0, 0, 0]
    reference_am = [1, 1, 0, 0, 0, 0, 0]
    fused_am = [1, 0, 0, 0, 0, 1, 1]
    return FakeLauncher(
        {
            ROLE_BASE: _ok(ROLE_BASE, base_am),
            ROLE_NOISE: _ok(ROLE_NOISE, base_am),
            ROLE_REFERENCE: _ok(ROLE_REFERENCE, reference_am, adapter_applied=True),
            ROLE_FUSED: _ok(ROLE_FUSED, fused_am),
        }
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
    # the local byte-compare ran to completion (delta_map phase tracked, not silent)
    assert report.phase_outcomes["delta_map"] == "ok"


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


def test_tokenizer_metadata_only_mismatch_warns_but_runs_the_oracle(
    tiny_local_repos: _Repos,
) -> None:
    # F6 relaxation: a changed chat template alone (vocab unchanged) is metadata
    # a real `mlx_lm.fuse` output routinely re-serializes -- it does not affect
    # the fixed-id forward pass, so the tokenizer check WARNS instead of failing,
    # and the oracle still runs to a real verdict (workers ARE invoked).
    report = check_adapter_parity(
        base=tiny_local_repos.base,
        adapter=tiny_local_repos.adapter,
        fused=tiny_local_repos.fused_bad_tokenizer,
        options=_opts(_pass_launcher()),
    )
    assert any(
        result.check_id == "parity/tokenizer.identity" and result.status == "warn"
        for result in report.results
    )
    assert not any(
        result.check_id == "parity/tokenizer.identity" and result.status == "fail"
        for result in report.results
    )
    assert report.adapter_applied is True
    assert report.verdict is ParityVerdict.PASS
    assert parity_exit_code(report) == 0


def test_vocab_mismatch_still_gates_the_oracle_without_running_workers(
    tiny_local_repos: _Repos,
) -> None:
    # A genuine tokenizer VOCABULARY difference is still a real incompatibility
    # (F6): the fixed-id oracle's token ids would not mean the same thing under
    # a different token<->id map, so the tokenizer check fails, the oracle is
    # void-skipped (verdict null), no worker is launched, and the run is a
    # determined defect (exit 1), not a mere cannot-determine (2). Distinct from
    # the metadata-only case above, which now warns and runs the oracle.
    launcher = RaisingLauncher()
    report = check_adapter_parity(
        base=tiny_local_repos.base,
        adapter=tiny_local_repos.adapter,
        fused=tiny_local_repos.fused_vocab_mismatch,
        options=ParityOptions(launcher=launcher),
    )
    assert any(
        result.check_id == "parity/tokenizer.identity" and result.status == "fail"
        for result in report.results
    )
    assert report.verdict is None
    assert launcher.model_paths_by_role == {}
    assert set(report.worker_status.values()) == {"skipped"}
    assert parity_exit_code(report) == 1


def test_vocab_unavailable_is_cannot_determine_not_a_determined_mismatch(
    tmp_path: Path,
) -> None:
    # B1b: an UNREADABLE vocabulary (both sides lack tokenizer.json, so neither
    # side's token<->id map can even be compared) means we could not confirm
    # tokenizer identity -- not that we confirmed the two tokenizers differ. This
    # must be a cannot-determine outcome (exit 2), never folded into the
    # determined-bad-static exit-1 path a GENUINE vocab mismatch (the test above)
    # takes. One-line bug this catches: classifying "could not read the vocab" the
    # same way as "read both vocabs and they differ" (a `fail`), which used to flip
    # this to exit 1.
    #
    # The injected fixture's fingerprint is computed from this exact base repo (so
    # it equals base's own fingerprint, all-None fields included) -- this isolates
    # the base-vs-fused vocab-unavailable gate under test from the separate
    # fixture-vs-base equality gate, which must NOT also fire here.
    base = _write_repo_without_tokenizer_json(tmp_path / "base", q_bytes=_F4 * 64)
    fused = _write_repo_without_tokenizer_json(tmp_path / "fused", q_bytes=_F4_TWO * 64)
    adapter = _write_adapter(tmp_path / "adapter")
    fingerprint = tokenizer_fingerprint_for_path(str(base))
    token_ids = ((0, 10, 22, 45, 7, 33, 1),)
    fixture_ref = FixtureRef(
        id="fixture-matches-base-without-tokenizer-json",
        input_digest=compute_input_digest(token_ids),
        max_length=7,
        scored_positions=tuple(range(7)),
        tokenizer_fingerprint=fingerprint,
    )
    launcher = RaisingLauncher()

    report = check_adapter_parity(
        base=str(base),
        adapter=str(adapter),
        fused=str(fused),
        options=ParityOptions(launcher=launcher, fixture=(fixture_ref, token_ids)),
    )

    assert not any(
        result.check_id == "parity/tokenizer.identity" and result.status == "fail"
        for result in report.results
    )
    assert any(
        result.check_id == "parity/tokenizer.identity" and result.status == "warn"
        for result in report.results
    )
    assert report.verdict is None
    assert launcher.model_paths_by_role == {}
    assert set(report.worker_status.values()) == {"skipped"}
    assert parity_exit_code(report) == 2


def _write_repo_with_custom_tokenizer(root: Path, *, q_bytes: bytes) -> Path:
    """Write a repo using a tokenizer that is NOT the default fixture's reference
    tokenizer -- isolates the fixture-vs-base gate: base and fused below share this
    SAME non-reference tokenizer, so the base-vs-fused ``TokenizerIdentityCheck``
    agrees and only the fixture disagrees.
    """
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
        json.dumps({"bos_token": "t0", "eos_token": "t1", "chat_template": "{{ x }}"}),
        encoding="utf-8",
    )
    write_safetensors_file(root / "model.safetensors", _model_tensors(q_bytes))
    return root


def _write_repo_without_tokenizer_json(root: Path, *, q_bytes: bytes) -> Path:
    """Write a synthetic model repo with NO ``tokenizer.json`` -- the real shape of a
    loadable SentencePiece-based "slow" tokenizer repository (only
    ``tokenizer_config.json``'s special-token fields, no fast-tokenizer vocab file).
    Its ``base_tokenizer_fingerprint()`` therefore has ``vocab_size=None`` (and, by
    extension, ``token_id_map_digest=None``) -- the exact shape the fixture-vs-base
    equality gate (F6/F9) must still recognize as a match against its OWN fixture.
    """
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
    (root / "tokenizer_config.json").write_text(
        json.dumps({"bos_token": "<s>", "eos_token": "</s>"}), encoding="utf-8"
    )
    write_safetensors_file(root / "model.safetensors", _model_tensors(q_bytes))
    return root


def _write_model_repo_without_chat_template(root: Path, *, q_bytes: bytes) -> Path:
    """Write a synthetic repo with a REAL, readable tokenizer vocabulary and special
    tokens but NO chat template (unlike ``_write_repo_without_tokenizer_json``, whose
    fingerprint has an unavailable vocabulary too). Its fingerprint therefore has
    every field populated except ``chat_template_digest``, which is ``None``.

    Isolates the fixture-vs-base equality gate from the (separate) base-vs-fused
    ``TokenizerIdentityCheck``: base and fused below share this exact tokenizer, so
    that check only reaches its metadata-only ``chat_template_unavailable`` warn (it
    never gates the oracle) -- only the fixture-vs-base gate under test can void it.
    """
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
    tokenizer_config, tokenizer_json, _ = reference_tokenizer_files()
    del tokenizer_config["chat_template"]
    (root / "tokenizer.json").write_text(json.dumps(tokenizer_json), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    write_safetensors_file(root / "model.safetensors", _model_tensors(q_bytes))
    return root


def test_fixture_equal_to_base_fingerprint_runs_the_oracle_despite_an_unavailable_field(
    tmp_path: Path,
) -> None:
    """C2: the fixture-vs-base gate must compare fingerprints for EQUALITY
    (``fixture_ref.tokenizer_fingerprint != tokenizer_fingerprint``), never by reusing
    ``tokenizers_match`` -- the base-vs-FUSED comparator, which by design treats even
    an identical "both sides unavailable" field (here: no chat template) as a
    non-match (see ``tokenizers_match``'s own docstring). Base and fused below share
    the exact same tokenizer (so the unrelated base-vs-fused check only warns, never
    gates), and the injected fixture's fingerprint is computed from this exact base
    repo -- byte-identical to what ``base_tokenizer_fingerprint()`` itself returns for
    it. The oracle must therefore run, not void-skip on a false-positive fixture
    mismatch.

    One-line bug this catches: swapping the equality gate for
    ``not tokenizers_match(fixture_ref.tokenizer_fingerprint, tokenizer_fingerprint).matched``
    in ``check_adapter_parity`` -- proven by reverting to that form locally and
    confirming this test goes red (see the fix commit's verification notes).
    """
    base = _write_model_repo_without_chat_template(tmp_path / "base", q_bytes=_F4 * 64)
    fused = _write_model_repo_without_chat_template(tmp_path / "fused", q_bytes=_F4_TWO * 64)
    adapter = _write_adapter(tmp_path / "adapter")
    fingerprint = tokenizer_fingerprint_for_path(str(base))
    token_ids = ((0, 10, 22, 45, 7, 33, 1),)
    fixture_ref = FixtureRef(
        id="fixture-matches-base-without-chat-template",
        input_digest=compute_input_digest(token_ids),
        max_length=7,
        scored_positions=tuple(range(7)),
        tokenizer_fingerprint=fingerprint,
    )
    launcher = _pass_launcher()

    report = check_adapter_parity(
        base=str(base),
        adapter=str(adapter),
        fused=str(fused),
        options=ParityOptions(launcher=launcher, fixture=(fixture_ref, token_ids)),
    )

    assert launcher.model_paths_by_role  # the workers DID run
    assert report.phase_outcomes["oracle"] == "ok"
    assert not any(
        "the parity fixture's tokenizer fingerprint does not match the base model's" in reason
        for reason in report.reasons
    )


def test_fixture_tokenizer_mismatch_gates_the_oracle_without_running_workers(
    tmp_path: Path,
) -> None:
    """A fixture built for one tokenizer must never be silently scored against a
    base model with a DIFFERENT one: base and fused agree with each other (so the
    existing base-vs-fused check stays green), but neither matches the default
    fixture's reference tokenizer -- the oracle must void-skip, cleanly, with a
    reason distinct from the base-vs-fused one, and NO worker may run.
    """
    base = _write_repo_with_custom_tokenizer(tmp_path / "base", q_bytes=_F4 * 64)
    fused = _write_repo_with_custom_tokenizer(tmp_path / "fused", q_bytes=_F4_TWO * 64)
    adapter = _write_adapter(tmp_path / "adapter")
    launcher = RaisingLauncher()

    report = check_adapter_parity(
        base=str(base),
        adapter=str(adapter),
        fused=str(fused),
        options=ParityOptions(launcher=launcher),
    )

    assert not any(
        result.check_id == "parity/tokenizer.identity" and result.status == "fail"
        for result in report.results
    )
    assert report.verdict is None
    assert launcher.model_paths_by_role == {}
    assert set(report.worker_status.values()) == {"skipped"}
    assert any(
        "the parity fixture's tokenizer fingerprint does not match the base model's" in reason
        for reason in report.reasons
    )
    # The reason must point at the real CLI flow that builds a matching fixture,
    # not just name the mismatch.
    assert any("parity mlx --prompts" in reason for reason in report.reasons)
    assert parity_exit_code(report) == 2


def test_matching_fixture_tokenizer_does_not_gate_the_oracle(tiny_local_repos: _Repos) -> None:
    """The base repo's tokenizer IS the default fixture's own reference tokenizer
    (see ``_write_model_repo``): the fixture-vs-base gate must let the run through
    -- workers run and the oracle reaches a real verdict, not a false-positive skip.
    """
    launcher = _pass_launcher()

    report = check_adapter_parity(
        base=tiny_local_repos.base,
        adapter=tiny_local_repos.adapter,
        fused=tiny_local_repos.fused_diff,
        options=_opts(launcher),
    )

    assert launcher.model_paths_by_role  # the workers DID run
    assert report.phase_outcomes["oracle"] == "ok"
    assert not any(
        "the parity fixture's tokenizer fingerprint does not match the base model's" in reason
        for reason in report.reasons
    )


def test_uncovered_lora_target_is_determined_bad_exit_1(tiny_local_repos: _Repos) -> None:
    # A LoRA target key that names no base tensor is a statically-confirmed defect:
    # the coverage check fails, the oracle is void-skipped (verdict null), no worker
    # runs, and the exit code is 1 (determined bad), not 2.
    launcher = RaisingLauncher()
    report = check_adapter_parity(
        base=tiny_local_repos.base,
        adapter=tiny_local_repos.adapter_bad_key,
        fused=tiny_local_repos.fused_diff,
        options=ParityOptions(launcher=launcher),
    )
    assert any(
        result.check_id == "parity/target.coverage" and result.status == "fail"
        for result in report.results
    )
    assert report.verdict is None
    assert launcher.model_paths_by_role == {}
    assert parity_exit_code(report) == 1


def test_worker_crash_returns_report_with_error_status_exit_2(tiny_local_repos: _Repos) -> None:
    report = check_adapter_parity(
        base=tiny_local_repos.base,
        adapter=tiny_local_repos.adapter,
        fused=tiny_local_repos.fused_diff,
        options=_opts(_fused_crash_launcher()),
    )
    assert report.worker_status[ROLE_FUSED] == "error"
    assert report.verdict is None
    # F1: only the fused worker crashed; the reference succeeded, so its
    # adapter-applied signal (True here) must survive the crash, not be erased.
    assert report.adapter_applied is True
    assert parity_exit_code(report) == 2


def test_inconclusive_verdict_carries_an_honest_reason(tiny_local_repos: _Repos) -> None:
    # An INCONCLUSIVE verdict (the oracle ran but couldn't call a preference) must
    # not ship silently: it needs a user-facing reason string, and it still exits
    # 2 (cannot-determine), the same as any other unresolved oracle outcome.
    report = check_adapter_parity(
        base=tiny_local_repos.base,
        adapter=tiny_local_repos.adapter,
        fused=tiny_local_repos.fused_diff,
        options=_opts(_inconclusive_launcher()),
    )
    assert report.verdict is ParityVerdict.INCONCLUSIVE
    assert (
        "the fused model's outputs do not clearly track the base+adapter reference; "
        "adapter parity could not be confirmed (the fuse may partially degrade the "
        "adapter)."
    ) in report.reasons
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


def test_parity_options_fixture_overrides_get_fixture(tiny_local_repos: _Repos) -> None:
    """ParityOptions.fixture, when set, replaces get_fixture(options.fixture_id): every
    worker spec must carry the INJECTED fixture's id/token ids, and the report's own
    ``fixture`` field must record it -- not silently fall back to the default fixture.
    """
    default_ref, _ = get_fixture(DEFAULT_FIXTURE_ID)
    custom_token_ids = ((7, 8, 9),)
    custom_ref = FixtureRef(
        id="custom-fixture",
        input_digest=compute_input_digest(custom_token_ids),
        max_length=3,
        scored_positions=(0, 1, 2),
        tokenizer_fingerprint=default_ref.tokenizer_fingerprint,
    )
    launcher = _pass_launcher()

    report = check_adapter_parity(
        base=tiny_local_repos.base,
        adapter=tiny_local_repos.adapter,
        fused=tiny_local_repos.fused_diff,
        options=ParityOptions(launcher=launcher, fixture=(custom_ref, custom_token_ids)),
    )

    for role in (ROLE_BASE, ROLE_NOISE, ROLE_REFERENCE, ROLE_FUSED):
        assert launcher.specs_by_role[role].fixture_id == "custom-fixture"
        assert launcher.specs_by_role[role].token_ids == custom_token_ids
    assert report.fixture.id == "custom-fixture"
