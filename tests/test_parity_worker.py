"""Tests for the memory-safe parity worker (spec/result, adapter validation, JSON IPC)."""

import os
from pathlib import Path

import pytest

import mlx_model_doctor.parity.worker as worker_module
from mlx_model_doctor.errors import MemorySafetyError, ModelDoctorError, WorkerArtifactError
from mlx_model_doctor.parity.oracle import argmax_agreement, first_divergence, flip_count
from mlx_model_doctor.parity.worker import (
    AdapterValidation,
    MlxLmWorkerBackend,
    WorkerResult,
    WorkerSpec,
    main,
    read_worker_json,
    run_worker_body,
    validate_adapter_manifest,
    write_worker_json,
)
from tests.parity_fakes import (
    FakeLoraModule,
    FakeMlxLmModel,
    FakeMlxLmModule,
    FakeMxCore,
    LoadForbiddenMlxLmModule,
    NeverCalledBackend,
    OrderAwareCaps,
    OrderCheckingBackend,
    SequenceLogitsMlxLmModel,
    StubBackend,
    lora_tensor,
    make_import_module,
    write_adapter_dir,
)


def _spec(**overrides: object) -> WorkerSpec:
    defaults: dict[str, object] = {
        "model_path": "/models/base",
        "adapter_path": None,
        "token_ids": ((1, 2, 3),),
        "fixture_id": "fx-1",
        "role": "base",
    }
    defaults.update(overrides)
    return WorkerSpec(**defaults)  # type: ignore[arg-type]


def _result(**overrides: object) -> WorkerResult:
    defaults: dict[str, object] = {
        "argmax": [1, 2, 3],
        "adapter_applied": None,
        "peak_bytes": 4096,
        "role": "base",
        "fixture_id": "fx-1",
    }
    defaults.update(overrides)
    return WorkerResult(**defaults)  # type: ignore[arg-type]


# --- WorkerSpec / WorkerResult -------------------------------------------------


def test_worker_spec_holds_the_expected_fields() -> None:
    spec = _spec(adapter_path="/adapters/a", token_ids=((4, 5),))
    assert spec.model_path == "/models/base"
    assert spec.adapter_path == "/adapters/a"
    assert spec.token_ids == ((4, 5),)
    assert spec.fixture_id == "fx-1"
    assert spec.role == "base"


def test_worker_result_holds_the_expected_fields() -> None:
    result = _result(argmax=[7, 8], adapter_applied=True, peak_bytes=99, role="adapter")
    assert result.argmax == [7, 8]
    assert result.adapter_applied is True
    assert result.peak_bytes == 99
    assert result.role == "adapter"
    assert result.fixture_id == "fx-1"


# --- run_worker_body: order + refusal + length invariant -----------------------


def test_run_worker_body_installs_caps_before_calling_backend() -> None:
    """A mutant that swaps the order (load first, caps second) must fail this test."""
    caps = OrderAwareCaps()
    expected = _result()
    backend = OrderCheckingBackend(caps, expected)

    result = run_worker_body(_spec(), backend, caps_fn=caps.install)

    assert result is expected
    assert caps.installed_before_load is True


def test_run_worker_body_refuses_when_caps_are_uninstalled() -> None:
    backend = NeverCalledBackend()

    with pytest.raises(MemorySafetyError, match="uncapped"):
        run_worker_body(_spec(), backend, caps_fn=lambda: (0, 0))


@pytest.mark.parametrize("caps", [(0, 22), (20, 0), (-1, -1)])
def test_run_worker_body_refuses_for_any_non_positive_cap(caps: tuple[int, int]) -> None:
    backend = NeverCalledBackend()

    with pytest.raises(MemorySafetyError):
        run_worker_body(_spec(), backend, caps_fn=lambda: caps)


def test_run_worker_body_rejects_argmax_length_mismatch() -> None:
    backend = StubBackend(_result(argmax=[1, 2]))  # spec has 3 token ids (one sequence)

    with pytest.raises(ModelDoctorError, match="3"):
        run_worker_body(_spec(token_ids=((1, 2, 3),)), backend, caps_fn=lambda: (20, 22))


def test_run_worker_body_rejects_argmax_length_mismatch_across_multiple_sequences() -> None:
    """The expected length is the SUM across sequences, not the sequence count.

    A mutant comparing against ``len(spec.token_ids)`` (the sequence count, 2)
    instead of the total token count (3) must fail this test.
    """
    backend = StubBackend(_result(argmax=[1, 2]))  # total token ids across sequences is 3

    with pytest.raises(ModelDoctorError, match="3"):
        run_worker_body(_spec(token_ids=((1, 2), (3,))), backend, caps_fn=lambda: (20, 22))


def test_run_worker_body_returns_backend_result_on_success() -> None:
    expected = _result(argmax=[9, 9, 9])
    backend = StubBackend(expected)

    result = run_worker_body(_spec(), backend, caps_fn=lambda: (20, 22))

    assert result is expected


# --- validate_adapter_manifest (F3) ---------------------------------------------


def test_validate_adapter_manifest_ok_for_complete_lora_target(tmp_path: Path) -> None:
    adapter_dir = write_adapter_dir(
        tmp_path / "adapter",
        fine_tune_type="lora",
        tensors={
            "target.lora_a": lora_tensor((32, 8)),
            "target.lora_b": lora_tensor((8, 32)),
        },
    )

    validation = validate_adapter_manifest(adapter_dir, ["target"])

    assert validation == AdapterValidation(status="ok")


def test_validate_adapter_manifest_missing_a(tmp_path: Path) -> None:
    adapter_dir = write_adapter_dir(
        tmp_path / "adapter",
        tensors={"target.lora_b": lora_tensor((8, 32))},
    )

    validation = validate_adapter_manifest(adapter_dir, ["target"])

    assert validation.status == "incomplete"
    assert validation.missing == ["target"]
    assert validation.reason is not None
    assert "lora_a" in validation.reason


def test_validate_adapter_manifest_missing_b(tmp_path: Path) -> None:
    adapter_dir = write_adapter_dir(
        tmp_path / "adapter",
        tensors={"target.lora_a": lora_tensor((32, 8))},
    )

    validation = validate_adapter_manifest(adapter_dir, ["target"])

    assert validation.status == "incomplete"
    assert validation.missing == ["target"]
    assert validation.reason is not None
    assert "lora_b" in validation.reason


def test_validate_adapter_manifest_partial_target_across_multiple_targets(tmp_path: Path) -> None:
    adapter_dir = write_adapter_dir(
        tmp_path / "adapter",
        tensors={
            "good.lora_a": lora_tensor((32, 8)),
            "good.lora_b": lora_tensor((8, 32)),
            "bad.lora_a": lora_tensor((32, 8)),
            # "bad.lora_b" intentionally absent
        },
    )

    validation = validate_adapter_manifest(adapter_dir, ["good", "bad"])

    assert validation.status == "incomplete"
    assert validation.missing == ["bad"]


def test_validate_adapter_manifest_zero_scale_is_incomplete(tmp_path: Path) -> None:
    adapter_dir = write_adapter_dir(
        tmp_path / "adapter",
        scale=0.0,
        tensors={
            "target.lora_a": lora_tensor((32, 8)),
            "target.lora_b": lora_tensor((8, 32)),
        },
    )

    validation = validate_adapter_manifest(adapter_dir, ["target"])

    assert validation.status == "incomplete"
    assert validation.missing == ["target"]
    assert validation.reason is not None
    assert "scale" in validation.reason.lower()


def test_validate_adapter_manifest_supported_dora_requires_magnitude(tmp_path: Path) -> None:
    adapter_dir = write_adapter_dir(
        tmp_path / "adapter",
        fine_tune_type="dora",
        tensors={
            "target.lora_a": lora_tensor((32, 8)),
            "target.lora_b": lora_tensor((8, 32)),
            "target.m": lora_tensor((32,)),
        },
    )

    validation = validate_adapter_manifest(adapter_dir, ["target"])

    assert validation == AdapterValidation(status="ok")


def test_validate_adapter_manifest_dora_missing_magnitude_is_incomplete(tmp_path: Path) -> None:
    adapter_dir = write_adapter_dir(
        tmp_path / "adapter",
        fine_tune_type="dora",
        tensors={
            "target.lora_a": lora_tensor((32, 8)),
            "target.lora_b": lora_tensor((8, 32)),
        },
    )

    validation = validate_adapter_manifest(adapter_dir, ["target"])

    assert validation.status == "incomplete"
    assert validation.missing == ["target"]


def test_validate_adapter_manifest_dora_wrong_shape_magnitude_is_incomplete(
    tmp_path: Path,
) -> None:
    """A magnitude tensor that matches neither DoRA convention must be rejected.

    lora_a is (64, 8) [input_dims=64] and lora_b is (8, 16) [output_dims=16], so
    the only two legitimate magnitude dims are 64 (DoRAEmbedding convention,
    lora_a.shape[0]) or 16 (DoRALinear convention, lora_b.shape[-1]). A magnitude
    of shape (99,) matches neither.
    """
    adapter_dir = write_adapter_dir(
        tmp_path / "adapter",
        fine_tune_type="dora",
        tensors={
            "target.lora_a": lora_tensor((64, 8)),
            "target.lora_b": lora_tensor((8, 16)),
            "target.m": lora_tensor((99,)),
        },
    )

    validation = validate_adapter_manifest(adapter_dir, ["target"])

    assert validation.status == "incomplete"
    assert validation.missing == ["target"]
    assert validation.reason is not None
    assert "shape" in validation.reason.lower()


def test_validate_adapter_manifest_dora_accepts_embedding_magnitude_convention(
    tmp_path: Path,
) -> None:
    """A magnitude matching lora_a.shape[0] (the DoRAEmbedding convention) is valid."""
    adapter_dir = write_adapter_dir(
        tmp_path / "adapter",
        fine_tune_type="dora",
        tensors={
            "target.lora_a": lora_tensor((64, 8)),
            "target.lora_b": lora_tensor((8, 16)),
            "target.m": lora_tensor((64,)),  # matches lora_a.shape[0], not lora_b.shape[-1]
        },
    )

    validation = validate_adapter_manifest(adapter_dir, ["target"])

    assert validation == AdapterValidation(status="ok")


def test_validate_adapter_manifest_dora_rejects_rank2_magnitude(tmp_path: Path) -> None:
    """A magnitude tensor must be rank-1, even if a dimension coincidentally matches."""
    adapter_dir = write_adapter_dir(
        tmp_path / "adapter",
        fine_tune_type="dora",
        tensors={
            "target.lora_a": lora_tensor((32, 8)),
            "target.lora_b": lora_tensor((8, 32)),
            "target.m": lora_tensor((32, 1)),  # rank-2, not rank-1
        },
    )

    validation = validate_adapter_manifest(adapter_dir, ["target"])

    assert validation.status == "incomplete"
    assert validation.missing == ["target"]


def test_validate_adapter_manifest_unsupported_type(tmp_path: Path) -> None:
    adapter_dir = write_adapter_dir(
        tmp_path / "adapter",
        fine_tune_type="full",
        tensors={},
    )

    validation = validate_adapter_manifest(adapter_dir, ["target"])

    assert validation.status == "unsupported"
    assert validation.missing == []
    assert validation.reason is not None
    assert "full" in validation.reason


def test_validate_adapter_manifest_rejects_mismatched_rank_dims(tmp_path: Path) -> None:
    adapter_dir = write_adapter_dir(
        tmp_path / "adapter",
        tensors={
            "target.lora_a": lora_tensor((32, 8)),
            "target.lora_b": lora_tensor((4, 32)),  # rank 4 != rank 8
        },
    )

    validation = validate_adapter_manifest(adapter_dir, ["target"])

    assert validation.status == "incomplete"
    assert validation.missing == ["target"]


def test_validate_adapter_manifest_missing_config_file(tmp_path: Path) -> None:
    adapter_dir = write_adapter_dir(tmp_path / "adapter", omit_config=True)

    validation = validate_adapter_manifest(adapter_dir, ["target"])

    assert validation.status == "incomplete"
    assert validation.reason is not None


def test_validate_adapter_manifest_missing_weights_file(tmp_path: Path) -> None:
    adapter_dir = write_adapter_dir(tmp_path / "adapter", omit_weights=True)

    validation = validate_adapter_manifest(adapter_dir, ["target"])

    assert validation.status == "incomplete"
    assert validation.reason is not None


# --- JSON IPC (F9) ---------------------------------------------------------------


def test_write_then_read_worker_json_round_trips(tmp_path: Path) -> None:
    result = _result(argmax=[1, 2, 0], adapter_applied=True, peak_bytes=555, role="fused")
    path = tmp_path / "result.json"

    write_worker_json(path, result)
    read_back = read_worker_json(
        path, expect_fixture="fx-1", expect_role="fused", vocab_size=10, length=3
    )

    assert read_back == result


def test_read_worker_json_rejects_length_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    write_worker_json(path, _result(argmax=[1, 2, 3]))

    with pytest.raises(WorkerArtifactError):
        read_worker_json(path, expect_fixture="fx-1", expect_role="base", vocab_size=10, length=4)


def test_read_worker_json_rejects_out_of_vocab_ids(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    write_worker_json(path, _result(argmax=[1, 2, 99]))

    with pytest.raises(WorkerArtifactError):
        read_worker_json(path, expect_fixture="fx-1", expect_role="base", vocab_size=10, length=3)


def test_read_worker_json_rejects_negative_ids(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    write_worker_json(path, _result(argmax=[1, -1, 3]))

    with pytest.raises(WorkerArtifactError):
        read_worker_json(path, expect_fixture="fx-1", expect_role="base", vocab_size=10, length=3)


def test_read_worker_json_rejects_wrong_role(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    write_worker_json(path, _result(role="base"))

    with pytest.raises(WorkerArtifactError):
        read_worker_json(
            path, expect_fixture="fx-1", expect_role="adapter", vocab_size=10, length=3
        )


def test_read_worker_json_rejects_wrong_fixture(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    write_worker_json(path, _result(fixture_id="fx-1"))

    with pytest.raises(WorkerArtifactError):
        read_worker_json(
            path, expect_fixture="fx-other", expect_role="base", vocab_size=10, length=3
        )


def test_read_worker_json_rejects_empty_argmax_when_length_expected(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    write_worker_json(path, _result(argmax=[]))

    with pytest.raises(WorkerArtifactError):
        read_worker_json(path, expect_fixture="fx-1", expect_role="base", vocab_size=10, length=3)


def test_read_worker_json_rejects_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "does-not-exist.json"

    with pytest.raises(WorkerArtifactError):
        read_worker_json(path, expect_fixture="fx-1", expect_role="base", vocab_size=10, length=3)


def test_read_worker_json_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(WorkerArtifactError):
        read_worker_json(path, expect_fixture="fx-1", expect_role="base", vocab_size=10, length=3)


def test_read_worker_json_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(WorkerArtifactError):
        read_worker_json(path, expect_fixture="fx-1", expect_role="base", vocab_size=10, length=3)


def test_read_worker_json_rejects_non_integer_argmax_entries(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        '{"fixture_id": "fx-1", "role": "base", "argmax": [1, "2", 3], '
        '"peak_bytes": 10, "adapter_applied": null}',
        encoding="utf-8",
    )

    with pytest.raises(WorkerArtifactError):
        read_worker_json(path, expect_fixture="fx-1", expect_role="base", vocab_size=10, length=3)


def test_read_worker_json_rejects_nested_rank(tmp_path: Path) -> None:
    """argmax must be a flat vector (rank 1), not a nested rank-2 list."""
    path = tmp_path / "result.json"
    path.write_text(
        '{"fixture_id": "fx-1", "role": "base", "argmax": [[1, 2], [3, 4]], '
        '"peak_bytes": 10, "adapter_applied": null}',
        encoding="utf-8",
    )

    with pytest.raises(WorkerArtifactError):
        read_worker_json(path, expect_fixture="fx-1", expect_role="base", vocab_size=10, length=2)


def test_read_worker_json_rejects_missing_required_key(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        '{"fixture_id": "fx-1", "role": "base", "argmax": [1, 2, 3], "peak_bytes": 10}',
        encoding="utf-8",
    )

    with pytest.raises(WorkerArtifactError):
        read_worker_json(path, expect_fixture="fx-1", expect_role="base", vocab_size=10, length=3)


# --- MlxLmWorkerBackend (real backend, offline via faked mlx/mlx_lm) -----------


def test_mlx_lm_worker_backend_computes_argmax_without_adapter(monkeypatch) -> None:
    logits = [[0.1, 5.0, 0.2], [3.0, 0.1, 0.2]]
    model = FakeMlxLmModel(logits)
    mlx_lm = FakeMlxLmModule(model)
    mx = FakeMxCore(logits=logits, peak_memory=777)

    monkeypatch.setattr(
        worker_module.importlib,
        "import_module",
        make_import_module({"mlx.core": mx, "mlx_lm": mlx_lm}),
    )

    result = MlxLmWorkerBackend().load_argmax(
        _spec(token_ids=((1, 2),), adapter_path=None, role="base")
    )

    assert result.argmax == [1, 0]  # argmax of each row
    assert result.adapter_applied is None
    assert result.peak_bytes == 777
    assert result.role == "base"
    assert result.fixture_id == "fx-1"
    assert mlx_lm.load_calls == [
        {
            "path": "/models/base",
            "adapter_path": None,
            "tokenizer_config": {"trust_remote_code": False},
        }
    ]
    assert mx.eval_calls == 1


def test_mlx_lm_worker_backend_pins_tokenizer_trust_remote_code_false(monkeypatch) -> None:
    """The parity worker's tokenizer load must never enable ``trust_remote_code``.

    Catches: the parity worker loading an untrusted repo's tokenizer with remote
    code execution enabled (``mlx_lm.load`` forwards ``tokenizer_config`` to
    ``transformers.AutoTokenizer.from_pretrained(**tokenizer_config)``, so an
    omitted or ``True`` pin here reaches the tokenizer resolution).
    """
    logits = [[0.1, 5.0, 0.2]]
    model = FakeMlxLmModel(logits)
    mlx_lm = FakeMlxLmModule(model)
    mx = FakeMxCore(logits=logits)

    monkeypatch.setattr(
        worker_module.importlib,
        "import_module",
        make_import_module({"mlx.core": mx, "mlx_lm": mlx_lm}),
    )

    MlxLmWorkerBackend().load_argmax(_spec(token_ids=((1,),)))

    assert len(mlx_lm.load_calls) == 1
    assert mlx_lm.load_calls[0]["tokenizer_config"] == {"trust_remote_code": False}


def test_mlx_lm_worker_backend_concatenates_per_sequence_argmax_in_order(monkeypatch) -> None:
    """Two sequences (3 + 2 tokens): the flat argmax must be the per-sequence argmax
    vectors concatenated IN ORDER over a single model load, so scored positions that
    address the second sequence land on the right values.

    Catches: only the first sequence scored, a sequence silently dropped, or the
    concatenation offset/ordered wrong.
    """
    # sequence 1 (3 positions) -> argmax [2, 0, 1]; sequence 2 (2 positions) -> argmax [0, 1]
    logits_per_call = [
        [[0.1, 0.2, 5.0], [4.0, 0.1, 0.2], [0.1, 3.0, 0.2]],
        [[6.0, 0.1], [0.1, 2.0]],
    ]
    model = SequenceLogitsMlxLmModel(logits_per_call)
    mlx_lm = FakeMlxLmModule(model)
    mx = FakeMxCore()

    monkeypatch.setattr(
        worker_module.importlib,
        "import_module",
        make_import_module({"mlx.core": mx, "mlx_lm": mlx_lm}),
    )

    result = MlxLmWorkerBackend().load_argmax(
        _spec(token_ids=((1, 2, 3), (4, 5)), adapter_path=None, role="base")
    )

    assert result.argmax == [2, 0, 1, 0, 1]
    assert len(model.calls) == 2  # one model load, one forward per sequence
    # Each forward must receive its OWN sequence's ids, in order -- not e.g. the
    # first sequence forwarded twice, which would still pass the argmax/call-count
    # assertions above by coincidence of this fixture's logits.
    assert model.calls == [[[1, 2, 3]], [[4, 5]]]
    assert mlx_lm.load_calls == [
        {
            "path": "/models/base",
            "adapter_path": None,
            "tokenizer_config": {"trust_remote_code": False},
        }
    ]  # loaded ONCE

    # Feed the concatenation into the oracle's scored-position reducer, differing
    # only at index 3 (the second sequence's first position), to prove positions
    # addressing the second sequence are correctly aligned in the concatenation.
    other = [2, 0, 1, 9, 1]
    scored = [1, 3, 4]
    assert argmax_agreement(result.argmax, other, scored) == 2 / 3
    assert first_divergence(result.argmax, other, scored) == 3
    assert flip_count(result.argmax, other, scored) == 1


def test_mlx_lm_worker_backend_rejects_non_finite_logits(monkeypatch) -> None:
    logits = [[float("nan"), 5.0, 0.2], [3.0, 0.1, 0.2]]
    model = FakeMlxLmModel(logits)
    mlx_lm = FakeMlxLmModule(model)
    mx = FakeMxCore(logits=logits)

    monkeypatch.setattr(
        worker_module.importlib,
        "import_module",
        make_import_module({"mlx.core": mx, "mlx_lm": mlx_lm}),
    )

    with pytest.raises(ModelDoctorError, match="non-finite"):
        MlxLmWorkerBackend().load_argmax(_spec(token_ids=((1, 2),)))


def test_mlx_lm_worker_backend_determines_adapter_applied_true_for_nonzero_delta(
    monkeypatch, tmp_path: Path
) -> None:
    adapter_dir = write_adapter_dir(
        tmp_path / "adapter",
        tensors={
            "layers.0.target.lora_a": lora_tensor((32, 8)),
            "layers.0.target.lora_b": lora_tensor((8, 32)),
        },
    )
    logits = [[0.1, 5.0, 0.2]]
    lora_module = FakeLoraModule(lora_a=[0.0] * 8, lora_b=[0.01, 0.0, 0.0])
    model = FakeMlxLmModel(logits, modules=[("layers.0.target", lora_module)])
    mlx_lm = FakeMlxLmModule(model)
    mx = FakeMxCore(logits=logits)

    monkeypatch.setattr(
        worker_module.importlib,
        "import_module",
        make_import_module({"mlx.core": mx, "mlx_lm": mlx_lm}),
    )

    result = MlxLmWorkerBackend().load_argmax(
        _spec(token_ids=((1,),), adapter_path=str(adapter_dir), role="adapter")
    )

    assert result.adapter_applied is True


def test_mlx_lm_worker_backend_adapter_applied_false_for_negligible_delta(
    monkeypatch, tmp_path: Path
) -> None:
    adapter_dir = write_adapter_dir(
        tmp_path / "adapter",
        tensors={
            "layers.0.target.lora_a": lora_tensor((32, 8)),
            "layers.0.target.lora_b": lora_tensor((8, 32)),
        },
    )
    logits = [[0.1, 5.0, 0.2]]
    # lora_b all-zero: the LoRA family's untrained init state (no learned effect).
    lora_module = FakeLoraModule(lora_a=[0.0] * 8, lora_b=[0.0, 0.0, 0.0])
    model = FakeMlxLmModel(logits, modules=[("layers.0.target", lora_module)])
    mlx_lm = FakeMlxLmModule(model)
    mx = FakeMxCore(logits=logits)

    monkeypatch.setattr(
        worker_module.importlib,
        "import_module",
        make_import_module({"mlx.core": mx, "mlx_lm": mlx_lm}),
    )

    result = MlxLmWorkerBackend().load_argmax(
        _spec(token_ids=((1,),), adapter_path=str(adapter_dir), role="adapter")
    )

    assert result.adapter_applied is False


def test_mlx_lm_worker_backend_adapter_applied_none_for_unsupported_type(
    monkeypatch, tmp_path: Path
) -> None:
    adapter_dir = write_adapter_dir(
        tmp_path / "adapter",
        fine_tune_type="full",
        tensors={},
    )
    logits = [[0.1, 5.0, 0.2]]
    lora_module = FakeLoraModule(lora_a=[0.0] * 8, lora_b=[0.01, 0.0, 0.0])
    model = FakeMlxLmModel(logits, modules=[("layers.0.target", lora_module)])
    mlx_lm = FakeMlxLmModule(model)
    mx = FakeMxCore(logits=logits)

    monkeypatch.setattr(
        worker_module.importlib,
        "import_module",
        make_import_module({"mlx.core": mx, "mlx_lm": mlx_lm}),
    )

    result = MlxLmWorkerBackend().load_argmax(
        _spec(token_ids=((1,),), adapter_path=str(adapter_dir), role="adapter")
    )

    assert result.adapter_applied is None


def test_mlx_lm_worker_backend_missing_dependencies_raise_install_hint(monkeypatch) -> None:
    def import_module(_name: str) -> object:
        raise ImportError("missing")

    monkeypatch.setattr(worker_module.importlib, "import_module", import_module)

    from mlx_model_doctor.errors import DependencyError

    with pytest.raises(DependencyError, match="Install it with") as exc_info:
        MlxLmWorkerBackend().load_argmax(_spec())
    assert exc_info.value.missing_package == "mlx"


# --- main() (offline, faked mlx/mlx_lm) -----------------------------------------


def test_main_writes_worker_json_and_prints_sentinel(monkeypatch, tmp_path, capsys) -> None:
    logits = [[0.1, 5.0, 0.2], [3.0, 0.1, 0.2]]
    model = FakeMlxLmModel(logits)
    mlx_lm = FakeMlxLmModule(model)
    mx = FakeMxCore(logits=logits, peak_memory=123)

    monkeypatch.setattr(
        worker_module.importlib,
        "import_module",
        make_import_module({"mlx.core": mx, "mlx_lm": mlx_lm}),
    )

    out_path = tmp_path / "out" / "result.json"
    exit_code = main(
        [
            "--model-path",
            "/models/base",
            "--token-ids",
            "1,2",
            "--fixture-id",
            "fx-1",
            "--role",
            "base",
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 0
    read_back = read_worker_json(
        out_path, expect_fixture="fx-1", expect_role="base", vocab_size=10, length=2
    )
    assert read_back.peak_bytes == 123
    captured = capsys.readouterr()
    assert "::PARITY_WORKER::ok role=base fixture=fx-1" in captured.out
    assert mx.cache_limit == 4 * 1024**3


def test_main_refuses_when_caps_cannot_be_installed(monkeypatch, tmp_path) -> None:
    mx = FakeMxCore(cap_failure="set_wired_limit")
    mlx_lm = LoadForbiddenMlxLmModule()

    monkeypatch.setattr(
        worker_module.importlib,
        "import_module",
        make_import_module({"mlx.core": mx, "mlx_lm": mlx_lm}),
    )

    with pytest.raises(MemorySafetyError):
        main(
            [
                "--model-path",
                "/models/base",
                "--token-ids",
                "1,2",
                "--fixture-id",
                "fx-1",
                "--role",
                "base",
                "--out",
                str(tmp_path / "result.json"),
            ]
        )
    assert mlx_lm.load_calls == 0


def test_main_wires_watchdog_abort_to_do_abort(monkeypatch, tmp_path) -> None:
    """A mutant dropping the on_abort wiring (e.g. ``lambda reason: None``) must fail this."""
    logits = [[0.1, 5.0, 0.2]]

    class SlowModel(FakeMlxLmModel):
        def __call__(self, ids: object) -> list[list[list[float]]]:
            import time

            time.sleep(0.2)
            return super().__call__(ids)

    model = SlowModel(logits)
    mlx_lm = FakeMlxLmModule(model)
    # active + cache memory already exceeds any ceiling derived from memory_size.
    mx = FakeMxCore(logits=logits, active_memory=10**12, cache_memory=0)

    monkeypatch.setattr(
        worker_module.importlib,
        "import_module",
        make_import_module({"mlx.core": mx, "mlx_lm": mlx_lm}),
    )

    aborts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        worker_module,
        "_do_abort",
        lambda out_dir, reason: aborts.append((out_dir, reason)),
    )

    out_path = tmp_path / "result.json"
    main(
        [
            "--model-path",
            "/models/base",
            "--token-ids",
            "1",
            "--fixture-id",
            "fx-1",
            "--role",
            "base",
            "--out",
            str(out_path),
            "--wall-deadline-s",
            "5",
            "--poll-s",
            "0.01",
        ]
    )

    assert len(aborts) >= 1
    out_dir, reason = aborts[0]
    assert out_dir == str(out_path.parent)
    assert "memory" in reason.lower()


def test_main_creates_output_directory_before_watchdog_can_abort(monkeypatch, tmp_path) -> None:
    """The output directory must exist before the watchdog can fire.

    ``_do_abort``'s marker write is wrapped in a bare ``except Exception: pass``
    (by design, F2 — a failing marker write must never block termination), so a
    missing directory silently swallows the marker instead of raising. Exercise
    the REAL ``_do_abort`` (only ``os._exit`` is mocked) so this test actually
    proves the directory got created early enough, not just that some abort
    callback ran.
    """
    logits = [[0.1, 5.0, 0.2]]

    class SlowModel(FakeMlxLmModel):
        def __call__(self, ids: object) -> list[list[list[float]]]:
            import time

            time.sleep(0.2)
            return super().__call__(ids)

    model = SlowModel(logits)
    mlx_lm = FakeMlxLmModule(model)
    mx = FakeMxCore(logits=logits, active_memory=10**12, cache_memory=0)

    monkeypatch.setattr(
        worker_module.importlib,
        "import_module",
        make_import_module({"mlx.core": mx, "mlx_lm": mlx_lm}),
    )
    exit_calls: list[int] = []
    monkeypatch.setattr(os, "_exit", exit_calls.append)

    out_path = tmp_path / "nested" / "does" / "not" / "exist" / "result.json"
    assert not out_path.parent.exists()

    main(
        [
            "--model-path",
            "/models/base",
            "--token-ids",
            "1",
            "--fixture-id",
            "fx-1",
            "--role",
            "base",
            "--out",
            str(out_path),
            "--wall-deadline-s",
            "5",
            "--poll-s",
            "0.01",
        ]
    )

    assert exit_calls, "the watchdog never fired; the test setup is broken"
    marker = out_path.parent / "parity_worker_abort.txt"
    assert marker.exists(), "abort marker was silently dropped (output dir missing)"
    assert "memory" in marker.read_text(encoding="utf-8").lower()


# --- Additional edge cases (malformed config, dependency errors, helpers) -------


def test_validate_adapter_manifest_malformed_config_json(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{not json", encoding="utf-8")
    (adapter_dir / "adapters.safetensors").write_bytes(b"")

    validation = validate_adapter_manifest(adapter_dir, ["target"])

    assert validation.status == "incomplete"
    assert validation.reason is not None
    assert "not valid JSON" in validation.reason


def test_validate_adapter_manifest_non_object_config_json(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("[1, 2, 3]", encoding="utf-8")
    (adapter_dir / "adapters.safetensors").write_bytes(b"")

    validation = validate_adapter_manifest(adapter_dir, ["target"])

    assert validation.status == "incomplete"
    assert validation.reason == "adapter_config.json must be a JSON object"


def test_validate_adapter_manifest_non_numeric_scale_is_incomplete(
    tmp_path: Path,
) -> None:
    """A *present but garbled* scale is untrustworthy (F3) — unlike an absent one,
    which falls back to mlx-lm's own default (see the missing-scale-key test)."""
    adapter_dir = write_adapter_dir(
        tmp_path / "adapter",
        config_overrides={"lora_parameters": {"rank": 8, "scale": "not-a-number"}},
        tensors={
            "target.lora_a": lora_tensor((32, 8)),
            "target.lora_b": lora_tensor((8, 32)),
        },
    )

    validation = validate_adapter_manifest(adapter_dir, ["target"])

    assert validation.status == "incomplete"
    assert validation.missing == ["target"]
    assert validation.reason is not None
    assert "scale" in validation.reason.lower()
    assert "not-a-number" in validation.reason


def test_validate_adapter_manifest_missing_scale_key_uses_default(tmp_path: Path) -> None:
    adapter_dir = write_adapter_dir(
        tmp_path / "adapter",
        config_overrides={"lora_parameters": {"rank": 8}},  # no "scale" key at all
        tensors={
            "target.lora_a": lora_tensor((32, 8)),
            "target.lora_b": lora_tensor((8, 32)),
        },
    )

    validation = validate_adapter_manifest(adapter_dir, ["target"])

    assert validation == AdapterValidation(status="ok")


def test_read_local_safetensors_header_rejects_truncated_prefix(tmp_path: Path) -> None:
    path = tmp_path / "adapters.safetensors"
    path.write_bytes(b"\x01\x02")  # fewer than 8 bytes

    with pytest.raises(worker_module.SafetensorsHeaderError, match="truncated"):
        worker_module._read_local_safetensors_header(path)


def test_read_local_safetensors_header_rejects_oversized_header(tmp_path: Path) -> None:
    import struct

    path = tmp_path / "adapters.safetensors"
    path.write_bytes(struct.pack("<Q", worker_module._MAX_HEADER_BYTES + 1))

    with pytest.raises(worker_module.SafetensorsHeaderError, match="too large"):
        worker_module._read_local_safetensors_header(path)


def test_adapter_applied_is_false_when_manifest_is_incomplete(monkeypatch, tmp_path: Path) -> None:
    # Adapter manifest is missing lora_b for the discovered target -> "incomplete".
    adapter_dir = write_adapter_dir(
        tmp_path / "adapter",
        tensors={"target.lora_a": lora_tensor((32, 8))},
    )
    lora_module = FakeLoraModule(lora_a=[0.0] * 8, lora_b=[0.01])
    model = FakeMlxLmModel([[0.1, 5.0]], modules=[("target", lora_module)])

    applied = worker_module._adapter_applied_from_reference(
        model, _spec(adapter_path=str(adapter_dir), token_ids=((1,),))
    )

    assert applied is False


def test_adapter_applied_is_false_when_no_lora_family_modules_are_discovered(
    tmp_path: Path,
) -> None:
    adapter_dir = write_adapter_dir(tmp_path / "adapter", tensors={})
    model = FakeMlxLmModel([[0.1, 5.0]], modules=[])  # no LoRA-family modules at all

    applied = worker_module._adapter_applied_from_reference(
        model, _spec(adapter_path=str(adapter_dir), token_ids=((1,),))
    )

    assert applied is False


def test_has_nonnegligible_delta_skips_targets_missing_from_the_module_tree(
    monkeypatch,
) -> None:
    mx = FakeMxCore()
    monkeypatch.setattr(
        worker_module.importlib, "import_module", make_import_module({"mlx.core": mx})
    )
    model = FakeMlxLmModel([[0.1]], modules=[])  # "phantom" target absent from the tree

    assert worker_module._has_nonnegligible_delta(model, ["phantom"]) is False


def test_has_nonnegligible_delta_skips_modules_without_lora_b(monkeypatch) -> None:
    mx = FakeMxCore()
    monkeypatch.setattr(
        worker_module.importlib, "import_module", make_import_module({"mlx.core": mx})
    )
    # lora_a present but lora_b absent (shouldn't happen post-validation, but the
    # helper must not crash if it does).
    lora_module = FakeLoraModule(lora_a=[0.0])
    model = FakeMlxLmModel([[0.1]], modules=[("target", lora_module)])

    assert worker_module._has_nonnegligible_delta(model, ["target"]) is False


def test_import_mlx_lm_missing_raises_dependency_error(monkeypatch) -> None:
    mx = FakeMxCore()

    monkeypatch.setattr(
        worker_module.importlib,
        "import_module",
        make_import_module({"mlx.core": mx}),  # mlx_lm intentionally absent
    )

    from mlx_model_doctor.errors import DependencyError

    with pytest.raises(DependencyError, match="Install it with") as exc_info:
        worker_module._import_mlx_lm()
    assert exc_info.value.missing_package == "mlx_lm"


def test_cwd_file_names_returns_empty_set_on_oserror(monkeypatch) -> None:
    def _raise_iterdir(self: Path) -> None:
        raise OSError("cannot list directory")

    monkeypatch.setattr(Path, "iterdir", _raise_iterdir)

    assert worker_module._cwd_file_names() == set()


def test_parse_token_id_sequences_rejects_empty_string() -> None:
    """A fixture must have at least one nonempty sequence -- empty input is invalid."""
    with pytest.raises(ValueError, match="empty"):
        worker_module._parse_token_id_sequences("")
    with pytest.raises(ValueError, match="empty"):
        worker_module._parse_token_id_sequences("   ")


def test_parse_token_id_sequences_parses_a_single_sequence() -> None:
    assert worker_module._parse_token_id_sequences("1,2,3") == ((1, 2, 3),)


def test_parse_token_id_sequences_parses_multiple_sequences() -> None:
    assert worker_module._parse_token_id_sequences("1,2,3;4,5") == ((1, 2, 3), (4, 5))


def test_parse_token_id_sequences_rejects_an_empty_sequence_between_separators() -> None:
    with pytest.raises(ValueError, match="empty"):
        worker_module._parse_token_id_sequences("1,2;;3,4")


def test_parse_token_id_sequences_rejects_a_trailing_empty_sequence() -> None:
    with pytest.raises(ValueError, match="empty"):
        worker_module._parse_token_id_sequences("1,2;")


def test_read_worker_json_rejects_non_positive_length(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    write_worker_json(path, _result())

    with pytest.raises(ValueError, match="positive"):
        read_worker_json(path, expect_fixture="fx-1", expect_role="base", vocab_size=10, length=0)


def test_read_worker_json_rejects_non_string_fixture_id(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        '{"fixture_id": 1, "role": "base", "argmax": [1, 2, 3], '
        '"peak_bytes": 10, "adapter_applied": null}',
        encoding="utf-8",
    )

    with pytest.raises(WorkerArtifactError, match="fixture_id"):
        read_worker_json(path, expect_fixture="fx-1", expect_role="base", vocab_size=10, length=3)


def test_read_worker_json_rejects_non_string_role(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        '{"fixture_id": "fx-1", "role": 7, "argmax": [1, 2, 3], '
        '"peak_bytes": 10, "adapter_applied": null}',
        encoding="utf-8",
    )

    with pytest.raises(WorkerArtifactError, match="role"):
        read_worker_json(path, expect_fixture="fx-1", expect_role="base", vocab_size=10, length=3)


def test_read_worker_json_rejects_non_integer_peak_bytes(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        '{"fixture_id": "fx-1", "role": "base", "argmax": [1, 2, 3], '
        '"peak_bytes": "lots", "adapter_applied": null}',
        encoding="utf-8",
    )

    with pytest.raises(WorkerArtifactError, match="peak_bytes"):
        read_worker_json(path, expect_fixture="fx-1", expect_role="base", vocab_size=10, length=3)


def test_read_worker_json_rejects_non_bool_adapter_applied(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        '{"fixture_id": "fx-1", "role": "base", "argmax": [1, 2, 3], '
        '"peak_bytes": 10, "adapter_applied": "yes"}',
        encoding="utf-8",
    )

    with pytest.raises(WorkerArtifactError, match="adapter_applied"):
        read_worker_json(path, expect_fixture="fx-1", expect_role="base", vocab_size=10, length=3)


def test_main_rejects_non_integer_memory_size(monkeypatch, tmp_path: Path) -> None:
    class BadDeviceInfoMx(FakeMxCore):
        def device_info(self) -> dict[str, object]:
            info = super().device_info()
            info["memory_size"] = "not-an-int"
            return info

    mx = BadDeviceInfoMx()
    mlx_lm = LoadForbiddenMlxLmModule()

    monkeypatch.setattr(
        worker_module.importlib,
        "import_module",
        make_import_module({"mlx.core": mx, "mlx_lm": mlx_lm}),
    )

    with pytest.raises(MemorySafetyError, match="memory_size"):
        main(
            [
                "--model-path",
                "/models/base",
                "--token-ids",
                "1",
                "--fixture-id",
                "fx-1",
                "--role",
                "base",
                "--out",
                str(tmp_path / "result.json"),
            ]
        )
    assert mlx_lm.load_calls == 0
