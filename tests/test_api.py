import errno
import json
from pathlib import Path
from typing import Any

import pytest

import mlx_model_doctor.api as api
from mlx_model_doctor.api import check_hf_model, check_local_model
from mlx_model_doctor.checks.base import ModelCheck
from mlx_model_doctor.context import CheckContext, CheckOptions
from mlx_model_doctor.errors import ModelDoctorError, TargetError
from mlx_model_doctor.exit_codes import exit_code_for
from mlx_model_doctor.report import CheckResult, DoctorReport
from mlx_model_doctor.targets import HfTarget, LocalTarget
from tests.fakes import STANDARD_INDEX, FakeShardHub, shard_repo, write_repo


def test_check_local_model_returns_text_report_for_valid_local_model(tmp_path: Path) -> None:
    model = write_local_model(tmp_path)

    report = check_local_model(model)

    assert report.target == str(model.resolve())
    assert report.source == "local"
    assert report.plugin == "text"
    assert report.summary["fail"] == 0
    assert {result.check_id for result in report.results} == {
        "text/files.required",
        "text/config.json",
        "text/config.model_type",
        "text/compat.mlx_signal",
        "text/tokenizer.files",
        "text/tokenizer.special_tokens",
        "text/chat_template.presence",
        "text/chat_template.special_tokens",
        "text/safetensors.index",
        "text/safetensors.offsets",
        "text/weights.param_count",
        "text/weights.tied_embedding",
        "text/quantization.metadata",
        "text/quantization.mode",
        "text/quantization.shape",
        "text/generation_config.tokens",
        "text/vlm.image_processor",
        "text/memory.estimate",
    }


def test_default_check_runs_weight_checks_that_skip_without_safetensors(tmp_path: Path) -> None:
    model = write_local_model(tmp_path)
    report = check_local_model(model)
    weight_ids = {
        "text/safetensors.offsets",
        "text/weights.param_count",
        "text/weights.tied_embedding",
        "text/quantization.shape",
    }
    statuses = {r.check_id: r.status for r in report.results if r.check_id in weight_ids}
    assert statuses == dict.fromkeys(weight_ids, "skip")
    assert report.summary["fail"] == 0


def test_sample_options_suppress_weight_checks(tmp_path: Path) -> None:
    model = write_local_model(tmp_path)
    options = CheckOptions(
        max_memory_bytes=None,
        context_length=4096,
        include_weights=False,
        smoke=False,
        verbosity="normal",
    )
    report = check_local_model(model, options=options)
    weight_ids = {
        "text/safetensors.offsets",
        "text/weights.param_count",
        "text/weights.tied_embedding",
        "text/quantization.shape",
    }
    assert not ({r.check_id for r in report.results} & weight_ids)


def test_check_local_model_reports_missing_config_without_crashing(tmp_path: Path) -> None:
    model = tmp_path / "missing-config"
    model.mkdir()

    report = check_local_model(model)

    assert report.plugin == "text"
    assert report.summary["fail"] >= 1
    # Assert this specific check FAILED, not merely that it ran: an empty dir also
    # fails tokenizer/other checks, so summary["fail"] >= 1 would stay true even if
    # files.required silently downgraded from fail to warn.
    assert result_by_id(report.results, "text/files.required").status == "fail"


def test_check_local_model_propagates_custom_options_to_memory_estimate(
    tmp_path: Path,
) -> None:
    model = write_local_model(tmp_path)
    options = CheckOptions(
        max_memory_bytes=1,
        context_length=8,
        include_weights=True,
        smoke=False,
        verbosity="normal",
    )

    report = check_local_model(model, options=options)

    memory = result_by_id(report.results, "text/memory.estimate")
    assert memory.status == "fail"
    assert memory.severity == "high"
    assert memory.details["context_length"] == 8
    assert memory.details["max_memory_bytes"] == 1


def test_check_local_model_rejects_unknown_plugin_with_package_error(tmp_path: Path) -> None:
    model = write_local_model(tmp_path)

    with pytest.raises(ModelDoctorError, match="Unknown plugin"):
        check_local_model(model, plugin_name="vision")


def test_check_hf_model_returns_text_report_for_valid_fake_repo() -> None:
    hub = FakeHub(files=valid_hf_files())

    report = check_hf_model("org/model", hub=hub)

    assert report.target == "org/model"
    assert report.source == "hf"
    assert report.plugin == "text"
    assert report.summary["fail"] == 0
    assert result_by_id(report.results, "text/config.json").status == "pass"
    assert result_by_id(report.results, "text/tokenizer.files").status == "pass"


def test_check_local_model_runs_vlm_plugin_with_weight_checks(tmp_path: Path) -> None:
    model = write_vlm_local_model(tmp_path)

    report = check_local_model(model, plugin_name="vlm")

    assert report.plugin == "vlm"
    ids = {result.check_id for result in report.results}
    assert "vlm/image_processor" in ids
    assert "vlm/image_token.wiring" in ids
    assert "vlm/safetensors.offsets" in ids
    assert "text/vlm.image_processor" not in ids


def test_check_local_model_vlm_skip_weights_excludes_only_weight_checks(
    tmp_path: Path,
) -> None:
    model = write_vlm_local_model(tmp_path)
    options = CheckOptions(
        max_memory_bytes=None,
        context_length=4096,
        include_weights=False,
        smoke=False,
        verbosity="normal",
    )

    report = check_local_model(model, plugin_name="vlm", options=options)
    ids = {result.check_id for result in report.results}

    assert "vlm/image_token.wiring" in ids
    assert "vlm/safetensors.offsets" not in ids
    assert "vlm/weights.param_count" not in ids
    assert "vlm/weights.tied_embedding" not in ids
    assert "vlm/quantization.shape" not in ids


def test_check_local_model_vlm_smoke_gate_blocks_on_tiny_budget(tmp_path: Path) -> None:
    model = write_vlm_local_model(tmp_path)
    # A measurable weight file lets the memory pre-flight gate compute a lower
    # bound; without one, VlmMemoryEstimateCheck has nothing to estimate from
    # and the gate can never engage.
    (model / "model.safetensors").write_bytes(b"\x00" * 4096)
    options = CheckOptions(
        max_memory_bytes=1,
        context_length=4096,
        include_weights=False,
        smoke=True,
        verbosity="normal",
    )

    report = check_local_model(model, plugin_name="vlm", options=options)

    assert result_by_id(report.results, "vlm/smoke.memory_budget").status == "fail"


def test_check_hf_model_runs_vlm_plugin_for_fake_repo() -> None:
    hub = FakeHub(files=valid_vlm_hf_files())

    report = check_hf_model("org/vlm", plugin_name="vlm", hub=hub)

    assert report.plugin == "vlm"
    assert result_by_id(report.results, "vlm/image_processor").status == "pass"


def test_check_hf_model_reports_missing_config_without_crashing() -> None:
    hub = FakeHub(files={"tokenizer.json": b"{}"})

    report = check_hf_model("org/missing-config", hub=hub)

    assert report.plugin == "text"
    assert report.source == "hf"
    assert report.summary["fail"] >= 1
    assert result_by_id(report.results, "text/files.required").status == "fail"


def test_check_hf_model_reports_bad_config_metadata_without_crashing() -> None:
    hub = FakeHub(files={"config.json": b"{not-json", "tokenizer.json": b"{}"})

    report = check_hf_model("org/bad-config", hub=hub)

    assert result_by_id(report.results, "text/config.json").status == "fail"


def test_check_hf_model_propagates_custom_options_to_memory_estimate() -> None:
    options = CheckOptions(
        max_memory_bytes=1,
        context_length=8,
        include_weights=True,
        smoke=False,
        verbosity="normal",
    )
    hub = FakeHub(files=valid_hf_files())

    report = check_hf_model("org/model", options=options, hub=hub)

    memory = result_by_id(report.results, "text/memory.estimate")
    assert memory.status == "fail"
    assert memory.severity == "high"
    assert memory.details["context_length"] == 8
    assert memory.details["max_memory_bytes"] == 1


def test_check_hf_model_rejects_unknown_plugin_with_package_error() -> None:
    hub = FakeHub(files=valid_hf_files())

    with pytest.raises(ModelDoctorError, match="Unknown plugin"):
        check_hf_model("org/model", plugin_name="vision", hub=hub)


def test_check_hf_model_maps_model_info_error_to_target_error() -> None:
    hub = FakeHub(model_info_error=RuntimeError("gated repo"))

    with pytest.raises(TargetError, match="Could not inspect Hugging Face model") as exc_info:
        check_hf_model("org/gated", hub=hub)

    assert exc_info.value.source == "hf"


def test_check_hf_model_propagates_download_target_error() -> None:
    hub = FakeHub(
        files=valid_hf_files(),
        download_errors={"config.json": RuntimeError("network unavailable")},
    )

    with pytest.raises(TargetError, match="Could not read Hugging Face model file") as exc_info:
        check_hf_model("org/model", hub=hub)

    assert exc_info.value.source == "hf"


def test_check_local_model_includes_smoke_results_only_when_requested(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model = write_local_model(tmp_path)
    monkeypatch.setattr(api, "get_plugin", lambda _name: FakePlugin())

    without_smoke = check_local_model(
        model,
        options=CheckOptions(
            max_memory_bytes=None,
            context_length=4096,
            include_weights=False,
            smoke=False,
            verbosity="normal",
        ),
    )
    with_smoke = check_local_model(
        model,
        options=CheckOptions(
            max_memory_bytes=None,
            context_length=4096,
            include_weights=False,
            smoke=True,
            verbosity="normal",
        ),
    )

    assert [result.check_id for result in without_smoke.results] == ["text/static.fake"]
    assert [result.check_id for result in with_smoke.results] == [
        "text/static.fake",
        "text/smoke.fake",
    ]


def test_check_hf_model_includes_smoke_results_when_requested(monkeypatch) -> None:
    monkeypatch.setattr(api, "get_plugin", lambda _name: FakePlugin())

    report = check_hf_model(
        "org/model",
        hub=FakeHub(files=valid_hf_files()),
        options=CheckOptions(
            max_memory_bytes=None,
            context_length=4096,
            include_weights=False,
            smoke=True,
            verbosity="normal",
        ),
    )

    assert [result.check_id for result in report.results] == [
        "text/static.fake",
        "text/smoke.fake",
    ]


def test_zero_check_local_run_populates_zero_check_reason(monkeypatch, tmp_path: Path) -> None:
    model = write_local_model(tmp_path)
    monkeypatch.setattr(api, "get_plugin", lambda _name: EmptyPlugin())

    report = check_local_model(model)

    assert report.results == ()
    assert report.zero_check_reason is not None
    assert "empty" in report.zero_check_reason


def test_zero_check_hf_run_populates_zero_check_reason(monkeypatch) -> None:
    monkeypatch.setattr(api, "get_plugin", lambda _name: EmptyPlugin())

    report = check_hf_model("org/model", hub=FakeHub(files=valid_hf_files()))

    assert report.results == ()
    assert report.zero_check_reason is not None
    assert "empty" in report.zero_check_reason


def test_normal_local_run_leaves_zero_check_reason_null(tmp_path: Path) -> None:
    model = write_local_model(tmp_path)

    report = check_local_model(model)

    assert report.results
    assert report.zero_check_reason is None


def test_check_local_model_chat_model_passes_new_checks(tmp_path: Path) -> None:
    model = tmp_path / "chat-model"
    model.mkdir()
    config = {
        "model_type": "llama",
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": 256,
        "intermediate_size": 512,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "quantization": {"mode": "affine", "bits": 4, "group_size": 64},
    }
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "eos_token": "<|im_end|>",
                "added_tokens_decoder": {"2": {"content": "<|im_end|>"}},
                "chat_template": "{{ messages }}<|im_end|>",
            }
        ),
        encoding="utf-8",
    )
    (model / "generation_config.json").write_text(
        json.dumps({"eos_token_id": 2, "pad_token_id": 0}), encoding="utf-8"
    )

    report = check_local_model(model)
    by_id = {result.check_id: result for result in report.results}
    assert by_id["text/chat_template.presence"].status == "pass"
    assert by_id["text/chat_template.special_tokens"].status == "pass"
    assert by_id["text/generation_config.tokens"].status == "pass"
    assert by_id["text/quantization.mode"].status == "pass"
    assert report.summary["fail"] == 0


def write_local_model(root: Path) -> Path:
    model = root / "model"
    model.mkdir()
    config = {
        "model_type": "llama",
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": 256,
        "intermediate_size": 512,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "quantization": {"bits": 4, "group_size": 64},
    }
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    return model


def write_vlm_local_model(root: Path) -> Path:
    model = root / "vlm"
    model.mkdir()
    for name, data in valid_vlm_hf_files().items():
        (model / name).write_bytes(data)
    return model


def result_by_id(results, check_id: str):
    return next(result for result in results if result.check_id == check_id)


def valid_hf_files() -> dict[str, bytes]:
    config = {
        "model_type": "llama",
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": 256,
        "intermediate_size": 512,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "quantization": {"bits": 4, "group_size": 64},
    }
    return {"config.json": json.dumps(config).encode(), "tokenizer.json": b"{}"}


def valid_vlm_hf_files() -> dict[str, bytes]:
    config = {
        "model_type": "qwen2_5_vl",
        "vision_config": {},
        "image_token_id": 151655,
        "quantization": {"bits": 4, "group_size": 64},
        "pad_token_id": 0,
        "eos_token_id": 1,
    }
    tokenizer_config = {
        "added_tokens_decoder": {"151655": {"content": "<|image_pad|>"}},
        "extra_special_tokens": {"image_token": "<|image_pad|>"},
        "chat_template": "<|vision_start|><|image_pad|><|vision_end|>",
    }
    preproc = {"image_processor_type": "Qwen2VLImageProcessor"}
    return {
        "config.json": json.dumps(config).encode(),
        "tokenizer.json": b"{}",
        "tokenizer_config.json": json.dumps(tokenizer_config).encode(),
        "preprocessor_config.json": json.dumps(preproc).encode(),
    }


class FakeSibling:
    def __init__(self, *, rfilename: str, size: int | None) -> None:
        self.rfilename = rfilename
        self.size = size


class FakeModelInfo:
    def __init__(
        self,
        *,
        siblings: tuple[FakeSibling, ...],
        tags: tuple[str, ...] | None = None,
        library_name: str | None = None,
    ) -> None:
        self.siblings = siblings
        self.tags = tags
        self.library_name = library_name


class FakeHub:
    def __init__(
        self,
        *,
        files: dict[str, bytes] | None = None,
        model_info_error: Exception | None = None,
        download_errors: dict[str, Exception] | None = None,
        safetensors: object | None = None,
    ) -> None:
        self.files = files if files is not None else {}
        self.model_info_error = model_info_error
        self.download_errors = download_errors if download_errors is not None else {}
        self.safetensors = safetensors

    def model_info(self, repo_id: str, *, files_metadata: bool) -> FakeModelInfo:
        if self.model_info_error is not None:
            raise self.model_info_error
        return FakeModelInfo(
            siblings=tuple(
                FakeSibling(rfilename=path, size=len(data)) for path, data in self.files.items()
            )
        )

    def download_bytes(self, repo_id: str, filename: str) -> bytes:
        if filename in self.download_errors:
            raise self.download_errors[filename]
        return self.files[filename]

    def safetensors_metadata(self, repo_id: str) -> object | None:
        return self.safetensors


class FakePlugin:
    name = "text"

    def static_checks(self) -> tuple[ModelCheck, ...]:
        return (FakeCheck("text/static.fake"),)

    def weight_checks(self) -> tuple[ModelCheck, ...]:
        return (FakeCheck("text/weights.fake"),)

    def smoke_checks(self) -> tuple[ModelCheck, ...]:
        return (FakeCheck("text/smoke.fake"),)


class EmptyPlugin:
    name = "empty"

    def static_checks(self) -> tuple[ModelCheck, ...]:
        return ()

    def weight_checks(self) -> tuple[ModelCheck, ...]:
        return ()

    def smoke_checks(self) -> tuple[ModelCheck, ...]:
        return ()


class FakeCheck:
    title = "Fake check"

    def __init__(self, check_id: str) -> None:
        self.check_id = check_id

    def run(self, _ctx: CheckContext) -> CheckResult:
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status="pass",
            severity="info",
            message="ok",
        )


# --- stale / partial root index end to end (0080, 0082) and local/HF parity (0079) -------

_LAYOUT_CHECKS = ("text/safetensors.index", "text/safetensors.offsets", "text/weights.param_count")


def test_hf_stale_index_report_warns_and_weight_checks_read_shipped_shards() -> None:
    # Bug caught: the default HF run dying on the stale index (exit 2), or the weight
    # checks judging the stale weight_map instead of the shipped tensors.
    files, shards = shard_repo("stale")
    hub = FakeShardHub(files=files, shards=shards)

    report = check_hf_model("org/stale", hub=hub)

    index = result_by_id(report.results, "text/safetensors.index")
    assert (index.status, index.severity) == ("warn", "medium")
    assert "names 2 shards" in index.message
    assert "3 other shards ship" in index.message
    params = result_by_id(report.results, "text/weights.param_count")
    assert params.status == "pass"
    assert params.details["stored_element_count"] == 18  # 2*3 + 5 + 7
    assert hub.safetensors_metadata_calls == []
    assert report.summary["fail"] == 0


def test_hf_partial_index_returns_a_report_that_fails_the_index_check() -> None:
    # Bug caught (0080): a partly missing index aborting the run with a tool error and
    # no report instead of a failing index check (exit 1).
    files, shards = shard_repo("partial")
    hub = FakeShardHub(files=files, shards=shards)

    report = check_hf_model("org/partial", hub=hub)

    assert result_by_id(report.results, "text/safetensors.index").status == "fail"
    assert result_by_id(report.results, "text/safetensors.offsets").status == "fail"
    assert exit_code_for(report, fail_on="error") == 1
    # Bug caught (T3): HfTarget not classifying, leaving the fake's 404 as the backstop.
    assert hub.safetensors_metadata_calls == []


_MODEL_SHARDS = (
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
)


def test_stale_index_beside_consolidated_reads_only_the_model_shards(tmp_path: Path) -> None:
    # Bug caught (T1): either target reading every top-level shard on the stale path, so
    # consolidated.safetensors (which no MLX loader reads beside model* shards) feeds
    # its tensors into the header.
    files, shards = shard_repo("stale_with_consolidated")
    hub = FakeShardHub(files=files, shards=shards)

    local = LocalTarget(write_repo(tmp_path / "repo", files, shards)).safetensors_header()
    hf = HfTarget("org/stale", hub=hub).safetensors_header()

    assert local is not None
    assert hf is not None
    assert [fh.filename for fh in local.files] == list(_MODEL_SHARDS)
    assert [fh.filename for fh in hf.files] == list(_MODEL_SHARDS)
    assert hub.file_metadata_calls == list(_MODEL_SHARDS)
    assert local.total_stored_element_count() == hf.total_stored_element_count() == 18


@pytest.mark.parametrize(
    "layout",
    [
        "stale",
        "partial",
        "nested",
        "consistent",
        "consolidated_only",
        "stale_with_consolidated",
        "dot_prefixed",
        "escaping",
    ],
)
def test_local_and_hf_agree_on_the_same_file_set(tmp_path: Path, layout: str) -> None:
    # Bug caught: the local and HF targets classifying the same repo differently (the
    # 0079 nested-index warn, or one target excusing what the other fails), or reading a
    # different shard set (T2: stored_element_count), or HfTarget skipping its own
    # classification (T3: the repo-metadata call made or not).
    files, shards = shard_repo(layout)
    hub = FakeShardHub(files=files, shards=shards)
    local = check_local_model(write_repo(tmp_path / layout, files, shards))
    hf = check_hf_model("org/" + layout, hub=hub)

    def statuses(report: DoctorReport) -> dict[str, str]:
        return {r.check_id: r.status for r in report.results if r.check_id in _LAYOUT_CHECKS}

    def observed(report: DoctorReport) -> tuple[str, object]:
        index = result_by_id(report.results, "text/safetensors.index")
        params = result_by_id(report.results, "text/weights.param_count")
        return index.message, params.details.get("stored_element_count")

    assert statuses(local) == statuses(hf) == EXPECTED_LAYOUT_STATUSES[layout]
    assert observed(local) == observed(hf)
    assert observed(hf)[1] == EXPECTED_STORED_ELEMENTS[layout]
    assert bool(hub.safetensors_metadata_calls) is EXPECTED_REPO_METADATA_CALL[layout]


EXPECTED_LAYOUT_STATUSES = {
    "stale": {
        "text/safetensors.index": "warn",
        "text/safetensors.offsets": "pass",
        "text/weights.param_count": "pass",
    },
    "partial": {
        "text/safetensors.index": "fail",
        "text/safetensors.offsets": "fail",
        "text/weights.param_count": "skip",
    },
    "nested": {
        "text/safetensors.index": "pass",
        "text/safetensors.offsets": "pass",
        "text/weights.param_count": "pass",
    },
    "consistent": {
        "text/safetensors.index": "pass",
        "text/safetensors.offsets": "pass",
        "text/weights.param_count": "pass",
    },
    # Bug caught: consolidated.safetensors excusing the stale index on either target.
    "consolidated_only": {
        "text/safetensors.index": "fail",
        "text/safetensors.offsets": "fail",
        "text/weights.param_count": "skip",
    },
    "stale_with_consolidated": {
        "text/safetensors.index": "warn",
        "text/safetensors.offsets": "pass",
        "text/weights.param_count": "pass",
    },
    # Bug caught (C1): "./model-00002..." read as absent by the header classifier while
    # the index check finds it, failing the offsets check on a repo MLX loads.
    "dot_prefixed": {
        "text/safetensors.index": "pass",
        "text/safetensors.offsets": "pass",
        "text/weights.param_count": "pass",
    },
    # Bug caught (RC3): a "../" or absolute value failed as invalid on local but warned
    # as stale on HF. The index fails on both; the header reads the shipped shard.
    "escaping": {
        "text/safetensors.index": "fail",
        "text/safetensors.offsets": "pass",
        "text/weights.param_count": "pass",
    },
}

# Element counts from the fakes.shard_repo shard specs; None where param_count skips.
EXPECTED_STORED_ELEMENTS = {
    "stale": 18,  # 2*3 + 5 + 7
    "partial": None,
    "nested": 6,  # root model.safetensors only
    "consistent": 11,  # 2*3 + 5
    "consolidated_only": None,
    "stale_with_consolidated": 18,  # consolidated's 100 excluded
    "dot_prefixed": 11,
    "escaping": 6,  # the shipped 2*3 shard
}

# Whether HfTarget asks huggingface_hub to walk the index (only when it is usable).
EXPECTED_REPO_METADATA_CALL = {
    "stale": False,
    "partial": False,
    "nested": True,
    "consistent": True,
    "consolidated_only": False,
    "stale_with_consolidated": False,
    "dot_prefixed": True,
    "escaping": False,
}


def test_dot_prefixed_value_gets_the_same_file_header_on_both_targets(tmp_path: Path) -> None:
    # Bug caught (RC2): HF keys the shard by the raw "./model-00002..." value, so its
    # FileHeader is named differently and has no size (the offsets EOF bound is skipped).
    files, shards = shard_repo("dot_prefixed")
    local = LocalTarget(write_repo(tmp_path / "repo", files, shards)).safetensors_header()
    hf = HfTarget("org/dot", hub=FakeShardHub(files=files, shards=shards)).safetensors_header()

    assert local is not None
    assert hf is not None
    expected = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    local_files = sorted((fh.filename, fh.file_size) for fh in local.files)
    hf_files = sorted((fh.filename, fh.file_size) for fh in hf.files)
    assert [name for name, _ in hf_files] == expected
    assert hf_files == local_files
    assert all(size is not None for _, size in hf_files)


def test_overlong_index_value_is_absent_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug caught (RC1): on Python 3.11-3.13 Path.is_file raises ENAMETOOLONG for a
    # repo-controlled 300-character index value; the raw OSError escaped the index check
    # and the header. Patched here so the failure shows on every Python.
    long_name = "x" * 300 + ".safetensors"
    files = {STANDARD_INDEX: json.dumps({"weight_map": {"w": long_name}}).encode()}
    shards = {"model-00001-of-00001.safetensors": {"a": ("BF16", (2, 3))}}
    root = write_repo(tmp_path / "repo", files, shards)
    real_is_file = Path.is_file

    # Forward whatever the caller passed: follow_symlinks exists only from Python 3.13.
    def is_file(self: Path, *args: Any, **kwargs: Any) -> bool:
        if len(self.name) > 255:
            raise OSError(errno.ENAMETOOLONG, "File name too long", str(self))
        return real_is_file(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_file", is_file)

    header = LocalTarget(root).safetensors_header()
    report = check_local_model(root)

    assert header is not None
    assert [fh.filename for fh in header.files] == ["model-00001-of-00001.safetensors"]
    index = result_by_id(report.results, "text/safetensors.index")
    assert index.status == "warn"
    assert index.details["shard_layout"] == "stale_all_absent_with_shipped"
    assert index.details["missing_shards"] == (long_name,)
    assert result_by_id(report.results, "text/safetensors.offsets").status == "pass"
