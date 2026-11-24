from dataclasses import dataclass
from pathlib import Path

import pytest

import mlx_model_doctor.checks.smoke as smoke_module
from mlx_model_doctor.checks.smoke import MlxLmBackend, MlxLmSmokeCheck, SmokeGeneration
from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.errors import DependencyError, MemorySafetyError, ModelDoctorError
from mlx_model_doctor.memory import GIB
from mlx_model_doctor.targets import HfTarget, LocalTarget
from tests.fakes import (
    FakeShardHub,
    FakeTarget,
    check_options,
    use_fake_home,
    write_hf_cache_snapshot,
)

# Small, known-good VLM pinned for the live canary (mlx-community, 4-bit quantized).
CANARY_REPO = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"


def canary_snapshot_dir(*, cache_dir: Path | None = None) -> Path:
    """Return the canary's snapshot folder from the local Hugging Face cache only.

    Never contacts the Hub. A missing snapshot fails the test: a --run-smoke run is
    prepared on purpose, so the model must already be downloaded.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        folder = snapshot_download(CANARY_REPO, cache_dir=cache_dir, local_files_only=True)
    except LocalEntryNotFoundError:
        pytest.fail(
            f"{CANARY_REPO} is not in the local Hugging Face cache. "
            f"Run `hf download {CANARY_REPO}` before `pytest --run-smoke`.",
            pytrace=False,
        )
    return Path(folder)


def test_mlx_lm_smoke_check_passes_for_non_empty_generation() -> None:
    check = MlxLmSmokeCheck(backend=FakeSmokeBackend(generation=SmokeGeneration(text="ok")))

    result = check.run(CheckContext(target=FakeTarget(files={}), options=check_options()))

    assert result.status == "pass"
    assert result.severity == "info"
    assert result.details["generated_text_chars"] == 2
    assert "text" not in result.details


def test_mlx_lm_smoke_check_fails_high_for_empty_generation() -> None:
    check = MlxLmSmokeCheck(backend=FakeSmokeBackend(generation=SmokeGeneration(text="")))

    result = check.run(CheckContext(target=FakeTarget(files={}), options=check_options()))

    assert result.status == "fail"
    assert result.severity == "high"
    assert "empty" in result.message


def test_mlx_lm_smoke_check_converts_ordinary_backend_exception_to_failure() -> None:
    check = MlxLmSmokeCheck(backend=FakeSmokeBackend(error=RuntimeError("generate failed")))

    result = check.run(CheckContext(target=FakeTarget(files={}), options=check_options()))

    assert result.status == "fail"
    assert result.severity == "high"
    assert "generate failed" in result.message


def test_mlx_lm_smoke_check_propagates_dependency_errors() -> None:
    check = MlxLmSmokeCheck(
        backend=FakeSmokeBackend(
            error=DependencyError(
                missing_package="mlx-lm",
                extra_name="mlx-lm",
                executable="/python",
            )
        )
    )

    with pytest.raises(DependencyError, match="mlx-lm"):
        check.run(CheckContext(target=FakeTarget(files={}), options=check_options()))


def test_mlx_lm_smoke_check_propagates_memory_safety_errors() -> None:
    # MemorySafetyError is a ModelDoctorError and must propagate as a tool error
    # (exit 2), never be demoted to a `fail` CheckResult (exit 1) by the broad
    # `except Exception` branch. DependencyError alone cannot pin this: a mutation
    # adding `except MemorySafetyError: return fail` before the ModelDoctorError
    # re-raise would leave the DependencyError test green.
    check = MlxLmSmokeCheck(backend=FakeSmokeBackend(error=MemorySafetyError("caps unavailable")))

    with pytest.raises(MemorySafetyError, match="caps"):
        check.run(CheckContext(target=FakeTarget(files={}), options=check_options()))


def test_mlx_lm_backend_lazily_imports_installs_caps_and_records_peak(
    monkeypatch, tmp_path: Path
) -> None:
    mx = FakeMx()
    mlx_lm = FakeMlxLm(mx)

    def import_module(name: str) -> object:
        if name == "mlx.core":
            return mx
        if name == "mlx_lm":
            return mlx_lm
        raise ImportError(name)

    monkeypatch.setattr(smoke_module.importlib, "import_module", import_module)

    # The smoke load reads a real directory; an empty one carries no guarded file.
    generation = MlxLmBackend().generate(
        CheckContext(target=FakeTarget(files={}, name=str(tmp_path)), options=check_options())
    )

    assert generation.text.strip()
    assert generation.peak_memory_bytes == 1234
    assert generation.memory_caps_gib == (20, 22)
    assert mx.wired_limit == 20 * GIB
    assert mx.memory_limit == 22 * GIB
    assert mlx_lm.load_saw_reset is True
    assert mlx_lm.load_saw_caps == (20 * GIB, 22 * GIB)
    # Pin the load-bearing memory-safety knobs; a behavior-preserving prompt tweak should
    # not break this test, but a bounded max_tokens and quiet generation are contractual.
    assert mlx_lm.generate_kwargs is not None
    assert mlx_lm.generate_kwargs["max_tokens"] == 8
    assert mlx_lm.generate_kwargs["verbose"] is False
    prompt = mlx_lm.generate_kwargs["prompt"]
    assert isinstance(prompt, str)
    assert prompt.strip()  # a real prompt, not blanked to "" or whitespace


@pytest.mark.parametrize(
    "failure",
    ["device_info", "set_wired_limit", "set_memory_limit"],
)
def test_mlx_lm_backend_refuses_to_load_when_memory_caps_are_unavailable(
    monkeypatch,
    failure: str,
) -> None:
    mx = FakeMx(cap_failure=failure)
    mlx_lm = LoadForbiddenMlxLm()

    def import_module(name: str) -> object:
        if name == "mlx.core":
            return mx
        if name == "mlx_lm":
            return mlx_lm
        raise ImportError(name)

    monkeypatch.setattr(smoke_module.importlib, "import_module", import_module)

    with pytest.raises(MemorySafetyError, match="memory caps") as exc_info:
        MlxLmBackend().generate(CheckContext(target=FakeTarget(files={}), options=check_options()))

    assert isinstance(exc_info.value, ModelDoctorError)
    assert mlx_lm.load_calls == 0


def test_mlx_lm_backend_missing_dependencies_raise_install_hint(monkeypatch) -> None:
    def import_module(_name: str) -> object:
        raise ImportError("missing")

    monkeypatch.setattr(smoke_module.importlib, "import_module", import_module)

    with pytest.raises(DependencyError, match="Install it with") as exc_info:
        MlxLmBackend().generate(CheckContext(target=FakeTarget(files={}), options=check_options()))

    assert exc_info.value.missing_package == "mlx"


@dataclass
class FakeSmokeBackend:
    generation: SmokeGeneration | None = None
    error: Exception | None = None

    def generate(self, ctx: CheckContext) -> SmokeGeneration:
        if self.error is not None:
            raise self.error
        assert self.generation is not None
        return self.generation


class FakeMx:
    def __init__(self, *, cap_failure: str | None = None) -> None:
        self._cap_failure = cap_failure
        self.wired_limit: int | None = None
        self.memory_limit: int | None = None
        self.peak_was_reset = False

    def device_info(self) -> dict[str, object]:
        if self._cap_failure == "device_info":
            raise RuntimeError("device unavailable")
        return {"max_recommended_working_set_size": 25 * GIB}

    def set_wired_limit(self, value: int) -> None:
        if self._cap_failure == "set_wired_limit":
            raise RuntimeError("wired limit unavailable")
        self.wired_limit = value

    def set_memory_limit(self, value: int) -> None:
        if self._cap_failure == "set_memory_limit":
            raise RuntimeError("memory limit unavailable")
        self.memory_limit = value

    def reset_peak_memory(self) -> None:
        self.peak_was_reset = True

    def get_peak_memory(self) -> int:
        return 1234


class FakeMlxLm:
    def __init__(self, mx: FakeMx) -> None:
        self._mx = mx
        self.load_saw_reset = False
        self.load_saw_caps: tuple[int | None, int | None] | None = None
        self.generate_kwargs: dict[str, object] | None = None

    def load(
        self, path_or_repo: str, tokenizer_config: dict[str, object] | None = None
    ) -> tuple[object, object]:
        self.load_saw_reset = self._mx.peak_was_reset
        self.load_saw_caps = (self._mx.wired_limit, self._mx.memory_limit)
        return (object(), object())

    def generate(
        self,
        model: object,
        tokenizer: object,
        *,
        prompt: str,
        max_tokens: int,
        verbose: bool,
    ) -> str:
        self.generate_kwargs = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "verbose": verbose,
        }
        return "generated"


class LoadForbiddenMlxLm:
    load_calls = 0

    def load(
        self, path_or_repo: str, tokenizer_config: dict[str, object] | None = None
    ) -> tuple[object, object]:
        self.load_calls += 1
        raise AssertionError("mlx_lm.load() must not be called without MLX memory caps")

    def generate(
        self,
        model: object,
        tokenizer: object,
        *,
        prompt: str,
        max_tokens: int,
        verbose: bool,
    ) -> str:
        raise AssertionError("mlx_lm.generate() must not be called without MLX memory caps")


# --- VLM smoke fakes (pure Python, no real MLX) ---
#
# apply_chat_template's `config` parameter was added after verifying the real
# mlx-vlm 0.6.12 signature (inspect.signature), unchanged in 0.7.12 (the
# upstream surface tests, 2026-11-16); apply_chat_template requires
# a model-config argument (`processor, config, prompt, ...`), unlike the
# text-only mlx_lm.load()/generate() surface above.


@dataclass
class FakeVlmGenerationResult:
    text: str


@dataclass
class FakeVlmModule:
    """Fake mlx-vlm for offline testing."""

    _text: str = "A blue image"
    _raise_on_load: Exception | None = None
    _raise_on_generate: Exception | None = None
    load_calls: int = 0
    load_saw_trust_remote_code: bool | None = None
    generate_kwargs: dict[str, object] | None = None

    def load(self, path: str, *, trust_remote_code: bool = True) -> tuple[object, object]:
        self.load_calls += 1
        self.load_saw_trust_remote_code = trust_remote_code
        if self._raise_on_load is not None:
            raise self._raise_on_load
        return ("fake_model", "fake_processor")

    def apply_chat_template(
        self, processor: object, config: object, prompt: str, *, num_images: int = 1
    ) -> str:
        return f"<image>{prompt}"

    def generate(
        self,
        model: object,
        processor: object,
        prompt: str,
        image: object,
        *,
        max_tokens: int = 8,
        verbose: bool = False,
    ) -> FakeVlmGenerationResult:
        self.generate_kwargs = {
            "formatted_prompt": prompt,
            "max_tokens": max_tokens,
            "verbose": verbose,
        }
        if self._raise_on_generate is not None:
            raise self._raise_on_generate
        return FakeVlmGenerationResult(text=self._text)


class LoadForbiddenVlmModule:
    load_calls = 0

    def load(self, path: str, *, trust_remote_code: bool = False) -> tuple[object, object]:
        self.load_calls += 1
        raise AssertionError("mlx_vlm.load() must not be called without MLX memory caps")

    def apply_chat_template(
        self, processor: object, config: object, prompt: str, *, num_images: int = 1
    ) -> str:
        raise AssertionError("not expected")

    def generate(self, *a, **kw) -> object:
        raise AssertionError("not expected")


# --- VLM smoke check-level tests (FakeSmokeBackend boundary) ---


def test_vlm_smoke_passes_on_nonempty_output() -> None:
    from mlx_model_doctor.checks.smoke import MlxVlmSmokeCheck

    check = MlxVlmSmokeCheck(
        backend=FakeSmokeBackend(generation=SmokeGeneration(text="A blue image"))
    )
    result = check.run(CheckContext(target=FakeTarget(files={}), options=check_options()))
    assert result.status == "pass"
    assert result.check_id == "vlm/smoke.mlx_vlm"
    assert result.details["generated_text_chars"] > 0


def test_vlm_smoke_fails_on_empty_output() -> None:
    from mlx_model_doctor.checks.smoke import MlxVlmSmokeCheck

    check = MlxVlmSmokeCheck(backend=FakeSmokeBackend(generation=SmokeGeneration(text="")))
    result = check.run(CheckContext(target=FakeTarget(files={}), options=check_options()))
    assert result.status == "fail"


def test_vlm_smoke_fails_on_generate_exception() -> None:
    from mlx_model_doctor.checks.smoke import MlxVlmSmokeCheck

    check = MlxVlmSmokeCheck(backend=FakeSmokeBackend(error=RuntimeError("generate failed")))
    result = check.run(CheckContext(target=FakeTarget(files={}), options=check_options()))
    assert result.status == "fail"
    assert "generate failed" in result.message


def test_vlm_smoke_propagates_memory_safety_error() -> None:
    from mlx_model_doctor.checks.smoke import MlxVlmSmokeCheck

    check = MlxVlmSmokeCheck(backend=FakeSmokeBackend(error=MemorySafetyError("caps unavailable")))
    with pytest.raises(MemorySafetyError, match="caps"):
        check.run(CheckContext(target=FakeTarget(files={}), options=check_options()))


# --- VLM backend-level tests (FakeVlmModule + FakeMx, monkeypatched imports) ---


def test_vlm_backend_installs_caps_loads_and_reinstalls_caps(monkeypatch, tmp_path: Path) -> None:
    mx = FakeMx()
    vlm = FakeVlmModule()

    def import_module(name: str) -> object:
        if name == "mlx.core":
            return mx
        if name == "mlx_vlm":
            return vlm
        raise ImportError(name)

    monkeypatch.setattr(smoke_module.importlib, "import_module", import_module)

    from mlx_model_doctor.checks.smoke import MlxVlmBackend

    generation = MlxVlmBackend(vlm_module=vlm, mx_module=mx).generate(
        CheckContext(target=FakeTarget(files={}, name=str(tmp_path)), options=check_options())
    )

    assert generation.text.strip()
    assert generation.peak_memory_bytes == 1234
    assert generation.memory_caps_gib == (20, 22)
    # Caps installed before load
    assert mx.wired_limit == 20 * GIB
    assert mx.memory_limit == 22 * GIB
    # Load was called with trust_remote_code=False
    assert vlm.load_saw_trust_remote_code is False
    # Generate was called with bounded tokens
    assert vlm.generate_kwargs is not None
    assert vlm.generate_kwargs["max_tokens"] == 8
    assert vlm.generate_kwargs["verbose"] is False


@pytest.mark.parametrize(
    "failure",
    ["device_info", "set_wired_limit", "set_memory_limit"],
)
def test_vlm_backend_refuses_to_load_when_memory_caps_unavailable(
    monkeypatch, failure: str
) -> None:
    mx = FakeMx(cap_failure=failure)
    vlm = LoadForbiddenVlmModule()

    from mlx_model_doctor.checks.smoke import MlxVlmBackend

    with pytest.raises(MemorySafetyError, match="memory caps"):
        MlxVlmBackend(vlm_module=vlm, mx_module=mx).generate(
            CheckContext(target=FakeTarget(files={}), options=check_options())
        )
    assert vlm.load_calls == 0


def test_vlm_backend_missing_dependencies_raise_install_hint(monkeypatch) -> None:
    def import_module(_name: str) -> object:
        raise ImportError("missing")

    monkeypatch.setattr(smoke_module.importlib, "import_module", import_module)

    from mlx_model_doctor.checks.smoke import MlxVlmBackend

    with pytest.raises(DependencyError, match="Install it with") as exc_info:
        MlxVlmBackend().generate(CheckContext(target=FakeTarget(files={}), options=check_options()))
    assert exc_info.value.extra_name == "mlx-vlm"


def test_vlm_backend_generate_failure_returns_fail_result(tmp_path: Path) -> None:
    """Test that a generate-time (not load-time) exception becomes a fail result."""
    from mlx_model_doctor.checks.smoke import MlxVlmBackend, MlxVlmSmokeCheck

    mx = FakeMx()
    vlm = FakeVlmModule(_raise_on_generate=RuntimeError("vision encoder crash"))
    backend = MlxVlmBackend(vlm_module=vlm, mx_module=mx)
    check = MlxVlmSmokeCheck(backend=backend)
    result = check.run(
        CheckContext(target=FakeTarget(files={}, name=str(tmp_path)), options=check_options())
    )
    assert result.status == "fail"
    assert "vision encoder crash" in result.message


def _vlm_smoke_context(model_dir: Path) -> CheckContext:
    # The smoke load reads a real directory; an empty one carries no guarded file.
    return CheckContext(target=FakeTarget(files={}, name=str(model_dir)), options=check_options())


def test_vlm_backend_refuses_when_caps_fail_after_load(tmp_path: Path) -> None:
    """Post-load cap reinstall failure must raise MemorySafetyError."""
    from mlx_model_doctor.checks.smoke import MlxVlmBackend

    # Create a FakeMx that succeeds on first install_mlx_memory_caps but fails on second
    class CapsFailAfterLoadMx:
        def __init__(self) -> None:
            self.install_count = 0

        def device_info(self) -> dict[str, object]:
            self.install_count += 1
            if self.install_count <= 1:
                return {"max_recommended_working_set_size": 34_359_738_368}
            return {"max_recommended_working_set_size": 0}  # Too small -> (0,0) caps

        def set_wired_limit(self, value: int) -> None:
            pass

        def set_memory_limit(self, value: int) -> None:
            pass

        def reset_peak_memory(self) -> None:
            pass

        def get_peak_memory(self) -> int:
            return 0

    mx = CapsFailAfterLoadMx()
    backend = MlxVlmBackend(vlm_module=FakeVlmModule(), mx_module=mx)
    ctx = _vlm_smoke_context(tmp_path)
    with pytest.raises(MemorySafetyError, match="reinstalled after VLM load"):
        backend.generate(ctx)


def test_vlm_smoke_fails_when_text_attribute_is_none(tmp_path: Path) -> None:
    """A GenerationResult with .text=None must fail, not false-pass via str()."""
    from mlx_model_doctor.checks.smoke import MlxVlmBackend, MlxVlmSmokeCheck

    class NoneTextResult:
        text = None

    class NoneTextVlm(FakeVlmModule):
        def generate(
            self,
            model: object,
            processor: object,
            formatted_prompt: str,
            image: object,
            *,
            max_tokens: int = 8,
            verbose: bool = False,
        ) -> NoneTextResult:
            return NoneTextResult()

    backend = MlxVlmBackend(vlm_module=NoneTextVlm(), mx_module=FakeMx())
    check = MlxVlmSmokeCheck(backend=backend)
    ctx = _vlm_smoke_context(tmp_path)
    result = check.run(ctx)
    assert result.status == "fail"
    assert "empty" in result.message.lower()


# --- VLM live smoke canary (real mlx-vlm runtime, opt-in via --run-smoke) ---


@pytest.mark.smoke
def test_vlm_smoke_canary_on_real_model() -> None:
    """Live: load a small VLM model through mlx-vlm and verify non-empty output + cap persistence."""
    from mlx_model_doctor.checks.smoke import MlxVlmBackend, MlxVlmSmokeCheck
    from mlx_model_doctor.memory import install_mlx_memory_caps
    from mlx_model_doctor.targets import LocalTarget

    # The cached snapshot folder, read without the Hub, checked as a local target.
    snapshot = canary_snapshot_dir()

    caps_before = install_mlx_memory_caps()
    assert caps_before[0] > 0, "Memory caps must be available for the canary"

    backend = MlxVlmBackend()
    check = MlxVlmSmokeCheck(backend=backend)
    ctx = CheckContext(target=LocalTarget(snapshot), options=check_options())
    result = check.run(ctx)
    assert result.status == "pass", f"VLM canary failed: {result.message}"
    assert result.details["generated_text_chars"] > 0

    # Verify caps are still installed after generation.
    caps_after = install_mlx_memory_caps()
    assert caps_after == caps_before, (
        f"Memory caps drifted: before={caps_before}, after={caps_after}"
    )


# --- canary snapshot resolution (offline, from the local Hugging Face cache) ---

CANARY_COMMIT = "46d4cf06a06ffc1a766c214174f9cbed2f45bcab"


def test_canary_snapshot_resolution_returns_the_cached_snapshot_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: the canary asks the Hub for the snapshot, so it fails with the Hub unreachable
    # although the snapshot is cached.
    import huggingface_hub

    def hub_contacted(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the canary snapshot resolution contacted the Hub")

    monkeypatch.setattr(huggingface_hub.HfApi, "repo_info", hub_contacted)
    snapshot = write_hf_cache_snapshot(
        tmp_path,
        "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
        {"config.json": b"{}"},
        commit=CANARY_COMMIT,
    )
    refs = tmp_path / "models--mlx-community--Qwen2.5-VL-3B-Instruct-4bit" / "refs"
    refs.mkdir()
    (refs / "main").write_text(CANARY_COMMIT, encoding="utf-8")

    assert canary_snapshot_dir(cache_dir=tmp_path) == snapshot


def test_canary_snapshot_resolution_fails_naming_the_download_command_when_not_cached(
    tmp_path: Path,
) -> None:
    # Bug: a missing snapshot skips the canary (a prepared --run-smoke run then reports
    # nothing), or fails without saying how to fetch the model.
    # BaseException, then the type: pytest.raises(Failed) would let a skip through.
    with pytest.raises(
        BaseException, match=r"`hf download mlx-community/Qwen2\.5-VL-3B-Instruct-4bit`"
    ) as exc_info:
        canary_snapshot_dir(cache_dir=tmp_path)

    assert exc_info.type is pytest.fail.Exception


# --- failure text keeps no local path (the user's home, the model folder) ---

_SHA = "0123456789abcdef0123456789abcdef01234567"


def _spaced_home_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Return a fake home and a model folder below it, behind a spaced folder.

    The spaced folder keeps the home cut alone from shortening the model path: without
    the model folder passed to the reduction, the text keeps ``~/my models/m``.
    """
    home = tmp_path / "Users" / "John Smith"
    model = home / "my models" / "m"
    model.mkdir(parents=True)
    use_fake_home(monkeypatch, home=str(home))
    return home, model


def test_smoke_failure_names_the_model_folder_not_its_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: smoke.py copies the loader's text raw, so the message carries the home and
    # the full model path; or it skips the target's folder and keeps "~/my models/m".
    _home, model = _spaced_home_model(tmp_path, monkeypatch)
    target = LocalTarget(model)
    check = MlxLmSmokeCheck(
        backend=FakeSmokeBackend(error=OSError(f"No safetensors found in {target.name}"))
    )

    result = check.run(CheckContext(target=target, options=check_options()))

    assert result.status == "fail"
    assert result.message == "Smoke generation failed: No safetensors found in m"


def test_vlm_smoke_failure_names_the_file_not_its_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: the VLM smoke message has no reduction at all, so a missing processor file
    # prints the home and the whole model path.
    from mlx_model_doctor.checks.smoke import MlxVlmSmokeCheck

    _home, model = _spaced_home_model(tmp_path, monkeypatch)
    target = LocalTarget(model)
    missing = f"{target.name}/processor/preprocessor_config.json"
    check = MlxVlmSmokeCheck(
        backend=FakeSmokeBackend(error=OSError(f"No such file or directory: '{missing}'"))
    )

    result = check.run(CheckContext(target=target, options=check_options()))

    assert result.message == (
        "VLM smoke generation failed: No such file or directory: 'preprocessor_config.json'"
    )


def test_hf_smoke_failure_names_the_repo_not_the_cache_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: `check hf --smoke` prints the cache snapshot path, which the user never typed
    # and which holds their home.
    home = tmp_path / "Users" / "John Smith"
    use_fake_home(monkeypatch, home=str(home))
    snapshot = home / ".cache" / "huggingface" / "hub" / "models--o--n" / "snapshots" / _SHA
    error = OSError(f"{snapshot}/config.json not found; snapshot {snapshot}")
    target = HfTarget("o/n", hub=FakeShardHub(files={}, shards={}))
    check = MlxLmSmokeCheck(backend=FakeSmokeBackend(error=error))

    result = check.run(CheckContext(target=target, options=check_options()))

    assert result.message == "Smoke generation failed: config.json not found; snapshot o/n"


def test_smoke_failure_for_a_missing_model_folder_names_only_the_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: the NotADirectoryError from the guard embeds the typed model path, and the
    # smoke message prints it with the home.
    home = tmp_path / "Users" / "John Smith"
    missing = home / "my models" / "gone"
    use_fake_home(monkeypatch, home=str(home))
    mx = FakeMx()
    mlx_lm = FakeMlxLm(mx)

    def import_module(name: str) -> object:
        if name == "mlx.core":
            return mx
        if name == "mlx_lm":
            return mlx_lm
        raise ImportError(name)

    monkeypatch.setattr(smoke_module.importlib, "import_module", import_module)
    target = FakeTarget(files={}, name=str(missing))

    result = MlxLmSmokeCheck().run(CheckContext(target=target, options=check_options()))

    assert result.message == (
        "Smoke generation failed: model path is not an existing directory: gone"
    )
