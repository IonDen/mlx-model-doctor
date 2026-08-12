from dataclasses import dataclass

import pytest

import mlx_model_doctor.checks.smoke as smoke_module
from mlx_model_doctor.checks.smoke import MlxLmBackend, MlxLmSmokeCheck, SmokeGeneration
from mlx_model_doctor.context import CheckContext
from mlx_model_doctor.errors import DependencyError, MemorySafetyError, ModelDoctorError
from mlx_model_doctor.memory import GIB
from tests.fakes import FakeTarget, check_options


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


def test_mlx_lm_backend_lazily_imports_installs_caps_and_records_peak(monkeypatch) -> None:
    mx = FakeMx()
    mlx_lm = FakeMlxLm(mx)

    def import_module(name: str) -> object:
        if name == "mlx.core":
            return mx
        if name == "mlx_lm":
            return mlx_lm
        raise ImportError(name)

    monkeypatch.setattr(smoke_module.importlib, "import_module", import_module)

    generation = MlxLmBackend().generate(
        CheckContext(target=FakeTarget(files={}), options=check_options())
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

    def load(self, path_or_repo: str) -> tuple[object, object]:
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

    def load(self, path_or_repo: str) -> tuple[object, object]:
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
# mlx-vlm==0.6.12 signature (inspect.signature); apply_chat_template requires
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


def test_vlm_backend_installs_caps_loads_and_reinstalls_caps(monkeypatch) -> None:
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
        CheckContext(target=FakeTarget(files={}, name="test-vlm"), options=check_options())
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


def test_vlm_backend_generate_failure_returns_fail_result() -> None:
    """Test that a generate-time (not load-time) exception becomes a fail result."""
    from mlx_model_doctor.checks.smoke import MlxVlmBackend, MlxVlmSmokeCheck

    mx = FakeMx()
    vlm = FakeVlmModule(_raise_on_generate=RuntimeError("vision encoder crash"))
    backend = MlxVlmBackend(vlm_module=vlm, mx_module=mx)
    check = MlxVlmSmokeCheck(backend=backend)
    result = check.run(
        CheckContext(target=FakeTarget(files={}, name="test"), options=check_options())
    )
    assert result.status == "fail"
    assert "vision encoder crash" in result.message


def _vlm_smoke_context() -> CheckContext:
    return CheckContext(target=FakeTarget(files={}, name="test-vlm"), options=check_options())


def test_vlm_backend_refuses_when_caps_fail_after_load() -> None:
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
    ctx = _vlm_smoke_context()
    with pytest.raises(MemorySafetyError, match="reinstalled after VLM load"):
        backend.generate(ctx)


def test_vlm_smoke_fails_when_text_attribute_is_none() -> None:
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
    ctx = _vlm_smoke_context()
    result = check.run(ctx)
    assert result.status == "fail"
    assert "empty" in result.message.lower()


# --- VLM live smoke canary (real mlx-vlm runtime, opt-in via --run-smoke) ---


@pytest.mark.smoke
def test_vlm_smoke_canary_on_real_model() -> None:
    """Live: load a small VLM model through mlx-vlm and verify non-empty output + cap persistence."""
    from mlx_model_doctor.checks.smoke import MlxVlmBackend, MlxVlmSmokeCheck
    from mlx_model_doctor.memory import install_mlx_memory_caps

    caps_before = install_mlx_memory_caps()
    assert caps_before[0] > 0, "Memory caps must be available for the canary"

    # Small, known-good VLM model pinned for the canary (mlx-community, 4-bit quantized).
    backend = MlxVlmBackend()
    check = MlxVlmSmokeCheck(backend=backend)
    ctx = CheckContext(
        target=FakeTarget(files={}, name="mlx-community/nanoLLaVA-1.5-4bit", _source="hf"),
        options=check_options(),
    )
    result = check.run(ctx)
    assert result.status == "pass", f"VLM canary failed: {result.message}"
    assert result.details["generated_text_chars"] > 0

    # Verify caps are still installed after generation.
    caps_after = install_mlx_memory_caps()
    assert caps_after == caps_before, (
        f"Memory caps drifted: before={caps_before}, after={caps_after}"
    )
