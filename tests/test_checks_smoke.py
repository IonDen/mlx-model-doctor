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
