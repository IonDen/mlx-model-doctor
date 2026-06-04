# mlx-model-doctor

[![PyPI version](https://img.shields.io/pypi/v/mlx-model-doctor.svg)](https://pypi.org/project/mlx-model-doctor/)
[![Python versions](https://img.shields.io/pypi/pyversions/mlx-model-doctor.svg)](https://pypi.org/project/mlx-model-doctor/)
[![License: Apache 2.0](https://img.shields.io/pypi/l/mlx-model-doctor.svg)](https://github.com/IonDen/mlx-model-doctor/blob/master/LICENSE)

Validate an MLX / Hugging Face model repository before you load it.

A model repo can be broken in ways you only discover halfway through `load()`: a `config.json` that disagrees with the weights, a missing tokenizer file, a `model.safetensors.index.json` that points at shards that aren't there, quantization metadata that doesn't match the tensors, or a model that simply won't fit in the memory you have. `mlx-model-doctor` checks those up front and prints a report, so a bad repo fails fast with a clear reason instead of a confusing crash.

The static checks read repository metadata only — `config.json`, the tokenizer files, the safetensors index, quantization fields. They need no GPU or MLX and don't download the weights, so they're cheap to run anywhere. An optional `--smoke` check loads the model through `mlx-lm` (Apple Silicon) under a memory cap, to confirm it loads and generates.

```bash
mlx-model-doctor check local ./my-model
mlx-model-doctor check hf mlx-community/Llama-3.2-3B-Instruct-4bit
```

See [EXAMPLES.md](EXAMPLES.md) for real, dated transcripts of every command.

## Install

```bash
pip install mlx-model-doctor
# With the optional mlx-lm smoke check (Apple Silicon):
pip install "mlx-model-doctor[mlx-lm]"
```

Or with `uv`:

```bash
uv add mlx-model-doctor
uv add "mlx-model-doctor[mlx-lm]"
```

Verify the install:

```bash
mlx-model-doctor version
mlx-model-doctor --help
```

Requires Python ≥ 3.11. The static checks are pure Python (`huggingface-hub` + `safetensors`); only the optional `--smoke` runtime check needs `mlx-lm` and Apple Silicon.

## What it checks

The built-in `text` plugin runs these against a model repository, in order:

- **Required files** — `config.json` is present and readable.
- **Config consistency** — `config.json` parses, and its `model_type` is set.
- **Tokenizer** — the tokenizer files a text model needs are present, and the special-token configuration is coherent.
- **Safetensors index** — when the weights are sharded, `model.safetensors.index.json` is valid and every shard it references exists.
- **Quantization metadata** — quantization fields in the config are consistent with what the weights actually carry.
- **Memory budget** — an estimate of the memory the model needs at your context length, compared against a budget you pass with `--max-memory`.

Each check returns a result with a status (`pass` / `warn` / `fail` / `skip`), a message, and — when something is wrong — a remediation hint. The report aggregates them, and the process exit code reflects the worst result under your fail policy.

## Python API

```python
from mlx_model_doctor import check_local_model, check_hf_model

report = check_local_model("./my-model")
print(report.summary)            # {"pass": 6, "warn": 1, "fail": 0, "skip": 1}
for result in report.results:
    print(result.status, result.check_id, result.message)

# Hugging Face repos (hits the Hub):
report = check_hf_model("mlx-community/Llama-3.2-3B-Instruct-4bit")
```

`DoctorReport` renders to text, JSON, or Markdown (`render_text` / `render_json` / `render_markdown`), and the result objects are frozen dataclasses, so the output is stable to diff in CI.

## Commands

| Command | What it does |
|---|---|
| `version` | Print the version plus the active Python, virtualenv, and dependency status. |
| `man` | Print usage examples and the exit-code table. |
| `plugins` | List registered check plugins (`text` today). |
| `check local <path>` | Validate a model directory on disk. |
| `check hf <repo_id>` | Validate a model repository on the Hugging Face Hub (network). |
| `sample hf` | Survey likely-MLX repos for an author and validate a deterministic sample. |

`check` accepts `--format {text,json,markdown}`, `--output <file>`, `--max-memory <e.g. 32gb>`, `--context-length <n>`, `--fail-on {error,warn,never}`, `--include-weights`, and `--smoke`.

Exit codes: `0` checks passed (under the fail policy), `1` checks found failures, `2` tool error — a bad target, a missing dependency, or zero checks run.

## The Hugging Face path

`check hf` and `sample hf` talk to the Hub through `huggingface-hub`. They read repository metadata (the file list, sizes, and the small text files) rather than downloading the weights, but they do need network access, and an auth or rate-limit problem surfaces as a clear tool error rather than a stack trace. `sample hf` is a survey: it lists an author's repos, keeps the ones that look like MLX models, validates a deterministic sample of them, and reports each as its own batch item — a per-model failure is recorded and the run continues.

## Status

**Alpha (0.1.0).** The static `check local` path and the report/CLI surface are solid and well tested. The Hugging Face path (`check hf`, `sample hf`) is implemented and tested offline against fakes; its live behavior is exercised by opt-in network tests. The API may still shift before 1.0 — pin a version if you depend on it.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Validating a model repository does not touch the model's own weights or license; those belong to their respective authors.

## Acknowledgements

- [Apple ML Explore](https://github.com/ml-explore/mlx) for MLX and [`mlx-lm`](https://github.com/ml-explore/mlx-lm).
- [Hugging Face](https://github.com/huggingface/huggingface_hub) for the Hub client and `safetensors`.

## Sister projects

Other MLX libraries for Apple Silicon:

- [mlx-taef](https://github.com/IonDen/mlx-taef) — tiny autoencoders for fast diffusion-latent previews and low-memory decode (FLUX / SD).
- [mlx-teacache](https://github.com/IonDen/mlx-teacache) — TeaCache residual caching to skip redundant FLUX denoising steps.

---

By Denis Ineshin · [ineshin.space](https://ineshin.space)
