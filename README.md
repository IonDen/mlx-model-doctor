# mlx-model-doctor

[![PyPI version](https://img.shields.io/pypi/v/mlx-model-doctor.svg)](https://pypi.org/project/mlx-model-doctor/)
[![Python versions](https://img.shields.io/pypi/pyversions/mlx-model-doctor.svg)](https://pypi.org/project/mlx-model-doctor/)
[![License: Apache 2.0](https://img.shields.io/pypi/l/mlx-model-doctor.svg)](https://github.com/IonDen/mlx-model-doctor/blob/master/LICENSE)

Validate an MLX / Hugging Face model repository before you load it.

A model repo can be broken in ways you only discover halfway through `load()`: a `config.json` that's missing or internally inconsistent, a missing tokenizer file, a `model.safetensors.index.json` that points at shards that aren't there, quantization metadata that uses a mode or group size MLX rejects, a chat template that's absent or whose stop token has a typo, a corrupt safetensors header, a quantized layer whose packed weight and scales shapes disagree, or a model that simply won't fit in the memory you have. `mlx-model-doctor` checks those up front and prints a report, so a bad repo fails fast with a clear reason instead of a confusing crash.

The checks read repository metadata and the safetensors *header* — `config.json`, the tokenizer files, the safetensors index, quantization fields, and the tensor map (dtypes, shapes, byte-offsets) parsed from the header alone. They need no GPU or MLX and never download the weights; on the Hub the header arrives over a small HTTP range request, so the checks stay cheap to run anywhere. An optional `--smoke` check loads the model through `mlx-lm` (Apple Silicon) under a memory cap, to confirm it loads and generates.

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

Requires Python ≥ 3.11. The static checks are pure Python and need only `huggingface-hub`; the optional `--smoke` runtime check is the one part that needs `mlx-lm` and Apple Silicon.

## What it checks

The built-in `text` plugin runs these against a model repository, broadly in this order:

- **Required files** — `config.json` is present and readable.
- **Config consistency** — `config.json` parses, and its `model_type` is set.
- **Tokenizer** — the tokenizer files a text model needs are present, and the special-token configuration is coherent.
- **Chat template** — a chat/instruct model declares a chat template (in `tokenizer_config.json` or a `chat_template.jinja`), and the end-of-turn token its template emits is a registered special token. A typo'd stop token loads fine and then never stops generating.
- **Safetensors index** — when the weights are sharded, `model.safetensors.index.json` is valid and every shard it references exists.
- **Tensor header** — read the safetensors header itself (the tensor map: dtypes, shapes, byte-offsets; no weight download) to catch a corrupt header (overlapping or out-of-bounds tensor offsets), a weight map that points at tensors no shard contains, a declared tied embedding that contradicts the stored weights, and an MLX-quantized layer whose packed-weight and scales shapes don't agree. These run by default; pass `--skip-weights` to skip them for a faster config-only pass.
- **Quantization metadata** — quantization fields are present and use a valid MLX mode with a valid group size and bit width (`affine`, `mxfp4`, `mxfp8`, `nvfp4`). This reads the metadata, not the tensors.
- **Generation tokens** — the `eos` / `pad` / `bos` token IDs are present and agree across `config.json`, `generation_config.json`, and `tokenizer_config.json`.
- **Memory budget** — an estimate of the memory the model needs at your context length, compared against a budget you pass with `--max-memory`.

Each check returns a result with a status (`pass` / `warn` / `fail` / `skip`), a message, and — when something is wrong — a remediation hint. The report aggregates them, and the process exit code reflects the worst result under your fail policy.

## Python API

```python
from mlx_model_doctor import check_local_model, check_hf_model

report = check_local_model("./my-model")
print(report.summary)            # {"pass": 9, "warn": 1, "fail": 0, "skip": 2}
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

`check` accepts `--format {text,json,markdown,github}`, `--output <file>`, `--max-memory <e.g. 32gb>`, `--context-length <n>`, `--fail-on {error,warn,never}`, `--skip-weights` (skip the tensor-header checks for a faster config-only pass), and `--smoke`. The `github` format prints GitHub Actions annotations (see [Use it in CI](#use-it-in-ci)).

Exit codes: `0` checks passed (under the fail policy), `1` checks found failures, `2` tool error — a bad target, a missing dependency, or zero checks run.

## The Hugging Face path

`check hf` and `sample hf` talk to the Hub through `huggingface-hub`. They read repository metadata (the file list, sizes, the small text files, and the safetensors header over a range request) rather than downloading the weights, but they do need network access, and an auth or rate-limit problem surfaces as a clear tool error rather than a stack trace. `sample hf` is a survey: it lists an author's repos, keeps the ones that look like MLX models, validates a deterministic sample of them, and reports each as its own batch item — a per-model failure is recorded and the run continues.

## Use it in CI

Gate a pull request on a model repository with the GitHub Action. It runs the static checks (no weights downloaded, no GPU), writes the report to the job summary, and fails the job under your fail policy:

```yaml
- uses: IonDen/mlx-model-doctor@v0
  with:
    source: hf
    target: mlx-community/Llama-3.2-3B-Instruct-4bit
    fail-on: warn
```

Add `version: "==0.5.2"` to pin the tool to a release; without it the action installs the latest published version.

For a model directory you keep in git, validate it on every commit with the pre-commit hook:

```yaml
repos:
  - repo: https://github.com/IonDen/mlx-model-doctor
    rev: v0.5.2
    hooks:
      - id: mlx-model-doctor
        args: ["path/to/model"]
```

## Output contract

`--format json` prints a stable, versioned payload. The top-level fields are:

| Field | Type | Description |
|---|---|---|
| `schema_version` | string | Schema major.minor version (e.g. `"1.0"`), independent of the package version. |
| `target` | string | The model path or repo ID that was checked. |
| `source` | `"local"` or `"hf"` | Where the model came from. |
| `plugin` | string | The check plugin that ran (e.g. `"text"`). |
| `summary` | object | Check counts: `pass`, `warn`, `fail`, `skip` (integers). |
| `environment` | object | Open object, currently empty — reserved for future environment metadata. |
| `zero_check_reason` | string or null | Non-null if no checks ran, explains why. |
| `results` | array | One entry per check; see below. |

Each result in `results[]` has:

| Field | Type | Description |
|---|---|---|
| `check_id` | string | Namespaced identifier, e.g. `"text/files.required"`. |
| `title` | string | Short human-readable check name. |
| `status` | string | `"pass"`, `"warn"`, `"fail"`, or `"skip"`. |
| `severity` | string | `"info"`, `"low"`, `"medium"`, or `"high"`. |
| `message` | string | What was found. |
| `remediation` | string or null | What to do if the check fired. |
| `details` | object | Open object with check-specific key/value pairs. |
| `duration_s` | number or null | How long this check took, in seconds. |

The machine-readable schema ships with the package at `mlx_model_doctor/schema/report.v1.schema.json` and is validated against real output in CI.

Exit codes: `0` checks passed under the fail policy, `1` failures found, `2` a tool error or zero checks run. `--format github` reports the same results as GitHub Actions annotations; inside a workflow it also writes the Markdown report to the job summary and the `pass` / `warn` / `fail` / `skip` / `exit-code` / `schema-version` values to the step outputs.

## Stability policy

### Public API

The names you can depend on — only change on a major release:

`check_local_model`, `check_hf_model`, `CheckOptions`, `DoctorReport`, `CheckResult`, `render_json`, `render_text`, `render_markdown`, `render_github`, `exit_code_for`, and the error types `ModelDoctorError`, `TargetError`, `DependencyError`, `MemorySafetyError`. `exit_code_for` raises `ValueError` on an unrecognized `fail-on` value.

### Internal layer

The check, plugin, and target Protocols; `CheckContext`; the plugin registry; and the `hub=` parameter on `check_hf_model` (a test injection seam whose type may change) are internal and not stable across releases.

### Schema versioning

`schema_version` is `MAJOR.MINOR`, versioned independently of the package. A minor bump adds new optional fields or new values to open fields (such as the memory check's `estimate_source` values). A major bump means a documented field was removed, renamed, or retyped, or a closed enum (`status`, `severity`, `source`) changed.

The top-level object, `summary`, and each entry in `results[]` are closed (`additionalProperties: false`), so a new field there is a coordinated schema edit plus a minor version bump. Validate against the schema that matches the payload's `schema_version`, not a pinned older copy — otherwise a newer payload's added field will fail your validator.

### Promoted `details` keys

`details` is otherwise free-form, but three keys from the memory check are stable across the 1.x schema line: `lower_bound_bytes`, `estimate_source`, and `memory_lower_bound_kind`. `lower_bound_bytes` is a structural lower-bound floor: it counts attention, MLP, and embedding parameters at ≤16-bit weights (or quantized-equivalent) plus KV cache, but excludes norms, biases, and an untied `lm_head`. It sits below real runtime use and is not a fit guarantee.

### Batch output

The published schema covers the single `check` report only. The `sample hf` batch output is not yet under the schema guarantee.

## Status

**Alpha (0.6.0).** The static `check local` path and the report/CLI surface are solid and well tested. The safetensors header (read without downloading weights) backs four tensor-level checks — offset corruption, weight-map parameter sanity, tied-embedding consistency, and MLX quantized-layer shape consistency — which run by default (`--skip-weights` opts out). A single `check` also reports whether a repository looks like an MLX model and why, and flags a vision-language repository that declares no way to resolve its image processor. The quantized-shape and quantization-mode checks read each layer's own `bits`/`group_size`/`mode`, so a mixed-precision model (4-bit experts with 8-bit dense and router layers) is validated per layer rather than reported as broken. The memory estimate handles mixed precision the same way: when a model mixes bit widths it takes the weight figure from the stored file sizes instead of the model-level setting. The Hugging Face path (`check hf`, `sample hf`) is implemented and tested offline against fakes; its live behavior is exercised by opt-in network tests. It also ships a GitHub Action and a pre-commit hook. The public API and JSON output now have a documented, versioned stability contract — see [Output contract](#output-contract) and [Stability policy](#stability-policy). Pin a version if you depend on the schema or the API.

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
