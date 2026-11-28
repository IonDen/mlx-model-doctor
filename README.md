# mlx-model-doctor

<p align="center">
  <img src="https://raw.githubusercontent.com/IonDen/mlx-model-doctor/master/docs/assets/mlx-model-doctor.jpeg" alt="mlx-model-doctor" width="100%">
</p>

[![PyPI version](https://img.shields.io/pypi/v/mlx-model-doctor.svg)](https://pypi.org/project/mlx-model-doctor/)
[![Python versions](https://img.shields.io/pypi/pyversions/mlx-model-doctor.svg)](https://pypi.org/project/mlx-model-doctor/)
[![License: Apache 2.0](https://img.shields.io/pypi/l/mlx-model-doctor.svg)](https://github.com/IonDen/mlx-model-doctor/blob/master/LICENSE)

A model repository can be broken in ways you find only halfway through loading it: a missing tokenizer file, quantization settings MLX rejects, or a model too big for your memory. `mlx-model-doctor` checks a model repository before you load it with MLX, Apple's machine-learning framework, on Apple Silicon. The checks read only metadata and file headers, never the weights; they need no GPU and also run on Linux.

You need this if you download, convert, quantize or publish MLX models, by hand or in CI.

With it you can:

- check a local folder or a Hugging Face repository;
- fail a pull request with the GitHub Action;
- compare a model's estimated memory with a budget you set;
- load the model and generate a few tokens (`--smoke`, optional);
- check that a fused LoRA adapter kept what it learned (`parity mlx`);
- survey one author's repositories (`sample hf`).

## Try it

```bash
pip install mlx-model-doctor   # Python 3.11 or later
```

```console
$ mlx-model-doctor check hf mlx-community/Qwen2.5-0.5B-Instruct-4bit
MLX Model Doctor: mlx-community/Qwen2.5-0.5B-Instruct-4bit

Summary:
  pass: 16
  warn: 0
  fail: 0
  skip: 2

PASS info text/files.required
  Required config file: config.json is present.

PASS info text/config.json
  Config JSON: config.json contains a valid JSON object.

PASS info text/config.model_type
  Model type: config.json declares model_type=qwen2.

PASS info text/compat.mlx_signal
  MLX compatibility signal: MLX-compatibility signals: tag:mlx, library:mlx, author:mlx-community, config:quantization, weights:mlx-quant, repo-name.

PASS info text/tokenizer.files
  Tokenizer files: Tokenizer artifacts are present.

SKIP info text/tokenizer.special_tokens
  Special token IDs: pad_token_id or eos_token_id is unavailable in config.json.

PASS info text/chat_template.presence
  Chat template: A chat template is present.

PASS info text/chat_template.special_tokens
  Chat template tokens: Chat-template token literals are registered special tokens.

PASS info text/safetensors.index
  Safetensors index: Safetensors indexes reference shard files that are present.

PASS info text/quantization.metadata
  Quantization metadata: config.json contains MLX quantization metadata.

PASS info text/quantization.mode
  Quantization mode: MLX quantization mode 'affine' has valid group_size/bits.

PASS info text/generation_config.tokens
  Generation-config tokens: Generation token IDs are present and consistent.

SKIP info text/vlm.image_processor
  VLM image processor: Not a vision-language repo; skipped.

PASS info text/memory.estimate
  Memory estimate: Estimated lower bound memory is advisory and may be below runtime use.

PASS info text/safetensors.offsets
  Safetensors offsets: Safetensors tensor offsets are well-formed; the data-section upper bound was not checked (header length unavailable on this target).

PASS info text/weights.param_count
  Weight parameter count: The weight map resolves to present tensors with non-zero parameters.

PASS info text/weights.tied_embedding
  Tied embeddings: Embedding tying is consistent with the stored tensors.

PASS info text/quantization.shape
  Quantization shape: Quantized tensor shapes are consistent with config bits/group_size.
```

A clean repository exits `0`; a failed check exits `1`. This output was captured with 0.11.1 on 2026-11-28. [EXAMPLES.md](https://github.com/IonDen/mlx-model-doctor/blob/master/EXAMPLES.md) has more, including repositories that fail.

## Install

```bash
pip install mlx-model-doctor
pip install "mlx-model-doctor[mlx-lm]"    # adds the --smoke and parity checks (Apple Silicon)
pip install "mlx-model-doctor[mlx-vlm]"   # adds --smoke for vision-language models
```

Or with `uv`: `uv add mlx-model-doctor`, or `uv add "mlx-model-doctor[mlx-lm]"`. Check the install with `mlx-model-doctor version` and `mlx-model-doctor --help`.

It needs Python 3.11 or later. The static checks are pure Python and need only `huggingface-hub`. The extras need Apple Silicon:

- `[mlx-lm]` asks for `mlx-lm>=0.32.0,<0.33` and `mlx>=0.32.4,<0.33`, and is for `--smoke` and `parity mlx` on text models.
- `[mlx-vlm]` asks for `mlx-vlm>=0.6.12,<0.8` and `mlx>=0.32.4,<0.33`, and is for `--smoke` with `--plugin vlm`.

Installing an extra upgrades `mlx` to 0.32.4 or later and, for `[mlx-lm]`, `mlx-lm` to 0.32.0 or later. A package in the same environment that pins an older `mlx` or `mlx-lm` cannot share it. mlx-train-perf 0.9.1 is one: its base install pins `mlx==0.32.0`, and its own `[mlx-lm]` extra keeps `mlx-lm` below 0.32, so it cannot share an environment with either extra.

## What it checks

- Required files and a consistent `config.json`.
- Tokenizer files and special tokens.
- A chat template whose end-of-turn token is a registered special token.
- The safetensors index, and that every shard it names exists.
- The safetensors header (the table of tensors at the start of each weights file): offsets, weight map, tied embeddings, and quantized-layer shapes (`--skip-weights` skips these).
- Quantization metadata: a valid MLX mode, group size and bit width.
- The `eos` / `pad` / `bos` token IDs agree across the config files.
- A memory estimate against a budget from `--max-memory`.
- Whether the repository looks like an MLX model, and why (for information; it never fails the run).
- For a vision-language repository: whether an image processor can be resolved; with `--plugin vlm`, also the image-token wiring.

Each check returns `pass`, `warn`, `fail` or `skip`, a message, and a fix hint when something is wrong. The exit code reflects the worst result under your fail policy. More on each check is on [the checks page](https://github.com/IonDen/mlx-model-doctor/blob/master/docs/checks.md). The default `text` profile covers text models; for vision-language repositories add `--plugin vlm`.

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

`DoctorReport` renders to text, JSON or Markdown (`render_text`, `render_json`, `render_markdown`). The result objects are frozen dataclasses.

## Commands

| Command | What it does |
|---|---|
| `version` | Print the version plus the active Python, virtualenv, and dependency status. |
| `man` | Print usage examples and the exit-code table. |
| `plugins` | List registered check plugins (`text`, `vlm`). |
| `check local <path>` | Validate a model directory on disk. |
| `check hf <repo_id>` | Validate a model repository on the Hugging Face Hub (network). |
| `sample hf` | Survey likely-MLX repos for an author and validate a deterministic sample. |
| `parity mlx` | Check whether fusing a LoRA adapter into a base model kept the behavior the adapter learned. |

`check` accepts:

- `--format {text,json,markdown,github}` and `--output <file>`;
- `--max-memory <e.g. 32gb>` and `--context-length <n>`;
- `--fail-on {error,warn,never}`;
- `--skip-weights`, to skip the tensor-header checks for a faster config-only pass;
- `--plugin vlm`, for vision-language repositories (omit it for `text`);
- `--smoke`, for the load test below.

The `github` format prints GitHub Actions annotations (see [Use it in CI](https://github.com/IonDen/mlx-model-doctor#use-it-in-ci)).

JSON output keeps the full local path, home folder included, in `target` and, for `parity mlx`, in `path` and `original_ref`. Check JSON output before you share it. The fields and their types are on [the output contract page](https://github.com/IonDen/mlx-model-doctor/blob/master/docs/output-contract.md).

Exit codes: `0` checks passed (under the fail policy), `1` checks found failures, `2` tool error: a bad target, a missing dependency, or zero checks run.

## Smoke load test

`--smoke` loads the model through `mlx-lm` (through `mlx-vlm` with `--plugin vlm`) and generates, to confirm it works. It needs the `[mlx-lm]` extra (`[mlx-vlm]` with `--plugin vlm`) and Apple Silicon. It runs under memory caps worked out from your Mac's memory, and it refuses to load if it cannot install them. With `check hf`, it first downloads the repository snapshot, weights included (without `.py` files), checks it, and loads from that local copy. The smoke result records the snapshot revision. A failed snapshot download is a tool error (exit 2).

`--smoke` runs the model, so run it only on repositories you trust. Before it loads anything, the tool reads the model folder and refuses the load if the repository names its own loader code: a `model_file` in `config.json`. On VLM `--smoke` it also refuses an `auto_map`, `configuration_files` and, for six processor families, a chat template the repository ships itself. There is no option to override this. Versions before 0.10.1 do not refuse; upgrade before you run `--smoke` or `parity mlx` on a repository you do not control.

A refused smoke check fails (exit `1` under the default policy), and the static checks still run. They read only metadata and the safetensors header, so they are the safe way to look at a repository you do not trust. The exact rules are on [the smoke safety page](https://github.com/IonDen/mlx-model-doctor/blob/master/docs/smoke-safety.md).

## The Hugging Face path

`check hf` and `sample hf` talk to the Hub through `huggingface-hub` and need network access. The checks read the file list, sizes, the small text files and the safetensors header (over a range request), not the weights; `--smoke` is the exception, because it downloads the snapshot. An auth or rate-limit problem surfaces as a clear tool error, not a stack trace.

`sample hf` lists an author's repos, keeps the ones that look like MLX models, and validates a deterministic sample. Each repo is its own batch item: a failure on one is recorded and the run continues.

```bash
mlx-model-doctor sample hf --author mlx-community --limit 5
```

`--max-candidates`, `--signal-filter` and `--no-cache` / `--cache-ttl` tune the scan.

## Checking a model in your Hugging Face cache

After `hf download <repo>`, the model sits in your cache under `~/.cache/huggingface/hub/models--ORG--NAME/snapshots/<commit>/`. Point `check local` and `parity mlx --base` straight at that folder:

```bash
mlx-model-doctor check local ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-4bit/snapshots/<commit>
```

The report header names the repository and the snapshot, for example `mlx-community/Qwen3.5-0.8B-4bit (cache snapshot da28692)`.

Pass the snapshot folder itself (`.../snapshots/<commit>`), not a subfolder of it. The files in it are symlinks into the cache's own stored copies. Snapshots downloaded with huggingface_hub 1.x or 2.x are read. That includes the shared store that versions 1.32 and later use for large files. The tool follows a symlink only when it leads to one of those stored copies. Any other symlink that points outside the folder is refused. [EXAMPLES.md](https://github.com/IonDen/mlx-model-doctor/blob/master/EXAMPLES.md) has a real run.

## Validate before uploading to Hugging Face

For a model you build or convert locally, run the static checks before you upload, and treat warnings as release blockers unless you have reviewed them:

```bash
mlx-model-doctor check local ./dist/my-mlx-model --fail-on warn
hf upload my-org/my-mlx-model ./dist/my-mlx-model --repo-type model
mlx-model-doctor check hf my-org/my-mlx-model --fail-on warn
```

If your publisher is Python-based, keep the same order:

```python
from huggingface_hub import upload_folder

# Run `mlx-model-doctor check local ./dist/my-mlx-model --fail-on warn` first.
upload_folder(
    repo_id="my-org/my-mlx-model",
    folder_path="./dist/my-mlx-model",
    repo_type="model",
)
```

`--fail-on warn` is a clean release gate. The default `--fail-on error` lets warnings through and still blocks hard failures. After the upload, run `check hf` on the Hub copy: it runs the same checks there.

## Check a fused adapter kept its behavior

Fusing a LoRA adapter (a small set of extra weights from fine-tuning) into a base model should bake what it learned into the weights. With mlx-lm's default fuse the merged weights are re-quantized to the base's format, and that can erase much of what the adapter learned. `parity mlx` runs the same input through the base, the base with the adapter, and the fused model, and compares their top-token predictions.

```bash
mlx-model-doctor parity mlx \
  --base ./Qwen2.5-0.5B-Instruct-4bit \
  --adapter ./my-lora \
  --fused ./my-fused-model \
  --prompts ./examples.json
```

`--prompts` takes a JSON list of `{"prompt", "completion"}` examples; use ones the adapter should have changed. The verdict is `pass`, `fail_tracks_base` (the fuse fell back to the base), `fail_gross` (it matches neither), or `inconclusive` (the difference is within noise). Exit `0` is a confirmed pass, `1` a determined regression, `2` a run that cannot decide. It needs the `[mlx-lm]` extra and a base repository that ships a `tokenizer.json`. Each model loads in its own memory-capped subprocess, one at a time. If the base or fused folder names loader code (the `--smoke` rule above), it loads nothing and exits `2`; the parity page has the one case that exits `1`. If a fuse is `inconclusive` or worse, re-fuse with `mlx_lm.fuse --dequantize`. More on [the parity page](https://github.com/IonDen/mlx-model-doctor/blob/master/docs/parity.md).

## Use it in CI

Gate a pull request on a model repository with the [GitHub Marketplace Action](https://github.com/marketplace/actions/mlx-model-doctor). It runs the static checks (no weights downloaded, no GPU), writes the report to the job summary, and fails the job under your fail policy:

```yaml
- uses: IonDen/mlx-model-doctor@v0
  with:
    source: hf
    target: mlx-community/Llama-3.2-3B-Instruct-4bit
    plugin: text
    fail-on: warn
```

For a vision-language model, opt into the VLM profile:

```yaml
- uses: IonDen/mlx-model-doctor@v0
  with:
    source: hf
    target: mlx-community/InternVL3-2B-4bit
    plugin: vlm
```

Add `version: "==0.11.1"` to pin the tool to a release; without it the action installs the latest published version.

For a model directory you keep in git, validate it on every commit with the pre-commit hook:

```yaml
repos:
  - repo: https://github.com/IonDen/mlx-model-doctor
    rev: v0.11.1
    hooks:
      - id: mlx-model-doctor
        args: ["path/to/model"]
```

## Output contract

`--format json` prints a stable, versioned payload, with a published schema. The fields, their types and the exit codes are on [the output contract page](https://github.com/IonDen/mlx-model-doctor/blob/master/docs/output-contract.md).

## Stability policy

The public API names, the internal layer and the schema versioning rules are on [the stability page](https://github.com/IonDen/mlx-model-doctor/blob/master/docs/stability.md). The public API changes only on a major release, and `schema_version` is versioned apart from the package. Pin a version if you depend on the schema or the API.

## Version sensitivity

The checks encode behavior from specific upstream versions (MLX, transformers, safetensors). When a value falls outside the known set, such as a quantization mode added in a newer MLX or an unfamiliar safetensors dtype, the check warns instead of failing. Only structurally invalid metadata (a wrong type, missing required fields) fails. Each version-bound table carries a comment naming the upstream version it was verified against.

## Status

Beta, 0.11.1, released on PyPI. This version changes the README and the pages under `docs/`, not the code: the CLI, exit codes, JSON output and Python API behave as in 0.11.0. See [CHANGELOG.md](https://github.com/IonDen/mlx-model-doctor/blob/master/CHANGELOG.md) for every release and [ROADMAP.md](https://github.com/IonDen/mlx-model-doctor/blob/master/ROADMAP.md) for what is planned.

Questions and shared results go to [GitHub Discussions](https://github.com/IonDen/mlx-model-doctor/discussions); bugs go to [Issues](https://github.com/IonDen/mlx-model-doctor/issues).

## License

Apache-2.0. See [LICENSE](https://github.com/IonDen/mlx-model-doctor/blob/master/LICENSE) and [NOTICE](https://github.com/IonDen/mlx-model-doctor/blob/master/NOTICE). Validating a model repository does not touch the model's own weights or license; those belong to their respective authors.

## Acknowledgements

- [Apple ML Explore](https://github.com/ml-explore/mlx) for MLX and [`mlx-lm`](https://github.com/ml-explore/mlx-lm).
- [Hugging Face](https://github.com/huggingface/huggingface_hub) for the Hub client and `safetensors`.

## Sister projects

Other MLX libraries for Apple Silicon:

- [mlx-taef](https://github.com/IonDen/mlx-taef) — tiny autoencoders for fast diffusion-latent previews and low-memory decode (FLUX / SD).
- [mlx-teacache](https://github.com/IonDen/mlx-teacache) — TeaCache residual caching to skip redundant FLUX denoising steps.
- [mlx-quant-fidelity](https://github.com/IonDen/mlx-quant-fidelity) — measure how much quality a quantization costs (KL divergence, top-token flips, perplexity delta).

---

By Denis Ineshin · [ineshin.space](https://ineshin.space)
