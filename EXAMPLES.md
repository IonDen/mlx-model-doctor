# Examples

Real output from `mlx-model-doctor`, captured by running the tool — so you can see exactly what you get before installing it. Each block shows a command, its actual response, and a short read of what the result means.

> Captured with **mlx-model-doctor 0.6.1** on **2026-06-28**. Your venv paths will differ, and the Hugging Face examples (`check hf`, `sample hf`) are live snapshots of the Hub, so they drift over time — that's why they're dated. The two deliberately-broken repos in sections 6 and 7 (`ybelkada/opt-350m-lora` and `TheBloke/Llama-2-7B-GGUF`) are long-standing archival repos, picked because they keep failing the same way.

## 1. `version` — environment and dependency status

```console
$ mlx-model-doctor version
mlx-model-doctor 0.6.1
Python: 3.14.5
Executable: /path/to/.venv/bin/python3
Virtualenv: /path/to/.venv
Dependencies:
  huggingface-hub: 1.19.0
  safetensors: not installed
  mlx: not installed
  mlx-lm: not installed
```

Exit code `0`. Static checks need only `huggingface-hub`; `safetensors`, `mlx`, and `mlx-lm` show up once the optional `[mlx-lm]` extra is installed (for the `--smoke` check).

## 2. `man` — usage and exit codes

```console
$ mlx-model-doctor man
mlx-model-doctor manual

Examples:
  mlx-model-doctor version
  mlx-model-doctor plugins
  mlx-model-doctor check local ./model
  mlx-model-doctor check hf mlx-community/Llama-3.2-3B-Instruct-4bit
  mlx-model-doctor sample hf --author mlx-community --limit 5

Exit codes:
  0: checks passed or informational command completed
  1: checks found failures under the selected fail policy
  2: tool error, bad target, missing dependency, or zero checks
```

## 3. `check local` — a healthy model on disk

**Model:** `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, downloaded to disk. The everyday case: you have a model locally and want to confirm it before loading. Locally the tool also knows the file size and header length, so it can verify the tensor offsets are in bounds and `safetensors.offsets` passes outright (over the Hub it passes with a note instead — see section 5).

The `text/compat.mlx_signal` line reports *why* the repo looks like an MLX model. On disk it sees the `quantization` block, the quantized weights in the header, and the `4bit` in the name; the `mlx-community` author and the MLX tags are Hub metadata, so those extra signals only show up in the `check hf` run (section 5).

```console
$ mlx-model-doctor check local ./Qwen2.5-0.5B-Instruct-4bit
MLX Model Doctor: /path/to/Qwen2.5-0.5B-Instruct-4bit

Summary:
  pass: 15
  warn: 1
  fail: 0
  skip: 2

PASS info text/files.required
  Required config file: config.json is present.

PASS info text/config.json
  Config JSON: config.json contains a valid JSON object.

PASS info text/config.model_type
  Model type: config.json declares model_type=qwen2.

PASS info text/compat.mlx_signal
  MLX compatibility signal: MLX-compatibility signals: config:quantization, weights:mlx-quant, repo-name.

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

WARN low text/memory.estimate
  Memory estimate: Estimated lower bound memory is advisory and may be below runtime use.
  Fix: Treat this as a floor; account for runtime overhead before loading the model.

PASS info text/safetensors.offsets
  Safetensors offsets: Safetensors header tensor offsets are well-formed and in bounds.

PASS info text/weights.param_count
  Weight parameter count: The weight map resolves to present tensors with non-zero parameters.

PASS info text/weights.tied_embedding
  Tied embeddings: Embedding tying is consistent with the stored tensors.

PASS info text/quantization.shape
  Quantization shape: Quantized tensor shapes are consistent with config bits/group_size.
```

**Result:** exit code `0` — a clean MLX repo. The one `warn` is the always-advisory memory floor, and the two `skip`s are checks that don't apply (no special-token IDs in this config; not a vision model). The `--skip-weights` flag drops the four tensor-header checks for a faster config-only pass.

## 4. `check local` — catching a corrupt safetensors before you load it

**Model:** a local directory whose `model.safetensors` has two tensors with overlapping byte ranges — a corrupt header that would blow up at load time. `mlx-model-doctor` reads only the tensor header (never the weight body) and flags it.

```console
$ mlx-model-doctor check local ./model
MLX Model Doctor: /path/to/model

Summary:
  pass: 6
  warn: 2
  fail: 1
  skip: 9

PASS info text/files.required
  Required config file: config.json is present.

PASS info text/config.json
  Config JSON: config.json contains a valid JSON object.

PASS info text/config.model_type
  Model type: config.json declares model_type=llama.

PASS info text/compat.mlx_signal
  MLX compatibility signal: No MLX-compatibility signals found; this may not be an MLX/mlx-lm model.

PASS info text/tokenizer.files
  Tokenizer files: Tokenizer artifacts are present.

SKIP info text/tokenizer.special_tokens
  Special token IDs: pad_token_id or eos_token_id is unavailable in config.json.

SKIP info text/chat_template.presence
  Chat template: No tokenizer metadata, so a chat template cannot be checked.

SKIP info text/chat_template.special_tokens
  Chat template tokens: No chat template string, so token consistency cannot be checked.

SKIP info text/safetensors.index
  Safetensors index: No safetensors index was found.

WARN low text/quantization.metadata
  Quantization metadata: config.json does not contain quantization metadata.
  Fix: Add MLX top-level quantization metadata when the model is quantized.

SKIP info text/quantization.mode
  Quantization mode: No MLX quantization object to validate.

SKIP info text/generation_config.tokens
  Generation-config tokens: No generation token IDs declared, so consistency cannot be checked.

SKIP info text/vlm.image_processor
  VLM image processor: Not a vision-language repo; skipped.

WARN low text/memory.estimate
  Memory estimate: Estimated lower bound memory is advisory and may be below runtime use.
  Fix: Treat this as a floor; account for runtime overhead before loading the model.

FAIL high text/safetensors.offsets
  Safetensors offsets: Safetensors header has corrupt tensor offsets (overlap or out of bounds).
  Fix: Re-save the safetensors shard; its tensor offsets are inconsistent.

PASS info text/weights.param_count
  Weight parameter count: The weight map resolves to present tensors with non-zero parameters.

SKIP info text/weights.tied_embedding
  Tied embeddings: No recognized embedding or output-head tensor; cannot check tying.

SKIP info text/quantization.shape
  Quantization shape: No MLX quantization metadata or safetensors header to check.
```

**Result:** exit code `1`. The `FAIL` on `text/safetensors.offsets` is the catch — overlapping tensor ranges that would corrupt the load. Only the header is read to find it, so you learn the shard is bad without downloading or loading the weights.

## 5. `check hf` — a healthy model on the Hugging Face Hub

**Model:** the same `mlx-community/Qwen2.5-0.5B-Instruct-4bit` as section 3, validated over the network instead of on disk. It reads repository metadata and the safetensors header; it does not download the weights. (`safetensors.offsets` passes with a note: the Hub doesn't expose the header length, so the data-section upper bound isn't checkable there, but the overlap and ordering checks still run.)

Over the Hub the `compat.mlx_signal` line picks up more than the on-disk run: the `mlx` tag, the `mlx` library metadata, and the `mlx-community` author all come from the repository listing.

```console
$ mlx-model-doctor check hf mlx-community/Qwen2.5-0.5B-Instruct-4bit
MLX Model Doctor: mlx-community/Qwen2.5-0.5B-Instruct-4bit

Summary:
  pass: 15
  warn: 1
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

WARN low text/memory.estimate
  Memory estimate: Estimated lower bound memory is advisory and may be below runtime use.
  Fix: Treat this as a floor; account for runtime overhead before loading the model.

PASS info text/safetensors.offsets
  Safetensors offsets: Safetensors tensor offsets are well-formed; the data-section upper bound was not checked (header length unavailable on this target).

PASS info text/weights.param_count
  Weight parameter count: The weight map resolves to present tensors with non-zero parameters.

PASS info text/weights.tied_embedding
  Tied embeddings: Embedding tying is consistent with the stored tensors.

PASS info text/quantization.shape
  Quantization shape: Quantized tensor shapes are consistent with config bits/group_size.
```

**Result:** exit code `0`. Same verdict as the local run, reached without downloading a single weight file.

## 6. `check hf` — catching a repo with no `config.json` (a LoRA adapter)

**Model:** `ybelkada/opt-350m-lora`, a LoRA adapter rather than a full model. It ships `adapter_model.safetensors` and `adapter_config.json` but no top-level `config.json`. Pointing the doctor at an adapter when you meant the base model is an easy mistake, and `mlx-lm` would crash partway through loading. The doctor stops you up front.

```console
$ mlx-model-doctor check hf ybelkada/opt-350m-lora
MLX Model Doctor: ybelkada/opt-350m-lora

Summary:
  pass: 1
  warn: 2
  fail: 2
  skip: 13

FAIL high text/files.required
  Required config file: Missing required config.json.
  Fix: Add config.json to the model repository.

FAIL high text/config.json
  Config JSON: Missing config.json; cannot validate model configuration.
  Fix: Add a valid config.json file to the model repository.

SKIP info text/config.model_type
  Model type: config.json is unavailable, so model_type cannot be checked.

PASS info text/compat.mlx_signal
  MLX compatibility signal: No MLX-compatibility signals found; this may not be an MLX/mlx-lm model.

WARN medium text/tokenizer.files
  Tokenizer files: No tokenizer artifacts were found.
  Fix: Add tokenizer.json or another tokenizer artifact to the model repository.

SKIP info text/tokenizer.special_tokens
  Special token IDs: config.json is unavailable, so special token IDs cannot be checked.

SKIP info text/chat_template.presence
  Chat template: No tokenizer metadata, so a chat template cannot be checked.

SKIP info text/chat_template.special_tokens
  Chat template tokens: No chat template string, so token consistency cannot be checked.

SKIP info text/safetensors.index
  Safetensors index: No safetensors index was found.

SKIP info text/quantization.metadata
  Quantization metadata: config.json is unavailable, so quantization metadata cannot be checked.

SKIP info text/quantization.mode
  Quantization mode: No MLX quantization object to validate.

SKIP info text/generation_config.tokens
  Generation-config tokens: No generation token IDs declared, so consistency cannot be checked.

SKIP info text/vlm.image_processor
  VLM image processor: Not a vision-language repo; skipped.

WARN low text/memory.estimate
  Memory estimate: Estimated lower bound memory is advisory and may be below runtime use.
  Fix: Treat this as a floor; account for runtime overhead before loading the model.

SKIP info text/safetensors.offsets
  Safetensors offsets: No safetensors header to scan.

SKIP info text/weights.param_count
  Weight parameter count: No safetensors header to check parameter counts.

SKIP info text/weights.tied_embedding
  Tied embeddings: No safetensors header or config to check embedding tying.

SKIP info text/quantization.shape
  Quantization shape: No MLX quantization metadata or safetensors header to check.
```

**Result:** exit code `1`. Two `FAIL`s — `files.required` and `config.json` — both pointing at the same root cause: there's no `config.json`, so this isn't a loadable model repository. With no config and no model weights to read, most downstream checks have nothing to inspect and `skip`. The fix is to point at the base model the adapter was trained on.

## 7. `check hf` — a stale, non-MLX repo (GGUF weights)

**Model:** `TheBloke/Llama-2-7B-GGUF`, a 2023-era GGUF conversion. It has a valid `config.json` (so the basic checks pass), but it carries no MLX quantization metadata and ships `.gguf` files instead of safetensors. This is the "right idea, wrong format" case: a real model, but not one `mlx-lm` will load.

```console
$ mlx-model-doctor check hf TheBloke/Llama-2-7B-GGUF
MLX Model Doctor: TheBloke/Llama-2-7B-GGUF

Summary:
  pass: 4
  warn: 2
  fail: 0
  skip: 12

PASS info text/files.required
  Required config file: config.json is present.

PASS info text/config.json
  Config JSON: config.json contains a valid JSON object.

PASS info text/config.model_type
  Model type: config.json declares model_type=llama.

PASS info text/compat.mlx_signal
  MLX compatibility signal: No MLX-compatibility signals found; this may not be an MLX/mlx-lm model.

WARN medium text/tokenizer.files
  Tokenizer files: No tokenizer artifacts were found.
  Fix: Add tokenizer.json or another tokenizer artifact to the model repository.

SKIP info text/tokenizer.special_tokens
  Special token IDs: pad_token_id or eos_token_id is unavailable in config.json.

SKIP info text/chat_template.presence
  Chat template: No tokenizer metadata, so a chat template cannot be checked.

SKIP info text/chat_template.special_tokens
  Chat template tokens: No chat template string, so token consistency cannot be checked.

SKIP info text/safetensors.index
  Safetensors index: No safetensors index was found.

WARN low text/quantization.metadata
  Quantization metadata: config.json does not contain quantization metadata.
  Fix: Add MLX top-level quantization metadata when the model is quantized.

SKIP info text/quantization.mode
  Quantization mode: No MLX quantization object to validate.

SKIP info text/generation_config.tokens
  Generation-config tokens: No generation token IDs declared, so consistency cannot be checked.

SKIP info text/vlm.image_processor
  VLM image processor: Not a vision-language repo; skipped.

SKIP info text/memory.estimate
  Memory estimate: Memory estimate skipped because of insufficient metadata.

SKIP info text/safetensors.offsets
  Safetensors offsets: No safetensors header to scan.

SKIP info text/weights.param_count
  Weight parameter count: No safetensors header to check parameter counts.

SKIP info text/weights.tied_embedding
  Tied embeddings: No safetensors header or config to check embedding tying.

SKIP info text/quantization.shape
  Quantization shape: No MLX quantization metadata or safetensors header to check.
```

**Result:** exit code `0` — nothing is corrupt, so there's no hard failure. But the report still tells the story: `compat.mlx_signal` finds no MLX signals, and every tensor-header check `skip`s because there's no safetensors to read (the weights are GGUF). If you were expecting an MLX repo, that cluster of skips plus the "not an MLX/mlx-lm model" line is your answer before you waste a download.

## 8. `sample hf` — survey an author's likely-MLX repos

Lists an author's repositories, keeps the ones that look like MLX models, and validates a deterministic sample. `--limit 10` over-fetches the listing so it checks ten MLX candidates even when some of the first listed repos aren't MLX. Each model is its own batch item; a per-model error is recorded and the run continues. The survey stays config-only — it doesn't fetch tensor headers per repo. A survey that matches no MLX repositories is a valid empty result and still exits `0`: finding nothing is informational, not a tool error.

```console
$ mlx-model-doctor sample hf --author mlx-community --limit 10
MLX Model Doctor HF Sample
Author: mlx-community
Task: any
Limit: 10
Plugin: text

Summary:
  checked: 10
  tool-error: 0

CHECKED mlx-community/Boogu-Image-0.1-Base-4bit
  Signal: tag:mlx
  Results: pass=1 warn=2 fail=2 skip=9

CHECKED mlx-community/DeepSeek-OCR-6bit
  Signal: tag:mlx
  Results: pass=11 warn=2 fail=0 skip=1

CHECKED mlx-community/DeepSeek-OCR-bf16
  Signal: tag:mlx
  Results: pass=9 warn=3 fail=0 skip=2

CHECKED mlx-community/DeepSeek-R1-4bit
  Signal: tag:mlx
  Results: pass=11 warn=1 fail=0 skip=2

CHECKED mlx-community/DeepSeek-R1-Distill-Llama-70B-8bit
  Signal: tag:mlx
  Results: pass=11 warn=1 fail=0 skip=2

CHECKED mlx-community/DeepSeek-V3.2-4bit
  Signal: tag:mlx
  Results: pass=8 warn=3 fail=0 skip=3

CHECKED mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit
  Signal: tag:mlx
  Results: pass=10 warn=2 fail=1 skip=1

CHECKED mlx-community/GLM-5-4bit
  Signal: tag:mlx
  Results: pass=10 warn=3 fail=0 skip=1

CHECKED mlx-community/GLM-5.2-4bit
  Signal: tag:mlx
  Results: pass=10 warn=3 fail=0 skip=1

CHECKED mlx-community/GLM-5.2-DQ4plus-q8
  Signal: tag:mlx
  Results: pass=10 warn=3 fail=0 skip=1
```

**Result:** exit code `0`. Ten MLX repos checked. Two carry hard failures: `Boogu-Image-0.1-Base-4bit` is an image model, so the `text` engine's checks fail on it, and `Devstral-Small-2-24B-Instruct-2512-4bit` trips one check. The survey still exits `0` because it records each model as its own batch item and reports per-repo results rather than gating on them; a per-model failure never fails the run. The `pass`/`warn`/`skip` spread across the rest reflects how complete each repo's metadata is. Add `--format json` or `--format markdown` to any `check` / `sample` command for machine-readable output, or `--format github` to a `check` command (next section) for GitHub Actions annotations.

## 9. `--format github` — annotations for CI

**Model:** any `check` run with `--format github`, shown here on a directory with no `config.json` so the failure annotations are visible. GitHub renders the `::error` / `::warning` lines as inline annotations on the changed files and the `::notice` line as a run summary. The [GitHub Action](README.md#use-it-in-ci) wraps this format for you.

```console
$ mlx-model-doctor check local ./model --format github
::error title=text/files.required::Required config file: Missing required config.json.
::error title=text/config.json::Config JSON: Missing config.json; cannot validate model configuration.
::warning title=text/tokenizer.files::Tokenizer files: No tokenizer artifacts were found.
::notice title=mlx-model-doctor::/path/to/model — pass=1 warn=1 fail=2 skip=14
```

**Result:** exit code `1`. Each failing or warning check becomes one annotation; passing and skipped checks stay quiet. Inside a GitHub Actions job the same run also writes the full Markdown report to the job summary and the `pass` / `warn` / `fail` / `skip` / `exit-code` / `schema-version` counts to the step outputs, so a later step can read them.
