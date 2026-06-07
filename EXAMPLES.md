# Examples

Real output from `mlx-model-doctor`, captured by running the tool — so you can see exactly what you get before installing it. Each block shows a command and its actual response.

> Captured with **mlx-model-doctor 0.4.0** on **2026-06-07**. Your venv paths will differ, and the Hugging Face examples (`check hf`, `sample hf`) are live snapshots of the Hub — they change over time, which is why they're dated.

## 1. `version` — environment and dependency status

```console
$ mlx-model-doctor version
mlx-model-doctor 0.4.0
Python: 3.13.12
Executable: /path/to/.venv/bin/python3
Virtualenv: /path/to/.venv
Dependencies:
  huggingface-hub: 1.17.0
  safetensors: 0.7.0
  mlx: not installed
  mlx-lm: not installed
```

Exit code `0`. Static checks need only `huggingface-hub` and `safetensors`; `mlx` / `mlx-lm` appear once the optional `[mlx-lm]` extra is installed (for the `--smoke` check).

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

## 3. `check local` — validate a real model on disk

The everyday case: you've downloaded a model and want to confirm it before loading. This is `mlx-community/Qwen2.5-0.5B-Instruct-4bit` on disk — the same model checked over the Hub in section 5. Locally the tool can also verify the tensor offsets are in bounds (it knows the file size and the header length), so `safetensors.offsets` passes outright rather than with the Hub's "upper bound not checked" note.

The `text/compat.mlx_signal` line reports why the repo looks like an MLX model. On disk it sees the `quantization` block, the quantized weights in the header, and the `4bit` name; the `mlx-community` author and the MLX tags are Hub metadata, so those extra signals show up in the `check hf` run (section 5), not here.

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

Exit code `0`. The `--skip-weights` flag drops the four tensor-header checks for a faster config-only pass.

## 4. `check local` — catching a corrupt safetensors before you load it

This directory's `model.safetensors` has two tensors whose byte ranges overlap — a corrupt header that would fail at load. `mlx-model-doctor` reads only the tensor header (not the weights) and flags it. The `compat.mlx_signal` line is informational and stays a `pass` even when nothing marks the repo as MLX; the failure comes from `safetensors.offsets`. Exit code `1`.

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

Only the tensor header is read to find this — never the weight body — so the corruption is caught without loading the model.

## 5. `check hf` — a healthy model on the Hugging Face Hub

The same model as section 3, validated over the network instead of on disk. It reads repository metadata and the safetensors header — it does not download the weights. The tensor-header checks confirm the quantized layer shapes, the tied embedding, and the parameter map. (`safetensors.offsets` passes with a note: the Hub doesn't expose the header length, so the data-section upper bound isn't checkable there — the overlap and ordering checks still run.)

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

Exit code `0`.

## 6. `sample hf` — survey an author's likely-MLX repos

Lists an author's repositories, keeps the ones that look like MLX models, and validates a deterministic sample. `--limit 5` over-fetches the listing so it checks five MLX candidates even when the first listed repos aren't all MLX. Each model is its own batch item; a per-model error is recorded and the run continues. The survey stays config-only (it doesn't fetch tensor headers per repo).

```console
$ mlx-model-doctor sample hf --author mlx-community --limit 5
MLX Model Doctor HF Sample
Author: mlx-community
Task: any
Limit: 5
Plugin: text

Summary:
  checked: 5
  tool-error: 0

CHECKED mlx-community/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-mlx-8bit
  Signal: tag:mlx
  Results: pass=11 warn=2 fail=0 skip=1

CHECKED mlx-community/LFM2.5-8B-A1B-MLX-8bit
  Signal: tag:mlx
  Results: pass=11 warn=2 fail=0 skip=1

CHECKED mlx-community/LocateAnything-3B-4bit
  Signal: tag:mlx
  Results: pass=11 warn=2 fail=0 skip=1

CHECKED mlx-community/MiniCPM5-1B-OptiQ-4bit
  Signal: tag:mlx
  Results: pass=10 warn=3 fail=0 skip=1

CHECKED mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit
  Signal: tag:mlx
  Results: pass=11 warn=2 fail=0 skip=1
```

Exit code `0`. (Add `--format json` or `--format markdown` to any `check` / `sample` command for machine-readable output.)
