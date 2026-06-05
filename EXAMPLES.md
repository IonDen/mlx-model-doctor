# Examples

Real output from `mlx-model-doctor`, captured by running the tool — so you can see exactly what you get before installing it. Each block shows a command and its actual response.

> Captured with **mlx-model-doctor 0.2.0** on **2026-06-05**. Your venv paths will differ, and the Hugging Face examples (`check hf`, `sample hf`) are live snapshots of the Hub — they change over time, which is why they're dated.

## 1. `version` — environment and dependency status

```console
$ mlx-model-doctor version
mlx-model-doctor 0.2.0
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

Exit code `0`.

## 3. `check local` — validate a local model directory

This directory is deliberately broken: it has a `config.json` but no tokenizer, and a `model.safetensors.index.json` that points at a shard which isn't there. The report flags both, skips the checks it can't run, and confirms the quantization mode is valid.

```console
$ mlx-model-doctor check local ./model
MLX Model Doctor: /path/to/model

Summary:
  pass: 5
  warn: 2
  fail: 1
  skip: 4

PASS info text/files.required
  Required config file: config.json is present.

PASS info text/config.json
  Config JSON: config.json contains a valid JSON object.

PASS info text/config.model_type
  Model type: config.json declares model_type=llama.

WARN medium text/tokenizer.files
  Tokenizer files: No tokenizer artifacts were found.
  Fix: Add tokenizer.json or another tokenizer artifact to the model repository.

SKIP info text/tokenizer.special_tokens
  Special token IDs: pad_token_id or eos_token_id is unavailable in config.json.

SKIP info text/chat_template.presence
  Chat template: No tokenizer metadata, so a chat template cannot be checked.

SKIP info text/chat_template.special_tokens
  Chat template tokens: No chat template string, so token consistency cannot be checked.

FAIL high text/safetensors.index
  Safetensors index: Safetensors index model.safetensors.index.json references invalid or missing shard model-00001-of-00002.safetensors.
  Fix: Add the missing safetensors shard or fix the index weight_map.

PASS info text/quantization.metadata
  Quantization metadata: config.json contains MLX quantization metadata.

PASS info text/quantization.mode
  Quantization mode: MLX quantization mode 'affine' has valid group_size/bits.

SKIP info text/generation_config.tokens
  Generation-config tokens: No generation token IDs declared, so consistency cannot be checked.

WARN low text/memory.estimate
  Memory estimate: Estimated lower bound memory is advisory and may be below runtime use.
  Fix: Treat this as a floor; account for runtime overhead before loading the model.
```

Exit code `1` — a `fail` result under the default `--fail-on error` policy. A healthy repo exits `0`.

## 4. `check hf` — validate a model on the Hugging Face Hub

Reads repository metadata over the network; it does not download the weights. This healthy instruct model passes the chat-template, quantization-mode, and generation-token checks.

```console
$ mlx-model-doctor check hf mlx-community/Qwen2.5-0.5B-Instruct-4bit
MLX Model Doctor: mlx-community/Qwen2.5-0.5B-Instruct-4bit

Summary:
  pass: 10
  warn: 1
  fail: 0
  skip: 1

PASS info text/files.required
  Required config file: config.json is present.

PASS info text/config.json
  Config JSON: config.json contains a valid JSON object.

PASS info text/config.model_type
  Model type: config.json declares model_type=qwen2.

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

WARN low text/memory.estimate
  Memory estimate: Estimated lower bound memory is advisory and may be below runtime use.
  Fix: Treat this as a floor; account for runtime overhead before loading the model.
```

Exit code `0`.

## 5. `sample hf` — survey an author's likely-MLX repos

Lists an author's repositories, keeps the ones that look like MLX models, and validates a deterministic sample. `--limit 5` over-fetches the listing so it checks five MLX candidates even when the author's first listed repos aren't all MLX. Each model is reported as its own batch item; a per-model error is recorded and the run continues.

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

CHECKED mlx-community/LFM2.5-8B-A1B-MLX-8bit
  Signal: tag:mlx
  Results: pass=10 warn=2 fail=0 skip=0

CHECKED mlx-community/LocateAnything-3B-4bit
  Signal: tag:mlx
  Results: pass=9 warn=2 fail=0 skip=1

CHECKED mlx-community/MiniCPM5-1B-OptiQ-4bit
  Signal: tag:mlx
  Results: pass=9 warn=3 fail=0 skip=0

CHECKED mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
  Signal: tag:mlx
  Results: pass=9 warn=2 fail=0 skip=1

CHECKED mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16
  Signal: tag:mlx
  Results: pass=5 warn=3 fail=0 skip=4
```

Exit code `0`. (Add `--format json` or `--format markdown` to any `check` / `sample` command for machine-readable output.)
