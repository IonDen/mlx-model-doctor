# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-06-05

Static correctness expansion: four config-level checks that catch the "loads
fine, then crashes at generation or fails at MLX convert" class of problems,
without downloading weights.

### Added
- Chat-template presence (`text/chat_template.presence`): a chat/instruct model
  declares a chat template in `tokenizer_config.json` or a sibling
  `chat_template.jinja`. A missing template only crashes at `apply_chat_template`
  time, never at load.
- Chat-template token consistency (`text/chat_template.special_tokens`): the
  end-of-turn token the template emits is a registered special token. A one-
  character typo in a stop token loads fine and then never stops generating.
- Generation token IDs (`text/generation_config.tokens`): `eos` / `pad` / `bos`
  IDs are present and agree across `config.json`, `generation_config.json`, and
  `tokenizer_config.json`.
- MLX quantization mode (`text/quantization.mode`): validates the quantization
  mode and its group size and bit width against what MLX accepts (`affine`,
  `mxfp4`, `mxfp8`, `nvfp4`). An unknown mode is a hard failure, since MLX
  rejects it at convert or load.
- Size-bounded reads: untrusted metadata files are checked against a size cap
  before they are read, so a malicious or corrupt repo cannot make the tool pull
  a huge file into memory.

### Fixed
- `sample hf --limit N` over-fetches before filtering, so it checks up to `N`
  MLX candidates even when an author's listing leads with non-MLX repos
  (best-effort within a capped window).
- `_positive_device_bytes` no longer treats a boolean device value as a byte
  count.
- README quantization wording no longer implies tensor-level validation; the
  quantization checks are config-level.
- The release workflow uses a Node 24 build of `actions/download-artifact`, ahead
  of the GitHub Node 20 sunset.

## [0.1.0] — 2026-06-04

Initial public release.

### Added
- Static validation for local model repositories (`check local <path>` /
  `check_local_model`): config presence and consistency, tokenizer files and
  special tokens, safetensors-index integrity, quantization metadata, and a
  context-length-aware memory-budget estimate. The static checks read repository
  metadata only — no MLX or GPU, and no weight download.
- Hugging Face target (`check hf <repo_id>` / `check_hf_model`): the same checks
  against a Hub repository, reading metadata over `huggingface-hub`. Auth,
  not-found, and rate-limit failures surface as a clear tool error.
- `sample hf`: survey an author's likely-MLX repositories and validate a
  deterministic sample as a batch report. A per-model error is recorded as a
  batch item and the run continues; a listing failure is a tool error.
- Optional memory-safe `mlx-lm` smoke check (`--smoke`, `mlx-lm` extra): loads
  the model under an MLX wired-memory cap and refuses to load if the cap cannot
  be installed, so a smoke run can't push the machine into a memory panic.
- Reports render to text, JSON, and Markdown; results are frozen dataclasses, so
  output is stable to diff. Exit codes: `0` pass, `1` fail-under-policy,
  `2` tool error or zero checks — tunable with `--fail-on`.
- `version`, `man`, and `plugins` commands; the built-in `text` plugin; a
  `py.typed` marker.
