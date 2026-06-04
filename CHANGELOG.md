# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
