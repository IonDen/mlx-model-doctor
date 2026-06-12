# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] — 2026-06-12

### Fixed
- The quantized-shape check (`text/quantization.shape`) no longer reports a
  load-blocking failure for valid mixed-precision models. An MLX `quantization`
  block can give individual layers their own `bits` and `group_size`; a common
  pattern is 4-bit experts alongside 8-bit dense, router, and gate layers. The
  check had applied the model-level values to every layer, so a model like
  `mlx-community/gpt-oss-20b-MXFP4-Q8` or an nvfp4 mixture-of-experts repository
  failed even though it loads fine. It now reads each layer's own values, and flags
  a layer whose bit width it cannot recognize as unverified instead of failing it.

## [0.4.0] — 2026-06-07

Two static checks, both metadata-only, added to the built-in `text` plugin.

### Added
- MLX-compatibility signal (`text/compat.mlx_signal`): a single `check local` or
  `check hf` now reports whether a repository looks like an MLX / mlx-lm model and
  which signals say so — an MLX `quantization` block, an `mlx-community` author,
  MLX tags or library metadata, quantized weights in the safetensors header, or an
  `mlx`/`4bit`/`8bit` name hint. It is informational and never fails the run. The
  same signal logic now backs the `sample hf` survey, so the single-repo and survey
  paths agree on what counts as MLX.
- Vision-language image-processor check (`text/vlm.image_processor`): a
  vision-language repository that declares no way to resolve an image processor —
  no `image_processor_type`, custom `auto_map`, feature extractor, or
  `processor_class` — is flagged before you load it, since the standard
  image-processor path may be unable to resolve it. Repositories that do declare a
  resolution path pass, and text-only repositories are skipped. Validated against
  live Qwen2.5-VL, InternVL3, and Qwen2-Audio repositories.

## [0.3.0] — 2026-06-06

Deep weight inspection: read the safetensors *header* — on the Hub over an HTTP
range request, still no weight download — to add four tensor-level checks that
JSON-only metadata can't see. They run by default on a single `check`; `sample
hf` stays a config-only survey.

### Added
- Safetensors offset scan (`text/safetensors.offsets`): the tensor byte-offsets
  in the header don't overlap and aren't out of bounds. A corrupt header fails
  at load; this catches it first. On a local file the data-section upper bound
  is checked too; on the Hub the header length isn't exposed, so the upper-bound
  check is skipped (and said so) while overlap and ordering still run.
- Weight-map parameter sanity (`text/weights.param_count`): every tensor the
  weight map references exists in a shard header, and the parameter count isn't
  zero — an internal consistency check, not a config-derived parameter recount.
- Tied-embedding consistency (`text/weights.tied_embedding`): a declared
  `tie_word_embeddings` matches which embedding and output-head tensors are
  actually stored. A declared-but-contradicted tie loads silently wrong.
- MLX quantized shape consistency (`text/quantization.shape`): each quantized
  layer's packed-weight and scales shapes agree with the config's bits and group
  size (`packed_last * 32 / bits == scales_last * group_size`). A mismatch won't
  load.
- A safetensors header reader: local targets parse the header off disk, Hugging
  Face targets fetch it through `huggingface_hub.get_safetensors_metadata` (a
  range request), with the tensor map exposed to checks as a shared, cached read.

### Changed
- The four tensor-header checks run by default on `check local` / `check hf`.
  The reserved `--include-weights` flag is replaced by an opt-out `--skip-weights`
  for a faster config-only pass. `sample hf` is unchanged (config-only).

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
