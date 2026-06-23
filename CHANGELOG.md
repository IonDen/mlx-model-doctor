# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] — 2026-06-23

The `--format json` output and the public Python API now come with a versioned,
documented stability contract. No checks, report fields, CLI flags, or exit codes
changed; this release writes down and pins what was already there, and makes the
stable API importable directly from the package.

### Added
- A versioned JSON Schema for the report, shipped inside the package at
  `mlx_model_doctor/schema/report.v1.schema.json` and validated against real
  `--format json` output in CI. It pins the payload shape and the
  `status` / `severity` / `source` enums. The `details` and `environment` objects
  stay open, so a check can add a key without a breaking change.
- A stability policy in the README: which Python API names you can depend on,
  which internals may still change, the `schema_version` bump rules, and the
  promoted `details` keys the `--smoke` memory gate reads.
- `CheckOptions`, the report renderers (`render_json`, `render_text`,
  `render_markdown`, `render_github`), `exit_code_for`, and `FailOn` are now
  exported from the package root, so `check_local_model` / `check_hf_model` and the
  types they use can all be imported from `mlx_model_doctor` directly.

## [0.5.2] — 2026-06-20

A robustness patch. The optional `--smoke` memory pre-flight is the only behavior
change; the rest is internal test and code hardening that leaves the checks, the
report, the CLI, and the exit codes untouched.

### Fixed
- The `--smoke` memory pre-flight no longer trusts an understated memory estimate.
  When a repository has no usable `config.json` and one or more weight files report
  no size (a real case for some Hugging Face repos), the estimate falls back to
  summing the file sizes it can read — a partial figure that is lower than the true
  weight total. That partial sum was being used as the gate's lower bound, so an
  over-budget smoke load could slip through on a number that was too low. The partial
  figure is still reported for context, but the gate now ignores it and lets the
  memory-capped load decide. Fully-measured repositories are unaffected. This mirrors
  the v0.4.3 mixed-precision fix, applied to the no-config fallback path.

## [0.5.1] — 2026-06-18

Dependency housekeeping. Nothing about the checks or the report output changed;
this release only adjusts what gets installed and which Python versions are tested.

### Changed
- The `huggingface-hub` floor is now `>=1.0`. The tool is built and tested against
  the 1.x line, so the old `>=0.24` floor described a setup that was never tested.

### Removed
- `safetensors` is no longer a runtime dependency. The validator reads the
  safetensors header straight from the file bytes and gets Hugging Face metadata
  through `huggingface-hub`, so it never imported the `safetensors` package.
  Installs are a little lighter. If you have it anyway (for example via the
  `[mlx-lm]` extra), the `version` command still reports it.

### Added
- Python 3.14 is now tested in CI and listed in the package classifiers.

## [0.5.0] — 2026-06-15

The integration on-ramp: run the validator in other people's CI and pre-commit,
not just by hand.

### Added
- GitHub Action (`IonDen/mlx-model-doctor@v0`): a composite action that runs the
  static checks on `ubuntu-latest` (no weights, no GPU), writes the report to the
  job summary, sets `pass` / `warn` / `fail` / `skip` / `exit-code` /
  `schema-version` step outputs, and fails the job under your fail policy. Inputs
  mirror the CLI (`source`, `target`, `fail-on`, `max-memory`, `context-length`,
  `skip-weights`, `version`); pin the installed release with `version: "==0.5.0"`.
- pre-commit hook (`id: mlx-model-doctor`): runs `check local` on a model
  directory you keep in git, with an overridable `args` for the path.
- `--format github`: emits GitHub Actions annotations — one `::error` or
  `::warning` per failing or warning check, plus a `::notice` summary. Inside a
  workflow it also appends the Markdown report to `$GITHUB_STEP_SUMMARY` and the
  counts to `$GITHUB_OUTPUT`.
- A documented output contract: `--format json` carries a `schema_version`
  (`1.0`), the `summary` counts, and a `results` array of frozen check records;
  the exit codes (`0` pass, `1` failures, `2` tool error or zero checks) are
  fixed. See the README "Output contract" section.

## [0.4.3] — 2026-06-14

### Fixed
- The memory estimate now accounts for mixed-precision quantization. A model can
  give individual layers their own bit width, such as 4-bit experts alongside
  8-bit dense, router, and head layers. The estimate had applied the model-level
  bit width to every weight, so it underreported the memory the model needs.
  It now takes the weight figure from the measured weight-file sizes, which
  already reflect each layer's precision, and adds the context-length KV-cache
  term. If the file sizes can't all be read, it reports the estimate as
  unverified rather than a number that is too low, so the optional `--smoke`
  preflight no longer lets through a load that won't fit. Single-precision models
  are unaffected.
- The source distribution no longer bundles local working-tree files. The sdist
  is built from an explicit list of what belongs in it (the package, the tests,
  and the README, license, and changelog files), so a local build can't pull in
  editor or tool state. The published wheel was already limited to the package.

## [0.4.2] — 2026-06-13

### Fixed
- The quantization-mode check (`text/quantization.mode`) now validates every
  layer, not just the model-level default. An MLX `quantization` block can give
  individual layers their own `mode`, `bits`, and `group_size` — a mixed-precision
  model often pairs 4-bit experts with 8-bit dense, router, and gate layers. The
  check had read only the top-level values, so a broken per-layer entry slipped
  through unnoticed. It now resolves each layer's own values and checks them
  against the MLX table: an unknown per-layer mode fails, and an off-table or
  otherwise invalid value warns. Valid mixed-precision models still pass. This is
  the companion to the v0.4.1 shape-check fix.

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
