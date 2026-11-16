# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.2] — 2026-11-17

The `[mlx-lm]` and `[mlx-vlm]` extras now follow MLX 0.32, mlx-lm 0.32 and
mlx-vlm 0.6.12 to 0.7. A text `--smoke` load that fails on a repository
naming `auto_map` code no longer puts your home folder path in the upstream
error kept in the report details. Nothing changes in the API, the CLI, the
checks or the output format.

### Changed
- The extras now require newer packages. `[mlx-lm]` asks for
  `mlx-lm>=0.32.0,<0.33` and `mlx>=0.32.4,<0.33` (it was `mlx-lm>=0.31.3,<0.32`
  and `mlx>=0.32,<0.33`). `[mlx-vlm]` asks for `mlx-vlm>=0.6.12,<0.8` and
  `mlx>=0.32.4,<0.33` (it was `mlx-vlm>=0.6.12,<0.7`). The runtime dependencies
  and the Python requirement are the same.
- mlx-lm 0.32.0 and later refuse a repository's own `model_file` unless the
  caller passes `trust_remote_code=True`. That is a second layer beside the
  refusal added in 0.10.1, which stays.
- With either extra installed, MLX is now 0.32.4 or later. Its release notes
  add 1-bit affine quantization on Apple GPUs, and the check accepts 1-bit from
  0.32.4 (0.32.0 to 0.32.3 reject it). The static check runs from the base
  install and cannot see which MLX will load the model, so a 1-bit repository
  checked in one environment and loaded on an older MLX in another can still
  fail at load.
- mlx-vlm 0.7.x no longer depends on mlx-lm. It brings more packages that only
  the extra installs: `datasets` (with pandas and pyarrow), `cryptography` and
  `websockets`. `fastapi`, `uvicorn` and `starlette` also came with mlx-vlm
  0.6.12.

### Fixed
- When a text `--smoke` load fails on a repository that names `auto_map` code,
  the upstream error kept in the report details (`upstream_error`) shows your
  home folder as `~`. This also holds when the user name has a space, when the
  home folder ends a quoted path, a sentence or a line, and when it appears
  percent-encoded in a `file://` or Hub URL. Other failed loads, text or VLM,
  still show the loader's own message, with its paths.
- `check hf --smoke --plugin vlm` now also downloads `*.jsonl` files, which one
  mlx-vlm 0.7.x processor family reads.

### Verified
On 2026-11-16, on this release's code (a clean tree), with mlx 0.32.4,
mlx-lm 0.32.2, mlx-vlm 0.7.12, transformers 5.15.0 and Python 3.13 on an Apple
M1 Max with 32 GB under macOS 27.2.1, these passed: the GPU-marked upstream
call tests and the source scans of mlx-vlm, a text `--smoke` on
Qwen2.5-0.5B-Instruct-4bit, and a VLM `--smoke` on Qwen2.5-VL-3B-Instruct-4bit.
Both `--smoke` runs exited 0.

The live VLM check ran three tests: the live VLM smoke test and two offline
snapshot-resolution tests. All three passed, and then the test process crashed
during teardown (exit 139).

At an earlier commit with the same extras and lock, `parity mlx` ran on a small
base and LoRA. The three parity runs returned their expected verdict codes (0
for pass, 2 for inconclusive, 1 for tracks-base) with no crash. One opt-in
parity test passed both of its tests and then ended with the same exit code
(139) during teardown. No traceback was kept, so a shared cause is not
confirmed. This was seen before 0.10.2 and is not fixed yet.

The lowest versions the extras admit (mlx-lm 0.32.0, mlx-vlm 0.6.12 and
mlx 0.32.4) were not run.

### Upgrade notes
- Installing an extra upgrades `mlx` to 0.32.4 or later and, for `[mlx-lm]`,
  `mlx-lm` to 0.32.0 or later.
- mlx-train-perf 0.9.1 pins `mlx==0.32.0` in its base install, and its own
  `[mlx-lm]` extra keeps `mlx-lm` below 0.32. Either way it cannot share an
  environment with these extras.

## [0.10.1] — 2026-11-16

`--smoke` and `parity mlx` now refuse to load a repository that names its own
loader code, or, on VLM `--smoke`, that ships a chat template for one of six
processor families. If you used `--smoke`, or `parity mlx` on a base or fused model you
do not control, upgrade.

### Security
- Some versions of the loaders that `--smoke` and `parity mlx` use can run
  Python code shipped inside a model repository while loading it: mlx-lm 0.30.6
  to 0.31.3 (the `[mlx-lm]` extra installs 0.31.3) and mlx-vlm 0.6.16 and later
  (the `[mlx-vlm]` extra is `>=0.6.12,<0.7`, so it reaches 0.6.16 and 0.6.17;
  0.7.12 is the newest release as of 2026-11-16). With such a repository, `--smoke` could
  run that code on your machine, and so could `parity mlx` for its base or
  fused model. These pins are the same in 0.10.1.
- The 0.10.0 README said the VLM smoke check refused remote code by default.
  That held only for the `trust_remote_code` flag, not for a repository that
  names a `model_file`.
- A refused smoke check is a failed check, so `--smoke` exits 1 under the
  default policy. A refused `parity mlx` run exits 2.
- Before any `--smoke` load, text or VLM, the tool now reads the model folder
  and refuses the load, with a plain message, in these cases:
  - Text `--smoke` and `parity mlx` (base and fused) load through mlx-lm. They
    refuse when `config.json` names a `model_file` at any depth, when
    `config.json` cannot be read, or when the model path is not a directory.
    The `--prompts` step loads the tokenizer through transformers with remote
    code turned off, and applies the same refusal before it does. Parity loads
    the tokenizer with remote code turned off, so an `auto_map` entry does not
    block it. Text `--smoke` also calls mlx-lm with
    remote code off for the tokenizer, and an `auto_map` there did not run
    code.
  - VLM `--smoke` (`--plugin vlm`) loads through mlx-vlm, and some mlx-vlm
    processors turn remote code back on. This path also refuses:
    - a `configuration_files` key in `config.json`;
    - an `auto_map` anywhere in `config.json`;
    - an `auto_map` anywhere in `tokenizer_config.json`;
    - an `auto_map` anywhere in `processor_config.json`;
    - an `auto_map` anywhere in `preprocessor_config.json`;
    - an `auto_map` anywhere in `video_preprocessor_config.json`;
    - any of those files that exists but cannot be read;
    - a chat template of the repository's own, when `model_type` in
      `config.json` (compared in lower case, or `speculators_model_type` when
      `model_type` is empty) is `molmo`, `molmo2`, `ernie4_5_moe_vl`, `phi3_v`
      (also written `phi3-v` or `phi3v`), `kimi_vl` or `locateanything`. These
      mlx-vlm processors render templates without a sandbox, which can let a
      template run code. This route was read in the mlx-vlm and transformers
      source; it has not been shown to run code. The list comes from mlx-vlm
      0.6.12 to 0.7.12; a later mlx-vlm could change it.
    - The tool looks for that template in `tokenizer_config.json`,
      `chat_template.json`, `processor_config.json`,
      `preprocessor_config.json`, `special_tokens_map.json` (a top-level
      `chat_template` key), `chat_template.jinja` and an
      `additional_chat_templates/` folder that holds a `.jinja` file. The check
      covers all of these files for every one of the six families. A template
      file that exists but cannot be read (for example a truncated download)
      also refuses the load, because the tool cannot tell whether the
      repository ships a template.
    - The message names the family and where the template was found, says the
      load was refused so the template was not rendered, and says the static
      checks still ran. A repository of these families with no template of its
      own still loads. Where mlx-vlm has a built-in template (molmo, molmo2) it
      is used; for the other four families such a repository can fail when the
      chat template is applied.
- The message reads: "This repository names loader code (`<key>` in `<file>`)
  that mlx-lm could import" (or mlx-vlm), then "The load was refused, so that
  code was not run." Smoke adds "The static checks above still ran." The advice
  is not to load the repository with mlx-lm or mlx-vlm unless you trust its
  author; the static checks read it without running its code. For a file that
  is unreadable or only partly downloaded, the message says the file could not
  be read and suggests downloading it again or repairing it.
- `parity mlx` and `check_adapter_parity` read the base and fused directories
  with the mlx-lm rule above before any model loads. On a hit, nothing is loaded; the
  report names the repository (base or fused), the key and the file, and says
  the load was refused so that code was not run. The run cannot reach a verdict,
  so it exits 2. A tokenizer mismatch or an uncovered target found in the same
  run still exits 1. The adapter directory is not checked, because mlx-lm reads
  only its config and weights.
- Who was exposed: users of `--smoke` (opt-in, needs an extra) on 0.10.0 and
  earlier who pointed it at a repository they do not control, and users of
  `parity mlx` with mlx-lm 0.30.6 to 0.31.3 who pointed the base or fused model at one.
  The static checks, which are the default, read only metadata and the safetensors header
  and are not affected.
- What to do: upgrade to 0.10.1. Until then, run `--smoke` and `parity mlx`
  only on repositories you trust.
- What this does not do: `--smoke` and `parity mlx` still load the model through
  mlx-lm or mlx-vlm. Outside the cases listed above, including chat templates
  in other model families, a repository is loaded as before, so run `--smoke`
  only on repositories you trust. The static checks remain the safe way to look
  at a repository you do not trust.

### Changed
- `check hf … --smoke` now downloads the repository snapshot first, without
  downloading `.py` files, checks it, and loads from that local copy. The smoke result
  records the snapshot revision.
- A failed snapshot download is now a tool error (exit 2). Before, it was a
  smoke failure (exit 1).
- When the Python API `check_adapter_parity` downloads a repository from the Hub
  (the `parity mlx` command never downloads), it now fetches only the
  files mlx-lm reads (JSON configs, safetensors weights, tokenizer files and
  chat templates) plus the adapter's config and weights.
- VLM `--smoke` now refuses any repository whose config files carry an
  `auto_map` entry, and some well-known vision repositories do. There is no
  option to override this. The static checks still run on them.
- VLM `--smoke` now also refuses repositories of the molmo, molmo2,
  ernie4_5_moe_vl, phi3_v, kimi_vl and locateanything families that ship their
  own chat template.
- On text `--smoke` and in parity, an `auto_map` does not block the load. But if
  the tokenizer needs that code to load, the load now fails with a message that
  names the file, instead of transformers' advice to turn remote code on.

No dependency changes in this release.

## [0.10.0] — 2026-11-13

`check local` and `parity mlx` now read a model straight from your Hugging Face
cache, and 1-bit affine quantization no longer draws a warning.

### Added
- `check local` and `parity mlx` accept a snapshot in the Hugging Face cache
  (`~/.cache/huggingface/hub/models--ORG--NAME/snapshots/<commit>/`), so you can
  check a model right after `hf download` without copying it first. Snapshots
  downloaded with huggingface_hub 1.x or 2.x are now read, including the shared
  store that 1.32 and later use for large files. Pass the snapshot folder
  itself, not a subfolder of it. The files in a
  snapshot are symlinks into the cache's own stored copies, and the tool follows
  a symlink only when it leads to one of those. Any other symlink that points
  outside the model directory is still refused. Before, a cache snapshot was
  reported as having files outside the model root, or missing.

### Changed
- 1-bit affine quantization is accepted as of MLX 0.32.4, so a repository that
  uses it no longer warns. MLX 0.32.0 through 0.32.3 reject it. The `[mlx-lm]`
  extra's `mlx>=0.32,<0.33` range already admits 0.32.4.

## [0.9.2] — 2026-10-24

The safetensors index check now tells a stale index apart from missing weights,
and `check hf` and `check local` agree on repositories with nested component
indexes.

### Fixed
- A stale safetensors index now gets a medium warning, not a failure. Some
  repositories ship a `model.safetensors.index.json` naming shards that are
  absent, next to other shards that ship
  (`mlx-community/Mistral-Small-3.1-24B-Instruct-2503-6bit`: 10 absent,
  4 shipped). The warning gives both counts and suggests regenerating or
  removing the index; the weight checks then read the shipped shards. A shard
  counts as shipped only if both loaders would take it: a top-level
  `model*.safetensors` other than `consolidated.safetensors`.
  `weights.safetensors` and `consolidated.safetensors` still fail, since
  mlx-lm's glob matches neither. No load was run. mlx-lm (0.31.3) globs
  `model*.safetensors` and never reads the index; mlx-vlm (0.6.12, 0.7.x) keeps
  the named shards that exist, else globs `*.safetensors` minus
  `consolidated.safetensors`. [mlx-vlm issue 624](https://github.com/Blaizzy/mlx-vlm/issues/624),
  fixed by [PR 638](https://github.com/Blaizzy/mlx-vlm/pull/638), has an opener
  who writes: "Model loads and works correctly in mlx-lm/mlx-vlm despite index
  mismatch."
- An index whose named shards are some or all missing, with no loader-eligible
  shard shipped, now produces a full report on `check hf`. `text/safetensors.index`
  and `text/safetensors.offsets` fail and the command exits 1. The index check
  names the first missing shard, and the full sorted list is in its details.
  Only the offsets message lists the absent shards: up to 8 names, then
  "and N more". It used to stop with a bare Hub 404 and exit 2.
  Other Hub errors, such as an auth failure or a dropped connection, still exit 2.
- `check local` and `check hf` now give the same result on a repository with a
  nested component index, such as `text_encoder/model.safetensors.index.json`.
  Both read only the root index for the header, as the Hub client does, so
  `check local` no longer warns `text/weights.param_count` about a component's
  tensors. The nested index is still validated by `text/safetensors.index`.
  On `check local`, a root index with some or all of its named shards missing
  and no loader-eligible shard shipped now reports a header error
  (`text/safetensors.offsets` fails and `text/weights.param_count` is skipped),
  which matches `check hf`. A stale index still reads its shipped shards, as
  in the first item.
- `check hf` no longer prints the `Parse safetensors files` progress bar on stderr.

### Behavior changes
On `check hf`, a stale index used to exit 2 by default (exit 1 with
`--skip-weights`). It now passes with a warning under the default `--fail-on error`
and exits 0. If your CI should still fail on it, use `--fail-on warn`. A
repository whose index names missing shards, with no loader-eligible shard
shipped, now exits 1 with a report where it used to exit 2 with a 404, so a script that treated exit 2 as "could not check"
will now see a failed check instead.

## [0.9.1] — 2026-09-12

A false-positive cleanup. Four checks were flagging valid, popular repositories;
each now passes, while the genuine problems they were meant to catch still fail.

### Fixed
- The safetensors index check no longer reports a false "missing shard" for a
  repository whose weight index sits in a subdirectory. An index at
  `text_encoder/model.safetensors.index.json` names its shards relative to its
  own directory, so `0.safetensors` there means `text_encoder/0.safetensors`.
  The check had instead looked for those names at the repository root and failed
  a valid repository (for example a diffusion model's text encoder). Shard names
  are now resolved relative to the index file, so a multi-component repository
  passes, while a genuinely missing shard still fails and is reported at its
  full path.
- The tied-embedding check no longer warns when `tie_word_embeddings` is absent
  from `config.json`. mlx-lm's model configs default that setting to true (llama
  and qwen2, for example), and tie-by-default families like Gemma ship no
  separate output head, so an MLX repo with the key absent loads fine and its
  absence is not an inconsistency. An explicit `tie_word_embeddings: false` with
  no stored output head still warns.
- The VLM image-token check now recognizes `[IMG]`, the image placeholder used
  by Pixtral and Mistral-3, so those repositories no longer warn that their
  image token is not an actual placeholder.
- The chat-template checks now read `chat_template.json`, the file transformers
  processors and mlx-vlm actually use. A repository that keeps its template
  there (common for vision-language models) is no longer reported as having no
  chat template, and the end-of-turn-token check reads the template the runtime
  will use rather than a stale copy in `tokenizer_config.json`.

## [0.9.0] — 2026-09-12

Adds an adapter-parity verifier: a way to check whether fusing a LoRA adapter
into a base model actually kept the behavior the adapter learned. Fusing is
supposed to be lossless, but with mlx-lm's default path it often is not, and
until now you had no cheap way to tell before shipping the fused model.

### Added
- `mlx-model-doctor parity mlx --base <B> --adapter <A> --fused <F>` and the
  `check_adapter_parity` Python API. It runs the same teacher-forced input
  through three models — the base alone, the base with the adapter loaded, and
  the fused model — and compares their top-token predictions. The verdict says
  whether the fused model tracks base+adapter (`pass`), fell back to the base
  and lost the adapter (`fail_tracks_base`), matches neither (`fail_gross`), or
  can't be told apart within measurement noise (`inconclusive`). Each model
  loads in its own memory-capped subprocess, one at a time, so the check is
  safe to run on a laptop.
- `parity mlx --prompts <file>` builds the comparison fixture from your own
  examples. Pass a JSON list of `{"prompt", "completion"}` pairs and the tool
  tokenizes them with the base model's own tokenizer and scores the completion
  tokens — so you check parity on text that matters for your model, not a
  stand-in. Without it, the built-in fixture only fits its reference tokenizer
  and a real model is refused with a message pointing you at `--prompts`.
- `parity.v1` JSON schema for `parity mlx --format json` (shipped in the wheel,
  validated against real output in the test suite), plus `text` and `markdown`
  renderers. Exit codes: `0` parity confirmed, `1` a determined regression
  (the fuse reverted to base, or a static incompatibility), `2` can't determine
  (inconclusive, a worker failure, or a setup error).
- Requires the optional `[mlx-lm]` extra for the runtime comparison, and a base
  repository that ships a `tokenizer.json`. Point `--base`/`--adapter`/`--fused`
  at directories of real files (a converted/fused model, or `hf download <repo>
  --local-dir ./dir`), not the raw Hugging Face cache snapshot path, whose
  symlinks the checker does not yet follow.

### Changed
- The verifier reports a degraded 4-bit fuse as `inconclusive` rather than a
  false `pass`. The reason it needs to: mlx-lm's default fuse re-quantizes the
  merged weights back to the base's format, and that round-trip erases part of
  the adapter's contribution. On a small text-to-SQL LoRA the default 4-bit fuse
  kept only about half the adapter's effect, while fusing with `--dequantize` (a
  float fuse) kept nearly all of it. If you fuse for distribution, prefer
  `--dequantize`, or run this check first.
- The optional `[mlx-lm]` extra now pins `mlx-lm>=0.31.3,<0.32` and
  `mlx>=0.32,<0.33` — the versions the parity check was verified against. This
  also backs the existing `--smoke` runtime check.

## [0.8.0] — 2026-08-11

A trust-and-depth release. Version-bound check tables now warn rather than
fail on a recognized-but-unlisted value, the project moves from Alpha to
Beta, and the VLM profile gets its first runtime smoke check alongside a
deeper `sample hf` survey.

### Added
- Version-aware check policy: version-bound allow-lists (MLX quantization
  modes, safetensors dtypes) are annotated with the upstream version they
  were verified against, and guarded by tests that pin the warn-not-fail
  behavior for an unrecognized-but-well-typed value. See the README's
  "Version sensitivity" section.
- VLM smoke check: `--smoke --plugin vlm` loads a vision-language model
  through `mlx-vlm` and generates from a dummy image, under the same
  advisory memory caps as the text smoke path. Remote code execution is
  refused by default (`trust_remote_code=False`). Requires the new
  `[mlx-vlm]` optional extra.
- `sample hf --max-candidates N`: configurable scan depth into an author's
  catalog, overriding the default fetch depth when set explicitly.
- `sample hf --signal-filter SIGNAL,...`: keep only candidates whose
  highest-priority MLX signal is in the given comma-separated list.
- `sample hf --no-cache` / `--cache-ttl SECONDS`: a local cache for Hugging
  Face listing responses, stored at `~/.cache/mlx-model-doctor/hf-listings/`
  with a default TTL of one hour (3600 seconds); `--no-cache` bypasses it.

### Changed
- An unrecognized MLX quantization mode now warns instead of failing. A
  repository using a mode added in a newer MLX release is flagged as
  unverified, not rejected outright.
- Development status promoted from Alpha to Beta.
- The `sample hf --format json` batch schema version moved from
  `sample-batch/1.0` to `sample-batch/1.1`, adding the new optional
  `max_candidates` and `signal_filter` fields.

## [0.7.0] — 2026-07-08

A reach/readiness minor release plus the first explicit non-text validation
profile. It adds a non-runtime `vlm` plugin for vision-language repositories
while keeping `text` as the default, preserving existing `text/...` check IDs,
leaving JSON schemas unchanged, and not adding runtime smoke/load behavior.

### Added
- Explicit `vlm` plugin (`--plugin vlm`) for vision-language repositories. It
  runs VLM-oriented metadata checks plus the existing no-download
  safetensors-header checks for `check local` / `check hf`.
- VLM image-token wiring check (`vlm/image_token.wiring`) for image placeholder
  consistency between config, tokenizer metadata, and chat templates.
- GitHub Action `plugin` input, defaulting to `text`, so CI can opt into `vlm`.
- A producer pre-upload workflow for Hugging Face publishing: validate local output
  with `check local --fail-on warn`, upload with `hf upload` or
  `upload_folder()`, then optionally verify the remote repo with `check hf`.
- An adoption scorecard process for release decisions, covering Action usage,
  pre-commit usage, generic mentions, PyPI downloads, repo metadata, and
  Marketplace discoverability.
- A profile-routing decision that preserves current `text` defaults and makes
  non-text validation profiles explicit opt-ins.

### Changed
- `plugins` now lists both built-ins: `text` and `vlm`.
- `sample hf --plugin vlm` requires an explicit VLM task filter such as
  `--task image-text-to-text`; automatic profile-aware sampling remains future
  work.
- The public roadmap no longer claims a wired docs-site milestone. README and
  EXAMPLES remain the source of truth until adoption evidence justifies a separate
  site.
- GitHub Action and pre-commit examples now pin the v0.7.0 release where a fixed
  version is shown.

## [0.6.2] — 2026-07-02

A correctness and hardening patch. One behavior change is worth knowing before you
upgrade: the memory estimate now passes a repository that fits and fails one that
exceeds `--max-memory`, where it used to always warn. That makes `--fail-on warn`
usable on a clean repo and lets `--max-memory` gate under the default policy, and it
means a memory result now affects the exit code where it never did before. The
`--smoke` pre-flight is unchanged, since it reads the estimate details rather than the
status, and nothing else changes the report shape, CLI flags, or exit-code policy.

### Added
- `--quiet` now drops the `pass` and `skip` lines from the text report while keeping the
  summary and any `warn`/`fail` lines. It was previously accepted and ignored.

### Changed
- The memory estimate reports `pass` when a repository is within budget (or when no
  `--max-memory` is set) and `fail` when its estimated lower bound exceeds the budget,
  instead of always warning.
- `zero_check_reason` now appears in the text and Markdown reports, not only in JSON.
- The weights check reports its stored-element total under a `stored_element_count`
  detail key, renamed from `total_parameter_count` because it counts bit-packed stored
  elements rather than logical parameters. Its tied-embedding message now reads "not
  enabled" instead of "not set", so it fits the case where the flag is explicitly false.
- `duration_s` and `environment` are documented as reserved output fields; they stay
  `null` and `{}` so the JSON stays stable to diff.

### Fixed
- A list-valued `eos_token_id` or `pad_token_id`, as recent Llama-family configs
  declare, no longer draws a false "should be integer" warning. The check still warns
  when a pad ID is also an end-of-sequence ID, and still flags genuinely malformed
  values.
- A local model directory aggregates only its canonical top-level (or index-named)
  safetensors shards, so nested component weights no longer merge into one header and a
  repository gets the same verdict on disk as it does from the Hub.
- The memory estimate no longer double-counts a repository that ships both
  `.safetensors` and PyTorch `.bin` copies of the same weights; it uses the safetensors
  set when both are present.
- Repository metadata is not read when its size is unknown, so an unsized file on the
  Hub path cannot trigger an unbounded download.
- The safetensors offset scan reports a gap before the first tensor.
- A check whose identifier is malformed can no longer crash the whole run through the
  crash-isolation path.

### Removed
- The unused `docs` dependency group.

## [0.6.1] — 2026-06-28

A hardening patch. A normal run is unchanged — same checks, report fields, CLI flags,
and exit codes. This finishes the stability-contract story for the batch output and
closes a release-time versioning trap.

### Added
- A published JSON Schema for the `sample hf` batch survey, shipped at
  `mlx_model_doctor/schema/sample-batch.v1.schema.json` and validated against real
  `--format json` output in CI. The batch already stamped a `sample-batch/1.0` version
  string; now there is an actual schema behind it. Each checked item embeds a full
  single-`check` report, which conforms to `report.v1.schema.json`.

### Changed
- `zero_check_reason` is now populated when a run produces no checks, with a message
  naming the plugin, instead of always being null. A zero-check run still exits `2`;
  the field just says why. A normal run leaves it null.

### Fixed
- The build no longer derives version "0" from the moving `v0` tag. `git describe` is
  restricted to full `vX.Y.Z` tags, with a `tag-pattern` backstop, so a `v0` that
  shares the release commit can't be picked instead of the real version tag — the
  failure that 400'd the v0.6.0 publish.

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
