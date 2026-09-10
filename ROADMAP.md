# mlx-model-doctor Roadmap

A non-binding sketch of where the library is headed. Items move between sections
as priorities change.

## Released

- **v0.9.0** (2026-09-10) — the adapter-parity verifier. A new `parity mlx`
  command and `check_adapter_parity` API check whether fusing a LoRA adapter
  into a base model kept the behavior the adapter learned, by comparing the
  top-token predictions of the base, the base with the adapter loaded, and the
  fused model on the same teacher-forced input. `--prompts` builds the
  comparison fixture from your own examples, tokenized with the base model's own
  tokenizer. Each model loads in its own memory-capped subprocess. The work
  surfaced a real gotcha: mlx-lm's default fuse re-quantizes the merged weights
  and loses part of the adapter, so a 4-bit fuse can quietly behave worse than
  the adapter it was built from — fuse with `--dequantize`, or run the check
  first. New `parity.v1` JSON schema.
- **v0.8.0** (2026-08-11) — trust and depth. Version-bound check tables (MLX
  quantization modes, safetensors dtypes) now warn instead of failing on a
  recognized-but-unlisted value, each annotated with the upstream version it
  was verified against, and the project's development status moves from
  Alpha to Beta. The VLM profile gets its first runtime check: `--smoke
  --plugin vlm` loads a model through `mlx-vlm` and generates from a dummy
  image under advisory memory caps, refusing remote code execution by
  default. `sample hf` can now scan deeper into an author's catalog
  (`--max-candidates`), filter by listing-visible MLX signal
  (`--signal-filter`), and cache Hub listing responses locally (`--no-cache`
  / `--cache-ttl`). The batch schema moves to `sample-batch/1.1` for the new
  optional fields.
- **v0.7.0** (2026-07-08) — reach/readiness plus first explicit VLM breadth.
  Keeps `text` defaults, schemas, exit codes, and runtime smoke/load behavior
  unchanged while adding `--plugin vlm` for non-runtime VLM metadata and
  safetensors-header validation. It also adds a producer pre-upload workflow for
  Hugging Face publishing, verifies the GitHub Marketplace Action listing,
  introduces an adoption-scorecard process for release decisions, records the
  profile-routing decision, and removes the stale docs-site milestone from the
  active 1.0 path.
- **v0.6.2** (2026-07-02) — a correctness and hardening patch. The memory estimate now
  passes a repository within budget and fails one that exceeds `--max-memory`, instead
  of always warning, so `--fail-on warn` works on a clean repo and `--max-memory` gates
  under the default policy. `--quiet` drops the pass/skip lines from the text report; a
  list-valued `eos_token_id` no longer draws a false warning; a local repository
  aggregates only its canonical safetensors shards to match the Hub path; plus a batch
  of smaller fixes. The unused `docs` dependency group is gone.
- **v0.6.1** (2026-06-28) — a hardening patch. The `sample hf` batch survey now has
  its own published JSON Schema (validated in CI), so both the single report and the
  batch output are under the contract. `zero_check_reason` is populated when a run
  produces no checks, naming the plugin, instead of always being null. And the build
  ignores the moving `v0` tag when deriving its version, so a `v0` on the release
  commit can no longer make it publish as version "0". A normal run is unchanged — same
  checks, report fields, CLI flags, and exit codes.
- **v0.6.0** (2026-06-23) — a versioned, documented stability contract for the
  JSON output and the public API. The `--format json` report now has a published
  JSON Schema (shipped in the package, validated in CI) and a written policy for
  what can change and when; `CheckOptions`, the report renderers, `exit_code_for`,
  and `FailOn` are exported from the package root. No checks, report fields, CLI
  flags, or exit codes changed.
- **v0.5.2** (2026-06-20) — a robustness patch. The optional `--smoke` memory
  pre-flight no longer treats a partial weight-size sum as its lower bound (the case
  where a repository has no usable config and one or more weight files report no
  size), so an over-budget smoke load can't slip through on a too-low estimate. The
  rest is internal test and code hardening with no change to the checks or the report.
- **v0.5.1** (2026-06-18) — dependency housekeeping. `safetensors` is no longer a
  runtime dependency (the validator parses the header itself), the `huggingface-hub`
  floor moved to the 1.x line it's tested against, and Python 3.14 joined the test
  matrix. No change to the checks or the report.
- **v0.5.0** (2026-06-15) — the integration on-ramp. A GitHub Action
  (`IonDen/mlx-model-doctor@v0`) and a pre-commit hook (`id: mlx-model-doctor`) run
  the static checks in CI and on every commit; a `--format github` output reports
  results as GitHub Actions annotations, with the Markdown report in the job summary
  and the counts in the step outputs. The JSON output and exit codes are now
  documented as a contract to build on — a `schema_version` field and fixed
  exit-code semantics.
- **v0.4.3** (2026-06-14) — the memory estimate now reflects mixed-precision
  quantization. When a model gives some layers a different bit width (4-bit
  experts with 8-bit dense, router, and head layers), the weight figure comes
  from the stored file sizes, which already account for each layer's precision,
  rather than from the model-level bit width alone; when the sizes can't all be
  read the estimate is marked unverified instead of too low. The source
  distribution is also built from an explicit file list, so a local build no
  longer pulls in working-tree state.
- **v0.4.2** (2026-06-13) — the quantization-mode check now validates per-layer
  overrides, not just the model-level default. A mixed-precision model that gives
  some layers their own `mode`/`bits`/`group_size` (4-bit experts with 8-bit dense,
  router, and gate layers) has each override checked against the MLX table, so a
  broken per-layer entry is surfaced instead of slipping past. Companion to the
  v0.4.1 shape-check fix.
- **v0.4.1** (2026-06-12) — fix the quantized-shape check reporting a false
  failure on valid mixed-precision models. MLX records per-layer `bits` and
  `group_size` overrides (4-bit experts with 8-bit dense and router layers); the
  check now resolves each layer's own values instead of the model-level ones, so
  models like `gpt-oss-20b-MXFP4-Q8` and nvfp4 mixture-of-experts repositories pass.
- **v0.4.0** (2026-06-07) — single-repo MLX-compatibility signal and a
  vision-language image-processor check. `check local` / `check hf` now report
  whether a repository looks like an MLX model and why; a vision-language repo
  that declares no image-processor resolution path is flagged before load, while
  text-only repos are skipped.
- **v0.3.0** (2026-06-06) — deep weight inspection. Reads the safetensors header
  (no weight download; on the Hub over a range request) to add four tensor-level
  checks: safetensors offset corruption, weight-map parameter sanity,
  tied-embedding consistency, and MLX quantized-layer shape consistency. They run
  by default; the reserved `--include-weights` flag became an opt-out
  `--skip-weights`.
- **v0.2.0** (2026-06-05) — static correctness expansion. Config-level checks for
  chat-template presence, end-of-turn token consistency, generation token IDs, and
  MLX quantization modes; size-bounded reads of untrusted metadata; a `sample hf
  --limit` over-fetch fix.
- **v0.1.0** (2026-06-04) — initial public release. Static validation for local
  model repositories; the Hugging Face target (`check hf`, `sample hf`); the
  optional memory-safe `mlx-lm` smoke check; text / JSON / Markdown reports; the
  `text` check plugin.

## Active

(Empty.)

## Future (no fixed release target)

Direction informed by a survey of how MLX / Hugging Face repos actually break
(2026-06-04). The theme: catch more of the "loads fine, then fails or generates
garbage" class of problems statically, before a load attempt.

- **Post-VLM hardening** — VLM-aware sampling and further VLM-specific
  checks as evidence demands.
- **Richer Hugging Face surveying** — broader candidate signals and
  survey-level diagnostics beyond the v0.8.0 caching and signal filter.
- **Documentation expansion** — README and EXAMPLES remain the source of truth for
  now. A separate docs site stays deferred until the adoption scorecard shows docs
  discovery or reference depth is the release bottleneck.

## Out of scope (deliberate non-goals)

- **Running or serving models.** `mlx-model-doctor` validates a repository;
  `mlx-lm` and `mflux` run it. The optional `--smoke` check is a minimal
  load-and-generate probe, not a runtime or a server.
- **Repairing repositories.** It reports problems and suggests fixes; it does not
  rewrite `config.json`, re-shard weights, or edit the repo.

Re-opening an out-of-scope item requires evidence that the original reasoning no
longer holds.

---

By Denis Ineshin · [ineshin.space](https://ineshin.space)
