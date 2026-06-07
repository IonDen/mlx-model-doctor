# mlx-model-doctor Roadmap

A non-binding sketch of where the library is headed. Items move between sections
as priorities change.

## Released

- **v0.4.0** (2026-06-07) — single-repo MLX-compatibility signal and a
  vision-language image-processor check. `check local` / `check hf` now report
  whether a repository looks like an MLX model and why; a vision-language repo
  missing its `image_processor_type` is caught before load, while text-only repos
  are skipped.
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

- **More check plugins beyond `text`** — vision-language, embedding, and other
  model families, each with its own ordered check list (the plugin protocol is
  already in place). The v0.4.0 vision-language image-processor check runs gated
  inside the `text` plugin today; a dedicated `vlm` plugin would group it with
  further vision-language checks.
- **Richer Hugging Face surveying** — caching, broader candidate signals, and
  pagination so `sample hf` can cover more of an author's catalog.
- **A docs site** — the `mkdocs-material` group is wired, but there's no
  published site yet.

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
