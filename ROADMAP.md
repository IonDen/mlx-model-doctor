# mlx-model-doctor Roadmap

A non-binding sketch of where the library is headed. Items move between sections
as priorities change.

## Released

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

- **Deeper pre-load consistency checks.** A model can load cleanly and still
  misbehave at generation time. Planned: chat-template presence across both
  `tokenizer_config.json` and a sibling `chat_template.jinja`; EOS / end-of-turn
  token literals matching what the chat template uses (a one-character typo makes
  generation never stop); `generation_config.json` token-id sanity; and
  tied-embedding consistency (a declared-but-missing tied weight loads silently
  wrong).
- **Reading the safetensors header without downloading weights.** Fetching just
  the tensor map (over an HTTP range request) unlocks real dtype/shape and
  parameter-count checks, plus per-layer quantization-divisibility — all on the
  cheap, no-download path.
- **MLX-aware quantization validation.** Validate `group_size` / `bits` against
  what each MLX quantization mode actually allows (affine vs mxfp4 / mxfp8 /
  nvfp4), and flag a quantized layer whose last dimension isn't divisible by its
  group size — MLX hard-errors on that at load.
- **More check plugins beyond `text`** — vision-language, embedding, and other
  model families, each with its own ordered check list (the plugin protocol is
  already in place). A vision-language repo missing its image-processor metadata
  is the first concrete VLM check.
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
