# mlx-model-doctor Roadmap

A non-binding sketch of where the library is headed. Items move between sections
as priorities change.

## Released

- **v0.1.0** (2026-06-04) — initial public release. Static validation for local
  model repositories; the Hugging Face target (`check hf`, `sample hf`); the
  optional memory-safe `mlx-lm` smoke check; text / JSON / Markdown reports; the
  `text` check plugin.

## Active

(Empty.)

## Future (no fixed release target)

- **More check plugins beyond `text`.** The plugin protocol is in place; vision
  and embedding model families would each get their own ordered check list.
- **Richer Hugging Face surveying.** Caching, broader candidate signals, and
  pagination so `sample hf` can cover more of an author's catalog.
- **A docs site.** The `mkdocs-material` dependency group is wired, but there is
  no published site yet.

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
