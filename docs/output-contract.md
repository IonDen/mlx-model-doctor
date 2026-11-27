# Output contract

`--format json` prints a stable, versioned payload. The top-level fields are:

| Field | Type | Description |
|---|---|---|
| `schema_version` | string | Schema major.minor version (e.g. `"1.0"`), independent of the package version. |
| `target` | string | The model path or repo ID that was checked. Text, Markdown and GitHub output show a local model's folder name instead; a Hugging Face target keeps its repo id, and a cache snapshot shows the repo id and the snapshot's short commit. |
| `source` | `"local"` or `"hf"` | Where the model came from. |
| `plugin` | string | The check plugin that ran (e.g. `"text"`). |
| `summary` | object | Check counts: `pass`, `warn`, `fail`, `skip` (integers). |
| `environment` | object | Reserved and currently always empty (`{}`); kept empty so JSON output stays stable to diff across environments. |
| `zero_check_reason` | string or null | When a run produced no checks (a zero-check run, which exits `2`), a message naming the responsible plugin; null on a normal run. |
| `results` | array | One entry per check; see below. |

JSON `target`, and the `path` and `original_ref` fields of `parity mlx --format json`, keep the full local path, home folder included, so check JSON output before you share it.

Each result in `results[]` has:

| Field | Type | Description |
|---|---|---|
| `check_id` | string | Namespaced identifier, e.g. `"text/files.required"`. |
| `title` | string | Short human-readable check name. |
| `status` | string | `"pass"`, `"warn"`, `"fail"`, or `"skip"`. |
| `severity` | string | `"info"`, `"low"`, `"medium"`, or `"high"`. |
| `message` | string | What was found. |
| `remediation` | string or null | What to do if the check fired. |
| `details` | object | Open object with check-specific key/value pairs. |
| `duration_s` | number or null | Reserved; currently always `null`. Per-check timing is not emitted so JSON output stays stable to diff run-to-run. |

The machine-readable schema ships with the package at `mlx_model_doctor/schema/report.v1.schema.json` and is validated against real output in CI.

Exit codes: `0` checks passed under the fail policy, `1` failures found, `2` a tool error (a bad target, a missing dependency, or zero checks run). `--format github` reports the same results as GitHub Actions annotations; inside a workflow it also writes the Markdown report to the job summary and the `pass` / `warn` / `fail` / `skip` / `exit-code` / `schema-version` values to the step outputs.
