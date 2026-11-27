# Stability policy

### Public API

The names you can depend on — only change on a major release:

`check_local_model`, `check_hf_model`, `CheckOptions`, `DoctorReport`, `CheckResult`, `render_json`, `render_text`, `render_markdown`, `render_github`, `exit_code_for`, `FailOn`, and the error types `ModelDoctorError`, `TargetError`, `DependencyError`, `MemorySafetyError`. `exit_code_for` raises `ValueError` on an unrecognized `fail-on` value.

### Internal layer

The check, plugin, and target Protocols; `CheckContext`; the plugin registry; and the `hub=` parameter on `check_hf_model` (a test injection seam whose type may change) are internal and not stable across releases.

### Schema versioning

`schema_version` is `MAJOR.MINOR`, versioned independently of the package. A minor bump adds new optional fields or new values to open fields (such as the memory check's `estimate_source` values). A major bump means a documented field was removed, renamed, or retyped, or a closed enum (`status`, `severity`, `source`) changed.

The top-level object, `summary`, and each entry in `results[]` are closed (`additionalProperties: false`), so a new field there is a coordinated schema edit plus a minor version bump. Validate against the schema that matches the payload's `schema_version`, not a pinned older copy — otherwise a newer payload's added field will fail your validator.

### Promoted `details` keys

`details` is otherwise free-form, but three keys from the memory check are stable across the 1.x schema line: `lower_bound_bytes`, `estimate_source`, and `memory_lower_bound_kind`. `lower_bound_bytes` is a structural lower bound — it counts attention, MLP, and embedding parameters at ≤16-bit weights (or quantized-equivalent) plus KV cache, but excludes norms, biases, and an untied `lm_head`. It sits below real runtime use and is not a fit guarantee.

### Batch output

The `sample hf --format json` survey has its own published schema, at `mlx_model_doctor/schema/sample-batch.v1.schema.json` and validated against real output in CI. It carries a `schema_version` of `sample-batch/MAJOR.MINOR` (same bump rules as above), and each checked item embeds a full single-`check` report that conforms to `report.v1.schema.json`.
