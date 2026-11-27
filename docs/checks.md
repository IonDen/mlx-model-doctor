# What it checks

The built-in `text` plugin runs these against a model repository, broadly in this order:

`text` is the default plugin. For vision-language repositories, pass `--plugin vlm`
to run the explicit non-runtime VLM profile. The VLM plugin keeps the same metadata
and safetensors-header posture: no weights are downloaded and no model runtime is
loaded, unless you add `--smoke`.

- **Required files** — `config.json` is present and readable.
- **Config consistency** — `config.json` parses, and its `model_type` is set.
- **Tokenizer** — the tokenizer files a text model needs are present, and the special-token configuration is coherent.
- **Chat template** — a chat/instruct model declares a chat template (in `tokenizer_config.json`, `chat_template.json`, or a `chat_template.jinja`), and the end-of-turn token its template emits is a registered special token. A typo'd stop token loads fine and then never stops generating.
- **Safetensors index** — when the weights are sharded, `model.safetensors.index.json` is valid and every shard it references exists.
- **Tensor header** — read the safetensors header itself (the tensor map: dtypes, shapes, byte-offsets; no weight download) to catch a corrupt header (overlapping or out-of-bounds tensor offsets), a weight map that points at tensors no shard contains, a declared tied embedding that contradicts the stored weights, and an MLX-quantized layer whose packed-weight and scales shapes don't agree. These run by default; pass `--skip-weights` to skip them for a faster config-only pass.
- **Quantization metadata** — quantization fields are present and use a valid MLX mode with a valid group size and bit width (`affine`, `mxfp4`, `mxfp8`, `nvfp4`). This reads the metadata, not the tensors.
- **Generation tokens** — the `eos` / `pad` / `bos` token IDs are present and agree across `config.json`, `generation_config.json`, and `tokenizer_config.json`.
- **Memory budget** — an estimate of the memory the model needs at your context length, compared against a budget you pass with `--max-memory`.

Each check returns a result with a status (`pass` / `warn` / `fail` / `skip`), a message, and — when something is wrong — a remediation hint. The report aggregates them, and the process exit code reflects the worst result under your fail policy.
