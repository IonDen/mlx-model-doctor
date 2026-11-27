# Check a fused adapter kept its behavior

Fusing a LoRA adapter into a base model is supposed to bake the adapter's
learned behavior into the weights. It doesn't always. With mlx-lm's default
fuse the merged weights get re-quantized back to the base's format, and that
round-trip can erase a good part of what the adapter learned — so the fused
model quietly behaves more like the plain base than like the model you
fine-tuned. `parity mlx` catches that before you ship the fused model.

```bash
mlx-model-doctor parity mlx \
  --base ./Qwen2.5-0.5B-Instruct-4bit \
  --adapter ./my-lora \
  --fused ./my-fused-model \
  --prompts ./examples.json
```

It runs the same teacher-forced input through three models — the base alone,
the base with the adapter loaded, and the fused model — and compares their
top-token predictions. If the fused model tracks base+adapter the verdict is
`pass`; if it fell back to the base and lost the adapter, `fail_tracks_base`; if
it matches neither, `fail_gross`; and if the difference is within measurement
noise, `inconclusive`. A default 4-bit fuse that only kept half the adapter lands
at `inconclusive` — parity not confirmed — rather than a misleading pass.
The exit code is `0` for a confirmed pass, `1` for a determined regression, and
`2` when the run can't decide (inconclusive or a setup problem).

`--prompts` takes a JSON list of `{"prompt", "completion"}` examples and builds
the comparison fixture from them, tokenized with the base model's own tokenizer
and scored over the completion tokens — so you measure parity on text that
matters for your model. Use examples the adapter should have changed. The check
needs the optional `[mlx-lm]` extra, a base repository that ships a
`tokenizer.json`, and loads each model in its own memory-capped subprocess (one
at a time), so it's safe on a laptop. If your fuse comes out `inconclusive` or
worse, re-fuse with `--dequantize` (a float fuse), which preserves the adapter.

Point `--base`, `--adapter`, and `--fused` at model directories: a locally
converted or fused model, a directory fetched with `hf download <repo>
--local-dir ./dir`, or a snapshot in the Hugging Face cache (see
[Checking a model in your Hugging Face cache](https://github.com/IonDen/mlx-model-doctor/blob/master/README.md#checking-a-model-in-your-hugging-face-cache)).

`parity mlx` and `check_adapter_parity` load the base and fused models through mlx-lm, so they apply the same refusal as text `--smoke` to those two directories. The `--prompts` step loads the tokenizer through transformers with remote code turned off, and applies the same refusal before it does. If either one names a `model_file` in `config.json` at any depth, or `config.json` cannot be read, or the path is not a directory, nothing is loaded. Parity loads the tokenizer with remote code turned off, so an `auto_map` does not block it. But if the tokenizer needs that code to load, the load fails with a message that names the file, instead of transformers' advice to turn remote code on. The report names the repository (base or fused), the key and the file, and the run exits `2`. A tokenizer mismatch or an uncovered target found in the same run still exits `1`. The adapter directory is not checked this way, because mlx-lm reads only its config and weights. The `parity mlx` command never downloads. When the Python API `check_adapter_parity` downloads a repository from the Hub, it fetches only the files mlx-lm reads (JSON configs, safetensors weights, tokenizer files and chat templates) plus the adapter's config and weights.

`parity mlx` also accepts `--format {text,json,markdown}`; the `json` output
follows the published `parity.v1` schema.
