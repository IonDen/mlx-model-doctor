# Examples

Mostly real output from `mlx-model-doctor`, captured by running the tool — so you can see exactly what you get before installing it. Captured transcript sections show a command, its actual response, and a short read of what the result means. Command-only examples are marked.

> Most transcripts were captured with **mlx-model-doctor 0.9.0** on **2026-09-10**. Your venv paths will differ, and the Hugging Face examples (`check hf`, `sample hf`) are live snapshots of the Hub, so they drift over time — that's why they're dated. The two deliberately-broken repos in sections 5 and 6 (`ybelkada/opt-350m-lora` and `TheBloke/Llama-2-7B-GGUF`) are long-standing archival repos, picked because they keep failing the same way.

## Producer pre-upload workflow

Use this when you have just converted or exported an MLX model directory and want to publish it to the Hub.

```bash
mlx-model-doctor check local ./dist/my-mlx-model --fail-on warn
hf upload my-org/my-mlx-model ./dist/my-mlx-model --repo-type model
mlx-model-doctor check hf my-org/my-mlx-model --fail-on warn
```

**Result:** the local directory must have no warnings or failures before upload. The final `check hf` confirms that the uploaded repository exposes the expected files and metadata through the Hub path.

## Vision-language repository

```bash
mlx-model-doctor check hf mlx-community/InternVL3-2B-4bit --plugin vlm
```

```console
$ mlx-model-doctor check hf mlx-community/InternVL3-2B-4bit --plugin vlm
MLX Model Doctor: mlx-community/InternVL3-2B-4bit

Summary:
  pass: 16
  warn: 0
  fail: 0
  skip: 3

PASS info vlm/files.required
  Required config file: config.json is present.

PASS info vlm/config.json
  Config JSON: config.json contains a valid JSON object.

PASS info vlm/config.model_type
  Model type: config.json declares model_type=internvl_chat.

PASS info vlm/compat.mlx_signal
  MLX compatibility signal: MLX-compatibility signals: tag:mlx, author:mlx-community, config:quantization, weights:mlx-quant, repo-name.

PASS info vlm/tokenizer.files
  Tokenizer files: Tokenizer artifacts are present.

SKIP info vlm/tokenizer.special_tokens
  Special token IDs: pad_token_id or eos_token_id is unavailable in config.json.

PASS info vlm/chat_template.presence
  Chat template: A chat template is present.

PASS info vlm/chat_template.special_tokens
  Chat template tokens: Chat-template token literals are registered special tokens.

PASS info vlm/safetensors.index
  Safetensors index: Safetensors indexes reference shard files that are present.

PASS info vlm/quantization.metadata
  Quantization metadata: config.json contains MLX quantization metadata.

PASS info vlm/quantization.mode
  Quantization mode: MLX quantization mode 'affine' has valid group_size/bits.

PASS info vlm/image_processor
  VLM image processor: No image_processor_type, but the repo resolves an image processor via feature_extractor_type.

PASS info vlm/image_token.wiring
  VLM image token wiring: Custom processor path exposes runtime image-token wiring.

SKIP info vlm/generation_config.tokens
  Generation-config tokens: No generation token IDs declared, so consistency cannot be checked.

PASS info vlm/memory.estimate
  VLM memory estimate: Estimated VLM file-size lower bound is advisory; no memory budget was configured.

PASS info vlm/safetensors.offsets
  Safetensors offsets: Safetensors tensor offsets are well-formed; the data-section upper bound was not checked (header length unavailable on this target).

PASS info vlm/weights.param_count
  Weight parameter count: The weight map resolves to present tensors with non-zero parameters.

SKIP info vlm/weights.tied_embedding
  Tied embeddings: No recognized embedding or output-head tensor; cannot check tying.

PASS info vlm/quantization.shape
  Quantization shape: Quantized tensor shapes are consistent with config bits/group_size.
```

**Result:** exit code `0`. The `vlm` plugin runs a superset of checks — metadata, tokenizer, chat template, quantization, safetensors, plus VLM-specific image-processor and image-token-wiring checks. Use `sample hf --plugin vlm --task image-text-to-text` for VLM sampling.

## 1. `version` — environment and dependency status

```console
$ mlx-model-doctor version
mlx-model-doctor 0.9.0
Python: 3.14.5
Executable: /path/to/.venv/bin/python3
Virtualenv: /path/to/.venv
Dependencies:
  huggingface-hub: 1.19.0
  safetensors: 0.8.0
  mlx: 0.32.0
  mlx-lm: 0.31.3
  mlx-vlm: not installed
```

**Result:** exit code `0`. The version report shows the Python environment, the tool's own version, and each dependency it knows about. `safetensors` is the serialization format (the tool parses headers itself); `mlx` and `mlx-lm` are needed only for the optional `--smoke` runtime check.

## 2. `man` — usage and exit codes

```console
$ mlx-model-doctor man
mlx-model-doctor manual

Examples:
  mlx-model-doctor version
  mlx-model-doctor plugins
  mlx-model-doctor check local ./model
  mlx-model-doctor check hf mlx-community/Llama-3.2-3B-Instruct-4bit
  mlx-model-doctor sample hf --author mlx-community --limit 5
  mlx-model-doctor parity mlx --base <B> --adapter <A> --fused <F>
  mlx-model-doctor parity mlx --base <B> --adapter <A> --fused <F> --prompts <examples.json>

Exit codes:
  0: checks passed or informational command completed
  1: checks found failures under the selected fail policy
  2: tool error, bad target, missing dependency, or zero checks

parity mlx exit codes:
  0: PASS -- the fused model tracks the adapter-applied reference
  1: a determined defect (tokenizer mismatch, uncovered LoRA target,
     or a confirmed regression/gross-divergence verdict)
  2: a tool/setup error or a runtime cannot-determine outcome
```

**Result:** exit code `0`.

## 3. `check hf` — a healthy model on the Hugging Face Hub

**Model:** `mlx-community/Qwen2.5-0.5B-Instruct-4bit`. A clean quantized MLX model with a complete config, tokenizer, chat template, and safetensors weights.

The `text/compat.mlx_signal` line reports *why* the repo looks like an MLX model. Over the Hub it picks up the `mlx` tag, the `mlx` library, the `mlx-community` author, the `quantization` config block, the quantized weights, and the `4bit` in the name.

```console
$ mlx-model-doctor check hf mlx-community/Qwen2.5-0.5B-Instruct-4bit
MLX Model Doctor: mlx-community/Qwen2.5-0.5B-Instruct-4bit

Summary:
  pass: 16
  warn: 0
  fail: 0
  skip: 2

PASS info text/files.required
  Required config file: config.json is present.

PASS info text/config.json
  Config JSON: config.json contains a valid JSON object.

PASS info text/config.model_type
  Model type: config.json declares model_type=qwen2.

PASS info text/compat.mlx_signal
  MLX compatibility signal: MLX-compatibility signals: tag:mlx, library:mlx, author:mlx-community, config:quantization, weights:mlx-quant, repo-name.

PASS info text/tokenizer.files
  Tokenizer files: Tokenizer artifacts are present.

SKIP info text/tokenizer.special_tokens
  Special token IDs: pad_token_id or eos_token_id is unavailable in config.json.

PASS info text/chat_template.presence
  Chat template: A chat template is present.

PASS info text/chat_template.special_tokens
  Chat template tokens: Chat-template token literals are registered special tokens.

PASS info text/safetensors.index
  Safetensors index: Safetensors indexes reference shard files that are present.

PASS info text/quantization.metadata
  Quantization metadata: config.json contains MLX quantization metadata.

PASS info text/quantization.mode
  Quantization mode: MLX quantization mode 'affine' has valid group_size/bits.

PASS info text/generation_config.tokens
  Generation-config tokens: Generation token IDs are present and consistent.

SKIP info text/vlm.image_processor
  VLM image processor: Not a vision-language repo; skipped.

PASS info text/memory.estimate
  Memory estimate: Estimated lower bound memory is advisory and may be below runtime use.

PASS info text/safetensors.offsets
  Safetensors offsets: Safetensors tensor offsets are well-formed; the data-section upper bound was not checked (header length unavailable on this target).

PASS info text/weights.param_count
  Weight parameter count: The weight map resolves to present tensors with non-zero parameters.

PASS info text/weights.tied_embedding
  Tied embeddings: Embedding tying is consistent with the stored tensors.

PASS info text/quantization.shape
  Quantization shape: Quantized tensor shapes are consistent with config bits/group_size.
```

**Result:** exit code `0` — a clean MLX repo. The two `skip`s are checks that don't apply (no special-token IDs in this config; not a vision model). The `--skip-weights` flag drops the four tensor-header checks for a faster config-only pass.

## 4. `check local` — catching a corrupt safetensors before you load it

**Model:** a local directory whose `model.safetensors` has two tensors with overlapping byte ranges — a corrupt header that would blow up at load time. `mlx-model-doctor` reads only the tensor header (never the weight body) and flags it.

```console
$ mlx-model-doctor check local ./model
MLX Model Doctor: /path/to/model

Summary:
  pass: 6
  warn: 2
  fail: 1
  skip: 9

PASS info text/files.required
  Required config file: config.json is present.

PASS info text/config.json
  Config JSON: config.json contains a valid JSON object.

PASS info text/config.model_type
  Model type: config.json declares model_type=llama.

PASS info text/compat.mlx_signal
  MLX compatibility signal: No MLX-compatibility signals found; this may not be an MLX/mlx-lm model.

WARN medium text/tokenizer.files
  Tokenizer files: No tokenizer artifacts were found.
  Fix: Add tokenizer.json or another tokenizer artifact to the model repository.

SKIP info text/tokenizer.special_tokens
  Special token IDs: pad_token_id or eos_token_id is unavailable in config.json.

SKIP info text/chat_template.presence
  Chat template: No tokenizer metadata, so a chat template cannot be checked.

SKIP info text/chat_template.special_tokens
  Chat template tokens: No chat template string, so token consistency cannot be checked.

SKIP info text/safetensors.index
  Safetensors index: No safetensors index was found.

WARN low text/quantization.metadata
  Quantization metadata: config.json does not contain quantization metadata.
  Fix: Add MLX top-level quantization metadata when the model is quantized.

SKIP info text/quantization.mode
  Quantization mode: No MLX quantization object to validate.

SKIP info text/generation_config.tokens
  Generation-config tokens: No generation token IDs declared, so consistency cannot be checked.

SKIP info text/vlm.image_processor
  VLM image processor: Not a vision-language repo; skipped.

PASS info text/memory.estimate
  Memory estimate: Estimated lower bound memory is advisory and may be below runtime use.

FAIL high text/safetensors.offsets
  Safetensors offsets: Safetensors header has corrupt tensor offsets (overlap or out of bounds).
  Fix: Re-save the safetensors shard; its tensor offsets are inconsistent.

PASS info text/weights.param_count
  Weight parameter count: The weight map resolves to present tensors with non-zero parameters.

SKIP info text/weights.tied_embedding
  Tied embeddings: No recognized embedding or output-head tensor; cannot check tying.

SKIP info text/quantization.shape
  Quantization shape: No MLX quantization metadata or safetensors header to check.
```

**Result:** exit code `1` — the overlapping offsets are caught at the header level. On disk the tool knows the file size and header length, so it can verify offsets are in bounds. This model would crash at load time; `mlx-model-doctor` catches it before any download or GPU allocation.

## 5. `check hf` — catching a repo with no `config.json` (a LoRA adapter)

**Model:** `ybelkada/opt-350m-lora` — a LoRA adapter repo that ships adapter weights but no `config.json`. This is a known archival repo that keeps failing the same way.

```console
$ mlx-model-doctor check hf ybelkada/opt-350m-lora
MLX Model Doctor: ybelkada/opt-350m-lora

Summary:
  pass: 2
  warn: 1
  fail: 2
  skip: 13

FAIL high text/files.required
  Required config file: Missing required config.json.
  Fix: Add config.json to the model repository.

FAIL high text/config.json
  Config JSON: Missing config.json; cannot validate model configuration.
  Fix: Add a valid config.json file to the model repository.

SKIP info text/config.model_type
  Model type: config.json is unavailable, so model_type cannot be checked.

PASS info text/compat.mlx_signal
  MLX compatibility signal: No MLX-compatibility signals found; this may not be an MLX/mlx-lm model.

WARN medium text/tokenizer.files
  Tokenizer files: No tokenizer artifacts were found.
  Fix: Add tokenizer.json or another tokenizer artifact to the model repository.

SKIP info text/tokenizer.special_tokens
  Special token IDs: config.json is unavailable, so special token IDs cannot be checked.

SKIP info text/chat_template.presence
  Chat template: No tokenizer metadata, so a chat template cannot be checked.

SKIP info text/chat_template.special_tokens
  Chat template tokens: No chat template string, so token consistency cannot be checked.

SKIP info text/safetensors.index
  Safetensors index: No safetensors index was found.

SKIP info text/quantization.metadata
  Quantization metadata: config.json is unavailable, so quantization metadata cannot be checked.

SKIP info text/quantization.mode
  Quantization mode: No MLX quantization object to validate.

SKIP info text/generation_config.tokens
  Generation-config tokens: No generation token IDs declared, so consistency cannot be checked.

SKIP info text/vlm.image_processor
  VLM image processor: Not a vision-language repo; skipped.

PASS info text/memory.estimate
  Memory estimate: Estimated lower bound memory is advisory and may be below runtime use.

SKIP info text/safetensors.offsets
  Safetensors offsets: No safetensors header to scan.

SKIP info text/weights.param_count
  Weight parameter count: No safetensors header to check parameter counts.

SKIP info text/weights.tied_embedding
  Tied embeddings: No safetensors header or config to check embedding tying.

SKIP info text/quantization.shape
  Quantization shape: No MLX quantization metadata or safetensors header to check.
```

**Result:** exit code `1`. Two hard failures: no `config.json`. This is what you see when someone uploads adapter weights without the base model config — a LoRA adapter, not a standalone model.

## 6. `check hf` — a stale, non-MLX repo (GGUF weights)

**Model:** `TheBloke/Llama-2-7B-GGUF` — a GGUF-format model, not an MLX model. No MLX signals, no safetensors, no quantization metadata. The tool treats it as an unknown model format and reports what it can check.

```console
$ mlx-model-doctor check hf TheBloke/Llama-2-7B-GGUF
MLX Model Doctor: TheBloke/Llama-2-7B-GGUF

Summary:
  pass: 4
  warn: 2
  fail: 0
  skip: 12

PASS info text/files.required
  Required config file: config.json is present.

PASS info text/config.json
  Config JSON: config.json contains a valid JSON object.

PASS info text/config.model_type
  Model type: config.json declares model_type=llama.

PASS info text/compat.mlx_signal
  MLX compatibility signal: No MLX-compatibility signals found; this may not be an MLX/mlx-lm model.

WARN medium text/tokenizer.files
  Tokenizer files: No tokenizer artifacts were found.
  Fix: Add tokenizer.json or another tokenizer artifact to the model repository.

SKIP info text/tokenizer.special_tokens
  Special token IDs: pad_token_id or eos_token_id is unavailable in config.json.

SKIP info text/chat_template.presence
  Chat template: No tokenizer metadata, so a chat template cannot be checked.

SKIP info text/chat_template.special_tokens
  Chat template tokens: No chat template string, so token consistency cannot be checked.

SKIP info text/safetensors.index
  Safetensors index: No safetensors index was found.

WARN low text/quantization.metadata
  Quantization metadata: config.json does not contain quantization metadata.
  Fix: Add MLX top-level quantization metadata when the model is quantized.

SKIP info text/quantization.mode
  Quantization mode: No MLX quantization object to validate.

SKIP info text/generation_config.tokens
  Generation-config tokens: No generation token IDs declared, so consistency cannot be checked.

SKIP info text/vlm.image_processor
  VLM image processor: Not a vision-language repo; skipped.

SKIP info text/memory.estimate
  Memory estimate: Memory estimate skipped because of insufficient metadata.

SKIP info text/safetensors.offsets
  Safetensors offsets: No safetensors header to scan.

SKIP info text/weights.param_count
  Weight parameter count: No safetensors header to check parameter counts.

SKIP info text/weights.tied_embedding
  Tied embeddings: No safetensors header or config to check embedding tying.

SKIP info text/quantization.shape
  Quantization shape: No MLX quantization metadata or safetensors header to check.
```

**Result:** exit code `0`. No failures — the repo has a valid `config.json`, so the required-config check passes. The warnings are about missing tokenizer artifacts and no quantization metadata, which is expected for a GGUF model. The 12 skips reflect checks that can't run without safetensors weights or MLX metadata. Under `--fail-on warn` this would exit `1`.

## 7. `sample hf` — survey an author's likely-MLX repos

Lists an author's repositories, keeps the ones that look like MLX models, and validates a deterministic sample. `--limit 10` over-fetches the listing so it checks ten MLX candidates even when some of the first listed repos aren't MLX. Each model is its own batch item; a per-model error is recorded and the run continues. The survey stays config-only — it doesn't fetch tensor headers per repo. A survey that matches no MLX repositories is a valid empty result and still exits `0`.

```console
$ mlx-model-doctor sample hf --author mlx-community --limit 10
MLX Model Doctor HF Sample
Author: mlx-community
Task: any
Limit: 10
Plugin: text

Summary:
  checked: 10
  tool-error: 0

CHECKED mlx-community/Cydonia-24B-v3.1-4bit
  Signal: tag:mlx
  Results: pass=10 warn=1 fail=0 skip=3

CHECKED mlx-community/DeepSeek-OCR-8bit
  Signal: tag:mlx
  Results: pass=12 warn=1 fail=0 skip=1

CHECKED mlx-community/DeepSeek-R1-Distill-Llama-70B-4bit
  Signal: tag:mlx
  Results: pass=12 warn=0 fail=0 skip=2

CHECKED mlx-community/Hermes-4-70B-8bit
  Signal: tag:mlx
  Results: pass=11 warn=1 fail=0 skip=2

CHECKED mlx-community/Josiefied-Qwen3-1.7B-abliterated-v1-bf16
  Signal: tag:mlx
  Results: pass=10 warn=1 fail=0 skip=3

CHECKED mlx-community/Kimi-K2.5
  Signal: tag:mlx
  Results: pass=11 warn=2 fail=0 skip=1

CHECKED mlx-community/LFM2.5-1.2B-Instruct-4bit
  Signal: tag:mlx
  Results: pass=13 warn=0 fail=0 skip=1

CHECKED mlx-community/LFM2.5-1.2B-Thinking-8bit
  Signal: tag:mlx
  Results: pass=12 warn=1 fail=0 skip=1

CHECKED mlx-community/LFM2.5-2.6B-4bit
  Signal: tag:mlx
  Results: pass=12 warn=2 fail=0 skip=0

CHECKED mlx-community/Llama-3.2-3B-Instruct-bf16
  Signal: tag:mlx
  Results: pass=10 warn=1 fail=0 skip=3
```

**Result:** exit code `0`. Ten MLX repos checked, all clean. Add `--format json` or `--format markdown` to any `check` / `sample` command for machine-readable output, or `--format github` to a `check` command (next section) for GitHub Actions annotations.

## 8. `sample hf` with `--max-candidates` and `--signal-filter`

The `--max-candidates` flag controls how deep the survey scans into an author's catalog before picking the `--limit` MLX candidates. The `--signal-filter` flag keeps only candidates whose highest-priority signal matches the filter. Filtering on the highest-priority signal means a model with both `tag:mlx` and `library:mlx-lm` is matched by `--signal-filter tag:mlx` only.

```console
$ mlx-model-doctor sample hf --author mlx-community --limit 5 --signal-filter tag:mlx --max-candidates 500
MLX Model Doctor HF Sample
Author: mlx-community
Task: any
Limit: 5
Plugin: text
Max candidates: 500
Signal filter: tag:mlx

Summary:
  checked: 5
  tool-error: 0

CHECKED mlx-community/ALMA-13B-R-4bit-mlx
  Signal: tag:mlx
  Results: pass=10 warn=1 fail=0 skip=3

CHECKED mlx-community/ALMA-7B-R-4bit-mlx
  Signal: tag:mlx
  Results: pass=10 warn=1 fail=0 skip=3

CHECKED mlx-community/AlphaMonarch-7B-mlx
  Signal: tag:mlx
  Results: pass=10 warn=1 fail=0 skip=3

CHECKED mlx-community/AlphaMonarch-7B-mlx-4bit
  Signal: tag:mlx
  Results: pass=12 warn=0 fail=0 skip=2

CHECKED mlx-community/BB-L-01-7B-mlx-4bit
  Signal: tag:mlx
  Results: pass=10 warn=1 fail=0 skip=3
```

**Result:** exit code `0`. With `--max-candidates 500`, the survey scans deeper into the catalog (500 repos instead of the default window), and `--signal-filter tag:mlx` keeps only repos whose primary signal is the `mlx` tag. The deeper scan reaches earlier repos alphabetically (starting with ALMA instead of DeepSeek). Use `--no-cache` to bypass the listing cache, or `--cache-ttl` to change its TTL (default: 1 hour).

## 9. `parity mlx` — did fusing the adapter keep its behavior?

`parity mlx` compares three models on the same teacher-forced input — the base
alone, the base with the LoRA adapter loaded, and the fused model — and reports
whether the fused model still tracks base+adapter. `--prompts` builds the
comparison from your own `{"prompt", "completion"}` examples, tokenized with the
base model's own tokenizer. Point `--base`/`--adapter`/`--fused` at directories
of real files (a converted/fused model, or `hf download <repo> --local-dir`), not
the raw Hugging Face cache path.

A faithful fuse (`mlx_lm.fuse --dequantize`, which keeps the merged weights in
float) preserves the adapter — `pass`, exit `0`:

```console
$ mlx-model-doctor parity mlx \
    --base ./Qwen2.5-0.5B-Instruct-4bit \
    --adapter ./wikisql-lora \
    --fused ./qwen-fused-fp16 \
    --prompts ./examples.json
MLX Model Doctor (parity): ./wikisql-lora -> ./qwen-fused-fp16

Verdict: pass

Agreement:
  fused vs adapter (agree_fa): 0.9740
  fused vs base    (agree_fb): 0.6364
  gap:                         0.3377
  noise:                       0.0000
  first divergence:            267
  flip count:                  2
  adapter applied:             true

Failing checks:
  WARN [parity] parity/tokenizer.identity: The base and fused targets' tokenizer vocabularies match, but their special-tokens/chat-template metadata differs (special_tokens_differ); the parity oracle's fixed-id comparison is still valid and will run.

Delta map: 628 tensors (121 unchanged, 507 non_comparable)
$ echo $?
0
```

A `--dequantize` fuse turns every quantized weight into float, so a byte
comparison against the 4-bit base does not apply — every tensor is
`non_comparable`, and the summary is all you need here.

The default fuse re-quantizes the merged weights back to 4-bit, and that
round-trip erases part of the adapter. The fused model then tracks neither the
base nor base+adapter cleanly — `inconclusive`, exit `2` (parity not confirmed):

```console
$ mlx-model-doctor parity mlx \
    --base ./Qwen2.5-0.5B-Instruct-4bit \
    --adapter ./wikisql-lora \
    --fused ./qwen-fused-q4 \
    --prompts ./examples.json
MLX Model Doctor (parity): ./wikisql-lora -> ./qwen-fused-q4

Verdict: inconclusive

Agreement:
  fused vs adapter (agree_fa): 0.7792
  fused vs base    (agree_fb): 0.8182
  gap:                         0.3377
  noise:                       0.0000
  first divergence:            43
  flip count:                  17
  adapter applied:             true

Failing checks:
  WARN [parity] parity/tokenizer.identity: The base and fused targets' tokenizer vocabularies match, but their special-tokens/chat-template metadata differs (special_tokens_differ); the parity oracle's fixed-id comparison is still valid and will run.

Delta map: 628 tensors (336 changed, 292 unchanged)
  changed model.layers.10.mlp.down_proj.weight
  changed model.layers.10.self_attn.q_proj.weight
  changed model.layers.10.self_attn.v_proj.weight
  … (the summary is followed by the notable tensors, capped, then a "… (N more)" line)
$ echo $?
2
```

The base+adapter reference disagrees with the base on 33.8% of the scored
tokens (`gap`), so the tool can measure how much of that the fuse kept: the
`--dequantize` fuse holds 97% agreement with base+adapter, while the default
4-bit fuse drops to 78% — about halfway back toward the plain base. When a fuse
comes out `inconclusive` or worse, re-fuse with `--dequantize`.

## 10. `--format github` — annotations for CI

**Model:** any `check` run with `--format github`, shown here on a directory with no `config.json` so the failure annotations are visible. GitHub renders the `::error` / `::warning` lines as inline annotations on the changed files and the `::notice` line as a run summary. The [GitHub Action](README.md#use-it-in-ci) wraps this format for you.

```console
$ mlx-model-doctor check local ./model --format github
::error title=text/files.required::Required config file: Missing required config.json.
::error title=text/config.json::Config JSON: Missing config.json; cannot validate model configuration.
::warning title=text/tokenizer.files::Tokenizer files: No tokenizer artifacts were found.
::notice title=mlx-model-doctor::/path/to/model — pass=1 warn=1 fail=2 skip=14
```

**Result:** exit code `1`. Each failing or warning check becomes one annotation; passing and skipped checks stay quiet. Inside a GitHub Actions job the same run also writes the full Markdown report to the job summary and the `pass` / `warn` / `fail` / `skip` / `exit-code` / `schema-version` counts to the step outputs, so a later step can read them.
