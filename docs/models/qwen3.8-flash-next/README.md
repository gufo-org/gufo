# Qwen3.8 Flash-Next

Hybrid recurrent/QSA mixture-of-experts text/image model on gfx1151.
Supported target: `unsloth/Qwen3.8-Flash-Next-GGUF`, **UD-Q4_K_XL** (four shards).
Optional shared-Q8 MTP predictor; optional BF16 vision projector.
Original unquantized-model and GGUF-conversion parity remain unqualified.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Load and run

```sh
nix develop -c hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision 38bb39ee97821de2c9009abb7e93950eec396e66 \
  --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" "mmproj-BF16.gguf" \
  --local-dir models/qwen3.8-flash-next
nix build
MODEL=/path/to/first-target-shard.gguf
MTP=/path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
./result/bin/gufo chat --model "$MODEL"
./result/bin/gufo serve llm --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --sessions 2 --context 32768
```

The loader discovers the remaining shards. Omit the speculative options for AR;
AR sessions allocate no predictor state even if a shared model has MTP loaded.
Adaptive MTP is default, with `--draft-tokens` capping 1–7 proposals. Sampled
requests use deterministic acceptance/cost control for seeded replay; all-greedy
C>1 batches may use measured cycle costs. Each request keeps private caches,
rollback and RNG. See [MTP qualification](QUALITY.md).

The official template defaults to thinking on, `xhigh` effort and preserving
prior reasoning. Use the [reasoning controls](../../SERVER.md#reasoning-controls)
for explicit effort/thinking overrides. Native context is 262144; YaRN extension
is unsupported. Memory grows with used context and selected rollback depth;
admission reserves the configured capacity before creating sessions.

### Memory-bounded expert cache

`serve llm --expert-cache <BYTES>` (e.g. `4G`, `512M`) caps the device bytes
held by the routed experts (issue #427). Without it — the default — all 512
experts per layer stay resident, exactly as before. With it, the layers in
the artifact's dominant per-expert size class (UD-Q4_K_XL: 43 of 48 trunk
layers at Q4_K gate/up + Q5_1 down, 2.93 MiB per expert) keep `slots` experts
in resident slabs sized from the budget, and read the rest through the OS
page cache from the mapped GGUF, like ds4's SSD expert streaming. A
per-layer LRU resolves each forward's routed ids on the host into slot
numbers, and the existing expert kernels run on the slabs with the same
weight bytes and reduction order: decode is bit-exact against the
fully-resident build for the same expert selections. Streamed layers run
the vector expert route at every pass width, so their prefill can round
differently from the resident build's tiled route (streamed layers'
prefill only; decode arithmetic is the same route). Size-class outliers (the UD Q8_0-down layers, the Q8_0
MTP block) stay fully resident. While streaming, decode graph replay is
off (routing resolves on the host); budgets covering all 512 experts are
rejected because full residency is then cheaper. Expect lower decode
throughput than the fully-resident build in exchange for far less resident
memory; load reports the plan under `--log-level=info`
(`event=expert_streaming layers=… slots_per_layer=…`).

## Images

Use this model's `mmproj-BF16.gguf`, discovered beside the target or selected
with `--mmproj`. PNG/JPEG CLI and HTTP requests use the
[same image interface](../qwen3.8-27b/README.md#images). Image state participates
in prefill, decoding, verification, multi-turn reuse and disk cache identity.
The predictor embeds shifted text IDs; visual information comes from target
hidden states and mRoPE. Image snapshots require matching prompt attachment.

## Tools and artifacts

Model tests are in `tests/models/qwen38_flash_next`, focused microbenchmarks in
`tools/qwen-flash`. [Quality](QUALITY.md) summarizes qualification and focused checks.
Build a microbenchmark with
`nix develop -c tools/bench/build.sh tools/qwen-flash/projection_plans.hip`;
`dense_blaslt_sweep.hip` times hipBLASLt on the dense prefill shapes.
No historical logit dump is required. New retained result summaries belong in
`artifacts/`; generated traces stay in the ignored top-level `artifacts/` tree.
