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

## Full Q8

The loader also reads the `Q8_0` checkpoint (six shards, about 188 GB), whose
per-layer token embedding table is `Q8_0` rather than IQ4_NL or BF16; the
table stays host-side and is decoded row by row as before. The checkpoint does
not fit one 128 GB host, so it needs two-host tensor parallelism (TP2).
SHA-256 of the files it was qualified with:

| Artifact | SHA-256 |
|---|---|
| `Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf` | `2dabcbb53ca537a7947bc7d20414fd464eeaf4d66d43021b5b2556cc87544ad2` |
| `Qwen3.8-Flash-Next-Q8_0-00002-of-00006.gguf` | `494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e` |
| `Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf` | `34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20` |
| `Qwen3.8-Flash-Next-Q8_0-00004-of-00006.gguf` | `bfa634025fabbd2658bf7694bc80b90e571699c768723f844c934c7ef06c691a` |
| `Qwen3.8-Flash-Next-Q8_0-00005-of-00006.gguf` | `232a8f14cc0fa4262e7efe8593774b136fe40909e39c7a020342ddaa27259a97` |
| `Qwen3.8-Flash-Next-Q8_0-00006-of-00006.gguf` | `538a93bca918064983409a41187ad4c68640f9aced6f29564da8f551bf86d7a5` |

See [Quality](QUALITY.md#full-q8) for how it compares with Q4 and llama.cpp.

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
