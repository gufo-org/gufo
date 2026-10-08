# Gemma 4 26B-A4B

Google's Gemma 4 26B-A4B IT, a mixture-of-experts model, in Unsloth's GGUFs:
`unsloth/gemma-4-26B-A4B-it-GGUF`, **gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf**
(15.8 GiB), **gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf** (21.7 GiB) and
**gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf** (25.7 GiB). The
tokenizer and chat template match the [31B](../gemma-4-31b/README.md), and the
same Gemma 4 engine serves both. Every mode, option and serving feature
described there applies here. Images use this repository's BF16 vision
sidecar (`mmproj-BF16.gguf`, the 31B's tower with a 2816-wide projection),
found beside the model or passed with `--mmproj`; `--image-tokens` works as
on the 31B ([VISION.md](../gemma-4-31b/VISION.md)).

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Architecture

- **Layers:** 30, hidden width 2816, 16 query heads. Sliding layers use 8 KV
  heads of 256; every sixth layer is global with 2 KV heads of 512 and no V
  projection.
- **Feed-forward:** each layer runs a dense 2112-wide GeGLU MLP beside 128
  routed experts. Eight experts per token, each a fused 2 × 704 gate/up
  projection plus a 704-wide down projection.
- **Router:** normalizes the attention residual, then selects the top eight
  of a softmax. Their weights are renormalized and multiplied by a
  per-expert output scale.

Every quant keeps attention, the dense MLP and the tied vocabulary head in
Q8_0 (about 73% of the bytes a token reads), except that UD-Q8_K_XL stores
all of layer 29 in BF16. They differ otherwise only in the experts:

| Quant | Gate/up experts | Down experts |
| --- | --- | --- |
| UD-Q4_K_XL | Q4_K (layer 29: Q5_K) | Q5_1 (layer 29: Q8_0) |
| UD-Q6_K_XL | Q6_K (layer 29: Q8_0) | Q8_0 |
| UD-Q8_K_XL | Q8_0 (layer 29: BF16) | Q8_0 (layer 29: BF16) |

BF16 tensors are rewritten as binary16 on upload: every BF16 value from 2^-17
to 65504 in magnitude is exactly a binary16 value, and the load fails if one
lies beyond the binary16 range.

By its header UD-Q4_K_M uses the same formats; it has not been run.

## Load and run

```sh
nix develop -c hf download unsloth/gemma-4-26B-A4B-it-GGUF \
  --revision c099eb48e663fd284577b04978a94ffccb261841 \
  --include gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf mtp-gemma-4-26B-A4B-it.gguf \
  mmproj-BF16.gguf \
  --local-dir models/gemma-4-26b-a4b
nix build
MODEL=models/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf
MTP=models/gemma-4-26b-a4b/mtp-gemma-4-26B-A4B-it.gguf
./result/bin/gufo chat --model "$MODEL"
./result/bin/gufo prompt --model "$MODEL" --image photo.jpg -p "Describe it."
./result/bin/gufo serve llm --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --context 131072
```

The MTP drafter is the repository's Q8_0 `gemma4-assistant`,
`mtp-gemma-4-26B-A4B-it.gguf`, and serves both quants. A 131072 context needs
about 2.3 GiB of KV and rings; the global layers keep few KV heads.

## How it runs

- **Decode and verification:** FP32 activations. Routed experts go through
  Gemma's grouped routed GEMV: each selected expert's weights are read once
  for every row routed to it. A row's arithmetic does not depend on the
  batch, so greedy MTP output equals single-token decoding. The dense MLP
  and the experts run on separate streams.
- **Prefill:** binary16 activations. Q8_0 projections go through a binary16
  WMMA GEMM, and experts through routed binary16 GEMMs over rows compacted
  by expert. Q8_1 activations (the 31B's prefill format and llama.cpp's)
  cost this model mean KL 0.74 against FP32 on a chat prompt; binary16 costs
  4e-5 ([Quality](QUALITY.md)).
- **MTP:** the calibrated draft policy uses this model's own cost table.
  Every verified row adds the experts it routes to, so verification grows
  faster per row than on the dense 31B.
