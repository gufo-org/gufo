# Ornith 1.5 122B

Sparse Mixture-of-Experts text and vision model (`qwen35moe`) on gfx1151, a
fine-tune of Qwen3.5-122B-A10B. 48 layers: every fourth layer is full
attention (12 layers), the other 36 are gated-delta-net recurrence. Attention
uses 32 query heads over 2 key-value heads at `head_dim` 256, so 16 query
heads share each KV head. Each MoE layer has 256 routed experts with 8 active
per token (expert width 1024) plus one shared expert (width 1024); the hidden
size is 3072. The GGUF carries one MTP layer that Gufo uses as a speculative
draft. Native context is 262144 tokens; 65536 is the longest context tested.

The tested GGUF is `Ornith-1.5-122B-Q4_K_M.gguf` (70.6 GiB). Most routed
experts are Q4_K, and the down projections of 25 layers are Q6_K. A loaded
server at context 65536 with MTP held about 85 GB of GTT.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Load and run

```sh
nix build
MODEL=path/to/Ornith-1.5-122B-Q4_K_M.gguf
./result/bin/gufo serve llm --model "$MODEL" --context 65536 --sessions 1 \
  --speculative mtp --mtp-model "$MODEL" --draft-tokens 2
./result/bin/gufo prompt --model "$MODEL" --prompt "Explain virtual memory." --temperature 0
```

Images need the projector, `mmproj-Ornith-1.5-122B-BF16.gguf`, passed with
`--mmproj`. It fits the existing Qwen vision tower at projection width 3072.

The model ships its own chat template. Gufo recognizes it by its SHA-256 and
renders it with a compiled implementation, as it does for the official Qwen
template.

## Modes

| Mode | Selection | Behavior |
| --- | --- | --- |
| AR | No speculative option | Target-only generation. |
| MTP | `--speculative mtp --mtp-model PATH` | Drafts with the GGUF's own MTP layer; pass the same file as the target. Two draft tokens decoded faster than three in our runs (see [Experiments](EXPERIMENTS.md)). |

Sampled requests draft too. Each draft token is drawn at the request's
temperature from the full head's top 64 candidates, and that exact
distribution goes to the verifier's rejection sampling, so replies keep the
target's distribution. Constrained, penalized and tool requests draft greedily
and are verified through the sampler.

## Kernel routes specific to this shape

- **Decode output projections.** The 48 per-token 8192 to 3072 Q4_K
  projections (gated-delta-net and attention outputs) run on the vendored
  MMVQ vector kernel. The same route serves batched decode rows.
- **Routed experts in decode.** Q4_K gate/up and down experts run on the
  vendored MMQ vector kernels. The 25 Q6_K down layers keep the per-slot GEMV.
- **Decode attention.** Split-K decode attention shares each FP16 KV tile
  across the 16 query heads of its KV head, for one row and for verification
  rows.

## Memory

Weights are uploaded to device memory when the heap has room above the
dispatch reserve. Kernel memory compaction never moves device memory, so
requests do not stall while the driver remaps pages. A loaded server uses
about 2.8 GB of host RSS. On a heap too small for the weights, the loader
falls back to a registered host copy.

An explicit `--cache-ram-bytes` keeps the RAM continuation cache small. On a
128 GB machine, 4 GiB avoided the slowdowns the automatic budget (about
21 GB here) caused once it filled.
