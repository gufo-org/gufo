# Nemotron 3.5 Bolt 74B-A9B

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

**Status: text serving in development.** Gufo runs the text trunk on ROCm
with its own tokenizer and chat template, serves it over the OpenAI-compatible
API (reasoning, tool calls, structured output, streaming, continuation cache)
and through `gufo prompt` / `gufo chat`, optionally with NVIDIA's LiLiCorr
speculative drafter. Up to 8 concurrent requests decode in one shared pass.
Not implemented yet: image and video input and the persistent disk cache.

NVIDIA Nemotron 3.5 Bolt is a 74B-parameter (about 9B active) hybrid of 40
Mamba-2 blocks, 40 LatentMoE blocks and 8 attention blocks, compressed from
Nemotron 3.5 Super VL. Weights use the OpenMDW-1.1 license, separate from
Gufo's MIT license.

## Weights

Gufo targets NVIDIA's quantization-aware 24GB GGUF only. It is the variant
NVIDIA's accuracy figures describe; community quantizations of the BF16
checkpoint use different tensor formats and are not supported.

| File | Revision | Size | SHA-256 |
| --- | --- | ---: | --- |
| [`NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/blob/e0fecb69cb3e5e4c5e8a056fa2aec98c78e91246/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf) | `e0fecb69` | 22.02 GB | `91d63363e782c8a3d5f4be42816eba8e0d42dd2530bd07a7d0d2dc440599af75` |
| [`mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/blob/e0fecb69cb3e5e4c5e8a056fa2aec98c78e91246/mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf) (vision, not used yet) | `e0fecb69` | 1.73 GB | `0ee891eadc7bf0cae6289b249c74eb1ef420534351f6ef8b64d49f0ca6a926b0` |
| [`NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF/blob/182d2987939e12f458298b5d5f6ef98912f4ad6c/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf) (LiLiCorr DFlash drafter) | `182d2987` | 1.00 GB | `080e1ea41c9562a7804dbf30c6fbd4d17e216c6b8ff666f88e6edb1816010194` |

The trunk stores routed and shared experts as IQ2_XXS, Mamba projections as
NVFP4 with a per-tensor F32 scale, and attention, latent projections,
embedding and head as Q8_0.

## Usage

```sh
MODEL=NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf
gufo serve --host 0.0.0.0 --port 8080 llm --model "$MODEL" --context 32768
gufo prompt --model "$MODEL" "What is the capital of France?"
gufo chat --model "$MODEL"
```

With the drafter, each decoding round drafts up to `--draft-tokens` (at most
7, the default) tokens and the trunk verifies them in one pass:

```sh
DRAFT=NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf
gufo serve --host 0.0.0.0 --port 8080 llm --model "$MODEL" --context 32768 \
  --speculative lilicorr --lilicorr-model "$DRAFT"
gufo prompt --model "$MODEL" --speculative lilicorr --lilicorr-model "$DRAFT" \
  "What is the capital of France?"
```

Drafting never changes the output distribution: a draft is kept only while
the request's own sampler, given the trunk's verified row, draws that same
token, and grammar-constrained requests (tools, JSON schema) verify under
their grammar. A verified row is bit-identical to decoding its token alone,
so greedy output with the drafter equals greedy output without it. Responses report `draft_tokens` and `draft_tokens_accepted`
in `usage`.

Thinking is on by default and its text arrives as `reasoning_content`.
`reasoning_effort: "none"` turns it off; `"minimal"`, `"low"` and `"medium"`
select the template's efficient mode. Tool calls use the Qwen3-Coder XML format that the
template defines. Defaults are temperature 1.0 and top-p 0.95, as NVIDIA
recommends. Each session holds a KV cache of 8 KiB per token plus about
165 MiB of Mamba state and 40 MiB of verification inputs; with the drafter
about 75 MiB more.

## Architecture

| Part | Shape |
| --- | --- |
| Blocks | 88, pre-norm RMSNorm with residual: 40 Mamba-2, 40 MoE, 8 attention |
| Width | hidden 4096, vocabulary 131,072, native context 262,144 |
| Attention | GQA, 32 query and 2 KV heads of 128, no positional encoding |
| Mamba-2 | 128 heads of 64, state 128, 8 B/C groups, causal conv of 4, silu(z) gate, RMSNorm per group of 1024 |
| MoE | sigmoid router with a selection-only bias, top-9, normalized weights × 5; 240–496 routed ReLU² experts per block (width 1536–2560) in a 1024-wide latent space; one 5376-wide shared ReLU² expert |

Per-block expert counts and widths come from the file; the runtime accepts
only this geometry otherwise.

## Runtime

- Chunks of more than 8 tokens (prefill) run the Mamba projections through
  a model-private FP16 WMMA kernel (`kernels/rocm/gemm.hip`): each NVFP4
  sub-block decodes exactly to FP16, and the norms ahead of the projections
  write their FP16 activations. The routed experts run through a BF16 WMMA
  kernel in the same file, which groups each block's token slots by expert
  and keeps the IQ2_XXS grid values exact. The shared expert, attention and
  latent projections use llama.cpp's MMQ kernels, copied into
  `kernels/rocm/mmq` with an `nb_` prefix, on Q8_1 activations.
- Chunks of up to 8 tokens (decode and verification) run the IQ2_XXS
  experts through a model-private vector kernel (`kernels/rocm/gemv.hip`,
  BF16 activations, `v_dot2_f32_bf16`; ReLU² and the shared expert's
  residual add happen as it stores) and the Q8_0/NVFP4 products through
  llama.cpp's MMVQ kernels. The norm kernels write the Q8_1 activations those
  read, and for one token the output projections add into the residual
  stream themselves. The shared expert runs on a second stream beside the
  router and the routed experts.
- Mamba-2 conv/scan, attention, routing and normalization are model-private
  kernels in `kernels/rocm/kernels.hip`. Attention is flash attention on
  FP16 WMMA: one wave takes the 16 query heads that share a KV head for one
  token; chunks of up to 8 tokens split the keys over 64 blocks per KV head.
  The scan keeps each head's state in registers, stored state-major
  (`[head][n][row]`).
- A single-token decode step is captured once per session as a HIP graph and
  replayed; positions are read from device memory.
- Concurrent requests (2 to 8) decode one token each in a shared pass
  (`Session::DecodeBatch`): the matrix products take the rows together and
  read the weights once, while each row's convolution, scan, KV append and
  attention run the one-token arithmetic on its own session's state, found
  through a device table. The rows are bit-identical to decoding each
  session alone. The pass is captured once per width. With the drafter, a
  request drafts while it decodes alone; in a shared pass each row's
  features still go into its session's drafter context.
- Prefill runs in chunks of 2,048 tokens; the HTTP runner yields after 2,048.
- Random sampling (top-k or top-p) reads a short head of each logits row
  that the GPU builds (`kernels/rocm/sample.hip`) instead of the whole
  vocabulary; [QUALITY.md](QUALITY.md#sampling-from-device-heads) describes
  why the draws stay those of the full row.
- With the drafter, every trunk pass also copies the residual stream entering
  blocks 2, 19, 36, 52, 69 and 86 and projects it into the drafter's K/V
  rings (the 1,024-position sliding window plus the block); rows older than
  that window skip it. A draft runs the anchor token and 7 mask tokens
  through the five drafter layers and the trunk's head, and the LiLiCorr head
  scores the top 8 candidates of each slot; the greedy walk of that lattice
  is the draft. Verification runs the block like a prefill but leaves the
  Mamba states in place and keeps their inputs; `Commit` replays the kept
  prefix through the same convolution and scan kernels, so the states equal
  those of appending it.
- A decode token at short context runs 757 kernels in about 26 ms of GPU
  time and reads about 5.1 GB: the NVFP4 Mamba projections 2.47 GB, the
  output head 0.57, the shared experts 0.45, the routed experts 0.44, the
  latent projections 0.36, attention 0.30, the F32 router 0.23 and the
  Mamba state 0.34. At the measured 238 GB/s that bounds decode near
  47 tok/s.

## Development checks

```sh
cmake --preset gpu-test
cmake --build --preset gpu-test --target nemotron35_bolt_tests
GUFO_NEMOTRON_BOLT_GGUF=$MODEL GUFO_NEMOTRON_BOLT_DRAFTER=$DRAFT \
  ctest --preset gpu-full -R '^nemotron35_bolt\.'
```

Without the model the three `external-model` checks skip; without the
drafter `nemotron35_bolt.runtime` skips its drafter checks. The full-model CPU
reference probe and the llama.cpp comparison are described in
[QUALITY.md](QUALITY.md).

The [functional API suites](../../../tests/functional/README.md) run with
`--sampling-preset nemotron35` and the suites of `all` except `cache`, which
needs the disk cache. Its expectations differ from Qwen's only where llama.cpp
on the same GGUF behaves the same or the chat template requires it. One
interaction remains: after a full run's few hundred conversations the RAM
cache holds 128 checkpoints of about 175 MB, below its byte budget, and the
entry limit refuses the optional 2,048-token checkpoint that `cache-edits`
needs to reuse an edited or rewound history. On a fresh server the suite
passes.
