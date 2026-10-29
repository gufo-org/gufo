# LFM3 1.7B

Native text inference for LiquidAI LFM3-1.7B on Linux x86-64 / Strix Halo
`gfx1151`. Implementation: [src/models/lfm3](../../../src/models/lfm3).

The 24-layer backbone mixes 13 gated width-seven convolutions, five sliding
attention layers (256 tokens) and six global attention layers. Attention has
Q/K RMSNorm, a sigmoid gate per head, and full-head paired RoPE with masked
frequencies on global layers. Both operator and FFN branches have sandwich
norms. Embedding and vocabulary projection share weights.

## Precision and acquisition

Use **Q8_0 for deployment** and **BF16 for numerical reference**. BF16 is the
original checkpoint precision. F16 is accepted for compatibility; it offers no
weight-memory advantage over BF16. Q4/Q5/Q6, multimodal and decision variants
are not supported by this initial backend.

Weights stay packed on the GPU. Activations, accumulation, convolution history
and KV caches remain FP32. Q8 prompt projections scale each activation row by a
power of two and split it into an FP16 high part plus the FP16 rounding of the
remainder (about 22 significant bits per element), chained into one FP32 F16
WMMA accumulator per Q8 K32 block. This approximation passes the independent
projection and model probability gates with lower error than the earlier
three-digit integer encoding; decode uses the existing Strix Halo quantized
GEMV kernels. BF16 prefill shares weights across rows using the existing exact
BF16/FP32 projection kernels. Attention stays FP32 for prompts and decode.

Download from the [official GGUF repository](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/tree/b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf):

```sh
hf download LiquidAI/LFM3-1.7B-GGUF LFM3-1.7B-Q8_0.gguf \
  --revision b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf \
  --local-dir models/lfm3
```

Replace the filename with `LFM3-1.7B-BF16.gguf` for reference weights. Model
weights use the [LFM Open License](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE)
and are downloaded separately.

## Usage

```sh
build/release/gufo prompt --model models/lfm3/LFM3-1.7B-Q8_0.gguf \
  --prompt 'What is the capital of France?' --think off
build/release/gufo chat --model models/lfm3/LFM3-1.7B-Q8_0.gguf --think off
build/release/gufo bench --model models/lfm3/LFM3-1.7B-Q8_0.gguf \
  -p 2048 -n 128 -r 3
build/release/gufo serve --sessions 2 llm \
  --model models/lfm3/LFM3-1.7B-Q8_0.gguf --context 4096
```

The CLI reserves 8192 tokens; HTTP accepts `--context` up to the native 262144.
The native limit is a loader limit, not a qualification of maximum-depth quality
or speed. HTTP supports Chat Completions, Responses and Messages, custom stops,
structured JSON and native tool calls. Sessions have independent FP32 state;
the scheduler batches up to eight autoregressive decode rows. RAM checkpoints,
prefix reuse, cancellation recovery and restart-safe disk snapshots are wired
through the shared serving cache. Speculative decoding and image inputs remain
unsupported. CLI chat retains its live prefix when framing permits.

The full release API matrix still fails: several prompts produce unfinished
reasoning or fail exact-answer/tool expectations. See [quality](QUALITY.md) for
retained failures and the bounded checks that pass; API wiring is not a claim
that this checkpoint meets the release quality bar.

Thinking defaults on; use `--think off` or the HTTP
`chat_template_kwargs.enable_thinking=false` control to disable it. Defaults are
temperature 0.1, top-k 50 and repetition penalty 1.1; explicit sampling controls
override them. Historical assistant thoughts are stripped as in the official
text-only template. The tokenizer uses LFM2's Unicode split and byte-level BPE,
including three-digit number groups.

[Quality](QUALITY.md) · [Benchmarks](BENCHMARKS.md) · [Experiments](EXPERIMENTS.md)
