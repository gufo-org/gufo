# YuE2 implementation notes

Reference: `multimodal-art-projection/YuE` `main` at 1116bea, package
version 0.1.6 (`src/yue2/pipeline.py`, `nar.py`, `sampling.py`,
`modeling_yue2.py`, `modeling_vae.py`; their numerics match the
`yue2-v0.1.6` tag), weights `m-a-p/YuE2-3B` 8f24312f, `m-a-p/YuE2-Vae`
152733a1.

## Layout

| File | Contents |
| --- | --- |
| `tokenizer.cpp` | tiktoken ranks (`qwen.tiktoken`), the Qwen2 split pattern, NFC, `encode_ordinary`; byte-pair merges in tiktoken order |
| `protocol.cpp` | Token ids, instructions, plan/semantic/negative prefixes, chunk ranges |
| `sampler.cpp` | Phase window, min-token end ban, windowed multiplicative penalty, temperature, top-k, top-p; BF16 CFG mix; the planning-off BF16 arithmetic |
| `noise.cpp` | `torch.randn` on a CPU generator: MT19937 uniforms and the 16-wide Box-Muller layout |
| `hip/backbone.cpp` | AR prefill/decode and the NAR velocity on the shared `src/families/qwen/eager` operators; the midpoint ODE |
| `hip/kernels.hip` | The acoustic attention, the 12-bit decode rows and their GEMV, the latent input and output, and the decoder's element-wise operators |
| `hip/vae.cpp` | Oobleck decoder: weight norm folded at load, k7 convolutions as seven shifted SGEMMs, transposed convolutions as polyphase SGEMMs |
| `provider.cpp` | `MusicRunner`: plan, semantic, acoustic, decode |

## Numerics

Activations keep PyTorch's eager BF16 boundaries, including YuE2's RMSNorm,
which rounds the reciprocal RMS to BF16 before scaling
(`RmsScale::kBfloat16`). KV caches are BF16: upstream stores the same BF16
values. Decode attention keeps scores and softmax in FP32 and rounds the
probabilities to BF16 before the value product, as PyTorch's fused attention
does; prefill and NAR attention use the fused WMMA kernel with an FP32 value
product. Logits are computed only for the phase's rows (planning: text plus
`</abc>`; semantic: `</music>` plus the 32768 codec rows) and rounded to BF16.

One-token decode reads the AR projections and the head from 12-bit copies
packed at load (`hip/kernels.hpp`). They hold every BF16 weight exactly, and
the GEMV adds in the BF16 kernel's order, so decode gives the same bits as
it does from the tensors.

The midpoint solver keeps upstream's BF16 state: it starts from the
BF16-rounded noise and rounds each update to BF16.

Departures from upstream, each at least as accurate:

- One KV cache serves the planning stage, the semantic stage and the acoustic
  prefix when the song fits one chunk (always, unless the prefix is longer
  than about 12k tokens): upstream re-prefills the same tokens.
- Sampling draws from a request-local MT19937-64, not torch's `multinomial`
  stream, so a seed reproduces Gufo's own output, not upstream's.

## Quality and speed

See [QUALITY.md](QUALITY.md), [BENCHMARKS.md](BENCHMARKS.md) and
[EXPERIMENTS.md](EXPERIMENTS.md).
