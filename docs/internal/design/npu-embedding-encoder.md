# EmbeddingGemma 2 on the NPU: one program per request

Status: design, building in milestones (`src/core/npu/kernels/embenc`). The
current `serve embedding --npu` path orchestrates `BfpGemm` products from the
CPU and runs norms, GELU, softmax and conversions there: about 24 CPU
core-seconds per 15,670-token request and 36% of GPU speed. The goal is all
math on the NPU, the CPU only staging token rows and reading embeddings.

## Measured facts that shape it

- `BfpGemm` cores are data-movement bound: a variant without the multiply runs
  as fast. A 512 x 1024 output tile costs about 43 us at K = 512.
- A bf16 residual state drops the worst golden cosine to 0.99884 (gate
  0.999); the state stays f32-equivalent as a hi/lo pair of bf16 rows.
- Core memory is 64 KB: a 32 KB C tile, double-buffered A (4 KB) and B (9 KB)
  objects and the stack leave no room for a second row-wide output buffer, so
  row operations cannot all live in the GEMM epilogue.
- Hardware-context switches cost about 2.3 ms: the whole request runs in one
  xclbin.

## Program

One IRON design whose 32 cores (8 columns x 4 rows) interpret a phase plan,
as LFM3's prefill does. Activations live in one device buffer `S`; the host
writes token rows and per-layer-input rows (the media towers splice their rows
here) and reads pooled embeddings.

Phases:
- **GEMM** (the ATB core loop, bf16 A converted on the cores): A rows from `S`
  (row-major), W from the arena, C to `S` (row-major drains).
- **Row** (LFM3's pattern): each core row's A stream carries whole rows to its
  8 cores; every core computes the row reductions and writes its column
  slice of the outputs through its C tile. Ops:
  - `addnorm`: s' = (s + rmsnorm(t) w_post) * scale, out hi'/lo' and
    rmsnorm(s') w_next (the next product's input), fused.
  - `gelu_gate`: gelu(g) * u (FFN), gelu(g) * ple_l (per-layer input).
  - `qkr`: per-head q/k RMS norms, unweighted v norm, NeoX RoPE (cos/sin
    tables in `S`).
  - `ple`: per-layer-input projection slice norms (scale folded into eps).
- **Attention** as products: QK^T with keys as the weight operand (a row phase
  writes K and V^T in the weight layout), an exp epilogue (an appended
  column of -offset shifts the logits; offset = |q| max|k|), and PV with an
  appended row of ones so the softmax denominator comes out as an extra
  output column; a row phase divides.
- **Pool**: per-sequence mean of normalized final states (linear, so pooling
  precedes `embedding_projection`), the 768-wide projection on the pooled
  rows, L2 normalize.

## Milestones

1. Row phases in a standalone design, checked against the CPU reference.
2. GEMM + row phases for one layer without attention, in one dispatch.
3. Attention products and epilogues.
4. All 24 layers and pooling in one dispatch; the host path becomes the
   fallback only for shapes the program cannot serve.
