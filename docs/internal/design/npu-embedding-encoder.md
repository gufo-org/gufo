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

## Progress and findings

**Milestone 1a (done, `kernels/embenc`, `test_embenc.py`):** the `addnorm`
row phase on all 32 cores, standalone. Records `[t | s_hi | s_lo | xn]`;
column c takes objects of 4 rows (c, c + 8, ...), core q of the column
computes row q and writes its `[s_hi' | s_lo' | xn']`, and the memtile join
of the four cores is already in record order, so input fill and drain are
plain 4-D taps with `n_obj` (objects per column) at dispatch time.
- Precision against f64: state 2.5e-6 relative, xn 1.66e-3 (the bf16 output
  rounding floor). It needs `aie::set_rounding(conv_even)` (Peano truncates
  accumulator-to-bf16 conversions otherwise, which doubles the error), t w
  split into hi/lo bf16, and per-row scales as hi/lo bf16 pairs.
- `w_next` folds into the next product's weights (xn' = s' / rms(s')).
- Cost: f32 vector multiplies and subtractions on vectors are emulated and
  slow (a Newton step on invsqrt made the kernel 4.6x slower; it was not
  needed: `aie::invsqrt` is accurate enough). Use bf16 x bf16 into accfloat
  and accumulator add/sub only. Data movement bound at about 1.3 ms for
  16,384 rows when the NPU is quiet (stub kernel: 1.5 ms).
- A join's memtile transfers go one core at a time, so a `to_stream`
  reorder across the joined cores is rejected: each core must own whole
  output rows (or the drain needs 5 dimensions).

**Open problem for milestone 2 (GEMM and row phases in one program):** the
GEMM's FIFOs (A per core row with the 8x8-block forward transforms, B per
column of 9,216-byte objects, a 32 KB C tile joined per column with the tile
inverse) use most of a core's 64 KB, so row phases cannot have FIFOs of their
own and must reuse these:
- Row inputs through B: 9,216 bytes is exactly 3 rows of `[t | s_hi | s_lo]`
  (1,536 bf16), not 4, so the "core q owns row q" mapping needs 3-row objects
  (one core idle) or another record width.
- Row outputs through C: a core writing whole rows of 4 slots (2,048 bf16 =
  16 C rows) fills a C object with exactly 8 token rows, which keeps the join
  in record order. Writing in the C tile's 8x8 layout (LFM3's `TileRow`) makes
  the drain's tile inverse produce row-major records.
- Alternatives: a GEMM C tile of 64 rows per core (16 KB, frees room for
  dedicated row FIFOs, but doubles weight streaming per band), or LFM3's
  layout in full (records sized so one B object is one record).
