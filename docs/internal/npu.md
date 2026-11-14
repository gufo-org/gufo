# NPU runtime

How Gufo runs models on the Strix Halo XDNA NPU through XRT, for contributors
changing `src/core/npu` or a model's `npu/` runtime.

## Pieces

| Piece | Source | Role |
| --- | --- | --- |
| Runtime | `src/core/npu/npu.hpp` | Devices, programs (one xclbin = one hardware context), buffers, instruction streams, dispatch libraries, `FindDesign` |
| `Q8Gemm` | `src/core/npu/q8_gemm.*`, `kernels/q8gemm` | Q8_0 weights, int16 rows: few-row products |
| `BfpGemm` | `src/core/npu/bfp_gemm.*`, `kernels/bfpgemm` | bfp16 rows x bfp16ebs8 weights: many-row products (prefill, encoders, embeddings); `bfpgemm_s` serves runs of short products in 128-row bands |
| `Qwen3Decoder` | `src/core/npu/qwen3_decoder.*`, `kernels/qwen3dec` | One dispatch per decode token for the Qwen3 dense block (28 x 2048): Qwen3-ASR's decoder and the Qwen3-TTS talker |
| Host operators | `src/core/npu/host_ops.*` | CPU norms, rotary embeddings, attention, activations, bf16 conversion on a persistent thread pool |

Designs are IRON (mlir-aie) programs compiled at build time when
`GUFO_IRON_PYTHON` names an IRON Python (`cmake/GufoNpuDesigns.cmake`, list in
`cmake/GufoNpuDesignList.cmake`; each reruns when a file in its
`kernels/<name>` changes). They install to `share/gufo/npu/<name>` (Windows:
`npu\<name>` beside `gufo.exe`); `FindDesign` also honors `GUFO_NPU_DESIGNS`.
The Nix package builds them as a derivation of their own
(`.devops/nix/npu-designs.nix`, from `cmake/npu-designs`) and installs it with
`GUFO_NPU_PREBUILT_DESIGNS`, so Cachix serves them until a design changes;
Windows CI keeps the IRON venv and IRON's design cache (`GUFO_NPU_CACHE_HOME`)
between runs (BUILD-WINDOWS.md).

`qwen3pre` keeps its 28 repeated layers in an IRON `range_` loop until
lowering emits the instruction stream. `Plan.layer_phases` advances weights,
norm constants and KV pieces; activations reuse their scratch regions. Keep
this loop staged: expanding it in Python makes MLIR verification expensive.
`npu_prefill_plan_test` checks every phase against the host layout at all
three row counts without requiring IRON or an NPU (about 1 ms).

On November 14, 2026, a cold Nix build of all three prefill variants took
18.417 s, including evaluation, configure and install, with mlir-aie 1.4.5
and the pinned Peano toolchain. The original 256-row and 512-row builds took
42.774 s and 93.919 s individually in `nix develop`. Both instruction streams
and host layouts match the packaged loop version byte for byte; all 32 core
ELFs also match for the 256-row comparison. The 1024-row layout passes the
phase check and its packaged design builds successfully; a full original
1024-row build was not repeated. The separate decoder dispatch C++ compilation
remains expensive. Measurement details: [prefill build evidence](artifacts/npu-prefill-build.json).

## Decoder build time

The decoder keeps its repeated layer schedule in `range_` with an internal
`DispatchTime[np.int32]` layer count. The C++ host supplies the model's fixed
depth, and `require` rejects other values. A constant-bound `range_` reduces
initial MLIR but is unrolled again before C++ emission. The checked dispatch
bound preserves the loop in the host builder; it adds no user setting.

On November 14, 2026, a cold build of the 28-layer program with GCC 16.2 took
292.658 s for the expanded schedule and 9.767 s for the retained loop, with
separate empty IRON caches. Generated C++ shrank from 7.19 MB to 0.84 MB;
the host compilation phase fell from 277.916 s to 5.355 s.

Clang 24 compiled the same compact builder in 4.322 s with about 217 MiB
peak RSS. On the expanded source it took 165.119 s but needed about 22 GiB,
so selecting Clang alone does not solve the build problem. Nix uses the
existing pinned ROCm Clang for host dispatch libraries; Peano and kernel
optimization settings stay fixed.

All 32 core ELFs and their probe copies match byte for byte. The decoded
DMA, RTP and lock traces match for both models, including context/head
boundaries; the raw instruction streams differ in BD-pool register writes.
The numerical NPU test passes for the decoder and predictor. Matched
20-token controls were 29.99/29.98 ms per decoder token and 2.46/2.41 ms per
predictor token (baseline/candidate), without a material regression. The
host checks IRON's optional `dispatch_abi` before calling a library, so
stale scalar signatures fail cleanly. Full measurements, commands, limits
and retained logs are in [decoder build evidence](artifacts/npu-decoder-build.json).

## BfpGemm

The cores are mlir-aie's asymmetric-tile-buffering GEMM (`bfpgemm.cc`): each
of the 32 cores owns a 128 x 128 bf16 output tile; A arrives as bf16 and is
converted to bfp16 on the core; W is pre-encoded. Shapes are dispatch-time
values written to runtime parameters, so one xclbin serves every matrix.

- A job is one 512-row band by up to 64 groups of 1024 outputs, one group by up
  to 64 bands, or a batch: one band from each of several calls whose matrices
  lie at a constant arena stride (`b_rep`, the weight stream's repeat stride).
- Jobs of one `Run` concatenate into one dispatch. Instruction streams are
  generated once per job shape and patched for the A, W and C offsets, which
  `dispatch_generate` writes linearly; the first use checks that.
- Two threads may stage at once (two lanes of A/C buffers), so one thread's
  packing, unpacking and CPU operators overlap the other's NPU time.
- `Scratch` regions hold per-request operands (attention keys and values).

Measured on Strix Halo (November 2026): about 1% relative error per product;
12-14 TFLOPS on 512 x 2048 x 4096-sized products; about 90 us per dispatch.
The program is limited by data movement, not arithmetic: a variant without
the matrix multiply runs at the same speed. A 512 x 1024 output tile costs
about 43 us at K = 512 and 92 us at K = 2048; A reaches each core row on one
shim stream at about 4.4 GB/s.

## Shim limits that shape the designs

- A buffer descriptor wraps at most 1023 in its inner dimensions and repeats
  at most 64 times; only the outermost (iteration) dimension may have stride 0.
- Offsets are int32 words: buffers stay under 2 GiB.
- Switching hardware contexts (xclbins) costs about 2-3 ms each way, so keep a
  phase of work inside one program.

## Testing

- `npu_host_ops_test` (CPU, hosted CI): vector paths against scalar formulas,
  attention against a naive loop, thread-pool rounds.
- `npu_device_test`, `npu_bfp_device_test` (label `npu`, skip without an NPU):
  products against double references; `npu_bfp_device_test <dir> bench n k
  rows` times one shape. `GUFO_NPU_PROFILE=1` prints per-dispatch pack, NPU
  and unpack times.
- Timings are only meaningful while no other process uses the NPU: designs
  time-share the whole array.
