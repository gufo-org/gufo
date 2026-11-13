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
`GUFO_IRON_PYTHON` names an IRON Python (`cmake/GufoNpuDesigns.cmake`; the
Nix package passes `.devops/nix/iron`). They install to
`share/gufo/npu/<name>`; `FindDesign` also honors `GUFO_NPU_DESIGNS`.

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
