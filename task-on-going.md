# Phase B — Direct Quant Prefill GEMM: Result

## Status: IMPLEMENTED, CORRECT, but MASSIVE pp128 REGRESSION

Phase B implemented per spec. The dequant-to-bf16 cast is GONE (`time_dequant=0`),
prefill computes the matmul directly on quantized weights. Correctness holds
(top1_match=yes). But the naive one-warp-per-row quant kernel is ~35x slower than
the hipBLAS BF16 GEMM it replaced. pp128 regressed 92.71 -> 2.61 tok/s.

## Changed files
- src/core/hip/qwen_gpu_prefill_ops.hip  (add BatchedQuantGEMVKernel + LaunchBatchedQuantGEMM)
- src/core/hip/qwen_gpu_ops.hpp          (declare LaunchBatchedQuantGEMM)
- src/core/hip/qwen_gpu_prefill.cpp      (wire gemm_weight quant branch -> LaunchBatchedQuantGEMM)
Untouched: decode_ops.hip, qwen_gpu_quant_ops.hpp (Phase A shared header).

## Validation
### qwen_gpu_ops_test (decode regression gate)
`./build/gpu-test/qwen_gpu_ops_test` EXIT=0, "All Qwen HIP GPU kernel tests passed on gfx1151."
(GEMV primary int-Q8 vs GPU max_rel=0.00; Q5_K/Q6_K dequant-dot max_rel=0.00; all fused
diff=0.00.)

### Builds
- cmake --preset gpu-test configure: OK (0.2s, gfx1151)
- cmake --build build/gpu-test --target qwen_gpu_ops_test: OK (EXIT 0; only pre-existing
  decode_ops.hip warnings, untouched file)
- nix build (prod ./result/bin/strix-server): OK (EXIT 0)

### Correctness (prod binary, Q8_K_L)
phb_full.log:
  [Prefill Validation] tokens=128 top1_match=yes finite=yes
    max_abs_diff=0.36560065 mean_abs_diff=0.05243833 rmse=0.06664532 cosine_similarity=0.99970269
  (OLD dequant path cosine=0.99961829 — new path numerically at least as good.)

### Throughput (prod binary)
| model | test | OLD (dequant+hipblas) | NEW (direct quant) |
| Qwen3.8-27B | pp128 | 92.71 tok/s | 2.61 tok/s |
| Qwen3.8-27B | tg16  | 2.15 tok/s | 2.14 tok/s |

### STRIX_PROFILE (phb_prof.log, -p 4)
[STRIX_PROFILE B=4] Total: 1568.37 ms (2.55 tok/s)
  - Input Proj: 330.22 ms
  - FFN (3 GEMM): 1149.01 ms
  - Dequant: 0 ms
  - GEMM(deq): 1560.11 ms
[STRIX_PROFILE B=32] Total: 12047.8 ms; Dequant: 0 ms; GEMM(deq): 12013.8 ms

## VERDICT
- `time_dequant = 0 ms` : the cast/dequant is GONE (user's hard rule satisfied functionally).
- `pp128` did NOT improve : REGRESSED 92.71 -> 2.61 tok/s (~35x slower); tg16 unchanged (2.14 vs 2.15).
- Cause: prescribed one-warp-per-row kernel has no weight reuse across batch, no tiling,
  scalar-ish block walk per output element — far slower than hipBLAS BF16 GEMM it replaced.
  Structural, not a correctness bug (top1_match=yes, cosine 0.99970).

## Open risks / decision needed (parent)
Phase B's performance premise is not met: avoiding dequant did not help, it collapsed pp
throughput. Options: (a) accept regression (correctness/refactor done, perf deferred to a
later tiled/reused kernel), (b) revert quant branch to dequant+hipblas for now, (c) design a
faster tiled direct-quant GEMM (weight-tile reuse across batch) before landing. Recommend (c)
as the path consistent with the no-dequant hard rule; (b) is the safe interim.
