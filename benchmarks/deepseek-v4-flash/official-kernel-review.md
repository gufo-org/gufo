# Official DS4 kernel review

Inspected 2026-09-10; DS4 Flash 0731 on gfx1151, primarily C1 autoregressive.
These are source inspections of pinned upstream checkouts, not upstream speed
claims extrapolated to this GPU.

| Repository | Revision | Useful direction for Gufo |
| --- | --- | --- |
| DeepGEMM | `66081d4c9c7d7c44f13fea402e5b622aa0f409c2` | MQA score reduction; fused gate and mHC stages |
| FlashMLA | `07a1089857b63e74e3133630c02b083b75e8d4b2` | Fuse query norm, RoPE, sparse attention, inverse RoPE |
| deepseek-recipe | `8cadfede7063c896b944e7bae05daa3549ae97ea` | V4 request/response encoding and streaming; inference is external |
| DeepSelect | `0f03b68748b304863fdf0181a11458d04ae533a9` | Filter and compact candidates instead of sorting every indexer score |

## Measured indexer change

The direct score kernel previously synchronized twice after each of 16 groups
of four heads. Giving each of the 64 heads its own shared-memory slot removes
those 32 loop barriers and needs one final barrier. Dot products, head weights,
scaling, four-head grouping, accumulation order, and causal visibility stay
unchanged. The launch remains 128 threads. No runtime selector is added.

The isolated production-flag A/B is bit-exact for 24 shapes: 1/6 query tokens,
513/1,024/2,048/4,096/8,192/32,768 compressed rows, and causal/noncausal modes.
The existing `ds4.attention` test adds a small permanent regression check against
the historical GPU arithmetic and a sampled independent double-precision CPU
formula, including repeated output, exact score ties, signed head weights,
compression boundaries within a verifier block, and poisoned invisible rows.

Two process pairs, run baseline/candidate then candidate/baseline, reproduced
**0.8–1.0% faster C1 generation at 16K**. All generated-token hashes match.

| Workload | Baseline tok/s | Candidate tok/s |
| --- | ---: | ---: |
| tg128 at 16K, first pair (3 repetitions) | 14.51 ± 0.00 | 14.63 ± 0.00 |
| tg128 at 16K, reverse pair (2 repetitions) | 14.51 ± 0.00 | 14.65 ± 0.01 |
| tg128 at depth 0 (2 repetitions) | 16.74 ± 0.01 | 16.76 ± 0.01 |
| pp2048 at 16K (3 repetitions) | 423.64 ± 2.06 | 424.65 ± 1.89 |

The 16K improvement repeats beyond the reported within-process variation;
shallow generation and prefill are effectively unchanged. Rejected launch-size
variants used 256/512/1,024 threads and were slower than 128 threads.
The [evidence](official-kernel-review.json) retains process summaries, repeated
hashes, kernel samples and resources, source/binary identities, and check logs.
The earlier complete qualification remains in `quality-qualification.json`
with its original source identity.

Final focused checks passed: the maintained attention test and target model
quality suite, including the unchanged 116/128 trajectory (rank sum 142,
worst rank 3), exact wide-prefill fingerprint, and session batch checks.
A C1 DSpark control at 16K repeated the same output hash and all reference draft
counters (105 accepted / 107 drafted, 22 steps), recording 36.58 tok/s. All runs
passed a 12 GiB memory guard; DSpark kept at least 22.87 GiB available.
The full model/serving suite, external continuation comparison, capability
evaluation, and full speed matrix were not repeated for this focused change.

Reproduce the small kernel comparison with the existing tool:

```sh
nix develop -c cmake --build --preset gpu-test --target ds4_attention_test
nix develop -c build/gpu-test/tests/models/deepseek_v4_flash/ds4_attention_test --benchmark
# Use each immutable Nix release in baseline/candidate/candidate/baseline order:
./result/bin/gufo bench --model "$MODEL" -c 1 -p 2048 -n 128 -d 16384 -r 3 -v
# Reverse-order pair and shallow control:
./result/bin/gufo bench --model "$MODEL" -c 1 -p 0 -n 128 -d 0,16384 -r 2 -v
```

A separate 4K/tg16 decode profile attributes 1,017.81 ms of GPU work to
16 generated tokens: ordinary Q8 projections take 167.38 ms, Q8 projections
with HC expansion 158.76 ms, MoE gate/up 124.16 ms, grouped Q8 output projection
107.89 ms, and the changed indexer 8.77 ms. Prefix preparation is excluded.
The candidate reduces the indexer's 336 launches from **8.77 to 7.56 ms**
(13.8% less kernel time), with the same 24 VGPRs and zero scratch spilling.
LDS grows from 528 to 768 bytes per block. These traced timings identify
bottlenecks; the table above uses unprofiled runs.

The [selection/projection follow-up](selection-projection.md) evaluates the
partial selector, C1 projection and HC normalization opportunities below.

## Next opportunities

- **Exact partial top-k:** DeepSelect's bounded candidate buffer is portable as
  an algorithm. Retain FP32 scores, index tie-breaking, and Gufo's final selected
  index order; changing attention reduction order would change logits. The
  CUDA cluster implementation and recommended unsorted/BF16 modes are not direct
  substitutions. Start with larger contexts, where full sorting costs more.
- **Projection and mHC fusion:** DeepGEMM Mega Gate/Mega mHC demonstrate useful
  launch and intermediate-buffer savings. Preserve Gufo's FP32 activation and
  ordered HC projection contract; normalizing after a projection or introducing
  BF16/FP8 intermediates needs separate numerical qualification. Projection
  kernels dominate the measured C1 decode profile.
- **Attention fusion:** FlashMLA combines five stages, with permuted weights and
  specific KV/output layouts. DS4 V4 requires query normalization; V4.1 settings
  cannot be copied blindly. Gufo already fuses norm/RoPE in wide prefill; extending
  fusion to C1 requires preserving the scalar path's rounding points.
- **Mega MoE:** the inspected DeepGEMM kernel overlaps expert dispatch, two
  projections, SwiGLU, and combine using NVLink and symmetric multiprocess
  memory. Its communication design targets a different machine. A gfx1151
  implementation would be a substantial new kernel, not a quick CUDA port.
- **Serving:** recipe supplies encoding/stream parsing, with mock inference in
  its example servers. It does not provide a replacement GPU scheduler or KV
  allocator to import for this task.

Relevant upstream files: DeepGEMM `tests/test_mega_gate.py`,
`tests/test_mega_mhc.py` and `deep_gemm/include/deep_gemm/impls/`;
FlashMLA `flash_mla/fused_norm_rope_attn_rope_cast.py` and its matching test;
DeepSelect `docs/DeepSelect-deep-dive.md`; recipe
`deepseek-recipe-encoding/src/v4/dsv4.rs`.
