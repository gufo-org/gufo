#ifndef GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_ATTENTION_HPP_
#define GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_ATTENTION_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::glm5_next::rocm {

// Absorbed MLA attention over the latent-only cache (NoPE), driven by the
// indexer top-k selection: every query attends to the union of the token
// slots expanded from its selected pool groups. V is never cached; the
// per-head attn_v_b projection maps the softmax-weighted latent combination
// to the per-head output inside this module (IMPLEMENTATION.md section 3).
//
// Contract: kv_rank is the fixed latent width, heads * 256 == q width, and
// selection entries are distinct cache row indices below `capacity`. The
// executor may call the decode trio once per (future prefill) query row.
constexpr std::uint32_t kMlaKvRank = 512;

// q_abs[h][i] = scale * dot(kb[h][i][:256], q[h][:256]).
// `q` is [heads][256], `kb` is [heads][512][256] (attn_k_b), `q_abs` receives
// [heads][512]. Blocks: one per head, one thread per latent column.
void MlaQabs(const float* q, const float* kb, std::uint32_t heads, float scale,
             float* q_abs, hipStream_t stream);

// Position tiles for `n_sel` selected slots at `tile` slots each.
std::uint32_t MlaGatherTiles(std::uint32_t n_sel, std::uint32_t tile);

// Score pass: score[sel j][head h] for the gathered latent rows.
// `q_abs` is [heads][512] F32 (already scaled), `q_pack` is a
// [2][heads][512] F16 scratch buffer holding the hi/lo split of q_abs
// (the dots of both halves accumulate into one fp32 tile, so the score
// matches the fp32 dot tier), `cache` is [capacity][512] F16 latent
// rows, `sel` is [n_sel] row indices. `scores` receives a dense
// [tiles][heads][tile] buffer with entry [t][h][l] = dot for slot
// t * tile + l (l within the tile); entries of the last tile beyond
// `n_sel` are never written and never read. Blocks: 64 threads per
// 32-slot tile, latent rows staged in LDS, dots on 16x16x16 F16 WMMA
// tiles over the packed q (heads % 16 == 0, tile a multiple of 32); the
// tile summation order differs from the warp scan, covered by the
// tolerance gate in
// staging/tests/models/glm5_next/rocm_attention_test.cpp.
void MlaGatherScore(const float* q_abs, __half* q_pack, const __half* cache,
                    const std::int32_t* sel, std::uint32_t n_sel,
                    std::uint32_t tile, std::uint32_t heads, float* scores,
                    hipStream_t stream);

// Combine pass: softmax over all n_sel scores of each head and the
// normalized weighted sum of the gathered latent rows.
// `scores`/`cache`/`sel` are as in the score pass; `latent` receives
// [heads][512] F32. Blocks: one per head, 64 threads holding 8-column
// slices of the latent as vector loads with j-strided partials (`tile`
// recovers each slot's dense score position).
void MlaGatherCombine(const float* scores, const __half* cache,
                      const std::int32_t* sel, std::uint32_t tile,
                      std::uint32_t n_sel, std::uint32_t heads, float* latent,
                      hipStream_t stream);

// Token-batched prefill forms of the two gather passes. Query token t
// (blockIdx.z / blockIdx.y) reads `sel` at t * sel_stride, writes and reads
// its score region at t * scores_stride on [tiles][heads][tile], takes its
// absorbed q at q_abs + t * q_stride (packed hi/lo to q_pack +
// t * 2 * heads * 512) and its latent at latent + t * latent_stride. Its
// selection count is
// derived from the query position pos_first + t exactly like the host:
// n_sel(t) = kpool * min(top_pools, (pos + 1) / kpool) +
//            (use_tail ? (pos + 1) % kpool : 0). `n_sel_max` (the largest
// of these over the token range) sizes the grids and score regions. Every
// block runs the per-token block arithmetic unchanged, so results are
// bit-identical to the single-token calls.
void MlaGatherScoreBatch(const float* q_abs, std::size_t q_stride,
                         __half* q_pack, const __half* cache,
                         const std::int32_t* sel, std::size_t sel_stride,
                         std::uint32_t n_sel_max, std::uint32_t pos_first,
                         std::uint32_t kpool, std::uint32_t top_pools,
                         std::uint32_t use_tail, std::uint32_t tokens,
                         std::uint32_t tile, std::uint32_t heads, float* scores,
                         std::size_t scores_stride, hipStream_t stream);
void MlaGatherCombineBatch(const float* scores, std::size_t scores_stride,
                           const __half* cache, const std::int32_t* sel,
                           std::size_t sel_stride, std::uint32_t n_sel_max,
                           std::uint32_t pos_first, std::uint32_t kpool,
                           std::uint32_t top_pools, std::uint32_t use_tail,
                           std::uint32_t tokens, std::uint32_t tile,
                           std::uint32_t heads, float* latent,
                            std::size_t latent_stride, hipStream_t stream);

// Fused single-pass replacement for the batch score+combine pair of a
// prefill chunk. Packs q like MlaGatherScoreBatch, then runs one block per
// query token (heads * 4 threads): wave pairs split the 512-dim scores and
// the output channels, the online chunk softmax (F32 accumulators, F16
// probabilities) removes the scores round-trip. `q_abs`, `q_stride`,
// `q_pack`, `cache`/`sel`/`sel_stride` and the per-token n_sel derivation
// are as in the batch pair; `latent` receives [tokens][heads][512].
// Requires heads % 16 == 0 and heads <= 64. The softmax runs in base 2 on
// the already-scaled q and the probabilities round through F16, so the
// output differs from the batch pair (NMSE ~1e-9 against the fp64 oracle,
// gate in staging/tests/models/glm5_next/rocm_attention_test.cpp).
void MlaFusedAttend(const float* q_abs, std::size_t q_stride, __half* q_pack,
                    const __half* cache, const std::int32_t* sel,
                    std::size_t sel_stride, std::uint32_t pos_first,
                    std::uint32_t kpool, std::uint32_t top_pools,
                    std::uint32_t use_tail, std::uint32_t tokens,
                    std::uint32_t heads, float* latent,
                    std::size_t latent_stride, hipStream_t stream);

// head_out[h][i] = dot(vb[h][i][:512], latent[h][:512]).
// `vb` is [heads][256][512] (attn_v_b rows), `latent` is [heads][512];
// `out` receives [heads][256]. Blocks: one per head, 256 threads.
void MlaApplyVb(const float* latent, const float* vb, std::uint32_t heads,
                float* out, hipStream_t stream);

}  // namespace gufo::models::glm5_next::rocm

#endif  // GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_ATTENTION_HPP_
