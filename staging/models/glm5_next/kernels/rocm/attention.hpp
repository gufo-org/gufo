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
// `q_abs` is [heads][512] (already scaled), `cache` is [capacity][512] F16
// latent rows, `sel` is [n_sel] row indices. `scores` receives a dense
// [tiles][heads][tile] buffer with entry [t][h][l] = dot for slot
// t * tile + l (l within the tile); entries of the last tile beyond
// `n_sel` are never written and never read. Blocks: 256 threads, one warp
// per slot loading its row as coalesced 16 B chunks, grid
// (ceil(n_sel/8), heads/hspan); each warp reduces its dot with shuffles,
// so summation order differs from a sequential scan within fp tolerance.
void MlaGatherScore(const float* q_abs, const __half* cache,
                    const std::int32_t* sel, std::uint32_t n_sel,
                    std::uint32_t tile, std::uint32_t heads, float* scores,
                    hipStream_t stream);

// Combine pass: softmax over all n_sel scores of each head and the
// normalized weighted sum of the gathered latent rows.
// `scores`/`cache`/`sel` are as in the score pass; `latent` receives
// [heads][512] F32. Blocks: one per head, threads along the latent
// columns (`tile` recovers each slot's dense score position).
void MlaGatherCombine(const float* scores, const __half* cache,
                      const std::int32_t* sel, std::uint32_t tile,
                      std::uint32_t n_sel, std::uint32_t heads, float* latent,
                      hipStream_t stream);

// head_out[h][i] = dot(vb[h][i][:512], latent[h][:512]).
// `vb` is [heads][256][512] (attn_v_b rows), `latent` is [heads][512];
// `out` receives [heads][256]. Blocks: one per head, 256 threads.
void MlaApplyVb(const float* latent, const float* vb, std::uint32_t heads,
                float* out, hipStream_t stream);

}  // namespace gufo::models::glm5_next::rocm

#endif  // GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_ATTENTION_HPP_
