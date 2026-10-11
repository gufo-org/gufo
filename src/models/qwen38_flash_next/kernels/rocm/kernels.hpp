#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KERNELS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KERNELS_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/sampling.hpp"
#include "src/models/qwen/vision/rope.hpp"

/// Model-private HIP launchers for everything outside the quantized GEMM
/// tier. Activations are row-major float32 [tokens][dim] unless noted; every
/// launch is asynchronous on `stream`. Weights referenced here are device
/// pointers in their GGUF encoding.
namespace gufo::models::qwen38_flash_next::rocm {

/// Stable per-prefix rollback addresses. Growing the depth does not move
/// existing buffers or invalidate graphs captured for a smaller batch.
struct RollbackRows {
  float* rows[7]{};
};

// Recurrent rollback keeps the state after the first token, then exact
// FP32 keys/decays/betas/errors for later tokens. No inverse or quantization.
constexpr std::size_t GdnRollbackRowFloats(std::uint32_t row,
                                           std::uint32_t k_heads,
                                           std::uint32_t v_heads,
                                           std::uint32_t dim) {
  return row == 0 ? std::size_t{v_heads} * dim * dim
                  : std::size_t{k_heads} * dim + v_heads * (dim + 2);
}
void RestoreGdnState(float* state, RollbackRows snapshots, std::uint32_t keep,
                     std::uint32_t k_heads, std::uint32_t v_heads,
                     hipStream_t stream);

/// GGUF type ids the runtime accepts for the small-matrix and lookup paths.
enum class WeightType : std::uint32_t {
  kF32 = 0,
  kF16 = 1,
  kQ5_1 = 7,
  kQ8_0 = 8,
  kQ4_K = 12,
  kQ5_K = 13,
  kBF16 = 30,
};

/// res[t][s][hidden] = table[tokens[t]] for every stream s (Q8_0 rows).
void EmbedTokens(const void* table, WeightType type, const std::int32_t* tokens,
                 float* res, std::uint32_t n_tokens, std::uint32_t hidden,
                 std::uint32_t streams, hipStream_t stream);

/// out[t][d] = rmsnorm(x[t][d]) * gamma[d]; `groups` independent norms of
/// `dim/groups` elements each share the gamma row. gamma may be null.
void RmsNormRows(const float* x, const float* gamma, float* out,
                 std::uint32_t n_rows, std::uint32_t dim, std::uint32_t groups,
                 float eps, hipStream_t stream);

/// Hyper-connection mix epilogue: mixed[t][i] = mean_s xn[t][s][i] *
/// sigmoid(gate[t][s][i]); with `inject_w` ([streams][hc_dim] f32) also
/// inject[t][s] = dot(inject_w[s], xn[t]).
/// Partial sums per inject logit the fused epilogue emits; `inject` holds
/// n_tokens * streams * HcInjectParts(hidden) floats.
std::uint32_t HcInjectParts(std::uint32_t hidden);
std::uint32_t HcInjectPartsVec4(std::uint32_t hidden);
/// A non-null `mixed_q8` (hidden % 32 == 0) also receives mixed as the
/// small-batch projections' Q8_1 rows, byte-identical to
/// qfn_mmq_quantize_q8_1.
void HcMixEpilogue(const float* xn, const float* gate, const float* inject_w,
                   float* mixed, float* inject, std::uint32_t n_tokens,
                   std::uint32_t hidden, std::uint32_t streams,
                   hipStream_t stream, void* mixed_q8 = nullptr);
/// Four adjacent hidden lanes per thread. The model's four-stream geometry
/// emits HcInjectPartsVec4(hidden) partial sums per inject logit.
void HcMixEpilogueVec4(const float* xn, const float* gate,
                       const float* inject_w, float* mixed, float* inject,
                       std::uint32_t n_tokens, std::uint32_t hidden,
                       std::uint32_t streams, hipStream_t stream);
/// The vec4 epilogue over an F16 `xn` (the F16 mixer input route); the
/// four-stream geometry is the caller's contract. Non-null `mixed_half` /
/// `mixed_q8` also receive the mixed rows as F16 and in the W8A8 tiled Q8
/// layout (K = hidden).
void HcMixEpilogueVec4F16(const __half* xn, const float* gate,
                          const float* inject_w, float* mixed,
                          __half* mixed_half, void* mixed_q8, float* inject,
                          std::uint32_t n_tokens, std::uint32_t hidden,
                          hipStream_t stream);

/// res[t][s][i] += block_out[t][i] * 2*sigmoid(inject[t][s] / streams), and
/// when `gamma` is non-null also the next mixer's grouped RMSNorm of the
/// updated residual into `xn`.
/// `inject` is [t][s][inject_parts] partial sums (1 part for a plain GEMM).
/// Concatenated decode requests retain scalar reduction with `decode=true`.
void HcCombine(float* res, const float* block_out, const float* inject,
               std::uint32_t inject_parts, const float* gamma, float* xn,
               std::uint32_t n_tokens, std::uint32_t hidden,
               std::uint32_t streams, float eps, hipStream_t stream,
               bool decode = false);
/// The one-block-per-stream HcCombine that also writes xn's Q8_1 rows (as
/// qfn_mmq_quantize_q8_1 would) to `xn_q8` for a small-batch mixer down
/// projection. False, with nothing launched, without `gamma` or unless
/// hidden % 32 == 0.
bool HcCombineQ8_1(float* res, const float* block_out, const float* inject,
                   std::uint32_t inject_parts, const float* gamma, float* xn,
                   void* xn_q8, std::uint32_t n_tokens, std::uint32_t hidden,
                   std::uint32_t streams, float eps, hipStream_t stream);
/// HcCombine writing the normalized output as F16, the width the next
/// mixer's projection and epilogue consume on the F16 mixer input route.
/// A non-null `xn_q8` also receives the norm quantized into the tiled Q8
/// layout (`Q8TiledBytes(n_tokens, streams * hidden)`) for the W8A8 down
/// projection; hidden must be a multiple of 32.
void HcCombineF16(float* res, const float* block_out, const float* inject,
                  std::uint32_t inject_parts, const float* gamma, __half* xn,
                  void* xn_q8, std::uint32_t n_tokens, std::uint32_t hidden,
                  std::uint32_t streams, float eps, hipStream_t stream);
/// HcCombineF16 with the MoE epilogue fused in: the block output is
/// MoeEpilogueVec4F16's result over `expert_out` ([tokens][used][hidden]
/// F16 rows), `weights`, the gated shared expert; it is formed in
/// registers and never written. Returns false (launching nothing) for a
/// geometry the fused kernel does not cover (four 2,560-wide streams).
/// A null gamma updates only the residual; otherwise xn_q8 is required.
bool HcCombineMoeF16(float* res, const __half* expert_out, const float* weights,
                     const float* shared_out, const float* gate,
                     std::uint32_t gate_stride, std::uint32_t used,
                     const float* inject, std::uint32_t inject_parts,
                     const float* gamma, __half* xn, void* xn_q8,
                     std::uint32_t n_tokens, std::uint32_t hidden,
                     std::uint32_t streams, float eps, hipStream_t stream);

/// x[i] = silu(x[i] * scale), in place over `count` floats.
void SiluScale(float* x, float scale, std::size_t count, hipStream_t stream);
/// SiluScale over `rows` rows of `k` that also writes x's Q8_1 rows, each
/// `k_padded` wide, to `x_q8` as qfn_mmq_quantize_q8_1 would (the padding
/// blocks are left as they are). False, with nothing launched, unless k and
/// k_padded >= k are multiples of 32.
bool SiluScaleQ8_1(float* x, float scale, std::uint32_t rows, std::uint32_t k,
                   std::uint32_t k_padded, void* x_q8, hipStream_t stream);
/// gate[i] = silu(gate[i]) * up[i], in place in `gate`.
void Swiglu(float* gate, const float* up, std::size_t count,
            hipStream_t stream);
/// out[i] = silu(gate[i]) * up[i] as F16 (`count` elements).
void SwigluHalf(const float* gate, const float* up, __half* out,
                std::size_t count, hipStream_t stream);
/// silu(gate) * up over n_rows rows of k elements, written only into the
/// W8A8 tiled Q8 layout `out_q8`; false (nothing launched) unless k % 32 == 0.
bool SwigluQ8Tiled(const float* gate, const float* up, void* out_q8,
                   std::size_t n_rows, std::size_t k, hipStream_t stream);
/// x[i] *= sigmoid(g[i]).
void SigmoidMul(float* x, const float* g, std::size_t count,
                hipStream_t stream);
/// SigmoidMul that also writes x's Q8_1 rows to `x_q8` as
/// qfn_mmq_quantize_q8_1 would for unpadded rows. False, with nothing
/// launched, unless count is a multiple of 32.
bool SigmoidMulQ8_1(float* x, const float* g, std::size_t count, void* x_q8,
                    hipStream_t stream);

/// out[t][m] = sum_k W[m][k] * x[t][k] for F32/BF16/F16 weights; meant for
/// the narrow projections (routers, alpha/beta, indexer, inject) that the
/// quantized tier does not cover.
/// Converts `count` floats to BF16 (or F16) for a 16-bit hipBLAS GEMM.
void NarrowActivations(const float* x, void* out, bool bf16, std::size_t count,
                       hipStream_t stream);

/// W8A8 route for wide batches over Q8_0 weights: activations quantized per
/// 32-wide block into the tiled fragment layout (`Q8TiledBytes(batch, k)`
/// bytes, padded to the GEMM's 128-token macro tile), then an int8 WMMA
/// GEMM. out is [batch][m]. W8A8Gemm returns false, launching nothing, when
/// k is not a multiple of 32.
std::size_t Q8TiledBytes(std::size_t batch, std::size_t k);
void QuantizeQ8Tiled(const float* x, void* out, std::size_t batch,
                     std::size_t k, hipStream_t stream);
bool W8A8Gemm(const void* w, const void* x_tiled, float* out, std::size_t batch,
              std::size_t m, std::size_t k, hipStream_t stream);
/// Native wave64 launch, selected by W8A8Gemm for wide output projections.
void W8A8GemmWave64(const void* w, const void* x_tiled, float* out,
                    std::size_t batch, std::size_t m, std::size_t k,
                    hipStream_t stream);

/// Q8_0 HC down projection [320,10240], SiLU(x / 4), then F16 output.
/// Matches the separate operators' rounding. Supports at least 96 tokens.
bool HcDownF16Gemm(const void* w, const void* x_tiled, __half* out,
                   std::uint32_t n_tokens, hipStream_t stream);

/// Stacked Q8_0 QKV projection [13312,2560], head normalization and RoPE.
/// Writes Q/gates in F32 and K/V caches in F16, preserving separate rounding.
/// Fixed geometry: 24 query heads, two KV heads, 256 dimensions, 64 rotary.
/// Requires at least 1024 tokens; cache capacity must include position +
/// tokens.
bool AttentionF16Gemm(const void* weights, const __half* input,
                      const float* q_gamma, const float* k_gamma, float* query,
                      float* gate, __half* keys, __half* values,
                      std::uint32_t n_tokens, const std::uint32_t* position,
                      float theta, float eps, hipStream_t stream,
                      const qwen::vision::DeviceRope* rope = nullptr);

/// Exact Q8_0 HC up projection and F16-input mixer for the Flash Next
/// 2560-hidden, rank-320 geometry. Returns false below 96 tokens or for other
/// shapes. low_rank must not overlap mixed_half; side outputs are optional.
bool HcMixF16Gemm(const void* up, const __half* low_rank, const __half* xn,
                  const float* inject_w, float* mixed, __half* mixed_half,
                  void* mixed_q8, float* inject, std::uint32_t n_tokens,
                  std::uint32_t hidden, std::uint32_t rank, hipStream_t stream);

/// F16 activation rows [batch][k], Q8_0 weights dequantized to F16 in LDS,
/// F32 accumulation. out is [batch][m]. Unsupported shapes launch nothing.
bool UnquantizedF16Gemm(const void* w, const __half* x, float* out,
                        std::size_t batch, std::size_t m, std::size_t k,
                        hipStream_t stream);
/// input_stride is in half elements; zero means k. Padded rows are supported
/// by the wide SSM output projection (batch >= 1024, m=2560, k=6144).
bool DenseF16Gemm(const void* w, const __half* x, float* out, std::size_t batch,
                  std::size_t m, std::size_t k, hipStream_t stream,
                  std::size_t input_stride = 0);
/// BF16 weight rows [m][k] times BF16 activation rows [batch][k] with one
/// F32 K16 chain per output. Token t's chain starts ((t / 32) % 4) * 128
/// elements into K and wraps: hipBLASLt's MT32x32x64 order (BlasLt::Gemm).
/// out is [batch][m].
bool DenseBf16Gemm(const void* w, const void* x, float* out, std::size_t batch,
                   std::size_t m, std::size_t k, hipStream_t stream);

/// SSM Q8_0 projection fused with its four-tap convolution. Supports
/// [m=16384,k=2560,channels=10240] and at least 1024 tokens. qkvz retains
/// every Z row plus the QKV rows required for tile boundaries and rolling
/// history; other QKV rows are not written. convolved is [tokens][channels].
/// history is read only; pass convolved=true to GatedDeltaNet to consume
/// this result and update history. Speculative snapshots need all raw QKV
/// rows and must use DenseF16Gemm instead. Unsupported shapes launch nothing.
bool DenseF16SsmGemm(const void* w, const __half* x, const float* conv_w,
                     const float* history, float* qkvz, float* convolved,
                     std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
                     std::uint32_t channels, std::uint32_t kernel,
                     hipStream_t stream, std::uint32_t checkpoint_tokens = 0);

/// Routed expert GEMMs. RoutedCompact sorts the (token, slot) assignments
/// by expert into `rows_token`/`rows_slot` (RoutedCompactRows(slots,
/// experts) entries, -1 for padding) with every bucket padded to 16 rows
/// (`pad_bounds`, experts + 1 entries) from the per-expert `counts`.
std::size_t RoutedCompactRows(std::size_t slots, std::size_t n_experts);
void RoutedCompact(const std::int32_t* ids, const std::uint32_t* counts,
                   std::int32_t* pad_bounds, std::int32_t* cursors,
                   std::int32_t* rows_token, std::int32_t* rows_slot,
                   std::uint32_t n_tokens, std::uint32_t k,
                   std::uint32_t n_experts, hipStream_t stream);

/// Routed expert GEMM on the F16 WMMA cores over F16 activation rows
/// `x` ([rows][k]): the weights are dequantized to F16 after the LDS read
/// and the whole K extent accumulates in F32. `tiles` (n_tiles entries)
/// packs each launched (expert, 48-row token tile) as expert | tile << 16,
/// so empty tiles cost nothing. Exactly one of `out` (F32) and `out_half`
/// (F16) receives the result; a non-null `swiglu_gate` (the gate
/// projection's F32 output over the same rows) turns it into
/// silu(gate) * result.
/// `tile_rows` is the token rows per tile the map was built with: 16 or 48,
/// or 64 for Q5_1/Q8_0.
/// Q4_K and Q5_K need k % 256 == 0, Q5_1 and Q8_0 k % 64 == 0; other types
/// and tile widths return false.
bool RoutedF16Gemm(const void* w, WeightType type, const __half* x,
                   const std::int32_t* tiles, std::uint32_t n_tiles,
                   std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                   const std::int32_t* rows_in, const std::int32_t* rows_out,
                   const float* swiglu_gate, float* out, __half* out_half,
                   std::size_t m, std::size_t k, hipStream_t stream);

/// Paired Q4_K/Q5_K gate/up GEMM with SwiGLU and F16 output. Uses the same
/// routing layout as RoutedF16Gemm, with 64 or 128 token rows per tile. The two
/// projections share one launch and never materialize the gate output.
bool RoutedGatedF16Gemm(const void* gate, const void* up, WeightType type,
                        const __half* x, const std::int32_t* tiles,
                        std::uint32_t n_tiles, std::uint32_t tile_rows,
                        const std::int32_t* pad_bounds,
                        const std::int32_t* rows_in,
                        const std::int32_t* rows_out, __half* out,
                        std::size_t m, std::size_t k, hipStream_t stream);

/// F32/BF16/F16 projection in groups of up to eight token rows. Each token
/// keeps the same accumulation order at every batch width.
void SmallGemm(const void* w, WeightType type, const float* x, float* out,
               std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
               hipStream_t stream);

/// PLE gate: for every stream s, sc = dot(key_n[t][s], query_n[t][s]) /
/// sqrt(hidden); gated[t][s][i] = value[t][i] * sigmoid(sgn(sc) *
/// sqrt(max(|sc|, 1e-6))).
void PleGate(const float* key_n, const float* query_n, const float* value,
             float* gated, std::uint32_t n_tokens, std::uint32_t hidden,
             std::uint32_t streams, hipStream_t stream);

/// Dilated depthwise causal conv over tokens with SiLU:
/// out[t][c] = silu(sum_k w[c*kernel+k] * in[t - (kernel-1-k)*dilation][c]),
/// where tokens before the chunk come from `history` ([hist][channels],
/// oldest first). `history` is then advanced past the chunk.
/// `snapshots`, when populated, receives the history as it stands after
/// each proper prefix: [n_tokens-1][hist][channels]. The full batch's
/// history remains in `history`.
void PleConv(const float* in, const float* w, float* history,
             float* history_scratch, float* out, RollbackRows snapshots,
             std::uint32_t n_tokens, std::uint32_t channels,
             std::uint32_t kernel, std::uint32_t dilation, hipStream_t stream);

/// res[t][c] += gated[t][c] + conv[t][c] over hc_dim channels.
void PleInject(float* res, const float* gated, const float* conv,
               std::size_t count, hipStream_t stream);

/// Copies rolling history at a prefix without advancing the live history.
void HistoryPrefix(const float* in, std::uint32_t stride, const float* history,
                   float* destination, std::uint32_t tokens,
                   std::uint32_t channels, std::uint32_t history_rows,
                   hipStream_t stream);

struct GdnCheckpoint {
  float* state = nullptr;
  float* history = nullptr;
  std::uint32_t tokens = 0;
};

/// Gated DeltaNet over a chunk of tokens for one layer. Runs the causal
/// conv (with rolling `conv_state`, [kernel-1][channels]) and the recurrence
/// on `state` ([v_heads][d][d]) sequentially over tokens, parallel over heads
/// and value dims. `alpha_beta` holds both projections per token:
/// [t][alpha(v_heads) | beta(v_heads)]. Outputs the normalized,
/// sigmoid-gated attention rows [tokens][v_heads*d] ready for the output
/// projection.
/// `state_snapshots` ([n_tokens-1][v_heads*d*d]) and `conv_snapshots`
/// ([n_tokens-1][(kernel-1)*channels]), when populated, receive the state after
/// each proper prefix. The live state already holds the full batch.
/// Scratch: `conv_scratch` ((n_tokens + kernel) * channels), `qn`/`kn`
/// (n_tokens * k_heads * d), `raw` (n_tokens * v_heads * d).
/// `qkv` rows are `qkv_stride` floats apart and `z` rows `z_stride`, so a
/// stacked [qkv|z] projection feeds both without unpacking. A non-null
/// `out_q8` receives the rows quantized into the W8A8 tiled layout
/// (K = v_heads * d) instead of `out`. With null `out_q8`, a non-null
/// `out_half` receives F16 rows instead of `out`. With both null, a non-null
/// `out_q8_1` also receives the rows' unpadded Q8_1 rows (as
/// qfn_mmq_quantize_q8_1 writes them; d == 128). `convolved` consumes
/// convolution rows already in conv_scratch; it requires null speculative
/// snapshots.
void GatedDeltaNet(const float* qkv, std::uint32_t qkv_stride, const float* z,
                   std::uint32_t z_stride, const float* alpha_beta,
                   const float* conv_w, const float* a, const float* dt,
                   const float* norm_w, float* conv_state, float* conv_scratch,
                   float* qn, float* kn, float* raw, float* state, float* out,
                   void* out_q8, RollbackRows state_snapshots,
                   RollbackRows conv_snapshots, std::uint32_t n_tokens,
                   std::uint32_t k_heads, std::uint32_t v_heads,
                   std::uint32_t d, std::uint32_t kernel, bool row_split,
                   bool convolved, float eps, hipStream_t stream,
                   __half* out_half = nullptr, GdnCheckpoint checkpoint = {},
                   std::uint32_t out_half_stride = 0, void* out_q8_1 = nullptr);

/// Private rows for one request in a decode batch. Scratch regions and all
/// recurrent/history/rollback buffers must be disjoint between requests.
struct GdnBatchItem {
  const float* qkv;
  const float* z;
  const float* alpha_beta;
  float* conv_state;
  float* conv_scratch;
  float* qn;
  float* kn;
  float* raw;
  float* state;
  float* out;
  RollbackRows state_snapshots;
  RollbackRows conv_snapshots;
  std::uint32_t n_tokens;
};

/// Runs the decode arithmetic for 1–8 independent requests of 1–8 rows,
/// 128-wide heads and four convolution taps. `items` is device-accessible;
/// inactive requests are untouched. The caller zeros their output rows.
bool GatedDeltaNetBatch(const GdnBatchItem* items, std::uint32_t count,
                        std::uint32_t max_tokens, std::uint32_t active,
                        std::uint32_t qkv_stride, std::uint32_t z_stride,
                        const float* conv_w, const float* a, const float* dt,
                        const float* norm_w, std::uint32_t k_heads,
                        std::uint32_t v_heads, float eps, hipStream_t stream);

/// Splits the interleaved [q|gate] projection (rows `qg_stride` apart) into
/// q [t][heads][d] and gate [t][heads*d]. With non-null `k`, the row
/// continues with k and v (`kv_width` each), copied out contiguously.
void UnpackQGate(const float* qg, std::uint32_t qg_stride, float* q,
                 float* gate, float* k, float* v, std::uint32_t n_tokens,
                 std::uint32_t heads, std::uint32_t d, std::uint32_t kv_width,
                 hipStream_t stream);

/// Unpacks a stacked Q/gate/K/V projection, normalizes and rotates Q/K,
/// and writes the F16 caches. Returns false for heads wider than 256.
/// With zero query heads, packed contains only K/V and q/gate may be null.
bool PrepareAttention(const float* packed, std::uint32_t stride,
                      const float* q_gamma, const float* k_gamma, float* q,
                      float* gate, __half* k_cache, __half* v_cache,
                      std::uint32_t n_tokens, std::uint32_t heads,
                      std::uint32_t kv_heads, std::uint32_t d,
                      std::uint32_t rotary_dim, const std::uint32_t* start_pos,
                      float theta, float eps, hipStream_t stream,
                      const qwen::vision::DeviceRope* rope = nullptr,
                      bool prefill = false);

/// NEOX partial rotary on x [t][heads][d] at positions start_pos + t.
/// Positions are read from device memory (`start_pos` points at the
/// session's control block) so a captured decode graph replays at any
/// position.
void Rope(float* x, std::uint32_t n_tokens, std::uint32_t heads,
          std::uint32_t d, std::uint32_t rotary_dim,
          const std::uint32_t* start_pos, float theta, hipStream_t stream,
          const qwen::vision::DeviceRope* rope = nullptr);

/// Stores f32 rows into the f16 cache at positions start_pos + t:
/// cache[(start_pos + t)][row_dim].
void StoreKv(const float* src, __half* cache, std::uint32_t n_tokens,
             std::uint32_t row_dim, const std::uint32_t* start_pos,
             hipStream_t stream);
/// Stores raw indexer rows in a power-of-two ring of `capacity` rows.
void StoreRows(const float* src, float* dst, std::uint32_t n_tokens,
               std::uint32_t row_dim, const std::uint32_t* start_pos,
               std::uint32_t capacity, hipStream_t stream);

/// Pools raw indexer keys ([pos % capacity][dim] f32) of the blocks the batch
/// completes, [*first_block, (*start_pos + n_tokens) / ratio), into block
/// keys: mean over `ratio` positions, RMSNorm with gamma, rotary at the
/// block start position, then round once to F16 for scoring.
/// `grid_blocks` bounds how many blocks one launch
/// can pool (a device-side bound is not available at launch time).
void PoolIndexerBlocks(const float* raw_keys, const float* gamma,
                       __half* blocks, const std::uint32_t* first_block,
                       const std::uint32_t* start_pos, std::uint32_t n_tokens,
                       std::uint32_t grid_blocks, std::uint32_t ratio,
                       std::uint32_t dim, std::uint32_t rotary_dim, float theta,
                       float eps, std::uint32_t capacity, hipStream_t stream,
                       const qwen::vision::DeviceRope* rope = nullptr);

/// Per query t (position *start_pos + first_token + t): scores every
/// complete block below its own tail, keeps the `budget` highest, and
/// writes a visibility bitmap (`mask_words` uint32 per query, bit b = block
/// b visible). Every block is visible when the count fits the budget.
/// Queries retain F32 precision; pooled cache keys are F16. Scores
/// still accumulate in FP32. `scores` holds n_tokens * max_blocks floats.
/// `live_blocks`, when nonzero, bounds the complete blocks of the last
/// query (eager launches only: a captured graph must cover max_blocks).
void SelectBlocks(const float* q, const __half* blocks, std::uint32_t* mask,
                  float* scores, std::uint32_t n_tokens,
                  const std::uint32_t* start_pos, std::uint32_t first_token,
                  std::uint32_t heads, std::uint32_t dim, std::uint32_t ratio,
                  std::uint32_t budget, std::uint32_t mask_words,
                  std::uint32_t max_blocks, hipStream_t stream,
                  std::uint32_t live_blocks = 0);

/// Per-token attention (decode and narrow batches). With `partials`
/// (n_tokens * heads * splits * (d + 2) floats) the key tiles are split
/// across `splits` blocks per row and merged in a second launch.
void Attention(const float* q, const __half* k_cache, const __half* v_cache,
               const std::uint32_t* mask, std::uint32_t mask_words, float* out,
               float* partials, std::uint32_t splits, std::uint32_t n_tokens,
               const std::uint32_t* start_pos, std::uint32_t heads,
               std::uint32_t kv_heads, std::uint32_t d, std::uint32_t ratio,
               hipStream_t stream);

/// Fused causal attention on the WMMA cores for wide batches: scores, online
/// softmax, PV and the sigmoid output gate in one launch. `mask` follows
/// Attention's block-selection contract (null for the dense window). Returns
/// false, launching nothing, when the geometry is not the model's 24 x 256
/// heads over two KV heads. `last_only` computes only the final dense query
/// tile, retaining its key sweep and leaving earlier output rows untouched.
bool WmmaCausalAttention(const float* q, const float* gate,
                         const __half* k_cache, const __half* v_cache,
                         const std::uint32_t* mask, std::uint32_t mask_words,
                         float* out, std::uint32_t n_tokens,
                         std::uint32_t start_pos, std::uint32_t heads,
                         std::uint32_t kv_heads, std::uint32_t d,
                         std::uint32_t ratio, hipStream_t stream,
                         bool last_only = false);

/// counts[e] = number of (token, slot) pairs routed to expert e.
void ExpertCounts(const std::int32_t* ids, std::uint32_t* counts,
                  std::uint32_t n_tokens, std::uint32_t n_experts,
                  std::uint32_t k, hipStream_t stream);

/// Softmax over `n_experts` logits per token (rows `stride` apart), top-k
/// selection, renormalized weights. ids [t][k] int32, weights [t][k] f32.
void RouterTopK(const float* logits, std::uint32_t stride, std::int32_t* ids,
                float* weights, std::uint32_t n_tokens, std::uint32_t n_experts,
                std::uint32_t k, hipStream_t stream);

/// out[t][i] = sum_s weights[t][s] * expert_out[t*k + s][i]
///           + sigmoid(gate[t * gate_stride]) * shared[t][i].
void MoeEpilogue(const float* expert_out, const float* weights,
                 const float* shared, const float* gate,
                 std::uint32_t gate_stride, float* out, std::uint32_t n_tokens,
                 std::uint32_t k, std::uint32_t dim, hipStream_t stream);
/// Same contract, four adjacent lanes per thread (`dim % 4 == 0`, else the
/// scalar kernel runs).
void MoeEpilogueVec4(const float* expert_out, const float* weights,
                     const float* shared, const float* gate,
                     std::uint32_t gate_stride, float* out,
                     std::uint32_t n_tokens, std::uint32_t k, std::uint32_t dim,
                     hipStream_t stream);
/// MoeEpilogueVec4 over F16 expert rows.
void MoeEpilogueVec4F16(const __half* expert_out, const float* weights,
                        const float* shared, const float* gate,
                        std::uint32_t gate_stride, float* out,
                        std::uint32_t n_tokens, std::uint32_t k,
                        std::uint32_t dim, hipStream_t stream);

/// dst[t] = *row < 0 ? alt[t] : base[*row + t] (rows of `width` floats).
void MtpHidden(const float* base, const float* alt, const std::int32_t* row,
               float* dst, std::uint32_t n_tokens, std::uint32_t width,
               hipStream_t stream);

/// Add the projected embedding once to each projected hidden branch.
void MtpAddEmbedding(const float* embedding, float* residual,
                     std::uint32_t n_tokens, std::uint32_t hidden,
                     std::uint32_t streams, hipStream_t stream);

inline constexpr std::uint32_t kArgmaxParts = 64;
struct ArgmaxCandidate {
  float value;
  std::int32_t index;
};
/// Full-vocabulary argmax into out[t], with the lowest index winning ties.
/// Matches std::max_element, including NaNs. Scratch holds
/// n_tokens * kArgmaxParts candidates.
void Argmax(const float* logits, ArgmaxCandidate* scratch, std::int32_t* out,
            std::uint32_t n_tokens, std::uint32_t vocab, hipStream_t stream);

/// Gather selected IDs and their original logits from independent rows.
/// The caller checks finite values only for the verification prefix it visits.
// Sorted sparse histories, one per verification row. Penalty arithmetic is
// FP64, matching SamplerState; output values remain the original logits.
struct GreedyPenaltyRows {
  const sampling::TokenPenalty* penalties;
  std::uint32_t offsets[8]{};
};
struct PenaltyArgmaxCandidate {
  double value;
  std::int32_t index;
};
void PenalizedArgmax(const float* logits, GreedyPenaltyRows penalties,
                     float repeat, float frequency, float presence,
                     PenaltyArgmaxCandidate* partial, ArgmaxCandidate* out,
                     std::uint32_t rows, std::uint32_t vocab,
                     hipStream_t stream);

void GatherArgmaxCandidates(const float* logits, const std::uint32_t* ids,
                            ArgmaxCandidate* out, std::uint32_t rows,
                            std::uint32_t vocab, hipStream_t stream);

/// Number of IDs in each of the two selection buffers.
std::uint32_t MtpCandidateWorkspaceSize(std::uint32_t vocab);

/// Exact MTP top-64 selection, descending score with lowest-token-ID ties.
/// Nonfinite scores become -infinity. Both ID buffers have room for
/// MtpCandidateWorkspaceSize(vocab); scores receives min(vocab, 64) values.
/// Scores may reuse the ID workspace starting at offset 64.
/// All candidate operations are graph-safe.
void MtpTopCandidates(const float* logits, std::uint32_t* ids,
                      std::uint32_t* scratch_ids, float* scores,
                      std::uint32_t vocab, hipStream_t stream);

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KERNELS_HPP_
