#ifndef GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_OPS_HPP_
#define GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_OPS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Fused per-token operators for the GLM 5.3 Flash (glm5-next) trunk that
/// are neither a pure GEMV nor attention: RMS/Layer norms, the mHC mixing
/// coefficients (sinkhorn), the KDA conv/gate/recurrence step, the DSA
/// indexer pooling score and top-pool selection, MoE routing and the
/// swiglu-clamp epilogue. Formula semantics follow
/// docs/models/glm-5.3-flash/IMPLEMENTATION.md verbatim; the parity test
/// checks every operator against an independent host double reference.
/// All buffers are F32 unless stated; the layout contract (head-major
/// channels, ch = head * head_dim + d) is fixed here and used unchanged by
/// the executor. Nothing here is shared with another model.
namespace gufo::models::glm5_next::rocm {

/// data[t] += bias[t % bias_len] for t in [0, total). Used for the KDA
/// ssm_dt.bias, router exp_probs_b.bias and other per-channel biases the
/// GEMV tier does not fold. Asynchronous on `stream` (null = default).
void AddBias(float* data, const float* bias, std::uint32_t total,
             std::uint32_t bias_len, hipStream_t stream);

/// dst[r*n + i] = w[i] * x[r*x_stride + i] * rsqrt(mean(x^2) + eps) for
/// rows rows of n elements. Fused (sum, scale, write) in one launch per row.
void RmsNorm(const float* x, const float* w, float* dst, std::uint32_t rows,
             std::uint32_t n, std::uint32_t x_stride, std::uint32_t dst_stride,
             float eps, hipStream_t stream);

/// LayerNorm of one row: dst[i] = w[i] * (x[i] - mean) / sqrt(var + eps) +
/// b[i] (the indexer k_norm, eps 1e-6; mean over n).
void LayerNorm(const float* x, const float* w, const float* b, float* dst,
               std::uint32_t n, float eps, hipStream_t stream);

/// mHC coefficients for one sub-layer (per token). mixes[24] is the
/// hc_*_fn GEMV output; role r in {pre (mixes[0:4]), post (mixes[4:8]),
/// comb (mixes[8:24])} applies affine mixes[m] * scale[r] + base[m], then
/// pre = sigmoid + eps, post = sigmoid * 2 and comb goes through the
/// Sinkhorn iterations: row softmax over dst (the row is one src stream),
/// column +eps normalize, repeated with plain row/col +eps normalization.
/// comb is stored row-major [src][dst]; the write-back consumes it as
/// comb[dst ...][src]. Single launch, one block.
void HcCoeffs(const float* mixes, const float* scale, const float* base,
              std::uint32_t sinkhorn_iters, float eps, float* pre, float* post,
              float* comb, hipStream_t stream);

/// Block input of one sub-layer: dst[i] = sum_src streams[src][i] * pre[src]
/// (hc_count four-stream mix into a single vector).
void HcInputs(const float* streams, const float* pre, std::uint32_t hc_count,
              std::uint32_t n, float* dst, hipStream_t stream);

/// Stream write-back around one sub-layer: out[dst][i] = post[dst] * y[i] +
/// sum_src res[src][i] * comb[src][dst], with comb as produced by HcCoeffs.
/// `out` may alias `res` (registers first); it must not alias `y`.
void HcWriteback(const float* y, const float* res, const float* post,
                 const float* comb, std::uint32_t hc_count, std::uint32_t n,
                 float* out, hipStream_t stream);

/// Causal short conv for one KDA projection channel set, one token.
/// `raw[n]` is the uncached projection output for the current position p,
/// `w[4*n]` the conv weight in the GGUF storage order [ne0 = taps], i.e.
/// channel-major w[ch * 4 + tap], `hist[3*n]` the ring of the previous
/// three positions' raw values with slot p % 3 holding position p-3.
/// dst[ch] = silu(sum_tap w[ch][tap] * x[p-3+tap]) where x[p] is `raw`;
/// after the read the ring slot p % 3 is overwritten with raw so position
/// p+1 sees it as p-2. Callers run it once per q/k/v.
void KdaConvStep(std::uint64_t pos, const float* raw, const float* w,
                 float* hist, std::uint32_t n, float* dst, hipStream_t stream);

/// Per head: fold the decay into the state and reduce the key dot.
/// sk[h][j] = sum_i k[h][i] * (state[h][i][j] * eg[h][i]) and the state is
/// updated in place to state * eg (the first two recurrence steps; dims
/// [heads][d][d], head-major channels throughout).
void KdaFoldSk(float* state, const float* eg, const float* k,
               std::uint32_t heads, std::uint32_t d, float* sk,
               hipStream_t stream);

/// Per head, completing the recurrence: d[j] = beta[h] * (v[h][j] - sk[h][j]),
/// state[i][j] += k[h][i] * d[j], o[h][j] = sum_i state[i][j] * q[h][i].
void KdaUpdateOut(float* state, const float* k, const float* q, const float* v,
                  const float* beta, const float* sk, std::uint32_t heads,
                  std::uint32_t d, float* o, hipStream_t stream);

/// KDA output gate per head: y[h*d + i] = w[i] * o * rsqrt(o^2.mean + eps)
/// * sigmoid(g2[h*d + i]) (ssm_norm over d plus the forget-gate sigmoid).
void KdaOutGate(const float* o, const float* w, const float* g2, float* dst,
                std::uint32_t heads, std::uint32_t d, float eps,
                hipStream_t stream);

/// DSA indexer score of every pool group. Pools hold `kpool` (4) cached
/// rows; a row packs [ik | gate] (gate immediately after ik) and the rows
/// of a pool are consecutive, so `pool_stride` is the stride between token
/// rows and a pool starts at pool[(P * kpool) * pool_stride]. For pool P,
/// dim j: probs = softmax over the present taps of (ape[tap * key_len + j]
/// + gate_tap[j]), k_pool[j] = sum probs * ik_tap[j] (taps beyond the token
/// count are dropped from the softmax; the ape tensor is stored tap-major,
/// [kpool][key_len]); scores[P] = q_scale * sum_h w[h] *
/// relu(dot(iq[h], k_pool[h * key_len : (h+1) * key_len])), with iq laid
/// out [heads][key_len] head-major. `tokens` is the number of cached tokens
/// (pools = ceil(tokens / kpool)); the last pool keeps only its
/// `tokens % kpool` (or kpool) valid rows.
void IdxScore(const float* pool, std::uint32_t pool_stride,
              std::uint32_t tokens, std::uint32_t kpool, std::uint32_t key_len,
              std::uint32_t heads, const float* iq, const float* ape,
              const float* w, float q_scale, float* scores, hipStream_t stream);

/// Stable top-k selection: writes the indices of the `k` largest scores,
/// ordered by score descending and by ascending index on ties (the order a
/// stable descending ggml_argsort produces). Requires k <= 2048 and reads
/// scores[0, n). `scratch` needs n + 4096 uint64 slots.
void TopKDesc(const float* scores, std::uint32_t n, std::uint32_t k,
              std::int32_t* out, std::uint64_t* scratch, hipStream_t stream);

/// MoE routing from the router logits (reference Gate.forward): the top
/// `top_k` experts are selected on sigmoid(logits) + bias (ties resolve to
/// the lower id, stable descending) but the weights are the *unbiased*
/// sigmoid values, renormalized by max(sum, 6.103515625e-5) (the F16
/// minimum the reference clamps to; no epsilon add) and multiplied by
/// `scale` (2.5, applied after the division). ids are int32. Requires
/// experts <= 1024; one block.
void MoERoute(const float* logits, const float* bias, std::uint32_t experts,
              std::uint32_t top_k, float scale, float* weights,
              std::int32_t* ids, hipStream_t stream);

/// x[i] *= s for i in [0, total).
void Scale(float* x, std::uint32_t total, float s, hipStream_t stream);

/// x[i] = 1 / (1 + exp(-x[i])) for i in [0, total) in place (the beta
/// gate; same expression as the reference ggml_sigmoid).
void SigmoidInPlace(float* x, std::uint32_t total, hipStream_t stream);

/// KDA decay gate: eg[h*d + i] = expf(lower_bound *
/// sigmoid(-(g1[h*d + i] * a[h]))). g1 is the dt-biased ssm_f_b output
/// (head-minor, channel = h*d + i), a[heads] ssm_a (= -exp(A_log)),
/// lower_bound the hparams gate lower bound (-5.0). KdaFoldSk consumes
/// eg directly (the reference's exp is folded here).
void KdaEGate(const float* g1, const float* a, std::uint32_t heads,
              std::uint32_t d, float lower_bound, float* eg,
              hipStream_t stream);

/// Per-head FLA l2 norm, in place, of `heads` rows of `d`:
/// x[h][i] = x[h][i] / sqrt(sum_j x[h][j]^2 + eps) (identical to
/// build_gdn_l2_norm = rms_norm(x, eps/n) / sqrt(n) for eps 1e-6). The
/// reference scales q by an extra 1/sqrt(d) after this; the caller does
/// that with Scale.
void L2NormHeads(float* x, std::uint32_t heads, std::uint32_t d, float eps,
                 hipStream_t stream);

/// Attention sel indices for one token of the kpool DSA path: expand the
/// `k` selected pools (TopKDesc order) pool-major with taps ascending,
/// sel[4*j + t] = 4*pool_sel[j] + t, then append the `n_tail` incomplete
/// tail slots in live order, sel[4*k + u] = pos - n_tail + 1 + u
/// (indexer_kpool_select_tail is on: hparams default, key absent in the
/// artifact). Writes 4*k + n_tail int32 rows; the last pool is always
/// complete, so no dead slots appear.
void BuildSel(const std::int32_t* pool_sel, std::uint32_t k, std::int32_t pos,
              std::uint32_t n_tail, std::int32_t* sel, hipStream_t stream);

/// SwiGLU with the reference clamp pair, elementwise over m * n:
/// g = min(gate, clamp_gate); out = g * sigmoid(g) * clamp(up, -c, c) with
/// c = clamp_up. Row strides allow expert-stack slices.
void SwigluClamp(const float* gate, const float* up, float* dst,
                 std::uint32_t m, std::uint32_t n, std::uint32_t gate_stride,
                 std::uint32_t up_stride, std::uint32_t dst_stride,
                 float clamp_gate, float clamp_up, hipStream_t stream);

/// dst[i] += sum_r weights[r] * src[r * stride + i] (accumulating the
/// weighted expert outputs; null `weights` means all 1).
void WeightedSumAdd(float* dst, const float* src, const float* weights,
                    std::uint32_t rows, std::uint32_t n, std::uint32_t stride,
                    hipStream_t stream);

}  // namespace gufo::models::glm5_next::rocm

#endif  // GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_OPS_HPP_
