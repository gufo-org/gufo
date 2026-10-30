#ifndef GUFO_MODELS_LFM3_HIP_OPS_HPP_
#define GUFO_MODELS_LFM3_HIP_OPS_HPP_
#include <hip/hip_runtime.h>

#include <span>

#include "src/models/lfm3/weights.hpp"
namespace gufo::models::lfm3::gpu {
void Embed(const Tensor& weight, const std::uint32_t* tokens, float* out, int n,
           hipStream_t stream);
// Prompt projections of n rows need SplitActivationBytes(n, K) here.
// row_max (one word per prompt row) lets the FFN down projection split its
// input while staging.
struct DenseWorkspace {
  void* quantized{};
  unsigned* row_max{};
};
std::size_t DenseWorkspaceBytes(std::size_t rows, std::size_t K);
struct Output {
  const Tensor* weight;
  float* out;
};
// True when these projections of n rows read one split prompt input.
bool SharesSplit(std::span<const Output> outputs, int n,
                 DenseWorkspace workspace);
// Projects one input through up to four weights with the same column count.
// Q8_0 prompts split the input once and share one launch. split_ready means
// the workspace already holds this input's split (see ResidualNorm).
void Project(std::span<const Output> outputs, const float* x, int n,
             hipStream_t stream, DenseWorkspace workspace,
             bool split_ready = false);
// silu(gate x) * (up x) into out; up_scratch holds up x when the projections
// cannot share the fused prompt launch.
void ProjectSwiGLU(const Tensor& gate, const Tensor& up, const float* x,
                   float* out, float* up_scratch, int n, hipStream_t stream,
                   DenseWorkspace workspace, bool split_ready = false);
// down(silu(gate x) * (up x)) into out, with the activation in hidden.
void ProjectFfn(const Tensor& gate, const Tensor& up, const Tensor& down,
                const float* x, float* hidden, float* up_scratch, float* out,
                int n, hipStream_t stream, DenseWorkspace workspace,
                bool split_ready = false);
// x += post_norm(branch) (branch may be null, post_norm may be null), then
// z = norm(x) unless norm is null. With split, z goes to the workspace as a
// split prompt input instead. Returns false when the width is unsupported.
bool ResidualNorm(const float* branch, const Tensor* post_norm, float* x,
                  const Tensor* norm, float* z, int n, int hidden, float eps,
                  bool split, hipStream_t stream, DenseWorkspace workspace);
void Dense(const Tensor& weight, const float* x, float* out, int n,
           hipStream_t stream, DenseWorkspace workspace = {});
void ShortConv(const float* projection, const float* weight, float* history,
               float* out, int n, int hidden, int width, hipStream_t stream);
// Writes the dim / 2 RoPE frequencies of theta for Rope.
void RopeFrequencies(float* out, int dim, float theta, hipStream_t stream);
void Rope(float* x, const float* frequencies, const float* factors, int n,
          int heads, int dim, int position, hipStream_t stream);
// Reads previous cached rows plus the current chunk. Cache writes follow this
// launch, so a sliding ring cannot overwrite history needed by early queries.
// Long histories split their keys across blocks through scratch of
// AttentionScratchBytes(heads, dim, context, rows) bytes, where rows bounds
// the queries of one call.
void Attention(const float* q, const float* k, const float* v,
               const float* gate, const float* cache_k, const float* cache_v,
               float* out, int n, int heads, int kv_heads, int dim,
               int position, int capacity, int window, hipStream_t stream,
               float* scratch = nullptr);
std::size_t AttentionScratchBytes(int heads, int dim, int context, int rows);
// Complete bidirectional prompt, without a persistent cache.
void BidirectionalAttention(const float* q, const float* k, const float* v,
                            const float* gate, float* out, int n, int heads,
                            int kv_heads, hipStream_t stream,
                            const std::int32_t* groups = nullptr);
// Packed row copies; preserves the tensor's encoded row bytes exactly.
void SelectRows(const Tensor& source, const std::uint32_t* rows, void* out,
                int count, hipStream_t stream);
void StoreKV(const float* k, const float* v, float* cache_k, float* cache_v,
             int n, int stride, int position, int capacity, hipStream_t stream);
}  // namespace gufo::models::lfm3::gpu
#endif
