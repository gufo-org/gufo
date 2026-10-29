#ifndef GUFO_MODELS_LFM3_HIP_OPS_HPP_
#define GUFO_MODELS_LFM3_HIP_OPS_HPP_
#include <hip/hip_runtime.h>

#include "src/models/lfm3/weights.hpp"
namespace gufo::models::lfm3::gpu {
void Embed(const Tensor& weight, const std::uint32_t* tokens, float* out, int n,
           hipStream_t stream);
// Prompt projections of n rows need SplitActivationBytes(n, K) here.
struct DenseWorkspace {
  void* quantized{};
};
std::size_t DenseWorkspaceBytes(std::size_t rows, std::size_t K);
void Dense(const Tensor& weight, const float* x, float* out, int n,
           hipStream_t stream, DenseWorkspace workspace = {});
void ShortConv(const float* projection, const float* weight, float* history,
               float* out, int n, int hidden, int width, hipStream_t stream);
void Rope(float* x, const float* factors, int n, int heads, int dim,
          int position, float theta, hipStream_t stream);
// Reads previous cached rows plus the current chunk. Cache writes follow this
// launch, so a sliding ring cannot overwrite history needed by early queries.
void Attention(const float* q, const float* k, const float* v,
               const float* gate, const float* cache_k, const float* cache_v,
               float* out, int n, int heads, int kv_heads, int dim,
               int position, int capacity, int window, hipStream_t stream);
void StoreKV(const float* k, const float* v, float* cache_k, float* cache_v,
             int n, int stride, int position, int capacity, hipStream_t stream);
}  // namespace gufo::models::lfm3::gpu
#endif
