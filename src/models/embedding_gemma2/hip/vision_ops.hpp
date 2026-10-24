#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_HIP_VISION_OPS_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_HIP_VISION_OPS_HPP_
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::embedding_gemma2::hip {
// Fixed released vision geometry: 768 channels, 12 heads of 64 channels,
// row-major patches, no padded keys. Void buffers and weights are BF16;
// the residual stream and GEMM outputs stay float32.

void LaunchVisionPixels(const float* pixels, void* bf16, std::size_t count,
                        hipStream_t stream);
void LaunchVisionPositions(const float* projected, const void* table,
                           float* state, std::size_t patches,
                           std::uint32_t columns, hipStream_t stream);
/// Head RMS normalization (unweighted for V) and independent x/y axial RoPE.
void LaunchVisionQkv(const float* qkv, const void* query_norm,
                     const void* key_norm, void* query, void* key, void* value,
                     std::size_t patches, std::uint32_t columns,
                     hipStream_t stream);
void LaunchVisionAttention(const void* query, const void* key,
                           const void* value, void* context,
                           std::size_t patches, hipStream_t stream);
/// Spatial 3x3 average, sqrt(768) scaling and projector input RMS norm.
void LaunchVisionPool(const float* state, void* pooled, std::uint32_t rows,
                      std::uint32_t columns, hipStream_t stream);
/// Replace text placeholder rows in both the residual and PLE input copy.
void LaunchVisionInsert(const float* features, float* state, void* embeddings,
                        std::size_t tokens, hipStream_t stream);
}  // namespace gufo::models::embedding_gemma2::hip
#endif
