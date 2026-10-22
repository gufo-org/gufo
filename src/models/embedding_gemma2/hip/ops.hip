#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/embedding_gemma2/hip/ops.hpp"

namespace gufo::models::embedding_gemma2::hip {
namespace {

constexpr unsigned int kBlockThreads = 256;
constexpr unsigned int kWaveSize = 32;

__device__ __forceinline__ float Bf16ToFloat(std::uint16_t bits) {
  return __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
}

// Rounds to nearest even. NaNs keep a quiet payload.
__device__ __forceinline__ std::uint16_t FloatToBf16(float value) {
  const std::uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16) | 0x40U);
  }
  const std::uint32_t rounding = 0x7fffU + ((bits >> 16) & 1U);
  return static_cast<std::uint16_t>((bits + rounding) >> 16);
}

__device__ __forceinline__ float GeluTanh(float x) {
  constexpr float kSqrt2OverPi = 0.7978845608028654F;
  const float inner = kSqrt2OverPi * (x + 0.044715F * x * x * x);
  return 0.5F * x * (1.0F + tanhf(inner));
}

// Sums `value` across the block. `shared` needs one slot per wave. Every
// thread receives the total.
__device__ float BlockSum(float value, float* shared) {
  for (int offset = kWaveSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor(value, offset, kWaveSize);
  }
  const unsigned int lane = threadIdx.x % kWaveSize;
  const unsigned int wave = threadIdx.x / kWaveSize;
  const unsigned int waves = (blockDim.x + kWaveSize - 1) / kWaveSize;
  __syncthreads();
  if (lane == 0) {
    shared[wave] = value;
  }
  __syncthreads();
  float total = 0.0F;
  for (unsigned int index = 0; index < waves; ++index) {
    total += shared[index];
  }
  return total;
}

__global__ void EmbeddingLookupKernel(const std::uint16_t* table,
                                      const std::uint32_t* token_ids,
                                      float* state, std::uint16_t* embeddings,
                                      std::size_t hidden, float scale) {
  const std::size_t token = blockIdx.x;
  const std::uint16_t* row =
      table + static_cast<std::size_t>(token_ids[token]) * hidden;
  for (std::size_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    // Mirrors the checkpoint's BF16 execution, which rounds the scaled
    // embedding. The CPU reference keeps the product in float32.
    const std::uint16_t scaled = FloatToBf16(Bf16ToFloat(row[i]) * scale);
    embeddings[token * hidden + i] = scaled;
    state[token * hidden + i] = Bf16ToFloat(scaled);
  }
}

__global__ void RmsNormToBf16Kernel(const float* input,
                                    const std::uint16_t* weight,
                                    std::uint16_t* output, std::size_t width,
                                    float epsilon) {
  __shared__ float shared[kBlockThreads / kWaveSize];
  const float* row = input + blockIdx.x * width;
  float squares = 0.0F;
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x) {
    squares += row[i] * row[i];
  }
  squares = BlockSum(squares, shared);
  const float scale = rsqrtf(squares / static_cast<float>(width) + epsilon);
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x) {
    output[blockIdx.x * width + i] =
        FloatToBf16(row[i] * scale * Bf16ToFloat(weight[i]));
  }
}

__global__ void RmsNormResidualKernel(const float* update,
                                      const std::uint16_t* weight, float* state,
                                      std::size_t width, float epsilon) {
  __shared__ float shared[kBlockThreads / kWaveSize];
  const float* row = update + blockIdx.x * width;
  float squares = 0.0F;
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x) {
    squares += row[i] * row[i];
  }
  squares = BlockSum(squares, shared);
  const float scale = rsqrtf(squares / static_cast<float>(width) + epsilon);
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x) {
    state[blockIdx.x * width + i] += row[i] * scale * Bf16ToFloat(weight[i]);
  }
}

__global__ void ScaledRmsNormKernel(float* values, const std::uint16_t* weight,
                                    std::size_t width, float scale,
                                    float epsilon) {
  __shared__ float shared[kBlockThreads / kWaveSize];
  float* row = values + blockIdx.x * width;
  float squares = 0.0F;
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x) {
    const float scaled = row[i] * scale;
    squares += scaled * scaled;
  }
  squares = BlockSum(squares, shared);
  const float norm = rsqrtf(squares / static_cast<float>(width) + epsilon);
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x) {
    row[i] = row[i] * scale * norm * Bf16ToFloat(weight[i]);
  }
}

__global__ void FloatToBf16Kernel(const float* input, std::uint16_t* output,
                                  std::size_t count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = FloatToBf16(input[index]);
  }
}

// One block per token and head slot. Slots enumerate the query heads, then
// the key heads, then the value heads of the fused projection row.
__global__ void QkvNormRopeKernel(
    const float* qkv, const std::uint16_t* query_norm,
    const std::uint16_t* key_norm, std::uint16_t* query, std::uint16_t* key,
    std::uint16_t* value, const std::uint32_t* positions,
    std::uint32_t num_heads, std::uint32_t num_key_value_heads,
    std::uint32_t head_dim, const float* rope_cos, const float* rope_sin,
    float epsilon) {
  __shared__ float shared[kBlockThreads / kWaveSize];
  const std::size_t token = blockIdx.x;
  const std::uint32_t slot = blockIdx.y;
  const std::size_t slots = num_heads + 2U * num_key_value_heads;
  const float* head = qkv + (token * slots + slot) * head_dim;
  const std::uint32_t half_dim = head_dim / 2U;
  const std::uint32_t i = threadIdx.x;

  float a = head[i];
  float b = head[i + half_dim];
  const float scale = rsqrtf(
      BlockSum(a * a + b * b, shared) / static_cast<float>(head_dim) + epsilon);
  a *= scale;
  b *= scale;

  if (slot >= num_heads + num_key_value_heads) {
    const std::size_t base = (token * num_key_value_heads +
                              (slot - num_heads - num_key_value_heads)) *
                             head_dim;
    value[base + i] = FloatToBf16(a);
    value[base + i + half_dim] = FloatToBf16(b);
    return;
  }

  const bool is_query = slot < num_heads;
  const std::uint16_t* weight = is_query ? query_norm : key_norm;
  a *= Bf16ToFloat(weight[i]);
  b *= Bf16ToFloat(weight[i + half_dim]);
  const std::size_t angle =
      static_cast<std::size_t>(positions[token]) * half_dim + i;
  const float cosine = rope_cos[angle];
  const float sine = rope_sin[angle];
  const float rotated_a = a * cosine - b * sine;
  const float rotated_b = b * cosine + a * sine;
  std::uint16_t* out =
      is_query
          ? query + (token * num_heads + slot) * head_dim
          : key + (token * num_key_value_heads + (slot - num_heads)) * head_dim;
  out[i] = FloatToBf16(rotated_a);
  out[i + half_dim] = FloatToBf16(rotated_b);
}

using Vec16 = __attribute__((__vector_size__(16))) unsigned int;

// One wave per (token, head). A lane owns HD / 32 elements held as 8-element
// chunks, so a row is read with coalesced 16-byte loads. Keys are consumed
// four at a time so their wave reductions overlap.
template<int HD>
__global__ __launch_bounds__(128) void AttentionKernel(
    const std::uint16_t* query, const std::uint16_t* key,
    const std::uint16_t* value, std::uint16_t* context,
    const std::uint32_t* sequence_begin, const std::uint32_t* sequence_end,
    std::uint32_t num_heads, std::uint32_t num_key_value_heads,
    std::uint32_t window) {
  constexpr int kChunks = HD / 256;
  constexpr int kPerLane = HD / kWaveSize;
  constexpr int kKeysPerStep = 4;

  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t head = blockIdx.y * blockDim.y + threadIdx.y;
  if (head >= num_heads) {
    return;
  }
  const std::size_t token = blockIdx.x;
  const std::uint32_t kv_head = head / (num_heads / num_key_value_heads);
  const std::int64_t begin = sequence_begin[token];
  const std::int64_t end = sequence_end[token];
  std::int64_t low = begin;
  std::int64_t high = end - 1;
  if (window != 0U) {
    low = max(begin, static_cast<std::int64_t>(token) - window);
    high = min(high, static_cast<std::int64_t>(token) + window);
  }

  float q[kPerLane];
  const Vec16* query_row =
      reinterpret_cast<const Vec16*>(query + (token * num_heads + head) * HD);
  for (int j = 0; j < kChunks; ++j) {
    const Vec16 packed = query_row[lane + kWaveSize * j];
    const std::uint16_t* bits = reinterpret_cast<const std::uint16_t*>(&packed);
    for (int e = 0; e < 8; ++e) {
      q[j * 8 + e] = Bf16ToFloat(bits[e]);
    }
  }

  float accumulator[kPerLane];
  for (int e = 0; e < kPerLane; ++e) {
    accumulator[e] = 0.0F;
  }
  float maximum = -INFINITY;
  float denominator = 0.0F;

  for (std::int64_t first = low; first <= high; first += kKeysPerStep) {
    float score[kKeysPerStep];
    for (int u = 0; u < kKeysPerStep; ++u) {
      float partial = 0.0F;
      if (first + u <= high) {
        const Vec16* key_row = reinterpret_cast<const Vec16*>(
            key + ((first + u) * num_key_value_heads + kv_head) * HD);
        for (int j = 0; j < kChunks; ++j) {
          const Vec16 packed = key_row[lane + kWaveSize * j];
          const std::uint16_t* bits =
              reinterpret_cast<const std::uint16_t*>(&packed);
          for (int e = 0; e < 8; ++e) {
            partial += q[j * 8 + e] * Bf16ToFloat(bits[e]);
          }
        }
      }
      score[u] = partial;
    }
    for (int offset = kWaveSize / 2; offset > 0; offset >>= 1) {
      for (int u = 0; u < kKeysPerStep; ++u) {
        score[u] += __shfl_xor(score[u], offset, kWaveSize);
      }
    }
    float step_maximum = maximum;
    for (int u = 0; u < kKeysPerStep; ++u) {
      if (first + u > high) {
        score[u] = -INFINITY;
      }
      step_maximum = fmaxf(step_maximum, score[u]);
    }
    const float correction = expf(maximum - step_maximum);
    maximum = step_maximum;
    denominator *= correction;
    for (int e = 0; e < kPerLane; ++e) {
      accumulator[e] *= correction;
    }
    for (int u = 0; u < kKeysPerStep; ++u) {
      if (first + u > high) {
        break;
      }
      const float weight = expf(score[u] - maximum);
      denominator += weight;
      const Vec16* value_row = reinterpret_cast<const Vec16*>(
          value + ((first + u) * num_key_value_heads + kv_head) * HD);
      for (int j = 0; j < kChunks; ++j) {
        const Vec16 packed = value_row[lane + kWaveSize * j];
        const std::uint16_t* bits =
            reinterpret_cast<const std::uint16_t*>(&packed);
        for (int e = 0; e < 8; ++e) {
          accumulator[j * 8 + e] += weight * Bf16ToFloat(bits[e]);
        }
      }
    }
  }

  const float inverse = 1.0F / denominator;
  Vec16* out_row =
      reinterpret_cast<Vec16*>(context + (token * num_heads + head) * HD);
  for (int j = 0; j < kChunks; ++j) {
    Vec16 packed;
    std::uint16_t* bits = reinterpret_cast<std::uint16_t*>(&packed);
    for (int e = 0; e < 8; ++e) {
      bits[e] = FloatToBf16(accumulator[j * 8 + e] * inverse);
    }
    out_row[lane + kWaveSize * j] = packed;
  }
}

__global__ void GegluKernel(const float* gate_up, std::uint16_t* output,
                            std::size_t width) {
  const float* row = gate_up + blockIdx.x * 2U * width;
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x) {
    output[blockIdx.x * width + i] =
        FloatToBf16(GeluTanh(row[i]) * row[width + i]);
  }
}

__global__ void PleGateKernel(const float* gate, const float* per_layer_input,
                              std::uint16_t* output, std::size_t count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = FloatToBf16(GeluTanh(gate[index]) * per_layer_input[index]);
  }
}

__global__ void PleResidualKernel(const float* update,
                                  const std::uint16_t* weight,
                                  const std::uint16_t* layer_scalar,
                                  float* state, std::size_t width,
                                  float epsilon) {
  __shared__ float shared[kBlockThreads / kWaveSize];
  const float* row = update + blockIdx.x * width;
  float squares = 0.0F;
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x) {
    squares += row[i] * row[i];
  }
  squares = BlockSum(squares, shared);
  const float norm = rsqrtf(squares / static_cast<float>(width) + epsilon);
  const float scalar = Bf16ToFloat(layer_scalar[0]);
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x) {
    float& target = state[blockIdx.x * width + i];
    target = (target + row[i] * norm * Bf16ToFloat(weight[i])) * scalar;
  }
}

constexpr unsigned int kMaxPoolWidth = 1024;

__global__ void MeanPoolNormalizeKernel(const float* projected,
                                        const std::uint32_t* offsets,
                                        float* output, std::size_t width) {
  __shared__ float shared[kBlockThreads / kWaveSize];
  const std::uint32_t begin = offsets[blockIdx.x];
  const std::uint32_t end = offsets[blockIdx.x + 1];
  constexpr unsigned int kSlots = kMaxPoolWidth / kBlockThreads;
  float sums[kSlots];
  float squares = 0.0F;
  for (unsigned int slot = 0; slot < kSlots; ++slot) {
    const std::size_t d = threadIdx.x + slot * kBlockThreads;
    float sum = 0.0F;
    if (d < width) {
      for (std::uint32_t t = begin; t < end; ++t) {
        sum += projected[static_cast<std::size_t>(t) * width + d];
      }
      sum /= static_cast<float>(end - begin);
    }
    sums[slot] = sum;
    squares += sum * sum;
  }
  const float norm = sqrtf(BlockSum(squares, shared));
  const float inverse = norm > 0.0F ? 1.0F / norm : 0.0F;
  for (unsigned int slot = 0; slot < kSlots; ++slot) {
    const std::size_t d = threadIdx.x + slot * kBlockThreads;
    if (d < width) {
      output[blockIdx.x * width + d] = sums[slot] * inverse;
    }
  }
}

}  // namespace

void LaunchEmbeddingLookup(const void* table_bf16,
                           const std::uint32_t* token_ids, float* state,
                           void* embeddings_bf16, std::size_t tokens,
                           std::size_t hidden, float scale,
                           hipStream_t stream) {
  EmbeddingLookupKernel<<<static_cast<unsigned int>(tokens), kBlockThreads, 0,
                          stream>>>(
      static_cast<const std::uint16_t*>(table_bf16), token_ids, state,
      static_cast<std::uint16_t*>(embeddings_bf16), hidden, scale);
}

void LaunchRmsNormToBf16(const float* input, const void* weight_bf16,
                         void* output_bf16, std::size_t rows, std::size_t width,
                         float epsilon, hipStream_t stream) {
  RmsNormToBf16Kernel<<<static_cast<unsigned int>(rows), kBlockThreads, 0,
                        stream>>>(
      input, static_cast<const std::uint16_t*>(weight_bf16),
      static_cast<std::uint16_t*>(output_bf16), width, epsilon);
}

void LaunchRmsNormResidual(const float* update, const void* weight_bf16,
                           float* state, std::size_t rows, std::size_t width,
                           float epsilon, hipStream_t stream) {
  RmsNormResidualKernel<<<static_cast<unsigned int>(rows), kBlockThreads, 0,
                          stream>>>(
      update, static_cast<const std::uint16_t*>(weight_bf16), state, width,
      epsilon);
}

void LaunchScaledRmsNorm(float* values, const void* weight_bf16,
                         std::size_t rows, std::size_t width, float scale,
                         float epsilon, hipStream_t stream) {
  ScaledRmsNormKernel<<<static_cast<unsigned int>(rows), kBlockThreads, 0,
                        stream>>>(
      values, static_cast<const std::uint16_t*>(weight_bf16), width, scale,
      epsilon);
}

void LaunchFloatToBf16(const float* input, void* output_bf16, std::size_t count,
                       hipStream_t stream) {
  FloatToBf16Kernel<<<static_cast<unsigned int>((count + kBlockThreads - 1) /
                                                kBlockThreads),
                      kBlockThreads, 0, stream>>>(
      input, static_cast<std::uint16_t*>(output_bf16), count);
}

void LaunchQkvNormRope(const float* qkv, const void* query_norm_bf16,
                       const void* key_norm_bf16, void* query_bf16,
                       void* key_bf16, void* value_bf16,
                       const std::uint32_t* positions, std::size_t tokens,
                       const QkvLayout& layout, const float* rope_cos,
                       const float* rope_sin, float epsilon,
                       hipStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(tokens),
                  layout.num_heads + 2U * layout.num_key_value_heads);
  QkvNormRopeKernel<<<grid, layout.head_dim / 2U, 0, stream>>>(
      qkv, static_cast<const std::uint16_t*>(query_norm_bf16),
      static_cast<const std::uint16_t*>(key_norm_bf16),
      static_cast<std::uint16_t*>(query_bf16),
      static_cast<std::uint16_t*>(key_bf16),
      static_cast<std::uint16_t*>(value_bf16), positions, layout.num_heads,
      layout.num_key_value_heads, layout.head_dim, rope_cos, rope_sin, epsilon);
}

bool LaunchAttention(const void* query_bf16, const void* key_bf16,
                     const void* value_bf16, void* context_bf16,
                     const std::uint32_t* sequence_begin,
                     const std::uint32_t* sequence_end, std::size_t tokens,
                     const QkvLayout& layout, std::uint32_t window,
                     hipStream_t stream) {
  if (layout.num_key_value_heads == 0 ||
      layout.num_heads % layout.num_key_value_heads != 0 || tokens == 0) {
    return false;
  }
  constexpr unsigned int kHeadsPerBlock = 4;
  const dim3 grid(static_cast<unsigned int>(tokens),
                  (layout.num_heads + kHeadsPerBlock - 1) / kHeadsPerBlock);
  const dim3 block(kWaveSize, kHeadsPerBlock);
  const auto* query = static_cast<const std::uint16_t*>(query_bf16);
  const auto* key = static_cast<const std::uint16_t*>(key_bf16);
  const auto* value = static_cast<const std::uint16_t*>(value_bf16);
  auto* context = static_cast<std::uint16_t*>(context_bf16);
  if (layout.head_dim == 256) {
    AttentionKernel<256><<<grid, block, 0, stream>>>(
        query, key, value, context, sequence_begin, sequence_end,
        layout.num_heads, layout.num_key_value_heads, window);
  } else if (layout.head_dim == 512) {
    AttentionKernel<512><<<grid, block, 0, stream>>>(
        query, key, value, context, sequence_begin, sequence_end,
        layout.num_heads, layout.num_key_value_heads, window);
  } else {
    return false;
  }
  return true;
}

void LaunchGeglu(const float* gate_up, void* output_bf16, std::size_t rows,
                 std::size_t width, hipStream_t stream) {
  GegluKernel<<<static_cast<unsigned int>(rows), kBlockThreads, 0, stream>>>(
      gate_up, static_cast<std::uint16_t*>(output_bf16), width);
}

void LaunchPleGate(const float* gate, const float* per_layer_input,
                   void* output_bf16, std::size_t count, hipStream_t stream) {
  PleGateKernel<<<static_cast<unsigned int>((count + kBlockThreads - 1) /
                                            kBlockThreads),
                  kBlockThreads, 0, stream>>>(
      gate, per_layer_input, static_cast<std::uint16_t*>(output_bf16), count);
}

void LaunchPleResidual(const float* update, const void* weight_bf16,
                       const void* layer_scalar_bf16, float* state,
                       std::size_t rows, std::size_t width, float epsilon,
                       hipStream_t stream) {
  PleResidualKernel<<<static_cast<unsigned int>(rows), kBlockThreads, 0,
                      stream>>>(
      update, static_cast<const std::uint16_t*>(weight_bf16),
      static_cast<const std::uint16_t*>(layer_scalar_bf16), state, width,
      epsilon);
}

bool LaunchMeanPoolNormalize(const float* projected,
                             const std::uint32_t* sequence_offsets,
                             float* output, std::size_t sequences,
                             std::size_t width, hipStream_t stream) {
  if (width > kMaxPoolWidth) {
    return false;
  }
  MeanPoolNormalizeKernel<<<static_cast<unsigned int>(sequences), kBlockThreads,
                            0, stream>>>(projected, sequence_offsets, output,
                                         width);
  return true;
}

}  // namespace gufo::models::embedding_gemma2::hip

#endif  // defined(ENGINE_ENABLE_HIP)
