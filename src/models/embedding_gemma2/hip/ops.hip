#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "ops.hpp"

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
using Fragment = __attribute__((ext_vector_type(16))) unsigned short;
using Accumulator = __attribute__((ext_vector_type(8))) float;

// Loads 16 contiguous BF16 values. `source` must be 16-byte aligned.
__device__ __forceinline__ Fragment LoadFragment(const std::uint16_t* source) {
  union {
    Fragment fragment;
    Vec16 words[2];
  } value;
  value.words[0] = *reinterpret_cast<const Vec16*>(source);
  value.words[1] = *reinterpret_cast<const Vec16*>(source + 8);
  return value.fragment;
}

// Returns the value held by the lane 16 positions away, in the other
// half-wave.
__device__ __forceinline__ float SwapHalves(float value) {
  return __uint_as_float(__builtin_amdgcn_permlanex16(
      __float_as_uint(value), __float_as_uint(value), 0x76543210U, 0xfedcba98U,
      false, false));
}

constexpr float kLog2e = 1.4426950408889634F;

// Transposes values from [token][column] to [column][token] through LDS so
// both the reads and the writes are coalesced. Tokens at or beyond `tokens`
// up to `ValueStride(tokens)` are written as zero.
__global__ void TransposeValuesKernel(const std::uint16_t* value,
                                      std::uint16_t* value_t,
                                      std::uint32_t tokens,
                                      std::uint32_t columns,
                                      std::uint32_t value_stride) {
  __shared__ std::uint16_t tile[32][33];
  const std::uint32_t token0 = blockIdx.x * 32U;
  const std::uint32_t column0 = blockIdx.y * 32U;
  for (std::uint32_t row = threadIdx.y; row < 32U; row += blockDim.y) {
    const std::uint32_t token = token0 + row;
    tile[row][threadIdx.x] =
        token < tokens ? value[static_cast<std::size_t>(token) * columns +
                               column0 + threadIdx.x]
                       : static_cast<std::uint16_t>(0);
  }
  __syncthreads();
  for (std::uint32_t row = threadIdx.y; row < 32U; row += blockDim.y) {
    value_t[static_cast<std::size_t>(column0 + row) * value_stride + token0 +
            threadIdx.x] = tile[threadIdx.x][row];
  }
}

// Bidirectional (windowed) attention with matrix cores.
//
// A block of four waves owns 16 consecutive query tokens of one head and
// walks the keys in tiles of 32. Scores are held transposed, as keys by
// queries, so a lane owns one query and the softmax needs no transposes. The
// four waves split the head width. Each wave computes the partial scores over
// its quarter of the dot product, and the partials are summed through LDS.
// Each wave then accumulates the output for its own quarter of the columns,
// which keeps the accumulators at 32 to 64 registers per lane.
//
// Values arrive transposed ([kv head][column][token]) so a lane reads sixteen
// consecutive keys of one column. Scores use the attention scale of 1. The
// softmax is the online base-two form, with FP32 statistics and accumulators.
template<int HD>
__global__ __launch_bounds__(128) void WmmaAttentionKernel(
    const std::uint16_t* query, const std::uint16_t* key,
    const std::uint16_t* value_t, std::uint16_t* context,
    const std::uint32_t* sequence_begin, const std::uint32_t* sequence_end,
    std::uint32_t tokens, std::uint32_t value_stride, std::uint32_t num_heads,
    std::uint32_t num_key_value_heads, std::uint32_t window) {
  constexpr int kQuarter = HD / 4;
  constexpr int kChunks = kQuarter / 16;
  constexpr int kPartialStride = 20;  // 16 floats per lane plus padding

  __shared__ __align__(16) float partial[2][4][32][kPartialStride];

  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int column = lane & 15;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t kv_head = head / (num_heads / num_key_value_heads);
  const std::uint32_t first_query = blockIdx.x * 16U;
  const std::uint32_t query_token = first_query + column;
  const bool query_valid = query_token < tokens;

  // Key range of this lane's query. Bounds only grow with the token index, so
  // the tile's range runs from its first query's low to its last query's high.
  const auto low_of = [&](std::uint32_t token) {
    const std::int64_t begin = sequence_begin[token];
    return window == 0U ? begin
                        : max(begin, static_cast<std::int64_t>(token) - window);
  };
  const auto high_of = [&](std::uint32_t token) {
    const std::int64_t last =
        static_cast<std::int64_t>(sequence_end[token]) - 1;
    return window == 0U ? last
                        : min(last, static_cast<std::int64_t>(token) + window);
  };
  const std::int64_t low = query_valid ? low_of(query_token) : 1;
  const std::int64_t high = query_valid ? high_of(query_token) : 0;
  const std::int64_t tile_low = low_of(first_query);
  const std::int64_t tile_high = high_of(min(first_query + 15U, tokens - 1U));

  Fragment q[kChunks];
  for (int c = 0; c < kChunks; ++c) {
    if (query_valid) {
      q[c] = LoadFragment(
          query +
          (static_cast<std::size_t>(query_token) * num_heads + head) * HD +
          wave * kQuarter + c * 16);
    } else {
      q[c] = Fragment{};
    }
  }

  Accumulator result[kChunks];
  for (int t = 0; t < kChunks; ++t) {
    result[t] = Accumulator{};
  }
  float maximum = -INFINITY;
  float sum = 0.0F;

  int parity = 0;
  for (std::int64_t base = tile_low & ~std::int64_t{31}; base <= tile_high;
       base += 32) {
    Accumulator scores[2] = {Accumulator{}, Accumulator{}};
    for (int c = 0; c < kChunks; ++c) {
      for (int h = 0; h < 2; ++h) {
        std::int64_t row = base + h * 16 + column;
        row = min(row, static_cast<std::int64_t>(tokens) - 1);
        const Fragment k = LoadFragment(
            key +
            (static_cast<std::size_t>(row) * num_key_value_heads + kv_head) *
                HD +
            wave * kQuarter + c * 16);
        scores[h] =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(k, q[c], scores[h]);
      }
    }

    // Sum the four waves' partial scores in a fixed order, so every wave sees
    // identical values and keeps identical softmax statistics. The two buffers
    // alternate, so one barrier per tile separates the reads of one tile from
    // the writes of the next.
    float* mine = partial[parity][wave][lane];
    for (int h = 0; h < 2; ++h) {
      for (int i = 0; i < 8; i += 4) {
        *reinterpret_cast<float4*>(mine + h * 8 + i) = make_float4(
            scores[h][i], scores[h][i + 1], scores[h][i + 2], scores[h][i + 3]);
      }
    }
    __syncthreads();
    float total[16];
    for (int v = 0; v < 16; v += 4) {
      float4 sum4 =
          *reinterpret_cast<const float4*>(&partial[parity][0][lane][v]);
      for (int w = 1; w < 4; ++w) {
        const float4 other =
            *reinterpret_cast<const float4*>(&partial[parity][w][lane][v]);
        sum4.x += other.x;
        sum4.y += other.y;
        sum4.z += other.z;
        sum4.w += other.w;
      }
      total[v] = sum4.x;
      total[v + 1] = sum4.y;
      total[v + 2] = sum4.z;
      total[v + 3] = sum4.w;
    }
    parity ^= 1;

    // Mask to this query's key range and move to the base-two domain. Register
    // i of half h holds the key at offset 2 * i + (lane / 16) of that half.
    float next = maximum;
    for (int h = 0; h < 2; ++h) {
      for (int i = 0; i < 8; ++i) {
        const std::int64_t position = base + h * 16 + 2 * i + (lane >> 4);
        float score = total[h * 8 + i] * kLog2e;
        if (position < low || position > high) {
          score = -INFINITY;
        }
        total[h * 8 + i] = score;
        next = fmaxf(next, score);
      }
    }
    next = fmaxf(next, SwapHalves(next));
    // Until a query has seen a key, its maximum is -INFINITY. Shift by zero
    // then, so the exponentials below stay finite.
    const float shifted = next == -INFINITY ? 0.0F : next;
    const float correction = __builtin_amdgcn_exp2f(maximum - shifted);

    float subtotal = 0.0F;
    Fragment probabilities[2];
    for (int h = 0; h < 2; ++h) {
      for (int i = 0; i < 8; ++i) {
        const float p = __builtin_amdgcn_exp2f(total[h * 8 + i] - shifted);
        // The denominator sums the rounded weights the value product uses.
        const unsigned own = FloatToBf16(p);
        subtotal += Bf16ToFloat(static_cast<std::uint16_t>(own));
        const unsigned peer = __builtin_amdgcn_permlanex16(
            own, own, 0x76543210U, 0xfedcba98U, false, false);
        probabilities[h][2 * i] = lane < 16 ? own : peer;
        probabilities[h][2 * i + 1] = lane < 16 ? peer : own;
      }
    }
    subtotal += SwapHalves(subtotal);
    sum = sum * correction + subtotal;
    maximum = next;
    for (int t = 0; t < kChunks; ++t) {
      result[t] *= correction;
    }

    for (int t = 0; t < kChunks; ++t) {
      const std::size_t column_row = static_cast<std::size_t>(kv_head) * HD +
                                     wave * kQuarter + t * 16 + column;
      for (int h = 0; h < 2; ++h) {
        const Fragment v =
            LoadFragment(value_t + column_row * value_stride + base + h * 16);
        result[t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
            v, probabilities[h], result[t]);
      }
    }
  }

  if (query_valid) {
    const float inverse = 1.0F / sum;
    std::uint16_t* out =
        context +
        (static_cast<std::size_t>(query_token) * num_heads + head) * HD;
    for (int t = 0; t < kChunks; ++t) {
      for (int i = 0; i < 8; ++i) {
        out[wave * kQuarter + t * 16 + 2 * i + (lane >> 4)] =
            FloatToBf16(result[t][i] * inverse);
      }
    }
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

void LaunchTransposeValues(const void* value_bf16, void* value_t_bf16,
                           std::size_t tokens, std::uint32_t columns,
                           hipStream_t stream) {
  const dim3 grid(static_cast<unsigned int>((tokens + 31U) / 32U),
                  columns / 32U);
  TransposeValuesKernel<<<grid, dim3(32, 8), 0, stream>>>(
      static_cast<const std::uint16_t*>(value_bf16),
      static_cast<std::uint16_t*>(value_t_bf16),
      static_cast<std::uint32_t>(tokens), columns,
      static_cast<std::uint32_t>(ValueStride(tokens)));
}

bool LaunchAttention(const void* query_bf16, const void* key_bf16,
                     const void* value_t_bf16, void* context_bf16,
                     const std::uint32_t* sequence_begin,
                     const std::uint32_t* sequence_end, std::size_t tokens,
                     const QkvLayout& layout, std::uint32_t window,
                     hipStream_t stream) {
  if (layout.num_key_value_heads == 0 ||
      layout.num_heads % layout.num_key_value_heads != 0 || tokens == 0) {
    return false;
  }
  const dim3 grid(static_cast<unsigned int>((tokens + 15U) / 16U),
                  layout.num_heads);
  const auto* query = static_cast<const std::uint16_t*>(query_bf16);
  const auto* key = static_cast<const std::uint16_t*>(key_bf16);
  const auto* value_t = static_cast<const std::uint16_t*>(value_t_bf16);
  auto* context = static_cast<std::uint16_t*>(context_bf16);
  const auto token_count = static_cast<std::uint32_t>(tokens);
  const auto value_stride = static_cast<std::uint32_t>(ValueStride(tokens));
  if (layout.head_dim == 256) {
    WmmaAttentionKernel<256><<<grid, 128, 0, stream>>>(
        query, key, value_t, context, sequence_begin, sequence_end, token_count,
        value_stride, layout.num_heads, layout.num_key_value_heads, window);
  } else if (layout.head_dim == 512) {
    WmmaAttentionKernel<512><<<grid, 128, 0, stream>>>(
        query, key, value_t, context, sequence_begin, sequence_end, token_count,
        value_stride, layout.num_heads, layout.num_key_value_heads, window);
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
