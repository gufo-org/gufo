#include <cmath>

#include "vision_ops.hpp"

namespace gufo::models::embedding_gemma2::hip {
namespace {
__device__ float Load(std::uint16_t value) {
  return __uint_as_float(static_cast<std::uint32_t>(value) << 16);
}
__device__ std::uint16_t Round(float value) {
  const auto bits = __float_as_uint(value);
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16) | 0x40U);
  }
  return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >>
                                    16);
}
__device__ float Sum(float value) {
  for (int offset = 16; offset; offset >>= 1) {
    value += __shfl_xor(value, offset, 32);
  }
  return value;
}
__global__ void Pixels(const float* input, std::uint16_t* output,
                       std::size_t count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    output[i] = Round(2.0F * (input[i] - 0.5F));
}
__global__ void Positions(const float* projected, const std::uint16_t* table,
                          float* state, std::size_t patches,
                          std::uint32_t columns) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= patches * 768)
    return;
  const std::size_t patch = i / 768, c = i % 768;
  state[i] = projected[i] + Load(table[(patch % columns) * 768 + c]) +
             Load(table[(10240 + patch / columns) * 768 + c]);
}
__global__ void Qkv(const float* qkv, const std::uint16_t* qnorm,
                    const std::uint16_t* knorm, std::uint16_t* query,
                    std::uint16_t* key, std::uint16_t* value,
                    std::uint32_t columns) {
  const std::size_t patch = blockIdx.x;
  const std::size_t head = blockIdx.y;
  const unsigned lane = threadIdx.x;
  const std::size_t offset = patch * 2304 + head * 64;
  float q[2], k[2], v[2];
  for (unsigned j = 0; j < 2; ++j) {
    q[j] = qkv[offset + lane + j * 32];
    k[j] = qkv[offset + 768 + lane + j * 32];
    v[j] = qkv[offset + 1536 + lane + j * 32];
  }
  const float qs = rsqrtf(Sum(q[0] * q[0] + q[1] * q[1]) / 64.0F + 1e-6F);
  const float ks = rsqrtf(Sum(k[0] * k[0] + k[1] * k[1]) / 64.0F + 1e-6F);
  const float vs = rsqrtf(Sum(v[0] * v[0] + v[1] * v[1]) / 64.0F + 1e-6F);
  for (unsigned j = 0; j < 2; ++j) {
    q[j] *= qs * Load(qnorm[lane + j * 32]);
    k[j] *= ks * Load(knorm[lane + j * 32]);
    const float other_q = __shfl_xor(q[j], 16, 32);
    const float other_k = __shfl_xor(k[j], 16, 32);
    // Each 32-channel axis rotates its own two 16-channel halves.
    const float position =
        static_cast<float>(j == 0 ? patch % columns : patch / columns);
    const float angle =
        position * powf(100.0F, -static_cast<float>(lane % 16) / 16.0F);
    const float co = cosf(angle), si = sinf(angle);
    const float sign = lane < 16 ? -1.0F : 1.0F;
    const std::size_t target = (patch * 12 + head) * 64 + lane + j * 32;
    query[target] = Round(q[j] * co + sign * other_q * si);
    key[target] = Round(k[j] * co + sign * other_k * si);
    value[target] = Round(v[j] * vs);
  }
}
__global__ void Attention(const std::uint16_t* query, const std::uint16_t* key,
                          const std::uint16_t* value, std::uint16_t* output,
                          std::size_t patches) {
  const std::size_t patch = blockIdx.x, head = blockIdx.y;
  const unsigned lane = threadIdx.x;
  const std::size_t target = (patch * 12 + head) * 64 + lane;
  const float q0 = Load(query[target]), q1 = Load(query[target + 32]);
  float a0 = 0.0F, a1 = 0.0F, denominator = 0.0F, maximum = -INFINITY;
  for (std::size_t p = 0; p < patches; ++p) {
    const std::size_t source = (p * 12 + head) * 64 + lane;
    const float score =
        Sum(q0 * Load(key[source]) + q1 * Load(key[source + 32]));
    const float next = fmaxf(maximum, score);
    const float correction = expf(maximum - next);
    const float weight = expf(score - next);
    denominator = denominator * correction + weight;
    a0 = a0 * correction + weight * Load(value[source]);
    a1 = a1 * correction + weight * Load(value[source + 32]);
    maximum = next;
  }
  output[target] = Round(a0 / denominator);
  output[target + 32] = Round(a1 / denominator);
}
__global__ void Pool(const float* state, std::uint16_t* output,
                     std::uint32_t columns) {
  const std::size_t token = blockIdx.x;
  const std::size_t x = token % (columns / 3) * 3;
  const std::size_t y = token / (columns / 3) * 3;
  float values[24];
  float squares = 0.0F;
  for (unsigned i = 0; i < 24; ++i) {
    const unsigned c = threadIdx.x + i * 32;
    float total = 0.0F;
    for (unsigned dy = 0; dy < 3; ++dy) {
      for (unsigned dx = 0; dx < 3; ++dx) {
        total += state[((y + dy) * columns + x + dx) * 768 + c] / 9.0F;
      }
    }
    values[i] = total * sqrtf(768.0F);
    squares += values[i] * values[i];
  }
  const float scale = rsqrtf(Sum(squares) / 768.0F + 1e-6F);
  for (unsigned i = 0; i < 24; ++i) {
    const unsigned c = threadIdx.x + i * 32;
    output[token * 768 + c] = Round(values[i] * scale);
  }
}
__global__ void Insert(const float* features, float* state,
                       std::uint16_t* embeddings, std::size_t count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    embeddings[i] = Round(features[i]);
    state[i] = Load(embeddings[i]);
  }
}
}  // namespace

void LaunchVisionPixels(const float* pixels, void* bf16, std::size_t count,
                        hipStream_t stream) {
  Pixels<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
      pixels, static_cast<std::uint16_t*>(bf16), count);
}
void LaunchVisionPositions(const float* projected, const void* table,
                           float* state, std::size_t patches,
                           std::uint32_t columns, hipStream_t stream) {
  Positions<<<static_cast<unsigned>((patches * 768 + 255) / 256), 256, 0,
              stream>>>(projected, static_cast<const std::uint16_t*>(table),
                        state, patches, columns);
}
void LaunchVisionQkv(const float* qkv, const void* query_norm,
                     const void* key_norm, void* query, void* key, void* value,
                     std::size_t patches, std::uint32_t columns,
                     hipStream_t stream) {
  Qkv<<<dim3(static_cast<unsigned>(patches), 12), 32, 0, stream>>>(
      qkv, static_cast<const std::uint16_t*>(query_norm),
      static_cast<const std::uint16_t*>(key_norm),
      static_cast<std::uint16_t*>(query), static_cast<std::uint16_t*>(key),
      static_cast<std::uint16_t*>(value), columns);
}
void LaunchVisionAttention(const void* query, const void* key,
                           const void* value, void* context,
                           std::size_t patches, hipStream_t stream) {
  Attention<<<dim3(static_cast<unsigned>(patches), 12), 32, 0, stream>>>(
      static_cast<const std::uint16_t*>(query),
      static_cast<const std::uint16_t*>(key),
      static_cast<const std::uint16_t*>(value),
      static_cast<std::uint16_t*>(context), patches);
}
void LaunchVisionPool(const float* state, void* pooled, std::uint32_t rows,
                      std::uint32_t columns, hipStream_t stream) {
  Pool<<<rows * columns / 9, 32, 0, stream>>>(
      state, static_cast<std::uint16_t*>(pooled), columns);
}
void LaunchVisionInsert(const float* features, float* state, void* embeddings,
                        std::size_t tokens, hipStream_t stream) {
  Insert<<<static_cast<unsigned>((tokens * 512 + 255) / 256), 256, 0, stream>>>(
      features, state, static_cast<std::uint16_t*>(embeddings), tokens * 512);
}
}  // namespace gufo::models::embedding_gemma2::hip
