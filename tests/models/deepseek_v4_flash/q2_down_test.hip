#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/models/deepseek_v4_flash/kernels/rocm/detail/grouped_q2_down.hip.hpp"
__device__ static float dev_f16_to_f32(uint16_t v) {
  return __half2float(*reinterpret_cast<const __half*>(&v));
}
__device__ static float warp_sum_f32(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    v += __shfl_down(v, offset, 32);
#else
    v += __shfl_down_sync(FULL_WARP_MASK, v, offset, 32);
#endif
  }
  return v;
}
__device__ __forceinline__ static void q2_K_scale_broadcast_w32(
    const unsigned char* blk, float* d, float* dmin) {
  float vd = 0.0f;
  float vm = 0.0f;
  if ((threadIdx.x & 31u) == 0u) {
    const uint16_t d_bits = (uint16_t)blk[80] | ((uint16_t)blk[81] << 8);
    const uint16_t dmin_bits = (uint16_t)blk[82] | ((uint16_t)blk[83] << 8);
    vd = dev_f16_to_f32(d_bits);
    vm = dev_f16_to_f32(dmin_bits);
  }
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  *d = __shfl(vd, 0, 32);
  *dmin = __shfl(vm, 0, 32);
#else
  *d = __shfl_sync(FULL_WARP_MASK, vd, 0, 32);
  *dmin = __shfl_sync(FULL_WARP_MASK, vm, 0, 32);
#endif
}
__device__ __forceinline__ static float q2_K_dequant_256_scaled_w32(
    const unsigned char* blk, uint32_t lane, uint32_t kk, float d, float dmin) {
  const unsigned char* sc = blk;
  const unsigned char* qs = blk + 16u;
  const uint32_t g = (lane >> 4u) + (kk << 1u);
  const uint32_t within = g & 7u;
  const uint32_t qi = (g >> 3u) * 32u + (within & 1u) * 16u + (lane & 15u);
  const uint32_t shift = (within >> 1u) * 2u;
  const float q = (float)((qs[qi] >> shift) & 3u);
  const float scale = (float)(sc[g] & 0x0fu);
  const float mn = (float)(sc[g] >> 4u);
  return d * scale * q - dmin * mn;
}
template<uint32_t TILES_PER_PASS, uint32_t ROWS_PER_WARP = 1u>
__global__ static void baseline_kernel(
    __half* down_out_h, const char* down_base, const float* mid,
    const uint32_t* expert_offsets, const uint32_t* sorted_tiles,
    const uint32_t* active_count, const uint32_t* active_experts,
    const uint32_t* tile_pair_counts, const uint32_t* tile_pairs,
    uint64_t down_expert_bytes, uint64_t down_row_bytes,
    uint32_t expert_mid_dim, uint32_t out_dim) {
  static_assert(TILES_PER_PASS >= 2u && TILES_PER_PASS <= 4u);
  static_assert(ROWS_PER_WARP == 1u || ROWS_PER_WARP == 2u);
  constexpr uint32_t PAIRS_PER_PASS = TILES_PER_PASS * 4u;
  const uint32_t active = blockIdx.y;
  if (active >= *active_count)
    return;

  const uint32_t expert = active_experts[active];
  const uint32_t first = expert_offsets[expert];
  const uint32_t count = expert_offsets[expert + 1u] - first;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 31u;
  const uint32_t rows_per_block = blockDim.x >> 5u;
  const uint32_t warp = tid >> 5u;
  const uint32_t row0 = blockIdx.x * rows_per_block * ROWS_PER_WARP + warp;
  uint32_t rows[ROWS_PER_WARP] = {};
  const unsigned char* down_rows[ROWS_PER_WARP] = {};
#pragma unroll
  for (uint32_t row_slot = 0u; row_slot < ROWS_PER_WARP; ++row_slot) {
    rows[row_slot] = row0 + row_slot * rows_per_block;
    down_rows[row_slot] =
        (const unsigned char*)down_base + (uint64_t)expert * down_expert_bytes +
        (uint64_t)(rows[row_slot] < out_dim ? rows[row_slot] : 0u) *
            down_row_bytes;
  }
  const uint32_t n_blocks = expert_mid_dim >> 8u;
  extern __shared__ float shmid[];

  for (uint32_t tile_index = 0u; tile_index < count;
       tile_index += TILES_PER_PASS) {
    uint32_t pair[PAIRS_PER_PASS] = {};
    uint32_t np = 0u;
#pragma unroll
    for (uint32_t tile_slot = 0u; tile_slot < TILES_PER_PASS; ++tile_slot) {
      if (tile_index + tile_slot >= count)
        break;
      const uint32_t tile = sorted_tiles[first + tile_index + tile_slot];
      const uint32_t tile_np = tile_pair_counts[tile];
#pragma unroll
      for (uint32_t p = 0u; p < 4u; ++p) {
        if (p < tile_np) {
          pair[np++] = tile_pairs[(uint64_t)tile * 4u + p];
        }
      }
    }

    float acc[ROWS_PER_WARP][PAIRS_PER_PASS] = {};
    for (uint32_t b = 0u; b < n_blocks; ++b) {
      const uint64_t mid_base = (uint64_t)b * 256u;
      for (uint32_t j = tid; j < np * 256u; j += blockDim.x) {
        const uint32_t p = j >> 8u;
        const uint32_t k = j & 255u;
        shmid[j] = mid[(uint64_t)pair[p] * expert_mid_dim + mid_base + k];
      }
      __syncthreads();
#pragma unroll
      for (uint32_t row_slot = 0u; row_slot < ROWS_PER_WARP; ++row_slot) {
        if (rows[row_slot] >= out_dim)
          continue;
        const unsigned char* weight_block =
            down_rows[row_slot] + (uint64_t)b * 84u;
        float d = 0.0f;
        float dmin = 0.0f;
        q2_K_scale_broadcast_w32(weight_block, &d, &dmin);
#pragma unroll
        for (uint32_t k = 0u; k < 8u; ++k) {
          const uint32_t i = lane + (k << 5u);
          const float weight =
              q2_K_dequant_256_scaled_w32(weight_block, lane, k, d, dmin);
#pragma unroll
          for (uint32_t p = 0u; p < PAIRS_PER_PASS; ++p) {
            if (p < np) {
              acc[row_slot][p] += weight * shmid[(p << 8u) + i];
            }
          }
        }
      }
      __syncthreads();
    }
#pragma unroll
    for (uint32_t row_slot = 0u; row_slot < ROWS_PER_WARP; ++row_slot) {
#pragma unroll
      for (uint32_t p = 0u; p < PAIRS_PER_PASS; ++p) {
        if (p < np) {
          acc[row_slot][p] = warp_sum_f32(acc[row_slot][p]);
          if (lane == 0u && rows[row_slot] < out_dim) {
            down_out_h[(uint64_t)pair[p] * out_dim + rows[row_slot]] =
                __float2half(acc[row_slot][p]);
          }
        }
      }
    }
  }
}
#define HIP_CHECK(expr)                                              \
  do {                                                               \
    const auto e = (expr);                                           \
    if (e != hipSuccess) {                                           \
      std::fprintf(stderr, "%s: %s\n", #expr, hipGetErrorString(e)); \
      std::exit(1);                                                  \
    }                                                                \
  } while (0)
template<class T>
struct Buffer {
  T* data;
  size_t count;
  explicit Buffer(size_t n) : count(n) {
    HIP_CHECK(hipMalloc(&data, n * sizeof(T)));
  }
  explicit Buffer(const std::vector<T>& v) : Buffer(v.size()) {
    HIP_CHECK(
        hipMemcpy(data, v.data(), count * sizeof(T), hipMemcpyHostToDevice));
  }
  ~Buffer() { HIP_CHECK(hipFree(data)); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  std::vector<T> read() const {
    std::vector<T> v(count);
    HIP_CHECK(
        hipMemcpy(v.data(), data, count * sizeof(T), hipMemcpyDeviceToHost));
    return v;
  }
};
__global__ void init_weights(char* w, size_t n) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  uint32_t x = uint32_t(i) * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  x = (x >> 22u) ^ x;
  const uint32_t at = i % 84;
  const uint32_t seed = uint32_t(i / 84) * 747796405u + 2891336453u;
  const uint16_t d = 0x2000u + (seed & 0x0fffu);
  const uint16_t dm = 0x1c00u + ((seed >> 12u) & 0x0fffu);
  w[i] = at == 80   ? d & 255
         : at == 81 ? d >> 8
         : at == 82 ? dm & 255
         : at == 83 ? dm >> 8
                    : x;
}
__global__ void init_mid(float* mid, size_t n) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    mid[i] = float(int((i * 73 + 19) % 2001) - 1000) * 0.0001f;
}
template<uint32_t TILES>
void run(uint32_t concurrency, bool ragged, bool benchmark,
         uint32_t warps = 32) {
  constexpr uint32_t in = 2048, out = 4096, experts = 96;
  std::vector<uint32_t> tile_experts, tile_counts, tile_pairs;
  uint32_t pair_count = 0;
  for (uint32_t c = 0; c < concurrency; ++c) {
    std::array<std::vector<uint32_t>, experts> pairs;
    const uint32_t width = ragged ? 1 + c % 4 : 4;
    for (uint32_t row = 0; row < width; ++row) {
      for (uint32_t slot = 0; slot < 6; ++slot) {
        const auto e = (slot * 13 + row / 2 + (c % 3) * 5) % experts;
        pairs[e].push_back(pair_count++);
      }
    }
    for (uint32_t e = 0; e < experts; ++e) {
      if (pairs[e].empty())
        continue;
      tile_experts.push_back(e);
      tile_counts.push_back(pairs[e].size());
      for (uint32_t p = 0; p < 4; ++p)
        tile_pairs.push_back(p < pairs[e].size() ? pairs[e][p] : 0);
    }
  }
  std::vector<uint32_t> offsets(257), sorted, active;
  for (uint32_t e = 0; e < 256; ++e) {
    offsets[e] = sorted.size();
    for (uint32_t t = 0; t < tile_experts.size(); ++t)
      if (tile_experts[t] == e)
        sorted.push_back(t);
    if (sorted.size() != offsets[e])
      active.push_back(e);
  }
  offsets[256] = sorted.size();
  Buffer<uint32_t> d_offsets(offsets), d_sorted(sorted), d_active(active),
      d_counts(tile_counts), d_pairs(tile_pairs),
      d_nactive(std::vector<uint32_t>{uint32_t(active.size())});
  const uint64_t row_bytes = in / 256 * 84;
  const uint64_t expert_bytes = out * row_bytes;
  Buffer<char> weights(experts * expert_bytes);
  Buffer<float> mid(size_t(pair_count) * in);
  Buffer<__half> base(size_t(pair_count) * out), cand(size_t(pair_count) * out);
  init_weights<<<(weights.count + 255) / 256, 256>>>(weights.data,
                                                     weights.count);
  init_mid<<<(mid.count + 255) / 256, 256>>>(mid.data, mid.count);
  HIP_CHECK(hipDeviceSynchronize());
  const dim3 grid((out + warps * 2 - 1) / (warps * 2), active.size());
  const uint32_t lds = TILES * 4 * 256 * sizeof(float);
  auto launch = [&](bool candidate) {
    if (candidate)
      gufo::models::deepseek_v4_flash::kernels::GroupedQ2Down<TILES>
          <<<grid, warps * 32, lds>>>(
              cand.data, weights.data, mid.data, d_offsets.data, d_sorted.data,
              d_nactive.data, d_active.data, d_counts.data, d_pairs.data,
              expert_bytes, row_bytes, in, out);
    else
      baseline_kernel<TILES, 2><<<grid, warps * 32, lds>>>(
          base.data, weights.data, mid.data, d_offsets.data, d_sorted.data,
          d_nactive.data, d_active.data, d_counts.data, d_pairs.data,
          expert_bytes, row_bytes, in, out);
    HIP_CHECK(hipGetLastError());
  };
  launch(false);
  launch(true);
  HIP_CHECK(hipDeviceSynchronize());
  const auto a = base.read(), b = cand.read();
  size_t mismatches = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(__half2float(a[i])) ||
        !std::isfinite(__half2float(b[i])) ||
        std::memcmp(&a[i], &b[i], sizeof(__half)))
      ++mismatches;
  }
  if (mismatches)
    std::exit(2);
  if (!benchmark) {
    std::printf("Q2 C%u ragged=%d T%u: %zu exact finite values\n", concurrency,
                ragged, TILES, a.size());
    return;
  }
  hipEvent_t begin, end;
  HIP_CHECK(hipEventCreate(&begin));
  HIP_CHECK(hipEventCreate(&end));
  std::vector<float> samples[2];
  for (uint32_t r = 0; r < 14; ++r) {
    for (uint32_t arm = 0; arm < 2; ++arm) {
      const bool candidate = (r + arm) % 2;
      HIP_CHECK(hipEventRecord(begin));
      for (uint32_t rep = 0; rep < 5; ++rep)
        launch(candidate);
      HIP_CHECK(hipEventRecord(end));
      HIP_CHECK(hipEventSynchronize(end));
      float ms = 0;
      HIP_CHECK(hipEventElapsedTime(&ms, begin, end));
      if (r > 1)
        samples[candidate].push_back(ms / 5);
    }
  }
  for (auto& s : samples)
    std::sort(s.begin(), s.end());
  const float ma = samples[0][samples[0].size() / 2],
              mb = samples[1][samples[1].size() / 2];
  std::printf(
      "C%u ragged=%d T%u warps=%u pairs=%u experts=%zu baseline=%.4f ms "
      "candidate=%.4f ms speedup=%.4f mismatch=%zu/%zu\n",
      concurrency, ragged, TILES, warps, pair_count, active.size(), ma, mb,
      ma / mb, mismatches, a.size());
  HIP_CHECK(hipEventDestroy(begin));
  HIP_CHECK(hipEventDestroy(end));
  if (mismatches)
    std::exit(2);
}
int main(int argc, char** argv) {
  const bool benchmark = argc == 2 && std::strcmp(argv[1], "--benchmark") == 0;
  if (argc > 1 && !benchmark)
    return 2;
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0)
    return 77;
  for (const bool ragged : {false, true}) {
    run<2>(2, ragged, benchmark);
    run<4>(4, ragged, benchmark);
    run<3>(6, ragged, benchmark);
    run<4>(8, ragged, benchmark);
  }
}
