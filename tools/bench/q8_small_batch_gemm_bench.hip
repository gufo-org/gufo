// Standalone microbenchmark + exactness harness for the Qwen3.8-27B decode
// equivalent Q8_0 small-batch GEMM used by speculative verification.
//
// Speculative verification must produce the same logits as sequential decode,
// so every candidate here is required to be *bit-exact* against the batch-1
// decode GEMV (Q8KBlockGEMVKernel_2Rows) row by row. A variant that is faster
// but not bit-exact is a failure, not a tradeoff.
//
// Self-contained (no repository headers) so it compiles directly with hipcc:
//
//   nix develop -c tools/bench/build.sh q8_small_batch_gemm_bench
//   /tmp/q8_small_batch_gemm_bench
//
// Roofline constants from tools/bench/gfx1151_peak.hip: DRAM read 241 GB/s.
//
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#define HIP_OK(expr)                                                    \
  do {                                                                  \
    hipError_t err_ = (expr);                                           \
    if (err_ != hipSuccess) {                                           \
      std::fprintf(stderr, "%s:%d hip error: %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(err_));                            \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

constexpr int kQ8_0BlockSize = 32;
constexpr int kMaxBatch = 8;

struct Q8_0Block {
  __half d;
  std::int8_t qs[kQ8_0BlockSize];
};
static_assert(sizeof(Q8_0Block) == 34);

// ---------------------------------------------------------------------------
// Reference: the batch-1 decode GEMV, two output rows per warp. This defines
// both the numeric contract and the bandwidth target.
// ---------------------------------------------------------------------------
template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void RefGemv2Rows(const void* __restrict__ A, const float* __restrict__ x,
                      float* __restrict__ y, std::size_t M, std::size_t K) {
  const std::size_t lane_id = threadIdx.x & 31u;
  const std::size_t warp_id = threadIdx.x >> 5u;
  const std::size_t pair = (blockIdx.x * WavesPerBlock) + warp_id;
  const std::size_t m0 = pair * 2;
  const std::size_t m1 = m0 + 1;
  if (m0 >= M) {
    return;
  }
  const std::size_t num_blocks = K / kQ8_0BlockSize;
  const auto* row0 = static_cast<const Q8_0Block*>(A) + (m0 * num_blocks);
  const auto* row1 =
      (m1 < M) ? static_cast<const Q8_0Block*>(A) + (m1 * num_blocks) : nullptr;
  float sum0 = 0.0F;
  float sum1 = 0.0F;
  for (std::size_t b = lane_id; b < num_blocks; b += 32) {
    const float* xb = x + (b * kQ8_0BlockSize);
    const Q8_0Block& w0 = row0[b];
    const float d0 = __half2float(w0.d);
    float dot0 = 0.0F;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      dot0 += static_cast<float>(w0.qs[i]) * xb[i];
    }
    sum0 += d0 * dot0;
    if (row1 != nullptr) {
      const Q8_0Block& w1 = row1[b];
      const float d1 = __half2float(w1.d);
      float dot1 = 0.0F;
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        dot1 += static_cast<float>(w1.qs[i]) * xb[i];
      }
      sum1 += d1 * dot1;
    }
  }
  for (int off = 16; off > 0; off >>= 1) {
    sum0 += __shfl_xor(sum0, off);
    sum1 += __shfl_xor(sum1, off);
  }
  if (lane_id == 0) {
    y[m0] = sum0;
    if (m1 < M) {
      y[m1] = sum1;
    }
  }
}

// ---------------------------------------------------------------------------
// Variant A: the shipped LDS-staged kernel (32 waves per block, activation
// tiles staged through shared memory, two barriers per staged tile).
// ---------------------------------------------------------------------------
template<std::uint32_t WavesPerBlock, std::size_t Batch>
__launch_bounds__(WavesPerBlock * 32, WavesPerBlock >= 32 ? 1 : 2) __global__
    void Lds8Gemm(const void* __restrict__ w, const float* __restrict__ x,
                  float* __restrict__ y, std::size_t m, std::size_t k) {
  constexpr std::size_t kStagedTokens = Batch < 4 ? Batch : 4;
  constexpr std::size_t kBlocksPerTile = 32;
  constexpr std::size_t kPaddedBlockSize = kQ8_0BlockSize + 1;
  constexpr std::size_t kTileStride = kBlocksPerTile * kPaddedBlockSize;
  __shared__ float staged_x[kStagedTokens * kTileStride];

  const std::size_t lane = threadIdx.x & 31u;
  const std::size_t warp_id = threadIdx.x >> 5u;
  const std::size_t row = (blockIdx.x * WavesPerBlock) + warp_id;
  const std::size_t num_blocks = k / kQ8_0BlockSize;
  const std::size_t safe_row = row < m ? row : m - 1;
  const auto* row_blocks =
      static_cast<const Q8_0Block*>(w) + (safe_row * num_blocks);
  float sums[Batch] = {};

  for (std::size_t tile_base = 0; tile_base < num_blocks;
       tile_base += kBlocksPerTile) {
    for (std::size_t token_base = 0; token_base < Batch;
         token_base += kStagedTokens) {
      const std::size_t staged = Batch - token_base < kStagedTokens
                                     ? Batch - token_base
                                     : kStagedTokens;
      const std::size_t tile_elements =
          staged * kBlocksPerTile * kQ8_0BlockSize;
      for (std::size_t flat = threadIdx.x; flat < tile_elements;
           flat += blockDim.x) {
        const std::size_t token = flat / (kBlocksPerTile * kQ8_0BlockSize);
        const std::size_t within = flat % (kBlocksPerTile * kQ8_0BlockSize);
        const std::size_t block = within / kQ8_0BlockSize;
        const std::size_t element = within % kQ8_0BlockSize;
        const std::size_t source_block = tile_base + block;
        staged_x[(token * kTileStride) + (block * kPaddedBlockSize) + element] =
            source_block < num_blocks
                ? x[((token_base + token) * k) +
                    (source_block * kQ8_0BlockSize) + element]
                : 0.0F;
      }
      __syncthreads();
      const std::size_t block = tile_base + lane;
      if (row < m && block < num_blocks) {
        const Q8_0Block& weights = row_blocks[block];
        const float scale = __half2float(weights.d);
        float block_dots[kStagedTokens] = {};
#pragma unroll
        for (std::size_t group = 0; group < kQ8_0BlockSize; group += 4) {
          std::uint32_t packed = 0;
          __builtin_memcpy(&packed, weights.qs + group, sizeof(packed));
          const float q0 =
              static_cast<float>(static_cast<std::int8_t>(packed & 0xFFU));
          const float q1 = static_cast<float>(
              static_cast<std::int8_t>((packed >> 8U) & 0xFFU));
          const float q2 = static_cast<float>(
              static_cast<std::int8_t>((packed >> 16U) & 0xFFU));
          const float q3 = static_cast<float>(
              static_cast<std::int8_t>((packed >> 24U) & 0xFFU));
#pragma unroll
          for (std::size_t token = 0; token < kStagedTokens; ++token) {
            if (token >= staged) {
              continue;
            }
            const float* input = staged_x + (token * kTileStride) +
                                 (lane * kPaddedBlockSize) + group;
            block_dots[token] += q0 * input[0];
            block_dots[token] += q1 * input[1];
            block_dots[token] += q2 * input[2];
            block_dots[token] += q3 * input[3];
          }
        }
#pragma unroll
        for (std::size_t token = 0; token < kStagedTokens; ++token) {
          if (token >= staged) {
            continue;
          }
          sums[token_base + token] += scale * block_dots[token];
        }
      }
      __syncthreads();
    }
  }
#pragma unroll
  for (std::size_t token = 0; token < Batch; ++token) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      sums[token] += __shfl_xor(sums[token], offset);
    }
  }
  if (lane == 0 && row < m) {
#pragma unroll
    for (std::size_t token = 0; token < Batch; ++token) {
      y[(token * m) + row] = sums[token];
    }
  }
}

// ---------------------------------------------------------------------------
// Variant B: keep the decode GEMV shape -- no shared memory, no barriers, high
// occupancy, RowsPerWave weight streams in flight -- and fold the batch into
// per-token accumulators. Activations stay in global memory; they are a few
// hundred KB and stay resident in cache across the whole grid.
// ---------------------------------------------------------------------------
template<std::uint32_t WavesPerBlock, std::size_t Batch,
         std::size_t RowsPerWave, std::uint32_t MinBlocksPerCu>
__launch_bounds__(WavesPerBlock * 32, MinBlocksPerCu) __global__
    void GemvShapedGemm(const void* __restrict__ w, const float* __restrict__ x,
                        float* __restrict__ y, std::size_t m, std::size_t k) {
  const std::size_t lane = threadIdx.x & 31u;
  const std::size_t warp_id = threadIdx.x >> 5u;
  const std::size_t group = (blockIdx.x * WavesPerBlock) + warp_id;
  const std::size_t row_base = group * RowsPerWave;
  if (row_base >= m) {
    return;
  }
  const std::size_t num_blocks = k / kQ8_0BlockSize;
  const auto* base = static_cast<const Q8_0Block*>(w);
  float sums[RowsPerWave][Batch] = {};

  for (std::size_t b = lane; b < num_blocks; b += 32) {
#pragma unroll
    for (std::size_t r = 0; r < RowsPerWave; ++r) {
      const std::size_t row = row_base + r;
      if (row >= m) {
        continue;
      }
      const Q8_0Block& weights = base[(row * num_blocks) + b];
      const float scale = __half2float(weights.d);
      // Eight packed dwords hold the whole 32-element block, so the weight
      // stream is read once per row regardless of the batch width.
      std::uint32_t packed[8];
#pragma unroll
      for (int g = 0; g < 8; ++g) {
        __builtin_memcpy(&packed[g], weights.qs + (g * 4),
                         sizeof(std::uint32_t));
      }
#pragma unroll
      for (std::size_t token = 0; token < Batch; ++token) {
        const float* xb = x + (token * k) + (b * kQ8_0BlockSize);
        float dot = 0.0F;
#pragma unroll
        for (int g = 0; g < 8; ++g) {
          const std::uint32_t p = packed[g];
          const float4 xv = *reinterpret_cast<const float4*>(xb + (g * 4));
          dot += static_cast<float>(static_cast<std::int8_t>(p & 0xFFU)) * xv.x;
          dot +=
              static_cast<float>(static_cast<std::int8_t>((p >> 8U) & 0xFFU)) *
              xv.y;
          dot +=
              static_cast<float>(static_cast<std::int8_t>((p >> 16U) & 0xFFU)) *
              xv.z;
          dot +=
              static_cast<float>(static_cast<std::int8_t>((p >> 24U) & 0xFFU)) *
              xv.w;
        }
        sums[r][token] += scale * dot;
      }
    }
  }

#pragma unroll
  for (std::size_t r = 0; r < RowsPerWave; ++r) {
#pragma unroll
    for (std::size_t token = 0; token < Batch; ++token) {
      for (int off = 16; off > 0; off >>= 1) {
        sums[r][token] += __shfl_xor(sums[r][token], off);
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (std::size_t r = 0; r < RowsPerWave; ++r) {
      const std::size_t row = row_base + r;
      if (row >= m) {
        continue;
      }
#pragma unroll
      for (std::size_t token = 0; token < Batch; ++token) {
        y[(token * m) + row] = sums[r][token];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Variant C: LDS-staged like variant A, but each warp owns RowsPerWave output
// rows. All those rows consume the same staged activation window, so the LDS
// read traffic per weight byte drops by RowsPerWave. That is what actually
// limits variant A at width 8: 34 bytes of weight per 8x32 activation reads.
// ---------------------------------------------------------------------------
template<std::uint32_t WavesPerBlock, std::size_t Batch,
         std::size_t RowsPerWave, std::size_t StagedTokens,
         std::uint32_t MinBlocksPerCu>
__launch_bounds__(WavesPerBlock * 32, MinBlocksPerCu) __global__
    void LdsMultiRowGemm(const void* __restrict__ w,
                         const float* __restrict__ x, float* __restrict__ y,
                         std::size_t m, std::size_t k) {
  constexpr std::size_t kBlocksPerTile = 32;
  constexpr std::size_t kPaddedBlockSize = kQ8_0BlockSize + 1;
  constexpr std::size_t kTileStride = kBlocksPerTile * kPaddedBlockSize;
  __shared__ float staged_x[StagedTokens * kTileStride];

  const std::size_t lane = threadIdx.x & 31u;
  const std::size_t warp_id = threadIdx.x >> 5u;
  const std::size_t row_base =
      ((blockIdx.x * WavesPerBlock) + warp_id) * RowsPerWave;
  const std::size_t num_blocks = k / kQ8_0BlockSize;
  const auto* base = static_cast<const Q8_0Block*>(w);
  float sums[RowsPerWave][Batch] = {};

  for (std::size_t tile_base = 0; tile_base < num_blocks;
       tile_base += kBlocksPerTile) {
    for (std::size_t token_base = 0; token_base < Batch;
         token_base += StagedTokens) {
      const std::size_t staged =
          Batch - token_base < StagedTokens ? Batch - token_base : StagedTokens;
      const std::size_t tile_elements =
          staged * kBlocksPerTile * kQ8_0BlockSize;
      for (std::size_t flat = threadIdx.x; flat < tile_elements;
           flat += blockDim.x) {
        const std::size_t token = flat / (kBlocksPerTile * kQ8_0BlockSize);
        const std::size_t within = flat % (kBlocksPerTile * kQ8_0BlockSize);
        const std::size_t block = within / kQ8_0BlockSize;
        const std::size_t element = within % kQ8_0BlockSize;
        const std::size_t source_block = tile_base + block;
        staged_x[(token * kTileStride) + (block * kPaddedBlockSize) + element] =
            source_block < num_blocks
                ? x[((token_base + token) * k) +
                    (source_block * kQ8_0BlockSize) + element]
                : 0.0F;
      }
      __syncthreads();

      const std::size_t block = tile_base + lane;
      if (block < num_blocks) {
        const Q8_0Block* rows[RowsPerWave];
        float scale[RowsPerWave];
#pragma unroll
        for (std::size_t r = 0; r < RowsPerWave; ++r) {
          const std::size_t row = row_base + r;
          const std::size_t safe_row = row < m ? row : m - 1;
          rows[r] = &base[(safe_row * num_blocks) + block];
          scale[r] = __half2float(rows[r]->d);
        }

        float block_dots[RowsPerWave][StagedTokens] = {};
#pragma unroll
        for (int g = 0; g < 8; ++g) {
          // Loading the weight dwords inside the group loop keeps only
          // RowsPerWave of them live, which is what holds occupancy up; the
          // block is already in cache after the scale read.
          std::uint32_t packed[RowsPerWave];
#pragma unroll
          for (std::size_t r = 0; r < RowsPerWave; ++r) {
            __builtin_memcpy(&packed[r], rows[r]->qs + (g * 4),
                             sizeof(std::uint32_t));
          }
#pragma unroll
          for (std::size_t token = 0; token < StagedTokens; ++token) {
            if (token >= staged) {
              continue;
            }
            const float* input = staged_x + (token * kTileStride) +
                                 (lane * kPaddedBlockSize) + (g * 4);
            const float x0 = input[0];
            const float x1 = input[1];
            const float x2 = input[2];
            const float x3 = input[3];
#pragma unroll
            for (std::size_t r = 0; r < RowsPerWave; ++r) {
              const std::uint32_t p = packed[r];
              block_dots[r][token] +=
                  static_cast<float>(static_cast<std::int8_t>(p & 0xFFU)) * x0;
              block_dots[r][token] +=
                  static_cast<float>(
                      static_cast<std::int8_t>((p >> 8U) & 0xFFU)) *
                  x1;
              block_dots[r][token] +=
                  static_cast<float>(
                      static_cast<std::int8_t>((p >> 16U) & 0xFFU)) *
                  x2;
              block_dots[r][token] +=
                  static_cast<float>(
                      static_cast<std::int8_t>((p >> 24U) & 0xFFU)) *
                  x3;
            }
          }
        }
#pragma unroll
        for (std::size_t r = 0; r < RowsPerWave; ++r) {
#pragma unroll
          for (std::size_t token = 0; token < StagedTokens; ++token) {
            if (token >= staged) {
              continue;
            }
            sums[r][token_base + token] += scale[r] * block_dots[r][token];
          }
        }
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (std::size_t r = 0; r < RowsPerWave; ++r) {
#pragma unroll
    for (std::size_t token = 0; token < Batch; ++token) {
      for (int off = 16; off > 0; off >>= 1) {
        sums[r][token] += __shfl_xor(sums[r][token], off);
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (std::size_t r = 0; r < RowsPerWave; ++r) {
      const std::size_t row = row_base + r;
      if (row >= m) {
        continue;
      }
#pragma unroll
      for (std::size_t token = 0; token < Batch; ++token) {
        y[(token * m) + row] = sums[r][token];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Variant D: variant C with a float4-aligned activation stride so each group
// read is one ds_read_b128 instead of four ds_read_b32. Pad is a template
// parameter because the stride controls both vectorization and bank conflicts.
// ---------------------------------------------------------------------------
template<std::uint32_t WavesPerBlock, std::size_t Batch,
         std::size_t RowsPerWave, std::size_t StagedTokens, std::size_t Pad,
         std::uint32_t MinBlocksPerCu>
__launch_bounds__(WavesPerBlock * 32, MinBlocksPerCu) __global__
    void LdsVecGemm(const void* __restrict__ w, const float* __restrict__ x,
                    float* __restrict__ y, std::size_t m, std::size_t k) {
  constexpr std::size_t kBlocksPerTile = 32;
  constexpr std::size_t kStride = kQ8_0BlockSize + Pad;
  constexpr std::size_t kTileStride = kBlocksPerTile * kStride;
  static_assert(kStride % 4 == 0, "activation stride must allow float4 reads");
  __shared__ float staged_x[StagedTokens * kTileStride];

  const std::size_t lane = threadIdx.x & 31u;
  const std::size_t warp_id = threadIdx.x >> 5u;
  const std::size_t row_base =
      ((blockIdx.x * WavesPerBlock) + warp_id) * RowsPerWave;
  const std::size_t num_blocks = k / kQ8_0BlockSize;
  const auto* base = static_cast<const Q8_0Block*>(w);
  float sums[RowsPerWave][Batch] = {};

  for (std::size_t tile_base = 0; tile_base < num_blocks;
       tile_base += kBlocksPerTile) {
    for (std::size_t token_base = 0; token_base < Batch;
         token_base += StagedTokens) {
      const std::size_t staged =
          Batch - token_base < StagedTokens ? Batch - token_base : StagedTokens;
      const std::size_t tile_vectors =
          staged * kBlocksPerTile * (kQ8_0BlockSize / 4);
      for (std::size_t flat = threadIdx.x; flat < tile_vectors;
           flat += blockDim.x) {
        constexpr std::size_t kVectorsPerBlock = kQ8_0BlockSize / 4;
        const std::size_t token = flat / (kBlocksPerTile * kVectorsPerBlock);
        const std::size_t within = flat % (kBlocksPerTile * kVectorsPerBlock);
        const std::size_t block = within / kVectorsPerBlock;
        const std::size_t vector = within % kVectorsPerBlock;
        const std::size_t source_block = tile_base + block;
        float4 value = {0.0F, 0.0F, 0.0F, 0.0F};
        if (source_block < num_blocks) {
          value = *reinterpret_cast<const float4*>(
              x + ((token_base + token) * k) + (source_block * kQ8_0BlockSize) +
              (vector * 4));
        }
        *reinterpret_cast<float4*>(staged_x + (token * kTileStride) +
                                   (block * kStride) + (vector * 4)) = value;
      }
      __syncthreads();

      const std::size_t block = tile_base + lane;
      if (block < num_blocks) {
        const Q8_0Block* rows[RowsPerWave];
        float scale[RowsPerWave];
#pragma unroll
        for (std::size_t r = 0; r < RowsPerWave; ++r) {
          const std::size_t row = row_base + r;
          const std::size_t safe_row = row < m ? row : m - 1;
          rows[r] = &base[(safe_row * num_blocks) + block];
          scale[r] = __half2float(rows[r]->d);
        }
        float block_dots[RowsPerWave][StagedTokens] = {};
#pragma unroll
        for (int g = 0; g < 8; ++g) {
          std::uint32_t packed[RowsPerWave];
#pragma unroll
          for (std::size_t r = 0; r < RowsPerWave; ++r) {
            __builtin_memcpy(&packed[r], rows[r]->qs + (g * 4),
                             sizeof(std::uint32_t));
          }
#pragma unroll
          for (std::size_t token = 0; token < StagedTokens; ++token) {
            if (token >= staged) {
              continue;
            }
            const float4 xv = *reinterpret_cast<const float4*>(
                staged_x + (token * kTileStride) + (lane * kStride) + (g * 4));
#pragma unroll
            for (std::size_t r = 0; r < RowsPerWave; ++r) {
              const std::uint32_t p = packed[r];
              block_dots[r][token] +=
                  static_cast<float>(static_cast<std::int8_t>(p & 0xFFU)) *
                  xv.x;
              block_dots[r][token] +=
                  static_cast<float>(
                      static_cast<std::int8_t>((p >> 8U) & 0xFFU)) *
                  xv.y;
              block_dots[r][token] +=
                  static_cast<float>(
                      static_cast<std::int8_t>((p >> 16U) & 0xFFU)) *
                  xv.z;
              block_dots[r][token] +=
                  static_cast<float>(
                      static_cast<std::int8_t>((p >> 24U) & 0xFFU)) *
                  xv.w;
            }
          }
        }
#pragma unroll
        for (std::size_t r = 0; r < RowsPerWave; ++r) {
#pragma unroll
          for (std::size_t token = 0; token < StagedTokens; ++token) {
            if (token >= staged) {
              continue;
            }
            sums[r][token_base + token] += scale[r] * block_dots[r][token];
          }
        }
      }
      __syncthreads();
    }
  }
#pragma unroll
  for (std::size_t r = 0; r < RowsPerWave; ++r) {
#pragma unroll
    for (std::size_t token = 0; token < Batch; ++token) {
      for (int off = 16; off > 0; off >>= 1) {
        sums[r][token] += __shfl_xor(sums[r][token], off);
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (std::size_t r = 0; r < RowsPerWave; ++r) {
      const std::size_t row = row_base + r;
      if (row >= m) {
        continue;
      }
#pragma unroll
      for (std::size_t token = 0; token < Batch; ++token) {
        y[(token * m) + row] = sums[r][token];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------
struct Shape {
  const char* name;
  std::size_t m;
  std::size_t k;
};

struct Result {
  std::string name;
  double ms;
  double gbps;
  bool exact;
  double max_abs;
};

double TimeKernel(const std::function<void()>& launch, int iters) {
  launch();
  HIP_OK(hipDeviceSynchronize());
  hipEvent_t start;
  hipEvent_t stop;
  HIP_OK(hipEventCreate(&start));
  HIP_OK(hipEventCreate(&stop));
  HIP_OK(hipEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    launch();
  }
  HIP_OK(hipEventRecord(stop));
  HIP_OK(hipEventSynchronize(stop));
  float ms = 0.0F;
  HIP_OK(hipEventElapsedTime(&ms, start, stop));
  HIP_OK(hipEventDestroy(start));
  HIP_OK(hipEventDestroy(stop));
  return static_cast<double>(ms) / iters;
}

int main(int argc, char** argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 20;
  const std::vector<Shape> shapes = {
      {"attn_qkv   m=10240 k=5120", 10240, 5120},
      {"attn_gate  m= 6144 k=5120", 6144, 5120},
      {"ffn_gate   m=17408 k=5120", 17408, 5120},
      {"ffn_down   m= 5120 k=17408", 5120, 17408},
  };

  std::printf("iters=%d  DRAM read roofline 241 GB/s\n", iters);
  for (const auto& shape : shapes) {
    const std::size_t num_blocks = shape.k / kQ8_0BlockSize;
    const std::size_t weight_blocks = shape.m * num_blocks;
    const std::size_t weight_bytes = weight_blocks * sizeof(Q8_0Block);

    std::vector<Q8_0Block> host_w(weight_blocks);
    for (std::size_t i = 0; i < weight_blocks; ++i) {
      host_w[i].d = __float2half(0.01F + 0.0001F * static_cast<float>(i % 97));
      for (int j = 0; j < kQ8_0BlockSize; ++j) {
        host_w[i].qs[j] =
            static_cast<std::int8_t>(((i * 31 + j * 7) % 255) - 127);
      }
    }
    std::vector<float> host_x(kMaxBatch * shape.k);
    for (std::size_t i = 0; i < host_x.size(); ++i) {
      host_x[i] = std::sin(0.001F * static_cast<float>(i)) * 1.5F;
    }

    void* d_w = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_ref = nullptr;
    HIP_OK(hipMalloc(&d_w, weight_bytes));
    HIP_OK(hipMalloc(&d_x, host_x.size() * sizeof(float)));
    HIP_OK(hipMalloc(&d_y, kMaxBatch * shape.m * sizeof(float)));
    HIP_OK(hipMalloc(&d_ref, kMaxBatch * shape.m * sizeof(float)));
    HIP_OK(hipMemcpy(d_w, host_w.data(), weight_bytes, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_x, host_x.data(), host_x.size() * sizeof(float),
                     hipMemcpyHostToDevice));

    // Reference rows: one batch-1 decode GEMV per token.
    {
      constexpr std::uint32_t kWaves = 4;
      const std::size_t pairs = shape.m / 2;
      const std::size_t grid = (pairs + kWaves - 1) / kWaves;
      for (std::size_t token = 0; token < kMaxBatch; ++token) {
        RefGemv2Rows<kWaves><<<grid, kWaves * 32>>>(
            d_w, d_x + (token * shape.k), d_ref + (token * shape.m), shape.m,
            shape.k);
      }
      HIP_OK(hipDeviceSynchronize());
    }
    std::vector<float> ref(kMaxBatch * shape.m);
    HIP_OK(hipMemcpy(ref.data(), d_ref, ref.size() * sizeof(float),
                     hipMemcpyDeviceToHost));

    std::printf("\n== %s  weights %.1f MB ==\n", shape.name,
                weight_bytes / 1e6);

    std::vector<Result> results;
    const auto check = [&](const std::string& name, double ms,
                           std::size_t batch) {
      std::vector<float> got(batch * shape.m);
      HIP_OK(hipMemcpy(got.data(), d_y, got.size() * sizeof(float),
                       hipMemcpyDeviceToHost));
      bool exact = true;
      double max_abs = 0.0;
      for (std::size_t token = 0; token < batch; ++token) {
        for (std::size_t row = 0; row < shape.m; ++row) {
          const float a = got[(token * shape.m) + row];
          const float b = ref[(token * shape.m) + row];
          if (a != b) {
            exact = false;
            max_abs = std::max(max_abs, static_cast<double>(std::fabs(a - b)));
          }
        }
      }
      results.push_back({name, ms, weight_bytes / (ms * 1e6), exact, max_abs});
    };

    // Batch-1 GEMV replayed `batch` times: the route narrow widths used to
    // take.
    {
      constexpr std::uint32_t kWaves = 4;
      const std::size_t pairs = shape.m / 2;
      const std::size_t grid = (pairs + kWaves - 1) / kWaves;
      for (std::size_t batch = 2; batch <= kMaxBatch; batch += 2) {
        const double ms = TimeKernel(
            [&] {
              for (std::size_t token = 0; token < batch; ++token) {
                RefGemv2Rows<kWaves><<<grid, kWaves * 32>>>(
                    d_w, d_x + (token * shape.k), d_y + (token * shape.m),
                    shape.m, shape.k);
              }
            },
            iters);
        check("gemv-per-row      b=" + std::to_string(batch), ms, batch);
      }
    }

#define RUN_LDS8(BATCH)                                                 \
  {                                                                     \
    constexpr std::uint32_t kWaves = 32;                                \
    const unsigned int grid =                                           \
        static_cast<unsigned int>((shape.m + kWaves - 1) / kWaves);     \
    const double ms = TimeKernel(                                       \
        [&] {                                                           \
          Lds8Gemm<kWaves, BATCH>                                       \
              <<<grid, kWaves * 32>>>(d_w, d_x, d_y, shape.m, shape.k); \
        },                                                              \
        iters);                                                         \
    check("lds8              b=" + std::to_string(BATCH), ms, BATCH);   \
  }

#define RUN_GEMV_SHAPED(BATCH, WAVES, ROWS, MINCU)                            \
  {                                                                           \
    const std::size_t groups = (shape.m + ROWS - 1) / ROWS;                   \
    const unsigned int grid =                                                 \
        static_cast<unsigned int>((groups + WAVES - 1) / WAVES);              \
    const double ms = TimeKernel(                                             \
        [&] {                                                                 \
          GemvShapedGemm<WAVES, BATCH, ROWS, MINCU>                           \
              <<<grid, WAVES * 32>>>(d_w, d_x, d_y, shape.m, shape.k);        \
        },                                                                    \
        iters);                                                               \
    check("gemvshaped w" #WAVES " r" #ROWS " b=" + std::to_string(BATCH), ms, \
          BATCH);                                                             \
  }

    RUN_LDS8(2)
    RUN_LDS8(4)
    RUN_LDS8(6)
    RUN_LDS8(8)

    RUN_GEMV_SHAPED(8, 4, 2, 4)

#define RUN_MULTIROW(BATCH, WAVES, ROWS, STAGED, MINCU)                  \
  {                                                                      \
    const std::size_t groups = (shape.m + (ROWS) - 1) / (ROWS);          \
    const unsigned int grid =                                            \
        static_cast<unsigned int>((groups + (WAVES) - 1) / (WAVES));     \
    const double ms = TimeKernel(                                        \
        [&] {                                                            \
          LdsMultiRowGemm<WAVES, BATCH, ROWS, STAGED, MINCU>             \
              <<<grid, (WAVES) * 32>>>(d_w, d_x, d_y, shape.m, shape.k); \
        },                                                               \
        iters);                                                          \
    check("multirow w" #WAVES " r" #ROWS " s" #STAGED " b=" +            \
              std::to_string(BATCH),                                     \
          ms, BATCH);                                                    \
  }

    RUN_MULTIROW(8, 16, 2, 4, 1)

#define RUN_VEC(BATCH, WAVES, ROWS, STAGED, PAD, MINCU)                  \
  {                                                                      \
    const std::size_t groups = (shape.m + (ROWS) - 1) / (ROWS);          \
    const unsigned int grid =                                            \
        static_cast<unsigned int>((groups + (WAVES) - 1) / (WAVES));     \
    const double ms = TimeKernel(                                        \
        [&] {                                                            \
          LdsVecGemm<WAVES, BATCH, ROWS, STAGED, PAD, MINCU>             \
              <<<grid, (WAVES) * 32>>>(d_w, d_x, d_y, shape.m, shape.k); \
        },                                                               \
        iters);                                                          \
    check("vec w" #WAVES " r" #ROWS " s" #STAGED " p" #PAD " b=" +       \
              std::to_string(BATCH),                                     \
          ms, BATCH);                                                    \
  }

    RUN_VEC(8, 16, 2, 8, 4, 1)
    RUN_VEC(8, 16, 1, 8, 4, 1)
    RUN_VEC(8, 32, 2, 8, 4, 1)
    RUN_VEC(8, 8, 2, 8, 4, 2)
    RUN_VEC(8, 16, 4, 8, 4, 1)
    RUN_VEC(7, 16, 2, 7, 4, 1)
    RUN_VEC(6, 16, 2, 6, 4, 1)
    RUN_VEC(5, 16, 2, 5, 4, 1)
    RUN_VEC(4, 16, 2, 4, 4, 1)
    RUN_VEC(3, 16, 2, 3, 4, 1)
    RUN_VEC(2, 16, 2, 2, 4, 1)

    for (const auto& r : results) {
      std::printf("  %-34s %8.3f ms  %7.1f GB/s  %s\n", r.name.c_str(), r.ms,
                  r.gbps, r.exact ? "exact" : "MISMATCH");
      if (!r.exact) {
        std::printf("      max abs diff %.9g\n", r.max_abs);
      }
    }

    HIP_OK(hipFree(d_w));
    HIP_OK(hipFree(d_x));
    HIP_OK(hipFree(d_y));
    HIP_OK(hipFree(d_ref));
  }
  return 0;
}
