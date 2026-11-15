// GLM-5.3-Flash Rows-tier microbenchmark: Q8_0 multi-token GEMV launches of
// the ssm_f_b / attn_k_b / attn_v_b shapes. Compares the shipped per-token
// slice launches (A), one launch with the slices in grid.y (B), and an
// LDS-staged-activation variant in both launch forms (C, D). All variants
// must produce bit-identical output to A. The F32 router/idx_proj shapes
// rerun the same three launch forms plus the four-row tile (T4, T4y).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct Q8_0Block {
  __half d;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8_0Block) == 34, "block_q8_0 must be 34 bytes");

__device__ __forceinline__ float WarpReduceSum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset);
  }
  return v;
}

// Shipped kernel (literal from gemv.hip.cpp) plus grid.y slice offset.
template <std::uint32_t Rows, bool kY>
__global__ void RowsCur(const Q8_0Block* __restrict__ w,
                        const float* __restrict__ x, std::uint32_t xs,
                        float* __restrict__ out, std::uint32_t os,
                        std::uint32_t rows, std::uint32_t cols,
                        std::uint32_t group_rows, std::uint32_t group_advance) {
  if (kY) {
    x += static_cast<std::size_t>(blockIdx.y) * Rows * xs;
    out += static_cast<std::size_t>(blockIdx.y) * Rows * os;
  }
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  if (group_rows != 0U) {
    x += static_cast<std::size_t>(r / group_rows) * group_advance;
  }
  const Q8_0Block* __restrict__ row =
      w + static_cast<std::size_t>(r) * (cols / 32U);
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc[Rows] = {};
#pragma unroll 2
  for (std::uint32_t i = lane; i < cols; i += 32U) {
    const Q8_0Block& blk = row[i >> 5];
    const float b = __half2float(blk.d) * static_cast<float>(blk.qs[i & 31U]);
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      acc[o] += b * x[static_cast<std::size_t>(o) * xs + i];
    }
  }
#pragma unroll
  for (std::uint32_t o = 0; o < Rows; ++o) {
    acc[o] = WarpReduceSum(acc[o]);
  }
  if (lane == 0) {
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      out[static_cast<std::size_t>(o) * os + r] = acc[o];
    }
  }
}

// LDS-staged variant: the block's 4 rows share one activation group (row
// blocks never straddle a group boundary for these shapes), so the token x
// slice is staged once into shared memory and the identical dot sequence
// reads it from LDS. Output is bit-identical to RowsCur.
template <std::uint32_t Rows, bool kY>
__global__ void RowsLds(const Q8_0Block* __restrict__ w,
                        const float* __restrict__ x, std::uint32_t xs,
                        float* __restrict__ out, std::uint32_t os,
                        std::uint32_t rows, std::uint32_t cols,
                        std::uint32_t group_rows, std::uint32_t group_advance) {
  if (kY) {
    x += static_cast<std::size_t>(blockIdx.y) * Rows * xs;
    out += static_cast<std::size_t>(blockIdx.y) * Rows * os;
  }
  __shared__ float sx[static_cast<std::size_t>(Rows) * 512U];
  const std::uint32_t xbase =
      group_rows != 0U
          ? (blockIdx.x * 4U) / group_rows * group_advance
          : 0U;
  for (std::uint32_t i = threadIdx.x; i < Rows * cols; i += 128U) {
    const std::uint32_t o = i / cols;
    const std::uint32_t j = i - o * cols;
    sx[static_cast<std::size_t>(o) * 512U + j] =
        x[static_cast<std::size_t>(o) * xs + xbase + j];
  }
  __syncthreads();
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const Q8_0Block* __restrict__ row =
      w + static_cast<std::size_t>(r) * (cols / 32U);
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc[Rows] = {};
#pragma unroll 2
  for (std::uint32_t i = lane; i < cols; i += 32U) {
    const Q8_0Block& blk = row[i >> 5];
    const float b = __half2float(blk.d) * static_cast<float>(blk.qs[i & 31U]);
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      acc[o] += b * sx[static_cast<std::size_t>(o) * 512U + i];
    }
  }
#pragma unroll
  for (std::uint32_t o = 0; o < Rows; ++o) {
    acc[o] = WarpReduceSum(acc[o]);
  }
  if (lane == 0) {
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      out[static_cast<std::size_t>(o) * os + r] = acc[o];
    }
  }
}

// Register-tiled variant: one wave accumulates RPT weight rows at once.
// The activation loads x[o*xs+i] are identical for all rows, so each load
// feeds RPT FMAs. Every row's accumulation order and shuffle tree are
// unchanged, so the per-row dot stays bit-identical to RowsCur.
template <std::uint32_t Rows, std::uint32_t RPT, bool kY>
__global__ void RowsTile(const Q8_0Block* __restrict__ w,
                         const float* __restrict__ x, std::uint32_t xs,
                         float* __restrict__ out, std::uint32_t os,
                         std::uint32_t rows, std::uint32_t cols,
                         std::uint32_t group_rows, std::uint32_t group_advance) {
  if (kY) {
    x += static_cast<std::size_t>(blockIdx.y) * Rows * xs;
    out += static_cast<std::size_t>(blockIdx.y) * Rows * os;
  }
  const std::uint32_t r0 = (blockIdx.x * 4U + (threadIdx.x >> 5)) * RPT;
  const std::uint32_t lane = threadIdx.x & 31U;
  if (r0 >= rows) {
    return;
  }
  if (group_rows != 0U) {
    x += static_cast<std::size_t>(r0 / group_rows) * group_advance;
  }
  const Q8_0Block* row[RPT];
#pragma unroll
  for (std::uint32_t u = 0; u < RPT; ++u) {
    row[u] = w + static_cast<std::size_t>(r0 + u) * (cols / 32U);
  }
  float acc[RPT][Rows] = {};
#pragma unroll 2
  for (std::uint32_t i = lane; i < cols; i += 32U) {
    float b[RPT];
#pragma unroll
    for (std::uint32_t u = 0; u < RPT; ++u) {
      const Q8_0Block& blk = row[u][i >> 5];
      b[u] = __half2float(blk.d) * static_cast<float>(blk.qs[i & 31U]);
    }
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      const float xv = x[static_cast<std::size_t>(o) * xs + i];
#pragma unroll
      for (std::uint32_t u = 0; u < RPT; ++u) {
        acc[u][o] += b[u] * xv;
      }
    }
  }
#pragma unroll
  for (std::uint32_t u = 0; u < RPT; ++u) {
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      acc[u][o] = WarpReduceSum(acc[u][o]);
    }
    if (lane == 0) {
#pragma unroll
      for (std::uint32_t o = 0; o < Rows; ++o) {
        out[static_cast<std::size_t>(o) * os + r0 + u] = acc[u][o];
      }
    }
  }
}

// F32 mirrors of the Rows-tier kernels (router/idx_proj shapes).
template <std::uint32_t Rows, bool kY>
__global__ void RowsCurF32(const float* __restrict__ w,
                           const float* __restrict__ x, std::uint32_t xs,
                           float* __restrict__ out, std::uint32_t os,
                           std::uint32_t rows, std::uint32_t cols) {
  if (kY) {
    x += static_cast<std::size_t>(blockIdx.y) * Rows * xs;
    out += static_cast<std::size_t>(blockIdx.y) * Rows * os;
  }
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const float* __restrict__ row = w + static_cast<std::size_t>(r) * cols;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc[Rows] = {};
#pragma unroll 2
  for (std::uint32_t i = lane; i < cols; i += 32U) {
    const float b = row[i];
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      acc[o] += b * x[static_cast<std::size_t>(o) * xs + i];
    }
  }
#pragma unroll
  for (std::uint32_t o = 0; o < Rows; ++o) {
    acc[o] = WarpReduceSum(acc[o]);
  }
  if (lane == 0) {
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      out[static_cast<std::size_t>(o) * os + r] = acc[o];
    }
  }
}

template <std::uint32_t Rows, std::uint32_t RPT>
__global__ void RowsTileF32(const float* __restrict__ w,
                            const float* __restrict__ x, std::uint32_t xs,
                            float* __restrict__ out, std::uint32_t os,
                             std::uint32_t rows, std::uint32_t cols) {
  x += static_cast<std::size_t>(blockIdx.y) * Rows * xs;
  out += static_cast<std::size_t>(blockIdx.y) * Rows * os;
  const std::uint32_t r0 = (blockIdx.x * 4U + (threadIdx.x >> 5)) * RPT;
  const std::uint32_t lane = threadIdx.x & 31U;
  const float* row[RPT];
#pragma unroll
  for (std::uint32_t u = 0; u < RPT; ++u) {
    row[u] = w + static_cast<std::size_t>(r0 + u) * cols;
  }
  float acc[RPT][Rows] = {};
#pragma unroll 2
  for (std::uint32_t i = lane; i < cols; i += 32U) {
    float wv[RPT];
#pragma unroll
    for (std::uint32_t u = 0; u < RPT; ++u) {
      wv[u] = row[u][i];
    }
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      const float xv = x[static_cast<std::size_t>(o) * xs + i];
#pragma unroll
      for (std::uint32_t u = 0; u < RPT; ++u) {
        acc[u][o] += wv[u] * xv;
      }
    }
  }
#pragma unroll
  for (std::uint32_t u = 0; u < RPT; ++u) {
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      acc[u][o] = WarpReduceSum(acc[u][o]);
    }
    if (lane == 0) {
#pragma unroll
      for (std::uint32_t o = 0; o < Rows; ++o) {
        out[static_cast<std::size_t>(o) * os + r0 + u] = acc[u][o];
      }
    }
  }
}

struct Shape {
  const char* name;
  std::uint32_t rows;   // total weight rows
  std::uint32_t cols;
  std::uint32_t group_rows;
  std::uint32_t group_advance;
  std::uint32_t xs;     // per-token activation stride
};

constexpr std::uint32_t kTokens = 16U;  // production slice width

auto Check = [](hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
};

template <bool kLds, bool kY, std::uint32_t Rows>
float RunOne(const Shape& s, const Q8_0Block* w, const float* x, float* out,
             std::uint32_t grid_y, std::uint32_t repeats) {
  const std::uint32_t grid = (s.rows + 3U) / 4U;
  hipEvent_t e0, e1;
  Check(hipEventCreate(&e0), "event");
  Check(hipEventCreate(&e1), "event");
  for (std::uint32_t it = 0; it < 2U; ++it) {
    for (std::uint32_t sl = 0; sl < (kY ? 1U : 16U); ++sl) {
      const std::uint32_t yy = kY ? grid_y : 1U;
      if (kLds) {
        RowsLds<Rows, kY><<<dim3(grid, yy), 128>>>(
            w, kY ? x : x + sl * Rows * s.xs, s.xs,
            kY ? out : out + static_cast<std::size_t>(sl) * Rows * s.rows,
            s.rows, s.rows, s.cols, s.group_rows, s.group_advance);
      } else {
        RowsCur<Rows, kY><<<dim3(grid, yy), 128>>>(
            w, kY ? x : x + sl * Rows * s.xs, s.xs,
            kY ? out : out + static_cast<std::size_t>(sl) * Rows * s.rows,
            s.rows, s.rows, s.cols, s.group_rows, s.group_advance);
      }
    }
  }
  float best = 1e30F;
  for (std::uint32_t it = 0; it < repeats; ++it) {
    Check(hipEventRecord(e0), "rec0");
    for (std::uint32_t sl = 0; sl < (kY ? 1U : 16U); ++sl) {
      const std::uint32_t yy = kY ? grid_y : 1U;
      if (kLds) {
        RowsLds<Rows, kY><<<dim3(grid, yy), 128>>>(
            w, kY ? x : x + sl * Rows * s.xs, s.xs,
            kY ? out : out + static_cast<std::size_t>(sl) * Rows * s.rows,
            s.rows, s.rows, s.cols, s.group_rows, s.group_advance);
      } else {
        RowsCur<Rows, kY><<<dim3(grid, yy), 128>>>(
            w, kY ? x : x + sl * Rows * s.xs, s.xs,
            kY ? out : out + static_cast<std::size_t>(sl) * Rows * s.rows,
            s.rows, s.rows, s.cols, s.group_rows, s.group_advance);
      }
    }
    Check(hipEventRecord(e1), "rec1");
    Check(hipEventSynchronize(e1), "sync");
    float ms = 0;
    Check(hipEventElapsedTime(&ms, e0, e1), "elapsed");
    if (ms < best) {
      best = ms;
    }
  }
  Check(hipEventDestroy(e0), "d0");
  Check(hipEventDestroy(e1), "d1");
  return best;  // min over timed production calls
}

template <std::uint32_t RPT, bool kY>
float RunTile(const Shape& s, const Q8_0Block* w, const float* x, float* out,
              std::uint32_t repeats) {
  const std::uint32_t grid = s.rows / (4U * RPT);
  hipEvent_t e0, e1;
  Check(hipEventCreate(&e0), "event");
  Check(hipEventCreate(&e1), "event");
  const auto once = [&] {
    for (std::uint32_t sl = 0; sl < (kY ? 1U : 16U); ++sl) {
      RowsTile<16U, RPT, kY><<<dim3(grid, kY ? 16U : 1U), 128>>>(
          w, kY ? x : x + sl * 16U * s.xs, s.xs,
          kY ? out : out + static_cast<std::size_t>(sl) * 16U * s.rows,
          s.rows, s.rows, s.cols, s.group_rows, s.group_advance);
    }
  };
  once();
  once();
  float best = 1e30F;
  for (std::uint32_t it = 0; it < repeats; ++it) {
    Check(hipEventRecord(e0), "rec0");
    once();
    Check(hipEventRecord(e1), "rec1");
    Check(hipEventSynchronize(e1), "sync");
    float ms = 0;
    Check(hipEventElapsedTime(&ms, e0, e1), "elapsed");
    if (ms < best) {
      best = ms;
    }
  }
  Check(hipEventDestroy(e0), "d0");
  Check(hipEventDestroy(e1), "d1");
  return best;
}

template <bool kY>
float RunF32(std::uint32_t rows, std::uint32_t cols, std::uint32_t xs,
             const float* w, const float* x, float* out,
             std::uint32_t repeats) {
  const std::uint32_t grid = (rows + 3U) / 4U;
  hipEvent_t e0, e1;
  Check(hipEventCreate(&e0), "event");
  Check(hipEventCreate(&e1), "event");
  const auto once = [&] {
    for (std::uint32_t sl = 0; sl < (kY ? 1U : 16U); ++sl) {
      RowsCurF32<16U, kY><<<dim3(grid, kY ? 16U : 1U), 128>>>(
          w, kY ? x : x + sl * 16U * xs, xs,
          kY ? out : out + static_cast<std::size_t>(sl) * 16U * rows, rows,
          rows, cols);
    }
  };
  once();
  once();
  float best = 1e30F;
  for (std::uint32_t it = 0; it < repeats; ++it) {
    Check(hipEventRecord(e0), "rec0");
    once();
    Check(hipEventRecord(e1), "rec1");
    Check(hipEventSynchronize(e1), "sync");
    float ms = 0;
    Check(hipEventElapsedTime(&ms, e0, e1), "elapsed");
    if (ms < best) {
      best = ms;
    }
  }
  Check(hipEventDestroy(e0), "d0");
  Check(hipEventDestroy(e1), "d1");
  return best;
}

template <std::uint32_t RPT, bool kY>
float RunTileF32(std::uint32_t rows, std::uint32_t cols, std::uint32_t xs,
                 const float* w, const float* x, float* out,
                 std::uint32_t repeats) {
  const std::uint32_t grid = rows / (4U * RPT);
  hipEvent_t e0, e1;
  Check(hipEventCreate(&e0), "event");
  Check(hipEventCreate(&e1), "event");
  const auto once = [&] {
    for (std::uint32_t sl = 0; sl < (kY ? 1U : 16U); ++sl) {
      RowsTileF32<16U, RPT><<<dim3(grid, kY ? 16U : 1U), 128>>>(
          w, kY ? x : x + sl * 16U * xs, xs,
          kY ? out : out + static_cast<std::size_t>(sl) * 16U * rows, rows,
          rows, cols);
    }
  };
  once();
  once();
  float best = 1e30F;
  for (std::uint32_t it = 0; it < repeats; ++it) {
    Check(hipEventRecord(e0), "rec0");
    once();
    Check(hipEventRecord(e1), "rec1");
    Check(hipEventSynchronize(e1), "sync");
    float ms = 0;
    Check(hipEventElapsedTime(&ms, e0, e1), "elapsed");
    if (ms < best) {
      best = ms;
    }
  }
  Check(hipEventDestroy(e0), "d0");
  Check(hipEventDestroy(e1), "d1");
  return best;
}

void BenchF32(const char* name, std::uint32_t rows, std::uint32_t cols,
              std::uint32_t xs, std::uint32_t repeats) {
  const std::size_t w_count = static_cast<std::size_t>(rows) * cols;
  const std::size_t x_count = static_cast<std::size_t>(256U) * xs;
  const std::size_t o_count = static_cast<std::size_t>(256U) * rows;
  float* w = nullptr;
  float* x = nullptr;
  float* o_ref = nullptr;
  float* o_new = nullptr;
  Check(hipMalloc(&w, w_count * sizeof(float)), "fw");
  Check(hipMalloc(&x, x_count * sizeof(float)), "fx");
  Check(hipMalloc(&o_ref, o_count * sizeof(float)), "fo");
  Check(hipMalloc(&o_new, o_count * sizeof(float)), "fn");
  std::vector<float> hw(w_count);
  for (auto& v : hw) {
    v = static_cast<float>(rand() % 1000) * 0.001F;
  }
  std::vector<float> hx(x_count);
  for (auto& v : hx) {
    v = static_cast<float>(rand() % 1000) * 0.001F;
  }
  Check(hipMemcpy(w, hw.data(), w_count * sizeof(float), hipMemcpyHostToDevice),
        "fcw");
  Check(hipMemcpy(x, hx.data(), x_count * sizeof(float), hipMemcpyHostToDevice),
        "fcx");
  Check(hipMemset(o_ref, 0, o_count * sizeof(float)), "fm0");
  Check(hipMemset(o_new, 0xFF, o_count * sizeof(float)), "fm1");

  const float t_a = RunF32<false>(rows, cols, xs, w, x, o_ref, repeats);
  const float t_b = RunF32<true>(rows, cols, xs, w, x, o_new, repeats);
  Check(hipDeviceSynchronize(), "fs");
  std::vector<float> href(o_count), hnew(o_count);
  Check(hipMemcpy(href.data(), o_ref, o_count * sizeof(float),
                  hipMemcpyDeviceToHost),
        "fgr");
  Check(hipMemcpy(hnew.data(), o_new, o_count * sizeof(float),
                  hipMemcpyDeviceToHost),
        "fgn");
  const bool eq_b =
      std::memcmp(href.data(), hnew.data(), o_count * sizeof(float)) == 0;
  Check(hipMemset(o_new, 0xFF, o_count * sizeof(float)), "fm2");
  const float t_t4 = RunTileF32<4, false>(rows, cols, xs, w, x, o_new, repeats);
  Check(hipMemcpy(hnew.data(), o_new, o_count * sizeof(float),
                  hipMemcpyDeviceToHost),
        "fgt4");
  const bool eq_t4 =
      std::memcmp(href.data(), hnew.data(), o_count * sizeof(float)) == 0;
  Check(hipMemset(o_new, 0xFF, o_count * sizeof(float)), "fm3");
  const float t_t4y = RunTileF32<4, true>(rows, cols, xs, w, x, o_new, repeats);
  Check(hipMemcpy(hnew.data(), o_new, o_count * sizeof(float),
                  hipMemcpyDeviceToHost),
        "fgt4y");
  const bool eq_t4y =
      std::memcmp(href.data(), hnew.data(), o_count * sizeof(float)) == 0;

  std::printf("%-18s A=%7.1fus B(y)=%7.1fus T4=%7.1fus T4y=%7.1fus"
              "  exact B/T4/T4y=%d/%d/%d\n",
              name, t_a * 1000.0F, t_b * 1000.0F, t_t4 * 1000.0F,
              t_t4y * 1000.0F, eq_b, eq_t4, eq_t4y);
  hipFree(w);
  hipFree(x);
  hipFree(o_ref);
  hipFree(o_new);
}

}  // namespace

int main() {
  Check(hipSetDevice(0), "setdev");
  const std::vector<Shape> shapes = {
      {"ssm_f_b  8192x128", 8192U, 128U, 0U, 0U, 128U},
      {"attn_k_b 32768x256", 32768U, 256U, 512U, 256U, 16384U},
      {"attn_v_b 16384x512", 16384U, 512U, 256U, 512U, 32768U},
  };
  constexpr std::uint32_t kTokTotal = kTokens * 16U;  // 256 tokens (1 chunk)
  for (const Shape& s : shapes) {
    const std::size_t w_bytes =
        static_cast<std::size_t>(s.rows) * (s.cols / 32U) * sizeof(Q8_0Block);
    const std::size_t x_count =
        static_cast<std::size_t>(kTokTotal) * s.xs;
    const std::size_t o_count =
        static_cast<std::size_t>(kTokTotal) * s.rows;
    Q8_0Block* w = nullptr;
    float* x = nullptr;
    float* o_ref = nullptr;
    float* o_new = nullptr;
    Check(hipMalloc(&w, w_bytes), "w");
    Check(hipMalloc(&x, x_count * sizeof(float)), "x");
    Check(hipMalloc(&o_ref, o_count * sizeof(float)), "oref");
    Check(hipMalloc(&o_new, o_count * sizeof(float)), "onew");
    std::vector<Q8_0Block> hw(w_bytes / sizeof(Q8_0Block));
    for (auto& b : hw) {
      b.d = __float2half(0.01F);
      for (auto& q : b.qs) {
        q = static_cast<std::int8_t>(rand() & 0xFF);
      }
    }
    std::vector<float> hx(x_count);
    for (auto& v : hx) {
      v = static_cast<float>(rand() % 1000) * 0.001F;
    }
    Check(hipMemcpy(w, hw.data(), w_bytes, hipMemcpyHostToDevice), "cw");
    Check(hipMemcpy(x, hx.data(), x_count * sizeof(float),
                    hipMemcpyHostToDevice),
          "cx");
    Check(hipMemset(o_ref, 0, o_count * sizeof(float)), "m0");
    Check(hipMemset(o_new, 0xFF, o_count * sizeof(float)), "m1");

    const float t_a = RunOne<false, false, 16>(s, w, x, o_ref, 1U, 20U);
    const float t_b = RunOne<false, true, 16>(s, w, x, o_new, 16U, 20U);
    Check(hipDeviceSynchronize(), "s");
    std::vector<float> href(o_count), hnew(o_count);
    Check(hipMemcpy(href.data(), o_ref, o_count * sizeof(float),
                    hipMemcpyDeviceToHost),
          "gr");
    Check(hipMemcpy(hnew.data(), o_new, o_count * sizeof(float),
                    hipMemcpyDeviceToHost),
          "gn");
    const bool eq_b = std::memcmp(href.data(), hnew.data(),
                                  o_count * sizeof(float)) == 0;
    Check(hipMemset(o_new, 0xFF, o_count * sizeof(float)), "m2");
    const float t_e = RunTile<4, false>(s, w, x, o_new, 20U);
    Check(hipMemcpy(hnew.data(), o_new, o_count * sizeof(float),
                    hipMemcpyDeviceToHost),
          "gn2");
    const bool eq_e = std::memcmp(href.data(), hnew.data(),
                                  o_count * sizeof(float)) == 0;
    Check(hipMemset(o_new, 0xFF, o_count * sizeof(float)), "m3");
    const float t_f = RunTile<2, false>(s, w, x, o_new, 20U);
    Check(hipMemcpy(hnew.data(), o_new, o_count * sizeof(float),
                    hipMemcpyDeviceToHost),
          "gn3");
    const bool eq_f = std::memcmp(href.data(), hnew.data(),
                                  o_count * sizeof(float)) == 0;
    Check(hipMemset(o_new, 0xFF, o_count * sizeof(float)), "m4");
    const float t_g = RunTile<8, false>(s, w, x, o_new, 20U);
    Check(hipMemcpy(hnew.data(), o_new, o_count * sizeof(float),
                    hipMemcpyDeviceToHost),
          "gn4");
    const bool eq_g = std::memcmp(href.data(), hnew.data(),
                                  o_count * sizeof(float)) == 0;
    Check(hipMemset(o_new, 0xFF, o_count * sizeof(float)), "m5");
    const float t_h = RunTile<4, true>(s, w, x, o_new, 20U);
    Check(hipMemcpy(hnew.data(), o_new, o_count * sizeof(float),
                    hipMemcpyDeviceToHost),
          "gn5");
    const bool eq_h = std::memcmp(href.data(), hnew.data(),
                                  o_count * sizeof(float)) == 0;

    std::printf("%-18s A=%7.1fus E(r4)=%7.1fus F(r2)=%7.1fus G(r8)=%7.1fus"
                " H(r4y)=%7.1fus  exact E/F/G/H=%d/%d/%d/%d\n",
                s.name, t_a * 1000.0F, t_e * 1000.0F, t_f * 1000.0F,
                t_g * 1000.0F, t_h * 1000.0F, eq_e, eq_f, eq_g, eq_h);
    hipFree(w);
    hipFree(x);
    hipFree(o_ref);
    hipFree(o_new);
  }
  BenchF32("router 288x4096", 288U, 4096U, 4096U, 20U);
  BenchF32("idx_proj 32x4096", 32U, 4096U, 4096U, 20U);
  return 0;
}
