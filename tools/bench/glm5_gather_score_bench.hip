// GLM-5.3-Flash MLA GatherScore: scalar per-head oracle vs LDS-staged
// warp-tiled candidate. Production scores 64 heads x 512-dim F16 selected
// KV rows per token (kSelTile=64, one thread per key per head block), which
// re-reads every selected row once per head (64x redundant global traffic,
// scalar half loads). The candidate stages a 32-key row tile into LDS
// (padded stride, conflict-free), lanes = 32 keys, and loops a head slice
// with broadcast q and vector dim-pair reads.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <vector>

#define CHECK(call)                                                    \
  do {                                                                 \
    hipError_t err = (call);                                           \
    if (err != hipSuccess) {                                           \
      std::fprintf(stderr, "HIP %s @%d: %s\n", #call, __LINE__,        \
                   hipGetErrorString(err));                            \
      std::exit(1);                                                    \
    }                                                                  \
  } while (0)

namespace {

constexpr uint32_t kHeads = 64;
constexpr uint32_t kRank = 512;
constexpr uint32_t kOracleTile = 64;
constexpr uint32_t kTile = 32;
constexpr uint32_t kChunks = kRank / 2U;  // 4 B chunks per row
constexpr uint32_t kStride = kRank + 2;   // padded halves

__global__ void OracleKernel(const float* __restrict__ q_abs,
                             const __half* __restrict__ cache,
                             const int32_t* __restrict__ sel, uint32_t n_sel,
                             uint32_t tile, uint32_t heads,
                             float* __restrict__ scores) {
  const uint32_t tile_i = blockIdx.x;
  const uint32_t head = blockIdx.y;
  const uint32_t j = tile_i * tile + threadIdx.x;
  if (j >= n_sel) {
    return;
  }
  const float* qa = q_abs + static_cast<size_t>(head) * kRank;
  const __half* row = cache + static_cast<size_t>(sel[j]) * kRank;
  float acc = 0.0F;
  for (uint32_t i = 0; i < kRank; ++i) {
    acc += qa[i] * __half2float(row[i]);
  }
  scores[(static_cast<size_t>(tile_i) * heads + head) * tile + threadIdx.x] =
      acc;
}

// Lanes are the 32 keys of the staged tile. Staging: one pass per uint4
// chunk, lane = row (32 independent streams, full lines consumed like the
// oracle does); LDS stores hit distinct banks (word = row*257 + 4*chunk).
// Compute: per head, 256 conflict-free half2 loads (word = lane*257 + d2)
// against broadcast q pairs. Output layout matches the oracle exactly.
__global__ __launch_bounds__(32) void StagedKernel(
    const float* __restrict__ q_abs, const __half* __restrict__ cache,
    const int32_t* __restrict__ sel, uint32_t n_sel, uint32_t hspan,
    float* __restrict__ scores) {
  __shared__ __half rows[kTile][kStride];  // stride 514 halves = 257 words
  const uint32_t lane = threadIdx.x;
  const uint32_t row_in_tile = lane;
  const uint32_t jrow = blockIdx.x * kTile + row_in_tile;
  const bool valid = jrow < n_sel;
  const uint32_t* src = reinterpret_cast<const uint32_t*>(
      cache + static_cast<size_t>(valid ? sel[jrow] : 0) * kRank);
  for (uint32_t u = 0; u < kChunks; ++u) {
    const uint32_t v = valid ? src[u] : 0U;
    *reinterpret_cast<uint32_t*>(&rows[row_in_tile][u * 2U]) = v;
  }
  __syncwarp();
  const uint32_t head0 = blockIdx.y * hspan;
  for (uint32_t h = 0; h < hspan; ++h) {
    const float* qa = q_abs + static_cast<size_t>(head0 + h) * kRank;
    float acc = 0.0F;
    for (uint32_t d2 = 0; d2 < kRank / 2U; ++d2) {
      const __half2 hv = *reinterpret_cast<const __half2*>(&rows[lane][d2 * 2U]);
      const float2 f = __half22float2(hv);
      acc = __fmaf_rn(f.x, qa[d2 * 2U], acc);
      acc = __fmaf_rn(f.y, qa[d2 * 2U + 1U], acc);
    }
    const uint32_t key = blockIdx.x * kTile + lane;
    if (key < n_sel) {
      scores[(static_cast<size_t>(key) / kOracleTile) * (kHeads * kOracleTile) +
             static_cast<size_t>(head0 + h) * kOracleTile +
             (key % kOracleTile)] = acc;
    }
  }
}

// Vectorized oracle: identical grid/thread mapping and same ascending dim
// order, but the row is streamed in 16 B chunks and q in float4 pairs.
__global__ void VectorKernel(const float* __restrict__ q_abs,
                             const __half* __restrict__ cache,
                             const int32_t* __restrict__ sel, uint32_t n_sel,
                             uint32_t tile, uint32_t heads,
                             float* __restrict__ scores) {
  const uint32_t tile_i = blockIdx.x;
  const uint32_t head = blockIdx.y;
  const uint32_t j = tile_i * tile + threadIdx.x;
  if (j >= n_sel) {
    return;
  }
  const float4* qa = reinterpret_cast<const float4*>(
      q_abs + static_cast<size_t>(head) * kRank);
  const uint4* row = reinterpret_cast<const uint4*>(
      cache + static_cast<size_t>(sel[j]) * kRank);
  float acc = 0.0F;
  for (uint32_t c = 0; c < kRank / 8U; ++c) {
    const uint4 r = row[c];
    const __half2* h2 = reinterpret_cast<const __half2*>(&r);
    const float4 q0 = qa[2U * c];
    const float4 q1 = qa[2U * c + 1U];
    const float2 f0 = __half22float2(h2[0]);
    const float2 f1 = __half22float2(h2[1]);
    const float2 f2 = __half22float2(h2[2]);
    const float2 f3 = __half22float2(h2[3]);
    acc = __fmaf_rn(f0.x, q0.x, acc);
    acc = __fmaf_rn(f0.y, q0.y, acc);
    acc = __fmaf_rn(f1.x, q0.z, acc);
    acc = __fmaf_rn(f1.y, q0.w, acc);
    acc = __fmaf_rn(f2.x, q1.x, acc);
    acc = __fmaf_rn(f2.y, q1.y, acc);
    acc = __fmaf_rn(f3.x, q1.z, acc);
    acc = __fmaf_rn(f3.y, q1.w, acc);
  }
  scores[(static_cast<size_t>(tile_i) * heads + head) * tile + threadIdx.x] =
      acc;
}

// Chain: identical grid and access pattern as the oracle, but the 512-long
// FMA dependency chain is split into 4 independent quarter-row chains
// combined in ascending order. Isolates accumulator latency; tiny legal
// reassociation (NMSE contract), not bitwise.
__global__ void ChainKernel(const float* __restrict__ q_abs,
                            const __half* __restrict__ cache,
                            const int32_t* __restrict__ sel, uint32_t n_sel,
                            uint32_t tile, uint32_t heads,
                            float* __restrict__ scores) {
  const uint32_t tile_i = blockIdx.x;
  const uint32_t head = blockIdx.y;
  const uint32_t j = tile_i * tile + threadIdx.x;
  if (j >= n_sel) {
    return;
  }
  const float* qa = q_abs + static_cast<size_t>(head) * kRank;
  const __half* row = cache + static_cast<size_t>(sel[j]) * kRank;
  constexpr uint32_t q = kRank / 4U;
  float a0 = 0.0F, a1 = 0.0F, a2 = 0.0F, a3 = 0.0F;
  for (uint32_t i = 0; i < q; ++i) {
    a0 = __fmaf_rn(__half2float(row[i]), qa[i], a0);
    a1 = __fmaf_rn(__half2float(row[q + i]), qa[q + i], a1);
    a2 = __fmaf_rn(__half2float(row[2U * q + i]), qa[2U * q + i], a2);
    a3 = __fmaf_rn(__half2float(row[3U * q + i]), qa[3U * q + i], a3);
  }
  scores[(static_cast<size_t>(tile_i) * heads + head) * tile + threadIdx.x] =
      ((a0 + a1) + a2) + a3;
}

// RowHeads: one warp per selected key; the lane loads two consecutive uint4
// chunks (fully coalesced 1 KB per warp), unpacks its 16 halves into
// registers, then loops a head slice, pairing each register against the
// float4 q pair covering the same dims, and reduces with shuffles.
// Eliminates the oracle's 32-transaction scattered warp loads and its 64x
// per-row instruction redundancy.
__global__ __launch_bounds__(256) void RowHeadsKernel(
    const float* __restrict__ q_abs, const __half* __restrict__ cache,
    const int32_t* __restrict__ sel, uint32_t n_sel, uint32_t hspan,
    float* __restrict__ scores) {
  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t key = blockIdx.x * 8U + (threadIdx.x >> 5U);
  if (key >= n_sel) {
    return;
  }
  const uint4* rp = reinterpret_cast<const uint4*>(
      cache + static_cast<size_t>(sel[key]) * kRank);
  const uint4 r0 = rp[lane];
  const uint4 r1 = rp[lane + 32U];
  float v[16];
  const __half2* p0 = reinterpret_cast<const __half2*>(&r0);
  const __half2* p1 = reinterpret_cast<const __half2*>(&r1);
#pragma unroll
  for (uint32_t k = 0; k < 4U; ++k) {
    const float2 f0 = __half22float2(p0[k]);
    const float2 f1 = __half22float2(p1[k]);
    v[2U * k] = f0.x;
    v[2U * k + 1U] = f0.y;
    v[8U + 2U * k] = f1.x;
    v[8U + 2U * k + 1U] = f1.y;
  }
  const uint32_t head0 = blockIdx.y * hspan;
  for (uint32_t h = 0; h < hspan; ++h) {
    const uint32_t head = head0 + h;
    const float4* qa = reinterpret_cast<const float4*>(
        q_abs + static_cast<size_t>(head) * kRank);
    const float4 c0 = qa[2U * lane];
    const float4 c1 = qa[2U * lane + 1U];
    const float4 c2 = qa[kRank / 8U + 2U * lane];
    const float4 c3 = qa[kRank / 8U + 2U * lane + 1U];
    float acc = 0.0F;
    acc = __fmaf_rn(v[0], c0.x, acc);
    acc = __fmaf_rn(v[1], c0.y, acc);
    acc = __fmaf_rn(v[2], c0.z, acc);
    acc = __fmaf_rn(v[3], c0.w, acc);
    acc = __fmaf_rn(v[4], c1.x, acc);
    acc = __fmaf_rn(v[5], c1.y, acc);
    acc = __fmaf_rn(v[6], c1.z, acc);
    acc = __fmaf_rn(v[7], c1.w, acc);
    acc = __fmaf_rn(v[8], c2.x, acc);
    acc = __fmaf_rn(v[9], c2.y, acc);
    acc = __fmaf_rn(v[10], c2.z, acc);
    acc = __fmaf_rn(v[11], c2.w, acc);
    acc = __fmaf_rn(v[12], c3.x, acc);
    acc = __fmaf_rn(v[13], c3.y, acc);
    acc = __fmaf_rn(v[14], c3.z, acc);
    acc = __fmaf_rn(v[15], c3.w, acc);
    acc += __shfl_down_sync(0xffffffffULL, acc, 16U);
    acc += __shfl_down_sync(0xffffffffULL, acc, 8U);
    acc += __shfl_down_sync(0xffffffffULL, acc, 4U);
    acc += __shfl_down_sync(0xffffffffULL, acc, 2U);
    acc += __shfl_down_sync(0xffffffffULL, acc, 1U);
    if (lane == 0U) {
      scores[(static_cast<size_t>(key) / kOracleTile) *
                 (kHeads * kOracleTile) +
             static_cast<size_t>(head) * kOracleTile + (key % kOracleTile)] =
          acc;
    }
  }
}

uint32_t Splitmix(uint32_t x) {
  x += 0x9e3779b9U;
  x = (x ^ (x >> 16)) * 0x85ebca6bU;
  x = (x ^ (x >> 13)) * 0xc2b2ae35U;
  return x ^ (x >> 16);
}

struct Case {
  uint32_t n_sel;
  uint32_t prefix;  // sel drawn from [0, prefix)
};

void RunConfig(const float* q, const __half* cache, const int32_t* sel,
               uint32_t n_sel, float* ref, float* out, const char* label) {
  const uint32_t tiles_o = (n_sel + kOracleTile - 1) / kOracleTile;
  const uint32_t tiles_s = (n_sel + kTile - 1) / kTile;
  const size_t nscores = static_cast<size_t>((n_sel + kOracleTile - 1) /
                                             kOracleTile) *
                         kOracleTile * kHeads;
  OracleKernel<<<dim3(tiles_o, kHeads), kOracleTile>>>(q, cache, sel, n_sel,
                                                       kOracleTile, kHeads,
                                                       ref);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());

  // Candidate with head slice hspan 16.
  auto launch_cand = [&](uint32_t hspan) {
    StagedKernel<<<dim3(tiles_s, kHeads / hspan), 32>>>(q, cache, sel, n_sel,
                                                        hspan, out);
  };
  launch_cand(16);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  std::vector<float> got(nscores), want(nscores);
  CHECK(hipMemcpy(got.data(), out, nscores * sizeof(float),
                  hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(want.data(), ref, nscores * sizeof(float),
                  hipMemcpyDeviceToHost));
  double se = 0.0, ww = 0.0, mx = 0.0;
  for (size_t i = 0; i < nscores; ++i) {
    const double d = double(got[i]) - double(want[i]);
    se += d * d;
    ww += double(want[i]) * double(want[i]);
    mx = std::max(mx, std::fabs(d));
  }
  std::printf("  NMSE=%.3e maxabs=%.3e\n", se / (ww + 1e-30), mx);

  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);
  auto bench = [&](const char* name, auto run) {
    for (int i = 0; i < 5; ++i) {
      run();
    }
    CHECK(hipDeviceSynchronize());
    std::vector<float> ts;
    for (int i = 0; i < 20; ++i) {
      hipEventRecord(e0);
      run();
      hipEventRecord(e1);
      CHECK(hipEventSynchronize(e1));
      float ms = 0.0F;
      hipEventElapsedTime(&ms, e0, e1);
      ts.push_back(ms);
    }
    std::sort(ts.begin(), ts.end());
    const float med = ts[ts.size() / 2];
    const double gb =
        double(n_sel) * kRank * 2.0 / (med * 1e-3) / 1e9;
    std::printf("  %-14s %8.1f us  uniqKV %7.1f GB/s\n", name, med * 1e3, gb);
    return med;
  };
  auto nmse = [&](float* o) {
    std::vector<float> got(nscores), want(nscores);
    CHECK(hipMemcpy(got.data(), o, nscores * sizeof(float),
                    hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(want.data(), ref, nscores * sizeof(float),
                    hipMemcpyDeviceToHost));
    double se = 0.0, ww = 0.0;
    for (size_t i = 0; i < nscores; ++i) {
      se += (got[i] - want[i]) * (got[i] - want[i]);
      ww += double(want[i]) * double(want[i]);
    }
    return se / (ww + 1e-30);
  };
  float* vec = out;
  const float to = bench("oracle(64x)", [&] {
    OracleKernel<<<dim3(tiles_o, kHeads), kOracleTile>>>(
        q, cache, sel, n_sel, kOracleTile, kHeads, ref);
  });
  VectorKernel<<<dim3(tiles_o, kHeads), kOracleTile>>>(
      q, cache, sel, n_sel, kOracleTile, kHeads, out);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  {
    std::vector<float> got(nscores), want(nscores);
    CHECK(hipMemcpy(got.data(), out, nscores * sizeof(float),
                    hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(want.data(), ref, nscores * sizeof(float),
                    hipMemcpyDeviceToHost));
    double se = 0.0, ww = 0.0;
    for (size_t i = 0; i < nscores; ++i) {
      se += (got[i] - want[i]) * (got[i] - want[i]);
      ww += double(want[i]) * double(want[i]);
    }
    std::printf("  vector NMSE=%.3e\n", se / (ww + 1e-30));
  }
  const float tv = bench("vector", [&] {
    VectorKernel<<<dim3(tiles_o, kHeads), kOracleTile>>>(
        q, cache, sel, n_sel, kOracleTile, kHeads, vec);
  });
  std::printf("  vector speedup %.2fx\n", to / tv);
  const float tchain = bench("chain", [&] {
    ChainKernel<<<dim3(tiles_o, kHeads), kOracleTile>>>(
        q, cache, sel, n_sel, kOracleTile, kHeads, vec);
  });
  std::printf("  chain NMSE=%.3e speedup %.2fx\n", nmse(vec), to / tchain);
  for (uint32_t hs : {8U, 16U, 32U, 64U}) {
    const std::string nm = "rowheads" + std::to_string(hs);
    const float tr = bench(nm.c_str(), [&] {
      RowHeadsKernel<<<dim3((n_sel + 7U) / 8U, kHeads / hs), 256>>>(
          q, cache, sel, n_sel, hs, vec);
    });
    std::printf("  rowheads%u NMSE=%.3e speedup %.2fx\n", hs, nmse(vec),
                to / tr);
  }
  float best = 1e30F;
  const char* bn = "";
  for (uint32_t hs : {8U, 16U, 32U, 64U}) {
    const std::string nm = "staged" + std::to_string(hs);
    const float tc = bench(nm.c_str(), [&] { launch_cand(hs); });
    (void)label;
    if (tc < best) {
      best = tc;
      bn = "staged";
    }
  }
  std::printf("  speedup %.2fx (%s hspan-sweep vs oracle)\n", to / best, bn);
  hipEventDestroy(e0);
  hipEventDestroy(e1);
}

}  // namespace

int main() {
  const std::vector<Case> cases = {{256, 256},
                                   {1024, 1024},
                                   {2048, 2048},
                                   {2048, 65536},
                                   {8192, 65536}};
  const uint32_t max_prefix = 65536;
  const size_t rows = max_prefix + kTile;
  std::vector<float> hq(static_cast<size_t>(kHeads) * kRank);
  std::vector<__half> hc(rows * kRank);
  for (auto& x : hq) {
    x = float(Splitmix(uint32_t(&x - hq.data())) % 2000) / 1000.0F - 1.0F;
  }
  for (size_t i = 0; i < hc.size(); ++i) {
    hc[i] = __float2half_rn(float(Splitmix(uint32_t(i)) % 2000) / 1000.0F -
                            1.0F);
  }
  float* q;
  __half* cache;
  int32_t* sel;
  float* ref;
  float* out;
  CHECK(hipMalloc(&q, hq.size() * sizeof(float)));
  CHECK(hipMalloc(&cache, hc.size() * sizeof(__half)));
  CHECK(hipMalloc(&sel, rows * sizeof(int32_t)));
  CHECK(hipMalloc(&ref, static_cast<size_t>(512) * kHeads * kOracleTile *
                            sizeof(float)));
  CHECK(hipMalloc(&out, static_cast<size_t>(512) * kHeads * kOracleTile *
                            sizeof(float)));
  CHECK(hipMemcpy(q, hq.data(), hq.size() * sizeof(float),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(cache, hc.data(), hc.size() * sizeof(__half),
                  hipMemcpyHostToDevice));
  for (const Case& c : cases) {
    std::vector<int32_t> hs(c.n_sel);
    for (uint32_t j = 0; j < c.n_sel; ++j) {
      hs[j] = int32_t(Splitmix(j + c.n_sel * 7U + c.prefix) % c.prefix);
    }
    CHECK(hipMemcpy(sel, hs.data(), hs.size() * sizeof(int32_t),
                    hipMemcpyHostToDevice));
    std::printf("n_sel=%u prefix=%u\n", c.n_sel, c.prefix);
    RunConfig(q, cache, sel, c.n_sel, ref, out, "run");
  }
  return 0;
}
