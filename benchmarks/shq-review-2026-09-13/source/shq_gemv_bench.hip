// Throwaway SHQ4-T16-G64-U4Z / SHQ8-T16-G64 decode GEMV microbenchmark for
// gfx1151 (investigation only; lives in the session scratchpad).
//
//   nix develop -c tools/bench/build.sh <abs path>/shq_gemv_bench.hip
//   /tmp/shq_gemv_bench [passes]
//
// Layouts follow docs/QUANTIZATION.md: 128-byte K16xN16 microtiles for 4-bit
// (256 bytes for 8-bit), output-lane-major, low nibble = even k, BF16 scale
// and packed UINT4 zero planes per (lane, K64). A CPU oracle checks results.
// Each shape is replicated into a >= 1 GiB arena and copies are rotated so
// timed passes stream cold memory (the 32 MB MALL would otherwise hide DRAM).

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CHECK(expr)                                                          \
  do {                                                                       \
    const hipError_t status = (expr);                                        \
    if (status != hipSuccess) {                                              \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,                 \
                   hipGetErrorString(status));                               \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

namespace {

constexpr double kPeak = 241.0;
constexpr std::uint32_t kG = 64;
constexpr std::uint32_t kSub = kG / 16;

__host__ __device__ inline float Bf16(std::uint16_t b) {
  std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
  float f;
  __builtin_memcpy(&f, &u, sizeof(f));
  return f;
}
std::uint16_t ToBf16(float f) {
  std::uint32_t u;
  __builtin_memcpy(&u, &f, sizeof(u));
  u += 0x7FFFu + ((u >> 16) & 1u);
  return static_cast<std::uint16_t>(u >> 16);
}

struct Planes {
  std::vector<std::uint8_t> w4;      // [tile][group][sub][lane][8]
  std::vector<std::uint8_t> w4l;     // [tile][group][lane][sub][8] (lane-contiguous variant)
  std::vector<std::uint8_t> w8;      // [tile][group][lane][64] int8
  std::vector<std::uint16_t> scale;  // [tile][group][lane]
  std::vector<std::uint8_t> zero;    // [tile][group][8]
  std::vector<float> ref4, ref8;
  std::uint32_t n = 0, k = 0;
};

Planes Make(std::uint32_t n, std::uint32_t k, const std::vector<float>& x) {
  Planes p;
  p.n = n; p.k = k;
  const std::uint32_t kg = k / kG;
  const std::size_t tg_count = static_cast<std::size_t>(n / 16) * kg;
  p.w4.assign(tg_count * 512, 0);
  p.w4l.assign(tg_count * 512, 0);
  p.w8.assign(tg_count * 1024, 0);
  p.scale.assign(tg_count * 16, 0);
  p.zero.assign(tg_count * 8, 0);
  p.ref4.assign(n, 0.0F); p.ref8.assign(n, 0.0F);
  std::mt19937 rng(7);
  std::normal_distribution<float> dist(0.0F, 0.02F);
  std::vector<float> row(k);
  for (std::uint32_t nn = 0; nn < n; ++nn) {
    for (auto& v : row) v = dist(rng);
    const std::uint32_t tile = nn / 16, lane = nn % 16;
    float acc4 = 0, acc8 = 0;
    for (std::uint32_t g = 0; g < kg; ++g) {
      const float* r = row.data() + g * kG;
      float wmin = r[0], wmax = r[0], amax = 0;
      for (std::uint32_t j = 0; j < kG; ++j) { wmin = std::min(wmin, r[j]); wmax = std::max(wmax, r[j]); amax = std::max(amax, std::fabs(r[j])); }
      float s = (wmax - wmin) / 15.0F; if (s <= 0) s = 1;
      s = Bf16(ToBf16(s));
      const int z = std::clamp(static_cast<int>(std::lrint(-wmin / s)), 0, 15);
      const std::size_t tg = static_cast<std::size_t>(tile) * kg + g;
      p.scale[tg * 16 + lane] = ToBf16(s);
      p.zero[tg * 8 + lane / 2] |= static_cast<std::uint8_t>(z << (4 * (lane & 1)));
      // 8-bit tier shares the scale plane in this bench (own scale in the real
      // contract); use s8 = amax/127 stored in the same slot? Keep it simple:
      // separate scale computed on the fly below and packed after the 4-bit one.
      float part4 = 0, part8 = 0;
      const float s8 = Bf16(ToBf16(amax / 127.0F + 1e-30F));
      for (std::uint32_t j = 0; j < kG; ++j) {
        const int q = std::clamp(static_cast<int>(std::lrint(r[j] / s)) + z, 0, 15);
        p.w4[((tg * kSub + j / 16) * 16 + lane) * 8 + (j % 16) / 2] |= static_cast<std::uint8_t>(q << (4 * (j & 1)));
        p.w4l[(tg * 16 + lane) * 32 + (j / 16) * 8 + (j % 16) / 2] |= static_cast<std::uint8_t>(q << (4 * (j & 1)));
        const int q8 = std::clamp(static_cast<int>(std::lrint(r[j] / s8)), -127, 127);
        p.w8[(tg * 16 + lane) * 64 + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q8));
        part4 += static_cast<float>(q - z) * x[g * kG + j];
        part8 += static_cast<float>(q8) * x[g * kG + j];
      }
      acc4 += s * part4;
      acc8 += s8 * part8;
      // stash s8 in the zero plane's sibling: reuse scale plane for 4-bit, and
      // a second scale plane for 8-bit.
      (void)0;
    }
    p.ref4[nn] = acc4; p.ref8[nn] = acc8;
  }
  return p;
}

// SHQ8 needs its own scale plane; build it in a second pass to keep Make simple.
std::vector<std::uint16_t> MakeScale8(std::uint32_t n, std::uint32_t k) {
  const std::uint32_t kg = k / kG;
  std::vector<std::uint16_t> s8(static_cast<std::size_t>(n / 16) * kg * 16, 0);
  std::mt19937 rng(7);
  std::normal_distribution<float> dist(0.0F, 0.02F);
  std::vector<float> row(k);
  for (std::uint32_t nn = 0; nn < n; ++nn) {
    for (auto& v : row) v = dist(rng);
    const std::uint32_t tile = nn / 16, lane = nn % 16;
    for (std::uint32_t g = 0; g < kg; ++g) {
      float amax = 0;
      for (std::uint32_t j = 0; j < kG; ++j) amax = std::max(amax, std::fabs(row[g * kG + j]));
      s8[(static_cast<std::size_t>(tile) * kg + g) * 16 + lane] = ToBf16(amax / 127.0F + 1e-30F);
    }
  }
  return s8;
}

// ----------------------------------------------------------------------------
// Kernel A: one wave per N16 tile, half-waves alternate K64 groups (SHQ4).
// ----------------------------------------------------------------------------
template<std::uint32_t Waves, std::uint32_t U>
__launch_bounds__(Waves * 32, 2) __global__
    void Gemv4Split(const std::uint8_t* __restrict__ w, const std::uint16_t* __restrict__ sc,
                    const std::uint8_t* __restrict__ zp, const std::uint16_t* __restrict__ xb,
                    float* __restrict__ y, std::uint32_t n, std::uint32_t k) {
  extern __shared__ std::uint16_t xs[];
  const std::uint32_t tid = threadIdx.x;
  for (std::uint32_t i = tid; i < k; i += Waves * 32) xs[i] = xb[i];
  __syncthreads();
  const std::uint32_t lane = tid & 31u, half = lane >> 4, l = lane & 15u;
  const std::uint32_t tile = blockIdx.x * Waves + (tid >> 5);
  if (tile * 16 >= n) return;
  const std::uint32_t kg = k / kG;
  const std::uint8_t* wt = w + static_cast<std::size_t>(tile) * kg * 512;
  const std::uint16_t* st = sc + static_cast<std::size_t>(tile) * kg * 16;
  const std::uint8_t* zt = zp + static_cast<std::size_t>(tile) * kg * 8;
  float acc = 0;
  for (std::uint32_t g = half; g < kg; g += 2 * U) {
    uint2 c[U][kSub]; std::uint16_t sb[U]; std::uint8_t zb[U];
#pragma unroll
    for (std::uint32_t u = 0; u < U; ++u) {
      const std::uint32_t gg = g + 2 * u;
      if (gg < kg) {
        const std::uint8_t* wg = wt + static_cast<std::size_t>(gg) * 512;
#pragma unroll
        for (std::uint32_t s = 0; s < kSub; ++s) c[u][s] = *reinterpret_cast<const uint2*>(wg + s * 128 + l * 8);
        sb[u] = st[gg * 16 + l]; zb[u] = zt[gg * 8 + l / 2];
      } else {
#pragma unroll
        for (std::uint32_t s = 0; s < kSub; ++s) c[u][s] = make_uint2(0, 0);
        sb[u] = 0; zb[u] = 0;
      }
    }
#pragma unroll
    for (std::uint32_t u = 0; u < U; ++u) {
      const std::uint32_t gg = g + 2 * u;
      if (gg >= kg) break;
      const float s = Bf16(sb[u]);
      const int z = (zb[u] >> (4 * (l & 1))) & 15;
      const std::uint16_t* xg = xs + gg * kG;
      float part = 0;
#pragma unroll
      for (std::uint32_t sub = 0; sub < kSub; ++sub) {
        const std::uint32_t words[2] = {c[u][sub].x, c[u][sub].y};
#pragma unroll
        for (std::uint32_t wd = 0; wd < 2; ++wd) {
#pragma unroll
          for (std::uint32_t b = 0; b < 4; ++b) {
            const std::uint32_t byte = (words[wd] >> (8 * b)) & 0xFFu;
            const std::uint32_t kk = sub * 16 + wd * 8 + b * 2;
            part += static_cast<float>(static_cast<int>(byte & 15u) - z) * Bf16(xg[kk]);
            part += static_cast<float>(static_cast<int>(byte >> 4) - z) * Bf16(xg[kk + 1]);
          }
        }
      }
      acc += s * part;
    }
  }
  acc += __shfl_xor(acc, 16);
  if (half == 0) y[tile * 16 + l] = acc;
}


// ----------------------------------------------------------------------------
// Kernel A2: as A, but the weight plane is [tile][group][lane][sub][8], so a
// lane's 64 codes of a group are 32 contiguous bytes: two 16-byte loads.
// ----------------------------------------------------------------------------
template<std::uint32_t Waves, std::uint32_t U>
__launch_bounds__(Waves * 32, 2) __global__
    void Gemv4Lane(const std::uint8_t* __restrict__ w, const std::uint16_t* __restrict__ sc,
                   const std::uint8_t* __restrict__ zp, const std::uint16_t* __restrict__ xb,
                   float* __restrict__ y, std::uint32_t n, std::uint32_t k) {
  extern __shared__ std::uint16_t xs[];
  const std::uint32_t tid = threadIdx.x;
  for (std::uint32_t i = tid; i < k; i += Waves * 32) xs[i] = xb[i];
  __syncthreads();
  const std::uint32_t lane = tid & 31u, half = lane >> 4, l = lane & 15u;
  const std::uint32_t tile = blockIdx.x * Waves + (tid >> 5);
  if (tile * 16 >= n) return;
  const std::uint32_t kg = k / kG;
  const std::uint8_t* wt = w + static_cast<std::size_t>(tile) * kg * 512;
  const std::uint16_t* st = sc + static_cast<std::size_t>(tile) * kg * 16;
  const std::uint8_t* zt = zp + static_cast<std::size_t>(tile) * kg * 8;
  float acc = 0;
  for (std::uint32_t g = half; g < kg; g += 2 * U) {
    uint4 c[U][2]; std::uint16_t sb[U]; std::uint8_t zb[U];
#pragma unroll
    for (std::uint32_t u = 0; u < U; ++u) {
      const std::uint32_t gg = g + 2 * u;
      if (gg < kg) {
        const std::uint8_t* wg = wt + static_cast<std::size_t>(gg) * 512 + l * 32;
        c[u][0] = *reinterpret_cast<const uint4*>(wg);
        c[u][1] = *reinterpret_cast<const uint4*>(wg + 16);
        sb[u] = st[gg * 16 + l]; zb[u] = zt[gg * 8 + l / 2];
      } else {
        c[u][0] = make_uint4(0, 0, 0, 0); c[u][1] = make_uint4(0, 0, 0, 0); sb[u] = 0; zb[u] = 0;
      }
    }
#pragma unroll
    for (std::uint32_t u = 0; u < U; ++u) {
      const std::uint32_t gg = g + 2 * u;
      if (gg >= kg) break;
      const float s = Bf16(sb[u]);
      const int z = (zb[u] >> (4 * (l & 1))) & 15;
      const std::uint16_t* xg = xs + gg * kG;
      float part = 0;
      const std::uint32_t words[8] = {c[u][0].x, c[u][0].y, c[u][0].z, c[u][0].w, c[u][1].x, c[u][1].y, c[u][1].z, c[u][1].w};
#pragma unroll
      for (std::uint32_t wd = 0; wd < 8; ++wd) {
#pragma unroll
        for (std::uint32_t b = 0; b < 4; ++b) {
          const std::uint32_t byte = (words[wd] >> (8 * b)) & 0xFFu;
          const std::uint32_t kk = wd * 8 + b * 2;
          part += static_cast<float>(static_cast<int>(byte & 15u) - z) * Bf16(xg[kk]);
          part += static_cast<float>(static_cast<int>(byte >> 4) - z) * Bf16(xg[kk + 1]);
        }
      }
      acc += s * part;
    }
  }
  acc += __shfl_xor(acc, 16);
  if (half == 0) y[tile * 16 + l] = acc;
}

// ----------------------------------------------------------------------------
// Kernel B: two tiles per wave (one per half-wave), every lane walks all
// groups. Better for small K where kernel A has too little work per wave.
// ----------------------------------------------------------------------------
template<std::uint32_t Waves, std::uint32_t U>
__launch_bounds__(Waves * 32, 2) __global__
    void Gemv4TilePair(const std::uint8_t* __restrict__ w, const std::uint16_t* __restrict__ sc,
                       const std::uint8_t* __restrict__ zp, const std::uint16_t* __restrict__ xb,
                       float* __restrict__ y, std::uint32_t n, std::uint32_t k) {
  extern __shared__ std::uint16_t xs[];
  const std::uint32_t tid = threadIdx.x;
  for (std::uint32_t i = tid; i < k; i += Waves * 32) xs[i] = xb[i];
  __syncthreads();
  const std::uint32_t lane = tid & 31u, half = lane >> 4, l = lane & 15u;
  const std::uint32_t tile = (blockIdx.x * Waves + (tid >> 5)) * 2 + half;
  if (tile * 16 >= n) return;
  const std::uint32_t kg = k / kG;
  const std::uint8_t* wt = w + static_cast<std::size_t>(tile) * kg * 512;
  const std::uint16_t* st = sc + static_cast<std::size_t>(tile) * kg * 16;
  const std::uint8_t* zt = zp + static_cast<std::size_t>(tile) * kg * 8;
  float acc = 0;
  for (std::uint32_t g = 0; g < kg; g += U) {
    uint2 c[U][kSub]; std::uint16_t sb[U]; std::uint8_t zb[U];
#pragma unroll
    for (std::uint32_t u = 0; u < U; ++u) {
      const std::uint32_t gg = g + u;
      if (gg < kg) {
        const std::uint8_t* wg = wt + static_cast<std::size_t>(gg) * 512;
#pragma unroll
        for (std::uint32_t s = 0; s < kSub; ++s) c[u][s] = *reinterpret_cast<const uint2*>(wg + s * 128 + l * 8);
        sb[u] = st[gg * 16 + l]; zb[u] = zt[gg * 8 + l / 2];
      } else {
#pragma unroll
        for (std::uint32_t s = 0; s < kSub; ++s) c[u][s] = make_uint2(0, 0);
        sb[u] = 0; zb[u] = 0;
      }
    }
#pragma unroll
    for (std::uint32_t u = 0; u < U; ++u) {
      const std::uint32_t gg = g + u;
      if (gg >= kg) break;
      const float s = Bf16(sb[u]);
      const int z = (zb[u] >> (4 * (l & 1))) & 15;
      const std::uint16_t* xg = xs + gg * kG;
      float part = 0;
#pragma unroll
      for (std::uint32_t sub = 0; sub < kSub; ++sub) {
        const std::uint32_t words[2] = {c[u][sub].x, c[u][sub].y};
#pragma unroll
        for (std::uint32_t wd = 0; wd < 2; ++wd) {
#pragma unroll
          for (std::uint32_t b = 0; b < 4; ++b) {
            const std::uint32_t byte = (words[wd] >> (8 * b)) & 0xFFu;
            const std::uint32_t kk = sub * 16 + wd * 8 + b * 2;
            part += static_cast<float>(static_cast<int>(byte & 15u) - z) * Bf16(xg[kk]);
            part += static_cast<float>(static_cast<int>(byte >> 4) - z) * Bf16(xg[kk + 1]);
          }
        }
      }
      acc += s * part;
    }
  }
  y[tile * 16 + l] = acc;
}

// ----------------------------------------------------------------------------
// Kernel C: SHQ8 (INT8 codes, BF16 scale per lane/K64), half-waves alternate
// groups, 16-byte loads (a lane's 64 codes are 4 x 16 B contiguous).
// ----------------------------------------------------------------------------
template<std::uint32_t Waves, std::uint32_t U>
__launch_bounds__(Waves * 32, 2) __global__
    void Gemv8Split(const std::uint8_t* __restrict__ w, const std::uint16_t* __restrict__ sc,
                    const std::uint16_t* __restrict__ xb, float* __restrict__ y,
                    std::uint32_t n, std::uint32_t k) {
  extern __shared__ std::uint16_t xs[];
  const std::uint32_t tid = threadIdx.x;
  for (std::uint32_t i = tid; i < k; i += Waves * 32) xs[i] = xb[i];
  __syncthreads();
  const std::uint32_t lane = tid & 31u, half = lane >> 4, l = lane & 15u;
  const std::uint32_t tile = blockIdx.x * Waves + (tid >> 5);
  if (tile * 16 >= n) return;
  const std::uint32_t kg = k / kG;
  const std::uint8_t* wt = w + static_cast<std::size_t>(tile) * kg * 1024;
  const std::uint16_t* st = sc + static_cast<std::size_t>(tile) * kg * 16;
  float acc = 0;
  for (std::uint32_t g = half; g < kg; g += 2 * U) {
    uint4 c[U][4]; std::uint16_t sb[U];
#pragma unroll
    for (std::uint32_t u = 0; u < U; ++u) {
      const std::uint32_t gg = g + 2 * u;
      if (gg < kg) {
        const std::uint8_t* wg = wt + static_cast<std::size_t>(gg) * 1024 + l * 64;
#pragma unroll
        for (std::uint32_t q = 0; q < 4; ++q) c[u][q] = *reinterpret_cast<const uint4*>(wg + q * 16);
        sb[u] = st[gg * 16 + l];
      } else {
#pragma unroll
        for (std::uint32_t q = 0; q < 4; ++q) c[u][q] = make_uint4(0, 0, 0, 0);
        sb[u] = 0;
      }
    }
#pragma unroll
    for (std::uint32_t u = 0; u < U; ++u) {
      const std::uint32_t gg = g + 2 * u;
      if (gg >= kg) break;
      const float s = Bf16(sb[u]);
      const std::uint16_t* xg = xs + gg * kG;
      float part = 0;
#pragma unroll
      for (std::uint32_t q = 0; q < 4; ++q) {
        const std::uint32_t words[4] = {c[u][q].x, c[u][q].y, c[u][q].z, c[u][q].w};
#pragma unroll
        for (std::uint32_t wd = 0; wd < 4; ++wd) {
#pragma unroll
          for (std::uint32_t b = 0; b < 4; ++b) {
            const int v = static_cast<std::int8_t>((words[wd] >> (8 * b)) & 0xFFu);
            part += static_cast<float>(v) * Bf16(xg[q * 16 + wd * 4 + b]);
          }
        }
      }
      acc += s * part;
    }
  }
  acc += __shfl_xor(acc, 16);
  if (half == 0) y[tile * 16 + l] = acc;
}

// Pure streaming read of `bytes` with 16-byte loads, same grid size class.
__global__ void StreamRead(const uint4* __restrict__ src, std::uint32_t* sink, std::size_t count) {
  std::uint32_t s = 0;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
    const uint4 v = src[i];
    s += v.x ^ v.y ^ v.z ^ v.w;
  }
  if (s == 0x9E3779B9u) sink[0] = s;
}

struct Shape { const char* name; std::uint32_t n, k; };

template<typename F>
double TimeMs(F&& fn, int passes) {
  hipEvent_t a, b;
  CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
  CHECK(hipEventRecord(a));
  for (int p = 0; p < passes; ++p) fn(p);
  CHECK(hipEventRecord(b)); CHECK(hipEventSynchronize(b));
  float ms = 0; CHECK(hipEventElapsedTime(&ms, a, b));
  CHECK(hipEventDestroy(a)); CHECK(hipEventDestroy(b));
  return ms / passes;
}

void Report(const char* label, double ms, double bytes, const float* got, const std::vector<float>& ref, std::uint32_t n) {
  const double gbs = bytes / (ms / 1e3) / 1e9;
  std::printf("  %-30s %8.3f ms %7.1f GB/s %6.1f%%", label, ms, gbs, 100.0 * gbs / kPeak);
  if (got != nullptr) {
    double max_rel = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
      const double d = std::fabs(static_cast<double>(got[i]) - ref[i]) / (std::fabs(static_cast<double>(ref[i])) + 1e-3);
      max_rel = std::max(max_rel, d);
    }
    std::printf("  max rel err %.1e %s", max_rel, max_rel < 1e-3 ? "ok" : "MISMATCH");
  }
  std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
  const int passes = argc > 1 ? std::atoi(argv[1]) : 40;
  hipDeviceProp_t props{};
  CHECK(hipGetDeviceProperties(&props, 0));
  std::printf("device: %s %s  peak read %.0f GB/s  passes %d\n", props.name, props.gcnArchName, kPeak, passes);
  const Shape shapes[] = {
      {"qwen ffn_gate 17408x5120", 17408, 5120},
      {"qwen ffn_down 5120x17408", 5120, 17408},
      {"qwen attn_qkv 10240x5120", 10240, 5120},
      {"ds4 attn_q_b 32768x1024", 32768, 1024},
      {"ds4 attn_output_a 8192x4096", 8192, 4096},
      {"ds4 attn_output_b 4096x8192", 4096, 8192},
      {"ds4 shexp gate 2048x4096", 2048, 4096},
  };
  for (const Shape& sh : shapes) {
    std::vector<float> x(sh.k);
    std::vector<std::uint16_t> xb(sh.k);
    std::mt19937 rng(11);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    for (std::uint32_t i = 0; i < sh.k; ++i) { xb[i] = ToBf16(dist(rng)); x[i] = Bf16(xb[i]); }
    Planes p = Make(sh.n, sh.k, x);
    std::vector<std::uint16_t> s8 = MakeScale8(sh.n, sh.k);
    const std::size_t b4 = p.w4.size() + p.scale.size() * 2 + p.zero.size();
    const std::size_t b8 = p.w8.size() + s8.size() * 2;
    const std::uint32_t copies = static_cast<std::uint32_t>(std::max<std::size_t>(2, ((std::size_t{1} << 30) + b4 - 1) / b4));
    const std::uint32_t copies8 = static_cast<std::uint32_t>(std::max<std::size_t>(2, ((std::size_t{1} << 30) + b8 - 1) / b8));
    std::uint8_t *dw4, *dw4l, *dz, *dw8; std::uint16_t *ds4, *ds8, *dx; float* dy; std::uint32_t* sink;
    CHECK(hipMalloc(&dw4, p.w4.size() * copies)); CHECK(hipMalloc(&dw4l, p.w4l.size() * copies)); CHECK(hipMalloc(&ds4, p.scale.size() * 2 * copies)); CHECK(hipMalloc(&dz, p.zero.size() * copies));
    CHECK(hipMalloc(&dw8, p.w8.size() * copies8)); CHECK(hipMalloc(&ds8, s8.size() * 2 * copies8));
    CHECK(hipMalloc(&dx, sh.k * 2)); CHECK(hipMalloc(&dy, sh.n * 4)); CHECK(hipMalloc(&sink, 4));
    for (std::uint32_t c = 0; c < copies; ++c) {
      CHECK(hipMemcpy(dw4 + c * p.w4.size(), p.w4.data(), p.w4.size(), hipMemcpyHostToDevice));
      CHECK(hipMemcpy(dw4l + c * p.w4l.size(), p.w4l.data(), p.w4l.size(), hipMemcpyHostToDevice));
      CHECK(hipMemcpy(ds4 + c * p.scale.size(), p.scale.data(), p.scale.size() * 2, hipMemcpyHostToDevice));
      CHECK(hipMemcpy(dz + c * p.zero.size(), p.zero.data(), p.zero.size(), hipMemcpyHostToDevice));
    }
    for (std::uint32_t c = 0; c < copies8; ++c) {
      CHECK(hipMemcpy(dw8 + c * p.w8.size(), p.w8.data(), p.w8.size(), hipMemcpyHostToDevice));
      CHECK(hipMemcpy(ds8 + c * s8.size(), s8.data(), s8.size() * 2, hipMemcpyHostToDevice));
    }
    CHECK(hipMemcpy(dx, xb.data(), sh.k * 2, hipMemcpyHostToDevice));
    std::vector<float> got(sh.n);
    const std::size_t lds = sh.k * 2;
    std::printf("\n%s: shq4 %.1f MB (%.4f B/w) x%u, shq8 %.1f MB x%u, %u tiles\n", sh.name, b4 / 1e6, static_cast<double>(b4) / (static_cast<double>(sh.n) * sh.k), copies, b8 / 1e6, copies8, sh.n / 16);

    {  // streaming read ceiling over the 4-bit arena bytes
      const std::size_t cnt = p.w4.size() / 16;
      auto fn = [&](int c) { hipLaunchKernelGGL(StreamRead, dim3(4096), dim3(256), 0, 0, reinterpret_cast<const uint4*>(dw4 + (c % copies) * p.w4.size()), sink, cnt); CHECK(hipGetLastError()); };
      for (std::uint32_t c = 0; c < copies; ++c) fn(c);
      CHECK(hipDeviceSynchronize());
      Report("stream read (16 B loads)", TimeMs(fn, passes), static_cast<double>(p.w4.size()), nullptr, p.ref4, sh.n);
    }
#define RUN4(KERNEL, W, U, LABEL)                                                                                   \
    {                                                                                                               \
      const std::uint32_t tiles = sh.n / 16;                                                                        \
      const std::uint32_t per_block = (std::string(#KERNEL) == "Gemv4TilePair") ? W * 2 : W;                         \
      const dim3 grid((tiles + per_block - 1) / per_block);                                                         \
      auto fn = [&](int c) {                                                                                        \
        hipLaunchKernelGGL((KERNEL<W, U>), grid, dim3(W * 32), lds, 0, dw4 + (c % copies) * p.w4.size(),            \
                           ds4 + (c % copies) * p.scale.size(), dz + (c % copies) * p.zero.size(), dx, dy, sh.n, sh.k); \
        CHECK(hipGetLastError());                                                                                   \
      };                                                                                                            \
      for (std::uint32_t c = 0; c < copies; ++c) fn(c);                                                             \
      CHECK(hipDeviceSynchronize());                                                                                \
      const double ms = TimeMs(fn, passes);                                                                         \
      CHECK(hipMemcpy(got.data(), dy, sh.n * 4, hipMemcpyDeviceToHost));                                            \
      Report(LABEL, ms, static_cast<double>(b4), got.data(), p.ref4, sh.n);                                          \
    }
#define RUN8(W, U, LABEL)                                                                                           \
    {                                                                                                               \
      const std::uint32_t tiles = sh.n / 16;                                                                        \
      const dim3 grid((tiles + W - 1) / W);                                                                         \
      auto fn = [&](int c) {                                                                                        \
        hipLaunchKernelGGL((Gemv8Split<W, U>), grid, dim3(W * 32), lds, 0, dw8 + (c % copies8) * p.w8.size(),       \
                           ds8 + (c % copies8) * s8.size(), dx, dy, sh.n, sh.k);                                    \
        CHECK(hipGetLastError());                                                                                   \
      };                                                                                                            \
      for (std::uint32_t c = 0; c < copies8; ++c) fn(c);                                                            \
      CHECK(hipDeviceSynchronize());                                                                                \
      const double ms = TimeMs(fn, passes);                                                                         \
      CHECK(hipMemcpy(got.data(), dy, sh.n * 4, hipMemcpyDeviceToHost));                                            \
      Report(LABEL, ms, static_cast<double>(b8), got.data(), p.ref8, sh.n);                                          \
    }
#define RUNL(W, U, LABEL)                                                                                           \
    {                                                                                                               \
      const std::uint32_t tiles = sh.n / 16;                                                                        \
      const dim3 grid((tiles + W - 1) / W);                                                                         \
      auto fn = [&](int c) {                                                                                        \
        hipLaunchKernelGGL((Gemv4Lane<W, U>), grid, dim3(W * 32), lds, 0, dw4l + (c % copies) * p.w4l.size(),        \
                           ds4 + (c % copies) * p.scale.size(), dz + (c % copies) * p.zero.size(), dx, dy, sh.n, sh.k); \
        CHECK(hipGetLastError());                                                                                   \
      };                                                                                                            \
      for (std::uint32_t c = 0; c < copies; ++c) fn(c);                                                             \
      CHECK(hipDeviceSynchronize());                                                                                \
      const double ms = TimeMs(fn, passes);                                                                         \
      CHECK(hipMemcpy(got.data(), dy, sh.n * 4, hipMemcpyDeviceToHost));                                            \
      Report(LABEL, ms, static_cast<double>(b4), got.data(), p.ref4, sh.n);                                          \
    }
    RUN4(Gemv4Split, 4, 2, "shq4 split w4 u2");
    RUN4(Gemv4Split, 4, 8, "shq4 split w4 u8");
    RUN4(Gemv4Split, 8, 8, "shq4 split w8 u8");
    RUNL(4, 2, "shq4 lane-contig w4 u2");
    RUNL(4, 4, "shq4 lane-contig w4 u4");
    RUNL(4, 8, "shq4 lane-contig w4 u8");
    RUNL(8, 4, "shq4 lane-contig w8 u4");
    RUN8(4, 2, "shq8 split w4 u2");
    CHECK(hipFree(dw4)); CHECK(hipFree(dw4l)); CHECK(hipFree(ds4)); CHECK(hipFree(dz)); CHECK(hipFree(dw8)); CHECK(hipFree(ds8));
    CHECK(hipFree(dx)); CHECK(hipFree(dy)); CHECK(hipFree(sink));
  }
  return 0;
}

