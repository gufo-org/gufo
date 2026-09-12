// Achievable-peak calibration for gfx1151 (Strix Halo).
//
// Build: nix develop -c tools/bench/build.sh gfx1151_peak
// Run: /tmp/gfx1151_peak
//
// Every accumulator is consumed, every thread has a separate checksum, and
// host references validate the work outside the timed region. Compute probes
// measure sustained issue rates; their only global traffic is the final
// checksum. These are measured ceilings, not model throughput predictions.
#include <hip/hip_runtime.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

#define HIP_OK(expr)                                                    \
  do {                                                                  \
    const hipError_t err_ = (expr);                                     \
    if (err_ != hipSuccess) {                                           \
      std::fprintf(stderr, "%s:%d hip error: %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(err_));                            \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

using u32x4 = __attribute__((__vector_size__(16))) std::uint32_t;
using i32x4 = __attribute__((__vector_size__(16))) int;
using i32x8 = __attribute__((__vector_size__(32))) int;
using bf16x16 = __attribute__((__vector_size__(32))) short;
using f16x16 = __attribute__((__vector_size__(32))) _Float16;
using f32x8 = __attribute__((__vector_size__(32))) float;

constexpr int kIters = 4096;
constexpr int kThreads = 256;
constexpr std::uint32_t kHashSeed = 2166136261U;

__host__ __device__ std::uint32_t Fold(std::uint32_t hash,
                                       std::uint32_t value) {
  return (hash ^ value) * 16777619U;
}

template<typename T>
__host__ __device__ std::uint32_t Word(T value) {
  static_assert(sizeof(T) == sizeof(std::uint32_t));
  return std::bit_cast<std::uint32_t>(value);
}

template<typename Vector>
__device__ std::uint32_t HashVector(std::uint32_t hash, Vector value) {
#pragma unroll
  for (unsigned i = 0; i < sizeof(Vector) / sizeof(value[0]); ++i)
    hash = Fold(hash, Word(value[i]));
  return hash;
}

__device__ unsigned ThreadIndex() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

// Distinct seeds keep all four dependency chains live. Uniform matrix inputs
// give exact integer-valued reference results for every output component.
__global__ void PeakWmmaI8(std::uint32_t* sink, int reps) {
  const i32x4 a = {0x01010101, 0x01010101, 0x01010101, 0x01010101};
  const i32x4 b = a;
  i32x8 c0 = {0, 1, 2, 3, 4, 5, 6, 7};
  i32x8 c1 = {8, 9, 10, 11, 12, 13, 14, 15};
  i32x8 c2 = {16, 17, 18, 19, 20, 21, 22, 23};
  i32x8 c3 = {24, 25, 26, 27, 28, 29, 30, 31};
  for (int r = 0; r < reps; ++r) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c0,
                                                      true);
      c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c1,
                                                      true);
      c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c2,
                                                      true);
      c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c3,
                                                      true);
    }
  }
  sink[ThreadIndex()] =
      HashVector(HashVector(HashVector(HashVector(kHashSeed, c0), c1), c2), c3);
}

template<bool Bf16>
__global__ void PeakWmmaFloat(std::uint32_t* sink, int reps) {
  using Input = std::conditional_t<Bf16, bf16x16, f16x16>;
  Input a;
  Input b;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    if constexpr (Bf16) {
      a[i] = 0x3f80;
      b[i] = 0x3f00;
    } else {
      a[i] = 1.0F;
      b[i] = 0.5F;
    }
  }
  const auto wmma = [&](f32x8 c) {
    if constexpr (Bf16)
      return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
    else
      return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
  };
  f32x8 c0 = {0, 1, 2, 3, 4, 5, 6, 7};
  f32x8 c1 = {8, 9, 10, 11, 12, 13, 14, 15};
  f32x8 c2 = {16, 17, 18, 19, 20, 21, 22, 23};
  f32x8 c3 = {24, 25, 26, 27, 28, 29, 30, 31};
  for (int r = 0; r < reps; ++r) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      c0 = wmma(c0);
      c1 = wmma(c1);
      c2 = wmma(c2);
      c3 = wmma(c3);
    }
  }
  sink[ThreadIndex()] =
      HashVector(HashVector(HashVector(HashVector(kHashSeed, c0), c1), c2), c3);
}

__global__ void PeakFmaF32(std::uint32_t* sink, int reps) {
  f32x8 acc;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    acc[i] = 1.0F + i + 0.5F * (threadIdx.x & 1U);
  for (int r = 0; r < reps; ++r) {
#pragma unroll
    for (int i = 0; i < 32; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        acc[j] = fmaf(acc[j], 1.0000001F, 0.0000001F);
    }
  }
  sink[ThreadIndex()] = HashVector(kHashSeed, acc);
}

// Two persistent WMMA updates plus N cvt/mul/fma triples. Keeping C across
// iterations prevents hoisting invariant matrix products out of the loop.
// This measures an instruction mix, not the complete Q8 dequantization loop.
template<int EpiTriples>
__global__ void PeakWmmaI8WithEpilogue(std::uint32_t* sink, int reps, float dw,
                                       float dx) {
  const i32x4 a = {0x01010101, 0x01010101, 0x01010101, 0x01010101};
  const i32x4 b = a;
  i32x8 c = {0, 1, 2, 3, 4, 5, 6, 7};
  f32x8 acc{};
  for (int r = 0; r < reps; ++r) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, true);
      c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, true);
#pragma unroll
      for (int j = 0; j < EpiTriples; ++j)
        acc[j] = fmaf(dw * static_cast<float>(c[j]), dx, acc[j]);
    }
  }
  sink[ThreadIndex()] = HashVector(HashVector(kHashSeed, c), acc);
}

__global__ void StreamRead(const u32x4* __restrict__ src, std::uint32_t* sink,
                           std::size_t n) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  u32x4 acc{};
  for (std::size_t i = ThreadIndex(); i < n; i += stride)
    acc += src[i];  // Unsigned wraparound is intentional and defined.
  sink[ThreadIndex()] = HashVector(kHashSeed, acc);
}

__global__ void StreamWrite(u32x4* __restrict__ dst, std::size_t n) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = ThreadIndex(); i < n; i += stride)
    dst[i] = u32x4{1, 2, 3, 4};
}

__global__ void StreamCopy(const u32x4* __restrict__ src,
                           u32x4* __restrict__ dst, std::size_t n) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = ThreadIndex(); i < n; i += stride)
    dst[i] = src[i];
}

__global__ void PeakLdsRead(std::uint32_t* sink, int reps) {
  // Volatile keeps repeated LDS loads in the timed loop.
  __shared__ volatile u32x4 buf[1024];
  for (unsigned i = threadIdx.x; i < 1024; i += blockDim.x)
    buf[i] = u32x4{i, i ^ 0x1234U, i * 3U + 7U, ~i};
  __syncthreads();
  u32x4 acc{};
  for (int r = 0; r < reps; ++r) {
#pragma unroll
    for (unsigned i = 0; i < 32; ++i)
      acc += buf[(threadIdx.x + i * 37U) & 1023U];
  }
  sink[ThreadIndex()] = HashVector(kHashSeed, acc);
}

template<typename Expected>
void Validate(const char* name, const std::uint32_t* sink, std::size_t count,
              Expected&& expected) {
  std::vector<std::uint32_t> actual(count);
  HIP_OK(hipGetLastError());
  HIP_OK(hipMemcpy(actual.data(), sink, count * sizeof(actual[0]),
                   hipMemcpyDeviceToHost));
  for (std::size_t i = 0; i < count; ++i) {
    if (actual[i] != expected(i)) {
      std::fprintf(stderr, "%s: thread %zu checksum %08x != %08x\n", name, i,
                   actual[i], expected(i));
      std::exit(1);
    }
  }
}

template<typename Launch>
double BenchSeconds(Launch&& launch, int iters = 3) {
  launch();
  HIP_OK(hipDeviceSynchronize());
  hipEvent_t a{};
  hipEvent_t b{};
  HIP_OK(hipEventCreate(&a));
  HIP_OK(hipEventCreate(&b));
  HIP_OK(hipEventRecord(a));
  for (int i = 0; i < iters; ++i)
    launch();
  HIP_OK(hipEventRecord(b));
  HIP_OK(hipEventSynchronize(b));
  float ms = 0.0F;
  HIP_OK(hipEventElapsedTime(&ms, a, b));
  HIP_OK(hipEventDestroy(a));
  HIP_OK(hipEventDestroy(b));
  return static_cast<double>(ms) / 1e3 / iters;
}

int main() {
  hipDeviceProp_t prop{};
  HIP_OK(hipGetDeviceProperties(&prop, 0));
  std::printf("device: %s  arch=%s  HIP multiprocessors=%d  clock=%.2f GHz\n",
              prop.name, prop.gcnArchName, prop.multiProcessorCount,
              prop.clockRate / 1e6);
  const int blocks = prop.multiProcessorCount * 16;
  const int stream_blocks = blocks * 2;
  const std::size_t compute_threads = blocks * kThreads;
  const std::size_t stream_threads = stream_blocks * kThreads;
  const double waves = static_cast<double>(compute_threads) / 32;
  std::uint32_t* sink{};
  HIP_OK(hipMalloc(&sink, stream_threads * sizeof(*sink)));

  const auto measure = [&](const char* name, auto launch, auto expected,
                           double work, const char* unit, std::size_t count) {
    HIP_OK(hipMemset(sink, 0, stream_threads * sizeof(*sink)));
    launch();
    Validate(name, sink, count, expected);
    const double seconds = BenchSeconds(launch);
    std::printf("  %-24s %10.2f %s  [checksums verified]\n", name,
                work / seconds, unit);
    return work / seconds;
  };

  std::uint32_t i8_hash = kHashSeed;
  std::uint32_t float_hash = kHashSeed;
  for (int i = 0; i < 32; ++i) {
    i8_hash = Fold(i8_hash, Word(i + 16 * 16 * kIters));
    float_hash =
        Fold(float_hash, Word(static_cast<float>(i + 8 * 16 * kIters)));
  }
  std::printf("\n-- sustained compute issue rates --\n");
  measure(
      "WMMA int8", [&] { PeakWmmaI8<<<blocks, kThreads>>>(sink, kIters); },
      [&](auto) { return i8_hash; }, waves * 4 * 16 * kIters * 8192 / 1e12,
      "TOPS", compute_threads);
  const auto floating = [&]<bool Bf16>(const char* name) {
    measure(
        name, [&] { PeakWmmaFloat<Bf16><<<blocks, kThreads>>>(sink, kIters); },
        [&](auto) { return float_hash; }, waves * 4 * 16 * kIters * 8192 / 1e12,
        "TFLOPS", compute_threads);
  };
  floating.template operator()<true>("WMMA bf16");
  floating.template operator()<false>("WMMA fp16");
  std::array<std::uint32_t, 2> fma_hash{};
  for (unsigned parity = 0; parity < 2; ++parity) {
    auto& hash = fma_hash[parity];
    hash = kHashSeed;
    for (int j = 0; j < 8; ++j) {
      float acc = 1.0F + j + 0.5F * parity;
      for (int i = 0; i < kIters * 32; ++i)
        acc = std::fma(acc, 1.0000001F, 0.0000001F);
      hash = Fold(hash, Word(acc));
    }
  }
  measure(
      "VALU fp32 fma", [&] { PeakFmaF32<<<blocks, kThreads>>>(sink, kIters); },
      [&](auto i) { return fma_hash[i & 1U]; },
      static_cast<double>(compute_threads) * 8 * 32 * kIters * 2 / 1e12,
      "TFLOPS", compute_threads);

  std::printf(
      "\n-- two dependent INT8 WMMA updates plus cvt/mul/fma triples --\n");
  const auto epilogue = [&]<int Triples>() {
    constexpr float dw = 0.00390625F;
    constexpr float dx = 0.0078125F;
    std::uint32_t hash = kHashSeed;
    for (int j = 0; j < 8; ++j)
      hash = Fold(hash, Word(j + kIters * 16 * 32));
    for (int j = 0; j < 8; ++j) {
      float acc = 0;
      if (j < Triples)
        for (int i = 1; i <= kIters * 16; ++i)
          acc = std::fma(dw * static_cast<float>(j + i * 32), dx, acc);
      hash = Fold(hash, Word(acc));
    }
    char name[32];
    std::snprintf(name, sizeof(name), "%d fp32 triples", Triples);
    measure(
        name,
        [&] {
          PeakWmmaI8WithEpilogue<Triples>
              <<<blocks, kThreads>>>(sink, kIters, dw, dx);
        },
        [&](auto) { return hash; }, waves * 16 * kIters * 2 * 8192 / 1e12,
        "TOPS", compute_threads);
  };
  epilogue.template operator()<0>();
  epilogue.template operator()<2>();
  epilogue.template operator()<4>();
  epilogue.template operator()<8>();

  std::printf("\n-- memory bandwidth --\n");
  std::array<std::uint32_t, kThreads> lds_hash{};
  for (unsigned lane = 0; lane < kThreads; ++lane) {
    std::array<std::uint32_t, 4> sum{};
    for (unsigned i = 0; i < 32; ++i) {
      const unsigned index = (lane + i * 37U) & 1023U;
      sum[0] += index;
      sum[1] += index ^ 0x1234U;
      sum[2] += index * 3U + 7U;
      sum[3] += ~index;
    }
    lds_hash[lane] = kHashSeed;
    for (auto value : sum)
      lds_hash[lane] = Fold(lds_hash[lane], value * kIters);
  }
  measure(
      "LDS read", [&] { PeakLdsRead<<<blocks, kThreads>>>(sink, kIters); },
      [&](auto i) { return lds_hash[i % kThreads]; },
      static_cast<double>(compute_threads) * 32 * kIters * 16 / 1e9, "GB/s",
      compute_threads);

  constexpr std::size_t bytes = 2ULL << 30;
  constexpr std::size_t n = bytes / sizeof(u32x4);
  u32x4* a{};
  u32x4* b{};
  HIP_OK(hipMalloc(&a, bytes));
  HIP_OK(hipMalloc(&b, bytes));
  HIP_OK(hipMemset(a, 1, bytes));
  const auto stream_hash = [&](std::size_t id, bool sequence) {
    const auto count = static_cast<std::uint32_t>(
        id < n ? 1 + (n - id - 1) / stream_threads : 0);
    auto hash = kHashSeed;
    for (unsigned j = 0; j < 4; ++j)
      hash = Fold(hash, count * (sequence ? j + 1 : 0x01010101U));
    return hash;
  };
  measure(
      "DRAM read", [&] { StreamRead<<<stream_blocks, kThreads>>>(a, sink, n); },
      [&](auto i) { return stream_hash(i, false); }, bytes / 1e9, "GB/s",
      stream_threads);
  // Validate the entire written/copied buffer through the checked read kernel.
  HIP_OK(hipMemset(b, 0, bytes));
  StreamWrite<<<stream_blocks, kThreads>>>(b, n);
  StreamRead<<<stream_blocks, kThreads>>>(b, sink, n);
  Validate("DRAM write", sink, stream_threads,
           [&](auto i) { return stream_hash(i, true); });
  const double write_s =
      BenchSeconds([&] { StreamWrite<<<stream_blocks, kThreads>>>(b, n); });
  std::printf("  %-24s %10.2f GB/s  [checksums verified]\n", "DRAM write",
              bytes / write_s / 1e9);
  HIP_OK(hipMemset(b, 0, bytes));
  StreamCopy<<<stream_blocks, kThreads>>>(a, b, n);
  StreamRead<<<stream_blocks, kThreads>>>(b, sink, n);
  Validate("DRAM copy", sink, stream_threads,
           [&](auto i) { return stream_hash(i, false); });
  const double copy_s =
      BenchSeconds([&] { StreamCopy<<<stream_blocks, kThreads>>>(a, b, n); });
  std::printf("  %-24s %10.2f GB/s  [checksums verified]\n",
              "DRAM copy (read+write)", 2.0 * bytes / copy_s / 1e9);
  HIP_OK(hipFree(a));
  HIP_OK(hipFree(b));
  HIP_OK(hipFree(sink));
}
