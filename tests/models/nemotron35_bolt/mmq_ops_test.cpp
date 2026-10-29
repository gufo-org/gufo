// SPDX-License-Identifier: MIT
// GPU check of Nemotron 3.5 Bolt's vendored quantized matrix kernels
// (nb_mmq.h) against an independent CPU oracle: the core ggml dequantizers
// and double accumulation over the unquantized activations.
//
// The kernels quantize activations to Q8_1 (one scale per 32 values, 8-bit
// codes), so outputs differ from the oracle by rounding noise. Each output's
// expected noise is computed from its own weights and activation scales:
//   var = sum_b (d_b^2 / 12) * sum_{k in b} w_k^2      (code rounding)
//       + sum_b (w_b . x_b)^2 * (2^-11)^2 / 3          (fp16 activation scale)
// with d_b = max|x| / 127 over each 32-value block. Every output must lie
// within kMaxZ standard deviations, the relative RMS error must stay at the
// Q8_1 level, and outputs must correlate with the oracle (so all-zero or
// misrouted outputs fail).

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <thread>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/nemotron35_bolt/kernels/rocm/mmq/nb_mmq.h"

namespace {

constexpr double kMaxZ = 7.0;
constexpr double kMaxRelRms = 0.012;
constexpr double kMinCorrelation = 0.999;

#define HIP_OK(expr)                                                           \
  do {                                                                         \
    const hipError_t err_ = (expr);                                            \
    if (err_ != hipSuccess) {                                                  \
      std::fprintf(stderr, "%s failed: %s\n", #expr, hipGetErrorString(err_)); \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct Rng {
  std::uint64_t state;
  std::uint64_t Next() {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
  }
  std::uint32_t Below(std::uint32_t n) {
    return static_cast<std::uint32_t>(Next() % n);
  }
  double Uniform() { return static_cast<double>(Next() >> 11U) * 0x1.0p-53; }
  float Normal() {
    const double u1 = std::max(Uniform(), 1e-300);
    const double u2 = Uniform();
    return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) *
                              std::cos(6.283185307179586 * u2));
  }
};

const char* TypeName(int type) {
  switch (type) {
    case NB_MMQ_Q8_0:
      return "Q8_0";
    case NB_MMQ_IQ2_XXS:
      return "IQ2_XXS";
    case NB_MMQ_NVFP4:
      return "NVFP4";
    default:
      return "?";
  }
}

std::size_t RowBytes(int type, int k) {
  switch (type) {
    case NB_MMQ_Q8_0:
      return static_cast<std::size_t>(k) / 32 * 34;
    case NB_MMQ_IQ2_XXS:
      return static_cast<std::size_t>(k) / 256 * 66;
    case NB_MMQ_NVFP4:
      return static_cast<std::size_t>(k) / 64 * 36;
    default:
      return 0;
  }
}

// Positive fp16 with a biased exponent in [e_lo, e_hi] and random mantissa.
std::uint16_t RandomHalf(Rng& rng, unsigned e_lo, unsigned e_hi) {
  const unsigned e = e_lo + rng.Below(e_hi - e_lo + 1);
  return static_cast<std::uint16_t>((e << 10U) | rng.Below(1024));
}

void PutU16(std::uint8_t* p, std::uint16_t v) {
  p[0] = static_cast<std::uint8_t>(v & 0xFFU);
  p[1] = static_cast<std::uint8_t>(v >> 8U);
}

// Random but valid blocks. Q8_0: scale ~1e-3..1.5e-2, codes in [-127, 127].
// IQ2_XXS: every grid index, sign index and group scale is valid, so the
// payload is uniform random bits. NVFP4: UE4M3 scales with exponents 4..9
// (never 0x7F or the huge range), uniform E2M1 nibbles.
std::vector<std::uint8_t> RandomWeights(int type, std::size_t rows, int k,
                                        Rng& rng) {
  std::vector<std::uint8_t> w(rows * RowBytes(type, k));
  std::uint8_t* p = w.data();
  std::uint8_t* const end = p + w.size();
  while (p < end) {
    switch (type) {
      case NB_MMQ_Q8_0:
        PutU16(p, RandomHalf(rng, 5, 8));
        for (int i = 0; i < 32; ++i) {
          p[2 + i] = static_cast<std::uint8_t>(
              static_cast<std::int8_t>(static_cast<int>(rng.Below(255)) - 127));
        }
        p += 34;
        break;
      case NB_MMQ_IQ2_XXS:
        PutU16(p, RandomHalf(rng, 4, 8));
        for (int i = 2; i < 66; i += 8) {
          const std::uint64_t bits = rng.Next();
          for (int b = 0; b < 8; ++b) {
            p[i + b] = static_cast<std::uint8_t>(bits >> (8U * b));
          }
        }
        p += 66;
        break;
      case NB_MMQ_NVFP4:
        for (int s = 0; s < 4; ++s) {
          p[s] = static_cast<std::uint8_t>(((4 + rng.Below(6)) << 3U) |
                                           rng.Below(8));
        }
        for (int i = 4; i < 36; i += 8) {
          const std::uint64_t bits = rng.Next();
          for (int b = 0; b < 8; ++b) {
            p[i + b] = static_cast<std::uint8_t>(bits >> (8U * b));
          }
        }
        p += 36;
        break;
      default:
        std::exit(1);
    }
  }
  return w;
}

void DequantizeRow(int type, const std::uint8_t* row, float* out, int k) {
  switch (type) {
    case NB_MMQ_Q8_0:
      gufo::quant::DequantizeQ8_0(row, out, static_cast<std::size_t>(k));
      break;
    case NB_MMQ_IQ2_XXS:
      gufo::quant::DequantizeIQ2_XXS(row, out, static_cast<std::size_t>(k));
      break;
    case NB_MMQ_NVFP4:
      gufo::quant::DequantizeNVFP4(row, out, static_cast<std::size_t>(k));
      break;
    default:
      std::exit(1);
  }
}

void ParallelFor(std::size_t n, const std::function<void(std::size_t)>& fn) {
  const std::size_t workers = std::max<std::size_t>(
      1, std::min<std::size_t>(std::thread::hardware_concurrency(), n));
  std::vector<std::thread> pool;
  pool.reserve(workers);
  for (std::size_t w = 0; w < workers; ++w) {
    pool.emplace_back([&, w] {
      for (std::size_t i = w; i < n; i += workers)
        fn(i);
    });
  }
  for (auto& t : pool)
    t.join();
}

// Per-32-block Q8_1 scale d_b = max|x| / 127 of every activation row.
std::vector<double> ActivationScales(const std::vector<float>& x, int k) {
  std::vector<double> d(x.size() / 32);
  for (std::size_t b = 0; b < d.size(); ++b) {
    float amax = 0.0F;
    for (int i = 0; i < 32; ++i)
      amax = std::max(amax, std::fabs(x[b * 32 + i]));
    d[b] = static_cast<double>(amax) / 127.0;
  }
  (void)k;
  return d;
}

struct Oracle {
  double value = 0.0;
  double var = 0.0;
};

// One output: weight row w (dequantized) against activation row x.
Oracle Dot(const float* w, const float* x, const double* d, int k) {
  constexpr double kHalfScale = 0x1.0p-22 / 3.0;  // (2^-11)^2 / 3
  Oracle o;
  double abs_sum = 0.0;
  for (int b = 0; b < k / 32; ++b) {
    double s = 0.0;
    double ww = 0.0;
    for (int i = b * 32; i < b * 32 + 32; ++i) {
      const double wi = w[i];
      s += wi * x[i];
      ww += wi * wi;
      abs_sum += std::fabs(wi * x[i]);
    }
    o.value += s;
    o.var += d[b] * d[b] / 12.0 * ww + s * s * kHalfScale;
  }
  // Float accumulation on the GPU.
  const double fp = 2e-6 * abs_sum;
  o.var += fp * fp;
  return o;
}

struct Stats {
  double max_abs = 0.0;
  double max_z = 0.0;
  double rel_rms = 0.0;
  double corr = 0.0;
  double rms_ref = 0.0;
  double rms_out = 0.0;
  bool finite = true;
};

Stats Compare(const std::vector<float>& out, const std::vector<Oracle>& ref) {
  Stats s;
  double se = 0.0;
  double sr = 0.0;
  double so = 0.0;
  double sro = 0.0;
  double mr = 0.0;
  double mo = 0.0;
  const double n = static_cast<double>(out.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (!std::isfinite(out[i]))
      s.finite = false;
    mr += ref[i].value;
    mo += out[i];
  }
  mr /= n;
  mo /= n;
  for (std::size_t i = 0; i < out.size(); ++i) {
    const double r = ref[i].value;
    const double o = out[i];
    const double e = o - r;
    s.max_abs = std::max(s.max_abs, std::fabs(e));
    s.max_z = std::max(s.max_z, std::fabs(e) / std::sqrt(ref[i].var));
    se += e * e;
    sr += (r - mr) * (r - mr);
    so += (o - mo) * (o - mo);
    sro += (r - mr) * (o - mo);
    s.rms_ref += r * r;
    s.rms_out += o * o;
  }
  s.rms_ref = std::sqrt(s.rms_ref / n);
  s.rms_out = std::sqrt(s.rms_out / n);
  s.rel_rms = std::sqrt(se / n) / s.rms_ref;
  s.corr = sro / std::sqrt(sr * so);
  return s;
}

int g_failures = 0;

void Report(const char* label, const Stats& s) {
  const bool ok = s.finite && s.max_z <= kMaxZ && s.rel_rms <= kMaxRelRms &&
                  s.corr >= kMinCorrelation && s.rms_out >= 0.5 * s.rms_ref;
  std::printf(
      "%-4s %-58s max|err| %.3e (%.4f rms) max z %5.2f rel rms %.5f "
      "corr %.7f\n",
      ok ? "ok" : "FAIL", label, s.max_abs, s.max_abs / s.rms_ref, s.max_z,
      s.rel_rms, s.corr);
  if (!ok)
    ++g_failures;
}

template<typename T>
T* Upload(const std::vector<T>& host) {
  T* dev = nullptr;
  HIP_OK(hipMalloc(&dev, host.size() * sizeof(T)));
  HIP_OK(hipMemcpy(dev, host.data(), host.size() * sizeof(T),
                   hipMemcpyHostToDevice));
  return dev;
}

std::vector<float> RandomActivations(std::size_t n, Rng& rng) {
  std::vector<float> x(n);
  for (float& v : x)
    v = rng.Normal();
  return x;
}

void DenseCase(int type, int m, int k, const std::vector<int>& ns,
               hipStream_t stream, Rng& rng) {
  const std::vector<std::uint8_t> w = RandomWeights(type, m, k, rng);
  const std::size_t row_bytes = RowBytes(type, k);
  void* w_dev = Upload(w);
  for (const int n : ns) {
    const std::vector<float> x =
        RandomActivations(static_cast<std::size_t>(n) * k, rng);
    const std::vector<double> d = ActivationScales(x, k);
    std::vector<Oracle> ref(static_cast<std::size_t>(m) * n);
    ParallelFor(static_cast<std::size_t>(m), [&](std::size_t row) {
      std::vector<float> wrow(static_cast<std::size_t>(k));
      DequantizeRow(type, w.data() + row * row_bytes, wrow.data(), k);
      for (int c = 0; c < n; ++c) {
        ref[static_cast<std::size_t>(c) * m + row] =
            Dot(wrow.data(), x.data() + static_cast<std::size_t>(c) * k,
                d.data() + static_cast<std::size_t>(c) * (k / 32), k);
      }
    });

    float* x_dev = Upload(x);
    float* out_dev = nullptr;
    HIP_OK(hipMalloc(&out_dev, ref.size() * sizeof(float)));
    HIP_OK(hipMemsetAsync(out_dev, 0xFF, ref.size() * sizeof(float), stream));
    const int rc = nb_mmq_dense(type, w_dev, x_dev, out_dev, m, n, k, stream);
    HIP_OK(hipStreamSynchronize(stream));
    std::vector<float> out(ref.size());
    HIP_OK(hipMemcpy(out.data(), out_dev, out.size() * sizeof(float),
                     hipMemcpyDeviceToHost));
    HIP_OK(hipFree(x_dev));
    HIP_OK(hipFree(out_dev));

    char label[128];
    std::snprintf(label, sizeof(label), "dense %-7s M=%5d K=%4d N=%3d (%s)",
                  TypeName(type), m, k, n, n <= 8 ? "vector" : "tiled");
    if (rc != 0) {
      std::printf("FAIL %s: nb_mmq_dense returned %d\n", label, rc);
      ++g_failures;
      continue;
    }
    Report(label, Compare(out, ref));
  }
  HIP_OK(hipFree(w_dev));
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::fprintf(stderr, "no HIP device\n");
    return 1;
  }
  if (nb_mmq_init(0) != 0) {
    std::fprintf(stderr, "nb_mmq_init failed\n");
    return 1;
  }
  hipStream_t stream = nullptr;
  HIP_OK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  Rng rng{0x6E656D6F74726F6EULL};

  // Dense: N <= 8 takes the MMVQ vector kernels, larger N the MMQ tiles.
  const std::vector<int> all_n = {1, 2, 5, 8, 33, 256};
  DenseCase(NB_MMQ_Q8_0, 4096, 4096, all_n, stream, rng);
  DenseCase(NB_MMQ_Q8_0, 18560, 4096, {1, 5, 33}, stream, rng);
  DenseCase(NB_MMQ_Q8_0, 4096, 8192, {1, 8, 33}, stream, rng);
  // M not a multiple of the 64-row tile exercises the bounds-checked MMQ.
  DenseCase(NB_MMQ_Q8_0, 1000, 2304, {2, 33, 256}, stream, rng);
  DenseCase(NB_MMQ_NVFP4, 5376, 4096, all_n, stream, rng);
  DenseCase(NB_MMQ_NVFP4, 4096, 5376, {1, 8, 256}, stream, rng);
  DenseCase(NB_MMQ_IQ2_XXS, 1024, 4096, all_n, stream, rng);
  DenseCase(NB_MMQ_IQ2_XXS, 256, 1024, {1, 33}, stream, rng);

  HIP_OK(hipStreamDestroy(stream));
  nb_mmq_cleanup();
  if (g_failures != 0) {
    std::printf("FAIL: %d case(s)\n", g_failures);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
