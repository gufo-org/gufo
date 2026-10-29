// Checks the trunk's HIP kernels (other than the quantized matrix products)
// against direct float/double CPU evaluations of the same formulas, including
// state carried across chunks.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/nemotron35_bolt/kernels/rocm/kernels.hpp"
#include "src/models/nemotron35_bolt/kernels/rocm/sample.hpp"

namespace k = gufo::models::nemotron35_bolt::rocm;

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    ++failures;
  }
}

void Hip(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "HIP error in %s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}

template<class T>
T* Device(const std::vector<T>& host) {
  T* d = nullptr;
  Hip(hipMalloc(&d, std::max<std::size_t>(1, host.size()) * sizeof(T)),
      "malloc");
  if (!host.empty())
    Hip(hipMemcpy(d, host.data(), host.size() * sizeof(T),
                  hipMemcpyHostToDevice),
        "upload");
  return d;
}

template<class T>
std::vector<T> Host(const T* d, std::size_t n) {
  std::vector<T> out(n);
  Hip(hipMemcpy(out.data(), d, n * sizeof(T), hipMemcpyDeviceToHost),
      "download");
  return out;
}

std::vector<float> Random(std::size_t n, std::mt19937& rng, float lo = -1.0F,
                          float hi = 1.0F) {
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (float& x : v)
    x = dist(rng);
  return v;
}

/// FP16 bits of a device half, or of a float rounded to nearest even.
std::uint16_t HalfBits(__half h) {
  std::uint16_t bits;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}
std::uint16_t HalfBits(float f) {
  const auto h = static_cast<_Float16>(f);
  std::uint16_t bits;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}

/// Max |a - b| relative to max |b|.
double RelErr(const std::vector<float>& a, const std::vector<float>& b) {
  double err = 0.0;
  double scale = 1e-12;
  for (std::size_t i = 0; i < a.size(); ++i) {
    err = std::max(err, std::abs(static_cast<double>(a[i]) - b[i]));
    scale = std::max(scale, std::abs(static_cast<double>(b[i])));
  }
  return err / scale;
}

float Sigmoid(float x) {
  return 1.0F / (1.0F + std::exp(-x));
}
float Silu(float x) {
  return x * Sigmoid(x);
}
float Softplus(float x) {
  return x > 20.0F ? x : std::log1p(std::exp(x));
}

void TestEmbedAndNorm(std::mt19937& rng) {
  constexpr int kHidden = 256;
  constexpr int kVocab = 7;
  std::vector<std::uint8_t> table(kVocab * (kHidden / 32) * 34);
  std::vector<float> expected_rows(kVocab * kHidden);
  std::uniform_int_distribution<int> q(-127, 127);
  for (int r = 0; r < kVocab; ++r) {
    for (int b = 0; b < kHidden / 32; ++b) {
      std::uint8_t* blk = table.data() + (r * (kHidden / 32) + b) * 34;
      const __half d = __float2half(0.01F * static_cast<float>(b + 1));
      std::memcpy(blk, &d, 2);
      for (int i = 0; i < 32; ++i) {
        const auto v = static_cast<std::int8_t>(q(rng));
        blk[2 + i] = static_cast<std::uint8_t>(v);
        expected_rows[r * kHidden + b * 32 + i] =
            __half2float(d) * static_cast<float>(v);
      }
    }
  }
  const std::vector<std::int32_t> tokens = {3, 0, 6, 3};
  auto* d_table = Device(table);
  auto* d_tokens = Device(tokens);
  float* d_x = Device(std::vector<float>(tokens.size() * kHidden));
  Hip(k::EmbedQ8_0(d_table, d_tokens, d_x, 4, kHidden, nullptr), "embed");
  const auto x = Host(d_x, tokens.size() * kHidden);
  std::vector<float> want;
  for (int t : tokens)
    want.insert(want.end(), expected_rows.begin() + t * kHidden,
                expected_rows.begin() + (t + 1) * kHidden);
  Check(RelErr(x, want) == 0.0, "Q8_0 embedding rows");

  const auto w = Random(kHidden, rng);
  float* d_w = Device(w);
  float* d_out = Device(std::vector<float>(x.size()));
  Hip(k::RmsNorm(d_x, d_w, d_out, 4, kHidden, 1e-5F, nullptr), "rmsnorm");
  const auto out = Host(d_out, x.size());
  std::vector<float> ref(x.size());
  for (int t = 0; t < 4; ++t) {
    double ss = 0;
    for (int i = 0; i < kHidden; ++i)
      ss += double(x[t * kHidden + i]) * x[t * kHidden + i];
    const float s = 1.0F / std::sqrt(float(ss / kHidden) + 1e-5F);
    for (int i = 0; i < kHidden; ++i)
      ref[t * kHidden + i] = x[t * kHidden + i] * s * w[i];
  }
  const double e = RelErr(out, ref);
  std::printf("rmsnorm rel err %.2e\n", e);
  Check(e < 1e-5, "RMSNorm");
  // The FP16 variant stores the same values rounded to nearest even.
  __half* d_out16 = Device(std::vector<__half>(x.size()));
  Hip(k::RmsNorm(d_x, d_w, d_out16, 4, kHidden, 1e-5F, nullptr), "rmsnorm16");
  const auto out16 = Host(d_out16, x.size());
  bool same16 = true;
  for (std::size_t i = 0; i < x.size(); ++i)
    same16 = same16 && HalfBits(out16[i]) == HalfBits(out[i]);
  Check(same16, "FP16 RMSNorm rounds the F32 result");
  Hip(hipFree(d_out16), "free");
  for (void* p : {static_cast<void*>(d_table), static_cast<void*>(d_tokens),
                  static_cast<void*>(d_x), static_cast<void*>(d_w),
                  static_cast<void*>(d_out)})
    Hip(hipFree(p), "free");
}

/// Conv, scan and gated norm over two chunks against a token-by-token CPU
/// evaluation that carries the same state.
void TestMamba(std::mt19937& rng) {
  constexpr int kHeads = 16;
  constexpr int kHeadDim = 64;
  constexpr int kGroups = 4;
  constexpr int kState = 128;
  constexpr int kKernel = 4;
  constexpr int kInner = kHeads * kHeadDim;
  constexpr int kChannels = kInner + 2 * kGroups * kState;
  constexpr int kStride = kInner + kChannels + kHeads;
  constexpr float kScale = 0.75F;
  constexpr float kEps = 1e-5F;
  const int chunks[] = {5, 1, 19};  // 19 spans several staged tiles.
  int total = 0;
  for (int c : chunks)
    total += c;

  const auto proj = Random(static_cast<std::size_t>(total) * kStride, rng);
  const auto conv_w = Random(kChannels * kKernel, rng, -0.5F, 0.5F);
  const auto conv_b = Random(kChannels, rng, -0.1F, 0.1F);
  const auto dt_b = Random(kHeads, rng, -2.0F, 0.0F);
  auto a = Random(kHeads, rng, -2.0F, -0.1F);
  const auto d = Random(kHeads, rng);
  const auto norm_w = Random(kInner, rng, 0.5F, 1.5F);

  // CPU reference, one token at a time.
  std::vector<float> conv_state((kKernel - 1) * kChannels, 0.0F);
  std::vector<float> ssm(static_cast<std::size_t>(kInner) * kState, 0.0F);
  std::vector<float> want(static_cast<std::size_t>(total) * kInner);
  for (int t = 0; t < total; ++t) {
    const float* row = proj.data() + static_cast<std::size_t>(t) * kStride;
    std::vector<float> xbc(kChannels);
    for (int c = 0; c < kChannels; ++c) {
      float acc = 0.0F;
      for (int j = 0; j + 1 < kKernel; ++j)
        acc += conv_state[j * kChannels + c] * conv_w[c * kKernel + j];
      acc += row[kInner + c] * kScale * conv_w[c * kKernel + kKernel - 1];
      xbc[c] = Silu(acc + conv_b[c]);
    }
    for (int j = 0; j + 2 < kKernel; ++j)
      std::copy_n(conv_state.begin() + (j + 1) * kChannels, kChannels,
                  conv_state.begin() + j * kChannels);
    for (int c = 0; c < kChannels; ++c)
      conv_state[(kKernel - 2) * kChannels + c] = row[kInner + c] * kScale;
    float* y = want.data() + static_cast<std::size_t>(t) * kInner;
    for (int h = 0; h < kHeads; ++h) {
      const float dt_sp =
          Softplus(row[kInner + kChannels + h] * kScale + dt_b[h]);
      const float dA = std::exp(dt_sp * a[h]);
      const int g = h / (kHeads / kGroups);
      for (int i = 0; i < kHeadDim; ++i) {
        const int ii = h * kHeadDim + i;
        double sum = 0;
        for (int n = 0; n < kState; ++n) {
          float& s = ssm[static_cast<std::size_t>(ii) * kState + n];
          s = s * dA + xbc[kInner + g * kState + n] * (xbc[ii] * dt_sp);
          sum += double(s) * xbc[kInner + kGroups * kState + g * kState + n];
        }
        y[ii] = float(sum) + xbc[ii] * d[h];
      }
    }
    const int gs = kInner / kGroups;
    for (int g = 0; g < kGroups; ++g) {
      double ss = 0;
      for (int j = 0; j < gs; ++j) {
        float& v = y[g * gs + j];
        v *= Silu(row[g * gs + j] * kScale);
        ss += double(v) * v;
      }
      const float s = 1.0F / std::sqrt(float(ss / gs) + kEps);
      for (int j = 0; j < gs; ++j)
        y[g * gs + j] *= s * norm_w[g * gs + j];
    }
  }

  auto* d_proj = Device(proj);
  auto* d_conv_w = Device(conv_w);
  auto* d_conv_b = Device(conv_b);
  auto* d_dt_b = Device(dt_b);
  auto* d_a = Device(a);
  auto* d_d = Device(d);
  auto* d_norm = Device(norm_w);
  auto* d_conv_state =
      Device(std::vector<float>((kKernel - 1) * kChannels, 0.0F));
  auto* d_ssm =
      Device(std::vector<float>(static_cast<std::size_t>(kInner) * kState));
  auto* d_xbc =
      Device(std::vector<float>(static_cast<std::size_t>(total) * kChannels));
  auto* d_y =
      Device(std::vector<float>(static_cast<std::size_t>(total) * kInner));
  int offset = 0;
  for (int n : chunks) {
    const float* p = d_proj + static_cast<std::size_t>(offset) * kStride;
    float* xbc = d_xbc + static_cast<std::size_t>(offset) * kChannels;
    float* y = d_y + static_cast<std::size_t>(offset) * kInner;
    Hip(k::MambaConv(p, kStride, kInner, kScale, d_conv_w, d_conv_b,
                     d_conv_state, xbc, n, kChannels, kKernel, nullptr),
        "conv");
    Hip(k::MambaConvState(p, kStride, kInner, kScale, d_conv_state, n,
                          kChannels, kKernel, nullptr),
        "conv state");
    k::MambaScanArgs args{
        xbc,    kChannels, p,      kStride,  kInner + kChannels,
        kScale, d_dt_b,    d_a,    d_d,      d_ssm,
        y,      n,         kHeads, kHeadDim, kGroups};
    Hip(k::MambaScan(args, nullptr), "scan");
    // The FP16 variant on a copy stores the F32 result rounded.
    const std::size_t len = static_cast<std::size_t>(n) * kInner;
    float* y_copy = Device(std::vector<float>(len));
    __half* y16 = Device(std::vector<__half>(len));
    Hip(hipMemcpy(y_copy, y, len * sizeof(float), hipMemcpyDeviceToDevice),
        "copy");
    Hip(k::MambaGateNorm(y_copy, p, kStride, kScale, d_norm, y16, n, kInner,
                         kGroups, kEps, nullptr),
        "gate norm16");
    Hip(k::MambaGateNorm(y, p, kStride, kScale, d_norm, n, kInner, kGroups,
                         kEps, nullptr),
        "gate norm");
    const auto f32 = Host(y, len);
    const auto f16 = Host(y16, len);
    bool same16 = true;
    for (std::size_t i = 0; i < len; ++i)
      same16 = same16 && HalfBits(f16[i]) == HalfBits(f32[i]);
    Check(same16, "FP16 gated norm rounds the F32 result");
    Hip(hipFree(y_copy), "free");
    Hip(hipFree(y16), "free");
    offset += n;
  }
  const auto got = Host(d_y, want.size());
  const double e = RelErr(got, want);
  std::printf("mamba conv/scan/gate-norm rel err %.2e\n", e);
  Check(e < 1e-4, "Mamba-2 chunked conv, scan and gated group norm");
  // The device keeps the state state-major per head: [head][n][row].
  const auto device_state = Host(d_ssm, ssm.size());
  std::vector<float> got_state(ssm.size());
  for (int h = 0; h < kHeads; ++h)
    for (int n = 0; n < kState; ++n)
      for (int i = 0; i < kHeadDim; ++i)
        got_state[(static_cast<std::size_t>(h) * kHeadDim + i) * kState + n] =
            device_state[(static_cast<std::size_t>(h) * kState + n) * kHeadDim +
                         i];
  const double es = RelErr(got_state, ssm);
  std::printf("mamba carried state rel err %.2e\n", es);
  Check(es < 1e-4, "Mamba-2 state after the last chunk");
  for (void* p : {static_cast<void*>(d_proj), static_cast<void*>(d_conv_w),
                  static_cast<void*>(d_conv_b), static_cast<void*>(d_dt_b),
                  static_cast<void*>(d_a), static_cast<void*>(d_d),
                  static_cast<void*>(d_norm), static_cast<void*>(d_conv_state),
                  static_cast<void*>(d_ssm), static_cast<void*>(d_xbc),
                  static_cast<void*>(d_y)})
    Hip(hipFree(p), "free");
}

/// Two chunks through the KV cache; the second chunk spans several key tiles.
void TestAttention(std::mt19937& rng) {
  // Bolt's grouping: 16 query heads per KV head, one WMMA row tile.
  constexpr int kHeads = 32;
  constexpr int kKvHeads = 2;
  constexpr int kHd = 128;
  constexpr int kKv = kKvHeads * kHd;
  const int chunks[] = {150, 3, 140};
  int total = 0;
  for (int c : chunks)
    total += c;
  const auto q = Random(static_cast<std::size_t>(total) * kHeads * kHd, rng);
  const auto kk = Random(static_cast<std::size_t>(total) * kKv, rng);
  const auto vv = Random(static_cast<std::size_t>(total) * kKv, rng);
  const float scale = 1.0F / std::sqrt(float(kHd));

  // CPU reference on half-rounded K/V, as stored in the cache.
  std::vector<float> kh(kk.size());
  std::vector<float> vh(vv.size());
  for (std::size_t i = 0; i < kk.size(); ++i) {
    kh[i] = __half2float(__float2half(kk[i]));
    vh[i] = __half2float(__float2half(vv[i]));
  }
  std::vector<float> want(q.size());
  for (int p = 0; p < total; ++p) {
    for (int h = 0; h < kHeads; ++h) {
      const int kvh = h / (kHeads / kKvHeads);
      std::vector<double> s(p + 1);
      double m = -INFINITY;
      for (int j = 0; j <= p; ++j) {
        double dot = 0;
        for (int dd = 0; dd < kHd; ++dd)
          dot += double(q[(std::size_t(p) * kHeads + h) * kHd + dd] * scale) *
                 kh[std::size_t(j) * kKv + kvh * kHd + dd];
        s[j] = dot;
        m = std::max(m, dot);
      }
      double denom = 0;
      for (double& x : s) {
        x = std::exp(x - m);
        denom += x;
      }
      for (int dd = 0; dd < kHd; ++dd) {
        double acc = 0;
        for (int j = 0; j <= p; ++j)
          acc += s[j] * vh[std::size_t(j) * kKv + kvh * kHd + dd];
        want[(std::size_t(p) * kHeads + h) * kHd + dd] = float(acc / denom);
      }
    }
  }

  // The chunked kernel's arithmetic: scores in the exp2 domain from the
  // scaled query rounded to FP16, keys taken in 64-key tiles from position
  // 0 with a running maximum, probabilities rounded to FP16 for P V while
  // the normalizer sums them unrounded. Only accumulation order and the
  // exp2 approximation remain between it and the kernel; the latter can
  // move a probability across an FP16 rounding boundary, so the two agree to
  // about 1e-4, not bit for bit.
  constexpr double kLog2e = 1.4426950408889634;
  constexpr int kTile = 64;
  std::vector<float> want_kernel(q.size());
  for (int p = 0; p < total; ++p) {
    for (int h = 0; h < kHeads; ++h) {
      const int kvh = h / (kHeads / kKvHeads);
      std::vector<float> qr(kHd);
      for (int dd = 0; dd < kHd; ++dd)
        qr[dd] = __half2float(
            __float2half(q[(std::size_t(p) * kHeads + h) * kHd + dd] *
                         float(scale * kLog2e)));
      double m = -INFINITY;
      double l = 0;
      std::vector<double> o(kHd, 0.0);
      for (int k0 = 0; k0 <= p; k0 += kTile) {
        const int k1 = std::min(p + 1, k0 + kTile);
        std::vector<double> sc(k1 - k0);
        double tile_max = -INFINITY;
        for (int j = k0; j < k1; ++j) {
          double dot = 0;
          for (int dd = 0; dd < kHd; ++dd)
            dot += double(qr[dd]) * kh[std::size_t(j) * kKv + kvh * kHd + dd];
          sc[j - k0] = dot;
          tile_max = std::max(tile_max, dot);
        }
        const double new_m = std::max(m, tile_max);
        const double corr = std::exp2(m - new_m);
        l *= corr;
        for (double& x : o)
          x *= corr;
        for (int j = k0; j < k1; ++j) {
          const double pj = std::exp2(sc[j - k0] - new_m);
          l += pj;
          const double ph = __half2float(__float2half(float(pj)));
          for (int dd = 0; dd < kHd; ++dd)
            o[dd] += ph * vh[std::size_t(j) * kKv + kvh * kHd + dd];
        }
        m = new_m;
      }
      for (int dd = 0; dd < kHd; ++dd)
        want_kernel[(std::size_t(p) * kHeads + h) * kHd + dd] =
            float(o[dd] / l);
    }
  }

  auto* d_q = Device(q);
  auto* d_k = Device(kk);
  auto* d_v = Device(vv);
  __half* kc = nullptr;
  __half* vc = nullptr;
  Hip(hipMalloc(&kc, std::size_t(total) * kKv * sizeof(__half)), "malloc");
  Hip(hipMalloc(&vc, std::size_t(total) * kKv * sizeof(__half)), "malloc");
  auto* d_out = Device(std::vector<float>(q.size()));
  int* d_pos = Device(std::vector<int>(1, 0));
  int pos = 0;
  for (int n : chunks) {
    Hip(hipMemcpy(d_pos, &pos, sizeof(int), hipMemcpyHostToDevice), "pos");
    Hip(k::KvAppend(d_k + std::size_t(pos) * kKv, d_v + std::size_t(pos) * kKv,
                    kc, vc, d_pos, n, kKv, nullptr),
        "kv append");
    Hip(k::Attention(d_q + std::size_t(pos) * kHeads * kHd, kc, vc,
                     d_out + std::size_t(pos) * kHeads * kHd, d_pos, n, kHeads,
                     kKvHeads, scale, nullptr),
        "attention");
    pos += n;
  }
  const auto got = Host(d_out, q.size());
  const double e = RelErr(got, want_kernel);
  const double e_exact = RelErr(got, want);
  std::printf("attention rel err %.2e (exact softmax %.2e)\n", e, e_exact);
  Check(e < 3e-4, "chunked causal GQA attention over the half KV cache");
  Check(e_exact < 1e-3, "FP16 query and probabilities stay near exact");

  // Split short chunks (decode, verification) ending at several depths,
  // including fewer keys than splits. The cache already holds every key, so
  // a chunk ending at p sees exactly keys 0..p of each of its tokens.
  float* d_partial =
      Device(std::vector<float>(std::size_t(k::kAttentionShortTokens) * kHeads *
                                k::kAttentionSplits * (kHd + 2)));
  float* d_short = Device(
      std::vector<float>(std::size_t(k::kAttentionShortTokens) * kHeads * kHd));
  double worst = 0;
  for (const auto [last, n] :
       {std::pair{0, 1}, std::pair{5, 1}, std::pair{31, 3}, std::pair{64, 8},
        std::pair{200, 5}, std::pair{total - 1, 1}, std::pair{total - 1, 8}}) {
    const int first = last + 1 - n;
    Hip(hipMemcpy(d_pos, &first, sizeof(int), hipMemcpyHostToDevice), "pos");
    Hip(k::AttentionShort(d_q + std::size_t(first) * kHeads * kHd, kc, vc,
                          d_short, d_pos, n, kHeads, kKvHeads, scale, d_partial,
                          nullptr),
        "short attention");
    const auto got_short = Host(d_short, std::size_t(n) * kHeads * kHd);
    const std::vector<float> ref(
        want.begin() + std::ptrdiff_t(first) * kHeads * kHd,
        want.begin() + std::ptrdiff_t(last + 1) * kHeads * kHd);
    worst = std::max(worst, RelErr(got_short, ref));
  }
  std::printf("split short attention rel err %.2e\n", worst);
  Check(worst < 1e-3, "split short-chunk attention");

  // A verified block's rows equal one-token calls at their positions, bit
  // for bit, though the block's later keys are already in the cache: blocks
  // within a tile, across a tile boundary and at the end of the cache.
  std::size_t differ = 0;
  for (const auto [first, n] :
       {std::pair{36, 2}, std::pair{36, 8}, std::pair{60, 8}, std::pair{100, 8},
        std::pair{200, 4}, std::pair{total - 8, 8}}) {
    Hip(hipMemcpy(d_pos, &first, sizeof(int), hipMemcpyHostToDevice), "pos");
    Hip(k::AttentionShort(d_q + std::size_t(first) * kHeads * kHd, kc, vc,
                          d_short, d_pos, n, kHeads, kKvHeads, scale, d_partial,
                          nullptr),
        "block attention");
    const auto block = Host(d_short, std::size_t(n) * kHeads * kHd);
    for (int i = 0; i < n; ++i) {
      const int at = first + i;
      Hip(hipMemcpy(d_pos, &at, sizeof(int), hipMemcpyHostToDevice), "pos");
      Hip(k::AttentionShort(d_q + std::size_t(at) * kHeads * kHd, kc, vc,
                            d_short, d_pos, 1, kHeads, kKvHeads, scale,
                            d_partial, nullptr),
          "one-token attention");
      const auto one = Host(d_short, std::size_t(kHeads) * kHd);
      for (std::size_t j = 0; j < one.size(); ++j)
        differ += one[j] != block[std::size_t(i) * kHeads * kHd + j];
    }
  }
  std::printf("verified blocks vs one-token attention: %zu values differ\n",
              differ);
  Check(differ == 0, "a verified block's rows equal one-token attention");
  Hip(hipFree(d_partial), "free");
  Hip(hipFree(d_short), "free");
  for (void* p :
       {static_cast<void*>(d_q), static_cast<void*>(d_k),
        static_cast<void*>(d_v), static_cast<void*>(kc), static_cast<void*>(vc),
        static_cast<void*>(d_out), static_cast<void*>(d_pos)})
    Hip(hipFree(p), "free");
}

/// Verified blocks that cross 4,096 and 8,192 keys, where the number of
/// tiles per split would change within the block: each row equals a
/// one-token pass at its position, bit for bit.
void TestAttentionSplitBoundary(std::mt19937& rng) {
  constexpr int kHeads = 32;
  constexpr int kKvHeads = 2;
  constexpr int kHd = 128;
  constexpr int kKv = kKvHeads * kHd;
  constexpr int kKeys = 8200;
  constexpr int kBlock = 8;
  const float scale = 1.0F / std::sqrt(float(kHd));
  std::vector<__half> kh(std::size_t(kKeys) * kKv);
  std::vector<__half> vh(kh.size());
  const auto kf = Random(kh.size(), rng);
  const auto vf = Random(vh.size(), rng);
  for (std::size_t i = 0; i < kh.size(); ++i) {
    kh[i] = __float2half(kf[i]);
    vh[i] = __float2half(vf[i]);
  }
  auto* kc = Device(kh);
  auto* vc = Device(vh);
  auto* d_q = Device(Random(std::size_t(kBlock) * kHeads * kHd, rng));
  int* d_pos = Device(std::vector<int>(1, 0));
  float* d_partial =
      Device(std::vector<float>(std::size_t(k::kAttentionShortTokens) * kHeads *
                                k::kAttentionSplits * (kHd + 2)));
  float* d_out = Device(std::vector<float>(std::size_t(kBlock) * kHeads * kHd));
  std::size_t differ = 0;
  for (const int first : {4092, 8188}) {
    Hip(hipMemcpy(d_pos, &first, sizeof(int), hipMemcpyHostToDevice), "pos");
    Hip(k::AttentionShort(d_q, kc, vc, d_out, d_pos, kBlock, kHeads, kKvHeads,
                          scale, d_partial, nullptr),
        "block attention");
    const auto block = Host(d_out, std::size_t(kBlock) * kHeads * kHd);
    for (int i = 0; i < kBlock; ++i) {
      const int at = first + i;
      Hip(hipMemcpy(d_pos, &at, sizeof(int), hipMemcpyHostToDevice), "pos");
      Hip(k::AttentionShort(d_q + std::size_t(i) * kHeads * kHd, kc, vc, d_out,
                            d_pos, 1, kHeads, kKvHeads, scale, d_partial,
                            nullptr),
          "one-token attention");
      const auto one = Host(d_out, std::size_t(kHeads) * kHd);
      for (std::size_t j = 0; j < one.size(); ++j)
        differ += one[j] != block[std::size_t(i) * kHeads * kHd + j];
    }
  }
  std::printf("verified blocks across split boundaries: %zu values differ\n",
              differ);
  Check(differ == 0,
        "verified blocks across split boundaries equal one-token attention");
  for (void* p : {static_cast<void*>(kc), static_cast<void*>(vc),
                  static_cast<void*>(d_q), static_cast<void*>(d_pos),
                  static_cast<void*>(d_partial), static_cast<void*>(d_out)})
    Hip(hipFree(p), "free");
}

void TestRouterLogits(std::mt19937& rng) {
  constexpr int kCols = 4096;
  for (int rows : {240, 496}) {
    for (int tokens : {1, 3, 8}) {
      const auto w = Random(static_cast<std::size_t>(rows) * kCols, rng);
      const auto x = Random(static_cast<std::size_t>(tokens) * kCols, rng);
      std::vector<float> want(static_cast<std::size_t>(tokens) * rows);
      for (int t = 0; t < tokens; ++t)
        for (int r = 0; r < rows; ++r) {
          double acc = 0;
          for (int c = 0; c < kCols; ++c)
            acc += double(w[std::size_t(r) * kCols + c]) *
                   x[std::size_t(t) * kCols + c];
          want[std::size_t(t) * rows + r] = float(acc);
        }
      auto* d_w = Device(w);
      auto* d_x = Device(x);
      auto* d_out = Device(std::vector<float>(want.size()));
      Hip(k::RouterLogits(d_w, d_x, d_out, rows, kCols, tokens, nullptr),
          "router logits");
      const double e = RelErr(Host(d_out, want.size()), want);
      Check(e < 1e-5, "router logits " + std::to_string(rows) + "x" +
                          std::to_string(tokens));
      for (void* p : {static_cast<void*>(d_w), static_cast<void*>(d_x),
                      static_cast<void*>(d_out)})
        Hip(hipFree(p), "free");
    }
  }
}

void TestRouting(std::mt19937& rng) {
  constexpr int kTokens = 5;
  constexpr int kTop = 9;
  constexpr int kWidth = 96;
  for (int experts : {240, 496}) {
    auto logits = Random(std::size_t(kTokens) * experts, rng, -4.0F, 4.0F);
    const auto bias = Random(experts, rng, -0.05F, 0.05F);
    // A tie in the selection score must choose the lower expert.
    logits[7] = logits[3];
    std::vector<std::int32_t> want_ids;
    std::vector<float> want_w;
    for (int t = 0; t < kTokens; ++t) {
      std::vector<int> order(experts);
      std::iota(order.begin(), order.end(), 0);
      auto sel = [&](int e) {
        return Sigmoid(logits[std::size_t(t) * experts + e]) + bias[e];
      };
      std::stable_sort(order.begin(), order.end(),
                       [&](int x, int y) { return sel(x) > sel(y); });
      float sum = 0;
      for (int r = 0; r < kTop; ++r)
        sum += Sigmoid(logits[std::size_t(t) * experts + order[r]]);
      for (int r = 0; r < kTop; ++r) {
        want_ids.push_back(order[r]);
        want_w.push_back(Sigmoid(logits[std::size_t(t) * experts + order[r]]) /
                         std::max(sum, 6.103515625e-5F) * 5.0F);
      }
    }
    auto* d_logits = Device(logits);
    auto* d_bias = Device(bias);
    auto* d_ids = Device(std::vector<std::int32_t>(kTokens * kTop));
    auto* d_w = Device(std::vector<float>(kTokens * kTop));
    Hip(k::RouterTopK(d_logits, d_bias, experts, kTokens, kTop, true, 5.0F,
                      d_ids, d_w, nullptr),
        "router");
    const auto ids = Host(d_ids, want_ids.size());
    const auto w = Host(d_w, want_w.size());
    Check(ids == want_ids, "router top-9 ids for " + std::to_string(experts));
    Check(RelErr(w, want_w) < 1e-6, "router weights");

    const auto slots = Random(std::size_t(kTokens) * kTop * kWidth, rng);
    auto* d_slots = Device(slots);
    auto* d_sum = Device(std::vector<float>(kTokens * kWidth));
    Hip(k::WeightedSlotSum(d_slots, d_w, d_sum, kTokens, kTop, kWidth, nullptr),
        "slot sum");
    std::vector<float> want_sum(kTokens * kWidth, 0.0F);
    for (int t = 0; t < kTokens; ++t)
      for (int s = 0; s < kTop; ++s)
        for (int c = 0; c < kWidth; ++c)
          want_sum[t * kWidth + c] +=
              w[t * kTop + s] * slots[(std::size_t(t) * kTop + s) * kWidth + c];
    Check(RelErr(Host(d_sum, want_sum.size()), want_sum) < 1e-6,
          "weighted expert sum");
    for (void* p : {static_cast<void*>(d_logits), static_cast<void*>(d_bias),
                    static_cast<void*>(d_ids), static_cast<void*>(d_w),
                    static_cast<void*>(d_slots), static_cast<void*>(d_sum)})
      Hip(hipFree(p), "free");
  }
}

/// Device logit heads against the host: the maximum, an F64 softmax sum in
/// sequential order, every token at least the head's smallest value, the
/// coverage the filters need, and the core sampler drawing the same token
/// from head and row.
void TestLogitHeads(std::mt19937& rng) {
  std::normal_distribution<float> normal(0.0F, 1.0F);
  for (const int vocab : {131072, 5000}) {
    // Rows: sharp, moderate, top ties, flat, no finite value, and moderate
    // rows with NaN and infinities.
    constexpr int kRows = 8;
    std::vector<float> logits(static_cast<std::size_t>(kRows) * vocab);
    for (int r = 0; r < kRows; ++r) {
      float* row = logits.data() + static_cast<std::size_t>(r) * vocab;
      const float spread = r == 0 ? 3.0F : r == 3 ? 0.05F : 2.0F;
      for (int i = 0; i < vocab; ++i)
        row[i] = spread * normal(rng);
      if (r == 0)
        row[vocab / 3] = 25.0F;
      if (r == 2)
        for (int i = 0; i < 6; ++i)
          row[(i * 7919) % vocab] = 9.5F;
      if (r == 4)
        std::fill(row, row + vocab, -INFINITY);
      if (r >= 5) {
        row[(r * 104729) % vocab] = NAN;
        row[(r * 1299709) % vocab] = INFINITY;
        row[(r * 15485863) % vocab] = -INFINITY;
        row[(r * 17) % vocab] = 7.0F;
      }
    }
    float* d_logits = Device(logits);
    void* scratch = nullptr;
    Hip(hipMalloc(&scratch, k::LogitHeadScratchBytes(kRows, vocab)), "malloc");
    std::vector<k::LogitHeadRecord> no_records(kRows);
    std::vector<k::LogitHeadCandidate> no_candidates(
        static_cast<std::size_t>(kRows) * k::kLogitHeadCandidates);
    auto* d_records = Device(no_records);
    auto* d_candidates = Device(no_candidates);
    const std::vector<gufo::sampling::SamplingConfig> configs{
        {.temperature = 1.0F, .top_p = 0.95F},
        {.temperature = 0.7F, .top_p = 0.9F, .min_keep = 3},
        {.temperature = 1.0F, .top_k = 40, .top_p = 0.95F},
        {.temperature = 1.3F, .top_k = 50, .min_p = 0.05F},
    };
    for (const auto& config : configs) {
      const auto filter = *config.logit_head_filter();
      const k::LogitHeadParams params{
          filter.temperature, static_cast<int>(filter.keep), filter.top_p,
          static_cast<int>(filter.minimum)};
      Hip(k::LogitHeads(d_logits, kRows, vocab, params, scratch, d_records,
                        d_candidates, nullptr),
          "LogitHeads");
      const auto records = Host(d_records, kRows);
      const auto candidates =
          Host(d_candidates,
               static_cast<std::size_t>(kRows) * k::kLogitHeadCandidates);
      const std::string where = "logit head (vocab " + std::to_string(vocab) +
                                ", T " + std::to_string(filter.temperature) +
                                ", keep " + std::to_string(filter.keep) +
                                ") row ";
      for (int r = 0; r < kRows; ++r) {
        const float* row = logits.data() + static_cast<std::size_t>(r) * vocab;
        const std::string at = where + std::to_string(r);
        float maximum = -INFINITY;
        std::vector<float> finite;
        for (int i = 0; i < vocab; ++i)
          if (std::isfinite(row[i])) {
            maximum = std::max(maximum, row[i]);
            finite.push_back(row[i]);
          }
        double sum = 0.0;
        for (int i = 0; i < vocab; ++i)
          if (std::isfinite(row[i]))
            sum += std::exp((static_cast<double>(row[i]) - maximum) /
                            filter.temperature);
        // The tokens the filters need: top-k, or the top-p nucleus.
        std::sort(finite.begin(), finite.end(), std::greater<>());
        std::size_t needed = filter.keep;
        if (filter.keep == 0) {
          double cumulative = 0.0;
          needed = 0;
          while (needed < finite.size() &&
                 (needed < filter.minimum || cumulative < filter.top_p * sum))
            cumulative +=
                std::exp((static_cast<double>(finite[needed++]) - maximum) /
                         filter.temperature);
        }
        if (finite.empty() || needed > k::kLogitHeadCandidates)
          Check(records[r].count == 0, at + ": no head where none fits");
        else if (needed <= 256 && r != 3)
          // The flat row crowds more than a head into the bin of its cut.
          Check(records[r].count > 0, at + ": a head where one fits");
        if (records[r].count == 0)
          continue;
        Check(records[r].maximum == maximum, at + ": maximum");
        Check(std::abs(records[r].softmax_sum - sum) <= 1e-12 * sum,
              at + ": softmax sum");
        gufo::sampling::LogitHead head{.maximum = records[r].maximum,
                                       .temperature = filter.temperature,
                                       .softmax_sum = records[r].softmax_sum};
        std::vector<char> seen(vocab, 0);
        float smallest = INFINITY;
        bool valid = true;
        for (int c = 0; c < records[r].count; ++c) {
          const auto& cand =
              candidates[static_cast<std::size_t>(r) * k::kLogitHeadCandidates +
                         c];
          valid = valid && cand.token < static_cast<std::uint32_t>(vocab) &&
                  !seen[cand.token] && cand.value == row[cand.token];
          if (cand.token < static_cast<std::uint32_t>(vocab))
            seen[cand.token] = 1;
          smallest = std::min(smallest, cand.value);
          head.candidates.push_back({cand.token, cand.value});
        }
        Check(valid, at + ": unique candidates with their logits");
        int above = 0;
        for (int i = 0; i < vocab; ++i)
          above += std::isfinite(row[i]) && row[i] >= smallest;
        Check(above == records[r].count,
              at + ": every token above the head's floor");
        std::vector<float> sorted;
        for (const auto& c : head.candidates)
          sorted.push_back(c.value);
        std::sort(sorted.begin(), sorted.end(), std::greater<>());
        double mass = 0.0;
        for (const float v : sorted)
          mass +=
              std::exp((static_cast<double>(v) - maximum) / filter.temperature);
        Check(filter.keep > 0 ? sorted.size() >= filter.keep
                              : sorted.size() >= filter.minimum &&
                                    mass >= filter.top_p * sum,
              at + ": covers the filters");
        const std::span<const float> full(row, vocab);
        bool same = true;
        for (std::int64_t seed = 0; seed < 32; ++seed) {
          auto seeded = config;
          seeded.seed = seed;
          gufo::sampling::SamplerState a(seeded), b(seeded);
          same = same && a.Sample(full) == b.Sample(full, head) &&
                 a.rng_state() == b.rng_state();
        }
        Check(same, at + ": the head draws the row's token");
      }
    }
    Hip(hipFree(d_logits), "free");
    Hip(hipFree(scratch), "free");
    Hip(hipFree(d_records), "free");
    Hip(hipFree(d_candidates), "free");
  }
}

}  // namespace

/// One token of each of several sequences with their own states: the row
/// kernels match one-token calls of the sequence kernels bit for bit.
void TestBatchRows(std::mt19937& rng) {
  constexpr int kRows = 3;
  {
    constexpr int kHeads = 16;
    constexpr int kHeadDim = 64;
    constexpr int kGroups = 4;
    constexpr int kState = 128;
    constexpr int kKernel = 4;
    constexpr int kInner = kHeads * kHeadDim;
    constexpr int kChannels = kInner + 2 * kGroups * kState;
    constexpr int kStride = kInner + kChannels + kHeads;
    constexpr float kScale = 0.75F;
    constexpr std::size_t kConv = (kKernel - 1) * kChannels;
    constexpr std::size_t kSsm = static_cast<std::size_t>(kInner) * kState;
    const auto proj = Random(kRows * kStride, rng);
    const auto conv_w = Random(kChannels * kKernel, rng, -0.5F, 0.5F);
    const auto conv_b = Random(kChannels, rng, -0.1F, 0.1F);
    const auto dt_b = Random(kHeads, rng, -2.0F, 0.0F);
    const auto a = Random(kHeads, rng, -2.0F, -0.1F);
    const auto d = Random(kHeads, rng);
    auto* d_proj = Device(proj);
    auto* d_conv_w = Device(conv_w);
    auto* d_conv_b = Device(conv_b);
    auto* d_dt_b = Device(dt_b);
    auto* d_a = Device(a);
    auto* d_d = Device(d);
    std::vector<float*> seq_conv, seq_ssm, row_conv, row_ssm;
    for (int r = 0; r < kRows; ++r) {
      const auto conv = Random(kConv, rng);
      const auto ssm = Random(kSsm, rng, -0.5F, 0.5F);
      seq_conv.push_back(Device(conv));
      seq_ssm.push_back(Device(ssm));
      row_conv.push_back(Device(conv));
      row_ssm.push_back(Device(ssm));
    }
    auto* seq_xbc = Device(std::vector<float>(kRows * kChannels));
    auto* seq_y = Device(std::vector<float>(kRows * kInner));
    for (int r = 0; r < kRows; ++r) {
      const float* p = d_proj + static_cast<std::size_t>(r) * kStride;
      float* xbc = seq_xbc + static_cast<std::size_t>(r) * kChannels;
      Hip(k::MambaConv(p, kStride, kInner, kScale, d_conv_w, d_conv_b,
                       seq_conv[r], xbc, 1, kChannels, kKernel, nullptr),
          "conv");
      Hip(k::MambaConvState(p, kStride, kInner, kScale, seq_conv[r], 1,
                            kChannels, kKernel, nullptr),
          "conv state");
      k::MambaScanArgs args{xbc,
                            kChannels,
                            p,
                            kStride,
                            kInner + kChannels,
                            kScale,
                            d_dt_b,
                            d_a,
                            d_d,
                            seq_ssm[r],
                            seq_y + static_cast<std::size_t>(r) * kInner,
                            1,
                            kHeads,
                            kHeadDim,
                            kGroups};
      Hip(k::MambaScan(args, nullptr), "scan");
    }
    auto* d_conv_rows = Device(row_conv);
    auto* d_ssm_rows = Device(row_ssm);
    auto* row_xbc = Device(std::vector<float>(kRows * kChannels));
    auto* row_y = Device(std::vector<float>(kRows * kInner));
    Hip(k::MambaConvRows(d_proj, kStride, kInner, kScale, d_conv_w, d_conv_b,
                         d_conv_rows, row_xbc, kRows, kChannels, kKernel,
                         nullptr),
        "conv rows");
    k::MambaScanArgs rows{
        row_xbc, kChannels, d_proj, kStride,  kInner + kChannels,
        kScale,  d_dt_b,    d_a,    d_d,      nullptr,
        row_y,   kRows,     kHeads, kHeadDim, kGroups};
    rows.row_states = d_ssm_rows;
    Hip(k::MambaScan(rows, nullptr), "scan rows");
    const auto same = [](const std::vector<float>& x,
                         const std::vector<float>& y) {
      return x.size() == y.size() &&
             std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0;
    };
    Check(same(Host(row_xbc, kRows * kChannels),
               Host(seq_xbc, kRows * kChannels)),
          "row convolution matches one-token convolutions");
    Check(same(Host(row_y, kRows * kInner), Host(seq_y, kRows * kInner)),
          "row scan matches one-token scans");
    bool states = true;
    for (int r = 0; r < kRows; ++r)
      states = states &&
               same(Host(row_conv[r], kConv), Host(seq_conv[r], kConv)) &&
               same(Host(row_ssm[r], kSsm), Host(seq_ssm[r], kSsm));
    Check(states, "row states match one-token states");
    for (int r = 0; r < kRows; ++r)
      for (float* p : {seq_conv[r], seq_ssm[r], row_conv[r], row_ssm[r]})
        Hip(hipFree(p), "free");
    for (void* p :
         {static_cast<void*>(d_proj), static_cast<void*>(d_conv_w),
          static_cast<void*>(d_conv_b), static_cast<void*>(d_dt_b),
          static_cast<void*>(d_a), static_cast<void*>(d_d),
          static_cast<void*>(d_conv_rows), static_cast<void*>(d_ssm_rows),
          static_cast<void*>(seq_xbc), static_cast<void*>(seq_y),
          static_cast<void*>(row_xbc), static_cast<void*>(row_y)})
      Hip(hipFree(p), "free");
  }
  {
    constexpr int kHeads = 32;
    constexpr int kKvHeads = 2;
    constexpr int kHd = 128;
    constexpr int kKv = kKvHeads * kHd;
    constexpr int kContext = 512;
    const int positions[kRows] = {0, 70, 300};
    const float scale = 1.0F / std::sqrt(static_cast<float>(kHd));
    const auto q = Random(kRows * kHeads * kHd, rng);
    const auto k_new = Random(kRows * kKv, rng);
    const auto v_new = Random(kRows * kKv, rng);
    auto* d_q = Device(q);
    auto* d_k = Device(k_new);
    auto* d_v = Device(v_new);
    std::vector<__half*> seq_k, seq_v, row_k, row_v;
    for (int r = 0; r < kRows; ++r) {
      std::vector<__half> kc(kContext * kKv), vc(kContext * kKv);
      for (auto& x : kc)
        x = __float2half(Random(1, rng)[0]);
      for (auto& x : vc)
        x = __float2half(Random(1, rng)[0]);
      seq_k.push_back(Device(kc));
      seq_v.push_back(Device(vc));
      row_k.push_back(Device(kc));
      row_v.push_back(Device(vc));
    }
    auto* partial = Device(
        std::vector<float>(static_cast<std::size_t>(k::kAttentionShortTokens) *
                           kHeads * k::kAttentionSplits * (kHd + 2)));
    auto* seq_out = Device(std::vector<float>(kRows * kHeads * kHd));
    for (int r = 0; r < kRows; ++r) {
      auto* pos = Device(std::vector<int>{positions[r]});
      Hip(k::KvAppend(d_k + r * kKv, d_v + r * kKv, seq_k[r], seq_v[r], pos, 1,
                      kKv, nullptr),
          "kv append");
      Hip(k::AttentionShort(
              d_q + static_cast<std::size_t>(r) * kHeads * kHd, seq_k[r],
              seq_v[r], seq_out + static_cast<std::size_t>(r) * kHeads * kHd,
              pos, 1, kHeads, kKvHeads, scale, partial, nullptr),
          "attention");
      Hip(hipDeviceSynchronize(), "sync");
      Hip(hipFree(pos), "free");
    }
    auto* d_pos = Device(std::vector<int>(positions, positions + kRows));
    auto* d_k_rows = Device(row_k);
    auto* d_v_rows = Device(row_v);
    auto* row_out = Device(std::vector<float>(kRows * kHeads * kHd));
    Hip(k::KvAppendRows(d_k, d_v, d_k_rows, d_v_rows, d_pos, kRows, kKv,
                        nullptr),
        "kv append rows");
    Hip(k::AttentionShortRows(d_q, d_k_rows, d_v_rows, row_out, d_pos, kRows,
                              kHeads, kKvHeads, scale, partial, nullptr),
        "attention rows");
    const auto got = Host(row_out, kRows * kHeads * kHd);
    const auto want = Host(seq_out, kRows * kHeads * kHd);
    Check(std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0,
          "row attention matches one-token attention");
    bool caches = true;
    for (int r = 0; r < kRows; ++r) {
      const auto a = Host(row_k[r], kContext * kKv);
      const auto b = Host(seq_k[r], kContext * kKv);
      const auto c = Host(row_v[r], kContext * kKv);
      const auto e = Host(seq_v[r], kContext * kKv);
      caches =
          caches &&
          std::memcmp(a.data(), b.data(), a.size() * sizeof(__half)) == 0 &&
          std::memcmp(c.data(), e.data(), c.size() * sizeof(__half)) == 0;
    }
    Check(caches, "row KV appends match one-token appends");
    for (int r = 0; r < kRows; ++r)
      for (__half* p : {seq_k[r], seq_v[r], row_k[r], row_v[r]})
        Hip(hipFree(p), "free");
    for (void* p : {static_cast<void*>(d_q), static_cast<void*>(d_k),
                    static_cast<void*>(d_v), static_cast<void*>(partial),
                    static_cast<void*>(seq_out), static_cast<void*>(d_pos),
                    static_cast<void*>(d_k_rows), static_cast<void*>(d_v_rows),
                    static_cast<void*>(row_out)})
      Hip(hipFree(p), "free");
  }
}

int main() {
  std::mt19937 rng(1234);
  TestEmbedAndNorm(rng);
  TestMamba(rng);
  TestAttention(rng);
  TestAttentionSplitBoundary(rng);
  TestBatchRows(rng);
  TestRouterLogits(rng);
  TestRouting(rng);
  TestLogitHeads(rng);
  if (failures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("all kernel checks passed\n");
  return 0;
}
