#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "staging/models/glm5_next/kernels/rocm/ops.hpp"
#include "staging/tests/models/glm5_next/hip_test.hpp"

// Parity checks for the glm5_next fused operators against independent host
// double references of the formulas in IMPLEMENTATION.md: the norms, the
// mHC mixing coefficients (sinkhorn) and stream combinations, the KDA
// causal conv ring and delta-recurrence step, the DSA indexer pooling
// score, stable descending top-k selection, MoE routing, the swiglu clamp
// epilogue and the weighted expert accumulation. Synthetic inputs only;
// no model artifact required.
namespace {

namespace q = gufo::models::glm5_next::rocm;
namespace t = gufo::tests::glm5_next;

constexpr unsigned kSinkhornIters = 20U;

double SigmoidD(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}

bool Check(const char* what, double worst, double limit) {
  const bool ok = worst <= limit;
  std::printf("%-34s worst %.3e limit %.0e %s\n", what, worst, limit,
              ok ? "OK" : "FAIL");
  return ok;
}

float RoundTie(float v, float step) {
  return std::floor(v / step) * step;
}

bool TestAddBias() {
  const std::uint32_t total = 1000;
  const std::uint32_t bias_len = 288;
  const auto data = t::MakeValues(total, 0xA1B2C3D4U, 2.0F);
  const auto bias = t::MakeValues(bias_len, 0x0F1E2D3CU, 1.0F);
  t::HipBuffer<float> d_data(total);
  t::Upload(&d_data, data);
  t::HipBuffer<float> d_bias(bias_len);
  t::Upload(&d_bias, bias);
  q::AddBias(d_data.get(), d_bias.get(), total, bias_len, nullptr);
  const auto got = t::Download(&d_data, total);
  std::vector<float> ref(total);
  for (std::uint32_t i = 0; i < total; ++i) {
    ref[i] = data[i] + bias[i % bias_len];
  }
  return Check("add_bias (exact)", t::WorstRelative(ref, got), 0.0);
}

bool TestRmsNorm() {
  const std::uint32_t rows = 3;
  const std::uint32_t n = 4096;
  const std::uint32_t x_stride = 4200;
  const std::uint32_t dst_stride = 4111;
  const auto x = t::MakeValues(rows * x_stride, 0x55AA55AAU, 1.0F);
  const auto w = t::MakeValues(n, 0x33445566U, 2.0F, 1.0F);
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  t::HipBuffer<float> d_w(n);
  t::Upload(&d_w, w);
  t::HipBuffer<float> d_out(rows * dst_stride);
  bool ok = true;
  for (const float eps : {1e-5F, 1e-6F}) {
    q::RmsNorm(d_x.get(), d_w.get(), d_out.get(), rows, n, x_stride, dst_stride,
               eps, nullptr);
    const auto got = t::Download(&d_out, rows * dst_stride);
    std::vector<float> ref(got);
    for (std::uint32_t r = 0; r < rows; ++r) {
      double ss = 0.0;
      for (std::uint32_t i = 0; i < n; ++i) {
        const double v = x[static_cast<std::size_t>(r) * x_stride + i];
        ss += v * v;
      }
      const double scale = 1.0 / std::sqrt(ss / n + eps);
      for (std::uint32_t i = 0; i < n; ++i) {
        const auto idx = static_cast<std::size_t>(r) * dst_stride + i;
        ref[idx] = static_cast<float>(
            w[i] * x[static_cast<std::size_t>(r) * x_stride + i] * scale);
      }
    }
    ok &=
        Check((std::string("rms_norm eps ") + (eps == 1e-5F ? "1e-5" : "1e-6"))
                  .c_str(),
              t::WorstRelative(ref, got), 2e-5);
  }
  return ok;
}

bool TestLayerNorm() {
  const std::uint32_t n = 128;
  const auto x = t::MakeValues(n, 0x1234ABCDU, 3.0F, 0.5F);
  const auto w = t::MakeValues(n, 0x9FE0D1C2U, 0.5F, 1.0F);
  const auto b = t::MakeValues(n, 0x0ABCDEF1U, 0.25F);
  t::HipBuffer<float> d_x(n);
  t::Upload(&d_x, x);
  t::HipBuffer<float> d_w(n);
  t::Upload(&d_w, w);
  t::HipBuffer<float> d_b(n);
  t::Upload(&d_b, b);
  t::HipBuffer<float> d_out(n);
  q::LayerNorm(d_x.get(), d_w.get(), d_b.get(), d_out.get(), n, 1e-6F, nullptr);
  const auto got = t::Download(&d_out, n);
  std::vector<float> ref(n);
  double mean = 0.0;
  for (const float v : x) {
    mean += v;
  }
  mean /= n;
  double var = 0.0;
  for (const float v : x) {
    var += (v - mean) * (v - mean);
  }
  const double inv = 1.0 / std::sqrt(var / n + 1e-6);
  for (std::uint32_t i = 0; i < n; ++i) {
    ref[i] = static_cast<float>(w[i] * ((x[i] - mean) * inv) + b[i]);
  }
  return Check("layer_norm", t::WorstRelative(ref, got), 2e-5);
}

bool TestHcCoeffs() {
  const auto mixes = t::MakeValues(24, 0x77FEED12U, 4.0F);
  const auto scale = t::MakeValues(3, 0x5150A5A5U, 0.5F, 1.0F);
  const auto base = t::MakeValues(24, 0x2468ACE0U, 1.0F);
  t::HipBuffer<float> d_mixes(mixes.size());
  t::Upload(&d_mixes, mixes);
  t::HipBuffer<float> d_scale(scale.size());
  t::Upload(&d_scale, scale);
  t::HipBuffer<float> d_base(base.size());
  t::Upload(&d_base, base);
  t::HipBuffer<float> d_pre(4);
  t::HipBuffer<float> d_post(4);
  t::HipBuffer<float> d_comb(16);
  q::HcCoeffs(d_mixes.get(), d_scale.get(), d_base.get(), kSinkhornIters, 1e-6F,
              d_pre.get(), d_post.get(), d_comb.get(), 1U, nullptr);
  const auto got_pre = t::Download(&d_pre, 4);
  const auto got_post = t::Download(&d_post, 4);
  const auto got_comb = t::Download(&d_comb, 16);

  double a[24];
  for (int i = 0; i < 24; ++i) {
    const int role = i < 4 ? 0 : (i < 8 ? 1 : 2);
    a[i] = static_cast<double>(mixes[i]) * scale[role] + base[i];
  }
  std::vector<float> ref_pre(4);
  std::vector<float> ref_post(4);
  for (int i = 0; i < 4; ++i) {
    ref_pre[i] = static_cast<float>(SigmoidD(a[i]) + 1e-6);
    ref_post[i] = static_cast<float>(SigmoidD(a[4 + i]) * 2.0);
  }
  double c[4][4];
  for (int s = 0; s < 4; ++s) {
    for (int d = 0; d < 4; ++d) {
      c[s][d] = a[8 + s * 4 + d];
    }
  }
  for (int s = 0; s < 4; ++s) {
    double mx = c[s][0];
    for (int d = 1; d < 4; ++d) {
      mx = std::max(mx, c[s][d]);
    }
    double sum = 0.0;
    for (int d = 0; d < 4; ++d) {
      c[s][d] = std::exp(c[s][d] - mx);
      sum += c[s][d];
    }
    const double inv = 1.0 / sum;
    for (int d = 0; d < 4; ++d) {
      c[s][d] = c[s][d] * inv + 1e-6;
    }
  }
  for (int d = 0; d < 4; ++d) {
    double sum = 1e-6;
    for (int s = 0; s < 4; ++s) {
      sum += c[s][d];
    }
    const double inv = 1.0 / sum;
    for (int s = 0; s < 4; ++s) {
      c[s][d] *= inv;
    }
  }
  for (unsigned it = 1; it < kSinkhornIters; ++it) {
    for (int s = 0; s < 4; ++s) {
      double sum = 1e-6;
      for (int d = 0; d < 4; ++d) {
        sum += c[s][d];
      }
      const double inv = 1.0 / sum;
      for (int d = 0; d < 4; ++d) {
        c[s][d] *= inv;
      }
    }
    for (int d = 0; d < 4; ++d) {
      double sum = 1e-6;
      for (int s = 0; s < 4; ++s) {
        sum += c[s][d];
      }
      const double inv = 1.0 / sum;
      for (int s = 0; s < 4; ++s) {
        c[s][d] *= inv;
      }
    }
  }
  std::vector<float> ref_comb(16);
  for (int s = 0; s < 4; ++s) {
    for (int d = 0; d < 4; ++d) {
      ref_comb[static_cast<std::size_t>(s) * 4 + d] =
          static_cast<float>(c[s][d]);
    }
  }
  bool ok = Check("hc_coeffs pre", t::WorstRelative(ref_pre, got_pre), 2e-5);
  ok &= Check("hc_coeffs post", t::WorstRelative(ref_post, got_post), 2e-5);
  ok &= Check("hc_coeffs sinkhorn comb", t::WorstRelative(ref_comb, got_comb),
              2e-5);
  double deviation = 0.0;
  for (int s = 0; s < 4; ++s) {
    double row = 0.0;
    double col = 0.0;
    for (int d = 0; d < 4; ++d) {
      row += got_comb[static_cast<std::size_t>(s) * 4 + d];
      col += got_comb[static_cast<std::size_t>(d) * 4 + s];
    }
    deviation = std::max({deviation, std::abs(row - 1.0), std::abs(col - 1.0)});
  }
  // The sinkhorn normalization adds eps to every denominator, so the exact
  // comb carries a structural mass bias of ~1e-3 against a unit-sum
  // doubly-stochastic ideal; the bit-exact sinkhorn check above pins the
  // kernel, this bound only sanity-checks the deviation scale.
  ok &= Check("hc_comb doubly stochastic", deviation, 5e-3);
  return ok;
}

bool TestHcCombine() {
  const std::uint32_t n = 4096;
  const auto streams = t::MakeValues(4 * n, 0x13579BDFU, 1.0F);
  const auto pre = t::MakeValues(4, 0x9AC3E5F7U, 0.5F, 0.75F);
  const auto post = t::MakeValues(4, 0x2468B5A1U, 1.0F, 0.5F);
  const auto comb = t::MakeValues(16, 0x8ACE1357U, 0.25F, 0.125F);
  const auto y = t::MakeValues(n, 0xFEEDFACEU, 1.0F);
  t::HipBuffer<float> d_streams(streams.size());
  t::Upload(&d_streams, streams);
  t::HipBuffer<float> d_pre(pre.size());
  t::Upload(&d_pre, pre);
  t::HipBuffer<float> d_post(post.size());
  t::Upload(&d_post, post);
  t::HipBuffer<float> d_comb(comb.size());
  t::Upload(&d_comb, comb);
  t::HipBuffer<float> d_y(y.size());
  t::Upload(&d_y, y);

  t::HipBuffer<float> d_in(n);
  q::HcInputs(d_streams.get(), d_pre.get(), 4U, 4, n, d_in.get(), 1U,
              nullptr);
  const auto got_in = t::Download(&d_in, n);
  std::vector<float> ref_in(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    double acc = 0.0;
    for (unsigned s = 0; s < 4; ++s) {
      acc += static_cast<double>(streams[static_cast<std::size_t>(s) * n + i]) *
             pre[s];
    }
    ref_in[i] = static_cast<float>(acc);
  }
  // Cancellation in the 4-term dot makes a few outputs land near the
  // 1e-3 floor, where float chain rounding exceeds 2e-5 relative.
  bool ok = Check("hc_inputs", t::WorstRelative(ref_in, got_in), 1e-4);

  std::vector<float> ref(streams.size());
  for (std::uint32_t d = 0; d < 4; ++d) {
    for (std::uint32_t i = 0; i < n; ++i) {
      const auto idx = static_cast<std::size_t>(d) * n + i;
      ref[idx] = static_cast<float>(
          post[d] * y[i] + static_cast<double>(streams[i]) * comb[d] +
          static_cast<double>(streams[n + i]) * comb[4 + d] +
          static_cast<double>(streams[2 * n + i]) * comb[8 + d] +
          static_cast<double>(streams[3 * n + i]) * comb[12 + d]);
    }
  }

  t::HipBuffer<float> d_out(streams.size());
  q::HcWriteback(d_y.get(), d_streams.get(), d_post.get(), d_comb.get(), 4, n,
                 d_out.get(), 1U, nullptr);
  const auto got = t::Download(&d_out, streams.size());
  // Same cancellation-floor effect as hc_inputs.
  ok &= Check("hc_writeback", t::WorstRelative(ref, got), 1e-4);

  t::HipBuffer<float> d_alias(streams.size());
  t::Upload(&d_alias, streams);
  q::HcWriteback(d_y.get(), d_alias.get(), d_post.get(), d_comb.get(), 4, n,
                 d_alias.get(), 1U, nullptr);
  const auto got_alias = t::Download(&d_alias, streams.size());
  ok &= Check("hc_writeback aliased", t::WorstRelative(ref, got_alias), 1e-4);
  return ok;
}

bool TestKdaConv() {
  const std::uint32_t n = 8192;
  const auto w = t::MakeValues(4 * n, 0xD1C0B0A0U, 0.25F);
  t::HipBuffer<float> d_hist(3 * n);
  t::CheckHip(hipMemset(d_hist.get(), 0, 3 * n * sizeof(float)), "memset");
  t::HipBuffer<float> d_w(w.size());
  t::Upload(&d_w, w);
  t::HipBuffer<float> d_raw(n);
  t::HipBuffer<float> d_dst(n);
  std::vector<float> hist(3 * n, 0.0F);  // host mirror of the ring
  bool ok = true;
  const std::uint32_t base_seed = 0xA9E1D1E0U;
  for (std::uint64_t pos = 0; pos < 6; ++pos) {
    const auto raw =
        t::MakeValues(n, base_seed + static_cast<std::uint32_t>(pos), 2.0F);
    t::Upload(&d_raw, raw);
    q::KdaConvStep(pos, d_raw.get(), d_w.get(), d_hist.get(), n, d_dst.get(),
                   nullptr);
    const auto got = t::Download(&d_dst, n);
    const std::uint64_t s0 = pos % 3;
    const std::size_t slot[3] = {static_cast<std::size_t>(s0),
                                 static_cast<std::size_t>((s0 + 1) % 3),
                                 static_cast<std::size_t>((s0 + 2) % 3)};
    std::vector<float> ref(n);
    for (std::uint32_t ch = 0; ch < n; ++ch) {
      const double x3 = hist[slot[0] * n + ch];
      const double x2 = hist[slot[1] * n + ch];
      const double x1 = hist[slot[2] * n + ch];
      const double acc = w[ch * 4 + 0] * x3 + w[ch * 4 + 1] * x2 +
                         w[ch * 4 + 2] * x1 + w[ch * 4 + 3] * raw[ch];
      ref[ch] = static_cast<float>(acc / (1.0 + std::exp(-acc)));
      hist[slot[0] * n + ch] = raw[ch];
    }
    ok &= Check((std::string("kda_conv step ") + std::to_string(pos)).c_str(),
                t::WorstRelative(ref, got), 2e-5);
  }
  return ok;
}

bool TestKdaRecurrent() {
  const unsigned heads = 8;
  const unsigned d = 128;
  const std::size_t channels = static_cast<std::size_t>(heads) * d;
  const auto w_state = t::MakeValues(channels * d, 0x0BADC0DEU, 0.2F);
  t::HipBuffer<float> d_state(w_state.size());
  t::Upload(&d_state, w_state);
  std::vector<double> state(w_state.begin(), w_state.end());
  const auto beta = t::MakeValues(heads, 0x3C3C3C3CU, 0.5F, 0.5F);
  std::vector<double> beta_state(beta.begin(), beta.end());
  t::HipBuffer<float> d_beta(heads);
  t::Upload(&d_beta, beta);
  std::vector<std::uint32_t> seeds = {0xFADE0001U, 0xFADE0002U, 0xFADE0003U};
  bool ok = true;
  for (unsigned step = 0; step < 3; ++step) {
    const std::uint32_t s = seeds[step];
    const auto eg = t::MakeValues(channels, s ^ 0x1111U, 0.4F, 0.5F);
    const auto k = t::MakeValues(channels, s ^ 0x2222U, 1.0F);
    const auto qv = t::MakeValues(channels, s ^ 0x3333U, 1.0F);
    const auto v = t::MakeValues(channels, s ^ 0x4444U, 1.0F);
    t::HipBuffer<float> d_eg(eg.size());
    t::Upload(&d_eg, eg);
    t::HipBuffer<float> d_k(k.size());
    t::Upload(&d_k, k);
    t::HipBuffer<float> d_q(qv.size());
    t::Upload(&d_q, qv);
    t::HipBuffer<float> d_v(v.size());
    t::Upload(&d_v, v);
    t::HipBuffer<float> d_sk(channels);
    t::HipBuffer<float> d_o(channels);
    q::KdaFoldSk(d_state.get(), d_eg.get(), d_k.get(), heads, d, d_sk.get(),
                 nullptr);
    q::KdaUpdateOut(d_state.get(), d_k.get(), d_q.get(), d_v.get(),
                    d_beta.get(), d_sk.get(), heads, d, d_o.get(), nullptr);
    const auto got_sk = t::Download(&d_sk, channels);
    const auto got_o = t::Download(&d_o, channels);
    std::vector<float> ref_sk(channels);
    std::vector<float> ref_o(channels);
    for (unsigned h = 0; h < heads; ++h) {
      for (unsigned j = 0; j < d; ++j) {
        double acc = 0.0;
        for (unsigned i = 0; i < d; ++i) {
          double& cell = state[(static_cast<std::size_t>(h) * d + i) * d + j];
          cell *= eg[h * d + i];
          acc += k[h * d + i] * cell;
        }
        ref_sk[h * d + j] = static_cast<float>(acc);
      }
      const std::size_t base = static_cast<std::size_t>(h) * d;
      for (unsigned j = 0; j < d; ++j) {
        const double dj = beta_state[h] * (v[base + j] - ref_sk[base + j]);
        double acc = 0.0;
        for (unsigned i = 0; i < d; ++i) {
          double& cell = state[(static_cast<std::size_t>(h) * d + i) * d + j];
          cell += k[base + i] * dj;
          acc += qv[base + i] * cell;
        }
        ref_o[base + j] = static_cast<float>(acc);
      }
    }
    const std::string tag = std::to_string(step);
    // The fold/out dots run in float and are cancellation-prone; the chain
    // noise against the double reference sits at the low e-4 level.
    ok &= Check(("kda_fold_sk step " + tag).c_str(),
                t::WorstRelative(ref_sk, got_sk), 5e-4);
    ok &= Check(("kda_out step " + tag).c_str(), t::WorstRelative(ref_o, got_o),
                2e-3);
  }
  const auto got_state = t::Download(&d_state, channels * d);
  std::vector<float> ref_state(got_state.size());
  for (std::size_t i = 0; i < ref_state.size(); ++i) {
    ref_state[i] = static_cast<float>(state[i]);
  }
  ok &= Check("kda_state after 3 steps", t::WorstRelative(ref_state, got_state),
              1e-3);
  return ok;
}

bool TestKdaConvChunkBitExact() {
  const std::uint32_t n = 8192;
  const std::uint32_t ntok = 5;
  const std::uint64_t pos0 = 7;
  const auto w = t::MakeValues(4 * n, 0xD1C0B0A0U, 0.25F);
  t::HipBuffer<float> d_w(w.size());
  t::Upload(&d_w, w);
  std::vector<float> raw(n * ntok);
  for (std::uint32_t r = 0; r < ntok; ++r) {
    const auto row = t::MakeValues(n, 0x5171A000U + r, 2.0F);
    std::copy(row.begin(), row.end(), raw.begin() + static_cast<std::size_t>(r) * n);
  }
  t::HipBuffer<float> d_raw(raw.size());
  t::Upload(&d_raw, raw);
  t::HipBuffer<float> d_hist_seq(3 * n);
  t::HipBuffer<float> d_hist_chunk(3 * n);
  t::CheckHip(hipMemset(d_hist_seq.get(), 0, 3 * n * sizeof(float)), "memset");
  t::CheckHip(hipMemset(d_hist_chunk.get(), 0, 3 * n * sizeof(float)),
              "memset");
  t::HipBuffer<float> d_dst_seq(raw.size());
  t::HipBuffer<float> d_dst_chunk(raw.size());
  for (std::uint32_t r = 0; r < ntok; ++r) {
    q::KdaConvStep(pos0 + r, d_raw.get() + static_cast<std::size_t>(r) * n,
                   d_w.get(), d_hist_seq.get(), n,
                   d_dst_seq.get() + static_cast<std::size_t>(r) * n, nullptr);
  }
  q::KdaConvStepChunk(pos0, d_raw.get(), d_w.get(), d_hist_chunk.get(), n,
                      ntok, d_dst_chunk.get(), nullptr);
  const auto dst_seq = t::Download(&d_dst_seq, raw.size());
  const auto dst_chunk = t::Download(&d_dst_chunk, raw.size());
  const auto hist_seq = t::Download(&d_hist_seq, 3 * n);
  const auto hist_chunk = t::Download(&d_hist_chunk, 3 * n);
  bool ok_dst = dst_seq.size() == dst_chunk.size();
  for (std::size_t i = 0; ok_dst && i < dst_seq.size(); ++i) {
    ok_dst &= dst_seq[i] == dst_chunk[i];
  }
  bool ok_hist = hist_seq.size() == hist_chunk.size();
  for (std::size_t i = 0; ok_hist && i < hist_seq.size(); ++i) {
    ok_hist &= hist_seq[i] == hist_chunk[i];
  }
  bool ok = Check("kda_conv_chunk dst bit-exact", ok_dst ? 0.0 : 1.0, 0.0);
  ok &= Check("kda_conv_chunk hist bit-exact", ok_hist ? 0.0 : 1.0, 0.0);
  return ok;
}

bool TestKdaFoldUpdateChunkBitExact() {
  // Reference: per-token KdaFoldSk + KdaUpdateOut chain; candidate: the
  // fused chunk launch. Both must agree bitwise (o and final state).
  const unsigned heads = 8;
  const unsigned d = 128;
  const std::uint32_t ntok = 4;
  const std::size_t channels = static_cast<std::size_t>(heads) * d;
  const auto state0 = t::MakeValues(channels * d, 0x0BADC0DEU, 0.2F);
  std::vector<float> eg(ntok * channels);
  std::vector<float> k(ntok * channels);
  std::vector<float> qv(ntok * channels);
  std::vector<float> v(ntok * channels);
  std::vector<float> beta(ntok * heads);
  for (std::uint32_t r = 0; r < ntok; ++r) {
    const std::uint32_t s = 0xFADE0001U + r;
    auto put = [&](std::vector<float>& dst, unsigned seed_xor, float scale,
                   float off = 0.0F) {
      const auto row = t::MakeValues(dst.size() / ntok, s ^ seed_xor, scale,
                                     off);
      std::copy(row.begin(), row.end(), dst.begin() + r * (dst.size() / ntok));
    };
    put(eg, 0x1111U, 0.4F, 0.5F);
    put(k, 0x2222U, 1.0F);
    put(qv, 0x3333U, 1.0F);
    put(v, 0x4444U, 1.0F);
    put(beta, 0x5555U, 0.5F, 0.5F);
  }
  t::HipBuffer<float> d_state_ref(state0.size());
  t::HipBuffer<float> d_state_chunk(state0.size());
  t::Upload(&d_state_ref, state0);
  t::Upload(&d_state_chunk, state0);
  t::HipBuffer<float> d_eg(eg.size());
  t::Upload(&d_eg, eg);
  t::HipBuffer<float> d_k(k.size());
  t::Upload(&d_k, k);
  t::HipBuffer<float> d_q(qv.size());
  t::Upload(&d_q, qv);
  t::HipBuffer<float> d_v(v.size());
  t::Upload(&d_v, v);
  t::HipBuffer<float> d_beta(beta.size());
  t::Upload(&d_beta, beta);
  t::HipBuffer<float> d_sk(channels);
  t::HipBuffer<float> d_o_ref(ntok * channels);
  t::HipBuffer<float> d_o_chunk(ntok * channels);
  for (std::uint32_t r = 0; r < ntok; ++r) {
    q::KdaFoldSk(d_state_ref.get(), d_eg.get() + r * channels,
                 d_k.get() + r * channels, heads, d, d_sk.get(), nullptr);
    q::KdaUpdateOut(d_state_ref.get(), d_k.get() + r * channels,
                    d_q.get() + r * channels, d_v.get() + r * channels,
                    d_beta.get() + r * heads, d_sk.get(), heads, d,
                    d_o_ref.get() + r * channels, nullptr);
  }
  const bool launched =
      q::KdaFoldUpdateChunk(d_state_chunk.get(), d_eg.get(), d_k.get(),
                            d_q.get(), d_v.get(), d_beta.get(), heads, d, ntok,
                            d_o_chunk.get(), nullptr);
  if (!launched) {
    std::printf("kda_fold_update_chunk NOT LAUNCHED (no LDS)\n");
    return false;
  }
  const auto o_ref = t::Download(&d_o_ref, ntok * channels);
  const auto o_chunk = t::Download(&d_o_chunk, ntok * channels);
  const auto st_ref = t::Download(&d_state_ref, channels * d);
  const auto st_chunk = t::Download(&d_state_chunk, channels * d);
  bool ok_o = o_ref.size() == o_chunk.size();
  for (std::size_t i = 0; ok_o && i < o_ref.size(); ++i) {
    ok_o &= o_ref[i] == o_chunk[i];
  }
  bool ok_st = st_ref.size() == st_chunk.size();
  for (std::size_t i = 0; ok_st && i < st_ref.size(); ++i) {
    ok_st &= st_ref[i] == st_chunk[i];
  }
  bool ok = Check("kda_fold_update_chunk o bit-exact", ok_o ? 0.0 : 1.0, 0.0);
  ok &= Check("kda_fold_update_chunk state bit-exact", ok_st ? 0.0 : 1.0, 0.0);
  return ok;
}

bool BitEqual(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

// Row batching (blockIdx.y) must reproduce the per-row launches exactly,
// so the executor can run one launch per chunk.
bool TestRowBatchBitExact() {
  bool ok = true;
  {
    const auto mixes = t::MakeValues(3 * 24, 0x1111AABBU, 4.0F);
    const auto scale = t::MakeValues(3, 0x5150A5A5U, 0.5F, 1.0F);
    const auto base = t::MakeValues(24, 0x2468ACE0U, 1.0F);
    t::HipBuffer<float> d_mixes(mixes.size());
    t::Upload(&d_mixes, mixes);
    t::HipBuffer<float> d_scale(scale.size());
    t::Upload(&d_scale, scale);
    t::HipBuffer<float> d_base(base.size());
    t::Upload(&d_base, base);
    t::HipBuffer<float> d_pre(3 * 4);
    t::HipBuffer<float> d_post(3 * 4);
    t::HipBuffer<float> d_comb(3 * 16);
    for (std::uint32_t r = 0; r < 3; ++r) {
      q::HcCoeffs(d_mixes.get() + r * 24, d_scale.get(), d_base.get(),
                  kSinkhornIters, 1e-6F, d_pre.get() + r * 4,
                  d_post.get() + r * 4, d_comb.get() + r * 16, 1U, nullptr);
    }
    const auto pre_seq = t::Download(&d_pre, 3 * 4);
    const auto post_seq = t::Download(&d_post, 3 * 4);
    const auto comb_seq = t::Download(&d_comb, 3 * 16);
    q::HcCoeffs(d_mixes.get(), d_scale.get(), d_base.get(), kSinkhornIters,
                1e-6F, d_pre.get(), d_post.get(), d_comb.get(), 3U, nullptr);
    ok &= Check("hc_coeffs rows bit-exact",
                BitEqual(pre_seq, t::Download(&d_pre, 3 * 4)) &&
                        BitEqual(post_seq, t::Download(&d_post, 3 * 4)) &&
                        BitEqual(comb_seq, t::Download(&d_comb, 3 * 16))
                    ? 0.0
                    : 1.0,
                0.0);
  }
  const std::uint32_t n = 137;
  {
    const auto streams = t::MakeValues(3 * 4 * n, 0x2323B0B0U, 1.0F);
    const auto pre = t::MakeValues(3 * 4, 0x3434C0C0U, 0.5F, -0.25F);
    t::HipBuffer<float> d_streams(streams.size());
    t::Upload(&d_streams, streams);
    t::HipBuffer<float> d_pre(pre.size());
    t::Upload(&d_pre, pre);
    t::HipBuffer<float> d_dst(3 * n);
    for (std::uint32_t r = 0; r < 3; ++r) {
      q::HcInputs(d_streams.get() + static_cast<std::size_t>(r) * 4 * n,
                  d_pre.get() + r * 4, 4U, 4, n, d_dst.get() + r * n, 1U,
                  nullptr);
    }
    const auto in_seq = t::Download(&d_dst, 3 * n);
    q::HcInputs(d_streams.get(), d_pre.get(), 4U, 4, n, d_dst.get(), 3U,
                nullptr);
    ok &= Check("hc_inputs rows bit-exact",
                BitEqual(in_seq, t::Download(&d_dst, 3 * n)) ? 0.0 : 1.0, 0.0);
    std::vector<float> bcast(3 * n);
    for (std::uint32_t r = 0; r < 3; ++r) {
      q::HcInputs(d_streams.get() + static_cast<std::size_t>(r) * 4 * n,
                  d_pre.get(), 0U, 4, n, d_dst.get() + r * n, 1U, nullptr);
      const auto part = t::Download(&d_dst, static_cast<std::size_t>(r + 1) * n);
      std::copy(part.begin() + static_cast<std::size_t>(r) * n,
                part.end(), bcast.begin() + static_cast<std::size_t>(r) * n);
    }
    q::HcInputs(d_streams.get(), d_pre.get(), 0U, 4, n, d_dst.get(), 3U,
                nullptr);
    ok &= Check("hc_inputs broadcast pre rows bit-exact",
                BitEqual(bcast, t::Download(&d_dst, 3 * n)) ? 0.0 : 1.0, 0.0);
  }
  {
    const auto post = t::MakeValues(3 * 4, 0x4545D0D0U, 0.5F);
    const std::vector<float> comb_flat = [&] {
      std::vector<float> v(3 * 16);
      const auto src = t::MakeValues(3 * 16, 0x5656E0E0U, 0.25F, -0.1F);
      for (std::uint32_t r = 0; r < 3; ++r) {
        // Doubly stochastic-ish rows: normalize each [src][dst] row.
        for (int s = 0; s < 4; ++s) {
          float sum = 0.0F;
          for (int d = 0; d < 4; ++d) {
            sum += src[r * 16 + s * 4 + d] + 1.5F;
          }
          for (int d = 0; d < 4; ++d) {
            v[r * 16 + s * 4 + d] = (src[r * 16 + s * 4 + d] + 1.5F) / sum;
          }
        }
      }
      return v;
    }();
    const auto y = t::MakeValues(3 * n, 0x6767F0F0U, 1.0F);
    const auto res = t::MakeValues(3 * 4 * n, 0x78781010U, 1.0F);
    t::HipBuffer<float> d_post(post.size());
    t::Upload(&d_post, post);
    t::HipBuffer<float> d_comb(comb_flat.size());
    t::Upload(&d_comb, comb_flat);
    t::HipBuffer<float> d_y(y.size());
    t::Upload(&d_y, y);
    t::HipBuffer<float> d_res(res.size());
    t::Upload(&d_res, res);
    t::HipBuffer<float> d_out(3 * 4 * n);
    for (std::uint32_t r = 0; r < 3; ++r) {
      q::HcWriteback(d_y.get() + r * n,
                     d_res.get() + static_cast<std::size_t>(r) * 4 * n,
                     d_post.get() + r * 4, d_comb.get() + r * 16, 4, n,
                     d_out.get() + static_cast<std::size_t>(r) * 4 * n, 1U,
                     nullptr);
    }
    const auto seq = t::Download(&d_out, 3 * 4 * n);
    q::HcWriteback(d_y.get(), d_res.get(), d_post.get(), d_comb.get(), 4, n,
                   d_out.get(), 3U, nullptr);
    ok &= Check("hc_writeback rows bit-exact",
                BitEqual(seq, t::Download(&d_out, 3 * 4 * n)) ? 0.0 : 1.0, 0.0);
    q::HcWriteback(d_y.get(), d_res.get(), d_post.get(), d_comb.get(), 4, n,
                   d_res.get(), 3U, nullptr);
    ok &= Check("hc_writeback rows aliased bit-exact",
                BitEqual(seq, t::Download(&d_res, 3 * 4 * n)) ? 0.0 : 1.0, 0.0);
  }
  {
    const std::uint32_t experts = 288;
    const std::uint32_t top_k = 8;
    auto logits = t::MakeValues(3 * experts, 0x89892020U, 3.0F, -1.5F);
    for (std::uint32_t i = 0; i < experts; ++i) {
      logits[experts + i] = logits[i];  // duplicate row ties the ids
    }
    const auto bias = t::MakeValues(experts, 0x9A9A3030U, 0.25F);
    t::HipBuffer<float> d_logits(logits.size());
    t::Upload(&d_logits, logits);
    t::HipBuffer<float> d_bias(bias.size());
    t::Upload(&d_bias, bias);
    t::HipBuffer<float> d_w(3 * top_k);
    t::HipBuffer<std::int32_t> d_ids(3 * top_k);
    for (std::uint32_t r = 0; r < 3; ++r) {
      q::MoERoute(d_logits.get() + static_cast<std::size_t>(r) * experts,
                  d_bias.get(), experts, top_k, 2.5F, d_w.get() + r * top_k,
                  d_ids.get() + r * top_k, 1U, nullptr);
    }
    const auto w_seq = t::Download(&d_w, 3 * top_k);
    std::vector<std::int32_t> ids_seq(3 * top_k);
    t::CheckHip(hipMemcpy(ids_seq.data(), d_ids.get(),
                          3 * top_k * sizeof(std::int32_t),
                          hipMemcpyDeviceToHost),
                "download ids");
    q::MoERoute(d_logits.get(), d_bias.get(), experts, top_k, 2.5F, d_w.get(),
                d_ids.get(), 3U, nullptr);
    const auto w_bat = t::Download(&d_w, 3 * top_k);
    std::vector<std::int32_t> ids_bat(3 * top_k);
    t::CheckHip(hipMemcpy(ids_bat.data(), d_ids.get(),
                          3 * top_k * sizeof(std::int32_t),
                          hipMemcpyDeviceToHost),
                "download ids");
    ok &= Check("moe_route rows bit-exact",
                BitEqual(w_seq, w_bat) && ids_seq == ids_bat ? 0.0 : 1.0, 0.0);
  }
  {
    const std::uint32_t rows = 6;
    const std::uint32_t m = 257;
    const std::uint32_t stride = 300;
    const auto src = t::MakeValues(3 * rows * stride, 0xABAB4040U, 1.0F);
    const auto weights = t::MakeValues(3 * rows, 0xBCBC5050U, 0.5F, 0.6F);
    t::HipBuffer<float> d_src(src.size());
    t::Upload(&d_src, src);
    t::HipBuffer<float> d_weights(weights.size());
    t::Upload(&d_weights, weights);
    t::HipBuffer<float> d_dst(3 * m);
    t::CheckHip(
        hipMemset(d_dst.get(), 0, 3 * m * sizeof(float)), "memset");
    for (std::uint32_t r = 0; r < 3; ++r) {
      q::WeightedSumAdd(d_dst.get() + r * m,
                        d_src.get() + static_cast<std::size_t>(r) * rows *
                                            stride,
                        d_weights.get() + r * rows, rows, m, stride, 1U,
                        nullptr);
    }
    const auto seq = t::Download(&d_dst, 3 * m);
    t::CheckHip(hipMemset(d_dst.get(), 0, 3 * m * sizeof(float)), "memset");
    q::WeightedSumAdd(d_dst.get(), d_src.get(), d_weights.get(), rows, m,
                      stride, 3U, nullptr);
    ok &= Check("weighted_sum_add rows bit-exact",
                BitEqual(seq, t::Download(&d_dst, 3 * m)) ? 0.0 : 1.0, 0.0);
  }
  return ok;
}

bool TestKdaOutGate() {
  const unsigned heads = 64;
  const unsigned d = 128;
  const std::size_t channels = static_cast<std::size_t>(heads) * d;
  const auto o = t::MakeValues(channels, 0x77C0FFEEU, 2.0F);
  const auto w = t::MakeValues(d, 0x0123ABCDU, 0.5F, 1.0F);
  const auto g2 = t::MakeValues(channels, 0x4242BABEU, 3.0F);
  t::HipBuffer<float> d_o(o.size());
  t::Upload(&d_o, o);
  t::HipBuffer<float> d_w(w.size());
  t::Upload(&d_w, w);
  t::HipBuffer<float> d_g2(g2.size());
  t::Upload(&d_g2, g2);
  t::HipBuffer<float> d_out(channels);
  q::KdaOutGate(d_o.get(), d_w.get(), d_g2.get(), d_out.get(), heads, d, 1e-5F,
                nullptr);
  const auto got = t::Download(&d_out, channels);
  std::vector<float> ref(channels);
  for (unsigned h = 0; h < heads; ++h) {
    double ss = 0.0;
    for (unsigned i = 0; i < d; ++i) {
      ss += static_cast<double>(o[h * d + i]) * o[h * d + i];
    }
    const double scale = 1.0 / std::sqrt(ss / d + 1e-5);
    for (unsigned i = 0; i < d; ++i) {
      ref[h * d + i] = static_cast<float>(w[i] * (o[h * d + i] * scale) *
                                          SigmoidD(g2[h * d + i]));
    }
  }
  return Check("kda_out_gate", t::WorstRelative(ref, got), 2e-5);
}

bool TestIdxScore() {
  const std::uint32_t kpool = 4;
  const std::uint32_t key_len = 8;
  const std::uint32_t heads = 32;
  const std::uint32_t tokens = 38;  // 9 full pools + one with 2 taps
  const std::uint32_t pools = (tokens + kpool - 1) / kpool;
  const std::uint32_t pool_stride = 2 * key_len;
  const std::size_t rows = static_cast<std::size_t>(pools) * kpool;
  const auto pool = t::MakeValues(rows * pool_stride, 0x1BADCAFEU, 1.0F);
  const auto iq = t::MakeValues(heads * key_len, 0x2E55AA55U, 1.0F);
  const auto ape = t::MakeValues(kpool * key_len, 0x4C663399U, 2.0F);
  const auto w = t::MakeValues(heads, 0x9933155CU, 0.5F);
  const float q_scale = 1.0F / 16.0F;
  t::HipBuffer<float> d_pool(pool.size());
  t::Upload(&d_pool, pool);
  t::HipBuffer<float> d_iq(iq.size());
  t::Upload(&d_iq, iq);
  t::HipBuffer<float> d_ape(ape.size());
  t::Upload(&d_ape, ape);
  t::HipBuffer<float> d_w(w.size());
  t::Upload(&d_w, w);
  t::HipBuffer<float> d_scores(pools);
  q::IdxScore(d_pool.get(), pool_stride, tokens, kpool, key_len, heads,
              d_iq.get(), d_ape.get(), d_w.get(), q_scale, d_scores.get(),
              nullptr);
  const auto got = t::Download(&d_scores, pools);
  std::vector<double> kp(key_len, 0.0);
  std::vector<float> ref(pools);
  for (std::uint32_t P = 0; P < pools; ++P) {
    const std::uint32_t seen = tokens - P * kpool;
    const std::uint32_t taps = seen < kpool ? seen : kpool;
    const float* rows_base =
        pool.data() + static_cast<std::size_t>(P) * kpool * pool_stride;
    for (std::uint32_t j = 0; j < key_len; ++j) {
      double g[4];
      for (std::uint32_t tap = 0; tap < taps; ++tap) {
        g[tap] =
            ape[tap * key_len + j] + rows_base[tap * pool_stride + key_len + j];
      }
      double mx = g[0];
      for (std::uint32_t tap = 1; tap < taps; ++tap) {
        mx = std::max(mx, g[tap]);
      }
      double sum = 0.0;
      for (std::uint32_t tap = 0; tap < taps; ++tap) {
        g[tap] = std::exp(g[tap] - mx);
        sum += g[tap];
      }
      double acc = 0.0;
      for (std::uint32_t tap = 0; tap < taps; ++tap) {
        acc += rows_base[tap * pool_stride + j] * (g[tap] / sum);
      }
      kp[j] = acc;
    }
    double score = 0.0;
    for (std::uint32_t h = 0; h < heads; ++h) {
      double dot = 0.0;
      for (std::uint32_t j = 0; j < key_len; ++j) {
        dot += iq[h * key_len + j] * kp[j];
      }
      if (dot < 0.0) {
        dot = 0.0;
      }
      score += dot * w[h];
    }
    ref[P] = static_cast<float>(score * q_scale);
  }
  return Check("idx_score pools", t::WorstRelative(ref, got), 5e-5);
}

bool TestTopKCase(const char* tag, std::uint32_t n, std::uint32_t k,
                  float tie_step) {
  auto scores = t::MakeValues(n, 0xC0FFEE00U ^ n, 3.0F);
  if (tie_step > 0.0F) {
    for (float& v : scores) {
      v = RoundTie(v, tie_step);
    }
  }
  t::HipBuffer<float> d_scores(scores.size());
  t::Upload(&d_scores, scores);
  t::HipBuffer<std::int32_t> d_ids(k);
  t::HipBuffer<std::uint64_t> d_scratch(n + 4096);
  q::TopKDesc(d_scores.get(), n, k, d_ids.get(), d_scratch.get(), nullptr);
  std::vector<std::int32_t> got_ids(k);
  t::CheckHip(hipMemcpy(got_ids.data(), d_ids.get(), k * sizeof(std::int32_t),
                        hipMemcpyDeviceToHost),
              "download ids");
  std::vector<std::uint32_t> order(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    order[i] = i;
  }
  std::stable_sort(
      order.begin(), order.end(),
      [&](std::uint32_t a, std::uint32_t b) { return scores[a] > scores[b]; });
  std::size_t mismatch = 0;
  for (std::uint32_t i = 0; i < k; ++i) {
    if (static_cast<std::uint32_t>(got_ids[i]) != order[i]) {
      if (mismatch == 0) {
        std::printf("  first mismatch at %u: got %d want %u\n",
                    static_cast<unsigned>(i), got_ids[i], order[i]);
      }
      ++mismatch;
    }
  }
  return Check((std::string("topk ") + tag).c_str(),
               static_cast<double>(mismatch), 0.0);  // exact expected ids
}

bool TestTopKDesc() {
  bool ok = true;
  ok &= TestTopKCase("5000/512", 5000, 512, 0.0F);
  ok &= TestTopKCase("5000/512 ties", 5000, 512, 0.25F);
  ok &= TestTopKCase("1500/128", 1500, 128, 0.0F);
  ok &= TestTopKCase("1024/1024 single", 1024, 1024, 0.0F);
  ok &= TestTopKCase("4098/2048 merge", 4098, 2048, 0.125F);
  return ok;
}

bool TestMoERouteCase(const char* tag, float tie_step) {
  const std::uint32_t experts = 288;
  const std::uint32_t top_k = 8;
  const float scale = 2.5F;
  auto logits = t::MakeValues(experts, 0x5EED1234U, 6.0F);
  // exp_probs_b-style selection bias (small, signed, ties still collide).
  auto bias = t::MakeValues(experts, 0xB1A5U, 0.4F);
  if (tie_step > 0.0F) {
    for (float& v : logits) {
      v = RoundTie(v, tie_step);
    }
    for (float& v : bias) {
      v = RoundTie(v, 0.25F);
    }
  }
  t::HipBuffer<float> d_logits(experts);
  t::HipBuffer<float> d_bias(experts);
  t::Upload(&d_logits, logits);
  t::Upload(&d_bias, bias);
  t::HipBuffer<float> d_weights(top_k);
  t::HipBuffer<std::int32_t> d_ids(top_k);
  q::MoERoute(d_logits.get(), d_bias.get(), experts, top_k, scale,
              d_weights.get(), d_ids.get(), 1U, nullptr);
  const auto got_weights = t::Download(&d_weights, top_k);
  std::vector<std::int32_t> got_ids(top_k);
  t::CheckHip(hipMemcpy(got_ids.data(), d_ids.get(),
                        top_k * sizeof(std::int32_t), hipMemcpyDeviceToHost),
              "download ids");
  std::vector<std::uint32_t> order(experts);
  for (std::uint32_t i = 0; i < experts; ++i) {
    order[i] = i;
  }
  // selection on sigmoid+bias, weights from the unbiased sigmoid (Gate.forward)
  std::stable_sort(order.begin(), order.end(), [&](std::uint32_t a,
                                                   std::uint32_t b) {
    return SigmoidD(logits[a]) + bias[a] > SigmoidD(logits[b]) + bias[b];
  });
  double sum = 0.0;
  std::vector<float> ref_weights(top_k);
  bool ids_ok = true;
  for (std::uint32_t r = 0; r < top_k; ++r) {
    const std::uint32_t id = order[r];
    ids_ok &= got_ids[r] == static_cast<std::int32_t>(id);
    ref_weights[r] = static_cast<float>(SigmoidD(logits[id]));
    sum += ref_weights[r];
  }
  const double div = sum > 6.103515625e-5 ? sum : 6.103515625e-5;
  for (float& weight : ref_weights) {
    weight = static_cast<float>(weight / div * scale);
  }
  bool ok = ids_ok;
  if (!ok) {
    std::printf("  %s: ids mismatch (got %d want %u)\n", tag, got_ids[0],
                order[0]);
  }
  ok &=
      Check((std::string("moe_route ids ") + tag).c_str(), ok ? 0.0 : 1.0, 0.0);
  ok &= Check((std::string("moe_route weights ") + tag).c_str(),
              t::WorstRelative(ref_weights, got_weights), 5e-5);
  return ok;
}

bool TestMoERoute() {
  return TestMoERouteCase("random", 0.0F) && TestMoERouteCase("ties", 0.5F);
}

bool TestScale() {
  const std::uint32_t total = 1000;
  const float s = 0.0883883476F;  // 1/sqrt(128)
  const auto x = t::MakeValues(total, 0x5CA1E5U, 4.0F);
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  q::Scale(d_x.get(), total, s, nullptr);
  const auto got = t::Download(&d_x, total);
  std::vector<float> ref(total);
  for (std::uint32_t i = 0; i < total; ++i) {
    ref[i] = static_cast<float>(static_cast<double>(x[i]) * s);
  }
  return Check("scale", t::WorstRelative(ref, got), 5e-7);
}

bool TestSigmoidInPlace() {
  const std::uint32_t total = 8192;
  const auto x = t::MakeValues(total, 0x51C0FEDU, 12.0F);
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  q::SigmoidInPlace(d_x.get(), total, nullptr);
  const auto got = t::Download(&d_x, total);
  std::vector<float> ref(total);
  for (std::uint32_t i = 0; i < total; ++i) {
    ref[i] = static_cast<float>(SigmoidD(x[i]));
  }
  return Check("sigmoid_in_place", t::WorstRelative(ref, got), 2e-5);
}

bool TestKdaEGate() {
  const std::uint32_t heads = 4;
  const std::uint32_t d = 128;
  const float lower_bound = -5.0F;
  const auto g1 = t::MakeValues(heads * d, 0xDA7E1U, 6.0F);
  const auto a = t::MakeValues(heads, 0xA11CEU, 8.0F, -9.0F);  // ssm_a < 0
  t::HipBuffer<float> d_g1(g1.size());
  t::Upload(&d_g1, g1);
  t::HipBuffer<float> d_a(a.size());
  t::Upload(&d_a, a);
  t::HipBuffer<float> d_eg(heads * d);
  q::KdaEGate(d_g1.get(), d_a.get(), heads, d, lower_bound, d_eg.get(), 1U,
              nullptr);
  const auto got = t::Download(&d_eg, heads * d);
  std::vector<float> ref(heads * d);
  for (std::uint32_t h = 0; h < heads; ++h) {
    for (std::uint32_t i = 0; i < d; ++i) {
      const double v = -(static_cast<double>(g1[h * d + i]) * a[h]);
      ref[h * d + i] =
          static_cast<float>(std::exp(lower_bound * SigmoidD(v)));
    }
  }
  bool ok = Check("kda_e_gate", t::WorstRelative(ref, got), 5e-5);
  // Multi-row batches keep the same per-head weight (a[(i/d) % heads]).
  std::vector<float> g3;
  for (unsigned r = 0; r < 3; ++r) {
    g3.insert(g3.end(), g1.begin(), g1.end());
  }
  t::HipBuffer<float> d_g3(g3.size());
  t::Upload(&d_g3, g3);
  std::vector<float> eg3(heads * d);
  t::HipBuffer<float> d_eg3(3 * heads * d);
  q::KdaEGate(d_g3.get(), d_a.get(), heads, d, lower_bound, d_eg3.get(), 3U,
              nullptr);
  const auto got3 = t::Download(&d_eg3, 3 * heads * d);
  bool batch_ok = true;
  for (unsigned r = 0; r < 3; ++r) {
    for (std::size_t i = 0; i < eg3.size(); ++i) {
      batch_ok &= got3[r * eg3.size() + i] == got[i];
    }
  }
  ok &= Check("kda_e_gate rows=3 bit-exact", batch_ok ? 0.0 : 1.0, 0.0);
  return ok;
}

bool TestL2NormHeadsCase(std::uint32_t heads, std::uint32_t d) {
  const float eps = 1e-6F;
  const auto x = t::MakeValues(heads * d, 0x1234ABCdu, 3.0F);
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  q::L2NormHeads(d_x.get(), heads, d, eps, nullptr);
  const auto got = t::Download(&d_x, heads * d);
  std::vector<float> ref(heads * d);
  double worst = 0.0;
  for (std::uint32_t h = 0; h < heads; ++h) {
    double sum = 0.0;
    for (std::uint32_t i = 0; i < d; ++i) {
      const double v = x[h * d + i];
      sum += v * v;
    }
    const double inv = 1.0 / std::sqrt(sum + eps);
    for (std::uint32_t i = 0; i < d; ++i) {
      ref[h * d + i] = static_cast<float>(x[h * d + i] * inv);
    }
  }
  worst = t::WorstRelative(ref, got);
  return Check(("l2norm_heads d" + std::to_string(d)).c_str(), worst, 2e-5);
}

bool TestL2NormHeads() {
  return TestL2NormHeadsCase(4, 128) && TestL2NormHeadsCase(2, 257);
}

bool TestBuildSel() {
  const std::int32_t pos = 100;
  struct Case {
    std::vector<std::int32_t> pools;
    std::uint32_t n_tail;
  };
  const Case cases[] = {
      {{7, 3, 0, 102, 45}, 3},
      {{11}, 0},
      {{}, 2},
  };
  bool ok = true;
  for (const Case& c : cases) {
    const std::uint32_t k = static_cast<std::uint32_t>(c.pools.size());
    const std::uint32_t total = 4U * k + c.n_tail;
    t::HipBuffer<std::int32_t> d_pool(k == 0 ? 1 : k);
    if (k != 0) {
      t::CheckHip(hipMemcpy(d_pool.get(), c.pools.data(),
                            k * sizeof(std::int32_t), hipMemcpyHostToDevice),
                  "upload pools");
    }
    t::HipBuffer<std::int32_t> d_sel(total == 0 ? 1 : total);
    t::CheckHip(hipMemset(d_sel.get(), 0xCD,
                          (total == 0 ? 1 : total) * sizeof(std::int32_t)),
                "memset sel");
    q::BuildSel(d_pool.get(), k, pos, c.n_tail, d_sel.get(), nullptr);
    std::vector<std::int32_t> got(total);
    if (total != 0) {
      t::CheckHip(
          hipMemcpy(got.data(), d_sel.get(), total * sizeof(std::int32_t),
                    hipMemcpyDeviceToHost),
          "download sel");
    }
    std::vector<std::int32_t> ref(total);
    for (std::uint32_t j = 0; j < k; ++j) {
      for (std::uint32_t t2 = 0; t2 < 4; ++t2) {
        ref[4 * j + t2] = c.pools[j] * 4 + static_cast<std::int32_t>(t2);
      }
    }
    for (std::uint32_t u = 0; u < c.n_tail; ++u) {
      ref[4 * k + u] =
          pos - static_cast<std::int32_t>(c.n_tail - u - 1U);
    }
    bool same = total == 0 || got == ref;
    ok &= Check(("build_sel k" + std::to_string(k)).c_str(),
                same ? 0.0 : 1.0, 0.0);
  }
  return ok;
}

bool TestSwigluClamp() {
  const std::uint32_t m = 3;
  const std::uint32_t n = 2048;
  const std::uint32_t gate_stride = 2100;
  const std::uint32_t up_stride = 2070;
  const std::uint32_t dst_stride = 2064;
  const auto gate = t::MakeValues(m * gate_stride, 0x12A1B2C3U, 15.0F);
  const auto up = t::MakeValues(m * up_stride, 0x34C5D6E7U, 15.0F);
  t::HipBuffer<float> d_gate(gate.size());
  t::Upload(&d_gate, gate);
  t::HipBuffer<float> d_up(up.size());
  t::Upload(&d_up, up);
  t::HipBuffer<float> d_out(m * dst_stride);
  q::SwigluClamp(d_gate.get(), d_up.get(), d_out.get(), m, n, gate_stride,
                 up_stride, dst_stride, 10.0F, 10.0F, nullptr);
  const auto got = t::Download(&d_out, m * dst_stride);
  std::vector<float> ref(got);
  for (std::uint32_t r = 0; r < m; ++r) {
    for (std::uint32_t c = 0; c < n; ++c) {
      const double g = std::min<double>(
          gate[static_cast<std::size_t>(r) * gate_stride + c], 10.0);
      const double u = std::min(
          10.0, std::max(-10.0,
                         static_cast<double>(
                             up[static_cast<std::size_t>(r) * up_stride + c])));
      ref[static_cast<std::size_t>(r) * dst_stride + c] =
          static_cast<float>(g * SigmoidD(g) * u);
    }
  }
  return Check("swiglu_clamp", t::WorstRelative(ref, got), 2e-5);
}

bool TestWeightedSumAdd() {
  const std::uint32_t rows = 8;
  const std::uint32_t n = 1000;
  const std::uint32_t stride = 1100;
  const auto src = t::MakeValues(rows * stride, 0x1E2E3E4EU, 1.0F);
  const auto weights = t::MakeValues(rows, 0x5F6F7F8FU, 0.5F, 0.6F);
  t::HipBuffer<float> d_src(src.size());
  t::Upload(&d_src, src);
  t::HipBuffer<float> d_weights(rows);
  t::Upload(&d_weights, weights);
  bool ok = true;
  for (unsigned variant = 0; variant < 2; ++variant) {
    const bool uses_weights = variant == 0;
    const auto base = uses_weights ? t::MakeValues(n, 0x0DA7A7A7U, 1.0F)
                                   : std::vector<float>(n, 1.0F);
    t::HipBuffer<float> d_dst(base.size());
    t::Upload(&d_dst, base);
    q::WeightedSumAdd(d_dst.get(), d_src.get(),
                      uses_weights ? d_weights.get() : nullptr, rows, n, stride,
                      1U, nullptr);
    const auto got = t::Download(&d_dst, n);
    std::vector<float> ref(n);
    for (std::uint32_t i = 0; i < n; ++i) {
      double acc = base[i];
      for (std::uint32_t r = 0; r < rows; ++r) {
        acc += (uses_weights ? weights[r] : 1.0F) *
               src[static_cast<std::size_t>(r) * stride + i];
      }
      ref[i] = static_cast<float>(acc);
    }
    ok &= Check(uses_weights ? "weighted_sum_add" : "weighted_sum_add null",
                t::WorstRelative(ref, got), 5e-5);
  }
  return ok;
}

}  // namespace

int main() {
  try {
    bool ok = true;
    ok &= TestAddBias();
    ok &= TestRmsNorm();
    ok &= TestLayerNorm();
    ok &= TestHcCoeffs();
    ok &= TestHcCombine();
    ok &= TestKdaConv();
    ok &= TestKdaConvChunkBitExact();
    ok &= TestKdaRecurrent();
    ok &= TestKdaFoldUpdateChunkBitExact();
    ok &= TestRowBatchBitExact();
    ok &= TestKdaOutGate();
    ok &= TestIdxScore();
    ok &= TestTopKDesc();
    ok &= TestMoERoute();
    ok &= TestScale();
    ok &= TestSigmoidInPlace();
    ok &= TestKdaEGate();
    ok &= TestL2NormHeads();
    ok &= TestBuildSel();
    ok &= TestSwigluClamp();
    ok &= TestWeightedSumAdd();
    t::CheckHip(hipDeviceSynchronize(), "device sync");
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "glm5_next_rocm_ops_test: %s\n", error.what());
    return 2;
  }
}
