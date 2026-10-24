// Gemma 4 attention kernels against an FP64 formula over the same binary16
// caches: window and ring addressing, the draft key limit, split-K and
// single-pass modes, batch invariance of the split mode, and image rows that
// see their whole image.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/gemma4/kernels/rocm/kernels.hpp"
#include "tests/models/gemma4/check.hpp"

namespace k = gufo::models::gemma4::rocm;
using gemma4_test::Require;

namespace {

std::uint16_t ToHalf(float x) {
  const _Float16 h = static_cast<_Float16>(x);
  std::uint16_t bits;
  std::memcpy(&bits, &h, 2);
  return bits;
}

float FromHalf(std::uint16_t bits) {
  _Float16 h;
  std::memcpy(&h, &bits, 2);
  return static_cast<float>(h);
}

struct Case {
  std::uint32_t head_dim;
  std::uint32_t heads;
  std::uint32_t kv_heads;
  std::uint32_t rows;
  std::uint32_t first_position;
  std::uint32_t window;
  std::uint32_t ring;
  bool shared_position;
  std::uint32_t key_limit;
  const char* name;
  bool timed = false;            ///< also report the launch time (deep shapes)
  std::uint32_t row_stride = 1;  ///< rows checked against FP64
  bool derived = false;          ///< keys rebuilt from V (no K cache)
  /// Rows [image_row, image_row + image_count) form one image.
  std::uint32_t image_row = 0;
  std::uint32_t image_count = 0;
};

/// A row's exclusive key end: its image's end for image rows.
std::uint32_t KeyEnd(const Case& c, std::uint32_t row) {
  const std::uint32_t position =
      c.shared_position ? c.first_position : c.first_position + row;
  if (row >= c.image_row && row < c.image_row + c.image_count) {
    return std::max(position + 1,
                    c.first_position + c.image_row + c.image_count);
  }
  return position + 1;
}

/// A derived-key case: the K cache holds the rotated dims of `pairs` pairs.
struct DerivedKeys {
  std::uint32_t pairs;
};

template<class T>
T* Device(const std::vector<T>& host) {
  T* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, host.size() * sizeof(T) + 16));
  HIP_CHECK(hipMemcpy(ptr, host.data(), host.size() * sizeof(T),
                      hipMemcpyHostToDevice));
  return ptr;
}

std::vector<float> RunKernel(const Case& c, const std::vector<float>& q,
                             const std::vector<std::uint16_t>& kc,
                             const std::vector<std::uint16_t>& vc,
                             std::uint32_t max_keys, bool timed = false,
                             const DerivedKeys* derived = nullptr) {
  float* dq = Device(q);
  auto* dk = Device(kc);
  auto* dv = Device(vc);
  std::vector<float> out(std::size_t{c.rows} * c.heads * c.head_dim, NAN);
  float* dout = Device(out);
  std::vector<float> partials(
      k::AttentionPartialFloats(c.rows, c.heads, c.head_dim, max_keys));
  float* dpart = Device(partials);
  k::AttentionArgs a{};
  a.q = dq;
  a.k_cache = dk;
  a.v_cache = dv;
  a.out = dout;
  a.partials = dpart;
  a.rows = c.rows;
  a.heads = c.heads;
  a.kv_heads = c.kv_heads;
  a.head_dim = c.head_dim;
  a.first_position = c.first_position;
  a.shared_position = c.shared_position;
  a.key_limit = c.key_limit;
  a.window = c.window;
  a.ring = c.ring;
  if (derived != nullptr) {
    a.rope_pairs = derived->pairs;
  }
  std::uint32_t* dends = nullptr;
  if (c.image_count != 0) {
    std::vector<std::uint32_t> ends(c.rows, 0);
    for (std::uint32_t r = c.image_row; r < c.image_row + c.image_count; ++r) {
      ends[r] = c.first_position + c.image_row + c.image_count;
    }
    dends = Device(ends);
    a.key_ends = dends;
  }
  k::Attention(a, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  if (timed) {
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    constexpr int kIters = 20;
    HIP_CHECK(hipEventRecord(start, nullptr));
    for (int i = 0; i < kIters; ++i) {
      k::Attention(a, nullptr);
    }
    HIP_CHECK(hipEventRecord(stop, nullptr));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    const double keys = c.window != 0 ? std::min(max_keys, c.window) : max_keys;
    const double bytes = 4.0 * keys * c.kv_heads * c.head_dim;
    const double us = 1e3 * ms / kIters;
    const double flops = 4.0 * c.rows * keys * c.heads * c.head_dim;
    std::cout << c.name << ": " << us << " us (" << bytes / us / 1e3
              << " GB/s of K/V, " << flops / us / 1e6 << " TFLOP/s)\n";
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
  }
  HIP_CHECK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));
  for (void* p : {static_cast<void*>(dq), static_cast<void*>(dk),
                  static_cast<void*>(dv), static_cast<void*>(dout),
                  static_cast<void*>(dpart), static_cast<void*>(dends)}) {
    HIP_CHECK(hipFree(p));
  }
  return out;
}

void Check(const Case& c, std::mt19937& rng) {
  const std::uint32_t D = c.head_dim;
  const std::uint32_t last =
      c.shared_position ? c.first_position : c.first_position + c.rows - 1;
  const std::uint32_t max_keys = std::min(last + 1, c.key_limit);
  const std::uint32_t slots = c.ring != 0 ? c.ring : max_keys;
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> q(std::size_t{c.rows} * c.heads * D);
  for (float& v : q)
    v = normal(rng) * 0.35F;
  std::vector<std::uint16_t> kc(std::size_t{slots} * c.kv_heads * D);
  std::vector<std::uint16_t> vc(kc.size());
  for (auto& v : kc)
    v = ToHalf(normal(rng));
  for (auto& v : vc)
    v = ToHalf(normal(rng));

  // Reference keys (FP32 copies of the binary16 cache).
  std::vector<float> kref(kc.size());
  for (std::size_t i = 0; i < kc.size(); ++i)
    kref[i] = FromHalf(kc[i]);
  // Derived keys: V is the normalized K projection, K = rope(w * V) turns
  // the first 64 pairs (Gemma 4 global rope, theta 1e6) and the K cache
  // keeps only those rotated dims; every other key dim is w * V, which the
  // producer folds into the query (q * w) so the kernel reads V.
  std::optional<DerivedKeys> derived;
  if (c.derived) {
    constexpr std::uint32_t kPairs = 64;
    std::uniform_real_distribution<float> scale(0.75F, 1.25F);
    std::vector<float> weight(D);
    for (float& w : weight)
      w = scale(rng);
    for (std::size_t row = 0; row < q.size() / D; ++row) {
      for (std::uint32_t d = 0; d < D; ++d) {
        if (d % (D / 2) >= kPairs) {
          q[row * D + d] *= weight[d];
        }
      }
    }
    kc.assign(std::size_t{slots} * c.kv_heads * 2 * kPairs, 0);
    for (std::uint32_t key = 0; key < max_keys; ++key) {
      const std::uint32_t slot = c.ring != 0 ? key % c.ring : key;
      for (std::uint32_t kvh = 0; kvh < c.kv_heads; ++kvh) {
        const std::size_t row = (std::size_t{slot} * c.kv_heads + kvh) * D;
        const std::size_t rot =
            (std::size_t{slot} * c.kv_heads + kvh) * 2 * kPairs;
        for (std::uint32_t i = 0; i < D / 2; ++i) {
          if (i >= kPairs) {
            kref[row + i] = FromHalf(vc[row + i]);
            kref[row + D / 2 + i] = FromHalf(vc[row + D / 2 + i]);
            continue;
          }
          const double x0 = double{FromHalf(vc[row + i])} * weight[i];
          const double x1 =
              double{FromHalf(vc[row + D / 2 + i])} * weight[D / 2 + i];
          const double theta =
              key * std::pow(1e6, -2.0 * i / static_cast<double>(D));
          const double cs = std::cos(theta);
          const double sn = std::sin(theta);
          kc[rot + i] = ToHalf(static_cast<float>(x0 * cs - x1 * sn));
          kc[rot + kPairs + i] = ToHalf(static_cast<float>(x0 * sn + x1 * cs));
          kref[row + i] = FromHalf(kc[rot + i]);
          kref[row + D / 2 + i] = FromHalf(kc[rot + kPairs + i]);
        }
      }
    }
    derived = DerivedKeys{kPairs};
  }
  const DerivedKeys* dk = derived ? &*derived : nullptr;

  const auto out = RunKernel(c, q, kc, vc, max_keys, c.timed, dk);
  // Repeated launches produce the same bits.
  Require(RunKernel(c, q, kc, vc, max_keys, false, dk) == out,
          std::string(c.name) + ": output differs between launches");
  double worst = 0.0;
  for (std::uint32_t r = 0; r < c.rows; r += c.row_stride) {
    const std::uint32_t pos =
        c.shared_position ? c.first_position : c.first_position + r;
    const std::uint32_t hi = std::min(KeyEnd(c, r), c.key_limit);
    const std::uint32_t lo =
        c.window != 0 && pos + 1 > c.window ? pos + 1 - c.window : 0;
    for (std::uint32_t h = 0; h < c.heads; ++h) {
      const std::uint32_t kvh = h / (c.heads / c.kv_heads);
      const float* qh = &q[(std::size_t{r} * c.heads + h) * D];
      std::vector<double> scores;
      double m = -INFINITY;
      for (std::uint32_t key = lo; key < hi; ++key) {
        const std::uint32_t slot = c.ring != 0 ? key % c.ring : key;
        double s = 0.0;
        for (std::uint32_t d = 0; d < D; ++d) {
          s += qh[d] *
               double{kref[(std::size_t{slot} * c.kv_heads + kvh) * D + d]};
        }
        scores.push_back(s);
        m = std::max(m, s);
      }
      double total = 0.0;
      for (double& s : scores) {
        s = std::exp(s - m);
        total += s;
      }
      for (std::uint32_t d = 0; d < D; ++d) {
        double acc = 0.0;
        for (std::uint32_t key = lo; key < hi; ++key) {
          const std::uint32_t slot = c.ring != 0 ? key % c.ring : key;
          acc += scores[key - lo] *
                 FromHalf(vc[(std::size_t{slot} * c.kv_heads + kvh) * D + d]);
        }
        const double want = acc / total;
        const double got = out[(std::size_t{r} * c.heads + h) * D + d];
        Require(std::isfinite(got),
                std::string(c.name) + ": non-finite output");
        worst = std::max(worst, std::fabs(got - want));
      }
    }
  }
  // Split mode accumulates in FP32; prefill runs binary16 WMMA operands
  // (queries and probabilities rounded to binary16).
  // With O(1) values, binary16 Q and P give ~1e-3 relative error per term.
  const double limit = c.rows <= k::kSplitRows ? 2e-5 : 1e-2;
  std::cout << c.name << ": max error " << worst << '\n';
  Require(worst < limit,
          std::string(c.name) + ": max error " + std::to_string(worst));

  // Split mode: every row equals its single-row evaluation bit for bit.
  if (c.rows > 1 && c.rows <= k::kSplitRows && !c.shared_position &&
      c.image_count == 0) {
    for (std::uint32_t r = 0; r < c.rows; ++r) {
      Case single = c;
      single.rows = 1;
      single.first_position = c.first_position + r;
      std::vector<float> q1(q.begin() + std::size_t{r} * c.heads * D,
                            q.begin() + std::size_t{r + 1} * c.heads * D);
      const auto one = RunKernel(single, q1, kc, vc, max_keys, false, dk);
      Require(std::memcmp(one.data(), &out[std::size_t{r} * c.heads * D],
                          one.size() * 4) == 0,
              std::string(c.name) + ": row " + std::to_string(r) +
                  " depends on its batch");
    }
  }
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cout << "SKIP: no HIP device\n";
    return 77;
  }
  return gemma4_test::Run([] {
    std::mt19937 rng(20260926);
    constexpr std::uint32_t kNoLimit =
        std::numeric_limits<std::uint32_t>::max();
    const Case cases[] = {
        {256, 32, 16, 1, 0, 1024, 3072, false, kNoLimit, "sliding first token"},
        {256, 32, 16, 1, 1023, 1024, 3072, false, kNoLimit,
         "sliding full window"},
        {256, 32, 16, 1, 5000, 1024, 3072, false, kNoLimit,
         "sliding ring wrap"},
        {256, 8, 4, 5, 1020, 1024, 1280, false, kNoLimit,
         "sliding verify window edge"},
        {256, 8, 4, 8, 3070, 1024, 1280, false, kNoLimit,
         "sliding verify wrap"},
        {256, 8, 4, 6, 2000, 1024, 1280, false, kNoLimit,
         "sliding verify six rows"},
        {256, 8, 4, 40, 1000, 1024, 1280, false, kNoLimit, "sliding prefill"},
        {256, 8, 4, 70, 3000, 1024, 1280, false, kNoLimit,
         "sliding prefill ring wrap"},
        {256, 32, 16, 37, 2, 1024, 3072, false, kNoLimit,
         "sliding prefill start"},
        // Image rows attend forward to the end of their image.
        {256, 8, 4, 300, 900, 1024, 1536, false, kNoLimit,
         "sliding prefill image across the window", false, 1, false, 20, 280},
        {256, 8, 4, 70, 3000, 1024, 1280, false, kNoLimit,
         "sliding prefill image ring wrap", false, 1, false, 10, 50},
        {256, 32, 16, 40, 0, 1024, 3072, false, kNoLimit,
         "sliding prefill image at start", false, 1, false, 0, 40},
        {256, 8, 4, 12, 2000, 1024, 1280, false, kNoLimit,
         "sliding split image", false, 1, false, 3, 7},
        {512, 32, 4, 1, 0, 0, 0, false, kNoLimit, "global first token"},
        {512, 32, 4, 1, 2999, 0, 0, false, kNoLimit, "global multi-split"},
        {512, 8, 1, 7, 1500, 0, 0, false, kNoLimit, "global verify"},
        {512, 8, 1, 33, 600, 0, 0, false, kNoLimit, "global prefill"},
        {512, 32, 4, 47, 0, 0, 0, false, kNoLimit, "global prefill from zero"},
        {512, 8, 1, 3, 1500, 0, 0, true, 1500, "draft global frontier"},
        {256, 8, 4, 3, 2000, 1024, 3072, true, 2000, "draft sliding frontier"},
        // Production global layers derive K from V.
        {512, 32, 4, 1, 2999, 0, 0, false, kNoLimit,
         "derived global multi-split", false, 1, true},
        {512, 8, 1, 7, 1500, 0, 0, false, kNoLimit, "derived global verify",
         false, 1, true},
        {512, 8, 1, 8, 2100, 0, 0, false, kNoLimit,
         "derived global verify eight rows", false, 1, true},
        {512, 8, 1, 33, 600, 0, 0, false, kNoLimit, "derived global prefill",
         false, 1, true},
        {512, 32, 4, 47, 0, 0, 0, false, kNoLimit,
         "derived global prefill from zero", false, 1, true},
        {512, 8, 1, 3, 1500, 0, 0, true, 1500, "derived draft global frontier",
         false, 1, true},
        {512, 32, 4, 1, 32767, 0, 0, false, kNoLimit, "global decode 32K", true,
         1, true},
        {512, 32, 4, 5, 32763, 0, 0, false, kNoLimit, "global verify 32K", true,
         1, true},
        {256, 32, 16, 1, 32767, 1024, 3072, false, kNoLimit,
         "sliding decode 32K", true},
        {256, 32, 16, 5, 32763, 1024, 3072, false, kNoLimit,
         "sliding verify 32K", true},
        {512, 32, 4, 512, 32768, 0, 0, false, kNoLimit, "global prefill 32K",
         true, 97, true},
        {512, 32, 4, 2048, 32768, 0, 0, false, kNoLimit,
         "global prefill 2048 rows at 32K", true, 389, true},
        {256, 32, 16, 512, 32768, 1024, 1536, false, kNoLimit,
         "sliding prefill 32K", true, 97},
        // The 26B-A4B's sliding layers: 16 query heads over 8 KV heads.
        {256, 16, 8, 2048, 32768, 1024, 3072, false, kNoLimit,
         "26B sliding prefill 2048 rows at 32K", true, 389},
        // The 26B-A4B's global layers: 16 query heads over 2 KV heads.
        {512, 16, 2, 2048, 32768, 0, 0, false, kNoLimit,
         "26B global prefill 2048 rows at 32K", true, 389, true},
    };
    for (const Case& c : cases) {
      Check(c, rng);
    }
  });
}
