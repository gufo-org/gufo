#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace {

// The model's routing: 512 experts, top-10 without replacement.
constexpr int kExperts = 512;
constexpr int kUsed = 10;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

/// Top-k ids per token without replacement, skewed toward low experts so
/// the buckets are uneven; every 50th token drops one slot (-1).
std::vector<std::int32_t> MakeIds(int n_tokens, std::uint32_t seed) {
  std::vector<std::int32_t> ids(static_cast<std::size_t>(n_tokens) * kUsed);
  for (int t = 0; t < n_tokens; ++t) {
    std::vector<std::int32_t> chosen;
    while (chosen.size() < kUsed) {
      const std::uint32_t r = NextRandom(&seed);
      const int e = static_cast<int>((r % 4 == 0 ? r / 4 : r / 8) % kExperts);
      if (std::find(chosen.begin(), chosen.end(), e) == chosen.end()) {
        chosen.push_back(e);
      }
    }
    if (t % 50 == 7) {
      chosen[3] = -1;
    }
    std::copy(chosen.begin(), chosen.end(),
              ids.begin() + static_cast<std::size_t>(t) * kUsed);
  }
  return ids;
}

struct Maps {
  std::vector<std::int32_t> src1;
  std::vector<std::int32_t> dst;
  std::vector<std::int32_t> bounds;
};

/// llama.cpp's mm_ids_helper contract on the host: rows sorted by expert
/// then assignment order; a negative id is counted below expert 0 and its
/// slot left zero; an id past the experts is dropped.
Maps Reference(const std::vector<std::int32_t>& ids, int n_tokens, int n_used,
               int si1, int sis1) {
  Maps m;
  const std::size_t rows = static_cast<std::size_t>(n_tokens) * n_used;
  m.src1.assign(rows, 0);
  m.dst.assign(rows, 0);
  m.bounds.assign(kExperts + 1, 0);
  std::vector<int> counts(kExperts, 0);
  int negatives = 0;
  for (int t = 0; t < n_tokens; ++t) {
    for (int i = 0; i < n_used; ++i) {
      const std::int32_t e = ids[static_cast<std::size_t>(t) * si1 + i];
      if (e < 0) {
        ++negatives;
      } else if (e < kExperts) {
        ++counts[e];
      }
    }
  }
  int running = negatives;
  for (int e = 0; e < kExperts; ++e) {
    m.bounds[e] = running;
    running += counts[e];
  }
  m.bounds[kExperts] = running;
  std::vector<int> cursor(m.bounds.begin(), m.bounds.end() - 1);
  for (int t = 0; t < n_tokens; ++t) {
    for (int i = 0; i < n_used; ++i) {
      const std::int32_t e = ids[static_cast<std::size_t>(t) * si1 + i];
      if (e < 0 || e >= kExperts) {
        continue;
      }
      const int pos = cursor[e]++;
      m.src1[pos] = t * sis1 + i % 1;
      m.dst[pos] = t * n_used + i;
    }
  }
  return m;
}

Maps Build(const std::vector<std::int32_t>& ids, int n_tokens, int n_used,
           int si1, int sis1) {
  const std::size_t rows = static_cast<std::size_t>(n_tokens) * n_used;
  std::int32_t* d_ids = nullptr;
  std::int32_t* d_src1 = nullptr;
  std::int32_t* d_dst = nullptr;
  std::int32_t* d_bounds = nullptr;
  CheckHip(hipMalloc(&d_ids, ids.size() * 4), "hipMalloc");
  CheckHip(hipMalloc(&d_src1, rows * 4), "hipMalloc");
  CheckHip(hipMalloc(&d_dst, rows * 4), "hipMalloc");
  CheckHip(hipMalloc(&d_bounds, (kExperts + 1) * 4), "hipMalloc");
  CheckHip(hipMemcpy(d_ids, ids.data(), ids.size() * 4, hipMemcpyHostToDevice),
           "upload");
  CheckHip(hipMemset(d_src1, 0, rows * 4), "memset");
  CheckHip(hipMemset(d_dst, 0, rows * 4), "memset");
  if (qfn_mmq_build_ids_maps(d_ids, d_src1, d_dst, d_bounds, kExperts, n_tokens,
                             n_used, 1, si1, sis1, nullptr) != 0) {
    throw std::runtime_error("id map build failed");
  }
  CheckHip(hipDeviceSynchronize(), "id maps");
  Maps m;
  m.src1.resize(rows);
  m.dst.resize(rows);
  m.bounds.resize(kExperts + 1);
  CheckHip(hipMemcpy(m.src1.data(), d_src1, rows * 4, hipMemcpyDeviceToHost),
           "download");
  CheckHip(hipMemcpy(m.dst.data(), d_dst, rows * 4, hipMemcpyDeviceToHost),
           "download");
  CheckHip(hipMemcpy(m.bounds.data(), d_bounds, (kExperts + 1) * 4,
                     hipMemcpyDeviceToHost),
           "download");
  (void)hipFree(d_ids);
  (void)hipFree(d_src1);
  (void)hipFree(d_dst);
  (void)hipFree(d_bounds);
  return m;
}

bool Same(const Maps& a, const Maps& b, const char* what) {
  const bool ok = a.src1 == b.src1 && a.dst == b.dst && a.bounds == b.bounds;
  std::cout << what << ": " << (ok ? "identical" : "DIFFER") << " ("
            << a.bounds.back() << " routed rows)\n";
  if (!ok) {
    for (std::size_t i = 0; i < a.bounds.size(); ++i) {
      if (a.bounds[i] != b.bounds[i]) {
        std::cout << "  first bounds mismatch at " << i << ": " << a.bounds[i]
                  << " vs " << b.bounds[i] << '\n';
        break;
      }
    }
    for (std::size_t i = 0; i < a.src1.size(); ++i) {
      if (a.src1[i] != b.src1[i] || a.dst[i] != b.dst[i]) {
        std::cout << "  first map mismatch at " << i << ": src1 " << a.src1[i]
                  << " vs " << b.src1[i] << ", dst " << a.dst[i] << " vs "
                  << b.dst[i] << '\n';
        break;
      }
    }
  }
  return ok;
}

// Check the router independently of the assignment-map builder: softmax,
// selection without replacement, lowest-index ties, and renormalization.
void CheckRouter(int tokens, int experts, int used, int pattern) {
  namespace q = gufo::models::qwen38_flash_next::rocm;
  const int stride = experts + 1;
  const std::size_t count = static_cast<std::size_t>(tokens) * used;
  constexpr std::int32_t kIdGuard = -771;
  constexpr float kWeightGuard = -71.0F;
  std::vector<float> logits(static_cast<std::size_t>(tokens) * stride,
                            std::numeric_limits<float>::quiet_NaN());
  std::uint32_t random = 0xBADC0FFEU;
  for (int t = 0; t < tokens; ++t) {
    for (int e = 0; e < experts; ++e) {
      const auto bits = NextRandom(&random);
      float value =
          static_cast<float>(static_cast<int>(bits % 16384) - 8192) / 1024.0F;
      if (pattern == 1) {
        value = static_cast<float>(bits % 8);
      } else if (pattern == 2) {
        value = 0.0F;
      } else if (pattern == 3 && e % 3 == 0) {
        value = -std::numeric_limits<float>::infinity();
      }
      logits[static_cast<std::size_t>(t) * stride + e] = value;
    }
  }
  float* d_logits = nullptr;
  float* d_weights = nullptr;
  std::int32_t* d_ids = nullptr;
  CheckHip(hipMalloc(&d_logits, logits.size() * sizeof(float)),
           "router logits");
  CheckHip(hipMalloc(&d_weights, (count + 2) * sizeof(float)),
           "router weights");
  CheckHip(hipMalloc(&d_ids, (count + 2) * sizeof(std::int32_t)), "router ids");
  const std::vector<float> weight_guards(count + 2, kWeightGuard);
  const std::vector<std::int32_t> id_guards(count + 2, kIdGuard);
  CheckHip(hipMemcpy(d_logits, logits.data(), logits.size() * sizeof(float),
                     hipMemcpyHostToDevice),
           "upload logits");
  CheckHip(
      hipMemcpy(d_weights, weight_guards.data(),
                weight_guards.size() * sizeof(float), hipMemcpyHostToDevice),
      "upload weight guards");
  CheckHip(
      hipMemcpy(d_ids, id_guards.data(),
                id_guards.size() * sizeof(std::int32_t), hipMemcpyHostToDevice),
      "upload id guards");
  std::vector<float> first_weights;
  std::vector<std::int32_t> first_ids;
  for (int replay = 0; replay < 2; ++replay) {
    q::RouterTopK(d_logits, stride, d_ids + 1, d_weights + 1, tokens, experts,
                  used, nullptr);
    CheckHip(hipDeviceSynchronize(), "router");
    std::vector<float> weights(count + 2);
    std::vector<std::int32_t> ids(count + 2);
    CheckHip(hipMemcpy(weights.data(), d_weights,
                       weights.size() * sizeof(float), hipMemcpyDeviceToHost),
             "download router weights");
    CheckHip(hipMemcpy(ids.data(), d_ids, ids.size() * sizeof(std::int32_t),
                       hipMemcpyDeviceToHost),
             "download router ids");
    if (weights.front() != kWeightGuard || weights.back() != kWeightGuard ||
        ids.front() != kIdGuard || ids.back() != kIdGuard) {
      throw std::runtime_error("router overwrote output guards");
    }
    if (replay != 0) {
      if (ids != first_ids ||
          !std::equal(weights.begin(), weights.end(), first_weights.begin(),
                      [](float a, float b) {
                        return std::bit_cast<std::uint32_t>(a) ==
                               std::bit_cast<std::uint32_t>(b);
                      })) {
        throw std::runtime_error("router replay differs");
      }
      continue;
    }
    first_weights = weights;
    first_ids = ids;
    for (int t = 0; t < tokens; ++t) {
      const float* row = logits.data() + static_cast<std::size_t>(t) * stride;
      const float maximum = *std::max_element(row, row + experts);
      std::vector<double> probabilities(experts);
      double denominator = 0.0;
      for (int e = 0; e < experts; ++e) {
        probabilities[e] = std::exp(static_cast<double>(row[e]) - maximum);
        denominator += probabilities[e];
      }
      for (double& probability : probabilities) {
        probability /= denominator;
      }
      std::vector<int> order(experts);
      std::iota(order.begin(), order.end(), 0);
      std::partial_sort(order.begin(), order.begin() + used, order.end(),
                        [&](int a, int b) {
                          return probabilities[a] == probabilities[b]
                                     ? a < b
                                     : probabilities[a] > probabilities[b];
                        });
      double sum = 0.0;
      for (int i = 0; i < used; ++i) {
        sum += probabilities[order[i]];
      }
      sum = std::max(sum, 6.103515625e-5);
      for (int i = 0; i < used; ++i) {
        const auto at = 1 + static_cast<std::size_t>(t) * used + i;
        const double expected = probabilities[order[i]] / sum;
        if (ids[at] != order[i] || !std::isfinite(weights[at]) ||
            std::abs(weights[at] - expected) > 2e-6) {
          throw std::runtime_error("router differs from softmax reference");
        }
      }
    }
  }
  (void)hipFree(d_logits);
  (void)hipFree(d_weights);
  (void)hipFree(d_ids);
}

/// A decode pass's router as one launch (RouterFusedKernel) against SmallGemm
/// and RouterTopK: every logit, id and weight byte, twice on the same counters.
template<typename Weight>
void CheckRouterFused(gufo::models::qwen38_flash_next::rocm::WeightType type,
                      std::uint32_t tokens) {
  namespace q = gufo::models::qwen38_flash_next::rocm;
  constexpr std::uint32_t kRows = 513;
  constexpr std::uint32_t kCols = 2560;
  constexpr std::uint32_t kExperts = 512;
  constexpr std::uint32_t kTop = 10;
  std::uint32_t seed = 0x70C7E400U + tokens;
  const auto next = [&] {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return static_cast<float>(static_cast<int>(seed & 0xFFFFU) - 32768) /
           32768.0F;
  };
  std::vector<Weight> w(std::size_t{kRows} * kCols);
  for (auto& value : w)
    value = static_cast<Weight>(0.05F * next());
  std::vector<float> x(std::size_t{tokens} * kCols);
  for (auto& value : x)
    value = 2.0F * next();
  void* d_w = nullptr;
  float* d_x = nullptr;
  float* d_logits[2] = {};
  float* d_weights[2] = {};
  std::int32_t* d_ids[2] = {};
  void* state = nullptr;
  const std::size_t logit_bytes = std::size_t{tokens} * kRows * sizeof(float);
  const std::size_t slot_count = std::size_t{tokens} * kTop;
  CheckHip(hipMalloc(&d_w, w.size() * sizeof(Weight)), "router weights");
  CheckHip(hipMalloc(&d_x, x.size() * sizeof(float)), "router input");
  CheckHip(hipMalloc(&state, q::RouterFusedStateBytes()), "router state");
  CheckHip(hipMemset(state, 0, q::RouterFusedStateBytes()), "router state");
  for (int route = 0; route < 2; ++route) {
    CheckHip(hipMalloc(&d_logits[route], logit_bytes), "router logits");
    CheckHip(hipMalloc(&d_weights[route], slot_count * sizeof(float)),
             "router weights");
    CheckHip(hipMalloc(&d_ids[route], slot_count * sizeof(std::int32_t)),
             "router ids");
  }
  CheckHip(hipMemcpy(d_w, w.data(), w.size() * sizeof(Weight),
                     hipMemcpyHostToDevice),
           "upload router weights");
  CheckHip(
      hipMemcpy(d_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice),
      "upload router input");
  q::SmallGemm(d_w, type, d_x, d_logits[0], tokens, kRows, kCols, nullptr);
  q::RouterTopK(d_logits[0], kRows, d_ids[0], d_weights[0], tokens, kExperts,
                kTop, nullptr);
  CheckHip(hipDeviceSynchronize(), "separate router");
  const auto download = [&](int route) {
    std::vector<std::uint8_t> bytes(logit_bytes + slot_count * 8);
    CheckHip(hipMemcpy(bytes.data(), d_logits[route], logit_bytes,
                       hipMemcpyDeviceToHost),
             "download logits");
    CheckHip(hipMemcpy(bytes.data() + logit_bytes, d_weights[route],
                       slot_count * 4, hipMemcpyDeviceToHost),
             "download weights");
    CheckHip(hipMemcpy(bytes.data() + logit_bytes + slot_count * 4,
                       d_ids[route], slot_count * 4, hipMemcpyDeviceToHost),
             "download ids");
    return bytes;
  };
  const auto reference = download(0);
  for (int replay = 0; replay < 2; ++replay) {
    CheckHip(hipMemset(d_logits[1], 0xFF, logit_bytes), "clear logits");
    CheckHip(hipMemset(d_ids[1], 0xFF, slot_count * 4), "clear ids");
    if (!q::RouterFused(d_w, type, d_x, d_logits[1], d_ids[1], d_weights[1],
                        tokens, kRows, kCols, kExperts, kTop, state, nullptr))
      throw std::runtime_error("fused router declined a decode shape");
    CheckHip(hipDeviceSynchronize(), "fused router");
    if (download(1) != reference)
      throw std::runtime_error("fused router differs at " +
                               std::to_string(tokens) + " rows");
  }
  for (int route = 0; route < 2; ++route) {
    (void)hipFree(d_logits[route]);
    (void)hipFree(d_weights[route]);
    (void)hipFree(d_ids[route]);
  }
  (void)hipFree(state);
  (void)hipFree(d_x);
  (void)hipFree(d_w);
}

}  // namespace

int main() {
  try {
    for (std::uint32_t tokens = 1; tokens <= 8; ++tokens) {
      namespace q = gufo::models::qwen38_flash_next::rocm;
      CheckRouterFused<float>(q::WeightType::kF32, tokens);
      CheckRouterFused<_Float16>(q::WeightType::kF16, tokens);
    }
    std::cout << "Fused router against separate launches: passed\n";
    for (int pattern : {0, 1, 2, 3}) {
      CheckRouter(1, 512, 10, pattern);
      CheckRouter(8, 512, 10, pattern);
      CheckRouter(2048, 512, 10, pattern);
    }
    CheckRouter(17, 8, 8, 1);
    CheckRouter(17, 513, 32, 0);
    CheckRouter(17, 1024, 32, 1);
    CheckRouter(17, 1024, 1, 3);
    std::cout << "Router softmax, ties, guards and replay: passed\n";
    bool ok = true;
    for (int n_tokens : {37, 512, 2048}) {
      const auto ids = MakeIds(n_tokens, 0x1234ABCDU + n_tokens);
      // Gate/up: [n_tokens][kUsed] ids, one activation row per token.
      const Maps pair_ref = Reference(ids, n_tokens, kUsed, kUsed, 1);
      const Maps pair_scan = Build(ids, n_tokens, kUsed, kUsed, 1);
      ok = Same(pair_ref, pair_scan,
                (std::string("gate/up maps, ") + std::to_string(n_tokens) +
                 " tokens")
                    .c_str()) &&
           ok;
      // Down: every (token, slot) row is its own single-expert token.
      const Maps down_ref = Reference(ids, n_tokens * kUsed, 1, 1, 1);
      const Maps down_scan = Build(ids, n_tokens * kUsed, 1, 1, 1);
      ok = Same(down_ref, down_scan,
                (std::string("down maps, ") + std::to_string(n_tokens) +
                 " tokens")
                    .c_str()) &&
           ok;
    }
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
