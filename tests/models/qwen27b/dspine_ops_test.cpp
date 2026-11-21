// DSpine kernels (dflash_kernels.hip) against CPU transcriptions of Tencent's
// reference, sglang_dspine/kernels/dspine.py at 321b0fc ("dspine.py"), at the
// Qwen3.8-27B-DSpine-b16 shapes. FP32 ops: max |gpu - ref| / max |ref| <=
// 1e-5 against a double-precision reference. IDs and chain tokens: exact, with
// planted ties (the lower ID wins).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "src/families/qwen/kernels/dflash_kernels.hpp"

namespace {

using namespace gufo::hip::kernels;

constexpr std::uint32_t kHidden = 5120, kCode = 1024, kRank = 512, kTopK = 16;
constexpr std::uint32_t kVocab = 248320, kRows = 16, kProposals = kRows - 1;
constexpr float kEps = 1e-6F;
constexpr double kTolerance = 1e-5;
int failures = 0;

void Check(hipError_t error, int line) {
  if (error != hipSuccess) {
    std::printf("HIP error %s at line %d\n", hipGetErrorString(error), line);
    std::exit(1);
  }
}
#define HIP_OK(call) Check((call), __LINE__)

template<typename T>
T* Upload(const std::vector<T>& host) {
  T* device = nullptr;
  HIP_OK(hipMalloc(&device, std::max<std::size_t>(1, host.size()) * sizeof(T)));
  HIP_OK(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                   hipMemcpyHostToDevice));
  return device;
}

template<typename T>
std::vector<T> Download(const T* device, std::size_t count) {
  std::vector<T> host(count);
  HIP_OK(hipDeviceSynchronize());
  HIP_OK(
      hipMemcpy(host.data(), device, count * sizeof(T), hipMemcpyDeviceToHost));
  return host;
}

std::vector<float> Random(std::size_t count, std::mt19937& rng,
                          float sigma = 1.0F) {
  std::normal_distribution<float> normal(0.0F, sigma);
  std::vector<float> values(count);
  for (auto& value : values)
    value = normal(rng);
  return values;
}

/// max |got - want| / max |want|.
double RelErr(const std::vector<float>& got, const std::vector<double>& want) {
  double worst = 0.0, scale = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    worst = std::max(worst, std::abs(double(got[i]) - want[i]));
    scale = std::max(scale, std::abs(want[i]));
  }
  return worst / std::max(scale, 1e-30);
}

void Expect(bool ok, const char* what, double value) {
  std::printf("%-44s %-4s %.3g\n", what, ok ? "ok" : "FAIL", value);
  failures += ok ? 0 : 1;
}

double Sigmoid(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}

// dspine.py:12-32 prepare_read: rms = sqrt(max(sum(x^2) / H, FLT_MIN)),
// normalized = x / (rms + EPS).
void TestRowStats(std::mt19937& rng) {
  const auto hidden = Random(kRows * kHidden, rng, 0.7F);
  std::vector<double> want_norm(hidden.size()), want_rms(kRows);
  for (std::uint32_t r = 0; r < kRows; ++r) {
    double sum = 0.0;
    for (std::uint32_t i = 0; i < kHidden; ++i)
      sum += double(hidden[r * kHidden + i]) * hidden[r * kHidden + i];
    want_rms[r] = std::sqrt(std::max(sum / kHidden, double(FLT_MIN)));
    for (std::uint32_t i = 0; i < kHidden; ++i)
      want_norm[r * kHidden + i] =
          hidden[r * kHidden + i] / (want_rms[r] + kEps);
  }
  auto* d_hidden = Upload(hidden);
  auto* d_norm = Upload(std::vector<float>(hidden.size()));
  auto* d_rms = Upload(std::vector<float>(kRows));
  LaunchDSpineRowStats(d_hidden, d_norm, d_rms, kRows, kHidden, kEps, nullptr);
  const double e1 = RelErr(Download(d_norm, hidden.size()), want_norm);
  const double e2 = RelErr(Download(d_rms, kRows), want_rms);
  Expect(e1 <= kTolerance, "row stats: normalized (dspine.py:31)", e1);
  Expect(e2 <= kTolerance, "row stats: rms (dspine.py:29)", e2);
  HIP_OK(hipFree(d_hidden));
  HIP_OK(hipFree(d_norm));
  HIP_OK(hipFree(d_rms));
}

// dspine.py:35-57 normalize_codes: the anchor row is sqrt(C) * codebook row,
// other rows sqrt(C) * z / (|z| + EPS).
void TestCodes(std::mt19937& rng) {
  const auto codes = Random(kRows * kCode, rng, 0.3F);
  const auto anchor = Random(kCode, rng, 0.05F);
  std::vector<double> want(codes.size());
  const double scale = std::sqrt(double(kCode));
  for (std::uint32_t i = 0; i < kCode; ++i)
    want[i] = scale * anchor[i];
  for (std::uint32_t r = 1; r < kRows; ++r) {
    double sum = 0.0;
    for (std::uint32_t i = 0; i < kCode; ++i)
      sum += double(codes[r * kCode + i]) * codes[r * kCode + i];
    for (std::uint32_t i = 0; i < kCode; ++i)
      want[r * kCode + i] =
          scale * codes[r * kCode + i] / (std::sqrt(sum) + kEps);
  }
  auto* d_codes = Upload(codes);
  auto* d_anchor = Upload(anchor);
  LaunchDSpineCodes(d_codes, d_anchor, kRows, kCode, kEps, nullptr);
  const double e = RelErr(Download(d_codes, codes.size()), want);
  Expect(e <= kTolerance, "codes (dspine.py:49-56)", e);
  HIP_OK(hipFree(d_codes));
  HIP_OK(hipFree(d_anchor));
}

// dspine.py:60-73 feature_message (bridge) and :571-579 (refinement:
// exact = sqrt(C) * codebook[sid], gate = sigmoid(q + exact P^T + b),
// message = gate * exact A^T).
void TestMessage(std::mt19937& rng) {
  const auto packed = Random(kRows * 3 * kRank, rng);
  const auto bias = Random(kRank, rng, 0.5F);
  std::vector<double> want(kRows * kRank, 0.0);
  for (std::uint32_t r = 1; r < kRows; ++r)
    for (std::uint32_t k = 0; k < kRank; ++k) {
      const double query = packed[r * 3 * kRank + k];
      const double gate = packed[(r - 1) * 3 * kRank + kRank + k];
      const double value = packed[(r - 1) * 3 * kRank + 2 * kRank + k];
      want[r * kRank + k] = Sigmoid(query + gate + bias[k]) * value;
    }
  auto* d_packed = Upload(packed);
  auto* d_bias = Upload(bias);
  auto* d_message = Upload(std::vector<float>(kProposals * kTopK * kRank));
  LaunchDSpineMessage(d_packed, nullptr, d_bias, d_message, kRows, kRank, 0,
                      1.0F, nullptr);
  double e = RelErr(Download(d_message, kRows * kRank), want);
  Expect(e <= kTolerance, "message, bridge (dspine.py:66-72)", e);

  const std::uint32_t pairs = kProposals * kTopK;
  const auto sender = Random(pairs * 2 * kRank, rng, 0.03F);
  const double scale = std::sqrt(double(kCode));
  want.assign(pairs * kRank, 0.0);
  for (std::uint32_t r = 0; r < pairs; ++r)
    for (std::uint32_t k = 0; k < kRank; ++k) {
      const double query = packed[(r / kTopK + 1) * 3 * kRank + k];
      const double gate = scale * sender[r * 2 * kRank + k];
      const double value = scale * sender[r * 2 * kRank + kRank + k];
      want[r * kRank + k] = Sigmoid(query + gate + bias[k]) * value;
    }
  auto* d_sender = Upload(sender);
  LaunchDSpineMessage(d_packed, d_sender, d_bias, d_message, pairs, kRank,
                      kTopK, float(scale), nullptr);
  e = RelErr(Download(d_message, pairs * kRank), want);
  Expect(e <= kTolerance, "message, refinement (dspine.py:574-579)", e);
  for (auto* pointer : {d_packed, d_bias, d_message, d_sender})
    HIP_OK(hipFree(pointer));
}

// dspine.py:76-108 feature_finish: raw + rms * delta, nothing on the anchor.
void TestWrite(std::mt19937& rng) {
  const auto hidden = Random(kRows * kHidden, rng);
  const auto delta = Random(kRows * kHidden, rng, 0.1F);
  const auto rms = Random(kRows, rng);
  std::vector<double> want(hidden.size());
  for (std::uint32_t r = 0; r < kRows; ++r)
    for (std::uint32_t i = 0; i < kHidden; ++i) {
      const auto at = r * kHidden + i;
      want[at] = r == 0 ? hidden[at] : hidden[at] + double(rms[r]) * delta[at];
    }
  auto* d_hidden = Upload(hidden);
  auto* d_delta = Upload(delta);
  auto* d_rms = Upload(rms);
  LaunchDSpineWrite(d_hidden, d_delta, d_rms, d_hidden, kRows, kHidden,
                    nullptr);
  const double e = RelErr(Download(d_hidden, hidden.size()), want);
  Expect(e <= kTolerance, "write, in place (dspine.py:98-101)", e);
  for (auto* pointer : {d_hidden, d_delta, d_rms})
    HIP_OK(hipFree(pointer));
}

// dspine.py:111-208 topk_ids: the k largest, ascending by ID; a tie at the
// k-th place goes to the lower ID.
void TestTopK(std::mt19937& rng) {
  auto logits = Random(std::size_t{kProposals} * kVocab, rng);
  for (std::uint32_t r = 0; r < kProposals; ++r) {
    float* row = logits.data() + std::size_t{r} * kVocab;
    // 24 equal values above every other logit: only the 16 lowest IDs fit.
    for (std::uint32_t i = 0; i < 24; ++i)
      row[(i * 10007U + r * 131U) % kVocab] = 9.0F;
    // An exact tie at a distinct lower level in another row pattern.
    if (r % 2 == 1)
      for (std::uint32_t i = 0; i < 24; ++i)
        row[(i * 10007U + r * 131U) % kVocab] = 9.0F - 0.5F * float(i % 3);
  }
  const std::uint32_t anchor = 4242;
  std::vector<std::uint32_t> want((kProposals + 1) * kTopK, anchor);
  for (std::uint32_t r = 0; r < kProposals; ++r) {
    const float* row = logits.data() + std::size_t{r} * kVocab;
    std::vector<std::uint32_t> order(kVocab);
    std::iota(order.begin(), order.end(), 0U);
    std::partial_sort(order.begin(), order.begin() + kTopK, order.end(),
                      [&](std::uint32_t a, std::uint32_t b) {
                        return row[a] > row[b] || (row[a] == row[b] && a < b);
                      });
    std::sort(order.begin(), order.begin() + kTopK);
    std::copy_n(order.begin(), kTopK, want.begin() + (r + 1) * kTopK);
  }
  auto* d_logits = Upload(logits);
  auto* d_anchor = Upload(std::vector<std::uint32_t>{anchor});
  auto* d_ids = Upload(std::vector<std::uint32_t>(want.size(), 0U));
  LaunchDSpineTopK(d_logits, kProposals, kVocab, kTopK, d_anchor, d_ids,
                   nullptr);
  const auto got = Download(d_ids, want.size());
  std::size_t wrong = 0;
  for (std::size_t i = 0; i < want.size(); ++i)
    wrong += got[i] != want[i];
  Expect(wrong == 0, "top-k IDs incl. ties (dspine.py:165-171)", double(wrong));
  for (auto* pointer :
       {static_cast<void*>(d_logits), static_cast<void*>(d_anchor),
        static_cast<void*>(d_ids)})
    HIP_OK(hipFree(pointer));
}

// dspine.py:575-585 serial_rs_table: refined = x + scale * (message W^T),
// normalized = refined / sqrt(mean(refined^2) + eps) * norm_w, table =
// normalized . head[cand].
void TestTable(std::mt19937& rng) {
  const std::uint32_t pairs = kProposals * kTopK;
  const auto raw = Random(kRows * kHidden, rng);
  const auto rms = Random(kRows, rng);
  const auto delta = Random(std::size_t{pairs} * kHidden, rng, 0.2F);
  const auto norm = Random(kHidden, rng, 0.5F);
  const auto heads = Random(std::size_t{pairs} * kHidden, rng, 0.02F);
  std::vector<double> want(pairs * kTopK);
  std::vector<double> refined(kHidden);
  for (std::uint32_t pair = 0; pair < pairs; ++pair) {
    const std::uint32_t t = pair / kTopK;
    double sum = 0.0;
    for (std::uint32_t i = 0; i < kHidden; ++i) {
      refined[i] = raw[(t + 1) * kHidden + i] +
                   double(rms[t + 1]) * delta[std::size_t{pair} * kHidden + i];
      sum += refined[i] * refined[i];
    }
    const double inverse = 1.0 / std::sqrt(sum / kHidden + kEps);
    for (std::uint32_t k = 0; k < kTopK; ++k) {
      double dot = 0.0;
      for (std::uint32_t i = 0; i < kHidden; ++i)
        dot += refined[i] * inverse * norm[i] *
               heads[(std::size_t{t} * kTopK + k) * kHidden + i];
      want[pair * kTopK + k] = dot;
    }
  }
  auto* d_raw = Upload(raw);
  auto* d_rms = Upload(rms);
  auto* d_delta = Upload(delta);
  auto* d_norm = Upload(norm);
  auto* d_heads = Upload(heads);
  auto* d_table = Upload(std::vector<float>(want.size()));
  LaunchDSpineTable(d_raw, d_rms, d_delta, d_norm, d_heads, d_table, kProposals,
                    kTopK, kHidden, kEps, nullptr);
  const double e = RelErr(Download(d_table, want.size()), want);
  Expect(e <= kTolerance, "table (dspine.py:580-585)", e);
  for (auto* pointer : {d_raw, d_rms, d_delta, d_norm, d_heads, d_table})
    HIP_OK(hipFree(pointer));
}

// dspine.py:287-320 serial_chain (greedy: lane 0 on row 1, then the best lane
// given the previous one, ties to the lower lane) and :589-641 (sampled: a
// draw from the softmax of the sender's row; gufo keeps DFlash2's unfiltered
// q and inverse CDF, so the reported q is exactly the one drawn from).
void TestChain(std::mt19937& rng) {
  auto table = Random(kProposals * kTopK * kTopK, rng, 2.0F);
  for (std::uint32_t row = 0; row < kProposals * kTopK; row += 3) {
    float* scores = table.data() + row * kTopK;
    const float top = *std::max_element(scores, scores + kTopK) + 1.0F;
    scores[11] = top;  // a tie: lane 4 must win
    scores[4] = top;
  }
  std::vector<std::uint32_t> ids((kProposals + 1) * kTopK);
  for (std::uint32_t i = 0; i < ids.size(); ++i)
    ids[i] = 1000U + 7U * i;
  std::uniform_real_distribution<float> uniform(0.0F, 1.0F);
  std::vector<float> uniforms(kProposals);
  for (auto& value : uniforms)
    value = uniform(rng);
  auto* d_table = Upload(table);
  auto* d_ids = Upload(ids);
  auto* d_uniforms = Upload(uniforms);
  auto* d_tokens = Upload(std::vector<std::uint32_t>(kProposals));
  auto* d_confidences = Upload(std::vector<float>(kProposals));
  auto* d_candidates = Upload(std::vector<std::uint32_t>(kProposals * kTopK));
  auto* d_q = Upload(std::vector<float>(kProposals * kTopK));
  for (const float temperature : {0.0F, 0.6F}) {
    std::vector<std::uint32_t> want(kProposals);
    std::vector<double> want_q(kProposals * kTopK);
    std::uint32_t lane = 0;
    for (std::uint32_t t = 0; t < kProposals; ++t) {
      const float* scores = table.data() + (t * kTopK + lane) * kTopK;
      std::uint32_t best = 0;
      for (std::uint32_t k = 1; k < kTopK; ++k)
        if (scores[k] > scores[best])
          best = k;
      std::uint32_t pick = best;
      if (temperature > 0.0F) {
        double total = 0.0;
        std::vector<double> weight(kTopK);
        for (std::uint32_t k = 0; k < kTopK; ++k)
          total += weight[k] =
              std::exp((scores[k] - scores[best]) / temperature);
        double cumulative = 0.0;
        for (pick = 0; pick + 1 < kTopK; ++pick)
          if ((cumulative += weight[pick]) > uniforms[t] * total)
            break;
        for (std::uint32_t k = 0; k < kTopK; ++k)
          want_q[t * kTopK + k] = weight[k] / total;
      } else {
        want_q[t * kTopK + best] = 1.0;
      }
      want[t] = ids[(t + 1) * kTopK + pick];
      lane = pick;
    }
    LaunchDSpineChain(d_table, d_ids, kProposals, kTopK, temperature,
                      d_uniforms, d_tokens, d_confidences, d_candidates, d_q,
                      nullptr);
    const auto tokens = Download(d_tokens, kProposals);
    const auto candidates = Download(d_candidates, kProposals * kTopK);
    const auto q = Download(d_q, kProposals * kTopK);
    const auto confidences = Download(d_confidences, kProposals);
    std::size_t wrong = 0, outside = 0;
    double worst_sum = 0.0;
    for (std::uint32_t t = 0; t < kProposals; ++t) {
      wrong += tokens[t] != want[t];
      const auto* row = candidates.data() + t * kTopK;
      const auto picked = std::find(row, row + kTopK, tokens[t]);
      outside += picked == row + kTopK ||
                 confidences[t] != q[t * kTopK + (picked - row)];
      double sum = 0.0;
      for (std::uint32_t k = 0; k < kTopK; ++k) {
        sum += q[t * kTopK + k];
        wrong += row[k] != ids[(t + 1) * kTopK + k];
      }
      worst_sum = std::max(worst_sum, std::abs(sum - 1.0));
    }
    const double e = RelErr(q, want_q);
    if (temperature == 0.0F) {
      Expect(wrong == 0, "chain greedy tokens, ties (dspine.py:307-320)",
             double(wrong));
      Expect(e == 0.0 && outside == 0, "chain greedy q one-hot", e);
    } else {
      Expect(wrong == 0, "chain sampled tokens (dspine.py:616-641)",
             double(wrong));
      Expect(e <= kTolerance, "chain sampled q (dspine.py:617-627)", e);
      Expect(worst_sum <= 1e-5 && outside == 0,
             "chain sampled q sums to 1, pick in cand", worst_sum);
    }
  }
  for (auto* pointer :
       {static_cast<void*>(d_table), static_cast<void*>(d_ids),
        static_cast<void*>(d_uniforms), static_cast<void*>(d_tokens),
        static_cast<void*>(d_confidences), static_cast<void*>(d_candidates),
        static_cast<void*>(d_q)})
    HIP_OK(hipFree(pointer));
}

// dspine.py's draft attention is causal inside the block (models/dspine.py:
// 398-402): block key p is visible to row t only when p <= t. Context keys
// keep the DFlash window. Also: rows 0..7 of a 16-row block equal an 8-row
// block bitwise (same kernel variant), which is what lets gufo run n + 1 rows.
void TestCausalAttention(std::mt19937& rng) {
  constexpr std::uint32_t heads = 32, kv_heads = 8, dim = 128;
  constexpr std::uint32_t window = 8192;
  const float scale = 1.0F / std::sqrt(float(dim));
  for (const std::uint32_t position : {1000U, 9000U}) {
    const std::uint32_t capacity = window;
    const auto q = Random(kRows * heads * dim, rng);
    const auto block_k = Random(kRows * kv_heads * dim, rng);
    const auto block_v = Random(kRows * kv_heads * dim, rng);
    const auto history_k = Random(std::size_t{capacity} * kv_heads * dim, rng);
    const auto history_v = Random(std::size_t{capacity} * kv_heads * dim, rng);
    std::vector<double> want(q.size());
    const std::uint32_t start =
        position + 1 > window ? position + 1 - window : 0;
    for (std::uint32_t t = 0; t < kRows; ++t)
      for (std::uint32_t h = 0; h < heads; ++h) {
        const std::uint32_t kv = h / (heads / kv_heads);
        const float* query = q.data() + (t * heads + h) * dim;
        std::vector<double> scores;
        std::vector<const float*> values;
        for (std::uint32_t p = start; p < position; ++p) {
          if (position + t - p >= window)
            continue;
          const auto slot = std::size_t{p % capacity} * kv_heads + kv;
          double dot = 0.0;
          for (std::uint32_t d = 0; d < dim; ++d)
            dot += double(query[d]) * history_k[slot * dim + d];
          scores.push_back(dot * scale);
          values.push_back(history_v.data() + slot * dim);
        }
        for (std::uint32_t p = 0; p <= t; ++p) {
          double dot = 0.0;
          for (std::uint32_t d = 0; d < dim; ++d)
            dot += double(query[d]) * block_k[(p * kv_heads + kv) * dim + d];
          scores.push_back(dot * scale);
          values.push_back(block_v.data() + (p * kv_heads + kv) * dim);
        }
        const double top = *std::max_element(scores.begin(), scores.end());
        double total = 0.0;
        for (auto& score : scores)
          total += score = std::exp(score - top);
        for (std::uint32_t d = 0; d < dim; ++d) {
          double acc = 0.0;
          for (std::size_t i = 0; i < scores.size(); ++i)
            acc += scores[i] / total * values[i][d];
          want[(t * heads + h) * dim + d] = acc;
        }
      }
    auto* d_q = Upload(q);
    auto* d_bk = Upload(block_k);
    auto* d_bv = Upload(block_v);
    auto* d_hk = Upload(history_k);
    auto* d_hv = Upload(history_v);
    auto* d_out = Upload(std::vector<float>(q.size()));
    LaunchDFlashNonCausalAttention(d_q, d_hk, d_hv, d_bk, d_bv, d_out, position,
                                   position, kRows, window, heads, kv_heads,
                                   dim, scale, nullptr, capacity, true);
    const auto full = Download(d_out, q.size());
    const double e = RelErr(full, want);
    Expect(e <= kTolerance,
           position < 4096 ? "causal attention, 2 heads/CTA, 1k context"
                           : "causal attention, 1 head/CTA, 8k window",
           e);
    LaunchDFlashNonCausalAttention(d_q, d_hk, d_hv, d_bk, d_bv, d_out, position,
                                   position, 8, window, heads, kv_heads, dim,
                                   scale, nullptr, capacity, true);
    const auto prefix = Download(d_out, 8 * heads * dim);
    const bool same = std::equal(prefix.begin(), prefix.end(), full.begin());
    Expect(same, "causal rows 0..7: 8-row block == 16-row block", same ? 0 : 1);
    // The split route (commit 3) must give the wave route's bits, also for a
    // block whose last PV group is partial (6 rows x 4 heads = 24 combos).
    auto* d_scores = Upload(std::vector<float>(
        DSpineAttentionScoreElements(kRows, heads, capacity), 0.0F));
    for (const std::uint32_t rows : {kRows, 6U}) {
      LaunchDFlashNonCausalAttention(
          d_q, d_hk, d_hv, d_bk, d_bv, d_out, position, position, rows, window,
          heads, kv_heads, dim, scale, nullptr, capacity, true);
      const auto wave = Download(d_out, std::size_t{rows} * heads * dim);
      HIP_OK(hipMemset(d_out, 0, q.size() * sizeof(float)));
      const bool launched = LaunchDSpineAttention(
          d_q, d_hk, d_hv, d_bk, d_bv, d_out, position, position, rows, window,
          heads, kv_heads, dim, scale, nullptr, capacity, d_scores);
      const auto split = Download(d_out, std::size_t{rows} * heads * dim);
      std::size_t differ = 0;
      for (std::size_t i = 0; i < split.size(); ++i)
        differ += std::memcmp(&split[i], &wave[i], sizeof(float)) != 0;
      Expect(launched && differ == 0,
             rows == kRows ? "split attention == wave route, bitwise, 16 rows"
                           : "split attention == wave route, bitwise, 6 rows",
             double(differ));
    }
    HIP_OK(hipFree(d_scores));
    for (auto* pointer : {d_q, d_bk, d_bv, d_hk, d_hv, d_out})
      HIP_OK(hipFree(pointer));
  }
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::printf("qwen_dspine_ops_test: no HIP device, skipped\n");
    return 77;
  }
  std::mt19937 rng(20261109);
  TestRowStats(rng);
  TestCodes(rng);
  TestMessage(rng);
  TestWrite(rng);
  TestTopK(rng);
  TestTable(rng);
  TestChain(rng);
  TestCausalAttention(rng);
  std::printf("qwen_dspine_ops_test: %s (%d failures)\n",
              failures == 0 ? "ALL TESTS PASSED" : "FAILED", failures);
  return failures == 0 ? 0 : 1;
}
