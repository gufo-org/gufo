#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

// The model's indexer: 4 heads of 128, 4-token blocks, a 2048-token budget.
constexpr std::uint32_t kHeads = 4;
constexpr std::uint32_t kDim = 128;
constexpr std::uint32_t kRatio = 4;
constexpr std::uint32_t kBudget = 2048 / kRatio;

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

std::vector<float> MakeValues(std::size_t count, std::uint32_t seed,
                              float scale) {
  std::vector<float> values(count);
  for (float& value : values) {
    value = scale *
            static_cast<float>(static_cast<int>(NextRandom(&seed) & 0xFFFFU) -
                               32768) /
            32768.0F;
  }
  return values;
}

template<typename T>
T* Upload(const std::vector<T>& host) {
  T* device = nullptr;
  CheckHip(hipMalloc(&device, host.size() * sizeof(T) + 4096), "hipMalloc");
  CheckHip(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                     hipMemcpyHostToDevice),
           "upload");
  return device;
}

template<typename T>
std::vector<T> Download(const T* device, std::size_t count) {
  std::vector<T> host(count);
  CheckHip(
      hipMemcpy(host.data(), device, count * sizeof(T), hipMemcpyDeviceToHost),
      "download");
  return host;
}

enum class ScoreLayout { kTight, kAligned };
enum class TiedBlocks { kPrefix, kSuffix };

bool Run(std::uint32_t n_tokens, std::uint32_t start_pos, std::uint32_t seed,
         bool zero_queries = false, std::uint32_t tied_high_blocks = 0,
         bool sample_scores = false, ScoreLayout layout = ScoreLayout::kTight,
         TiedBlocks tied_blocks = TiedBlocks::kPrefix) {
  const std::uint32_t max_context = start_pos + n_tokens + 64;
  const std::uint32_t blocks_needed = (max_context + kRatio - 1) / kRatio;
  const std::uint32_t mask_words = (blocks_needed + 31) / 32;
  const std::uint32_t max_blocks =
      layout == ScoreLayout::kAligned ? mask_words * 32 : blocks_needed;
  const std::size_t q_count =
      static_cast<std::size_t>(n_tokens) * kHeads * kDim;
  // Zero queries make every score tie at zero: the budget must then fill
  // in index order.
  auto qv = zero_queries ? std::vector<float>(q_count, 0.0F)
                         : MakeValues(q_count, seed, 1.0F);
  auto blocks = MakeValues(static_cast<std::size_t>(max_blocks) * kDim,
                           seed ^ 0x9999U, 1.0F);
  if (tied_high_blocks != 0) {
    std::fill(qv.begin(), qv.end(), 0.0F);
    std::fill(blocks.begin(), blocks.end(), 0.0F);
    for (std::size_t i = 0; i < qv.size(); i += kDim)
      qv[i] = 1.0F;
    for (std::uint32_t b = 0; b < tied_high_blocks; ++b) {
      const auto row =
          tied_blocks == TiedBlocks::kPrefix ? b : max_blocks - 1 - b;
      blocks[static_cast<std::size_t>(row) * kDim] = 1.0F;
    }
  }
  std::vector<__half> blocks_half(blocks.size());
  const auto half = [](float value) { return __float2half_rn(value); };
  std::transform(blocks.begin(), blocks.end(), blocks_half.begin(), half);
  float* d_q = Upload(qv);
  __half* d_blocks = Upload(blocks_half);
  std::uint32_t* d_pos = Upload(std::vector<std::uint32_t>{start_pos});
  std::uint32_t* d_mask = Upload(std::vector<std::uint32_t>(
      static_cast<std::size_t>(n_tokens) * mask_words, 0xA5A5A5A5U));
  float* d_scores = Upload(std::vector<float>(
      static_cast<std::size_t>(n_tokens) * max_blocks, -1.0F));
  q::SelectBlocks(d_q, d_blocks, d_mask, d_scores, n_tokens, d_pos, 0, kHeads,
                  kDim, kRatio, kBudget, mask_words, max_blocks, nullptr);
  CheckHip(hipDeviceSynchronize(), "selection");
  const auto mask =
      Download(d_mask, static_cast<std::size_t>(n_tokens) * mask_words);
  const auto scores =
      Download(d_scores, static_cast<std::size_t>(n_tokens) * max_blocks);
  q::SelectBlocks(d_q, d_blocks, d_mask, d_scores, n_tokens, d_pos, 0, kHeads,
                  kDim, kRatio, kBudget, mask_words, max_blocks, nullptr);
  if (Download(d_mask, mask.size()) != mask) {
    throw std::runtime_error("selection replay changed the mask");
  }
  // Reusing bounded score scratch must retain absolute query positions and
  // causal masks, including a ragged chunk and a different padded stride.
  constexpr std::uint32_t chunk = 31;
  const auto compact_stride = ((start_pos + n_tokens) / kRatio + 31) / 32 * 32;
  float* d_chunk_scores = nullptr;
  CheckHip(hipMalloc(&d_chunk_scores, std::size_t{std::min(chunk, n_tokens)} *
                                          compact_stride * sizeof(float)),
           "chunk score allocation");
  for (std::uint32_t first = 0; first < n_tokens; first += chunk) {
    const auto count = std::min(chunk, n_tokens - first);
    q::SelectBlocks(d_q + std::size_t{first} * kHeads * kDim, d_blocks,
                    d_mask + std::size_t{first} * mask_words, d_chunk_scores,
                    count, d_pos, first, kHeads, kDim, kRatio, kBudget,
                    mask_words, compact_stride, nullptr,
                    (start_pos + first + count) / kRatio);
  }
  if (Download(d_mask, mask.size()) != mask) {
    throw std::runtime_error("chunked selection changed the mask");
  }
  CheckHip(hipFree(d_chunk_scores), "free chunk scores");

  double worst_score = 0.0;
  std::size_t mismatches = 0;
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const std::uint32_t complete = (start_pos + t + 1) / kRatio;
    const std::uint32_t* words =
        mask.data() + static_cast<std::size_t>(t) * mask_words;
    std::vector<bool> expect(max_blocks, false);
    if (complete <= kBudget) {
      // Below the budget every block is visible.
      for (std::uint32_t w = 0; w < mask_words; ++w) {
        mismatches += words[w] == 0xFFFFFFFFU ? 0 : 1;
      }
      continue;
    }
    // Reference scores in double; the GPU value must agree closely.
    for (std::uint32_t b = 0; b < complete; ++b) {
      // The large selection case checks every mask bit and samples score
      // arithmetic already covered exhaustively by the smaller cases.
      if (sample_scores && b % 1024 != 0 && b + 1 != complete)
        continue;
      double total = 0.0;
      float rounded_total = 0.0F;
      for (std::uint32_t h = 0; h < kHeads; ++h) {
        double dot = 0.0;
        std::array<float, 32> partial{};
        for (std::uint32_t i = 0; i < kDim; ++i) {
          const auto query =
              qv[(static_cast<std::size_t>(t) * kHeads + h) * kDim + i];
          const auto key =
              __half2float(blocks_half[static_cast<std::size_t>(b) * kDim + i]);
          dot += static_cast<double>(query) * key;
          partial[i % 32] = std::fma(query, key, partial[i % 32]);
        }
        total += std::max(dot, 0.0);
        for (unsigned delta = 16; delta; delta >>= 1)
          for (unsigned lane = 0; lane < delta; ++lane)
            partial[lane] += partial[lane + delta];
        rounded_total += std::max(partial[0], 0.0F);
      }
      const float got = scores[static_cast<std::size_t>(t) * max_blocks + b];
      // A loose numerical tolerance cannot detect exchanges at the top-k
      // boundary. Keep the established F32 reduction as well as the FP64
      // formula check below when changing kernel layout.
      if (got != rounded_total)
        throw std::runtime_error("selector changed the F32 reduction");
      const double err = std::abs(total - got) / std::max(1.0, total);
      worst_score = std::max(worst_score, err);
    }
    // Selection contract on the GPU's own scores: the budget highest,
    // blocks strictly above the threshold first, ties by lowest index.
    std::vector<float> sorted(
        scores.begin() + static_cast<std::size_t>(t) * max_blocks,
        scores.begin() + static_cast<std::size_t>(t) * max_blocks + complete);
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    const float threshold = sorted[kBudget - 1];
    std::uint32_t remaining = kBudget;
    for (std::uint32_t b = 0; b < complete; ++b) {
      if (scores[static_cast<std::size_t>(t) * max_blocks + b] > threshold) {
        expect[b] = true;
        --remaining;
      }
    }
    for (std::uint32_t b = 0; b < complete && remaining > 0; ++b) {
      if (scores[static_cast<std::size_t>(t) * max_blocks + b] == threshold) {
        expect[b] = true;
        --remaining;
      }
    }
    for (std::uint32_t b = 0; b < max_blocks; ++b) {
      const bool got = ((words[b / 32] >> (b % 32)) & 1U) != 0;
      mismatches += got == expect[b] ? 0 : 1;
    }
  }
  std::cout << "block selection n=" << n_tokens << " start=" << start_pos
            << ": worst score error " << worst_score << ", mask mismatches "
            << mismatches << '\n';
  (void)hipFree(d_q);
  (void)hipFree(d_blocks);
  (void)hipFree(d_pos);
  (void)hipFree(d_mask);
  (void)hipFree(d_scores);
  // F32 queries against the stored F16 keys, independently scored in F64.
  return worst_score < 1e-5 && mismatches == 0;
}

void CheckQueryPrecisionBoundary() {
  constexpr unsigned blocks = kBudget + 1, words = (blocks + 31) / 32;
  std::vector<float> queries(kHeads * kDim, 0);
  queries[0] = 1.0F;
  queries[1] = 1.0001F;  // Both become 1.0 in F16.
  std::vector<__half> keys(blocks * kDim, __float2half(0));
  for (unsigned b = 0; b < kBudget - 1; ++b)
    keys[b * kDim] = __float2half(2);
  keys[(kBudget - 1) * kDim] = __float2half(1);
  keys[kBudget * kDim + 1] = __float2half(1);
  auto* dq = Upload(queries);
  auto* dk = Upload(keys);
  auto* dp = Upload(std::vector<unsigned>{blocks * kRatio - 1});
  auto* dm = Upload(std::vector<unsigned>(words, 0));
  auto* ds = Upload(std::vector<float>(blocks, 0));
  q::SelectBlocks(dq, dk, dm, ds, 1, dp, 0, kHeads, kDim, kRatio, kBudget,
                  words, blocks, nullptr);
  const auto mask = Download(dm, words);
  if ((mask[(kBudget - 1) / 32] & (1U << ((kBudget - 1) % 32))) != 0 ||
      (mask[kBudget / 32] & (1U << (kBudget % 32))) == 0)
    throw std::runtime_error("F16 narrowing changed the top-k boundary");
  for (void* p :
       {static_cast<void*>(dq), static_cast<void*>(dk), static_cast<void*>(dp),
        static_cast<void*>(dm), static_cast<void*>(ds)})
    CheckHip(hipFree(p), "free precision probe");
}

void CheckPooling(unsigned start, unsigned capacity) {
  constexpr unsigned tokens = 7;
  const unsigned first = start / kRatio;
  const unsigned blocks = (start + tokens) / kRatio + 3;
  constexpr unsigned rotary = 64;
  constexpr float theta = 1000000.0F, eps = 1e-6F;
  const auto raw = MakeValues((start + tokens) * kDim, 0x31415926U, 1.0F);
  const auto gamma = MakeValues(kDim, 0x27182818U, 1.0F);
  const std::vector<__half> initial(blocks * kDim, __float2half(3.0F));
  // Keep only the newest raw rows. Store the next batch through the same
  // ring operation used by the executor, including a wrap within a block
  // of speculative tokens.
  std::vector<float> ring(capacity * kDim, -123.0F);
  for (unsigned t = start > capacity ? start - capacity : 0; t < start; ++t)
    std::copy_n(raw.data() + std::size_t{t} * kDim, kDim,
                ring.data() + std::size_t{t & (capacity - 1)} * kDim);
  auto* d_raw = Upload(ring);
  auto* d_input =
      Upload(std::vector<float>(raw.begin() + start * kDim, raw.end()));
  auto* d_gamma = Upload(gamma);
  auto* d_blocks = Upload(initial);
  auto* d_first = Upload(std::vector<unsigned>{first});
  auto* d_start = Upload(std::vector<unsigned>{start});
  const auto run = [&] {
    q::StoreRows(d_input, d_raw, tokens, kDim, d_start, capacity, nullptr);
    q::PoolIndexerBlocks(d_raw, d_gamma, d_blocks, d_first, d_start, tokens, 4,
                         kRatio, kDim, rotary, theta, eps, capacity, nullptr);
  };
  run();
  const auto actual = Download(d_blocks, initial.size());
  run();
  const auto replay = Download(d_blocks, initial.size());
  if (std::memcmp(actual.data(), replay.data(),
                  actual.size() * sizeof(__half)) != 0) {
    throw std::runtime_error("pooling replay changed the block keys");
  }
  double worst = 0.0;
  for (unsigned block = 0; block < blocks; ++block) {
    if (block < first || block >= (start + tokens) / kRatio) {
      for (unsigned i = 0; i < kDim; ++i) {
        if (__half2float(actual[block * kDim + i]) != 3.0F) {
          throw std::runtime_error("pooling wrote an incomplete or old block");
        }
      }
      continue;
    }
    std::vector<double> values(kDim);
    double squares = 0.0;
    for (unsigned i = 0; i < kDim; ++i) {
      for (unsigned r = 0; r < kRatio; ++r)
        values[i] += raw[(block * kRatio + r) * kDim + i] / double(kRatio);
      squares += values[i] * values[i];
    }
    const double scale = 1.0 / std::sqrt(squares / kDim + eps);
    for (unsigned i = 0; i < kDim; ++i)
      values[i] *= scale * gamma[i];
    for (unsigned i = 0; i < rotary / 2; ++i) {
      const double angle =
          block * kRatio * std::pow(double(theta), -2.0 * i / rotary);
      const auto a = values[i], b = values[i + rotary / 2];
      values[i] = a * std::cos(angle) - b * std::sin(angle);
      values[i + rotary / 2] = a * std::sin(angle) + b * std::cos(angle);
    }
    for (unsigned i = 0; i < kDim; ++i) {
      const double value = __half2float(actual[block * kDim + i]);
      const double error =
          std::abs(value - values[i]) / std::max(1.0, std::abs(values[i]));
      if (!std::isfinite(value) || error > 1e-3) {
        throw std::runtime_error("pooled F16 key disagrees with FP64 formula");
      }
      worst = std::max(worst, error);
    }
  }
  for (void* pointer :
       {static_cast<void*>(d_raw), static_cast<void*>(d_input),
        static_cast<void*>(d_gamma), static_cast<void*>(d_blocks),
        static_cast<void*>(d_first), static_cast<void*>(d_start)})
    CheckHip(hipFree(pointer), "free pooling input");
  std::cout << "pooled F16 keys: FP64 error " << worst
            << ", boundaries and replay exact\n";
}

}  // namespace

int main() {
  try {
    CheckQueryPrecisionBoundary();
    CheckPooling(29, 64);
    CheckPooling(29, 16);     // batch wraps the raw-key ring
    CheckPooling(65533, 16);  // many wraps; absolute rotary position retained
    bool ok = true;
    ok = Run(100, 20000, 0x1234ABCDU) && ok;  // deep, ragged group
    ok = Run(7, 131069, 0x2468ACE0U) && ok;
    ok = Run(257, 131069, 0xC0FFEE01U, false, 0, true, ScoreLayout::kAligned) &&
         ok;
    ok = Run(81, 260013, 0xC0FFEE02U, false, 0, true, ScoreLayout::kAligned) &&
         ok;
    ok = Run(129, 131069, 0, true, 0, true) && ok;  // deep, partial word
    ok = Run(1, 9001, 0x0BADF00DU) && ok;           // decode
    ok = Run(40, 2040, 0xDEADBEEFU) && ok;          // straddles the budget
    ok = Run(3, 6000, 0x5EED5EEDU, true) && ok;     // all scores tie at zero
    // A threshold tie group that fits exactly, then one that needs the
    // lowest-index prefix. Both must produce the same CPU-sorted mask.
    ok = Run(1, 9001, 0, false, kBudget) && ok;
    ok = Run(1, 9001, 0, false, kBudget + 7) && ok;
    // Unrepresentative samples put the threshold below or above the
    // histogram window. Both tails must retain the exact sorted result.
    ok = Run(1, 20000, 0, false, 256) && ok;
    ok = Run(1, 20000, 0, false, kBudget + 128, false, ScoreLayout::kAligned,
             TiedBlocks::kSuffix) &&
         ok;
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
