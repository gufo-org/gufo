#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/testing/compare/logit_comparator.hpp"

namespace {
using Token = gufo::tokenization::TokenId;
using Executor = gufo::hip::QwenGpuExecutor;

void Expect(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct Case {
  std::vector<Token> tokens;
  std::vector<std::vector<float>> logits;
};

// Teacher forcing keeps quantization comparisons on identical input prefixes.
// Free-running text would conflate different inputs with numerical differences.
constexpr std::array<const char*, 3> kTexts{
    "Virtual memory lets an operating system give each process its own address "
    "space. Pages map virtual addresses to physical memory. A page fault "
    "occurs "
    "when the requested page is unavailable and the operating system must act.",
    "A train leaves the station at noon and travels at sixty kilometers per "
    "hour. "
    "Another train leaves one hour later at ninety kilometers per hour. To "
    "find "
    "when they meet, compare the distances each train has traveled.",
    "template <typename T> class RingBuffer { public: bool push(const T& "
    "value) "
    "{ if (count == capacity) return false; data[tail] = value; "
    "tail = (tail + 1) % capacity; ++count; return true; } };",
};

std::vector<float> Logits(Executor& executor) {
  const auto view = executor.CopyLastLogits();
  Expect(!view.empty() && std::ranges::all_of(
                              view, [](float v) { return std::isfinite(v); }),
         "target logits must be finite and nonempty");
  return {view.begin(), view.end()};
}

bool ByteEqual(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

std::vector<Case> Capture(const char* path, bool check_replay) {
  std::string error;
  auto owner = gufo::core::GgufReader::OpenFile(path, &error);
  Expect(owner != nullptr, error);
  auto executor = Executor::CreateFromGguf(
      std::shared_ptr<const gufo::core::GgufReader>(std::move(owner)), &error,
      128);
  Expect(executor != nullptr, error);
  Expect(executor->GetConfig().hidden_size == 5120 &&
             executor->GetConfig().vocab_size == 248320,
         "quality fixture requires Qwen3.8 27B");
  std::vector<Case> cases;
  for (const auto* text : kTexts) {
    Case row{.tokens = executor->GetTokenizer().Encode(text), .logits = {}};
    constexpr std::size_t prompt_size = 24;
    constexpr std::size_t continuation = 4;
    Expect(row.tokens.size() >= prompt_size + continuation,
           "quality fixture has too few tokens");
    row.tokens.resize(prompt_size + continuation);
    executor->Reset();
    (void)executor->ForwardPromptBatch(
        std::span(row.tokens).first(prompt_size));
    row.logits.push_back(Logits(*executor));
    auto snapshot = executor->SaveSnapshot(prompt_size);
    for (std::size_t pos = prompt_size; pos < row.tokens.size(); ++pos) {
      (void)executor->ForwardToken(row.tokens[pos], pos);
      row.logits.push_back(Logits(*executor));
    }
    if (check_replay) {
      executor->RestoreSnapshot(*snapshot);
      executor->SaveState(prompt_size);
      const auto suffix = std::span(row.tokens).subspan(prompt_size);
      const auto predictions =
          executor->ForwardVerificationChunk(suffix, prompt_size, true);
      Expect(predictions.size() == suffix.size(),
             "verification returned the wrong number of rows");
      for (std::size_t index = 0; index < suffix.size(); ++index) {
        const auto logits = executor->CopyVerificationLogits(index);
        const auto comparison =
            gufo::testing::CompareLogits(logits, row.logits[index + 1]);
        std::cout << "verification case=" << cases.size() << " row=" << index
                  << " exact=" << ByteEqual(logits, row.logits[index + 1])
                  << " max_abs=" << comparison.max_abs_diff << '\n';
        Expect(ByteEqual(logits, row.logits[index + 1]),
               "batched verification changed target logits");
      }
      executor->RestoreState();
      executor->CommitVerificationChunk(suffix.first(2), prompt_size);
      for (std::size_t index = 2; index < suffix.size(); ++index) {
        (void)executor->ForwardToken(suffix[index], prompt_size + index);
        Expect(ByteEqual(Logits(*executor), row.logits[index + 1]),
               "rejected draft replay changed target logits");
      }
      executor->RestoreSnapshot(*snapshot);
      for (std::size_t pos = prompt_size; pos < row.tokens.size(); ++pos) {
        (void)executor->ForwardToken(row.tokens[pos], pos);
        Expect(ByteEqual(Logits(*executor), row.logits[pos - prompt_size + 1]),
               "snapshot continuation changed target logits");
      }
      executor->Reset();
      (void)executor->ForwardPromptBatch(
          std::span(row.tokens).first(prompt_size));
      Expect(ByteEqual(Logits(*executor), row.logits.front()),
             "repeated prefill changed target logits");
      executor->Reset();
      for (std::size_t pos = 0; pos < prompt_size; ++pos) {
        (void)executor->ForwardToken(row.tokens[pos], pos,
                                     pos + 1 == prompt_size);
      }
      const auto comparison =
          gufo::testing::CompareLogits(Logits(*executor), row.logits.front());
      std::cout << "prefill case=" << cases.size()
                << " top1_match=" << comparison.top1_match
                << " rmse=" << comparison.root_mean_square_error
                << " max_abs=" << comparison.max_abs_diff << '\n';
      Expect(comparison.finite && comparison.top1_match,
             "prefill/scalar target choice differs");
    }
    cases.push_back(std::move(row));
  }
  return cases;
}

double LogNormalizer(std::span<const float> logits) {
  const double maximum = *std::ranges::max_element(logits);
  double sum = 0;
  for (const double value : logits)
    sum += std::exp(value - maximum);
  return maximum + std::log(sum);
}

void CompareReference(const std::vector<Case>& candidate,
                      const std::vector<Case>& reference) {
  Expect(candidate.size() == reference.size(), "reference case count differs");
  double total_kl = 0, total_tv = 0, total_nll_delta = 0;
  std::size_t rows = 0, labels = 0, top1 = 0;
  for (std::size_t c = 0; c < reference.size(); ++c) {
    Expect(candidate[c].tokens == reference[c].tokens,
           "reference and candidate tokenization differ");
    for (std::size_t r = 0; r < reference[c].logits.size(); ++r) {
      const auto& p = reference[c].logits[r];
      const auto& q = candidate[c].logits[r];
      Expect(p.size() == q.size(), "reference vocabulary differs");
      const double log_p = LogNormalizer(p), log_q = LogNormalizer(q);
      double kl = 0, tv = 0;
      for (std::size_t t = 0; t < p.size(); ++t) {
        const double probability = std::exp(p[t] - log_p);
        kl += probability * ((p[t] - log_p) - (q[t] - log_q));
        tv += std::abs(probability - std::exp(q[t] - log_q)) * 0.5;
      }
      const auto cmp = gufo::testing::CompareLogits(p, q);
      total_kl += kl;
      total_tv += tv;
      top1 += cmp.top1_match;
      ++rows;
      if (r < 4) {
        const auto label = reference[c].tokens[24 + r];
        total_nll_delta += (log_q - q[label]) - (log_p - p[label]);
        ++labels;
      }
      std::cout << "reference case=" << c << " row=" << r
                << " top1_match=" << cmp.top1_match << " kl=" << kl
                << " tv=" << tv << " rmse=" << cmp.root_mean_square_error
                << '\n';
    }
  }
  // These are measurements, not a claim that different quantizations must
  // produce identical distributions or that Gufo BF16 is independent truth.
  std::cout << "reference rows=" << rows << " top1=" << top1
            << " mean_kl=" << total_kl / rows << " mean_tv=" << total_tv / rows
            << " mean_nll_delta=" << total_nll_delta / labels << '\n';
}
}  // namespace

int main(int argc, const char* const* argv) {
  try {
    const char* model = argc > 1 ? argv[1] : std::getenv("GUFO_QWEN27B_MODEL");
    if (model == nullptr) {
      std::cout << "Set GUFO_QWEN27B_MODEL for the Qwen27B target check.\n";
      return 77;
    }
    const auto candidate = Capture(model, true);
    if (argc > 2) {
      // Load one model at a time: a reference check must not require both
      // targets to remain resident in unified memory.
      CompareReference(candidate, Capture(argv[2], false));
    }
    std::cout << "Qwen27B target checks passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
