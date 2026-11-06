// A wide single-session verification (QwenGpuExecutor::ForwardVerificationBatch
// with more than eight rows) must give every row the predictions, logits and
// DFlash2 target features of consecutive verifications of at most eight rows,
// and a rollback to any kept prefix must continue exactly as those
// verifications cut back at the same token. The check needs the production
// weight formats and shapes (the verification projections dispatch on them),
// so it loads the 27B GGUF named by GUFO_QWEN27B_MODEL.

#include <algorithm>
#include <cstdint>
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

namespace {

using Token = gufo::tokenization::TokenId;
using Executor = gufo::hip::QwenGpuExecutor;

constexpr std::size_t kSegment = 8;
// Taps like DFlash2's, so verification also captures target features.
constexpr std::uint32_t kTargetLayers[] = {1, 16, 31, 46, 61};

void Expect(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

bool BitEqual(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

std::vector<Token> Tokens(std::size_t count, std::uint64_t seed) {
  std::vector<Token> tokens(count);
  for (auto& token : tokens) {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    token = static_cast<Token>(1000 + (seed >> 33) % 100000);
  }
  return tokens;
}

struct Rows {
  std::vector<Token> predictions;
  std::vector<float> logits;
  std::vector<float> hidden;
};

void Append(Rows& rows, Executor& executor, std::vector<Token> predictions) {
  rows.predictions.insert(rows.predictions.end(), predictions.begin(),
                          predictions.end());
  const auto logits = executor.GetVerificationLogits();
  const auto hidden = executor.GetVerificationHiddenStates();
  Expect(logits.size() == predictions.size() * executor.GetConfig().vocab_size,
         "verification logits are incomplete");
  Expect(!hidden.empty() && hidden.size() % predictions.size() == 0,
         "verification target features are incomplete");
  rows.logits.insert(rows.logits.end(), logits.begin(), logits.end());
  rows.hidden.insert(rows.hidden.end(), hidden.begin(), hidden.end());
}

// Verifies chain rows [begin, end) as one pass; `keep` < end - begin cuts it
// back as a rejected draft does.
void Verify(Executor& executor, std::span<const Token> chain,
            std::uint32_t position, std::size_t begin, std::size_t end,
            std::size_t keep, Rows* rows) {
  const auto start = position + static_cast<std::uint32_t>(begin);
  executor.SaveState(start);
  auto predictions = executor.ForwardVerificationChunk(
      chain.subspan(begin, end - begin), start, true);
  if (rows != nullptr)
    Append(*rows, executor, std::move(predictions));
  if (keep < end - begin) {
    executor.RestoreState();
    executor.CommitVerificationChunk(chain.subspan(begin, keep), start);
  }
  executor.FinishVerification();
}

// Consecutive ordinary verifications of the chain, each accepted whole up to
// the one that holds the last kept row.
Rows Reference(Executor& executor, std::span<const Token> chain,
               std::uint32_t position, std::size_t kept) {
  Rows rows;
  for (std::size_t begin = 0; begin < kept; begin += kSegment) {
    const auto end = std::min(begin + kSegment, chain.size());
    Verify(executor, chain, position, begin, end, std::min(end, kept) - begin,
           &rows);
  }
  return rows;
}

std::vector<std::vector<float>> Decode(Executor& executor,
                                       std::uint32_t position) {
  std::vector<std::vector<float>> logits;
  for (const Token token : Tokens(3, position)) {
    (void)executor.ForwardToken(token, position++);
    const auto row = executor.CopyLastLogits();
    logits.emplace_back(row.begin(), row.end());
  }
  return logits;
}

std::size_t CheckChain(Executor& executor,
                       const gufo::hip::QwenGpuSnapshot& prompt_state,
                       std::uint32_t position, std::size_t width) {
  const auto chain = Tokens(width, 7919 * width + position);
  const std::size_t vocab = executor.GetConfig().vocab_size;
  const std::string label = "prompt " + std::to_string(position) + " width " +
                            std::to_string(width) + ": ";

  executor.RestoreSnapshot(prompt_state);
  const auto reference = Reference(executor, chain, position, width);
  const auto reference_decode =
      Decode(executor, position + static_cast<std::uint32_t>(width));

  executor.RestoreSnapshot(prompt_state);
  Rows wide;
  Verify(executor, chain, position, 0, width, width, &wide);
  const std::size_t feature_width = wide.hidden.size() / width;
  for (std::size_t row = 0; row < width; ++row) {
    Expect(wide.predictions[row] == reference.predictions[row],
           label + "prediction differs at row " + std::to_string(row));
    Expect(BitEqual(std::span(wide.logits).subspan(row * vocab, vocab),
                    std::span(reference.logits).subspan(row * vocab, vocab)),
           label + "logits differ at row " + std::to_string(row));
    Expect(
        BitEqual(
            std::span(wide.hidden).subspan(row * feature_width, feature_width),
            std::span(reference.hidden)
                .subspan(row * feature_width, feature_width)),
        label + "target features differ at row " + std::to_string(row));
  }
  Expect(Decode(executor, position + static_cast<std::uint32_t>(width)) ==
             reference_decode,
         label + "decode after the whole chain differs");

  std::size_t cases = 1;
  const std::size_t cut_backs[] = {1, 7, 8, 9, 16, 17, 40, 56, 57, width - 1};
  for (const std::size_t kept : cut_backs) {
    if (kept == 0 || kept >= width)
      continue;
    executor.RestoreSnapshot(prompt_state);
    (void)Reference(executor, chain, position, kept);
    const auto expected =
        Decode(executor, position + static_cast<std::uint32_t>(kept));
    executor.RestoreSnapshot(prompt_state);
    Verify(executor, chain, position, 0, width, kept, nullptr);
    Expect(Decode(executor, position + static_cast<std::uint32_t>(kept)) ==
               expected,
           label + "decode after keeping " + std::to_string(kept) +
               " rows differs");
    ++cases;
  }
  return cases;
}

}  // namespace

int main(int argc, const char* const* argv) {
  try {
    const char* path = argc > 1 ? argv[1] : std::getenv("GUFO_QWEN27B_MODEL");
    if (path == nullptr) {
      std::cout << "Set GUFO_QWEN27B_MODEL for the wide verification check.\n";
      return 77;
    }
    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(path, &error);
    Expect(reader != nullptr, error);
    auto model = gufo::hip::QwenGpuModel::CreateFromGguf(
        std::shared_ptr<const gufo::core::GgufReader>(std::move(reader)),
        &error);
    Expect(model != nullptr, error);
    auto executor = Executor::Create(model, &error, 4096);
    Expect(executor != nullptr, error);
    executor->SetPromptHiddenCapture(true, kTargetLayers);

    std::size_t cases = 0;
    // 100 tokens: the chain crosses the split-K attention threshold (128).
    // 2,000 tokens: every segment uses split-K attention.
    for (const std::uint32_t prompt_size : {100U, 2000U}) {
      const auto prompt = Tokens(prompt_size, prompt_size);
      (void)executor->ForwardPromptBatch(prompt, 0, true);
      const auto prompt_state = executor->SaveSnapshot(prompt_size);
      for (const std::size_t width : {Executor::kMaxWideVerificationRows,
                                      std::size_t{20}, std::size_t{9}})
        cases += CheckChain(*executor, *prompt_state, prompt_size, width);
    }
    std::cout << "Qwen27B wide verification checks passed (" << cases
              << " cases).\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
