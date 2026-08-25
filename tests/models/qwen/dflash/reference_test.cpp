#include "src/models/qwen/dflash_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int kSkipped = 77;

[[noreturn]] void Fail(std::string_view message) {
  std::cerr << "Assertion failed: " << message << '\n';
  std::exit(1);
}

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    Fail(message);
  }
}

void ExpectNear(float actual, float expected, float tolerance,
                std::string_view message) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << "Assertion failed: " << message << " actual=" << actual
              << " expected=" << expected << " tolerance=" << tolerance << '\n';
    std::exit(1);
  }
}

void TestDynamicConv2Tap() {
  // Test 2-tap grouped depthwise convolution:
  // y[t, c] = w0[c] * x[t, c] + w1[c] * x[t-1, c] + b[c]
  constexpr std::size_t kHidden = 4;
  constexpr std::size_t kTokens = 3;

  std::vector<float> weight = {
      2.0F, 0.5F,  // c=0: w0=2.0, w1=0.5
      1.0F, -1.0F, // c=1: w0=1.0, w1=-1.0
      3.0F, 0.0F,  // c=2: w0=3.0, w1=0.0
      0.5F, 1.5F   // c=3: w0=0.5, w1=1.5
  };
  std::vector<float> bias = {0.1F, -0.2F, 0.5F, 0.0F};

  strix::models::QwenTensorRef weight_ref{
      .data = weight.data(),
      .type = strix::core::GgmlType::kF32,
      .num_elements = weight.size(),
      .available_bytes = weight.size() * sizeof(float)};
  strix::models::QwenTensorRef bias_ref{
      .data = bias.data(),
      .type = strix::core::GgmlType::kF32,
      .num_elements = bias.size(),
      .available_bytes = bias.size() * sizeof(float)};

  std::vector<float> sequence = {
      1.0F, 2.0F, 3.0F, 4.0F,  // t=0
      2.0F, 1.0F, 0.0F, 2.0F,  // t=1
      0.0F, 3.0F, 1.0F, 1.0F   // t=2
  };

  strix::speculative::QwenDFlashReference::ApplyDynamicConv2Tap(
      sequence, kTokens, kHidden, weight_ref, bias_ref);

  // t=0: x_prev = x_curr
  // c=0: 2.0*1.0 + 0.5*1.0 + 0.1 = 2.6
  // c=1: 1.0*2.0 + (-1.0)*2.0 + (-0.2) = -0.2
  // c=2: 3.0*3.0 + 0.0*3.0 + 0.5 = 9.5
  // c=3: 0.5*4.0 + 1.5*4.0 + 0.0 = 8.0
  ExpectNear(sequence[0], 2.6F, 1e-4F, "t=0, c=0 dynamic conv");
  ExpectNear(sequence[1], -0.2F, 1e-4F, "t=0, c=1 dynamic conv");
  ExpectNear(sequence[2], 9.5F, 1e-4F, "t=0, c=2 dynamic conv");
  ExpectNear(sequence[3], 8.0F, 1e-4F, "t=0, c=3 dynamic conv");

  // t=1: x_prev = t=0 original
  // c=0: 2.0*2.0 + 0.5*1.0 + 0.1 = 4.6
  // c=1: 1.0*1.0 + (-1.0)*2.0 + (-0.2) = -1.2
  ExpectNear(sequence[4], 4.6F, 1e-4F, "t=1, c=0 dynamic conv");
  ExpectNear(sequence[5], -1.2F, 1e-4F, "t=1, c=1 dynamic conv");

  std::cout << "TestDynamicConv2Tap: passed\n";
}

void TestMarkovPathScoring() {
  constexpr std::size_t kNumTokens = 3;
  constexpr std::size_t kVocabSize = 4;
  constexpr std::size_t kRank = 2;

  // Base logits [3, 4]
  std::vector<float> logits = {
      1.0F, 2.0F, 0.5F, 0.1F,   // t=0
      0.5F, 1.0F, 2.0F, 0.2F,   // t=1
      2.0F, 0.5F, 1.0F, 3.0F    // t=2
  };

  // W1 [Rank=2, Vocab=4]
  std::vector<float> w1 = {
      0.5F, 1.0F,   // tok 0
      1.0F, 0.0F,   // tok 1
      0.0F, 2.0F,   // tok 2
      1.0F, 1.0F    // tok 3
  };

  // W2 [Rank=2, Vocab=4]
  std::vector<float> w2 = {
      1.0F, 0.0F,   // row 0
      0.5F, 0.5F,   // row 1
      0.0F, 1.0F,   // row 2
      2.0F, 1.0F    // row 3
  };

  strix::models::QwenTensorRef w1_ref{
      .data = w1.data(),
      .type = strix::core::GgmlType::kF32,
      .num_elements = w1.size(),
      .available_bytes = w1.size() * sizeof(float)};
  strix::models::QwenTensorRef w2_ref{
      .data = w2.data(),
      .type = strix::core::GgmlType::kF32,
      .num_elements = w2.size(),
      .available_bytes = w2.size() * sizeof(float)};
  strix::models::QwenTensorRef d2t_empty{};

  std::vector<strix::tokenization::TokenId> tokens;
  strix::speculative::QwenDFlashReference::ApplyMarkovPathScoring(
      logits, kNumTokens, kVocabSize, kVocabSize, w1_ref, w2_ref, d2t_empty,
      /*anchor_token=*/1, /*sample_from_anchor=*/true, tokens);

  Expect(tokens.size() == kNumTokens, "tokens size matches draft block");
  std::cout << "TestMarkovPathScoring: passed (tokens=[" << tokens[0] << ", "
            << tokens[1] << ", " << tokens[2] << "])\n";
}

}  // namespace

int main(int argc, const char* const* argv) {
  try {
    TestDynamicConv2Tap();
    TestMarkovPathScoring();

    if (argc > 1 && argv[1] != nullptr && std::string_view(argv[1]).size() > 0) {
      std::string error;
      auto reader = strix::core::GgufReader::OpenFile(argv[1], &error);
      if (reader != nullptr) {
        auto ref = strix::speculative::QwenDFlashReference::Create(
            std::shared_ptr<const strix::core::GgufReader>(std::move(reader)), 16, &error);
        if (ref != nullptr) {
          std::cout << "Successfully loaded DFlash model from " << argv[1] << '\n';
        }
      }
    }

    std::cout << "qwen_dflash_reference_test: ALL TESTS PASSED\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Test exception: " << ex.what() << '\n';
    return 1;
  }
}
