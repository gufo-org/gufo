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

void TestGroupedDynamicCausalConv() {
  constexpr std::size_t kHidden = 4;
  constexpr std::size_t kTokens = 3;
  constexpr std::size_t kKernel = 2;
  constexpr std::size_t kGroupSize = 2;

  // Layout: [side, tap, channel].
  std::vector<float> base = {
      1.0F, 1.0F, 1.0F, 1.0F,  // side 0, tap 0
      0.5F, 0.5F, 0.5F, 0.5F,  // side 0, tap 1
      0.0F, 0.0F, 0.0F, 0.0F,  // side 1, tap 0
      0.0F, 0.0F, 0.0F, 0.0F,  // side 1, tap 1
  };
  gufo::models::QwenTensorRef base_ref{
      .data = base.data(),
      .type = gufo::core::GgmlType::kF32,
      .num_elements = base.size(),
      .available_bytes = base.size() * sizeof(float)};

  const std::vector<float> sequence = {
      1.0F, 2.0F, 3.0F, 4.0F,  // t=0
      2.0F, 1.0F, 0.0F, 2.0F,  // t=1
      0.0F, 3.0F, 1.0F, 1.0F   // t=2
  };
  // Layout per token: [side, tap, group].
  std::vector<float> dynamic(kTokens * 2 * kKernel * 2, 0.0F);
  dynamic[0] = 0.1F;
  dynamic[1] = 0.2F;
  dynamic[8 + 2] = 0.25F;
  std::vector<float> output(sequence.size());

  gufo::speculative::QwenDFlashReference::ApplyGroupedDynamicCausalConv(
      sequence, dynamic, kTokens, kHidden, kKernel, kGroupSize,
      /*side=*/0, base_ref, output);

  ExpectNear(output[0], 1.1F, 1e-4F, "zero padding at t=0, group 0");
  ExpectNear(output[2], 3.6F, 1e-4F, "dynamic coefficient at t=0, group 1");
  ExpectNear(output[4], 2.75F, 1e-4F, "base plus dynamic previous tap");
  ExpectNear(output[5], 2.5F, 1e-4F, "grouped coefficient sharing");
  ExpectNear(output[8], 1.0F, 1e-4F, "t=2 causal previous tap");

  std::cout << "TestGroupedDynamicCausalConv: passed\n";
}

void TestCandidatePathSelector() {
  constexpr std::size_t kNumTokens = 2;
  constexpr std::size_t kVocabSize = 4;
  constexpr std::size_t kRank = 2;
  constexpr std::size_t kHidden = 2;
  constexpr std::size_t kTopK = 2;

  std::vector<float> logits = {
      3.0F, 0.0F, 2.9F, 0.0F,  // unary prefers token 0
      0.0F, 4.0F, 0.0F, 3.9F,  // unary prefers token 1
  };
  std::vector<float> hidden = {
      1.0F,
      0.0F,
      0.0F,
      1.0F,
  };
  std::vector<float> predecessor = {
      0.0F, 0.0F,  // token 0
      1.0F, 0.0F,  // anchor token 1
      0.0F, 1.0F,  // selected token 2
      0.0F, 0.0F,  // token 3
  };
  std::vector<float> successor = {
      0.0F, 0.0F,  // token 0
      0.0F, 0.0F,  // token 1
      2.0F, 0.0F,  // transition makes token 2 win at position 0
      0.0F, 2.0F,  // transition makes token 3 win at position 1
  };
  std::vector<float> hidden_projection = {
      1.0F,
      0.0F,
      0.0F,
      1.0F,
  };

  gufo::models::QwenTensorRef predecessor_ref{
      .data = predecessor.data(),
      .type = gufo::core::GgmlType::kF32,
      .num_elements = predecessor.size(),
      .available_bytes = predecessor.size() * sizeof(float)};
  gufo::models::QwenTensorRef successor_ref{
      .data = successor.data(),
      .type = gufo::core::GgmlType::kF32,
      .num_elements = successor.size(),
      .available_bytes = successor.size() * sizeof(float)};
  gufo::models::QwenTensorRef hidden_projection_ref{
      .data = hidden_projection.data(),
      .type = gufo::core::GgmlType::kF32,
      .num_elements = hidden_projection.size(),
      .available_bytes = hidden_projection.size() * sizeof(float)};

  std::vector<gufo::tokenization::TokenId> tokens;
  std::vector<float> confidences;
  gufo::speculative::QwenDFlashReference::SelectCandidatePath(
      hidden, logits, kNumTokens, kHidden, kVocabSize, kRank, kTopK,
      predecessor_ref, successor_ref, hidden_projection_ref,
      /*anchor_token=*/1, tokens, &confidences);

  Expect(tokens.size() == kNumTokens, "tokens size matches draft block");
  Expect(tokens[0] == 2, "anchor-conditioned selector chooses token 2");
  Expect(tokens[1] == 3, "path-conditioned selector chooses token 3");
  Expect(confidences.size() == kNumTokens, "selector returns confidences");
  std::cout << "TestCandidatePathSelector: passed\n";
}

}  // namespace

int main(int argc, const char* const* argv) {
  try {
    TestGroupedDynamicCausalConv();
    TestCandidatePathSelector();

    if (argc > 1 && argv[1] != nullptr &&
        std::string_view(argv[1]).size() > 0) {
      std::string error;
      auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
      if (reader != nullptr) {
        auto ref = gufo::speculative::QwenDFlashReference::Create(
            std::shared_ptr<const gufo::core::GgufReader>(std::move(reader)),
            16, &error);
        Expect(ref != nullptr, error);
        const auto& config = ref->GetConfig();
        Expect(config.target_layer_ids ==
                   std::vector<std::uint32_t>({5, 19, 33, 47, 61}),
               "official target layer taps normalize to layer outputs");
        std::cout << "Successfully loaded DFlash-2 model from " << argv[1]
                  << '\n';
      }
    }

    std::cout << "qwen_dflash_reference_test: ALL TESTS PASSED\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Test exception: " << ex.what() << '\n';
    return 1;
  }
}
