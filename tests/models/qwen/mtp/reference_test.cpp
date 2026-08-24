#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/mtp_reference.hpp"

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

bool IsFinite(std::span<const float> values) {
  return std::ranges::all_of(values,
                             [](float value) { return std::isfinite(value); });
}

bool BitwiseEqual(std::span<const float> lhs, std::span<const float> rhs) {
  return lhs.size() == rhs.size() &&
         std::memcmp(lhs.data(), rhs.data(), lhs.size_bytes()) == 0;
}

std::vector<float> MakeTargetHidden(std::size_t size) {
  std::vector<float> hidden(size);
  for (std::size_t index = 0; index < hidden.size(); ++index) {
    const auto centered = static_cast<int>(index % 257U) - 128;
    hidden[index] = static_cast<float>(centered) / 256.0F;
  }
  return hidden;
}

}  // namespace

int main() {
  const char* model_path = std::getenv("STRIX_MTP_MODEL");
  if (model_path == nullptr || std::string_view(model_path).empty()) {
    std::cout << "qwen_mtp_reference_test: skipped "
                 "(STRIX_MTP_MODEL not set)\n";
    return kSkipped;
  }

  std::string error;
  auto reader = strix::core::GgufReader::OpenFile(model_path, &error);
  if (reader == nullptr) {
    std::cerr << "Failed to open MTP model: " << error << '\n';
    return 1;
  }
  std::shared_ptr<const strix::core::GgufReader> shared_reader{
      std::move(reader)};
  auto reference =
      strix::speculative::QwenMtpReference::Create(shared_reader, 16, &error);
  if (reference == nullptr) {
    std::cerr << "Failed to create MTP reference: " << error << '\n';
    return 1;
  }

  const auto& config = reference->GetWeights().config;
  Expect(config.hidden_size == 5120, "MTP hidden size");
  Expect(config.intermediate_size == 17408, "MTP intermediate size");
  Expect(config.vocab_size == 248320, "MTP vocabulary size");

  const auto target_hidden = MakeTargetHidden(config.hidden_size);
  const auto first = reference->ForwardHidden(17, target_hidden, 0);
  Expect(first.size() == config.hidden_size, "first hidden size");
  Expect(IsFinite(first), "first hidden is finite");
  const std::vector<float> first_copy(first.begin(), first.end());
  Expect(std::ranges::any_of(first_copy,
                             [](float value) { return value != 0.0F; }),
         "first hidden is nonzero");
  const float first_logit_0 = reference->ComputeLogit(0);
  const float first_logit_42 = reference->ComputeLogit(42);
  Expect(std::isfinite(first_logit_0), "first token-zero logit is finite");
  Expect(std::isfinite(first_logit_42), "first token-42 logit is finite");

  struct HiddenFixture {
    std::size_t index;
    float value;
  };
  constexpr std::array<HiddenFixture, 14> kHiddenFixtures{{
      {0, -0.5464083552F},
      {1, -1.3646016121F},
      {2, -2.5607542992F},
      {3, -1.1097238064F},
      {15, -2.6411702633F},
      {31, 0.1561293751F},
      {63, -1.7275234461F},
      {127, -3.0553896427F},
      {255, -0.2710255682F},
      {511, 2.6408267021F},
      {1023, 1.1629519463F},
      {2047, 2.7863371372F},
      {4095, 3.5756802559F},
      {5119, 1.0536026955F},
  }};
  for (const auto& fixture : kHiddenFixtures) {
    ExpectNear(first_copy[fixture.index], fixture.value, 2.0e-2F,
               "independent position-zero hidden fixture");
  }
  constexpr std::array<HiddenFixture, 6> kLogitFixtures{{
      {0, 5.2424373627F},
      {1, 5.7938537598F},
      {17, 12.8167362213F},
      {42, 9.6316232681F},
      {198, 6.9586696625F},
      {248319, -4.9884157181F},
  }};
  for (const auto& fixture : kLogitFixtures) {
    ExpectNear(
        reference->ComputeLogit(static_cast<std::uint32_t>(fixture.index)),
        fixture.value, 5.0e-2F, "independent position-zero logit fixture");
  }
  double norm_squared = 0.0;
  double sum = 0.0;
  for (const float value : first_copy) {
    norm_squared += static_cast<double>(value) * value;
    sum += value;
  }
  ExpectNear(static_cast<float>(std::sqrt(norm_squared)), 158.3712768555F,
             1.0e-1F, "independent position-zero hidden norm");
  ExpectNear(static_cast<float>(sum), -374.1975665156F, 5.0e-1F,
             "independent position-zero hidden sum");

  Expect(reference->ForwardHidden(42, first_copy, 3).empty(),
         "non-sequential position is rejected");
  const auto second = reference->ForwardHidden(42, first_copy, 1);
  Expect(second.size() == config.hidden_size, "second hidden size");
  Expect(IsFinite(second), "second hidden is finite");
  Expect(!BitwiseEqual(first_copy, second),
         "second hidden differs from first hidden");

  reference->Reset();
  const auto repeated = reference->ForwardHidden(17, target_hidden, 0);
  Expect(BitwiseEqual(first_copy, repeated),
         "reset restores deterministic first position");
  Expect(reference->ComputeLogit(0) == first_logit_0,
         "reset restores token-zero logit");
  Expect(reference->ComputeLogit(42) == first_logit_42,
         "reset restores token-42 logit");

  std::cout << "qwen_mtp_reference_test: passed\n";
  return 0;
}
