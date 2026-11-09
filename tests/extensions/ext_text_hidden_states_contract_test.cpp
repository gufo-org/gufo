// Text hidden-state encoding contract, as an external consumer sees it.
//
// Links only `gufo_ext_runtime` and includes only <gufo/ext/...>, like an
// image extension that uses a registered text model as its frozen encoder.
// Numerical evidence for the Qwen adapter lives in
// tests/models/qwen/hip/text_hidden_states_test.cpp.

#include <cstddef>
#include <cstdio>
#include <functional>
#include <gufo/ext/text.hpp>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using gufo::ext::TextHiddenStateEncoding;
using gufo::ext::TextHiddenStateRequest;
using gufo::ext::TextHiddenStates;
using gufo::ext::TextHiddenStateStatus;
using gufo::ext::TextRunnerToken;

int failures = 0;

void Expect(bool condition, const char* what) {
  if (!condition) {
    ++failures;
    std::fprintf(stderr, "FAILED: %s\n", what);
  }
}

// The smallest complete text runner: every required operation and nothing
// else, so the hidden-state methods keep their defaults.
class PlainRunner : public gufo::ext::TextModelRunner {
public:
  [[nodiscard]] gufo::ext::TextRunnerDescriptor Descriptor() const override {
    gufo::ext::TextRunnerDescriptor descriptor;
    descriptor.model_id = "plain";
    descriptor.state_abi = "plain";
    descriptor.max_context = 8;
    return descriptor;
  }
  [[nodiscard]] gufo::ext::TextRunnerResourceClaim ResourceClaim()
      const override {
    return {};
  }
  [[nodiscard]] std::vector<gufo::ext::TextExecutionPlan> SupportedPlans()
      const override {
    return {{}};
  }
  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return std::vector<TextRunnerToken>(text.size(), 1);
  }
  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const gufo::ext::ChatRequest&) const override {
    return std::nullopt;
  }
  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken>) const override {
    return {};
  }
  [[nodiscard]] std::unique_ptr<gufo::ext::ComponentCacheResources>
  CreateComponentCacheResources(gufo::cache::ResourceLedger&,
                                std::size_t) const override {
    throw std::logic_error("not used");
  }
  [[nodiscard]] std::unique_ptr<gufo::ext::TextRunnerState> BindComponentState(
      gufo::cache::Adapter&, gufo::cache::Slot&) const override {
    throw std::logic_error("not used");
  }
  void ReconcileComponentState(gufo::ext::TextRunnerState&,
                               std::size_t) const override {
    throw std::logic_error("not used");
  }
  [[nodiscard]] gufo::ext::TextPrefillStep Prefill(
      gufo::ext::TextRunnerState&, std::span<const TextRunnerToken>,
      std::size_t, std::size_t) const override {
    throw std::logic_error("not used");
  }
  [[nodiscard]] gufo::ext::TextDecodeSelection SelectNext(
      gufo::ext::TextRunnerState&,
      gufo::sampling::SamplerState&) const override {
    throw std::logic_error("not used");
  }
  void Advance(gufo::ext::TextRunnerState&, TextRunnerToken) const override {
    throw std::logic_error("not used");
  }
  [[nodiscard]] std::size_t CheckpointPosition(
      const gufo::ext::TextRunnerState&) const override {
    return 0;
  }
};

// A conforming encoder with a known answer: row i, column c holds
// token_i * 100 + c, so order and layout are observable.
class EncoderRunner final : public PlainRunner {
public:
  static constexpr std::size_t kWidth = 3;
  static constexpr std::size_t kMaxTokens = 4;
  static constexpr TextRunnerToken kVocabulary = 50;

  [[nodiscard]] std::optional<TextHiddenStateEncoding> HiddenStateEncoding()
      const override {
    return TextHiddenStateEncoding{.width = kWidth, .max_tokens = kMaxTokens};
  }
  [[nodiscard]] TextHiddenStateStatus EncodeHiddenStates(
      const TextHiddenStateRequest& request, TextHiddenStates* states,
      std::string* error) const override {
    *states = {};
    const auto reject = [&](TextHiddenStateStatus status, const char* why) {
      if (error != nullptr)
        *error = why;
      return status;
    };
    if (request.tokens.empty() || request.tokens.size() > kMaxTokens)
      return reject(TextHiddenStateStatus::kInvalidInput, "token count");
    for (const auto token : request.tokens) {
      if (token >= kVocabulary)
        return reject(TextHiddenStateStatus::kInvalidInput, "vocabulary");
    }
    if (request.is_cancelled && request.is_cancelled())
      return reject(TextHiddenStateStatus::kCancelled, "cancelled");
    states->rows = request.tokens.size();
    states->width = kWidth;
    for (const auto token : request.tokens) {
      for (std::size_t column = 0; column < kWidth; ++column)
        states->values.push_back(static_cast<float>(token * 100 + column));
    }
    return TextHiddenStateStatus::kOk;
  }
};

std::shared_ptr<gufo::ext::TextModelRunner> CreateEncoder(
    const gufo::ext::TextRunnerOptions&, std::string*) {
  return std::make_shared<EncoderRunner>();
}

void CheckUnsupportedIsNotAFailure() {
  const PlainRunner runner;
  Expect(!runner.HiddenStateEncoding().has_value(),
         "runners do not offer hidden-state encoding by default");
  const std::vector<TextRunnerToken> tokens{1, 2};
  TextHiddenStates states;
  bool unsupported = false;
  try {
    (void)runner.EncodeHiddenStates({.tokens = tokens, .is_cancelled = {}},
                                    &states, nullptr);
  } catch (const std::logic_error&) {
    unsupported = true;
  }
  Expect(unsupported,
         "encoding on an unsupported runner throws instead of returning a "
         "failure status");
}

void CheckLayoutOrderAndOwnership() {
  gufo::ext::RegisterTextProvider(
      {.id = "test-hidden-encoder", .create = &CreateEncoder});
  const auto* provider = gufo::ext::FindTextProvider("test-hidden-encoder");
  Expect(provider != nullptr && provider->create != nullptr,
         "an encoder is created through the public provider factory");
  if (provider == nullptr || provider->create == nullptr)
    return;
  std::string error;
  auto runner = provider->create({}, &error);
  const auto encoding = runner->HiddenStateEncoding();
  Expect(encoding.has_value() && encoding->width == EncoderRunner::kWidth &&
             encoding->max_tokens == EncoderRunner::kMaxTokens,
         "the capability reports width and prompt limit before encoding");

  const std::vector<TextRunnerToken> tokens{7, 3, 9};
  TextHiddenStates states;
  Expect(
      runner->EncodeHiddenStates({.tokens = tokens, .is_cancelled = {}},
                                 &states, &error) == TextHiddenStateStatus::kOk,
      "a valid prompt encodes");
  // The result must stay usable after its producer is gone.
  runner.reset();
  Expect(states.rows == tokens.size() &&
             states.width == EncoderRunner::kWidth &&
             states.values.size() == states.rows * states.width,
         "states are rows x width");
  bool ordered = true;
  for (std::size_t row = 0; row < states.rows; ++row) {
    for (std::size_t column = 0; column < states.width; ++column) {
      ordered = ordered && states.values[row * states.width + column] ==
                               static_cast<float>(tokens[row] * 100 + column);
    }
  }
  Expect(ordered, "row i is token i, stored row-major");
  const TextHiddenStates copy = states;
  Expect(copy.values == states.values && copy.values.data() != nullptr &&
             copy.values.data() != states.values.data(),
         "states are an owned value");
}

void CheckInvalidInputAndCancellation() {
  const EncoderRunner runner;
  TextHiddenStates states{.rows = 1, .width = 1, .values = {1.0F}};
  std::string error;
  Expect(runner.EncodeHiddenStates({}, &states, &error) ==
                 TextHiddenStateStatus::kInvalidInput &&
             states.values.empty() && !error.empty(),
         "an empty prompt is invalid input and leaves no states");
  const std::vector<TextRunnerToken> too_long(EncoderRunner::kMaxTokens + 1, 1);
  Expect(runner.EncodeHiddenStates({.tokens = too_long, .is_cancelled = {}},
                                   &states, nullptr) ==
             TextHiddenStateStatus::kInvalidInput,
         "a prompt above max_tokens is invalid input");
  const std::vector<TextRunnerToken> outside{EncoderRunner::kVocabulary};
  Expect(runner.EncodeHiddenStates({.tokens = outside, .is_cancelled = {}},
                                   &states, nullptr) ==
             TextHiddenStateStatus::kInvalidInput,
         "a token outside the vocabulary is invalid input");
  const std::vector<TextRunnerToken> tokens{1};
  Expect(runner.EncodeHiddenStates(
             {.tokens = tokens, .is_cancelled = [] { return true; }}, &states,
             nullptr) == TextHiddenStateStatus::kCancelled &&
             states.values.empty(),
         "cancellation is its own status and leaves no states");
}

}  // namespace

int main() {
  CheckUnsupportedIsNotAFailure();
  CheckLayoutOrderAndOwnership();
  CheckInvalidInputAndCancellation();
  return failures == 0 ? 0 : 1;
}
