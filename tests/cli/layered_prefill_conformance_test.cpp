#include "tests/support/layered_prefill_conformance.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tests/cli/fake_cache_resources.hpp"

namespace {

using gufo::ext::TextDecodeSelection;
using gufo::ext::TextModelRunner;
using gufo::ext::TextPrefillStep;
using gufo::ext::TextRunnerDescriptor;
using gufo::ext::TextRunnerResourceClaim;
using gufo::ext::TextRunnerState;
using gufo::ext::TextRunnerToken;
using gufo::ext::TextSchedulingProfile;

void Expect(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "Assertion failed: %s\n", message);
    std::exit(1);
  }
}

// Model state is the cache adapter's hash of every row it holds.
class HashState final : public TextRunnerState {
public:
  HashState(gufo::cache::testing::FakeAdapter& adapter, gufo::cache::Slot& slot)
      : adapter(adapter), slot(slot) {}
  void Invalidate() noexcept override {
    (void)adapter.Invalidate(slot);
    position = 0;
  }
  [[nodiscard]] std::uint64_t Hash() const {
    return adapter.RecurrentHash(slot);
  }

  gufo::cache::testing::FakeAdapter& adapter;
  gufo::cache::Slot& slot;
  std::size_t position{0};
};

enum class Fault : std::uint8_t {
  kNone,
  // Decoding reuses the scratch a pass carries across its layer groups.
  kSharedScratch,
  // The pass never yields.
  kNoYield,
  // Only batched decoding reuses the pass's scratch.
  kBatchedScratch,
};

class HashRunner final : public TextModelRunner {
public:
  explicit HashRunner(Fault fault) : fault_(fault) {}

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = "layered-fake",
        .state_abi = "layered-fake-v1",
        .max_context = 4096,
        .capabilities = {.incremental_prefill = true, .prefix_reuse = false},
        .scheduling =
            TextSchedulingProfile{.pass_tokens = 8, .max_layer_groups = 4}};
  }
  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    return {.resident_weights_bytes = 0,
            .state_capacity_bytes = 8 * 64,
            .per_request_state_bytes = 64,
            .temporary_scratch_bytes = 0,
            .retained_snapshot_capacity_bytes = 1U << 25};
  }
  [[nodiscard]] std::vector<gufo::ext::TextExecutionPlan> SupportedPlans()
      const override {
    return {{.kind = gufo::ext::TextExecutionPlanKind::kSerial,
             .physical_width = 1},
            {.kind = gufo::ext::TextExecutionPlanKind::kBatched,
             .physical_width = 4}};
  }
  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return {static_cast<TextRunnerToken>(text.size())};
  }
  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const gufo::ext::ChatRequest&) const override {
    return std::nullopt;
  }
  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return std::to_string(tokens.size());
  }
  [[nodiscard]] std::unique_ptr<gufo::server::ComponentCacheResources>
  CreateComponentCacheResources(gufo::cache::ResourceLedger& ledger,
                                std::size_t) const override {
    return gufo::server::testing::MakeFakeCacheResources(
        ledger, std::make_unique<gufo::cache::testing::FakeAdapter>(64),
        {.private_blocks = 64,
         .row_backing_bytes = 1024,
         .total_row_bytes = 1U << 24});
  }
  [[nodiscard]] std::unique_ptr<TextRunnerState> BindComponentState(
      gufo::cache::Adapter& adapter, gufo::cache::Slot& slot) const override {
    return std::make_unique<HashState>(
        dynamic_cast<gufo::cache::testing::FakeAdapter&>(adapter), slot);
  }
  void ReconcileComponentState(TextRunnerState& state,
                               std::size_t position) const override {
    dynamic_cast<HashState&>(state).position = position;
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& base, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& state = dynamic_cast<HashState&>(base);
    const auto count = std::min(max_input_tokens, prompt.size() - offset);
    std::vector<TextRunnerToken> rows(prompt.begin() + offset,
                                      prompt.begin() + offset + count);
    // The first layer group leaves its output in scratch for the next.
    scratch_ = rows;
    if (fault_ != Fault::kNoYield)
      for (std::size_t group = 1; group < state.prefill_yield_groups(); ++group)
        state.YieldPrefill();
    state.adapter.Append(state.slot, scratch_, scratch_);
    state.position += count;
    return {.consumed_tokens = count,
            .decode_ready = state.position == prompt.size()};
  }
  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& base,
      gufo::sampling::SamplerState& sampler) const override {
    const auto hash = dynamic_cast<HashState&>(base).Hash();
    const auto token = static_cast<TextRunnerToken>(hash % 1000);
    std::optional<gufo::sampling::TokenLogprobs> logprobs;
    if (sampler.config().logprobs)
      logprobs = gufo::sampling::TokenLogprobs{
          {token, -static_cast<double>(hash % 4096) / 1024.0},
          {{token, -static_cast<double>(hash % 4096) / 1024.0},
           {token + 1, -static_cast<double>(hash % 8192) / 1024.0}}};
    return {.token = token, .piece = "t", .logprobs = std::move(logprobs)};
  }
  void Advance(TextRunnerState& base, TextRunnerToken token) const override {
    auto& state = dynamic_cast<HashState&>(base);
    if (fault_ == Fault::kSharedScratch && !scratch_.empty())
      scratch_.front() = token;
    state.adapter.Append(state.slot, {&token, 1}, {&token, 1});
    ++state.position;
  }
  void AdvanceBatch(
      std::span<const gufo::ext::TextRunnerAdvance> advances) const override {
    if (fault_ == Fault::kBatchedScratch && !scratch_.empty())
      scratch_.front() = advances.front().token;
    for (const auto& advance : advances)
      Advance(advance.state, advance.token);
  }
  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return dynamic_cast<const HashState&>(state).position;
  }

private:
  Fault fault_;
  mutable std::vector<TextRunnerToken> scratch_;
};

gufo::test::LayeredPrefillCase Case() {
  gufo::test::LayeredPrefillCase test;
  for (TextRunnerToken token = 1; token <= 30; ++token)
    test.prompt.push_back(token);
  test.decoder_prompt = {500, 501, 502};
  return test;
}

void TestConformingRunnerPasses() {
  const auto failures = gufo::test::CheckLayeredPrefill(
      std::make_shared<HashRunner>(Fault::kNone), Case());
  for (const auto& failure : failures)
    std::fprintf(stderr, "%s\n", failure.c_str());
  Expect(failures.empty(), "a runner that keeps pass scratch passes");
}

void TestSharedScratchIsCaught() {
  const auto failures = gufo::test::CheckLayeredPrefill(
      std::make_shared<HashRunner>(Fault::kSharedScratch), Case());
  Expect(!failures.empty() && failures.front().starts_with("prompt token 0"),
         "decoding into the pass's scratch changes the prompt's output");
}

void TestMissingYieldIsCaught() {
  const auto failures = gufo::test::CheckLayeredPrefill(
      std::make_shared<HashRunner>(Fault::kNoYield), Case());
  Expect(
      failures.size() == 1 &&
          failures.front() == "the runner never yielded between layer groups",
      "a profiled runner must yield");
}

gufo::test::LayeredPrefillCase BatchedCase() {
  auto test = Case();
  test.more_decoder_prompts = {{600, 601}, {700}};
  return test;
}

void TestBatchedDecodersPass() {
  const auto failures = gufo::test::CheckLayeredPrefill(
      std::make_shared<HashRunner>(Fault::kNone), BatchedCase());
  for (const auto& failure : failures)
    std::fprintf(stderr, "%s\n", failure.c_str());
  Expect(failures.empty(), "batched decoders inside the yields pass");
}

void TestBatchedScratchIsCaught() {
  // A single decoder never takes the batched route.
  Expect(gufo::test::CheckLayeredPrefill(
             std::make_shared<HashRunner>(Fault::kBatchedScratch), Case())
             .empty(),
         "a lone decoder does not batch");
  const auto failures = gufo::test::CheckLayeredPrefill(
      std::make_shared<HashRunner>(Fault::kBatchedScratch), BatchedCase());
  Expect(!failures.empty() && failures.front().starts_with("prompt token 0"),
         "a batched decode into the pass's scratch changes the prompt");
}

void TestCaseValidation() {
  auto test = Case();
  test.prompt.resize(8);
  Expect(
      gufo::test::CheckLayeredPrefill(
          std::make_shared<HashRunner>(Fault::kNone), test) ==
          std::vector<std::string>{"the prompt must span more than one pass"},
      "a one-pass prompt cannot check the split");
}

}  // namespace

int main() {
  TestConformingRunnerPasses();
  TestSharedScratchIsCaught();
  TestMissingYieldIsCaught();
  TestBatchedDecodersPass();
  TestBatchedScratchIsCaught();
  TestCaseValidation();
}
