// In-pass prompt checkpoints (PrefillThrough) on real weights, in AR and
// DFlash2 modes. Each capture equals, byte for byte and component by
// component, the slot after a prefill step that ends at its boundary; so do
// the completed frontier, a restored capture continued in place, and the
// passes a step cancelled at a pass's start or between its layers keeps.
// Greedy decoding afterwards matches, and a cancelled step keeps the token
// its frontier published.
//
// qwen27b_in_pass_test [MODEL.gguf [DFLASH.gguf]]
// Without arguments, GUFO_QWEN27B_MODEL and GUFO_QWEN27B_DFLASH_MODEL name
// the weights; the test is skipped without a target model and runs AR only
// without a draft.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "gufo/ext/transfer_pool.hpp"
#include "src/cache/adapter.hpp"
#include "src/cache/component_resources.hpp"
#include "src/cache/ledger.hpp"
#include "src/core/sampling.hpp"
#include "src/models/qwen/runner.hpp"

namespace {
namespace cache = gufo::cache;
namespace ext = gufo::ext;
using Token = ext::TextRunnerToken;

constexpr std::uint32_t kContext = 8192;
constexpr std::size_t kBudget = 4096;
// A pass of five rows, the 2,048-row grid checkpoint and a stable boundary
// 505 rows later, which leaves the five-row tail of the issue's prompt.
constexpr std::size_t kBoundaries[] = {5, 2048, 2553};
constexpr std::size_t kEnd = 2558;

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

template<class F>
void Reject(F&& fn, const std::string& message) {
  bool rejected = false;
  try {
    fn();
  } catch (const std::exception&) {
    rejected = true;
  }
  Require(rejected, message);
}

std::string Ledger(std::size_t entries) {
  std::string text = "Inventory ledger.\n";
  for (std::size_t i = 0; i < entries; ++i)
    text += "Entry " + std::to_string(i) + " holds " +
            std::to_string(i * 7919 % 1009) + " units of part " +
            std::to_string(i * 31 % 97) + ".\n";
  return text + "Which part does entry 42 hold? Answer briefly.";
}

void Wait(cache::Completion completion) {
  Require(completion.Wait() == cache::TransferResult::kSucceeded,
          "transfer failed");
}

struct Checkpoint {
  std::vector<cache::ComponentPosition> positions;
  std::vector<std::vector<std::byte>> bytes;
};

Checkpoint Capture(cache::Adapter& adapter, cache::Slot& slot,
                   gufo::hip::TransferPool& pool) {
  Checkpoint checkpoint{adapter.Positions(slot), {}};
  checkpoint.bytes.resize(adapter.Components().size());
  for (std::size_t i = 0; i < checkpoint.bytes.size(); ++i) {
    const auto& descriptor = adapter.Components()[i];
    auto& bytes = checkpoint.bytes[i];
    bytes.assign(descriptor.row_bytes
                     ? checkpoint.positions[i].valid_rows * descriptor.row_bytes
                     : descriptor.state_bytes,
                 std::byte{0x5a});
    auto stream = pool.TryAcquire();
    Require(bool(stream), "transfer pool did not recycle");
    if (descriptor.row_bytes)
      Wait(adapter.CopyRowsOut(slot, descriptor.id, 0,
                               checkpoint.positions[i].valid_rows, bytes,
                               *stream));
    else
      Wait(adapter.CapturePrivate(slot, descriptor.id, bytes, *stream));
  }
  return checkpoint;
}

void Load(cache::Adapter& adapter, cache::Slot& slot,
          const Checkpoint& checkpoint, gufo::hip::TransferPool& pool) {
  adapter.BeginRestore(slot, checkpoint.positions);
  for (std::size_t i = 0; i < checkpoint.bytes.size(); ++i) {
    const auto& descriptor = adapter.Components()[i];
    auto stream = pool.TryAcquire();
    Require(bool(stream), "transfer pool did not recycle");
    if (descriptor.row_bytes)
      Wait(adapter.CopyRowsIn(slot, descriptor.id, 0,
                              checkpoint.positions[i].valid_rows,
                              checkpoint.bytes[i], *stream));
    else
      Wait(adapter.LoadPrivate(slot, descriptor.id, checkpoint.bytes[i],
                               *stream));
  }
  Require(adapter.Validate(slot, checkpoint.positions), "restore rejected");
}

// Names the components whose bytes differ, for a failure message.
std::string Differences(const Checkpoint& actual, const Checkpoint& expected) {
  std::string differing;
  if (actual.positions != expected.positions)
    differing += " [positions]";
  for (std::size_t c = 0; c < expected.bytes.size(); ++c) {
    const auto& x = actual.bytes.at(c);
    const auto& y = expected.bytes[c];
    if (x == y)
      continue;
    std::size_t first = 0, count = 0;
    for (std::size_t j = 0; j < std::min(x.size(), y.size()); ++j)
      if (x[j] != y[j] && count++ == 0)
        first = j;
    differing += " [component " + std::to_string(c) + " bytes " +
                 std::to_string(x.size()) + "/" + std::to_string(y.size()) +
                 " first " + std::to_string(first) + " differing " +
                 std::to_string(count) + "]";
  }
  return differing;
}

void Same(const Checkpoint& actual, const Checkpoint& expected,
          const std::string& what) {
  const auto differing = Differences(actual, expected);
  Require(differing.empty(), what + " differs:" + differing);
}

struct NullGuard final : cache::MutationGuard {
  void BeforeOverwrite(cache::ComponentId, cache::Rows, cache::Rows) override {}
  void BeforeRelease(cache::ComponentId, cache::Rows,
                     cache::Rows) noexcept override {}
};

std::vector<Token> Greedy(const ext::TextModelRunner& runner,
                          ext::TextRunnerState& state, std::size_t count) {
  gufo::sampling::SamplerState sampler;
  std::vector<Token> tokens;
  while (tokens.size() < count) {
    const auto step = runner.DecodeStep(state, count - tokens.size(), sampler);
    for (const auto& selection : step.selections)
      tokens.push_back(selection.token);
    if (step.stop || step.selections.empty())
      break;
  }
  return tokens;
}

std::shared_ptr<ext::TextModelRunner> Create(const std::string& model,
                                             const std::string& draft) {
  ext::TextRunnerOptions options;
  options.model_path = model;
  options.max_context = kContext;
  options.session_count = 2;
  if (!draft.empty()) {
    options.speculative.backend = ext::TextSpeculativeBackend::kDFlash;
    options.speculative.draft_model_path = draft;
  }
  std::string error;
  auto runner = gufo::models::qwen::CreateQwenTextRunner(options, &error);
  if (!runner)
    throw std::runtime_error(error);
  return runner;
}

void Check(const std::string& name, const ext::TextModelRunner& runner) {
  const auto capabilities = runner.Descriptor().capabilities;
  Require(capabilities.in_pass_checkpoint &&
              capabilities.max_in_pass_checkpoints >= std::size(kBoundaries),
          "Qwen27B must capture its prompt checkpoints in place");
  constexpr auto kUnlimited = std::numeric_limits<std::size_t>::max() / 4;
  cache::ResourceLedger ledger(
      {kUnlimited, kUnlimited, kUnlimited, kUnlimited});
  const auto resources = runner.CreateComponentCacheResources(ledger, 0);
  auto& adapter = *resources->adapter;
  gufo::hip::TransferPool pool(4);
  NullGuard source_guard, reference_guard;
  auto source = adapter.CreateSlot(source_guard);
  auto reference = adapter.CreateSlot(reference_guard);
  auto a = runner.BindComponentState(adapter, *source);
  auto b = runner.BindComponentState(adapter, *reference);
  auto prompt = runner.Tokenize(Ledger(200));
  Require(prompt.size() >= kEnd, "in-pass prompt is too short");
  prompt.resize(kEnd);
  const auto first = [&](std::size_t end) {
    return std::span<const Token>(prompt).first(end);
  };

  // The reference ends a step at each boundary, as a runner without in-pass
  // checkpoints does for a prompt snapshot target: a complete frontier.
  std::vector<Checkpoint> split;
  std::size_t at = 0;
  for (const auto boundary : kBoundaries) {
    const auto step = runner.Prefill(*b, first(boundary), at, kBudget);
    Require(step.consumed_tokens == boundary - at && step.decode_ready,
            "reference step did not end at its boundary");
    split.push_back(Capture(adapter, *reference, pool));
    at = boundary;
  }
  Require(runner.Prefill(*b, prompt, at, kBudget).decode_ready,
          "reference prompt incomplete");
  const auto complete = Capture(adapter, *reference, pool);
  const auto expected = Greedy(runner, *b, 24);

  // Claims: only inside the step, never at its ends.
  for (const auto boundary : kBoundaries)
    Require(runner.PrefillCheckpointBytes(*a, prompt, 0, kBudget, boundary)
                .has_value(),
            "in-pass boundary refused");
  for (const std::size_t boundary : {std::size_t{0}, kEnd, kBudget})
    Require(!runner.PrefillCheckpointBytes(*a, prompt, 0, kBudget, boundary),
            "claim outside the step");
  Require(!runner.PrefillCheckpointBytes(*a, prompt, 0, 2000, 2048),
          "claim beyond the step's budget");

  // In place, with the step split into layer groups across its passes.
  std::vector<std::vector<cache::ComponentPosition>> predicted;
  for (const auto boundary : kBoundaries)
    predicted.push_back(adapter.PrefillCheckpointPositions(*source, boundary));
  // Every group boundary lands on its own layer: one yield fewer than groups.
  constexpr std::size_t kGroups = 5;
  std::size_t yields = 0;
  a->SetPrefillYield(kGroups, [&] { ++yields; });
  std::vector<Checkpoint> frozen;
  const auto step = runner.PrefillThrough(
      *a, prompt, 0, kBudget, kBoundaries, [&](std::size_t index) {
        Require(index == frozen.size(), "in-pass captures out of order");
        Require(adapter.Positions(*source) == predicted[index],
                "in-pass positions differ from the admission prediction");
        Reject(
            [&] {
              (void)runner.Prefill(*a, prompt, kBoundaries[index], kBudget);
            },
            "a pass ran while a consumer read its boundary");
        Require(!adapter.Invalidate(*source),
                "a consumer invalidated the frontier it reads");
        frozen.push_back(Capture(adapter, *source, pool));
      });
  a->SetPrefillYield(1, {});
  Require(step.consumed_tokens == kEnd && step.decode_ready &&
              !step.interrupted && frozen.size() == std::size(kBoundaries),
          "in-pass step did not complete the prompt");
  Require(yields == kGroups - 1, "in-pass step yielded " +
                                     std::to_string(yields) + " times, not " +
                                     std::to_string(kGroups - 1));
  for (std::size_t i = 0; i < frozen.size(); ++i)
    Same(frozen[i], split[i],
         "in-pass checkpoint at " + std::to_string(kBoundaries[i]));
  Same(Capture(adapter, *source, pool), complete, "completed frontier");
  Require(Greedy(runner, *a, 24) == expected,
          "greedy decoding after in-pass checkpoints differs");
  std::cout << name << ",in_pass,PASS\n";

  // A restored capture continues in place like the steps it replaces.
  Require(adapter.Invalidate(*source), "reset before restore failed");
  Load(adapter, *source, frozen[1], pool);
  runner.ReconcileComponentState(*a, kBoundaries[1]);
  runner.PreparePrefixReuse(*a, first(kBoundaries[1]));
  std::vector<Checkpoint> resumed;
  const std::size_t rest[] = {kBoundaries[2]};
  Require(runner.PrefillThrough(*a, prompt, kBoundaries[1], kBudget, rest,
                                [&](std::size_t) {
                                  resumed.push_back(
                                      Capture(adapter, *source, pool));
                                })
                  .decode_ready &&
              resumed.size() == 1,
          "restored in-pass step did not complete the prompt");
  Same(resumed.front(), split[2], "restored in-pass checkpoint");
  Same(Capture(adapter, *source, pool), complete, "restored frontier");
  Require(Greedy(runner, *a, 24) == expected,
          "greedy decoding after a restored checkpoint differs");
  std::cout << name << ",in_pass_restore,PASS\n";

  // Cancelling a later pass keeps the passes before it, as their separate
  // steps would: at its start, or between its layers while the step's layer
  // groups yield. Cancelling the first pass rolls the whole step back.
  enum class Cancel { kLaterPassStart, kLaterPassLayers, kFirstPass };
  for (const auto cancel : {Cancel::kLaterPassStart, Cancel::kLaterPassLayers,
                            Cancel::kFirstPass}) {
    Require(adapter.Invalidate(*source), "reset before cancellation failed");
    runner.ReconcileComponentState(*a, 0);
    // The checks the cancelled pass passes first: none, or twelve, its entry
    // and first chunk and then at least one per layer, so that it stops with
    // eight or more layers queued and four of them complete.
    const std::size_t pass_checks = cancel == Cancel::kLaterPassLayers ? 12 : 0;
    bool armed = cancel == Cancel::kFirstPass;
    std::size_t checks = 0;
    a->SetCancellationCheck([&] { return armed && checks++ >= pass_checks; });
    a->SetPrefillYield(kGroups, [] {});
    std::size_t consumed = 0;
    const auto interrupted = runner.PrefillThrough(
        *a, prompt, 0, kBudget, kBoundaries, [&](std::size_t index) {
          ++consumed;
          if (index == 1)
            armed = true;
        });
    a->SetPrefillYield(1, {});
    a->SetCancellationCheck({});
    if (cancel == Cancel::kFirstPass) {
      Require(interrupted.interrupted && !interrupted.consumed_tokens &&
                  runner.CheckpointPosition(*a) == 0 && consumed == 0,
              "a cancelled first pass did not roll the step back");
      continue;
    }
    Require(checks > pass_checks, "the later pass was not cancelled");
    Require(!interrupted.interrupted && !interrupted.decode_ready &&
                interrupted.consumed_tokens == kBoundaries[1] &&
                runner.CheckpointPosition(*a) == kBoundaries[1] &&
                consumed == 2,
            "a cancelled later pass did not keep the passes before it");
    Same(Capture(adapter, *source, pool), split[1],
         "frontier kept by a cancelled step");
  }
  std::cout << name << ",in_pass_cancel,PASS\n";

  // A cancelled step keeps the token its frontier published, as a cancelled
  // Prefill does: a prompt that continues with it still replays it with
  // decode arithmetic, a single-token step.
  Require(adapter.Invalidate(*source), "reset before resume failed");
  runner.ReconcileComponentState(*a, 0);
  constexpr std::size_t kFrontier = kBoundaries[0];
  Require(runner.Prefill(*a, first(kFrontier), 0, kBudget).decode_ready,
          "short prompt incomplete");
  gufo::sampling::SamplerState sampler;
  const auto published = runner.SelectNext(*a, sampler).token;
  runner.PreparePrefixReuse(*a, first(kFrontier));
  std::vector<Token> continued(prompt.begin(), prompt.begin() + 100);
  continued[kFrontier] = published;
  auto diverging = continued;
  diverging[kFrontier] = prompt[kFrontier] != published ? prompt[kFrontier]
                                                        : prompt[kFrontier + 1];
  Require(diverging[kFrontier] != published, "no diverging token");
  const std::size_t inside[] = {kFrontier + 8};
  a->SetCancellationCheck([] { return true; });
  const auto dropped = runner.PrefillThrough(*a, diverging, kFrontier, kBudget,
                                             inside, [](std::size_t) {});
  a->SetCancellationCheck({});
  Require(dropped.interrupted && runner.CheckpointPosition(*a) == kFrontier,
          "a cancelled diverging step moved the frontier");
  Require(
      runner.Prefill(*a, continued, kFrontier, kBudget).consumed_tokens == 1,
      "a cancelled step lost the published token");
  std::cout << name << ",in_pass_resume,PASS\n";
  Require(adapter.Invalidate(*source) && adapter.Invalidate(*reference),
          "final reset failed");
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, draft;
  if (argc > 3) {
    std::cerr << "Usage: qwen27b_in_pass_test [MODEL.gguf [DFLASH.gguf]]\n";
    return 2;
  }
  if (argc > 1) {
    model = argv[1];
    draft = argc > 2 ? argv[2] : "";
  } else {
    const char* target = std::getenv("GUFO_QWEN27B_MODEL");
    const char* head = std::getenv("GUFO_QWEN27B_DFLASH_MODEL");
    if (!target || !*target) {
      std::cerr << "GUFO_QWEN27B_MODEL is not set; skipping\n";
      return 77;
    }
    model = target;
    draft = head ? head : "";
  }
  try {
    // One runner at a time keeps a single copy of the weights resident.
    Check("AR", *Create(model, {}));
    if (!draft.empty())
      Check("DFlash2", *Create(model, draft));
    std::cout << "Qwen27B in-pass checkpoint checks passed.\n";
    return 0;
  } catch (const std::exception& failure) {
    std::cerr << "qwen27b_in_pass_test: " << failure.what() << '\n';
    return 1;
  }
}
