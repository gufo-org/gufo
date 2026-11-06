#include "src/models/deepseek_v4_flash/continuation_adapter.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ds4 = gufo::models::deepseek_v4_flash;
namespace cache = gufo::cache;

namespace {
void Require(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error(message);
}
template<class F>
void Reject(F&& call) {
  bool rejected = false;
  try {
    call();
  } catch (const std::exception&) {
    rejected = true;
  }
  Require(rejected, "invalid adapter operation was accepted");
}
struct Guard final : cache::MutationGuard {
  void BeforeOverwrite(cache::ComponentId, cache::Rows first,
                       cache::Rows end) override {
    overwrites += first < end;
  }
  void BeforeRelease(cache::ComponentId, cache::Rows,
                     cache::Rows) noexcept override {}
  void AfterReset() noexcept override { ++resets; }
  std::size_t overwrites{}, resets{};
};
struct Checkpoint {
  std::vector<cache::ComponentPosition> positions;
  std::vector<std::vector<std::byte>> bytes;
};
void Wait(cache::Completion done) {
  Require(done.Wait() == cache::TransferResult::kSucceeded, "transfer failed");
}
// Odd sizes make every tensor and ring boundary cross a piece boundary.
constexpr std::size_t kPiece = (1 << 18) + 13;
constexpr cache::Rows kRowPiece = 97;

auto Acquire(gufo::hip::TransferPool& streams) {
  auto stream = streams.TryAcquire();
  Require(bool(stream), "stream exhausted");
  return stream;
}

Checkpoint Capture(ds4::ContinuationAdapter& adapter, cache::Slot& slot,
                   gufo::hip::TransferPool& streams) {
  Checkpoint result{adapter.Positions(slot), {}};
  for (const auto& component : adapter.Components()) {
    const auto rows = result.positions[component.id.value - 1].valid_rows;
    auto& bytes = result.bytes.emplace_back(component.row_bytes
                                                ? rows * component.row_bytes
                                                : component.state_bytes);
    if (component.row_bytes) {
      for (cache::Rows first = 0; first < rows;) {
        const auto end = std::min(rows, first + kRowPiece);
        Wait(adapter.CopyRowsOut(
            slot, component.id, first, end,
            std::span(bytes).subspan(first * component.row_bytes,
                                     (end - first) * component.row_bytes),
            *Acquire(streams)));
        first = end;
      }
    } else {
      for (std::size_t offset = 0; offset < bytes.size();) {
        const auto count = std::min(kPiece, bytes.size() - offset);
        Wait(adapter.CapturePrivatePiece(
            slot, component.id, offset, std::span(bytes).subspan(offset, count),
            *Acquire(streams)));
        offset += count;
      }
    }
  }
  return result;
}

// Reverse order: metadata arrives after the device regions, and the first
// piece of each tensor arrives last.
void Load(ds4::ContinuationAdapter& adapter, cache::Slot& slot,
          const Checkpoint& checkpoint, gufo::hip::TransferPool& streams,
          bool omit_last = false) {
  adapter.BeginRestore(slot, checkpoint.positions);
  Require(!slot.IsValid(), "BeginRestore exposed executable state");
  Reject([&] { (void)adapter.GetSession(slot); });
  for (std::size_t i = adapter.Components().size(); i-- > 0;) {
    const auto& component = adapter.Components()[i];
    const auto bytes = std::span<const std::byte>(checkpoint.bytes[i]);
    if (component.row_bytes) {
      for (cache::Rows end = checkpoint.positions[i].valid_rows; end;) {
        const auto first = end > kRowPiece ? end - kRowPiece : 0;
        Wait(adapter.CopyRowsIn(
            slot, component.id, first, end,
            bytes.subspan(first * component.row_bytes,
                          (end - first) * component.row_bytes),
            *Acquire(streams)));
        end = first;
      }
    } else {
      for (std::size_t end = bytes.size(); end;) {
        const auto first = end > kPiece ? end - kPiece : 0;
        auto count = end - first;
        if (omit_last && component.id.value == 2 && !first)
          --count;
        Wait(adapter.LoadPrivatePiece(slot, component.id, first,
                                      bytes.subspan(first, count),
                                      *Acquire(streams)));
        end = first;
      }
    }
  }
}

std::vector<int> Prompt(std::size_t tokens) {
  std::vector<int> prompt(tokens);
  std::uint32_t state = 0x9e3779b9U;
  for (auto& token : prompt) {
    state = state * 1664525U + 1013904223U;
    token = 1000 + static_cast<int>((state >> 8) % 60000);
  }
  return prompt;
}

std::vector<std::uint8_t> Bytes(ds4::Session& session) {
  std::string error;
  const auto snapshot = session.SaveSnapshot(&error);
  Require(bool(snapshot), error);
  const auto bytes = snapshot->bytes();
  return {bytes.begin(), bytes.end()};
}

// Generates through the mode's own decode path and returns its tokens.
std::vector<int> Continue(ds4::Session& session, bool speculative,
                          std::size_t count) {
  std::vector<int> emitted;
  std::string error;
  while (emitted.size() < count) {
    if (speculative) {
      std::vector<int> step;
      Require(session.DsparkStep(count - emitted.size(), &step, &error), error);
      emitted.insert(emitted.end(), step.begin(), step.end());
    } else {
      const int token = session.SelectNext(0.0F, nullptr);
      Require(token >= 0, "greedy selection failed");
      Require(session.Evaluate(token, &error), error);
      emitted.push_back(token);
    }
  }
  return emitted;
}

// Prompt lengths exercise every component shape: below one compressor window,
// a partial sliding window, a wrapped ring, and whole sparse rows with a
// partial frontier.
void CheckRoundTrip(ds4::ContinuationAdapter& adapter, bool speculative,
                    gufo::hip::TransferPool& streams, std::size_t tokens) {
  Guard source_guard, target_guard;
  auto source = adapter.CreateSlot(source_guard);
  auto target = adapter.CreateSlot(target_guard);
  const auto prompt = Prompt(tokens);
  std::string error;
  Require(adapter.GetSession(*source).Sync(prompt, &error), error);
  Require(source_guard.overwrites > 0, "prefill bypassed the write guard");

  // A longer unrelated frontier must be fully replaced by a shorter restore.
  const auto other = Prompt(tokens + 300);
  Require(adapter.GetSession(*target).Sync(
              std::span(other).subspan(7, tokens + 200), &error),
          error);

  const auto checkpoint = Capture(adapter, *source, streams);
  auto& original = adapter.GetSession(*source);
  const auto expected_bytes = Bytes(original);
  Load(adapter, *target, checkpoint, streams);
  Require(adapter.Validate(*target, checkpoint.positions),
          "complete restore failed validation");
  Reject([&] { (void)adapter.Validate(*target, checkpoint.positions); });
  auto& restored = adapter.GetSession(*target);
  Require(Bytes(restored) == expected_bytes,
          "restored state differs from the source session");
  Require(restored.CopyLogits() == original.CopyLogits(),
          "restored logits differ");
  Require(Continue(restored, speculative, 24) ==
              Continue(original, speculative, 24),
          "restored decode diverged from an uninterrupted run");
  Require(Bytes(restored) == Bytes(original),
          "state diverged after decoding from a restore");

  // A speculative frontier is itself a valid checkpoint.
  const auto later = Capture(adapter, *source, streams);
  Require(adapter.Invalidate(*target), "invalidate failed");
  Require(target_guard.resets == 1, "invalidate skipped AfterReset");
  Require(adapter.Positions(*target).front().valid_rows == 0,
          "invalidate left a frontier");
  Load(adapter, *target, later, streams);
  Require(adapter.Validate(*target, later.positions), "later restore failed");
  Require(
      Bytes(adapter.GetSession(*target)) == Bytes(adapter.GetSession(*source)),
      "later restore differs");
}

void CheckRefusals(ds4::ContinuationAdapter& adapter,
                   gufo::hip::TransferPool& streams) {
  Guard guard;
  auto slot = adapter.CreateSlot(guard);
  const auto prompt = Prompt(300);
  std::string error;
  auto& session = adapter.GetSession(*slot);
  Require(session.Sync(prompt, &error), error);
  const auto checkpoint = Capture(adapter, *slot, streams);

  auto snapshot = session.SaveSnapshot(&error);
  Require(bool(snapshot), error);
  Reject([&] { (void)session.RestoreSnapshot(*snapshot, &error); });

  auto inconsistent = checkpoint.positions;
  inconsistent.back().valid_rows += 1;
  Reject([&] { adapter.BeginRestore(*slot, inconsistent); });
  Require(adapter.Positions(*slot) == checkpoint.positions,
          "refused restore changed the slot");

  // A missing byte latches failure until the slot is invalidated.
  Load(adapter, *slot, checkpoint, streams, true);
  Require(!adapter.Validate(*slot, checkpoint.positions),
          "incomplete restore passed validation");
  Require(!slot->IsValid(), "failed restore left an executable slot");
  Require(!adapter.Validate(*slot, checkpoint.positions),
          "a false validation must stay false");
  Reject([&] { (void)adapter.GetSession(*slot); });
  Require(adapter.Invalidate(*slot), "failed restore cannot be invalidated");
  Require(slot->IsValid(), "invalidate must leave a usable empty slot");

  // A cold prefill after failure reproduces the original state.
  Require(adapter.GetSession(*slot).Sync(prompt, &error), error);
  Require(adapter.Positions(*slot) == checkpoint.positions,
          "cold prefill after failure lost its frontier");
}

// Card 17's capacity rows: what one checkpoint costs at depth.
void MeasureCapacity(const std::shared_ptr<ds4::Model>& model,
                     bool speculative) {
  const auto mode = speculative ? gufo::core::SessionMode::kSpeculative
                                : gufo::core::SessionMode::kAutoregressive;
  ds4::ContinuationAdapter adapter(model, mode, model->MaxContext(), {1, 2, 3});
  gufo::hip::TransferPool streams(2);
  Guard source_guard, target_guard;
  auto source = adapter.CreateSlot(source_guard);
  auto target = adapter.CreateSlot(target_guard);
  auto& session = adapter.GetSession(*source);
  const auto prompt = Prompt(100000);
  std::size_t position = 0;
  std::string error;
  using Clock = std::chrono::steady_clock;
  const auto milliseconds = [](Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start)
        .count();
  };
  for (const std::size_t depth : {32768U, 100000U}) {
    while (position < depth) {
      position =
          std::min<std::size_t>(depth, position + session.PrefillCapacity());
      Require(session.Sync(std::span(prompt).first(position), &error), error);
    }
    auto start = Clock::now();
    const auto checkpoint = Capture(adapter, *source, streams);
    const auto capture_ms = milliseconds(start);
    std::size_t rows = 0, state = 0;
    for (std::size_t i = 0; i < checkpoint.bytes.size(); ++i)
      (adapter.Components()[i].row_bytes ? rows : state) +=
          checkpoint.bytes[i].size();
    start = Clock::now();
    Load(adapter, *target, checkpoint, streams);
    Require(adapter.Validate(*target, checkpoint.positions), "restore failed");
    const auto restore_ms = milliseconds(start);
    std::cout << (speculative ? "dspark" : "ar") << " depth=" << depth
              << " private_bytes=" << state << " row_bytes=" << rows
              << " row_bytes_per_token=" << double(rows) / double(depth)
              << " capture_ms=" << capture_ms << " restore_ms=" << restore_ms
              << '\n';
    Require(adapter.Invalidate(*target), "invalidate failed");
  }
}

void CheckMode(const std::shared_ptr<ds4::Model>& model, bool speculative) {
  const auto mode = speculative ? gufo::core::SessionMode::kSpeculative
                                : gufo::core::SessionMode::kAutoregressive;
  ds4::ContinuationAdapter adapter(model, mode, model->MaxContext(), {1, 2, 3});
  Require(adapter.GetCapabilities().persistent_encoding,
          "adapter is not persistent");
  gufo::hip::TransferPool streams(2);
  CheckRefusals(adapter, streams);
  for (const std::size_t tokens : {3U, 100U, 131U, 1031U, 2500U})
    CheckRoundTrip(adapter, speculative, streams, tokens);
}
}  // namespace

int main(int argc, char** argv) {
  const bool capacity = argc == 2 && std::string_view(argv[1]) == "--capacity";
  const char* model_path = std::getenv("GUFO_DEEPSEEK_V4_FLASH_MODEL");
  const char* support = std::getenv("GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL");
  if (!model_path || !*model_path || !support || !*support) {
    std::cout << "SKIP: DeepSeek model paths are not set\n";
    return 77;
  }
  try {
    std::string error;
    auto model = ds4::Model::Load(model_path,
                                  {.max_context = capacity ? 102400U : 8192U,
                                   .dspark_model_path = support},
                                  &error);
    Require(bool(model), error);
    if (capacity) {
      MeasureCapacity(model, false);
      MeasureCapacity(model, true);
      return 0;
    }
    CheckMode(model, false);
    std::cout << "autoregressive continuation checks passed\n";
    CheckMode(model, true);
    std::cout << "DSpark continuation checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
