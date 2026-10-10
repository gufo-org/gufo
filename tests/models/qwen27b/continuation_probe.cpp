// Model-local capacity and adapter step baseline, with committed D4 backing.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/committed_backing.hpp"
#include "src/models/qwen/continuation_adapter.hpp"

namespace cache = gufo::cache;
namespace qwen = gufo::models::qwen;
namespace {
using Clock = std::chrono::steady_clock;
void Require(bool ok, const std::string& error) {
  if (!ok)
    throw std::runtime_error(error);
}
void Check(hipError_t result) {
  Require(result == hipSuccess, hipGetErrorString(result));
}
void Wait(cache::Completion done) {
  Require(done.Wait() == cache::TransferResult::kSucceeded, "transfer failed");
}
double Ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}
struct Guard final : cache::MutationGuard {
  void BeforeOverwrite(cache::ComponentId, cache::Rows, cache::Rows) override {}
  void BeforeRelease(cache::ComponentId, cache::Rows,
                     cache::Rows) noexcept override {}
};
struct Piece {
  cache::ComponentId id;
  bool row;
  std::size_t block, begin, bytes, first, end;
};
}  // namespace
int main(int argc, char** argv) {
  try {
    Require(argc == 3, "Usage: continuation_probe MODEL.gguf DEPTH");
    const auto depth = static_cast<std::uint32_t>(std::stoul(argv[2]));
    Require(depth == 32768 || depth == 100000, "depth must be 32768 or 100000");
    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
    Require(bool(reader), error);
    auto model = gufo::hip::QwenGpuModel::CreateFromGguf(
        std::shared_ptr<const gufo::core::GgufReader>(std::move(reader)),
        &error);
    Require(bool(model), error);
    qwen::ContinuationAdapter adapter(model, depth + 128,
                                      {'p', 'r', 'o', 'b', 'e'});
    Guard first_guard, second_guard;
    std::vector<std::unique_ptr<cache::Slot>> slots;
    slots.push_back(adapter.CreateSlot(first_guard));
    slots.push_back(adapter.CreateSlot(second_guard));
    constexpr std::size_t block_bytes = 64ULL << 20, metadata = 1ULL << 20,
                          capacity = 8ULL << 30;
    cache::ResourceLedger ledger(
        {capacity + metadata, capacity + metadata, 0, capacity});
    const auto commit_start = Clock::now();
    gufo::hip::CommittedBackingPool backing(
        ledger, {capacity + metadata, block_bytes, metadata});
    gufo::hip::TransferPool streams(2);
    const auto a_usage = adapter.GetExecutor(*slots[0]).GetMemoryUsage();
    const auto b_usage = adapter.GetExecutor(*slots[1]).GetMemoryUsage();
    std::cout << "capacity,AR,context," << depth + 128
              << ",sessions,2,resident_bytes," << model->GetResidentBytes()
              << ",request_state_bytes,"
              << a_usage.request_state_bytes + b_usage.request_state_bytes
              << ",scratch_bytes,"
              << a_usage.temporary_scratch_bytes +
                     b_usage.temporary_scratch_bytes
              << ",committed_backing_bytes," << backing.CommittedBytes()
              << ",backing_commit_ms," << Ms(commit_start) << std::endl;
    const auto pattern = model->GetTokenizer().Encode(
        "The quick brown fox jumps over the lazy dog. Strix Halo executes this "
        "deterministic benchmark sequence. ");
    Require(!pattern.empty(), "empty token fixture");
    std::vector<gufo::tokenization::TokenId> prompt(depth);
    for (std::size_t i = 0; i < prompt.size(); ++i)
      prompt[i] = pattern[i % pattern.size()];
    auto& source = adapter.GetExecutor(*slots[0]);
    (void)source.ForwardPromptBatch(prompt);
    const auto positions = adapter.Positions(*slots[0]);
    std::vector<gufo::hip::BackingBlock> blocks;
    std::vector<Piece> pieces;
    std::size_t cursor = block_bytes;
    std::size_t private_bytes = 0, row_bytes = 0;
    double private_capture_ms = 0;
    const auto capture_start = Clock::now();
    // Pack the representation into already committed blocks. No full-payload
    // staging allocation, or one wasteful slab per small private component.
    for (const auto& component : adapter.Components()) {
      const auto component_start = Clock::now();
      const auto rows = positions[component.id.value - 1].valid_rows;
      const auto total = component.row_bytes ? rows * component.row_bytes
                                             : component.state_bytes;
      if (component.row_bytes)
        row_bytes += total;
      else
        private_bytes += total;
      std::size_t offset = 0;
      while (offset < total) {
        const auto unit = component.row_bytes ? component.row_bytes : 1;
        if (block_bytes - cursor < unit) {
          auto block =
              backing.TryAcquire(cache::ResourceCategory::kPrivateTail);
          Require(bool(block), "checkpoint exceeds committed backing");
          blocks.push_back(std::move(*block));
          cursor = 0;
        }
        const auto bytes =
            std::min(total - offset, (block_bytes - cursor) / unit * unit);
        auto stream = streams.TryAcquire();
        Require(bool(stream), "stream exhausted");
        auto buffer = blocks.back().Bytes().subspan(cursor, bytes);
        if (component.row_bytes)
          Wait(adapter.CopyRowsOut(*slots[0], component.id, offset / unit,
                                   (offset + bytes) / unit, buffer, *stream));
        else
          Wait(adapter.CapturePrivatePiece(*slots[0], component.id, offset,
                                           buffer, *stream));
        pieces.push_back(
            {component.id, bool(component.row_bytes), blocks.size() - 1, cursor,
             bytes, component.row_bytes ? offset / unit : offset,
             component.row_bytes ? (offset + bytes) / unit : offset + bytes});
        cursor += bytes;
        offset += bytes;
      }
      if (!component.row_bytes)
        private_capture_ms += Ms(component_start);
    }
    const auto capture_ms = Ms(capture_start);
    for (auto& block : blocks)
      block.Convert();
    const auto restore_start = Clock::now();
    adapter.BeginRestore(*slots[1], positions);
    for (const auto& piece : pieces) {
      auto stream = streams.TryAcquire();
      Require(bool(stream), "stream exhausted");
      const auto buffer =
          blocks[piece.block].Bytes().subspan(piece.begin, piece.bytes);
      if (piece.row)
        Wait(adapter.CopyRowsIn(*slots[1], piece.id, piece.first, piece.end,
                                buffer, *stream));
      else
        Wait(adapter.LoadPrivatePiece(*slots[1], piece.id, piece.first, buffer,
                                      *stream));
    }
    Require(adapter.Validate(*slots[1], positions), "complete restore failed");
    const auto restore_ms = Ms(restore_start);
    auto& destination = adapter.GetExecutor(*slots[1]);
    auto staging = backing.TryAcquire(cache::ResourceCategory::kPrivateTail);
    Require(bool(staging), "no verification staging block available");
    for (const auto& piece : pieces) {
      auto stream = streams.TryAcquire();
      Require(bool(stream), "stream exhausted");
      auto buffer = staging->Bytes().first(piece.bytes);
      if (piece.row)
        Wait(adapter.CopyRowsOut(*slots[1], piece.id, piece.first, piece.end,
                                 buffer, *stream));
      else
        Wait(adapter.CapturePrivatePiece(*slots[1], piece.id, piece.first,
                                         buffer, *stream));
      Require(std::ranges::equal(buffer, blocks[piece.block].Bytes().subspan(
                                             piece.begin, piece.bytes)),
              "restored device component differs");
    }
    for (std::uint32_t i = 0; i < 16; ++i) {
      const auto logits = source.CopyLastLogits();
      const auto other = destination.CopyLastLogits();
      Require(logits.size() == other.size() &&
                  std::memcmp(logits.data(), other.data(),
                              logits.size_bytes()) == 0,
              "continuation logits differ");
      const auto token = static_cast<gufo::tokenization::TokenId>(
          std::ranges::max_element(logits) - logits.begin());
      Require(source.ForwardToken(token, depth + i) ==
                  destination.ForwardToken(token, depth + i),
              "continuation token differs");
    }
    std::size_t free_bytes = 0, total_bytes = 0;
    Check(hipMemGetInfo(&free_bytes, &total_bytes));
    std::cout << "baseline,AR,tokens," << depth << ",private_bytes,"
              << private_bytes << ",row_bytes," << row_bytes
              << ",private_capture_ms," << private_capture_ms
              << ",full_capture_ms," << capture_ms << ",restore_ms,"
              << restore_ms << ",assigned_backing_bytes,"
              << blocks.size() * block_bytes << ",hip_free_bytes," << free_bytes
              << ",PASS" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
