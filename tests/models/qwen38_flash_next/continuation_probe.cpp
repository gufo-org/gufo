// Model-local capacity and adapter step baseline, with committed D4 backing.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/hip/committed_backing.hpp"
#include "src/models/qwen38_flash_next/continuation_adapter.hpp"

namespace cache = gufo::cache;
namespace qfn = gufo::models::qwen38_flash_next;
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
    std::string model_path, mtp_path, mode;
    std::uint32_t context = 0, count = 0;
    std::size_t backing_gib = 4;
    bool round_trip = false;
    for (int i = 1; i < argc; ++i) {
      const std::string_view arg = argv[i];
      if (arg == "--round-trip") {
        round_trip = true;
        continue;
      }
      Require(i + 1 < argc, "missing option value");
      const std::string value = argv[++i];
      if (arg == "--model")
        model_path = value;
      else if (arg == "--mtp-model")
        mtp_path = value;
      else if (arg == "--mode")
        mode = value;
      else if (arg == "--context")
        context = static_cast<std::uint32_t>(std::stoul(value));
      else if (arg == "--sessions")
        count = static_cast<std::uint32_t>(std::stoul(value));
      else if (arg == "--backing-gib")
        backing_gib = std::stoul(value);
      else
        throw std::invalid_argument("unknown option");
    }
    Require(!model_path.empty() && !mtp_path.empty() &&
                (mode == "ar" || mode == "mtp") && context > 0 &&
                context <= 100000 && count > 0 && count <= 8 &&
                backing_gib > 0 && backing_gib <= 16,
            "requires --model PATH --mtp-model PATH --mode ar|mtp --context N "
            "--sessions N [--backing-gib N] [--round-trip]");
    Require(count <= 2 || context <= 32768,
            "long high-concurrency histories are excluded from the admitted "
            "envelope");
    Require(!round_trip || count >= 2,
            "cross-slot baseline needs at least two slots");
    Check(hipInit(0));
    hipDeviceProp_t device{};
    Check(hipGetDeviceProperties(&device, 0));
    Require(std::string_view(device.gcnArchName).starts_with("gfx1151"),
            "requires gfx1151");
    std::cout << "device," << device.gcnArchName << ",visible_bytes,"
              << device.totalGlobalMem << '\n';
    std::string error;
    const auto slot_context = round_trip ? context + 128 : context;
    auto model =
        qfn::Model::Load(model_path,
                         {.max_context = slot_context,
                          .mtp_model_path = mode == "mtp" ? mtp_path : "",
                          .decode_concurrency = count},
                         &error);
    Require(bool(model), error);
    const auto session_mode = mode == "mtp"
                                  ? gufo::core::SessionMode::kSpeculative
                                  : gufo::core::SessionMode::kAutoregressive;
    qfn::ContinuationAdapter adapter(model, session_mode, slot_context,
                                     {'p', 'r', 'o', 'b', 'e'});
    std::vector<Guard> guards(count);
    std::vector<std::unique_ptr<cache::Slot>> slots;
    std::size_t allocated = 0;
    for (auto& guard : guards) {
      slots.push_back(adapter.CreateSlot(guard));
      allocated += adapter.GetSession(*slots.back()).AllocatedBytes();
    }
    constexpr std::size_t block_bytes = 64ULL << 20, metadata = 1ULL << 20;
    const auto capacity = backing_gib << 30;
    cache::ResourceLedger ledger(
        {capacity + metadata, capacity + metadata, 0, capacity});
    const auto commit_start = Clock::now();
    gufo::hip::CommittedBackingPool backing(
        ledger, {capacity + metadata, block_bytes, metadata});
    gufo::hip::TransferPool streams(2);
    std::cout << "capacity," << mode << ",context," << slot_context
              << ",sessions," << count << ",resident_bytes,"
              << model->ResidentBytes() << ",live_allocated_bytes," << allocated
              << ",live_claim_bytes,"
              << count * model->SessionBytes(session_mode, slot_context)
              << ",deferred_scratch_bytes," << model->DeferredScratchBytes()
              << ",committed_backing_bytes," << backing.CommittedBytes()
              << ",backing_commit_ms," << Ms(commit_start) << std::endl;
    const auto pattern = model->Tokenize(
        "The quick brown fox jumps over the lazy dog. Strix Halo executes this "
        "deterministic benchmark sequence. ");
    Require(!pattern.empty(), "empty pattern");
    std::vector<std::int32_t> prompt(
        round_trip ? context : context - std::min(context, 128U));
    for (std::size_t i = 0; i < prompt.size(); ++i)
      prompt[i] = pattern[i % pattern.size()];
    for (std::size_t i = 0; i < slots.size(); ++i) {
      for (std::size_t j = 0; j < prompt.size(); ++j)
        prompt[j] = pattern[(j + 5 * i) % pattern.size()];
      Require(adapter.GetSession(*slots[i]).Sync(prompt, &error), error);
    }
    auto& source = adapter.GetSession(*slots[0]);
    if (!round_trip) {
      std::vector<gufo::sampling::SamplerState> samplers;
      samplers.reserve(count);
      for (auto& slot : slots) {
        const auto tokens = adapter.GetSession(*slot).Tokens();
        samplers.emplace_back(
            gufo::sampling::SamplingConfig{.temperature = 0.0F},
            std::vector<gufo::sampling::TokenId>(tokens.begin(), tokens.end()));
      }
      std::vector<qfn::Session::DecodeResult> results(count);
      std::vector<qfn::Session::DecodeRequest> requests;
      for (std::size_t i = 0; i < count; ++i)
        requests.push_back({&adapter.GetSession(*slots[i]), 16, &samplers[i],
                            &results[i], false});
      for (unsigned step = 0; step < 4; ++step)
        Require(qfn::Session::DecodeBatch(requests, &error), error);
    }
    std::size_t live_bytes = 0, drafted = 0;
    for (auto& slot : slots) {
      live_bytes += adapter.GetSession(*slot).AllocatedBytes();
      drafted += adapter.GetSession(*slot).Statistics().drafted;
    }
    if (!round_trip && mode == "mtp")
      Require(drafted > 0, "capacity qualification executed no MTP drafts");
    std::size_t free_bytes = 0, total_bytes = 0;
    Check(hipMemGetInfo(&free_bytes, &total_bytes));
    const auto live_claim =
        count * model->SessionBytes(session_mode, slot_context);
    const auto remaining_claim =
        live_claim > live_bytes ? live_claim - live_bytes : 0;
    Require(free_bytes >= remaining_claim + model->DeferredScratchBytes(),
            "capacity does not leave space for worst-case rollback and "
            "deferred scratch");
    std::cout << "capacity_exercised," << mode << ",context," << slot_context
              << ",sessions," << count << ",prefilled_tokens_per_slot,"
              << prompt.size() << ",actual_live_bytes," << live_bytes
              << ",remaining_live_claim," << remaining_claim
              << ",deferred_scratch_bytes," << model->DeferredScratchBytes()
              << ",hip_used_bytes," << total_bytes - free_bytes
              << ",hip_free_bytes," << free_bytes << ",drafted," << drafted
              << ",PASS" << std::endl;
    if (!round_trip)
      return 0;
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
    auto& destination = adapter.GetSession(*slots[1]);
    Require(std::ranges::equal(source.Tokens(), destination.Tokens()) &&
                std::memcmp(source.Logits().data(), destination.Logits().data(),
                            source.Logits().size_bytes()) == 0,
            "restored host state differs");
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
    const auto initial = source.Tokens();
    const std::vector<gufo::sampling::TokenId> history(initial.begin(),
                                                       initial.end());
    gufo::sampling::SamplerState first_sampler({.temperature = 0.0F}, history);
    gufo::sampling::SamplerState second_sampler({.temperature = 0.0F}, history);
    qfn::Session::DecodeResult first_step, second_step;
    Require(source.DecodeStep(16, first_sampler, &first_step, &error, false),
            error);
    Require(
        destination.DecodeStep(16, second_sampler, &second_step, &error, false),
        error);
    Require(
        first_step.tokens == second_step.tokens &&
            std::memcmp(source.Logits().data(), destination.Logits().data(),
                        source.Logits().size_bytes()) == 0 &&
            source.Statistics().drafted == destination.Statistics().drafted &&
            source.Statistics().accepted == destination.Statistics().accepted,
        "next-step logits or drafts differ after restore");
    if (mode == "mtp")
      Require(source.Statistics().drafted > 0,
              "restored MTP executed no drafts");
    std::cout << "baseline," << mode << ",tokens," << context
              << ",private_bytes," << private_bytes << ",row_bytes,"
              << row_bytes << ",private_capture_ms," << private_capture_ms
              << ",full_capture_ms," << capture_ms << ",restore_ms,"
              << restore_ms << ",assigned_backing_bytes,"
              << blocks.size() * block_bytes << ",PASS" << std::endl;
    for (const auto& component : adapter.Components())
      std::cout << "component," << component.id.value << ",row_bytes,"
                << component.row_bytes << ",private_bytes,"
                << component.state_bytes << ",rows,"
                << positions[component.id.value - 1].valid_rows << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
