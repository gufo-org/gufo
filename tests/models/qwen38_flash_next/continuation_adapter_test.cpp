#include "src/models/qwen38_flash_next/continuation_adapter.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qfn = gufo::models::qwen38_flash_next;
namespace cache = gufo::cache;
namespace sampling = gufo::sampling;
void FailNextResetSubmission();
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
struct IncompatibleStream final : cache::Stream {
  cache::TransferResult Synchronize() noexcept override {
    return cache::TransferResult::kSucceeded;
  }
};
struct Checkpoint {
  std::vector<cache::ComponentPosition> positions;
  std::vector<std::vector<std::byte>> bytes;
};
void Wait(cache::Completion done) {
  Require(done.Wait() == cache::TransferResult::kSucceeded, "transfer failed");
}
Checkpoint Capture(qfn::ContinuationAdapter& adapter, cache::Slot& slot,
                   gufo::hip::TransferPool& streams) {
  Checkpoint result{adapter.Positions(slot), {}};
  constexpr std::size_t piece = (1 << 20) + 13;
  for (const auto& component : adapter.Components()) {
    const auto rows = result.positions[component.id.value - 1].valid_rows;
    auto& bytes = result.bytes.emplace_back(component.row_bytes
                                                ? rows * component.row_bytes
                                                : component.state_bytes);
    if (component.row_bytes) {
      for (cache::Rows first = 0; first < rows;) {
        const auto end = std::min(rows, first + 97);
        auto stream = streams.TryAcquire();
        Require(bool(stream), "stream exhausted");
        Wait(adapter.CopyRowsOut(
            slot, component.id, first, end,
            std::span(bytes).subspan(first * component.row_bytes,
                                     (end - first) * component.row_bytes),
            *stream));
        first = end;
      }
    } else {
      for (std::size_t offset = 0; offset < bytes.size();) {
        const auto count = std::min(piece, bytes.size() - offset);
        auto stream = streams.TryAcquire();
        Require(bool(stream), "stream exhausted");
        Wait(adapter.CapturePrivatePiece(
            slot, component.id, offset, std::span(bytes).subspan(offset, count),
            *stream));
        offset += count;
      }
    }
  }
  return result;
}
void Load(qfn::ContinuationAdapter& adapter, cache::Slot& slot,
          const Checkpoint& checkpoint, gufo::hip::TransferPool& streams,
          bool omit_last = false) {
  adapter.BeginRestore(slot, checkpoint.positions);
  Require(!slot.IsValid(), "BeginRestore exposed executable state");
  Reject([&] { (void)adapter.GetSession(slot); });
  // Reverse component and piece order: metadata arrives after the device
  // regions, and the first piece of each tensor arrives last.
  constexpr std::size_t piece = (1 << 20) + 13;
  for (std::size_t i = adapter.Components().size(); i-- > 0;) {
    const auto& component = adapter.Components()[i];
    const auto& bytes = checkpoint.bytes[i];
    if (component.row_bytes) {
      const auto rows = checkpoint.positions[i].valid_rows;
      for (cache::Rows end = rows; end;) {
        const auto first = end > 97 ? end - 97 : 0;
        auto stream = streams.TryAcquire();
        Require(bool(stream), "stream exhausted");
        Wait(adapter.CopyRowsIn(
            slot, component.id, first, end,
            std::span(bytes).subspan(first * component.row_bytes,
                                     (end - first) * component.row_bytes),
            *stream));
        end = first;
      }
    } else {
      for (std::size_t end = bytes.size(); end;) {
        const auto first = end > piece ? end - piece : 0;
        auto stream = streams.TryAcquire();
        Require(bool(stream), "stream exhausted");
        auto count = end - first;
        if (omit_last && component.id.value == 2 && !first)
          --count;
        Wait(adapter.LoadPrivatePiece(slot, component.id, first,
                                      std::span(bytes).subspan(first, count),
                                      *stream));
        end = first;
      }
    }
  }
}
struct Guard final : cache::MutationGuard {
  qfn::ContinuationAdapter* adapter{};
  cache::Slot* slot{};
  gufo::hip::TransferPool* streams{};
  const Checkpoint* borrowed{};
  std::vector<std::vector<std::byte>> preserved;
  std::vector<bool> detached;
  std::size_t overwrites{}, releases{}, resets{};
  bool release_failed{false};
  void Borrow(const Checkpoint& checkpoint) {
    borrowed = &checkpoint;
    preserved.resize(checkpoint.bytes.size());
    detached.assign(checkpoint.bytes.size(), false);
  }
  void BeforeOverwrite(cache::ComponentId id, cache::Rows first,
                       cache::Rows end) override {
    ++overwrites;
    if (!borrowed || detached[id.value - 1] || !end ||
        first >= borrowed->positions[id.value - 1].valid_rows)
      return;
    const auto& component = adapter->Components()[id.value - 1];
    if (!component.row_bytes)
      return;
    const auto rows = borrowed->positions[id.value - 1].valid_rows;
    auto& bytes = preserved[id.value - 1];
    bytes.resize(rows * component.row_bytes);
    auto stream = streams->TryAcquire();
    Require(bool(stream), "guard stream exhausted");
    Wait(adapter->CopyRowsOut(*slot, id, 0, rows, bytes, *stream));
    Require(bytes == borrowed->bytes[id.value - 1],
            "borrowed rows overwritten before preservation");
    detached[id.value - 1] = true;
  }
  void BeforeRelease(cache::ComponentId id, cache::Rows first,
                     cache::Rows end) noexcept override {
    ++releases;
    try {
      BeforeOverwrite(id, first, end);
    } catch (...) {
      release_failed = true;
    }
  }
  void AfterReset() noexcept override { ++resets; }
};
std::vector<std::int32_t> Decode(qfn::Session& session, std::size_t count,
                                 bool sampled = false) {
  const auto history = session.Tokens();
  std::vector<sampling::TokenId> initial(history.begin(), history.end());
  sampling::SamplerState sampler({.temperature = sampled ? 0.8F : 0.0F,
                                  .top_k = 20,
                                  .top_p = 0.95F,
                                  .seed = 73},
                                 initial);
  std::vector<std::int32_t> tokens;
  std::string error;
  while (tokens.size() < count) {
    qfn::Session::DecodeResult result;
    Require(session.DecodeStep(count - tokens.size(), sampler, &result, &error,
                               false),
            error);
    tokens.insert(tokens.end(), result.tokens.begin(), result.tokens.end());
  }
  return tokens;
}
void SameLogits(const qfn::Session& a, const qfn::Session& b) {
  Require(a.Logits().size() == b.Logits().size() &&
              std::memcmp(a.Logits().data(), b.Logits().data(),
                          a.Logits().size_bytes()) == 0,
          "restored next-step logits differ");
}
void Check(const std::shared_ptr<qfn::Model>& model, bool mtp,
           std::span<const std::int32_t> prompt) {
  gufo::hip::TransferPool streams(2);
  qfn::ContinuationAdapter adapter(
      model,
      mtp ? gufo::core::SessionMode::kSpeculative
          : gufo::core::SessionMode::kAutoregressive,
      16384, {'t', 'e', 's', 't'});
  Guard source_guard, destination_guard;
  auto source = adapter.CreateSlot(source_guard);
  auto destination = adapter.CreateSlot(destination_guard);
  source_guard.adapter = destination_guard.adapter = &adapter;
  source_guard.streams = destination_guard.streams = &streams;
  source_guard.slot = source.get();
  destination_guard.slot = destination.get();
  auto& a = adapter.GetSession(*source);
  auto& b = adapter.GetSession(*destination);
  std::string error;
  const auto capacity = model->PrefillCapacity();
  const auto through = model->PrefillThroughCapacity();
  Require(adapter.PlanPrefill(*source, 0, through) ==
              std::vector<cache::Rows>{through},
          "prefill plan split a supported tail");
  Require(adapter.PlanPrefill(*source, 0, through + 1) ==
              std::vector<cache::Rows>{capacity, through + 1},
          "prefill plan exceeds the executor pass capacity");
  Require(adapter.PlanPrefill(*source, 17, 17 + through) ==
              std::vector<cache::Rows>{17 + through},
          "prefill plan lost its continuation offset");
  Reject([&] { (void)adapter.PlanPrefill(*source, 2, 1); });
  Reject([&] { (void)adapter.PlanPrefill(*source, 0, 16385); });
  const auto empty = Capture(adapter, *source, streams);
  Load(adapter, *destination, empty, streams);
  Require(adapter.Validate(*destination, empty.positions),
          "empty restore rejected");
  Require(Capture(adapter, *destination, streams).bytes == empty.bytes,
          "empty state differs");
  for (const auto boundary : {2047U, 8203U}) {
    std::cout << (mtp ? "MTP" : "AR") << ",boundary," << boundary << std::endl;
    Require(a.Sync(prompt.first(boundary), &error), error);
    const auto checkpoint = Capture(adapter, *source, streams);
    source_guard.Borrow(checkpoint);
    Load(adapter, *destination, checkpoint, streams);
    Require(adapter.Validate(*destination, checkpoint.positions),
            "complete restore rejected");
    Require(Capture(adapter, *destination, streams).bytes == checkpoint.bytes,
            "component round trip differs, including chronological raw ring");
    SameLogits(a, b);
    Reject([&] { (void)adapter.Validate(*destination, checkpoint.positions); });
    const auto before = source_guard.overwrites;
    const auto continued = Decode(a, 24, boundary == 8203);
    Require(continued == Decode(b, 24, boundary == 8203),
            "restored draft/continuation tokens differ");
    SameLogits(a, b);
    Require(a.Statistics().drafted == b.Statistics().drafted &&
                a.Statistics().accepted == b.Statistics().accepted,
            "restored draft sequence or acceptance differs");
    if (mtp)
      Require(a.Statistics().drafted > 0, "MTP qualification ran no drafts");
    Require(source_guard.overwrites > before,
            "decode did not invoke mutation guards");
    // Capture the carried draft residual and trained acceptance controller,
    // rather than only the initial prefill frontier.
    const auto mid_decode = Capture(adapter, *source, streams);
    Load(adapter, *destination, mid_decode, streams);
    Require(adapter.Validate(*destination, mid_decode.positions),
            "mid-decode restore rejected");
    Require(Capture(adapter, *destination, streams).bytes == mid_decode.bytes,
            "mid-decode private state changed during restore");
    const auto next = Decode(a, 16, boundary == 8203);
    Require(next == Decode(b, 16, boundary == 8203),
            "carried draft state replay differs");
    SameLogits(a, b);
    if (mtp && boundary == 8203)
      Require(a.Statistics().accepted < a.Statistics().drafted,
              "sampled qualification exercised no rejected draft");

    // A shorter checkpoint replaces every recurrent component; subsequent
    // edits start from that exact boundary rather than truncating recurrence.
    Load(adapter, *source, checkpoint, streams);
    Require(adapter.Validate(*source, checkpoint.positions),
            "shorter restore rejected");
    Require(Capture(adapter, *source, streams).bytes == checkpoint.bytes,
            "earlier recurrent boundary was truncated");
    auto edited =
        std::vector<std::int32_t>(prompt.begin(), prompt.begin() + boundary);
    edited[boundary / 2] = edited[boundary / 2] == 1 ? 2 : 1;
    Require(a.Sync(edited, &error), error);
    Require(source_guard.overwrites > before, "edited prefill skipped guards");
    source_guard.borrowed = nullptr;
    Require(adapter.Invalidate(*source) && source->IsValid(), "reset failed");
    Require(adapter.Invalidate(*destination), "destination reset failed");
  }
  std::cout << (mtp ? "MTP" : "AR") << ",earlier_edit" << std::endl;
  Require(a.Sync(prompt.first(1023), &error), error);
  const auto earlier = Capture(adapter, *source, streams);
  Require(a.Sync(prompt.first(8203), &error), error);
  auto edited =
      std::vector<std::int32_t>(prompt.begin(), prompt.begin() + 8203);
  edited[2048] = edited[2048] == 1 ? 2 : 1;
  Load(adapter, *destination, earlier, streams);
  Require(adapter.Validate(*destination, earlier.positions),
          "earlier edit checkpoint rejected");
  Require(Capture(adapter, *destination, streams).bytes == earlier.bytes,
          "earlier edit checkpoint state changed");
  // Compare the edited branch with an uninterrupted run using the same
  // prefill pass boundaries. Different GEMM batch geometries are a separate
  // numerical contract, rather than a snapshot restore oracle.
  a.Reset();
  Require(a.Sync(std::span(edited).first(1023), &error), error);
  Require(a.Sync(edited, &error) && b.Sync(edited, &error), error);
  SameLogits(a, b);
  Require(Decode(a, 8) == Decode(b, 8),
          "edit after an earlier checkpoint changed drafts");
  Require(adapter.Invalidate(*source), "edit source reset failed");
  Require(a.Sync(prompt.first(8203), &error), error);
  const auto checkpoint = Capture(adapter, *source, streams);
  Load(adapter, *destination, checkpoint, streams);
  Require(adapter.Validate(*destination, checkpoint.positions),
          "peer overlap restore failed");
  {
    auto first_stream = streams.TryAcquire();
    auto second_stream = streams.TryAcquire();
    Require(first_stream && second_stream,
            "settled transfers retained stream leases");
    auto first_bytes = checkpoint.bytes[3];
    auto second_bytes = checkpoint.bytes[4];
    auto first_copy =
        adapter.CapturePrivate(*source, {4}, first_bytes, *first_stream);
    auto second_copy =
        adapter.CapturePrivate(*source, {5}, second_bytes, *second_stream);
    Require(b.Evaluate(prompt[0], &error), error);
    Wait(std::move(first_copy));
    Wait(std::move(second_copy));
    Require(first_bytes == checkpoint.bytes[3] &&
                second_bytes == checkpoint.bytes[4],
            "peer graph execution changed a frozen capture");
  }
  std::cout << (mtp ? "MTP" : "AR") << ",pending_and_failure" << std::endl;
  auto stream = streams.TryAcquire();
  auto bytes = checkpoint.bytes[3];
  auto capture = adapter.CapturePrivate(*source, {4}, bytes, *stream);
  Reject([&] { (void)a.Evaluate(prompt[0], &error); });
  Require(!adapter.Invalidate(*source), "reset ignored a pending read");
  Wait(std::move(capture));
  stream.reset();
  Load(adapter, *destination, checkpoint, streams, true);
  Require(!adapter.Validate(*destination, checkpoint.positions),
          "missing private byte accepted");
  Require(!adapter.Validate(*destination, checkpoint.positions),
          "validation failure was not latched");
  Require(adapter.Invalidate(*destination),
          "failed restore could not be invalidated");
  adapter.BeginRestore(*destination, checkpoint.positions);
  stream = streams.TryAcquire();
  IncompatibleStream incompatible;
  Reject([&] {
    (void)adapter.LoadPrivatePiece(
        *destination, {2}, 0,
        std::span<const std::byte>(checkpoint.bytes[1]).first(1), incompatible);
  });
  // Submission/type errors latch even when the caller discards the error.
  Require(!adapter.Validate(*destination, checkpoint.positions),
          "failed load was not latched");
  Require(adapter.Invalidate(*destination), "submission failure reset failed");
  stream.reset();
  FailNextResetSubmission();
  Require(!adapter.Invalidate(*destination) && !destination->IsValid(),
          "failed zero-fill submission exposed an empty slot");
  Reject([&] { (void)b.Sync(prompt.first(17), &error); });
  Reject([&] { (void)adapter.GetSession(*destination); });
  Require(adapter.Invalidate(*destination),
          "failed zero-fill submission could not recover");
  source_guard.Borrow(checkpoint);
  a.Reset();
  Require(!source_guard.release_failed, "reset preservation failed");
  for (std::size_t i = 0; i < adapter.Components().size(); ++i)
    if (adapter.Components()[i].row_bytes && checkpoint.positions[i].valid_rows)
      Require(source_guard.detached[i],
              "reset failed to preserve borrowed component");
  source_guard.borrowed = nullptr;
  const auto reset_state = Capture(adapter, *source, streams);
  Load(adapter, *destination, reset_state, streams);
  Require(adapter.Validate(*destination, reset_state.positions),
          "reset empty restore failed");
  Require(Capture(adapter, *destination, streams).bytes == reset_state.bytes,
          "reset state was captured before zero fills completed");
  Require(a.Sync(prompt.first(8203), &error), error);
  const auto destroyed = Capture(adapter, *source, streams);
  source_guard.Borrow(destroyed);
  source.reset();
  Require(!source_guard.release_failed, "destruction preservation failed");
  source_guard.borrowed = nullptr;
  std::cout << (mtp ? "MTP" : "AR") << ",private_bytes,";
  std::size_t private_bytes = 0;
  for (const auto& component : adapter.Components())
    private_bytes += component.state_bytes;
  std::cout << private_bytes << ",PASS\n";
}
void CheckImages(const std::shared_ptr<qfn::Model>& model, bool mtp) {
  namespace vision = gufo::models::qwen::vision;
  std::cout << (mtp ? "MTP" : "AR") << ",images" << std::endl;
  Require(bool(model->VisionEncoder()),
          "image adapter qualification requires the adjacent projector");
  auto image = [](bool blue) {
    auto prompt = std::make_shared<vision::Prompt>();
    const vision::ImageGrid grid{0, 2, 2};
    prompt->tokens = {vision::kImageToken, vision::kImageToken,
                      vision::kImageToken, vision::kImageToken, 42};
    prompt->rope.images.push_back(grid);
    gufo::core::Image pixels;
    pixels.width = pixels.height = 64;
    pixels.pixels.resize(64 * 64 * 3);
    for (std::size_t i = blue ? 2 : 0; i < pixels.pixels.size(); i += 3)
      pixels.pixels[i] = 255;
    prompt->images.push_back({std::move(pixels), grid});
    prompt->images.back().prefix_identity.fill(blue ? 2 : 1);
    prompt->cache_identity.assign(32, blue ? 2 : 1);
    return prompt;
  };
  const auto red = image(false), blue = image(true);
  const std::vector<std::int32_t> tokens(red->tokens.begin(),
                                         red->tokens.end());
  gufo::hip::TransferPool streams(2);
  qfn::ContinuationAdapter adapter(
      model,
      mtp ? gufo::core::SessionMode::kSpeculative
          : gufo::core::SessionMode::kAutoregressive,
      64, {'i', 'm', 'a', 'g', 'e'});
  Guard first_guard, second_guard;
  auto first = adapter.CreateSlot(first_guard);
  auto second = adapter.CreateSlot(second_guard);
  auto& a = adapter.GetSession(*first);
  auto& b = adapter.GetSession(*second);
  std::string error;
  a.ConfigureVision(red);
  Require(a.Sync(std::span(tokens).first(2), &error), error);
  const auto checkpoint = Capture(adapter, *first, streams);
  Load(adapter, *second, checkpoint, streams);
  Require(!adapter.Validate(*second, checkpoint.positions),
          "image restored without its attachment");
  Require(adapter.Invalidate(*second), "missing attachment reset failed");
  b.ConfigureVision(blue);
  Load(adapter, *second, checkpoint, streams);
  Require(!adapter.Validate(*second, checkpoint.positions),
          "different pixels accepted at the boundary");
  Require(adapter.Invalidate(*second), "wrong attachment reset failed");
  b.ConfigureVision(red);
  Load(adapter, *second, checkpoint, streams);
  Require(adapter.Validate(*second, checkpoint.positions),
          "matching image restore rejected");
  Require(a.Sync(tokens, &error) && b.Sync(tokens, &error), error);
  SameLogits(a, b);
  a.ConfigureVision(nullptr);
  const std::array<std::int32_t, 2> plain{42, 43};
  Require(a.Sync(plain, &error), error);
  const auto text = Capture(adapter, *first, streams);
  Load(adapter, *second, text, streams);
  Require(adapter.Validate(*second, text.positions),
          "text restore retained stale image input");
  Require(a.Evaluate(44, &error) && b.Evaluate(44, &error), error);
  SameLogits(a, b);
  std::cout << (mtp ? "MTP" : "AR") << ",image_restore,PASS\n";
}
}  // namespace
int main(int argc, char** argv) {
  if (argc != 5 || std::string_view(argv[1]) != "--model" ||
      std::string_view(argv[3]) != "--mtp-model") {
    std::cerr << "Usage: continuation_adapter_test --model FIRST.gguf "
                 "--mtp-model MTP.gguf\n";
    return 77;
  }
  try {
    std::string error;
    auto model = qfn::Model::Load(
        argv[2], {.max_context = 16384, .mtp_model_path = argv[4]}, &error);
    Require(bool(model), error);
    const auto pattern = model->Tokenize(
        "The quick brown fox jumps over the lazy dog. Strix Halo executes this "
        "deterministic benchmark sequence. ");
    Require(!pattern.empty(), "empty prompt pattern");
    std::vector<std::int32_t> prompt(10000);
    for (std::size_t i = 0; i < prompt.size(); ++i)
      prompt[i] = pattern[i % pattern.size()];
    Check(model, false, prompt);
    Check(model, true, prompt);
    CheckImages(model, false);
    CheckImages(model, true);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
