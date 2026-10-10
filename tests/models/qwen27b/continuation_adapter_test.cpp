#include "src/models/qwen/continuation_adapter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cache/slot.hpp"
#include "src/core/gguf_reader.hpp"

namespace cache = gufo::cache;
namespace hip = gufo::hip;
namespace qwen = gufo::models::qwen;
namespace vision = qwen::vision;
using Token = gufo::tokenization::TokenId;
void FailNextResetSubmission();
void FailNextTransferWait();
namespace {
void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}
template<class F>
void Reject(F&& fn) {
  bool rejected = false;
  try {
    fn();
  } catch (const std::exception&) {
    rejected = true;
  }
  Require(rejected, "invalid/busy operation was accepted");
}
void Wait(cache::Completion completion) {
  Require(completion.Wait() == cache::TransferResult::kSucceeded,
          "transfer failed");
}
struct Checkpoint {
  std::vector<cache::ComponentPosition> positions;
  std::vector<std::vector<std::byte>> bytes;
};
Checkpoint Capture(qwen::ContinuationAdapter& adapter, cache::Slot& slot,
                   hip::TransferPool& pool) {
  Checkpoint checkpoint{adapter.Positions(slot), {}};
  checkpoint.bytes.resize(adapter.Components().size());
  for (std::size_t i = 0; i < checkpoint.bytes.size(); ++i) {
    const auto& descriptor = adapter.Components()[i];
    auto& bytes = checkpoint.bytes[i];
    bytes.resize(descriptor.row_bytes
                     ? checkpoint.positions[i].valid_rows * descriptor.row_bytes
                     : descriptor.state_bytes);
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
void Load(qwen::ContinuationAdapter& adapter, cache::Slot& slot,
          const Checkpoint& checkpoint, hip::TransferPool& pool,
          bool missing = false) {
  adapter.BeginRestore(slot, checkpoint.positions);
  for (std::size_t i = 0; i < checkpoint.bytes.size(); ++i) {
    const auto& descriptor = adapter.Components()[i];
    const auto bytes = std::span(checkpoint.bytes[i]);
    if (descriptor.row_bytes) {
      const auto rows = checkpoint.positions[i].valid_rows;
      const auto cut = std::min<cache::Rows>(17, rows);
      auto first = pool.TryAcquire();
      auto second = pool.TryAcquire();
      Require(first && second, "independent row streams unavailable");
      auto tail =
          adapter.CopyRowsIn(slot, descriptor.id, cut, rows,
                             bytes.subspan(cut * descriptor.row_bytes), *first);
      auto head =
          adapter.CopyRowsIn(slot, descriptor.id, 0, cut,
                             bytes.first(cut * descriptor.row_bytes), *second);
      Wait(std::move(head));
      Wait(std::move(tail));
    } else {
      const auto end = bytes.size() - (missing && i == 1 ? 1 : 0);
      const auto cut = std::min<std::size_t>(37, end);
      auto first = pool.TryAcquire();
      auto second = pool.TryAcquire();
      Require(first && second, "independent private streams unavailable");
      auto tail = adapter.LoadPrivatePiece(
          slot, descriptor.id, cut, bytes.subspan(cut, end - cut), *first);
      auto head = adapter.LoadPrivatePiece(slot, descriptor.id, 0,
                                           bytes.first(cut), *second);
      Wait(std::move(head));
      Wait(std::move(tail));
    }
  }
}
struct Guard final : cache::MutationGuard {
  qwen::ContinuationAdapter* adapter{};
  cache::Slot* slot{};
  hip::TransferPool* pool{};
  const Checkpoint* borrowed{};
  std::vector<bool> detached;
  void Borrow(const Checkpoint& checkpoint) {
    borrowed = &checkpoint;
    detached.assign(checkpoint.bytes.size(), false);
  }
  void BeforeOverwrite(cache::ComponentId id, cache::Rows first,
                       cache::Rows end) override {
    if (refuse)
      throw std::runtime_error("guard refusal fixture");
    if (!borrowed || detached[id.value - 1] ||
        first >= borrowed->positions[id.value - 1].valid_rows || !end)
      return;
    const auto& descriptor = adapter->Components()[id.value - 1];
    if (!descriptor.row_bytes)
      return;
    auto stream = pool->TryAcquire();
    Require(bool(stream), "preservation stream unavailable");
    auto bytes = borrowed->bytes[id.value - 1];
    Wait(adapter->CopyRowsOut(*slot, id, 0,
                              borrowed->positions[id.value - 1].valid_rows,
                              bytes, *stream));
    Require(bytes == borrowed->bytes[id.value - 1],
            "borrowed bytes changed before preservation");
    detached[id.value - 1] = true;
  }
  void BeforeRelease(cache::ComponentId id, cache::Rows first,
                     cache::Rows end) noexcept override {
    try {
      BeforeOverwrite(id, first, end);
    } catch (...) {
      release_failed = true;
    }
  }
  void AfterReset() noexcept override { borrowed = nullptr; }
  void Preserved() {
    Require(!release_failed, "borrower release failed");
    for (std::size_t i = 0; i < detached.size(); ++i)
      if (adapter->Components()[i].row_bytes &&
          borrowed->positions[i].valid_rows)
        Require(detached[i], "mutation missed a borrowed component");
    borrowed = nullptr;
  }
  bool release_failed{};
  bool refuse{};
};
std::vector<float> Logits(hip::QwenGpuExecutor& executor) {
  auto values = executor.CopyLastLogits();
  Require(std::ranges::all_of(values, [](float x) { return std::isfinite(x); }),
          "nonfinite logits");
  return {values.begin(), values.end()};
}
void SameLogits(hip::QwenGpuExecutor& a, hip::QwenGpuExecutor& b) {
  const auto first = Logits(a), second = Logits(b);
  Require(first.size() == second.size() &&
              std::memcmp(first.data(), second.data(),
                          first.size() * sizeof(float)) == 0,
          "restored logits differ bytewise");
}
void CheckLeasedGuard(const std::shared_ptr<const hip::QwenGpuModel>& model) {
  qwen::ContinuationAdapter adapter(model, 64, {'l', 'e', 'a', 's', 'e'});
  hip::TransferPool pool(4);
  constexpr std::size_t budget = 8ULL << 20;
  cache::ResourceLedger ledger({budget, budget, 0, budget});
  const std::array<Token, 17> prompt{42, 43, 44, 45, 46, 47, 48, 49, 50,
                                     51, 52, 53, 54, 55, 56, 57, 58};
  for (int path = 0; path < 4; ++path) {
    auto facade_stream = pool.TryAcquire();
    cache::LeasedSlot facade(
        ledger, adapter, *facade_stream, {static_cast<std::uint64_t>(path + 1)},
        cache::IdleSpillConfig{4096,
                               [&]() -> std::unique_ptr<cache::Stream> {
                                 return pool.TryAcquire();
                               },
                               [] { return false; }});
    auto lease = facade.Acquire();
    auto& executor = adapter.GetExecutor(lease.Execution());
    (void)executor.ForwardPromptBatch(prompt);
    const auto positions = adapter.Positions(lease.Execution());
    auto legacy = executor.SaveSnapshot(prompt.size());
    std::vector<cache::ResourceCharge> charges;
    std::vector<std::shared_ptr<cache::BorrowedRows>> borrowed;
    std::vector<std::vector<std::byte>> expected;
    for (const auto& descriptor : adapter.Components()) {
      if (!descriptor.row_bytes)
        continue;
      const auto rows = positions[descriptor.id.value - 1].valid_rows;
      auto& oracle = expected.emplace_back(rows * descriptor.row_bytes);
      auto stream = pool.TryAcquire();
      Wait(adapter.CopyRowsOut(lease.Execution(), descriptor.id, 0, rows,
                               oracle, *stream));
      auto reservation =
          ledger.Reserve(cache::ResourceCategory::kBackingFree, oracle.size());
      auto backing = std::make_shared<std::vector<std::byte>>(oracle.size());
      auto charge = reservation.Convert();
      auto assigned =
          charge.ReserveBacking(cache::ResourceCategory::kBackingAssigned);
      charges.push_back(charge);
      borrowed.push_back(lease.Borrow(descriptor.id, 0, rows,
                                      std::move(assigned), backing, *backing));
    }
    if (path == 0)
      executor.Reset();
    if (path == 1)
      executor.RestoreSnapshot(*legacy);
    if (path == 2)
      adapter.BeginRestore(lease.Execution(), positions);
    if (path == 3) {
      executor.Reset();
      (void)executor.ForwardPromptBatch(std::span(prompt).first(8));
    }
    for (std::size_t i = 0; i < borrowed.size(); ++i) {
      Require(borrowed[i]->IsValid() && !borrowed[i]->Location(),
              "real guard left rows borrowed");
      auto owner = std::static_pointer_cast<const std::vector<std::byte>>(
          borrowed[i]->Owner());
      Require(owner && *owner == expected[i],
              "real guard preserved changed bytes");
    }
  }
  std::cout << "real_leased_guard,PASS\n";
}
void Check(const std::shared_ptr<const hip::QwenGpuModel>& model, bool fp16) {
  std::cout << (fp16 ? "FP16" : "FP32") << ",boundaries" << std::endl;
  hip::QwenExecutionPolicy policy;
  policy.kv_cache_storage =
      fp16 ? hip::QwenKvCacheStorage::kFp16 : hip::QwenKvCacheStorage::kFp32;
  qwen::ContinuationAdapter adapter(model, 12288, {'q', 'w', 'e', 'n'}, policy);
  hip::TransferPool pool(2);
  Guard ga, gb;
  auto source = adapter.CreateSlot(ga), destination = adapter.CreateSlot(gb);
  ga.adapter = gb.adapter = &adapter;
  ga.slot = source.get();
  gb.slot = destination.get();
  ga.pool = gb.pool = &pool;
  auto& a = adapter.GetExecutor(*source);
  auto& b = adapter.GetExecutor(*destination);
  const auto text = model->GetTokenizer().Encode(
      "Memory pages map virtual addresses to physical memory. A page fault "
      "asks the operating system to provide a missing page.\n");
  std::vector<Token> tokens(9000);
  for (std::size_t i = 0; i < tokens.size(); ++i)
    tokens[i] = text[i % text.size()];
  gufo::models::GenerationOptions generation;
  generation.max_new_tokens = 4;
  generation.sampling.temperature = 0;
  auto history = std::vector<Token>(tokens.begin(), tokens.begin() + 64);
  history.reserve(history.size() + generation.max_new_tokens);
  const auto generated = a.Generate(history, generation);
  history.insert(history.end(), generated.begin(), generated.end());
  const auto generated_frontier = Capture(adapter, *source, pool);
  Load(adapter, *destination, generated_frontier, pool);
  Require(adapter.Validate(*destination, generated_frontier.positions),
          "generated frontier restore rejected");
  Require(a.GenerateFromPrefix(history, history.size(), generation) ==
              b.GenerateFromPrefix(history, history.size(), generation),
          "cached generation frontier changed after restore");
  for (auto boundary : {257U, 8203U}) {
    a.Reset();
    b.Reset();
    (void)a.ForwardPromptBatch(std::span(tokens).first(boundary));
    const auto checkpoint = Capture(adapter, *source, pool);
    Load(adapter, *destination, checkpoint, pool);
    Reject([&] { (void)b.ForwardToken(tokens[boundary], boundary); });
    Require(adapter.Validate(*destination, checkpoint.positions),
            "complete restore rejected");
    Require(Capture(adapter, *destination, pool).bytes == checkpoint.bytes,
            "restored component bytes differ");
    SameLogits(a, b);
    for (std::uint32_t i = 0; i < 12; ++i) {
      const auto first = a.ForwardToken(tokens[boundary + i], boundary + i);
      const auto second = b.ForwardToken(tokens[boundary + i], boundary + i);
      Require(first == second, "next token differs");
      SameLogits(a, b);
    }
  }
  a.Reset();
  b.Reset();
  (void)a.ForwardPromptBatch(std::span(tokens).first(257));
  const auto earlier = Capture(adapter, *source, pool);
  (void)a.ForwardPromptBatch(std::span(tokens).subspan(257, 1025), 257);
  auto edited = tokens;
  edited[258] = edited[258] == 1 ? 2 : 1;
  Load(adapter, *destination, earlier, pool);
  Require(adapter.Validate(*destination, earlier.positions),
          "earlier frontier rejected");
  a.Reset();
  (void)a.ForwardPromptBatch(std::span(edited).first(257));
  (void)a.ForwardPromptBatch(std::span(edited).subspan(257, 1025), 257);
  (void)b.ForwardPromptBatch(std::span(edited).subspan(257, 1025), 257);
  SameLogits(a, b);
  const auto stable = Capture(adapter, *source, pool);
  for (int mutation = 0; mutation < 5; ++mutation) {
    Load(adapter, *source, stable, pool);
    Require(adapter.Validate(*source, stable.positions),
            "guard source restore failed");
    const auto legacy = a.SaveSnapshot(1282);
    ga.Borrow(stable);
    if (mutation == 0)
      a.Reset();
    if (mutation == 1)
      a.RestoreSnapshot(*legacy);
    if (mutation == 2) {
      std::vector<std::uint8_t> payload(legacy->CompactPayloadBytes());
      (void)legacy->SerializeCompact(payload);
      a.RestoreCompactSnapshot(payload, 1282);
    }
    if (mutation == 3) {
      a.SaveState(1282);
      Reject([&] { (void)Capture(adapter, *source, pool); });
      (void)a.ForwardToken(tokens[1282], 1282);
      a.RestoreState();
      a.FinishVerification();
    }
    if (mutation == 4)
      (void)a.ForwardPromptBatch(std::span(edited).subspan(5, 7), 5);
    ga.Preserved();
  }
  Load(adapter, *source, stable, pool);
  Require(adapter.Validate(*source, stable.positions),
          "pending source restore failed");
  const auto& descriptor = adapter.Components()[2];
  auto stream = pool.TryAcquire();
  auto bytes = stable.bytes[2];
  auto read = adapter.CapturePrivate(*source, descriptor.id, bytes, *stream);
  Reject([&] { (void)a.ForwardToken(tokens[1282], 1282); });
  Require(!adapter.Invalidate(*source), "invalidation ignored a pending read");
  Wait(std::move(read));
  stream.reset();
  Load(adapter, *destination, stable, pool, true);
  Require(!adapter.Validate(*destination, stable.positions),
          "incomplete private load accepted");
  Require(!adapter.Validate(*destination, stable.positions),
          "restore failure was not latched");
  Require(adapter.Invalidate(*destination), "failed restore could not recover");
  Reject([&] { (void)adapter.Validate(*destination, stable.positions); });
  Require(destination->IsValid(), "out-of-restore validation changed validity");
  for (bool move_assignment : {false, true}) {
    Load(adapter, *destination, stable, pool);
    hip::TransferPool failure_pool(2);
    auto stream = failure_pool.TryAcquire();
    auto load =
        adapter.LoadPrivatePiece(*destination, adapter.Components()[2].id, 0,
                                 std::span(stable.bytes[2]).first(1), *stream);
    FailNextTransferWait();
    if (move_assignment) {
      auto replacement_stream = failure_pool.TryAcquire();
      auto replacement_bytes = stable.bytes[3];
      auto replacement =
          adapter.CapturePrivate(*source, adapter.Components()[3].id,
                                 replacement_bytes, *replacement_stream);
      load = std::move(replacement);
      Wait(std::move(load));
    }
    // Destruction also settles the load, retaining its failure in the slot.
    else {
      auto discarded = std::move(load);
    }
    Require(!adapter.Validate(*destination, stable.positions),
            "discarded load error was not latched");
    Require(adapter.Invalidate(*destination),
            "discarded load error did not recover");
  }
  FailNextResetSubmission();
  Require(!adapter.Invalidate(*destination) && !destination->IsValid(),
          "reset submission failure exposed valid state");
  Reject([&] { (void)b.ForwardToken(tokens[0], 0); });
  Require(adapter.Invalidate(*destination), "reset failure did not recover");
  // Batched native writes preserve both sessions before touching the
  // coordinator stream.
  Load(adapter, *source, earlier, pool);
  Require(adapter.Validate(*source, earlier.positions),
          "batch source restore failed");
  Load(adapter, *destination, earlier, pool);
  Require(adapter.Validate(*destination, earlier.positions),
          "batch destination restore failed");
  ga.Borrow(earlier);
  gb.Borrow(earlier);
  const std::array<hip::QwenGpuBatchItem, 2> batch{
      {{&a, tokens[0], 0}, {&b, tokens[0], 0}}};
  gb.refuse = true;
  Reject([&] { (void)hip::QwenGpuExecutor::ForwardTokenBatch(batch); });
  gb.refuse = false;
  Require(source->IsValid() && destination->IsValid(),
          "guard refusal invalidated an untouched batch peer");
  Require(Capture(adapter, *source, pool).bytes == earlier.bytes &&
              Capture(adapter, *destination, pool).bytes == earlier.bytes,
          "guard refusal changed batch peer bytes");
  (void)hip::QwenGpuExecutor::ForwardTokenBatch(batch);
  ga.Preserved();
  gb.Preserved();
  Guard lifetime_guard;
  auto temporary = adapter.CreateSlot(lifetime_guard);
  Load(adapter, *temporary, earlier, pool);
  Require(adapter.Validate(*temporary, earlier.positions),
          "lifetime restore failed");
  auto lifetime_stream = pool.TryAcquire();
  auto lifetime_bytes = earlier.bytes[2];
  auto lifetime_copy = adapter.CapturePrivate(
      *temporary, adapter.Components()[2].id, lifetime_bytes, *lifetime_stream);
  temporary.reset();
  Wait(std::move(lifetime_copy));
  Require(lifetime_bytes == earlier.bytes[2],
          "slot destruction changed pending capture");
  std::cout << (fp16 ? "FP16" : "FP32") << ",restore_edit_guard,PASS\n";
}

void CheckBatchAdmission(
    const std::shared_ptr<const hip::QwenGpuModel>& model) {
  qwen::ContinuationAdapter adapter(model, 64, {'b', 'a', 't', 'c', 'h'});
  hip::TransferPool pool(2);
  Guard ga, gb;
  auto source = adapter.CreateSlot(ga), destination = adapter.CreateSlot(gb);
  auto& a = adapter.GetExecutor(*source);
  auto& b = adapter.GetExecutor(*destination);
  const std::array<Token, 3> prefix{42, 43, 44};
  (void)a.ForwardPromptBatch(prefix);
  (void)b.ForwardPromptBatch(prefix);
  const auto before_a = Capture(adapter, *source, pool);
  const auto before_b = Capture(adapter, *destination, pool);
  const std::array<hip::QwenGpuBatchItem, 2> batch{{{&a, 45, 3}, {&b, 45, 3}}};
  const std::array<Token, 2> tokens{45, 46};
  const std::array<hip::QwenGpuVerificationItem, 2> verification{
      {{&a, tokens, 3}, {&b, tokens, 3}}};
  gb.refuse = true;
  Reject([&] { (void)hip::QwenGpuExecutor::ForwardTokenBatch(batch); });
  Reject([&] {
    (void)hip::QwenGpuExecutor::ForwardVerificationBatch(verification);
  });
  gb.refuse = false;
  Require(source->IsValid() && destination->IsValid(),
          "batch admission invalidated untouched sessions");
  Require(Capture(adapter, *source, pool).bytes == before_a.bytes &&
              Capture(adapter, *destination, pool).bytes == before_b.bytes,
          "batch admission changed session bytes");
  (void)hip::QwenGpuExecutor::ForwardTokenBatch(batch);
  SameLogits(a, b);
  ga.adapter = gb.adapter = &adapter;
  ga.slot = source.get();
  gb.slot = destination.get();
  ga.pool = gb.pool = &pool;
  const auto borrowed_a = Capture(adapter, *source, pool);
  const auto borrowed_b = Capture(adapter, *destination, pool);
  ga.Borrow(borrowed_a);
  gb.Borrow(borrowed_b);
  const std::array<hip::QwenGpuVerificationItem, 2> overwrite{
      {{&a, tokens, 0}, {&b, tokens, 0}}};
  const auto predictions =
      hip::QwenGpuExecutor::ForwardVerificationBatch(overwrite);
  Require(predictions[0] == predictions[1], "verification batch peers differ");
  ga.Preserved();
  gb.Preserved();
  std::cout << "batch_admission,PASS\n";
}

void CheckMutationPaths(const std::shared_ptr<const hip::QwenGpuModel>& model) {
  const std::array<Token, 4> prefix{42, 43, 44, 45};
  const std::array<Token, 2> verified{46, 47};
  for (bool fp16 : {true, false}) {
    auto policy = hip::QwenExecutionPolicy::Production();
    if (!fp16)
      policy.kv_cache_storage = hip::QwenKvCacheStorage::kFp32;
    qwen::ContinuationAdapter adapter(model, 64, {'m', 'u', 't'}, policy);
    hip::TransferPool pool(2);
    Guard guard, other_guard;
    auto slot = adapter.CreateSlot(guard),
         other = adapter.CreateSlot(other_guard);
    auto& a = adapter.GetExecutor(*slot);
    auto& b = adapter.GetExecutor(*other);
    guard.adapter = &adapter;
    guard.slot = slot.get();
    guard.pool = &pool;
    (void)a.ForwardPromptBatch(prefix);
    const auto checkpoint = Capture(adapter, *slot, pool);
    // These calls intentionally overwrite existing KV rows; recurrence quality
    // after an edit is covered separately by restoring the exact boundary.
    for (bool commit : {false, true}) {
      Load(adapter, *slot, checkpoint, pool);
      Require(adapter.Validate(*slot, checkpoint.positions),
              "mutation restore rejected");
      guard.Borrow(checkpoint);
      if (commit)
        a.CommitVerificationChunk(verified, 0);
      else
        (void)a.ForwardToken(46, 0);
      guard.Preserved();
    }
    Load(adapter, *slot, checkpoint, pool);
    Require(adapter.Validate(*slot, checkpoint.positions),
            "replay source restore rejected");
    Load(adapter, *other, checkpoint, pool);
    Require(adapter.Validate(*other, checkpoint.positions),
            "replay control restore rejected");
    a.SaveState(4);
    (void)a.ForwardVerificationChunk(verified, 4);
    a.RestoreState();
    a.CommitVerificationChunk(std::span(verified).first(1), 4);
    a.FinishVerification();
    (void)b.ForwardToken(verified[0], 4, false);
    Require(Capture(adapter, *slot, pool).bytes ==
                Capture(adapter, *other, pool).bytes,
            "committed recurrent replay differs from scalar continuation");
  }
  std::cout << "scalar_commit_replay_guard,PASS\n";
}

std::shared_ptr<vision::Prompt> Image(bool blue) {
  auto prompt = std::make_shared<vision::Prompt>();
  vision::ImageGrid grid{0, 2, 2};
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
}
void CheckImages(const std::shared_ptr<const hip::QwenGpuModel>& model) {
  Require(bool(model->VisionEncoder()),
          "adjacent projector required for image qualification");
  qwen::ContinuationAdapter adapter(model, 64, {'i', 'm', 'g'});
  hip::TransferPool pool(2);
  Guard ga, gb;
  auto source = adapter.CreateSlot(ga), destination = adapter.CreateSlot(gb);
  auto& a = adapter.GetExecutor(*source);
  auto& b = adapter.GetExecutor(*destination);
  auto red = Image(false), blue = Image(true);
  a.ConfigureVision(red, model->VisionEncoder());
  (void)a.ForwardPromptBatch(std::span(red->tokens).first(2));
  const auto checkpoint = Capture(adapter, *source, pool);
  Load(adapter, *destination, checkpoint, pool);
  Require(!adapter.Validate(*destination, checkpoint.positions),
          "missing image attachment accepted");
  Require(adapter.Invalidate(*destination), "image reset failed");
  b.ConfigureVision(blue, model->VisionEncoder());
  Load(adapter, *destination, checkpoint, pool);
  Require(!adapter.Validate(*destination, checkpoint.positions),
          "changed image accepted");
  Require(adapter.Invalidate(*destination), "changed image reset failed");
  b.ConfigureVision(red, model->VisionEncoder());
  Load(adapter, *destination, checkpoint, pool);
  Require(adapter.Validate(*destination, checkpoint.positions),
          "matching image rejected");
  (void)a.ForwardPromptBatch(std::span(red->tokens).subspan(2), 2);
  (void)b.ForwardPromptBatch(std::span(red->tokens).subspan(2), 2);
  SameLogits(a, b);
  Require(a.VisionLayout().Position(4)[0] != 4,
          "image fixture did not change positions");
  Reject([&] { a.ConfigureVision(blue, model->VisionEncoder()); });
  a.Reset();
  a.ConfigureVision(nullptr, nullptr);
  const std::array<Token, 2> plain{42, 43};
  (void)a.ForwardPromptBatch(plain);
  const auto text = Capture(adapter, *source, pool);
  Load(adapter, *destination, text, pool);
  Require(adapter.Validate(*destination, text.positions),
          "text restore rejected");
  Require(b.VisionLayout().images.empty(),
          "text restore retained image layout");
  (void)a.ForwardToken(44, 2);
  (void)b.ForwardToken(44, 2);
  SameLogits(a, b);
  std::cout << "images,PASS\n";
}
void CheckImageSuffix(const std::shared_ptr<const hip::QwenGpuModel>& model) {
  auto prompt = std::make_shared<vision::Prompt>();
  prompt->tokens.push_back(42);
  for (bool blue : {false, true}) {
    const auto image = Image(blue);
    auto prepared = image->images.front();
    prepared.grid.offset = static_cast<std::uint32_t>(prompt->tokens.size());
    prompt->rope.images.push_back(prepared.grid);
    prompt->images.push_back(std::move(prepared));
    prompt->tokens.insert(prompt->tokens.end(), image->tokens.begin(),
                          image->tokens.end());
  }
  prompt->cache_identity.assign(32, 2);
  for (const std::size_t boundary : {1, 6}) {
    qwen::ContinuationAdapter adapter(model, 64,
                                      {'s', 'u', 'f', 'f', 'i', 'x'});
    hip::TransferPool pool(2);
    Guard ga, gb;
    auto source = adapter.CreateSlot(ga), destination = adapter.CreateSlot(gb);
    auto& a = adapter.GetExecutor(*source);
    auto& b = adapter.GetExecutor(*destination);
    a.ConfigureVision(prompt, model->VisionEncoder());
    b.ConfigureVision(prompt, model->VisionEncoder());
    (void)a.ForwardPromptBatch(std::span(prompt->tokens).first(boundary));
    const auto checkpoint = Capture(adapter, *source, pool);
    Load(adapter, *destination, checkpoint, pool);
    Require(adapter.Validate(*destination, checkpoint.positions),
            "image suffix checkpoint rejected");
    Require(b.VisionLayout() == prompt->rope,
            "restoration discarded future image positions");
    (void)a.ForwardPromptBatch(std::span(prompt->tokens).subspan(boundary),
                               static_cast<std::uint32_t>(boundary));
    (void)b.ForwardPromptBatch(std::span(prompt->tokens).subspan(boundary),
                               static_cast<std::uint32_t>(boundary));
    SameLogits(a, b);
    Require(Capture(adapter, *source, pool).bytes ==
                Capture(adapter, *destination, pool).bytes,
            "restoration changed image suffix component bytes");
  }
  std::cout << "image_suffix,PASS\n";
}
}  // namespace
int main(int argc, const char* const* argv) {
  try {
    const bool guard_only = argc == 3 && std::string(argv[2]) == "--guard-only";
    const bool image_only = argc == 3 && std::string(argv[2]) == "--image-only";
    if (argc != 2 && !guard_only && !image_only)
      throw std::invalid_argument(
          "Usage: continuation_adapter_test MODEL.gguf "
          "[--guard-only|--image-only]");
    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
    Require(bool(reader), error.c_str());
    auto encoder = vision::Encoder::Open(argv[1], {}, 5120);
    auto model = hip::QwenGpuModel::CreateFromGguf(
        std::shared_ptr<const gufo::core::GgufReader>(std::move(reader)),
        &error, encoder);
    Require(bool(model), error.c_str());
    if (!guard_only && !image_only) {
      Check(model, true);
      Check(model, false);
    }
    if (!image_only) {
      CheckLeasedGuard(model);
      CheckBatchAdmission(model);
      CheckMutationPaths(model);
    }
    if (!guard_only) {
      CheckImages(model);
      CheckImageSuffix(model);
    }
    std::cout << "Qwen continuation adapter: PASS\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
