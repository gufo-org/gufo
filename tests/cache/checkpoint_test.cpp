#include "src/cache/checkpoint.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tests/cache/fake_adapter.hpp"

using namespace gufo::cache;
using namespace gufo::cache::testing;

namespace {
class Guard final : public MutationGuard {
public:
  void BeforeOverwrite(ComponentId, Rows, Rows) override {}
  void BeforeRelease(ComponentId, Rows, Rows) noexcept override {}
};
constexpr ResourceLimits kLimits{1 << 20, 1 << 20, 1 << 20, 1 << 20};

struct Fixture {
  ResourceLedger ledger{kLimits};
  FakeAdapter adapter;
  Guard guard;
  FakeStream stream;
  std::unique_ptr<Slot> slot{adapter.CreateSlot(guard)};
  std::vector<Token> tokens;
  Payload Save(const PayloadRequest& request) {
    auto reservation =
        ledger.Reserve(ResourceCategory::kBackingFree, request.bytes);
    auto buffer = std::make_shared<std::vector<std::byte>>(request.bytes);
    auto pool = reservation.Convert();
    auto assigned = pool.ReserveBacking(request.category);
    auto transfer =
        request.category == ResourceCategory::kPrivateState
            ? adapter.CapturePrivate(*slot, request.component, *buffer, stream)
            : adapter.CopyRowsOut(*slot, request.component, request.first,
                                  request.end, *buffer, stream);
    if (transfer.Wait() != TransferResult::kSucceeded)
      throw std::runtime_error("fake capture failed");
    return Payload::Committed(assigned.Convert(), buffer);
  }
  std::shared_ptr<const Checkpoint> Capture(ExecutionHistory& history) {
    const auto positions = adapter.Positions(*slot);
    return history.Capture(
        {tokens, InputIdentity(tokens.size()), positions,
         CheckpointPurpose::kPrompt, 1},
        [this](const PayloadRequest& request) { return Save(request); });
  }
  void Append(unsigned count, Token value = 7) {
    std::vector<Token> suffix(count, value);
    adapter.Append(*slot, suffix, suffix);
    tokens.insert(tokens.end(), suffix.begin(), suffix.end());
  }
  std::vector<std::byte> Read(ComponentId component) {
    const auto components = adapter.Components();
    const auto descriptor =
        *std::find_if(components.begin(), components.end(),
                      [&](const auto& d) { return d.id == component; });
    const auto positions = adapter.Positions(*slot);
    const auto rows =
        std::find_if(positions.begin(), positions.end(), [&](const auto& p) {
          return p.id == component;
        })->valid_rows;
    std::vector<std::byte> bytes(descriptor.kind == ComponentKind::kPrivateState
                                     ? descriptor.state_bytes
                                     : rows * descriptor.row_bytes);
    auto transfer =
        descriptor.kind == ComponentKind::kPrivateState
            ? adapter.CapturePrivate(*slot, component, bytes, stream)
            : adapter.CopyRowsOut(*slot, component, 0, rows, bytes, stream);
    assert(transfer.Wait() == TransferResult::kSucceeded);
    return bytes;
  }
  void Restore(const Checkpoint& checkpoint) {
    std::vector<ComponentPosition> positions;
    for (const auto& c : checkpoint.Components())
      positions.push_back(c.position);
    adapter.BeginRestore(*slot, positions);
    for (const auto& c : checkpoint.Components()) {
      for (const auto& chunk : c.chunks) {
        auto buffer = std::static_pointer_cast<const std::vector<std::byte>>(
            chunk.Storage().Owner());
        auto transfer =
            adapter.CopyRowsIn(*slot, c.descriptor.id, chunk.First(),
                               chunk.End(), *buffer, stream);
        assert(transfer.Wait() == TransferResult::kSucceeded);
      }
      if (c.tail) {
        auto buffer = std::static_pointer_cast<const std::vector<std::byte>>(
            c.tail->Owner());
        const auto end = c.position.valid_rows;
        auto transfer = adapter.CopyRowsIn(
            *slot, c.descriptor.id, end - end % c.descriptor.rows_per_chunk,
            end, *buffer, stream);
        assert(transfer.Wait() == TransferResult::kSucceeded);
      }
      if (c.private_state) {
        auto buffer = std::static_pointer_cast<const std::vector<std::byte>>(
            c.private_state->Owner());
        auto transfer =
            adapter.LoadPrivate(*slot, c.descriptor.id, *buffer, stream);
        assert(transfer.Wait() == TransferResult::kSucceeded);
      }
    }
    assert(adapter.Validate(*slot, positions));
    tokens.assign(checkpoint.Tokens().begin(), checkpoint.Tokens().end());
  }
};
void AssertSameState(Fixture& restored, Fixture& control) {
  assert(restored.adapter.Positions(*restored.slot) ==
         control.adapter.Positions(*control.slot));
  for (const auto& component : control.adapter.Components())
    assert(restored.Read(component.id) == control.Read(component.id));
}
void SharingAndTeardown() {
  Fixture f;
  {
    auto history = ExecutionHistory::Cold(f.ledger, f.adapter.Components(),
                                          f.adapter.CompatibilityIdentity());
    f.Append(5);
    auto parent = f.Capture(history);
    assert(parent->Components()[0].chunks.size() == 1);
    assert(parent->Components()[0].tail);
    const auto parent_chunk = parent->Components()[0].chunks[0].Id();
    auto reader = parent->Components()[0].chunks[0].PinReader();
    auto writer = parent->Components()[0].chunks[0].PinPersistence();
    f.Append(3);
    auto child = f.Capture(history);
    assert(parent->Lineage() == child->Lineage());
    assert(child->Components()[0].chunks[0].Id() == parent_chunk);
    assert(child->Components()[0].chunks.size() == 2);
    assert(!child->Components()[0].tail);
    assert(parent->Components()[0].tail->Bytes() == sizeof(Token));
    assert((reader.References() == ChunkReferences{2, 1, 1}));
    child.reset();
    assert((reader.References() == ChunkReferences{1, 1, 1}));
    parent.reset();
    assert((reader.References() == ChunkReferences{0, 1, 1}));
    assert(f.ledger.Snapshot().persistence_pinned_bytes == 4 * sizeof(Token));
  }
  assert(f.ledger.Snapshot().total_bytes == 0);
}
void EmptyAndShortCheckpoints() {
  Fixture f;
  {
    auto history = ExecutionHistory::Cold(f.ledger, f.adapter.Components(),
                                          f.adapter.CompatibilityIdentity());
    auto empty = f.Capture(history);
    assert(empty->Boundary() == 0);
    for (const auto& c : empty->Components()) {
      assert(c.chunks.empty());
      assert(!c.tail);
    }
    auto empty_restore = ExecutionHistory::Restored(f.ledger, *empty);
    f.Restore(*empty);
    f.Append(3);
    auto short_checkpoint = f.Capture(empty_restore);
    const auto hash = f.adapter.RecurrentHash(*f.slot);
    for (std::size_t i = 0; i < 2; ++i) {
      assert(short_checkpoint->Components()[i].chunks.empty());
      assert(short_checkpoint->Components()[i].tail->Bytes() ==
             3 * sizeof(Token));
    }
    auto restored = ExecutionHistory::Restored(f.ledger, *short_checkpoint);
    f.Restore(*short_checkpoint);
    assert(f.adapter.RecurrentHash(*f.slot) == hash);
    f.Append(1);
    auto full = f.Capture(restored);
    assert(full->Components()[0].chunks.size() == 1);
    assert(!full->Components()[0].tail);
    assert(short_checkpoint->Components()[0].tail->Bytes() ==
           3 * sizeof(Token));
  }
  assert(f.ledger.Snapshot().total_bytes == 0);
}
void ColdPrefillsNeverShare() {
  Fixture f;
  {
    auto first = ExecutionHistory::Cold(f.ledger, f.adapter.Components(),
                                        f.adapter.CompatibilityIdentity());
    f.Append(8);
    auto a = f.Capture(first);
    const auto hash = f.adapter.RecurrentHash(*f.slot);
    assert(f.adapter.Invalidate(*f.slot));
    f.tokens.clear();
    auto second = ExecutionHistory::Cold(f.ledger, f.adapter.Components(),
                                         f.adapter.CompatibilityIdentity());
    f.Append(8);
    auto b = f.Capture(second);
    assert(hash == f.adapter.RecurrentHash(*f.slot));
    assert(a->Lineage() != b->Lineage());
    for (std::size_t i = 0; i < 2; ++i)
      for (std::size_t j = 0; j < a->Components()[i].chunks.size(); ++j)
        assert(a->Components()[i].chunks[j].Id() !=
               b->Components()[i].chunks[j].Id());
    assert(a->Components()[2].private_state->Owner() !=
           b->Components()[2].private_state->Owner());
  }
  assert(f.ledger.Snapshot().total_bytes == 0);
}

template<class Exception, class Function>
void Throws(Function&& function) {
  bool caught = false;
  try {
    function();
  } catch (const Exception&) {
    caught = true;
  }
  assert(caught);
}
void RestoredForksAndPrivateTails() {
  Fixture f;
  {
    auto original = ExecutionHistory::Cold(f.ledger, f.adapter.Components(),
                                           f.adapter.CompatibilityIdentity());
    f.Append(5);
    auto parent = f.Capture(original);
    const auto parent_hash = f.adapter.RecurrentHash(*f.slot);
    f.Append(3, 8);
    auto original_child = f.Capture(original);
    const auto original_hash = f.adapter.RecurrentHash(*f.slot);
    auto fork = ExecutionHistory::Restored(f.ledger, *parent);
    f.Restore(*parent);
    assert(f.adapter.RecurrentHash(*f.slot) == parent_hash);
    // A capture at the same partial frontier still owns a new private tail.
    auto same_boundary = f.Capture(fork);
    assert(same_boundary->Components()[0].tail->Owner() !=
           parent->Components()[0].tail->Owner());
    f.Append(3, 9);
    auto child = f.Capture(fork);
    assert(child->Lineage() == parent->Lineage());
    assert(child->Components()[0].chunks[0].Id() ==
           parent->Components()[0].chunks[0].Id());
    assert(child->Components()[0].chunks[1].Id() !=
           original_child->Components()[0].chunks[1].Id());
    assert(f.adapter.RecurrentHash(*f.slot) != original_hash);
    auto sibling = ExecutionHistory::Restored(f.ledger, *parent);
    f.Restore(*parent);
    f.Append(3, 9);  // Even equal suffix tokens are independent computation.
    auto sibling_child = f.Capture(sibling);
    assert(sibling_child->Components()[0].chunks[1].Id() !=
           child->Components()[0].chunks[1].Id());
    const auto expected = f.adapter.RecurrentHash(*f.slot);
    f.Restore(*child);
    assert(f.adapter.RecurrentHash(*f.slot) == expected);
    Fixture child_control;
    child_control.Append(5);
    child_control.Append(3, 9);
    AssertSameState(f, child_control);
    f.Append(1, 77);
    child_control.Append(1, 77);
    AssertSameState(f, child_control);
    f.Restore(*original_child);
    assert(f.adapter.RecurrentHash(*f.slot) == original_hash);
    Fixture original_control;
    original_control.Append(5);
    original_control.Append(3, 8);
    AssertSameState(f, original_control);
    f.Append(1, 77);
    original_control.Append(1, 77);
    AssertSameState(f, original_control);
  }
  assert(f.ledger.Snapshot().total_bytes == 0);
}
void FailedCapturesAreInvisible() {
  Fixture f;
  {
    auto history = ExecutionHistory::Cold(f.ledger, f.adapter.Components(),
                                          f.adapter.CompatibilityIdentity());
    f.Append(5);
    auto parent = f.Capture(history);
    f.Append(3);
    const auto before = f.ledger.Snapshot();
    f.adapter.FailNextTransfer();
    Throws<std::runtime_error>([&] { (void)f.Capture(history); });
    const auto failed = f.ledger.Snapshot();
    assert(failed.bytes == before.bytes);
    assert(failed.reserved_bytes == before.reserved_bytes);
    assert(failed.total_bytes == before.total_bytes);
    assert(parent->Components()[0].chunks[0].References().checkpoints == 1);
    assert(f.slot->IsValid());
    f.ledger.FailAfter(LedgerStep::kConvert, 1);
    Throws<std::bad_alloc>([&] { (void)f.Capture(history); });
    assert(f.ledger.Snapshot().bytes == before.bytes);
    auto child = f.Capture(history);
    assert(child->Components()[0].chunks[0].Id() ==
           parent->Components()[0].chunks[0].Id());
    assert(child->Components()[0].chunks[1].References().checkpoints == 1);
    // A callback failure after copying private state still cannot publish it.
    const auto retained = f.ledger.Snapshot().bytes;
    unsigned captures = 0;
    Throws<std::runtime_error>([&] {
      (void)history.Capture(
          {f.tokens, InputIdentity(f.tokens.size()),
           f.adapter.Positions(*f.slot), CheckpointPurpose::kGrid, 2},
          [&](const auto& request) {
            ++captures;
            auto result = f.Save(request);
            throw std::runtime_error("private capture failed");
            return result;
          });
    });
    assert(captures == 1);
    assert(f.ledger.Snapshot().bytes == retained);
    Throws<std::logic_error>([&] {
      (void)history.Capture(
          {f.tokens, InputIdentity(f.tokens.size()),
           f.adapter.Positions(*f.slot), CheckpointPurpose::kGrid, 2},
          [&](const auto& request) {
            (void)f.Capture(history);
            return f.Save(request);
          });
    });
    assert(f.ledger.Snapshot().bytes == retained);
  }
  assert(f.ledger.Snapshot().total_bytes == 0);
}
Payload AccountingPayload(ResourceLedger& ledger, const PayloadRequest& r) {
  // Sized fake backing: exercises the actual ownership/ledger with model-sized
  // capacities without allocating multi-GiB device buffers on the CPU host.
  auto pool = ledger.Reserve(ResourceCategory::kBackingFree, r.bytes).Convert();
  auto assigned = pool.ReserveBacking(r.category);
  auto owner = std::make_shared<std::size_t>(r.bytes);
  return Payload::Committed(assigned.Convert(), owner);
}
void PrivateOnlyCheckpoints() {
  ResourceLedger ledger(kLimits);
  {
    const std::array<ComponentDescriptor, 1> layout{
        {{kRecurrent, 1, ComponentKind::kPrivateState, 0, 0, 32}}};
    auto history = ExecutionHistory::Cold(ledger, layout, {1});
    std::array<Token, 3> tokens{7, 7, 7};
    const CheckpointRequest request{tokens,
                                    InputIdentity(3),
                                    {{kRecurrent, 3}},
                                    CheckpointPurpose::kPrompt,
                                    1};
    auto save = [&](const auto& r) { return AccountingPayload(ledger, r); };
    auto checkpoint = history.Capture(request, save);
    assert(checkpoint->Components()[0].chunks.empty());
    assert(checkpoint->Components()[0].private_state);
    auto restored = ExecutionHistory::Restored(ledger, *checkpoint);
    auto child = restored.Capture(request, save);
    assert(child->Lineage() == checkpoint->Lineage());
    assert(child->Components()[0].private_state->Owner() !=
           checkpoint->Components()[0].private_state->Owner());
  }
  assert(ledger.Snapshot().total_bytes == 0);
}
void GeometryLocationsAndPins() {
  ResourceLedger ledger(kLimits);
  const std::array<ComponentDescriptor, 3> layout{
      {{{1}, 1, ComponentKind::kAppendRows, 8, 3, 0},
       {{2}, 1, ComponentKind::kAppendRows, 16, 5, 0},
       {{3}, 1, ComponentKind::kPrivateState, 0, 0, 32}}};
  {
    auto history = ExecutionHistory::Cold(ledger, layout, {1, 2});
    std::array<Token, 8> tokens{};
    auto checkpoint = history.Capture(
        {tokens,
         InputIdentity(8, {3}),
         {{{2}, 6}, {{3}, 8}, {{1}, 8}},
         CheckpointPurpose::kGenerated,
         4},
        [&](const auto& r) { return AccountingPayload(ledger, r); });
    assert(checkpoint->Boundary() == 8);
    assert(checkpoint->Rank() == 4);
    assert(checkpoint->Purpose() == CheckpointPurpose::kGenerated);
    assert(checkpoint->Compatibility() == Identity({1, 2}));
    assert(checkpoint->Input() == Identity({3}));
    assert(checkpoint->Components()[0].chunks.size() == 2);
    assert(checkpoint->Components()[0].tail->Bytes() == 16);
    assert(checkpoint->Components()[1].chunks.size() == 1);
    assert(checkpoint->Components()[1].tail->Bytes() == 16);
    const auto& chunk = checkpoint->Components()[1].chunks[0];
    assert(chunk.First() == 0 && chunk.End() == 5);
    assert(!chunk.Storage().BorrowedFrom());
    auto read = chunk.PinReader();
    ledger.FailAfter(LedgerStep::kPin, 0);
    Throws<std::bad_alloc>([&] { (void)chunk.PinPersistence(); });
    assert((chunk.References() == ChunkReferences{1, 1, 0}));
    auto persistence = chunk.PinPersistence();
    auto copy = persistence;
    assert((chunk.References() == ChunkReferences{1, 1, 2}));
    assert(ledger.Snapshot().persistence_pinned_bytes == 80);
    checkpoint.reset();
    history.Prune();
    assert(ledger.Snapshot().bytes[static_cast<std::size_t>(
               ResourceCategory::kBackingMaterialized)] == 80);
    assert(ledger.Snapshot().bytes[static_cast<std::size_t>(
               ResourceCategory::kPrivateTail)] == 0);
    assert((read.References() == ChunkReferences{0, 1, 2}));
  }
  assert(ledger.Snapshot().total_bytes == 0);
}
void RejectInvalidDescriptionsAndProvenance() {
  Fixture f;
  {
    auto history = ExecutionHistory::Cold(f.ledger, f.adapter.Components(),
                                          f.adapter.CompatibilityIdentity());
    f.Append(5);
    auto parent = f.Capture(history);
    auto positions = f.adapter.Positions(*f.slot);
    unsigned calls = 0;
    auto save = [&](const auto& request) {
      ++calls;
      return f.Save(request);
    };
    positions.back().id = kTarget;
    Throws<std::invalid_argument>([&] {
      (void)history.Capture({f.tokens, InputIdentity(5), positions,
                             CheckpointPurpose::kPrompt, 1},
                            save);
    });
    positions = f.adapter.Positions(*f.slot);
    --positions.front().valid_rows;
    Throws<std::invalid_argument>([&] {
      (void)history.Capture({f.tokens, InputIdentity(5), positions,
                             CheckpointPurpose::kPrompt, 1},
                            save);
    });
    positions = f.adapter.Positions(*f.slot);
    f.tokens.front() = 99;
    Throws<std::invalid_argument>([&] {
      (void)history.Capture({f.tokens, InputIdentity(5), positions,
                             CheckpointPurpose::kPrompt, 1},
                            save);
    });
    f.tokens.front() = 7;
    Throws<std::invalid_argument>([&] {
      (void)history.Capture({f.tokens, InputIdentity(5, {99}), positions,
                             CheckpointPurpose::kPrompt, 1},
                            save);
    });
    assert(calls == 0);
    positions.front().valid_rows = std::numeric_limits<Rows>::max();
    Throws<std::overflow_error>([&] {
      (void)history.Capture({f.tokens, InputIdentity(5), positions,
                             CheckpointPurpose::kPrompt, 1},
                            save);
    });
    assert(calls == 0);
    f.Append(3);
    auto after_image = history.Capture(
        {f.tokens, InputIdentity(8, {99}, {{5, {}}}),
         f.adapter.Positions(*f.slot), CheckpointPurpose::kPrompt, 1},
        save);
    assert(after_image->Input() == Identity({99}));
    assert(parent->Input().empty());
    assert(after_image->Components()[0].chunks[0].Id() ==
           parent->Components()[0].chunks[0].Id());
  }
  assert(f.ledger.Snapshot().total_bytes == 0);
  auto layout = std::vector<ComponentDescriptor>(f.adapter.Components().begin(),
                                                 f.adapter.Components().end());
  layout.front().rows_per_chunk = 0;
  Throws<std::invalid_argument>(
      [&] { (void)ExecutionHistory::Cold(f.ledger, layout, {}); });
  layout.front() = layout.back();
  Throws<std::invalid_argument>(
      [&] { (void)ExecutionHistory::Cold(f.ledger, layout, {}); });
  assert(f.ledger.Snapshot().total_bytes == 0);
}
void FaultsAtEveryAdmissionAndConversion() {
  for (const auto step : {LedgerStep::kReserve, LedgerStep::kConvert}) {
    bool reached_success = false;
    for (std::size_t failure = 0; failure != 40; ++failure) {
      Fixture f;
      {
        auto history =
            ExecutionHistory::Cold(f.ledger, f.adapter.Components(),
                                   f.adapter.CompatibilityIdentity());
        f.Append(5);
        auto parent = f.Capture(history);
        f.Append(3);
        const auto before = f.ledger.Snapshot();
        f.ledger.FailAfter(step, failure);
        try {
          auto child = f.Capture(history);
          reached_success = true;
        } catch (const std::bad_alloc&) {
          const auto failed = f.ledger.Snapshot();
          assert(failed.bytes == before.bytes);
          assert(failed.reserved_bytes == before.reserved_bytes);
          assert(failed.persistence_pinned_bytes ==
                 before.persistence_pinned_bytes);
          assert(parent->Components()[0].chunks[0].References().checkpoints ==
                 1);
          auto retry = f.Capture(history);
          assert(retry->Components()[0].chunks[0].Id() ==
                 parent->Components()[0].chunks[0].Id());
          assert(retry->Components()[0].chunks[1].References().checkpoints ==
                 1);
        }
        f.ledger.ClearFaults();
      }
      assert(f.ledger.Snapshot().total_bytes == 0);
      if (reached_success)
        break;
    }
    assert(reached_success);
  }
}
void PayloadValidationAndReleaseOrder() {
  ResourceLedger ledger(kLimits);
  {
    auto first_charge =
        ledger.Reserve(ResourceCategory::kPrivateState, 64).Convert();
    auto second_charge =
        ledger.Reserve(ResourceCategory::kPrivateTail, 128).Convert();
    auto first_owner =
        std::shared_ptr<const void>(new int, [&](const void* pointer) {
          assert(ledger.Snapshot().bytes[static_cast<std::size_t>(
                     ResourceCategory::kPrivateState)] == 64);
          delete static_cast<const int*>(pointer);
        });
    auto second_owner =
        std::shared_ptr<const void>(new int, [&](const void* pointer) {
          assert(ledger.Snapshot().bytes[static_cast<std::size_t>(
                     ResourceCategory::kPrivateTail)] == 128);
          delete static_cast<const int*>(pointer);
        });
    auto first =
        Payload::Committed(std::move(first_charge), std::move(first_owner));
    auto second =
        Payload::Committed(std::move(second_charge), std::move(second_owner));
    first = std::move(second);
  }
  assert(ledger.Snapshot().total_bytes == 0);
  {
    auto pool = ledger.Reserve(ResourceCategory::kBackingFree, 64).Convert();
    auto assigned = pool.ReserveBacking(ResourceCategory::kBackingAssigned);
    auto materialized = assigned.Convert();
    Throws<std::invalid_argument>(
        [&] { (void)Payload::Committed(pool, std::make_shared<int>()); });
    auto owner =
        Payload::Committed(std::move(materialized), std::make_shared<int>());
    auto moved = std::move(owner);
    assert(!owner.Owner());
    assert(moved.Bytes() == 64);
  }
  assert(ledger.Snapshot().total_bytes == 0);
  Throws<std::invalid_argument>([&] { (void)Payload::Borrowed({}); });
  assert(ledger.Snapshot().total_bytes == 0);
}
void RejectedPayloadsReleaseOwnersFirst() {
  ResourceLedger ledger(kLimits);
  bool destroyed = false;
  {
    auto charge = ledger.Reserve(ResourceCategory::kMetadata, 64).Convert();
    auto owner = std::shared_ptr<const void>(new int, [&](const void* pointer) {
      assert(ledger.Snapshot().bytes[static_cast<std::size_t>(
                 ResourceCategory::kMetadata)] == 64);
      destroyed = true;
      delete static_cast<const int*>(pointer);
    });
    Throws<std::invalid_argument>(
        [&] { (void)Payload::Committed(std::move(charge), std::move(owner)); });
    assert(destroyed);
  }
  assert(ledger.Snapshot().total_bytes == 0);
}
void Baseline() {
  constexpr std::size_t gib = std::size_t{1} << 30;
  ResourceLedger ledger({64 * gib, 64 * gib, 0, 64 * gib});
  std::size_t payload = 0, checkpoint_metadata = 0, chunk_count = 0;
  std::vector<std::size_t> metadata_by_checkpoint;
  {
    FakeAdapter adapter;
    auto layout = std::vector<ComponentDescriptor>(adapter.Components().begin(),
                                                   adapter.Components().end());
    // Use the RFC's measured 27B DFlash2 combined sizes, distributed across the
    // fake's two row components. This is accounting evidence, not GPU numerics.
    layout[0].row_bytes = 32768;
    layout[1].row_bytes = 32768;
    layout[0].rows_per_chunk = layout[1].rows_per_chunk = 2048;
    layout[2].state_bytes = 243700000;
    auto history =
        ExecutionHistory::Cold(ledger, layout, adapter.CompatibilityIdentity());
    std::vector<std::shared_ptr<const Checkpoint>> checkpoints;
    for (Rows boundary = 86016; boundary <= 100352; boundary += 2048) {
      std::vector<Token> tokens(boundary, 7);
      auto checkpoint = history.Capture(
          {tokens,
           InputIdentity(boundary),
           {{kTarget, boundary}, {kDraft, boundary}, {kRecurrent, boundary}},
           CheckpointPurpose::kGrid,
           1},
          [&](const auto& r) { return AccountingPayload(ledger, r); });
      checkpoint_metadata += checkpoint->MetadataBytes();
      metadata_by_checkpoint.push_back(checkpoint->MetadataBytes());
      checkpoints.push_back(std::move(checkpoint));
    }
    const auto snapshot = ledger.Snapshot();
    payload =
        snapshot.bytes[static_cast<std::size_t>(
            ResourceCategory::kBackingMaterialized)] +
        snapshot
            .bytes[static_cast<std::size_t>(ResourceCategory::kPrivateState)];
    assert(payload == 100352ULL * 65536 + 8ULL * 243700000);
    assert(snapshot.bytes[static_cast<std::size_t>(
               ResourceCategory::kPrivateTail)] == 0);
    chunk_count = checkpoints.back()->Components()[0].chunks.size() +
                  checkpoints.back()->Components()[1].chunks.size();
    assert(chunk_count == 98);
    std::cout
        << "{\"payload_bytes\":" << payload << ",\"metadata_bytes\":"
        << snapshot.bytes[static_cast<std::size_t>(ResourceCategory::kMetadata)]
        << ",\"checkpoint_metadata_bytes\":" << checkpoint_metadata
        << ",\"chunk_metadata_bytes\":"
        << ExecutionHistory::ChunkMetadataBytes()
        << ",\"chunk_count\":" << chunk_count
        << ",\"checkpoint_metadata_by_boundary\":[";
    for (std::size_t i = 0; i < metadata_by_checkpoint.size(); ++i) {
      if (i)
        std::cout << ',';
      std::cout << metadata_by_checkpoint[i];
    }
    std::cout << "]}\n";
  }
  assert(ledger.Snapshot().total_bytes == 0);
}
}  // namespace
int main() {
  RejectedPayloadsReleaseOwnersFirst();
  EmptyAndShortCheckpoints();
  PrivateOnlyCheckpoints();
  SharingAndTeardown();
  ColdPrefillsNeverShare();
  RestoredForksAndPrivateTails();
  FailedCapturesAreInvisible();
  GeometryLocationsAndPins();
  RejectInvalidDescriptionsAndProvenance();
  FaultsAtEveryAdmissionAndConversion();
  PayloadValidationAndReleaseOrder();
  Baseline();
  std::cout << "PASS: checkpoint ownership\n";
}
