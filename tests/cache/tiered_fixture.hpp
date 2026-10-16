#ifndef GUFO_TESTS_CACHE_TIERED_FIXTURE_HPP_
#define GUFO_TESTS_CACHE_TIERED_FIXTURE_HPP_

#include <cassert>
#include <filesystem>
#include <numeric>

#include "src/cache/disk_catalog.hpp"
#include "src/cache/streaming.hpp"
#include "tests/cache/fake_adapter.hpp"

namespace gufo::cache::testing {
struct TieredDirectory {
  std::filesystem::path path;
  TieredDirectory() {
    char pattern[] = "/tmp/gufo-tiered-test-XXXXXX";
    path = mkdtemp(pattern);
  }
  ~TieredDirectory() { std::filesystem::remove_all(path); }
};
struct TieredGuard final : MutationGuard {
  void BeforeOverwrite(ComponentId, Rows, Rows) override {}
  void BeforeRelease(ComponentId, Rows, Rows) noexcept override {}
};
inline DiskFileId TieredFile(unsigned tag, std::uint64_t id,
                             unsigned component = 0) {
  DiskFileId file{};
  file[0] = tag;
  file[1] = component;
  for (unsigned i = 0; i < 8; ++i)
    file[2 + i] = id >> (i * 8);
  return file;
}
inline StagingAllocation TieredStaging(std::size_t bytes) {
  auto buffer = std::make_shared<std::vector<std::uint8_t>>(bytes);
  return {buffer, *buffer};
}
struct TieredFixture {
  ResourceLedger ledger{{1 << 25, 1 << 24, 16, 1 << 20}};
  TieredDirectory directory;
  FakeAdapter adapter;
  TieredGuard guard;
  FakeStream stream;
  std::unique_ptr<Slot> slot{adapter.CreateSlot(guard)};
  std::vector<Token> tokens, drafts;
  ExecutionHistory history{ExecutionHistory::Cold(
      ledger, adapter.Components(), adapter.CompatibilityIdentity())};
  CompatibilityDigest digest{};
  DiskStore disk{ledger, directory.path, 1 << 24};
  PrefixIndex index{ledger};
  DiskCatalog catalog{ledger, disk, index};
  StreamedStore streams{ledger, disk, 16, TieredStaging,
                        [] { return std::make_unique<FakeStream>(true); }};
  TieredFixture() {
    digest[0] = 42;
    index.Register(adapter.CompatibilityIdentity(), adapter.Components(),
                   digest);
  }
  void Append(unsigned target, unsigned draft) {
    std::vector<Token> suffix(target);
    std::iota(suffix.begin(), suffix.end(), tokens.size() + 7);
    auto draft_suffix = std::span(suffix).first(draft);
    adapter.Append(*slot, suffix, draft_suffix);
    tokens.insert(tokens.end(), suffix.begin(), suffix.end());
    drafts.insert(drafts.end(), draft_suffix.begin(), draft_suffix.end());
  }
  std::shared_ptr<const Checkpoint> Capture() {
    return history.Capture(
        {tokens, InputIdentity(tokens.size()), adapter.Positions(*slot),
         CheckpointPurpose::kPrompt, 1},
        [&](const PayloadRequest& request) {
          auto admission =
              ledger.Reserve(ResourceCategory::kBackingFree, request.bytes);
          auto bytes = std::make_shared<std::vector<std::byte>>(request.bytes);
          auto pool = admission.Convert();
          auto assigned = pool.ReserveBacking(request.category);
          auto completion =
              request.category == ResourceCategory::kPrivateState
                  ? adapter.CapturePrivate(*slot, request.component, *bytes,
                                           stream)
                  : adapter.CopyRowsOut(*slot, request.component, request.first,
                                        request.end, *bytes, stream);
          assert(completion.Wait() == TransferResult::kSucceeded);
          return Payload::Committed(assigned.Convert(), bytes);
        });
  }
  std::shared_ptr<const DiskDescription> Publish(
      const Checkpoint& checkpoint, bool track = true,
      const Slot* borrowed_source = nullptr) {
    DiskManifest manifest;
    manifest.checkpoint = checkpoint.Id();
    manifest.lineage = checkpoint.Lineage();
    manifest.compatibility = digest;
    manifest.input = checkpoint.Input();
    manifest.tokens.assign(checkpoint.Tokens().begin(),
                           checkpoint.Tokens().end());
    std::vector<DiskWriteBuffer> buffers;
    std::vector<Payload> copies;
    for (const auto& c : checkpoint.Components()) {
      DiskComponent component{c.descriptor, c.position, {}, {}, {}};
      auto add = [&](const Payload& payload, bool private_file, DiskFileId id,
                     Rows first = 0, Rows end = 0) {
        auto owner =
            payload.BorrowedFrom()
                ? std::shared_ptr<const std::vector<std::byte>>{}
                : std::static_pointer_cast<const std::vector<std::byte>>(
                      payload.Owner());
        if (!owner) {
          assert(borrowed_source);
          auto pin = payload.PinRows();
          assert(pin.Location());
          auto admission =
              ledger.Reserve(ResourceCategory::kBackingFree, payload.Bytes());
          auto copy = std::make_shared<std::vector<std::byte>>(payload.Bytes());
          auto pool = admission.Convert();
          auto assigned = pool.ReserveBacking(payload.Category());
          const auto location = payload.BorrowedFrom();
          assert(location);
          auto completion = adapter.CopyRowsOut(
              *borrowed_source, c.descriptor.id, first, end, *copy, stream);
          assert(completion.Wait() == TransferResult::kSucceeded);
          copies.push_back(Payload::Committed(assigned.Convert(), copy));
          owner = std::move(copy);
        }
        auto bytes =
            std::span(reinterpret_cast<const std::uint8_t*>(owner->data()),
                      owner->size());
        buffers.push_back({private_file, id, bytes});
        return DiskPayload{id, bytes.size(), DiskChecksum(bytes)};
      };
      for (const auto& chunk : c.chunks)
        component.chunks.push_back(add(chunk.Storage(), false,
                                       TieredFile(1, chunk.Id().value),
                                       chunk.First(), chunk.End()));
      if (c.tail)
        component.tail =
            add(*c.tail, true,
                TieredFile(2, checkpoint.Id().value, c.descriptor.id.value),
                c.position.valid_rows / c.descriptor.rows_per_chunk *
                    c.descriptor.rows_per_chunk,
                c.position.valid_rows);
      if (c.private_state)
        component.private_state =
            add(*c.private_state, true,
                TieredFile(3, checkpoint.Id().value, c.descriptor.id.value));
      manifest.components.push_back(std::move(component));
    }
    disk.Publish(TieredFile(4, checkpoint.Id().value), manifest, buffers);
    if (track) {
      catalog.Track(checkpoint.Id());
      return catalog.Find(checkpoint.Id());
    }
    return disk.Describe(checkpoint.Id());
  }
  std::vector<ComponentAvailability> Resident() const {
    std::vector<ComponentAvailability> availability;
    for (auto descriptor : adapter.Components())
      availability.push_back({descriptor.id, true, false});
    return availability;
  }
  PrefixQuery Query() const {
    return {adapter.CompatibilityIdentity(), tokens,
            InputIdentity(tokens.size()), 0};
  }
};
}  // namespace gufo::cache::testing
#endif
