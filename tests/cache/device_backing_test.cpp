#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

#include "src/core/hip/committed_backing.hpp"
#include "src/core/hip/component_cache_resources.hpp"
#include "tests/cache/fake_adapter.hpp"

namespace {
void DevicePressureFallsBackToHost() {
  using namespace gufo;
  constexpr std::size_t chunk = 512ULL << 10;
  class Adapter final : public cache::testing::FakeAdapter {
  public:
    bool SupportsDeviceRows() const override { return true; }
    std::span<const cache::ComponentDescriptor> Components() const override {
      return components;
    }
    std::vector<cache::BackingClass> BackingPlan(std::size_t) const override {
      return {{64ULL << 10, 32, 64ULL << 10}};
    }
    const std::array<cache::ComponentDescriptor, 1> components{
        cache::ComponentDescriptor{
            {1}, 1, cache::ComponentKind::kAppendRows, 1, chunk, 0, 0}};
  };
  cache::ResourceLedger ledger(
      {8ULL << 20, 6ULL << 20, 0, 8ULL << 20, 2 * chunk});
  {
    auto resources = hip::CreateHipComponentCacheResources(
        std::make_unique<Adapter>(), ledger,
        {[] { return 1ULL << 30; }, [](auto) {}, [](auto) {}, 2});
    resources->commit_backing(4ULL << 20);
    // A short history still uses host backing, even when device capacity is
    // available. The larger frontier below opts into the additional tier.
    auto short_history = resources->allocate(
        {{1}, cache::ResourceCategory::kBackingAssigned, 0, 1, chunk});
    assert(short_history.domain == cache::MemoryDomain::kHost);
    auto short_payload = short_history.finish();
    const cache::PayloadRequest request{
        {1}, cache::ResourceCategory::kBackingAssigned, 8, 9, chunk};
    const std::array requests{request, request, request};
    assert(resources->can_capture(requests));
    std::vector<cache::Payload> payloads;
    for (int i = 0; i != 3; ++i) {
      auto buffer = resources->allocate(request);
      if (buffer.domain == cache::MemoryDomain::kDevice) {
        assert(hipMemset(buffer.bytes.data(), 29, buffer.bytes.size()) ==
               hipSuccess);
        assert(hipDeviceSynchronize() == hipSuccess);
      } else {
        std::fill(buffer.bytes.begin(), buffer.bytes.end(), std::byte{29});
      }
      payloads.push_back(buffer.finish());
    }
    assert(payloads[2].Domain() == cache::MemoryDomain::kHost);
    if (ledger.Snapshot().device_bytes)
      assert(payloads[0].Domain() == cache::MemoryDomain::kDevice);
    std::cout << "device_bytes=" << ledger.Snapshot().device_bytes
              << " host_fallback=verified\n";
    std::vector<std::byte> copy(chunk);
    for (const auto& payload : payloads) {
      if (payload.Domain() == cache::MemoryDomain::kDevice) {
        assert(hipMemcpy(copy.data(), payload.Owner().get(), chunk,
                         hipMemcpyDeviceToHost) == hipSuccess);
      } else {
        const auto* bytes =
            static_cast<const std::byte*>(payload.Owner().get());
        std::copy_n(bytes, chunk, copy.data());
      }
      assert(std::ranges::all_of(
          copy, [](auto byte) { return byte == std::byte{29}; }));
    }
  }
  assert(ledger.Snapshot().total_bytes == 0);
}
}  // namespace

int main() {
  using namespace gufo;
  constexpr std::size_t segment = 128ULL << 20;
  constexpr std::size_t metadata = 2ULL << 20;
  std::size_t available{}, total{};
  if (hipMemGetInfo(&available, &total) != hipSuccess ||
      available < (1ULL << 30))
    return 77;
  cache::ResourceLedger ledger(
      {3 * segment + 3 * metadata, 3 * metadata, 0, 3 * segment, 3 * segment});
  auto pool = std::make_unique<hip::CommittedBackingPool>(
      ledger, hip::BackingPoolConfig{3 * segment + metadata,
                                     512ULL << 10,
                                     metadata,
                                     segment,
                                     {},
                                     {},
                                     cache::MemoryDomain::kDevice});
  std::vector<cache::Payload> payloads;
  std::optional<hip::BackingPin> held;
  for (int i = 0; i != 3; ++i) {
    auto block =
        pool->TryAcquire(cache::ResourceCategory::kBackingAssigned, segment);
    assert(block);
    assert(hipMemset(block->Bytes().data(), 17 + i, segment) == hipSuccess);
    assert(hipDeviceSynchronize() == hipSuccess);
    block->Convert();
    if (i == 2)
      held.emplace(block->PinPersistence());
    payloads.push_back(std::move(*block).TakePayload());
    assert(payloads.back().Domain() == cache::MemoryDomain::kDevice);
    if (i) {
      const auto a = payloads[i - 1].Owner(), b = payloads[i].Owner();
      assert(a.owner_before(b) || b.owner_before(a));
    }
  }
  assert(ledger.Snapshot().device_bytes == 3 * segment);
  assert(ledger.Snapshot().ram_bytes == metadata);
  assert(!pool->TryAcquire(cache::ResourceCategory::kBackingAssigned, segment));
  auto host = std::make_unique<hip::CommittedBackingPool>(
      ledger, hip::BackingPoolConfig{2 * metadata, 64ULL << 10, metadata});
  const std::array rows{cache::PayloadRequest{
      {1}, cache::ResourceCategory::kBackingAssigned, 0, 1, 512ULL << 10}};
  // Host free pages cannot admit another device allocation.
  assert(!pool->CanFit(rows));
  auto host_payload =
      host->TryAcquire(cache::ResourceCategory::kPrivateState, metadata);
  assert(host_payload);
  host_payload->Convert();
  pool.reset();
  std::vector<unsigned char> readback(4ULL << 20);
  for (int i = 0; i != 3; ++i)
    for (std::size_t offset = 0; offset != segment; offset += readback.size()) {
      const auto* source =
          static_cast<const std::byte*>(payloads[i].Owner().get());
      assert(hipMemcpy(readback.data(), source + offset, readback.size(),
                       hipMemcpyDeviceToHost) == hipSuccess);
      assert(std::ranges::all_of(readback,
                                 [i](auto byte) { return byte == 17 + i; }));
    }
  const void* pinned = payloads.back().Owner().get();
  payloads.clear();
  const std::array private_state{cache::PayloadRequest{
      {2}, cache::ResourceCategory::kPrivateState, 0, 0, metadata}};
  assert(ledger.Snapshot().device_free_backing_bytes == 2 * segment);
  // Device free pages cannot satisfy a host-only private capture or stop
  // that host pool from preparing its own reserve.
  assert(!host->CanFit(private_state));
  assert(ledger.Snapshot().device_bytes == 3 * segment);
  assert(ledger.Snapshot().persistence_pinned_bytes == segment);
  assert(hipMemcpy(readback.data(), pinned, readback.size(),
                   hipMemcpyDeviceToHost) == hipSuccess);
  assert(std::ranges::all_of(readback, [](auto byte) { return byte == 19; }));
  held.reset();
  host_payload.reset();
  host.reset();
  assert(ledger.Snapshot().total_bytes == 0);
  assert(ledger.Snapshot().device_bytes == 0);
  DevicePressureFallsBackToHost();
}
