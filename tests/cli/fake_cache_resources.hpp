#ifndef GUFO_TESTS_CLI_FAKE_CACHE_RESOURCES_HPP_
#define GUFO_TESTS_CLI_FAKE_CACHE_RESOURCES_HPP_

#include <cstring>
#include <memory>
#include <span>
#include <thread>
#include <vector>

#include "src/cli/serve/component_text_cache.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "tests/cache/fake_adapter.hpp"

namespace gufo::server::testing {

struct FakeCacheOptions {
  std::size_t private_blocks{128};
  std::size_t row_backing_bytes{16};
  std::size_t total_row_bytes{2048 * 16};
  // Idle work gets no transfer stream while this is set.
  const bool* defer_idle{};
  std::size_t* captures{};
};

// Host-memory backing for a FakeAdapter, so serving tests run the component
// cache without a device.
inline std::unique_ptr<ComponentCacheResources> MakeFakeCacheResources(
    cache::ResourceLedger& ledger,
    std::unique_ptr<cache::testing::FakeAdapter> adapter,
    FakeCacheOptions options = {}) {
  using cache::testing::FakeStream;
  struct Block {
    cache::ResourceCharge pool;
    std::span<std::byte> data;
    std::shared_ptr<const void> owner;
    bool private_state;
  };
  auto blocks = std::make_shared<std::vector<Block>>();
  // Physical backing commits once, before any request; assigned owner tokens
  // account for the entire block including unused bytes.
  const auto row_backing = options.total_row_bytes;
  const auto row_bytes = options.row_backing_bytes;
  const auto row_blocks = row_backing / row_bytes;
  auto data = std::make_shared<std::vector<std::byte>>(
      row_backing + options.private_blocks * 64);
  for (std::size_t i = 0; i < row_blocks + options.private_blocks; ++i) {
    const bool private_state = i >= row_blocks;
    const auto size = private_state ? 64ULL : row_bytes;
    const auto offset =
        private_state ? row_backing + (i - row_blocks) * 64 : i * row_bytes;
    auto reservation =
        ledger.Reserve(cache::ResourceCategory::kBackingFree, size);
    auto bytes = std::span(*data).subspan(offset, size);
    blocks->push_back({reservation.Convert(), bytes,
                       std::shared_ptr<const void>(data, bytes.data()),
                       private_state});
  }
  auto resources = std::make_unique<ComponentCacheResources>();
  resources->adapter = std::move(adapter);
  resources->owner = blocks;
  resources->stream =
      [defer_idle = options.defer_idle,
       foreground =
           std::this_thread::get_id()]() -> std::unique_ptr<cache::Stream> {
    if (defer_idle && *defer_idle && std::this_thread::get_id() != foreground)
      return {};
    return std::make_unique<FakeStream>(true);
  };
  resources->transfer_batch_size = 4;
  resources->allocate = [captures = options.captures, blocks,
                         data](const cache::PayloadRequest& request) {
    for (const auto& block : *blocks) {
      if (request.bytes > block.data.size())
        continue;
      if ((request.category == cache::ResourceCategory::kPrivateState) !=
          block.private_state)
        continue;
      // Occupied blocks are normal, not allocation failures. Keep the
      // reservation below authoritative if another thread races this scan.
      if (block.pool.Info().category != cache::ResourceCategory::kBackingFree)
        continue;
      try {
        auto reservation = std::make_shared<cache::ResourceReservation>(
            block.pool.ReserveBacking(request.category));
        if (captures)
          ++*captures;
        return ComponentCaptureBuffer{
            block.data,
            [reservation, owner = block.owner] {
              return cache::Payload::Committed(reservation->Convert(), owner);
            },
            data.get(),
            [reservation, owner = block.owner, bytes = block.data, request](
                cache::SlotLease& lease, auto source) {
              return cache::Payload::Borrowed(
                  lease.Borrow(request.component, request.first, request.end,
                               std::move(*reservation), owner,
                               bytes.first(request.bytes), source));
            }};
      } catch (const std::bad_alloc&) {
      }
    }
    throw cache::ResourceExhausted();
  };
  resources->staging = [](std::size_t size) {
    auto data = std::make_shared<std::vector<std::uint8_t>>(size);
    return cache::StagingAllocation{data, *data};
  };
  resources->copy_observed_rows = [](auto source, auto destination,
                                     cache::Stream& stream) {
    return dynamic_cast<FakeStream&>(stream).Submit([source, destination] {
      std::memcpy(destination.data(), source.data(), destination.size());
      return cache::TransferResult::kSucceeded;
    });
  };
  return resources;
}

}  // namespace gufo::server::testing

#endif
