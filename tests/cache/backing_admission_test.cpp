#include <hip/hip_runtime.h>

#include <atomic>
#include <barrier>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <latch>
#include <map>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/hip/committed_backing.hpp"
#include "src/core/hip/component_cache_resources.hpp"
#include "src/core/hip/transfer_pool.hpp"
#include "src/core/resource_error.hpp"
#include "src/families/qwen/vision/admission.hpp"
#include "src/models/qwen38_flash_next/cache_admission.hpp"
#include "tests/cache/fake_adapter.hpp"

namespace {
std::map<void*, std::size_t> registered;
std::atomic<std::size_t> registered_bytes{};
std::function<void()> verify_segment_ownership;
bool fail_registration{}, fail_query{};
std::atomic<unsigned> live_streams{}, live_events{};

constexpr std::size_t kPayloadBytes = 4 * 4096;
constexpr std::size_t kMetadataBytes = 16384;
constexpr std::size_t kBudgetBytes = 1 << 20;

class Adapter final : public gufo::cache::testing::FakeAdapter {
public:
  std::vector<gufo::cache::BackingClass> BackingPlan(
      std::size_t) const override {
    return {
        {.block_bytes = 4096, .blocks = 4, .metadata_bytes = kMetadataBytes}};
  }
};

void OnDemandLazyBacking(bool inject_registration_failure) {
  using namespace gufo;
  constexpr std::size_t block = 4096;
  constexpr std::size_t cache_bytes = kPayloadBytes + kMetadataBytes;
  constexpr std::size_t execution_bytes = block;
  constexpr std::size_t image_bytes = 2 * block;
  // Enough for the cache OR mixed images and partial backing. The configured
  // cache maximum is optional growth, not an upfront execution obligation.
  constexpr std::size_t available = execution_bytes + kPayloadBytes;
  std::atomic<std::size_t> image_committed{};
  models::qwen::vision::AdmissionBudget admission(
      [&] {
        return available - registered_bytes.load() - image_committed.load();
      },
      execution_bytes);
  cache::ResourceLedger cache_ledger({cache_bytes, cache_bytes, 0, 0});
  fail_registration = false;
  fail_query = false;
  std::barrier start(2);
  std::latch image_live(1), release_image(1);
  std::atomic<bool> synchronize_growth{false};
  const auto refused = [&](bool shared, std::size_t bytes) {
    bool exhausted = false;
    try {
      auto extra =
          shared ? admission.ReserveShared(bytes) : admission.Reserve(bytes);
    } catch (const core::ResourceAdmissionError&) {
      exhausted = true;
    }
    assert(exhausted);
  };
  {
    hip::CommittedBackingPool pool(
        cache_ledger,
        {.ram_bytes = cache_bytes,
         .block_bytes = block,
         .metadata_headroom_bytes = kMetadataBytes,
         .segment_bytes = block,
         .admit_segment =
             [&](std::size_t bytes) {
               assert(bytes == block);
               if (synchronize_growth.exchange(false)) {
                 start.arrive_and_wait();
                 image_live.wait();
               }
               return true;
             },
         .keep_segment =
             [](std::size_t, std::size_t before, std::size_t after) {
               return hip::RegistrationKeepsReserve(before, after);
             },
         .reserve_segment =
             [&](std::size_t bytes) {
               return admission.ReserveShared(bytes);
             }});
    assert(!pool.CommittedBytes());
    assert(admission.ProtectedBytes() == execution_bytes);
    assert(!admission.Outstanding());
    pool.GrowAhead(block);
    pool.WaitForGrowth();
    assert(pool.CommittedBytes() == block);
    assert(admission.Outstanding() == block);
    assert(!admission.PendingBytes());

    synchronize_growth = true;
    fail_registration = inject_registration_failure;
    std::thread image([&] {
      start.arrive_and_wait();
      auto reservation = admission.Reserve(image_bytes);
      std::vector<std::byte> pixels(image_bytes);
      image_committed = image_bytes;
      auto charge = reservation.Convert();
      image_live.count_down();
      release_image.wait();
      std::vector<std::byte>().swap(pixels);
      image_committed = 0;
      charge = {};
    });
    pool.GrowAhead(kPayloadBytes);
    pool.WaitForGrowth();
    assert(pool.CommittedBytes() ==
           (inject_registration_failure ? block : 2 * block));
    assert(cache_ledger.Snapshot().total_bytes ==
           kMetadataBytes + pool.CommittedBytes());
    assert(registered_bytes == pool.CommittedBytes());
    assert(!admission.PendingBytes());
    if (inject_registration_failure) {
      // The failed physical allocation leaked neither a pending reservation
      // nor a committed charge. One explicit recovery follows fault removal.
      fail_registration = false;
      pool.GrowAhead(kPayloadBytes);
      pool.WaitForGrowth();
      assert(pool.CommittedBytes() == 2 * block);
    }
    assert(admission.Outstanding() == 2 * block + image_bytes);
    refused(false, 1);
    refused(true, 1);
    release_image.count_down();
    image.join();
    assert(admission.Outstanding() == 2 * block);
    // Fresh observation reflects committed backing exactly once. Growth can
    // reuse released image capacity, but cannot spend the execution allowance.
    pool.GrowAhead(kPayloadBytes);
    pool.WaitForGrowth();
    assert(pool.CommittedBytes() == kPayloadBytes);
    assert(admission.Outstanding() == kPayloadBytes);
    assert(!admission.PendingBytes());
    refused(false, 1);
    refused(true, 1);
    auto checkpoint =
        pool.TryAcquire(cache::ResourceCategory::kPrivateState, 32);
    assert(checkpoint);
    checkpoint->Bytes().front() = std::byte{7};
    checkpoint->Convert();
  }
  assert(registered.empty() && !registered_bytes);
  assert(!cache_ledger.Snapshot().total_bytes && !admission.Outstanding());
  // A previous refusal does not poison later work or the reservation ledger.
  auto image = admission.Reserve(image_bytes);
  assert(admission.PendingBytes() == image_bytes);
}

void OnDemandHooksKeepLazyReducedGrants(bool image_pressure) {
  using namespace gufo;
  using namespace gufo::models::qwen38_flash_next;
  constexpr std::size_t execution = 4096;
  const auto admission =
      std::make_shared<models::qwen::vision::AdmissionBudget>(
          [] { return execution + kPayloadBytes - registered_bytes.load(); },
          execution);
  cache::ResourceLedger ledger(
      {kBudgetBytes, kBudgetBytes, kBudgetBytes, kBudgetBytes});
  auto image =
      image_pressure ? admission->Reserve(4096) : cache::ResourceReservation{};
  std::string warnings, info;
  {
    hip::ComponentCacheHooks hooks;
    const auto transfers = hip::PlanComponentTransfers(4);
    hooks.transfer_streams = transfers.native_streams;
    hooks.transfer_leases = transfers.leases;
    hooks.host_ceiling = [] { return kBudgetBytes; };
    hooks.warn = [&](std::string_view message) { warnings += message; };
    hooks.info = [&](std::string_view message) { info += message; };
    hooks = VisionCacheHooks(std::move(hooks), admission);
    assert(hooks.host_ceiling() == kBudgetBytes);
    assert(hooks.transfer_streams == 5 && hooks.transfer_leases == 73);
    auto resources = hip::CreateHipComponentCacheResources(
        std::make_unique<Adapter>(), ledger, std::move(hooks));
    assert(info.find("native_streams=5 event_leases=73 allocation=eager") !=
           std::string::npos);
    // Shared serving may grant less than the requested maximum after slots
    // allocate. The actual production hook accepts that grant unchanged.
    resources->commit_backing(kBudgetBytes / 2);
    assert(info.find("required_payload_bytes") == std::string::npos);
    assert(registered_bytes == (image_pressure ? 0 : kPayloadBytes));
    if (image_pressure) {
      assert(warnings.find("event=backing_limited reason=shared_reservation") !=
             std::string::npos);
      assert(admission->Outstanding() == 4096);
    } else {
      assert(warnings.empty());
      assert(admission->Outstanding() == kPayloadBytes);
      assert(!admission->PendingBytes());
    }
  }
  image = {};
  assert(!registered_bytes && !admission->Outstanding());
}

void SegmentTicketFailureAndLifetime(int failure) {
  using namespace gufo;
  constexpr std::size_t block = 4096;
  constexpr std::size_t cache_bytes = kMetadataBytes + kPayloadBytes;
  cache::ResourceLedger cache_ledger({cache_bytes, cache_bytes, 0, 0});
  const auto shared_capacity = failure == 1 ? 0 : kPayloadBytes;
  cache::ResourceLedger shared({shared_capacity, shared_capacity, 0, 0});
  if (failure == 4)
    shared.FailAfter(cache::LedgerStep::kConvert, 0);
  fail_registration = failure == 2;
  fail_query = false;
  verify_segment_ownership = [&] {
    // Both at physical allocation and at release, the corresponding shared
    // reservation/charge must still exist. No early accounting release.
    assert(shared.Snapshot().total_bytes >= registered_bytes.load());
  };
  std::optional<hip::BackingBlock> retained;
  unsigned keep_calls = 0;
  {
    hip::CommittedBackingPool pool(
        cache_ledger,
        {.ram_bytes = cache_bytes,
         .block_bytes = block,
         .metadata_headroom_bytes = kMetadataBytes,
         .segment_bytes = block,
         .admit_segment = {},
         .keep_segment =
             [&](std::size_t bytes, std::size_t, std::size_t) {
               ++keep_calls;
               assert(bytes == block && registered_bytes == block);
               assert(registered.size() == 1);
               assert(shared.Snapshot().reserved_bytes[static_cast<std::size_t>(
                          cache::ResourceCategory::kPrivateTail)] == block);
               if (failure == 5)
                 throw std::bad_alloc();
               if (failure == 6)
                 throw std::runtime_error("injected post-registration failure");
               return failure != 3;
             },
         .reserve_segment =
             [&](std::size_t bytes) {
               return shared.Reserve(cache::ResourceCategory::kPrivateTail,
                                     bytes);
             }});
    pool.GrowAhead(block);
    pool.WaitForGrowth();
    // bad_alloc declines growth; other exceptions are cleaned up and
    // contained by the worker. WaitForGrowth only joins that worker.
    assert(keep_calls == (failure == 1 || failure == 2 ? 0u : 1u));
    assert(pool.CommittedBytes() == (failure ? 0 : block));
    assert(shared.Snapshot().total_bytes == pool.CommittedBytes());
    assert(cache_ledger.Snapshot().total_bytes ==
           kMetadataBytes + pool.CommittedBytes());
    if (failure == 5 || failure == 6) {
      // Synchronous growth declines bad_alloc but propagates other errors.
      bool threw = false;
      try {
        auto result =
            pool.TryAcquire(cache::ResourceCategory::kPrivateState, 32);
        assert(!result);
      } catch (const std::runtime_error& error) {
        assert(std::string(error.what()) ==
               "injected post-registration failure");
        threw = true;
      }
      assert(threw == (failure == 6));
      assert(keep_calls == 2);
      assert(!pool.CommittedBytes() && !pool.PageCommitAllocations());
      assert(registered.empty() && !registered_bytes);
      assert(!shared.Snapshot().total_bytes);
      assert(cache_ledger.Snapshot().total_bytes == kMetadataBytes);
    }
    if (!failure)
      retained = pool.TryAcquire(cache::ResourceCategory::kPrivateState, 32);
  }
  // A capture owner can outlive the pool facade. Its registered segment and
  // shared accounting must survive together until the last physical user.
  assert(shared.Snapshot().total_bytes == (failure ? 0 : block));
  assert(registered_bytes == (failure ? 0 : block));
  retained.reset();
  assert(!shared.Snapshot().total_bytes && !registered_bytes);
  assert(registered.empty() && !cache_ledger.Snapshot().total_bytes);
  verify_segment_ownership = {};
  fail_registration = false;
}

}  // namespace

hipError_t hipHostMalloc(void** pointer, std::size_t bytes, unsigned) {
  *pointer = std::malloc(bytes);
  return *pointer ? hipSuccess : hipErrorOutOfMemory;
}
hipError_t hipHostFree(void* pointer) {
  std::free(pointer);
  return hipSuccess;
}
hipError_t hipHostRegister(void* pointer, std::size_t bytes, unsigned) {
  if (fail_registration)
    return hipErrorOutOfMemory;
  registered.emplace(pointer, bytes);
  registered_bytes.fetch_add(bytes);
  if (verify_segment_ownership)
    verify_segment_ownership();
  return hipSuccess;
}
hipError_t hipHostUnregister(void* pointer) {
  if (verify_segment_ownership)
    verify_segment_ownership();
  registered_bytes.fetch_sub(registered.at(pointer));
  assert(registered.erase(pointer) == 1);
  return hipSuccess;
}
hipError_t hipMemGetInfo(std::size_t* free, std::size_t* total) {
  *free = *total = std::size_t{8} << 30;
  return fail_query ? hipErrorOutOfMemory : hipSuccess;
}
// These tests exercise backing commitment, not asynchronous transfer behavior.
// Native resources are small eager handles; transfer ownership has its own
// suite.
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned) {
  *stream = new int;
  ++live_streams;
  return hipSuccess;
}
hipError_t hipStreamDestroy(hipStream_t stream) {
  delete static_cast<int*>(stream);
  --live_streams;
  return hipSuccess;
}
hipError_t hipStreamSynchronize(hipStream_t) {
  return hipSuccess;
}
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned) {
  *event = new int;
  ++live_events;
  return hipSuccess;
}
hipError_t hipEventDestroy(hipEvent_t event) {
  delete static_cast<int*>(event);
  --live_events;
  return hipSuccess;
}
hipError_t hipEventRecord(hipEvent_t, hipStream_t) {
  return hipSuccess;
}
hipError_t hipEventQuery(hipEvent_t) {
  return hipSuccess;
}
hipError_t hipEventSynchronize(hipEvent_t) {
  return hipSuccess;
}
hipError_t hipMemcpyAsync(void* destination, const void* source,
                          std::size_t bytes, hipMemcpyKind, hipStream_t) {
  std::memcpy(destination, source, bytes);
  return hipSuccess;
}

const char* hipGetErrorString(hipError_t) {
  return "simulated HIP failure";
}

int main() {
  OnDemandLazyBacking(false);
  OnDemandLazyBacking(true);
  OnDemandHooksKeepLazyReducedGrants(false);
  OnDemandHooksKeepLazyReducedGrants(true);
  for (int failure = 0; failure <= 6; ++failure)
    SegmentTicketFailureAndLifetime(failure);
  assert(!live_streams && !live_events && !registered_bytes);
}
