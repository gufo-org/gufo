#include <array>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <iostream>
#include <latch>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "tests/families/qwen/vision_memory_fixture.hpp"

namespace vision = gufo::models::qwen::vision;
using gufo::core::ResourceAdmissionError;

namespace {
void Require(bool condition, const char* description) {
  if (!condition)
    throw std::runtime_error(description);
}

template<class Error = ResourceAdmissionError, class Function>
std::string Refused(Function&& function) {
  try {
    function();
  } catch (const Error& error) {
    return error.what();
  }
  throw std::runtime_error("expected admission refusal");
}

void StartupPolicies() {
  constexpr std::size_t available = 40088076288ULL;
  constexpr std::size_t base = 7527108948ULL;
  constexpr std::size_t worst = 10214609236ULL;
  const auto conservative =
      vision::PlanStartupMemory(available + 1024, 1024, base, worst);
  Require(conservative.state_capacity_bytes == available &&
              conservative.per_session_bytes == worst && conservative.Fits(3) &&
              !conservative.Fits(4),
          "default retains the complete image-capable claim");
  const auto demand =
      vision::PlanStartupMemory(available + 1024, 1024, base, worst, true);
  Require(demand.Fits(4) && !demand.Fits(6),
          "text sessions exclude hypothetical images");
  const auto exact = vision::PlanStartupMemory(4 * base, 0, base, worst, true);
  Require(exact.Fits(4) &&
              !vision::PlanStartupMemory(4 * base - 1, 0, base, worst, true)
                   .Fits(4),
          "exact fit and genuine text shortage");
  Require(!vision::PlanStartupMemory(10, 11, base, worst).Fits(1),
          "deferred work saturates capacity");
  const auto maximum = std::numeric_limits<std::size_t>::max();
  Require(!exact.Fits(maximum), "session aggregation cannot wrap");
  Refused([&] { (void)exact.RequiredBytes(maximum); });
  Refused([&] { (void)exact.DeficitBytes(maximum, maximum); });
  const vision::StartupMemoryPlan sum_overflow{maximum, maximum - 2, 3};
  Refused([&] { (void)sum_overflow.RequiredBytes(1); });
}

void DeviceGateAndRuntimeCapacity() {
  constexpr std::size_t device = 40497639424ULL;
  constexpr std::size_t host_available = 38922952704ULL;
  constexpr std::size_t host_reserve = 4294967296ULL;
  constexpr std::size_t host_ceiling = 34627985408ULL;
  constexpr std::size_t base = 7527108948ULL;
  constexpr std::size_t worst = 10214609236ULL;
  constexpr std::size_t scratch = 418013184ULL;
  constexpr std::size_t cache = 4294967296ULL;
  const auto memory = vision::SelectStartupCapacity(device, host_ceiling);
  Require(host_available - host_reserve == host_ceiling &&
              std::string(memory.LimiterName()) == "host",
          "runtime host limit remains separate");
  const auto gate = vision::PlanStartupMemory(memory.device_admission_bytes,
                                              scratch, base, worst, true);
  Require(gate.Fits(4) &&
              gate.state_capacity_bytes - 4 * base == 9971190448ULL &&
              gate.RequiredBytes(4) == 4 * base + scratch,
          "device text gate does not promise the full optional cache");
  Require(host_available - 4 * base - cache - scratch == 4101536432ULL &&
              4101536432ULL < host_reserve,
          "text eligibility cannot qualify the full cache and image workload");
}

void OnDemandObligations() {
  const auto legacy = vision::PlanStartupMemory(350, 10, 40, 100);
  const auto demand = vision::PlanStartupMemory(350, 10, 40, 100, true);
  Require(!legacy.Fits(4) && demand.Fits(4),
          "images are not a text prerequisite");
  Require(
      demand.shared_reservation_bytes == 10 && demand.RequiredBytes(4) == 170,
      "only deferred execution work is shared startup work");
  Require(vision::PlanStartupMemory(170, 10, 40, 100, true).Fits(4) &&
              !vision::PlanStartupMemory(169, 10, 40, 100, true).Fits(4),
          "text boundary remains enforced");
  std::size_t free =
      160;  // Host/image budget can be below accepted device work.
  auto budget =
      std::make_shared<vision::AdmissionBudget>([&] { return free; }, 170);
  Require(budget->Outstanding() == 0 && budget->ProtectedBytes() == 170,
          "constructing runtime coordination does not repeat the host gate");
  Refused([&] { (void)budget->Reserve(1); });
  budget->ClaimExecutionOwner();
  Refused([&] { budget->ClaimExecutionOwner(); });
  // More headroom arrives. No free-memory query can erase pending claims.
  free = 300;
  auto image = budget->Reserve(100);
  auto copy = budget->ReserveShared(30);
  Refused([&] { (void)budget->Reserve(1); });
  Refused([&] { (void)budget->ReserveShared(1); });
  Require(budget->PendingBytes() == 130 && budget->AccountedBytes() == 300,
          "images and optional copies share pending admission atomically");
  // Actual execution allocations lower free memory; remove only their pending
  // obligation. They are never credited back to the observation.
  free -= 160;
  budget->CommitExecution(160);
  Refused([&] { (void)budget->Reserve(1); });
  Require(budget->ProtectedBytes() == 10 && budget->Capacity() == 130,
          "committed execution is neither charged twice nor added to free");
  // Optional cache growth has no permanent maximum reservation; its arriving
  // segment competes with images and snapshot copies.
  copy = {};
  auto segment = budget->ReserveShared(30);
  free -= 30;
  auto committed = segment.Convert();
  Require(budget->PendingBytes() == 100,
          "registered cache pages are reflected only in fresh observations");
  Refused([&] { (void)budget->Reserve(1); });
  image = {};
  auto next = budget->Reserve(100);
  next = {};
  committed = {};  // Physical pages released first in the real owner.
  free += 30;
  auto recovered = budget->Reserve(130);
  recovered = {};
  // Free memory can also fall due to another process. No stale initial cap.
  free = 10;
  Refused([&] { (void)budget->Reserve(1); });
  Require(!budget->Outstanding(),
          "refusal cannot leak reservations or harm text");
  Refused<std::logic_error>([&] { budget->CommitExecution(11); });
}

void ObservationOrdering() {
  // Simulate allocation/conversion between the pending sample and observer:
  // old pending must remain deducted, even if the observation is early.
  gufo::cache::ResourceReservation image;
  gufo::cache::ResourceCharge pixels;
  bool commit_during_observation = false;
  std::size_t free = 100;
  auto budget = std::make_shared<vision::AdmissionBudget>([&] {
    const auto observed = free;
    if (commit_during_observation) {
      commit_during_observation = false;
      free -= 60;
      pixels = image.Convert();
    }
    return observed;
  });
  image = budget->Reserve(60);
  commit_during_observation = true;
  Refused([&] { (void)budget->ReserveShared(41); });
  Require(budget->PendingBytes() == 0 && budget->Outstanding() == 60,
          "commitment during observation cannot hide an accepted allocation");
  auto exact = budget->ReserveShared(40);
  Refused([&] { (void)budget->Reserve(1); });
  exact = {};
  pixels = {};
  free += 60;
  vision::AdmissionLease retained(budget);
  retained.Grow(80);
  free -= 50;
  retained.SetCommitted(50);
  Require(budget->PendingBytes() == 30,
          "only remaining image work stays pending");
  auto peer = budget->Reserve(20);
  Refused([&] { (void)budget->Reserve(1); });
  // Replacing/shrinking a retained owner preserves its aggregate coverage.
  retained.SetCommitted(0);
  free += 50;
  retained.Shrink(30);
  retained.Clear();
  peer = {};
  Require(budget->Outstanding() == 0 && budget->PendingBytes() == 0,
          "reset releases materialization markers and tickets once");
}

void ShrinkingReservations() {
  auto budget =
      std::make_shared<vision::AdmissionBudget>([] { return 130; }, 30);
  vision::AdmissionLease image(budget);
  image.Grow(60);
  image.Grow(100);
  image.Shrink(75);
  Require(image.Bytes() == 75 && budget->Outstanding() == 75,
          "replacement peak shrinks without another allocation at an exact "
          "boundary");
  auto other = budget->Reserve(25);
  Refused([&] { image.Grow(76); });
  image.Shrink(50);
  Require(image.Bytes() == 50 && budget->Outstanding() == 75,
          "shrinking spans multiple growth reservations");
  image.Shrink(70);
  Require(image.Bytes() == 50, "shrink cannot increase an existing claim");
  image.Shrink(0);
  Require(budget->Outstanding() == 25,
          "only the genuinely freed owner releases");
  Refused<std::invalid_argument>([&] { other.Shrink(26); });
  Require(other.Info().bytes == 25, "invalid shrink leaves ledger unchanged");
  other.Shrink(10);
  auto committed = other.Convert();
  Require(committed.Info().bytes == 10 && budget->Outstanding() == 10,
          "conversion after shrink preserves retained allocation ownership");
  Refused<std::logic_error>([&] { other.Shrink(0); });
  committed = {};
  Require(budget->AccountedBytes() == 30,
          "released ownership keeps base protection");

  gufo::cache::ResourceLedger pool({100, 100, 0, 0});
  auto backing = pool.Reserve(gufo::cache::ResourceCategory::kBackingFree, 100);
  Refused<std::invalid_argument>([&] { backing.Shrink(50); });
  auto block = backing.Convert();
  auto assigned =
      block.ReserveBacking(gufo::cache::ResourceCategory::kPrivateState);
  Refused<std::invalid_argument>([&] { assigned.Shrink(50); });
  Require(pool.Snapshot().ram_bytes == 100,
          "pool backing cannot be partially released");

  auto prep_budget = std::make_shared<vision::AdmissionBudget>(
      [] {
        return vision::kPreparationScratchBytes + vision::kPreparedImageBytes +
               10;
      },
      10);
  {
    vision::PreparationReservation prep(prep_budget, 1);
    prep.Finish(123);
    Require(
        prep_budget->Outstanding() == 123,
        "completed preparation releases scratch and unused pixel allowance");
  }
  Require(prep_budget->Outstanding() == 0,
          "prepared pixel owner releases only when its content is gone");
}

void StartupCapacitySelection() {
  const auto device = vision::SelectStartupCapacity(100, 120);
  const auto host = vision::SelectStartupCapacity(120, 100);
  const auto equal = vision::SelectStartupCapacity(100, 100);
  Require(device.available_bytes == 100 &&
              std::string(device.LimiterName()) == "device" &&
              host.available_bytes == 100 &&
              std::string(host.LimiterName()) == "host" &&
              equal.available_bytes == 100 &&
              std::string(equal.LimiterName()) == "equal",
          "runtime capacity selects the limiting observation without adding "
          "committed bytes");
}

void GeometryAndHistory() {
  const vision::ImageDimensions tiny{32, 32};
  const vision::ImageDimensions medium{1024, 1024};
  const vision::ImageDimensions largest{4096, 4096};
  Require(vision::ImageScratchBytes(tiny) == 184704,
          "32px encoder allocation oracle includes pinned upload");
  Require(vision::ImageEmbeddingBytes(tiny, 2560) == 10240,
          "one merged image token is 2560 FP32 values");
  Require(vision::ImageScratchBytes(medium) == 194150400,
          "4096 patches use the special 64-row attention tile");
  Require(vision::ImageScratchBytes(largest) == 3471376384ULL,
          "maximum encoder working set includes the capped upload buffer");
  const std::array one{tiny};
  const std::array history{tiny, medium};
  Require(vision::testing::RequestMemoryBound(80, one, 2560) == 199936,
          "request accounts host and device positions and embeddings");
  Require(vision::testing::RequestMemoryBound(1200, history, 2560) == 207824000,
          "image history sums embeddings and takes peak sequential scratch");
  Require(vision::testing::RequestMemoryBound(80, {}, 2560) == 1920,
          "checkpoint layout restoration needs positional storage alone");
  Require(vision::testing::RequestMemoryBound(0, {}, 2560) == 0,
          "text requests need no dynamic vision storage");
  Refused<std::invalid_argument>(
      [] { (void)vision::ImageScratchBytes({0, 32}); });
  Refused<std::invalid_argument>(
      [] { (void)vision::ImageScratchBytes({33, 32}); });
  Refused<std::invalid_argument>(
      [] { (void)vision::ImageScratchBytes({4096, 4128}); });
  Refused<std::invalid_argument>(
      [&] { (void)vision::ImageEmbeddingBytes(tiny, 0); });
  const auto maximum = std::numeric_limits<std::size_t>::max();
  Refused([&] { (void)vision::CheckedAdd(maximum, 1); });
  Refused([&] { (void)vision::CheckedMultiply(maximum, 2); });
  Refused(
      [&] { (void)vision::testing::RequestMemoryBound(maximum, one, 2560); });
  Require(vision::CheckedMultiply(0, maximum) == 0 &&
              vision::CheckedAdd(maximum, 0) == maximum,
          "valid arithmetic boundaries remain valid");
}

void LifetimeAndGrowth() {
  auto budget = std::make_shared<vision::AdmissionBudget>([] { return 100; });
  vision::AdmissionLease a(budget), b(budget);
  a.Grow(40);
  a.Grow(30);
  Require(a.Bytes() == 40 && budget->Outstanding() == 40,
          "smaller reused request keeps its retained allocation claim");
  a.Grow(60);
  b.Grow(40);
  Require(budget->Outstanding() == 100, "exact request boundary admits");
  const auto error = Refused([&] { a.Grow(61); });
  Require(error.find("requested_bytes=1") != std::string::npos &&
              error.find("outstanding_bytes=100") != std::string::npos &&
              error.find("capacity_bytes=100") != std::string::npos,
          "resource error reports only accounting diagnostics");
  Require(a.Bytes() == 60 && budget->Outstanding() == 100,
          "refused growth leaves all previous state unchanged");
  // Neither repeated hipMemGetInfo-like observations nor a second caller
  // admitting against the same capacity can replenish the ledger.
  for (int observation = 0; observation != 3; ++observation) {
    Require(budget->Capacity() == 100,
            "unchanged observation cannot erase pending claims");
    Refused([&] { (void)budget->Reserve(1); });
  }
  a.Clear();  // physical buffers were released by its owner
  Require(budget->Outstanding() == 40 && !a.Bytes(),
          "text transition releases its image buffers and claim");
  a.Grow(60);
  a.Clear();
  a.Clear();
  b.Clear();
  Require(!budget->Outstanding(), "repeated reset cannot double-release");

  vision::AdmissionLease conservative;
  conservative.Grow(std::numeric_limits<std::size_t>::max());
  Require(!conservative.Bytes(), "default policy does not add a runtime pool");
  {
    vision::AdmissionLease survivor(budget);
    survivor.Grow(100);
    budget.reset();
    // The ledger's handles retain accounting even after its facade is gone.
    survivor.Clear();
  }
}

void ConcurrentReservations() {
  constexpr std::size_t workers = 8;
  auto budget = std::make_shared<vision::AdmissionBudget>([] { return 100; });
  std::barrier start(workers + 1);
  std::latch attempted(workers), release(1);
  std::atomic<std::size_t> admitted = 0, refused = 0;
  std::vector<std::thread> threads;
  for (std::size_t i = 0; i != workers; ++i) {
    threads.emplace_back([&] {
      vision::AdmissionLease lease(budget);
      start.arrive_and_wait();
      try {
        lease.Grow(60);
        ++admitted;
      } catch (const ResourceAdmissionError&) {
        ++refused;
      }
      attempted.count_down();
      release.wait();
    });
  }
  start.arrive_and_wait();
  attempted.wait();
  Require(
      admitted == 1 && refused == workers - 1 && budget->Outstanding() == 60,
      "concurrent requests cannot spend the same capacity");
  release.count_down();
  for (auto& thread : threads)
    thread.join();
  Require(!budget->Outstanding(), "request destruction releases reservations");
}

// The actual AdmissionLease owns logical memory while these tiny CPU buffers
// stand in for device positions, embeddings and temporary work. Inject after
// each possible partial-allocation stage; no large physical buffers are used.
void AllocationFailuresAndCancellation() {
  auto budget = std::make_shared<vision::AdmissionBudget>([] { return 100; });
  for (std::size_t fail_at = 0; fail_at != 4; ++fail_at) {
    vision::AdmissionLease lease(budget);
    std::vector<std::unique_ptr<std::array<std::byte, 4>>> allocations;
    lease.Grow(80);
    Refused<std::bad_alloc>([&] {
      for (std::size_t stage = 0; stage != 4; ++stage) {
        if (stage == fail_at)
          throw std::bad_alloc();
        allocations.push_back(std::make_unique<std::array<std::byte, 4>>());
      }
    });
    Require(budget->Outstanding() == 80,
            "allocation failure retains claim for partial device allocations");
    vision::AdmissionLease peer(budget);
    peer.Grow(20);
    Refused([&] { peer.Grow(21); });
    allocations.clear();
    lease.Clear();
    peer.Clear();
    Require(!budget->Outstanding(), "failed request cleanup restores capacity");
  }
  for (const bool cancel_after_allocation : {false, true}) {
    vision::AdmissionLease lease(budget);
    std::unique_ptr<std::array<std::byte, 4>> allocation;
    lease.Grow(100);
    if (cancel_after_allocation)
      allocation = std::make_unique<std::array<std::byte, 4>>();
    // Cancellation does not release the reservation behind live buffers.
    Require(budget->Outstanding() == 100, "cancelled state remains charged");
    allocation.reset();
    lease.Clear();
  }
  vision::AdmissionLease subsequent_text(budget);
  subsequent_text.Grow(0);
  Require(!budget->Outstanding(),
          "text remains usable after all failure stages");
}

void PreparedHistoryAndRestoredLayouts() {
  const std::array history{vision::ImageDimensions{32, 32},
                           vision::ImageDimensions{64, 32}};
  const auto peak = vision::testing::RequestMemoryBound(120, history, 2560);
  auto budget = std::make_shared<vision::AdmissionBudget>([=] { return peak; });
  vision::AdmissionLease state(budget);
  state.Grow(peak);
  const auto before = budget->Outstanding();
  // RAM checkpoint reuse keeps the same request attachment. Disk restore
  // reconstructs only RoPE layout; both use Grow and cannot release the
  // existing embedding claim just because the restored prefix is shorter.
  state.Grow(peak);
  state.Grow(vision::testing::RequestMemoryBound(80, {}, 2560));
  Require(budget->Outstanding() == before,
          "RAM reuse and shorter restored layout retain attached image claims");
  Refused([&] {
    state.Grow(vision::CheckedAdd(
        peak, vision::testing::RequestMemoryBound(160, history, 2560)));
  });
  Require(budget->Outstanding() == peak,
          "replacement peak refusal preserves old request buffers");
  state.Clear();
  // A fresh slot restoring a disk layout pays its complete positional bound;
  // subsequent image reattachment grows before encoding or continuation.
  state.Grow(vision::testing::RequestMemoryBound(80, {}, 2560));
  state.Grow(peak);
  Require(budget->Outstanding() == peak,
          "fresh restore followed by image history cannot bypass admission");
  state.Clear();
}

void PreparationLifetimes() {
  const auto scratch = vision::kPreparationScratchBytes;
  const auto pixels = vision::kPreparedImageBytes;
  const auto peak = scratch + 2 * pixels;
  auto budget = std::make_shared<vision::AdmissionBudget>([=] { return peak; });
  {
    vision::PreparationReservation text(budget, 0);
    vision::PreparationReservation conservative(nullptr, 1);
    Require(budget->Outstanding() == 0,
            "text/default preparation creates no new reservation");
  }
  {
    auto preparation =
        std::make_shared<vision::PreparationReservation>(budget, 2);
    Require(budget->Outstanding() == peak,
            "all history pixels and transient preprocessing fit exactly");
    Refused([&] { vision::PreparationReservation peer(budget, 1); });
    preparation->Finish();
    preparation->Finish();
    Require(budget->Outstanding() == 2 * pixels,
            "successful preparation releases scratch only and only once");
    auto queued = preparation;
    auto session_history = preparation;
    preparation.reset();
    queued.reset();  // Cancellation drops one request owner.
    Require(budget->Outstanding() == 2 * pixels,
            "cancelled queued request cannot release live image history");
    session_history.reset();
    Require(budget->Outstanding() == 0,
            "last prepared-pixel owner returns its capacity");
  }
  {
    auto too_small =
        std::make_shared<vision::AdmissionBudget>([=] { return peak - 1; });
    Refused([&] { vision::PreparationReservation request(too_small, 2); });
    Require(too_small->Outstanding() == 0,
            "failure of the second reservation rolls back transient scratch");
  }
  for (const bool finished : {false, true}) {
    Refused<std::bad_alloc>([&] {
      vision::PreparationReservation request(budget, 2);
      // Tiny storage stands in for decoded/prepared pixels. Its lifetime
      // precedes reservation destruction under cancellation or exceptions.
      auto allocation = std::make_unique<std::array<std::byte, 4>>();
      if (finished)
        request.Finish();
      throw std::bad_alloc();
    });
    Require(budget->Outstanding() == 0,
            "allocation/cancellation failure returns every preparation claim");
  }
  Refused([&] {
    vision::PreparationReservation overflow(
        budget, std::numeric_limits<std::size_t>::max());
  });
  Require(budget->Outstanding() == 0,
          "image-count overflow cannot leak claims");

  std::barrier start(3);
  std::latch attempted(2), release(1);
  std::atomic<unsigned> admitted{0}, refused{0};
  const auto prepare = [&] {
    std::shared_ptr<vision::PreparationReservation> reservation;
    start.arrive_and_wait();
    try {
      reservation = std::make_shared<vision::PreparationReservation>(budget, 2);
      ++admitted;
    } catch (const ResourceAdmissionError&) {
      ++refused;
    }
    attempted.count_down();
    release.wait();
  };
  std::thread a(prepare), b(prepare);
  start.arrive_and_wait();
  attempted.wait();
  Require(admitted == 1 && refused == 1 && budget->Outstanding() == peak,
          "concurrent preprocessing cannot spend the same capacity");
  release.count_down();
  a.join();
  b.join();
  Require(budget->Outstanding() == 0,
          "preprocessing cancellation returns the shared quota");
}
}  // namespace

int main() {
  StartupPolicies();
  DeviceGateAndRuntimeCapacity();
  OnDemandObligations();
  ObservationOrdering();
  ShrinkingReservations();
  StartupCapacitySelection();
  GeometryAndHistory();
  LifetimeAndGrowth();
  ConcurrentReservations();
  AllocationFailuresAndCancellation();
  PreparedHistoryAndRestoredLayouts();
  PreparationLifetimes();
  std::cout << "vision admission checks passed\n";
}
