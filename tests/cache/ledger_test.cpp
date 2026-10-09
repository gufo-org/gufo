#include "src/cache/ledger.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cassert>
#include <limits>
#include <memory>
#include <random>
#include <thread>
#include <utility>
#include <vector>

using namespace gufo::cache;

namespace {
constexpr auto Free = ResourceCategory::kBackingFree;
constexpr auto Spill = ResourceCategory::kBackingReserved;
constexpr auto Chunk = ResourceCategory::kBackingMaterialized;
constexpr auto Private = ResourceCategory::kPrivateState;
constexpr auto Tail = ResourceCategory::kPrivateTail;
constexpr auto Metadata = ResourceCategory::kMetadata;
constexpr auto Staging = ResourceCategory::kTransferStaging;
constexpr std::size_t Index(ResourceCategory category) {
  return static_cast<std::size_t>(category);
}
template<class Exception, class F>
void Throws(F&& function) {
  bool caught = false;
  try {
    function();
  } catch (const Exception&) {
    caught = true;
  }
  assert(caught);
}
void Empty(const ResourceLedger& ledger) {
  const auto snapshot = ledger.Snapshot();
  assert(snapshot.total_bytes == 0);
  assert(snapshot.ram_bytes == 0);
  assert(snapshot.persistence_pinned_bytes == 0);
  for (auto bytes : snapshot.bytes)
    assert(bytes == 0);
  for (auto bytes : snapshot.reserved_bytes)
    assert(bytes == 0);
}
void BackingAndPins() {
  ResourceLedger ledger({128, 128, 32, 64});
  auto reservation = ledger.Reserve(Free, 64);
  assert(ledger.Snapshot().reserved_bytes[Index(Free)] == 64);
  auto pool = reservation.Convert();  // caller has committed physical pages
  auto spill = pool.ReserveSpill();   // borrower admission, whole pool block
  assert(ledger.Snapshot().bytes[Index(Free)] == 0);
  assert(ledger.Snapshot().bytes[Index(Spill)] == 64);
  Throws<std::bad_alloc>([&] { auto duplicate = pool.ReserveSpill(); });
  auto chunk = spill.Convert();  // caller has completed preservation
  auto child = chunk;
  auto pin = chunk.PinPersistence();
  auto second_pin = child.PinPersistence();
  auto job_reference = pin;
  assert(ledger.Snapshot().persistence_pinned_bytes == 64);
  assert(ledger.Snapshot().total_bytes == 64);
  chunk = {};
  child = {};
  pool = {};
  pin = {};
  second_pin = {};
  assert(ledger.Snapshot().bytes[Index(Chunk)] == 64);
  job_reference = {};
  Empty(ledger);
  assert(ledger.Snapshot().peak_total_bytes == 64);

  auto block = ledger.Reserve(Free, 64).Convert();
  {
    auto cancelled = block.ReserveSpill();
  }
  assert(ledger.Snapshot().bytes[Index(Free)] == 64);
  {
    auto materialized = block.ReserveSpill().Convert();
  }
  assert(ledger.Snapshot().bytes[Index(Free)] == 64);
  Throws<std::logic_error>([&] { auto pin_free = block.PinPersistence(); });
  block = {};
  Empty(ledger);

  block = ledger.Reserve(Free, 64).Convert();
  auto borrowed = block.ReserveSpill();
  auto queued = borrowed.PinPersistence();
  assert(ledger.Snapshot().persistence_pinned_bytes == 64);
  borrowed = {};
  assert(ledger.Snapshot().bytes[Index(Spill)] == 64);
  Throws<std::bad_alloc>([&] { auto reuse = block.ReserveSpill(); });
  queued = {};
  assert(ledger.Snapshot().bytes[Index(Free)] == 64);
  borrowed = block.ReserveSpill();
  queued = borrowed.PinPersistence();
  auto preserved = borrowed.Convert();
  assert(ledger.Snapshot().bytes[Index(Chunk)] == 64);
  assert(ledger.Snapshot().persistence_pinned_bytes == 64);
  preserved = {};
  block = {};
  queued = {};
  Empty(ledger);
}
void LimitsAndFaults() {
  ResourceLedger ledger({100, 80, 30, 20});
  auto retained = ledger.Reserve(Private, 70).Convert();
  auto staging = ledger.Reserve(Staging, 30).Convert();
  auto unchanged = ledger.Snapshot();
  Throws<std::bad_alloc>([&] { auto r = ledger.Reserve(Metadata, 1); });
  Throws<std::bad_alloc>([&] { auto r = ledger.Reserve(Staging, 1); });
  Throws<std::bad_alloc>([&] { auto p = retained.PinPersistence(); });
  assert(ledger.Snapshot() == unchanged);
  retained = {};
  staging = {};
  for (auto category : {Free, Private, Tail, Metadata, Staging}) {
    ledger.FailAfter(LedgerStep::kReserve, 0);
    unchanged = ledger.Snapshot();
    Throws<std::bad_alloc>([&] { auto r = ledger.Reserve(category, 10); });
    assert(ledger.Snapshot() == unchanged);
    auto r = ledger.Reserve(category, 10);
    ledger.FailAfter(LedgerStep::kConvert, 0);
    unchanged = ledger.Snapshot();
    Throws<std::bad_alloc>([&] { auto c = r.Convert(); });
    assert(ledger.Snapshot() == unchanged);
    auto c = r.Convert();  // failed conversion retains its reservation
    Throws<std::logic_error>([&] { auto again = r.Convert(); });
  }
  auto pool = ledger.Reserve(Free, 20).Convert();
  ledger.FailAfter(LedgerStep::kReserve, 0);
  unchanged = ledger.Snapshot();
  Throws<std::bad_alloc>([&] { auto r = pool.ReserveSpill(); });
  assert(ledger.Snapshot() == unchanged);
  auto spill = pool.ReserveSpill();
  ledger.FailAfter(LedgerStep::kConvert, 0);
  unchanged = ledger.Snapshot();
  Throws<std::bad_alloc>([&] { auto c = spill.Convert(); });
  assert(ledger.Snapshot() == unchanged);
  auto chunk = spill.Convert();
  ledger.FailAfter(LedgerStep::kPin, 0);
  unchanged = ledger.Snapshot();
  Throws<std::bad_alloc>([&] { auto p = chunk.PinPersistence(); });
  assert(ledger.Snapshot() == unchanged);
  chunk = {};
  pool = {};
  Empty(ledger);

  ResourceLedger huge({std::numeric_limits<std::size_t>::max(),
                       std::numeric_limits<std::size_t>::max(),
                       std::numeric_limits<std::size_t>::max(), 0});
  auto all = huge.Reserve(Private, std::numeric_limits<std::size_t>::max());
  unchanged = huge.Snapshot();
  Throws<std::bad_alloc>([&] { auto overflow = huge.Reserve(Staging, 1); });
  assert(huge.Snapshot() == unchanged);
  for (auto invalid : {Spill, Chunk, static_cast<ResourceCategory>(255)})
    Throws<std::invalid_argument>([&] { auto r = ledger.Reserve(invalid, 1); });
  Throws<std::invalid_argument>([&] { auto r = ledger.Reserve(Private, 0); });

  ResourceLedger partitions({1000, 80, 30, 20});
  auto ram = partitions.Reserve(Metadata, 80).Convert();
  auto transfer = partitions.Reserve(Staging, 30).Convert();
  unchanged = partitions.Snapshot();
  Throws<std::bad_alloc>([&] { auto r = partitions.Reserve(Tail, 1); });
  Throws<std::bad_alloc>([&] { auto r = partitions.Reserve(Staging, 1); });
  assert(partitions.Snapshot() == unchanged);
  ram = {};
  auto small = partitions.Reserve(Tail, 20).Convert();
  auto pin = small.PinPersistence();
  auto same_allocation = small.PinPersistence();
  auto extra = partitions.Reserve(Private, 1).Convert();
  unchanged = partitions.Snapshot();
  Throws<std::bad_alloc>([&] { auto p = extra.PinPersistence(); });
  assert(partitions.Snapshot() == unchanged);
  assert(unchanged.peak_ram_bytes == 80);
  assert(unchanged.peak_bytes[Index(Staging)] == 30);
  assert(unchanged.peak_persistence_pinned_bytes == 20);

  ledger.FailAfter(LedgerStep::kReserve, 1);
  auto first = ledger.Reserve(Metadata, 1);
  Throws<std::logic_error>([&] { auto p = first.PinPersistence(); });
  unchanged = ledger.Snapshot();
  Throws<std::bad_alloc>([&] { auto r = ledger.Reserve(Metadata, 1); });
  assert(ledger.Snapshot() == unchanged);
  auto moved = std::move(first);
  assert(!first);
  auto replacement = ledger.Reserve(Tail, 1);
  replacement = std::move(moved);  // releases replaced reservation
  assert(ledger.Snapshot().bytes[Index(Tail)] == 0);
  ledger.FailAfter(LedgerStep::kConvert, 1);
  auto charge = replacement.Convert();
  auto retry = ledger.Reserve(Metadata, 1);
  unchanged = ledger.Snapshot();
  Throws<std::bad_alloc>([&] { auto c = retry.Convert(); });
  assert(ledger.Snapshot() == unchanged);
  auto succeeds = retry.Convert();
  auto other = ledger.Reserve(Tail, 1).Convert();
  charge = std::move(other);  // releases replaced ownership
  assert(ledger.Snapshot().bytes[Index(Metadata)] == 1);
  Throws<std::logic_error>([&] { auto r = ResourceCharge{}.ReserveSpill(); });
  Throws<std::logic_error>([&] { auto p = ResourceCharge{}.PinPersistence(); });
  Throws<std::logic_error>([&] { auto r = charge.ReserveSpill(); });
  Throws<std::invalid_argument>(
      [&] { ledger.FailAfter(LedgerStep::kRelease, 0); });
}

// Independent model: each record represents one physical allocation, regardless
// of copied handles. Pin references overlay it; spill states partition backing.
struct Record {
  ResourceCategory category;
  std::size_t bytes;
  ResourceReservation reservation;
  std::vector<ResourceCharge> owners;
  std::vector<PersistencePin> pins;
  ResourceCharge pool;
  bool reserved{true};
};
void RandomSequences() {
  for (unsigned seed = 0; seed != 64; ++seed) {
    std::mt19937 random(seed);
    ResourceLedger ledger({1024, 900, 200, 400});
    std::vector<Record> records;
    std::size_t peak = 0;
    auto check = [&] {
      std::array<std::size_t, kResourceCategoryCount> bytes{}, reserved{};
      std::size_t pins = 0, total = 0;
      for (const auto& r : records) {
        bytes[Index(r.category)] += r.bytes;
        if (r.reserved)
          reserved[Index(r.category)] += r.bytes;
        if (!r.pins.empty())
          pins += r.bytes;
        total += r.bytes;
      }
      peak = std::max(peak, total);
      const auto actual = ledger.Snapshot();
      assert(actual.bytes == bytes);
      assert(actual.reserved_bytes == reserved);
      assert(actual.persistence_pinned_bytes == pins);
      assert(actual.total_bytes == total && total <= 1024);
      assert(actual.ram_bytes == total - bytes[Index(Staging)]);
      assert(actual.ram_bytes <= 900 && bytes[Index(Staging)] <= 200);
      assert(pins <= 400 && actual.peak_total_bytes == peak);
    };
    for (unsigned step = 0; step != 2000; ++step) {
      if (records.empty() || random() % 4 == 0) {
        const std::array categories{Free, Private, Tail, Metadata, Staging};
        const auto category = categories[random() % categories.size()];
        const std::size_t size = 1 + random() % 100;
        const auto before = ledger.Snapshot();
        if (random() % 5 == 0)
          ledger.FailAfter(LedgerStep::kReserve, 0);
        try {
          auto r = ledger.Reserve(category, size);
          records.push_back({category, size, std::move(r), {}, {}, {}, true});
        } catch (const std::bad_alloc&) {
          assert(ledger.Snapshot() == before);
        }
        ledger.ClearFaults();
      } else {
        auto& r = records[random() % records.size()];
        const auto before = ledger.Snapshot();
        const auto operation = random() % 7;
        try {
          if (operation == 0 && r.reservation) {
            if (random() % 4 == 0)
              ledger.FailAfter(LedgerStep::kConvert, 0);
            auto c = r.reservation.Convert();
            r.reserved = false;
            if (r.category == Free)
              r.pool = std::move(c);
            else {
              r.owners.push_back(std::move(c));
              if (r.category == Spill)
                r.category = Chunk;
            }
          } else if (operation == 1 && r.category == Free && !r.reserved) {
            if (random() % 4 == 0)
              ledger.FailAfter(LedgerStep::kReserve, 0);
            r.reservation = r.pool.ReserveSpill();
            r.category = Spill;
            r.reserved = true;
          } else if (operation == 2 && !r.owners.empty()) {
            r.owners.push_back(r.owners.front());
          } else if (operation == 3 &&
                     (!r.owners.empty() ||
                      (r.category == Spill && r.reservation))) {
            if (random() % 4 == 0)
              ledger.FailAfter(LedgerStep::kPin, 0);
            r.pins.push_back(r.reservation ? r.reservation.PinPersistence()
                                           : r.owners.front().PinPersistence());
          } else if (operation == 4 && !r.pins.empty()) {
            if (random() % 2)
              r.pins.push_back(r.pins.front());
            else
              r.pins.pop_back();
          } else if (operation == 5) {
            if (r.reservation) {
              r.reservation = {};
              if (r.pins.empty())
                r.reserved = false;
            } else if (!r.owners.empty())
              r.owners.pop_back();
          } else if (operation == 6) {
            r.pins.clear();
            r.owners.clear();
            r.reservation = {};
            r.pool = {};
            r.reserved = false;
          }
        } catch (const std::bad_alloc&) {
          assert(ledger.Snapshot() == before);
        }
        ledger.ClearFaults();
        if (!r.reservation && r.owners.empty() && r.pins.empty()) {
          r.reserved = false;
          if (r.pool)
            r.category = Free;
          else
            records.erase(records.begin() + (&r - records.data()));
        }
      }
      check();
    }
    records.clear();
    Empty(ledger);
  }
}
void ContentionAndLifetime() {
  ResourceLedger ledger({4096, 4096, 0, 4096});
  auto common = ledger.Reserve(Private, 256).Convert();
  std::barrier start(8);
  std::vector<std::thread> threads;
  for (unsigned t = 0; t != 8; ++t)
    threads.emplace_back([&, common] {
      start.arrive_and_wait();
      for (unsigned i = 0; i != 4000; ++i) {
        auto reservation = ledger.Reserve(Metadata, 64);
        auto charge = reservation.Convert();
        auto reference = charge;
        auto pin = common.PinPersistence();
        assert(ledger.Snapshot().total_bytes <= 4096);
      }
    });
  for (auto& thread : threads)
    thread.join();
  assert(ledger.Snapshot().total_bytes == 256);
  common = {};
  Empty(ledger);
  PersistencePin survivor;
  {
    ResourceLedger short_lived({64, 64, 0, 64});
    survivor = short_lived.Reserve(Private, 64).Convert().PinPersistence();
  }
  survivor = {};  // shared ledger state outlives its public facade
}
void AdmissionRace() {
  for (bool spill : {false, true}) {
    ResourceLedger ledger({64, 64, 0, 0});
    ResourceCharge pool;
    if (spill)
      pool = ledger.Reserve(Free, 64).Convert();
    std::barrier start(8), admitted(8);
    std::atomic<unsigned> successes{0};
    std::vector<std::thread> threads;
    for (unsigned t = 0; t != 8; ++t)
      threads.emplace_back([&] {
        start.arrive_and_wait();
        ResourceReservation reservation;
        try {
          reservation =
              spill ? pool.ReserveSpill() : ledger.Reserve(Private, 64);
          ++successes;
        } catch (const std::bad_alloc&) {
        }
        admitted.arrive_and_wait();
        assert(successes == 1);
        const auto snapshot = ledger.Snapshot();
        assert(snapshot.total_bytes == 64);
        assert(snapshot.bytes[Index(spill ? Spill : Private)] == 64);
        admitted
            .arrive_and_wait();  // keep the winner alive through every check
      });
    for (auto& thread : threads)
      thread.join();
    assert(ledger.Snapshot().bytes[Index(Free)] == (spill ? 64 : 0));
    pool = {};
    Empty(ledger);
  }
}
}  // namespace

int main() {
  BackingAndPins();
  LimitsAndFaults();
  RandomSequences();
  ContentionAndLifetime();
  AdmissionRace();
}
