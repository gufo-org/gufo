#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#include "src/core/hip/committed_backing.hpp"
#include "src/core/hip/transfer_pool.hpp"

using namespace gufo;
using namespace gufo::cache;
using Clock = std::chrono::steady_clock;
namespace {
void Check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}
void Require(bool ok) {
  if (!ok)
    throw std::runtime_error("idle spill benchmark verification failed");
}
class ByteAdapter final : public Adapter {
  struct Execution final : Slot {
    MutationGuard& guard;
    Rows rows{};
    explicit Execution(MutationGuard& g) : guard(g) {}
    ~Execution() override { guard.BeforeRelease(ComponentId{1}, 0, rows); }
    bool IsValid() const noexcept override { return true; }
  };
  void* source{};
  std::size_t capacity;
  ComponentDescriptor component{
      ComponentId{1}, 1, ComponentKind::kAppendRows, 1, 128ULL << 20, 0};

public:
  explicit ByteAdapter(std::size_t bytes) : capacity(bytes) {
    Check(hipMalloc(&source, bytes));
    Check(hipMemset(source, 0x5a, bytes));
  }
  ~ByteAdapter() override { Check(hipFree(source)); }
  void Prepare(Slot& s) { static_cast<Execution&>(s).rows = capacity; }
  Capabilities GetCapabilities() const override { return {true, false}; }
  std::span<const ComponentDescriptor> Components() const override {
    return {&component, 1};
  }
  Identity CompatibilityIdentity() const override { return {}; }
  std::unique_ptr<Slot> CreateSlot(MutationGuard& g) override {
    return std::make_unique<Execution>(g);
  }
  std::vector<ComponentPosition> Positions(const Slot& s) const override {
    return {{ComponentId{1}, static_cast<const Execution&>(s).rows}};
  }
  std::vector<Rows> PlanPrefill(const Slot&, Rows, Rows end) const override {
    return {end};
  }
  void BeginRestore(Slot&, std::span<const ComponentPosition>) override {
    throw std::logic_error("unused benchmark restore");
  }
  Completion CapturePrivate(const Slot&, ComponentId, std::span<std::byte>,
                            Stream&) override {
    throw std::logic_error("unused benchmark private state");
  }
  Completion CopyRowsOut(const Slot& s, ComponentId c, Rows first, Rows end,
                         std::span<std::byte> bytes, Stream& stream) override {
    Require(c == ComponentId{1} && first < end && bytes.size() == end - first &&
            end <= static_cast<const Execution&>(s).rows);
    return dynamic_cast<hip::TransferStream&>(stream).Copy(
        bytes.data(), static_cast<std::byte*>(source) + first, bytes.size(),
        hip::CopyDirection::kDeviceToHost, bytes.size());
  }
  Completion CopyRowsIn(Slot&, ComponentId, Rows, Rows,
                        std::span<const std::byte>, Stream&) override {
    throw std::logic_error("unused benchmark load");
  }
  Completion LoadPrivate(Slot&, ComponentId, std::span<const std::byte>,
                         Stream&) override {
    throw std::logic_error("unused benchmark private load");
  }
  bool Validate(Slot&, std::span<const ComponentPosition>) override {
    return true;
  }
  bool Invalidate(Slot& s) noexcept override {
    auto& execution = static_cast<Execution&>(s);
    execution.guard.BeforeRelease(ComponentId{1}, 0, execution.rows);
    execution.rows = 0;
    execution.guard.AfterReset();
    return true;
  }
};
void Run(std::size_t capacity) {
  constexpr std::size_t block = 128ULL << 20, piece = 4ULL << 20;
  constexpr std::size_t metadata = 1ULL << 20;
  ResourceLedger ledger(
      {capacity + 2 * metadata, capacity + 2 * metadata, 0, capacity});
  hip::CommittedBackingPool backing(ledger,
                                    {capacity + metadata, block, metadata});
  hip::TransferPool streams(1);
  ByteAdapter adapter(capacity);
  for (const char* mode : {"idle", "zero-idle", "mid-spill"}) {
    for (int iteration = -1; iteration < 5; ++iteration) {
      std::atomic<bool> idle{false};
      LeasedSlot* facade{};
      // Stop admitting pieces after half the bytes for the controlled mid-spill
      // history. The worker and foreground still use the same production guard.
      IdleSpillConfig config{
          piece,
          [&]() -> std::unique_ptr<Stream> { return streams.TryAcquire(); },
          [&] {
            return idle.load() && (std::strcmp(mode, "mid-spill") != 0 ||
                                   facade->Metrics().idle_bytes < capacity / 2);
          }};
      // Configured preservation always leases a fresh stream; the legacy
      // foreground stream is unused. Provide a CPU dummy without leasing a
      // scarce HIP pair for the slot's lifetime.
      class UnusedStream final : public Stream {
        TransferResult Synchronize() noexcept override {
          return TransferResult::kSucceeded;
        }
      } foreground;
      LeasedSlot slot(ledger, adapter, foreground, SlotId{1},
                      std::move(config));
      facade = &slot;
      auto lease = slot.Acquire();
      adapter.Prepare(lease.Execution());
      std::vector<std::shared_ptr<BorrowedRows>> rows;
      for (std::size_t offset = 0; offset < capacity; offset += block) {
        auto held = backing.TryAcquire(ResourceCategory::kBackingAssigned);
        Require(bool(held));
        rows.push_back(std::move(*held).Borrow(lease, ComponentId{1}, offset,
                                               offset + block, block));
      }
      const auto initial = slot.Metrics();
      const auto start = Clock::now();
      lease.Commit();
      if (std::strcmp(mode, "zero-idle") != 0) {
        idle = true;
        const auto target =
            std::strcmp(mode, "idle") == 0 ? capacity : capacity / 2;
        while (slot.Metrics().idle_bytes < target) {
          Require(slot.Metrics().idle_failures == 0);
          std::this_thread::yield();
        }
      }
      idle = false;
      const auto reassignment = Clock::now();
      lease = slot.Acquire();
      const auto end = Clock::now();
      const auto metrics = slot.Metrics();
      Require(metrics.idle_bytes + metrics.foreground_bytes == capacity);
      Require(backing.PageCommitAllocations() == 1);
      // Full byte verification outside timing, each iteration, each chunk.
      for (const auto& r : rows) {
        Require(r->IsValid() && !r->Location());
        auto pin = r->Pin();
        const auto* bytes = static_cast<const std::byte*>(pin.Owner().get());
        Require(std::all_of(bytes, bytes + block,
                            [](std::byte b) { return b == std::byte{0x5a}; }));
      }
      if (iteration >= 0)
        std::printf(
            "sample,%zu,%s,%d,%.6f,%.6f,%zu,%zu,%.6f,%.6f,%.6f,%.6f\n",
            capacity, mode, iteration,
            std::chrono::duration<double, std::milli>(end - start).count(),
            std::chrono::duration<double, std::milli>(end - reassignment)
                .count(),
            metrics.idle_bytes, metrics.foreground_bytes,
            metrics.idle_copy_ns / 1e6, metrics.foreground_copy_ns / 1e6,
            (metrics.residual_wait_ns - initial.residual_wait_ns) / 1e6,
            metrics.stream_wait_ns / 1e6);
    }
  }
}
}  // namespace
int main() {
  Check(hipInit(0));
  Check(hipFree(nullptr));
  hipDeviceProp_t properties{};
  Check(hipGetDeviceProperties(&properties, 0));
  int runtime{};
  Check(hipRuntimeGetVersion(&runtime));
  std::printf("device,%s,%s,hip,%d,compiler,%s\n", properties.name,
              properties.gcnArchName, runtime, __VERSION__);
  std::puts(
      "type,bytes,history,iteration,total_ms,reassignment_ms,idle_bytes,"
      "foreground_bytes,idle_copy_ms,foreground_copy_ms,residual_wait_ms,"
      "stream_wait_ms");
  Run(1ULL << 30);
  Run(6ULL << 30);
}
