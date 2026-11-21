#include "src/mem/remap_barrier.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace gufo::mem {
namespace {

using Clock = std::chrono::steady_clock;

struct State {
  // Stamps handed out now, and the last stamp a finished Settle() covers.
  std::atomic<std::uint64_t> epoch{1};
  std::atomic<std::uint64_t> clean{0};
  // NoteRemap() calls, and how many of them a finished Settle() answered.
  std::atomic<std::uint64_t> notes{0};
  std::atomic<std::uint64_t> answered{0};
  // Device work other threads bracket: scopes open now, and scopes ever
  // opened (to tell whether one opened during an idle wait).
  std::atomic<std::int64_t> open{0};
  std::atomic<std::uint64_t> entered{0};
  std::atomic<bool> settling{false};
  // When the gate opens whatever happens, in steady-clock nanoseconds: a
  // Hold() nobody follows with a Settle() must not stop other threads.
  std::atomic<std::int64_t> gate_until{0};
  std::atomic<void (*)() noexcept> drain{nullptr};
  std::atomic<std::uint64_t> settles{0}, unquiet{0}, micros{0};
  std::atomic<std::thread::id> settler{};
  // One Settle() at a time; `gate` wakes the threads Enter() held back.
  std::mutex settle_mutex;
  std::mutex gate_mutex;
  std::condition_variable gate;
};

State& Shared() {
  // Never destroyed: regions and leases may still call during exit.
  static State* const state = new State();
  return *state;
}

std::int64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             Clock::now().time_since_epoch())
      .count();
}

// Whether another thread holds the gate against this one now.
bool Gated(const State& s) {
  return s.settling.load(std::memory_order_acquire) &&
         s.settler.load(std::memory_order_relaxed) !=
             std::this_thread::get_id() &&
         NowNs() < s.gate_until.load(std::memory_order_acquire);
}

void WaitForOthers(const State& s, std::chrono::milliseconds patience) {
  const auto end = Clock::now() + patience;
  while (s.open.load(std::memory_order_acquire) > 0 && Clock::now() < end)
    std::this_thread::yield();
}

void SpinFor(std::chrono::microseconds time) {
  // A sleep would round up by the timer slack, and a yield is enough to let
  // the threads that were held back reach their wait.
  const auto end = Clock::now() + time;
  while (Clock::now() < end)
    std::this_thread::yield();
}

}  // namespace

std::uint64_t RemapBarrier::Stamp() noexcept {
  return Shared().epoch.load(std::memory_order_acquire);
}

bool RemapBarrier::Clean(std::uint64_t stamp) noexcept {
  return stamp <= Shared().clean.load(std::memory_order_acquire);
}

void RemapBarrier::NoteRemap() noexcept {
  Shared().notes.fetch_add(1, std::memory_order_acq_rel);
}

bool RemapBarrier::Pending() noexcept {
  const auto& s = Shared();
  return s.notes.load(std::memory_order_acquire) >
         s.answered.load(std::memory_order_acquire);
}

void RemapBarrier::Enter() noexcept {
  auto& s = Shared();
  for (;;) {
    if (Gated(s)) {
      std::unique_lock lock(s.gate_mutex);
      const auto until = Clock::time_point(std::chrono::nanoseconds(
          s.gate_until.load(std::memory_order_acquire)));
      s.gate.wait_until(lock, until, [&] {
        return !s.settling.load(std::memory_order_acquire);
      });
    }
    s.open.fetch_add(1, std::memory_order_acq_rel);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    // A Hold() or Settle() that began between the check and the count has
    // not seen this scope: step back and wait for it.
    if (!Gated(s))
      break;
    s.open.fetch_sub(1, std::memory_order_acq_rel);
  }
  s.entered.fetch_add(1, std::memory_order_acq_rel);
}

void RemapBarrier::Hold() noexcept {
  auto& s = Shared();
  const auto me = std::this_thread::get_id();
  s.gate_until.store(NowNs() + std::chrono::nanoseconds(kHoldLimit).count(),
                     std::memory_order_release);
  if (s.settling.load(std::memory_order_acquire) &&
      s.settler.load(std::memory_order_relaxed) == me)
    return;
  s.settler.store(me, std::memory_order_relaxed);
  s.settling.store(true, std::memory_order_release);
  // Pairs with the fence in Enter(): either this thread sees the scope's
  // count or that thread sees the gate. Release/acquire alone lets both
  // loads run before the other thread's store is visible.
  std::atomic_thread_fence(std::memory_order_seq_cst);
  // Their work in flight ends within a piece; an unmap behind it would
  // wait for all of it.
  WaitForOthers(s, std::chrono::milliseconds(2));
}

void RemapBarrier::Release() noexcept {
  auto& s = Shared();
  if (s.settler.load(std::memory_order_relaxed) != std::this_thread::get_id())
    return;
  {
    const std::lock_guard lock(s.gate_mutex);
    s.settling.store(false, std::memory_order_release);
  }
  s.gate.notify_all();
}

void RemapBarrier::Leave() noexcept {
  Shared().open.fetch_sub(1, std::memory_order_acq_rel);
}

void RemapBarrier::SetDeviceDrain(void (*drain)() noexcept) noexcept {
  Shared().drain.store(drain, std::memory_order_release);
}

RemapBarrier::Counts RemapBarrier::Stats() noexcept {
  const auto& s = Shared();
  return {s.settles.load(std::memory_order_relaxed),
          s.unquiet.load(std::memory_order_relaxed),
          s.micros.load(std::memory_order_relaxed)};
}

void RemapBarrier::Settle() noexcept {
  using std::chrono::microseconds;
  auto& s = Shared();
  const std::lock_guard settle_lock(s.settle_mutex);
  s.gate_until.store(NowNs() + std::chrono::nanoseconds(kHoldLimit).count(),
                     std::memory_order_release);
  s.settler.store(std::this_thread::get_id(), std::memory_order_relaxed);
  s.settling.store(true, std::memory_order_release);
  // Pairs with the fence in Enter(): either this thread sees the scope's
  // count or that thread sees the gate. Release/acquire alone lets both
  // loads run before the other thread's store is visible.
  std::atomic_thread_fence(std::memory_order_seq_cst);
  // What this call covers: addresses stamped and remaps noted before it. A
  // thread that unmaps or maps during the idle time gets the next stamp and
  // waits for the next call.
  const auto covered = s.epoch.fetch_add(1, std::memory_order_acq_rel);
  const auto noted = s.notes.load(std::memory_order_acquire);
  // Bounded: a scope this thread opened itself, or one that never closes,
  // must not hang the model. The drain below still covers their kernels.
  const auto start = Clock::now();
  const auto deadline = start + std::chrono::milliseconds(20);
  bool quiet = false;
  for (;;) {
    WaitForOthers(s, std::chrono::milliseconds(2));
    const auto mark = s.entered.load(std::memory_order_acquire);
    const auto drain = s.drain.load(std::memory_order_acquire);
    if (drain)
      drain();
    SpinFor(microseconds(kIdleMicros));
    // Work nobody bracketed shows only while it runs: a wait that does not
    // return at once found some.
    const auto before = Clock::now();
    if (drain)
      drain();
    quiet = Clock::now() - before < microseconds(100) &&
            s.entered.load(std::memory_order_acquire) == mark &&
            s.open.load(std::memory_order_acquire) <= 0;
    if (quiet || Clock::now() >= deadline)
      break;
  }
  s.settles.fetch_add(1, std::memory_order_relaxed);
  if (!quiet)
    s.unquiet.fetch_add(1, std::memory_order_relaxed);
  s.micros.fetch_add(
      static_cast<std::uint64_t>(
          std::chrono::duration_cast<microseconds>(Clock::now() - start)
              .count()),
      std::memory_order_relaxed);
  s.clean.store(covered, std::memory_order_release);
  s.answered.store(noted, std::memory_order_release);
  {
    const std::lock_guard lock(s.gate_mutex);
    s.settling.store(false, std::memory_order_release);
  }
  s.gate.notify_all();
}

}  // namespace gufo::mem
