#ifndef GUFO_MEM_REMAP_BARRIER_HPP_
#define GUFO_MEM_REMAP_BARRIER_HPP_

#include <chrono>
#include <cstdint>

namespace gufo::mem {

/// When a remapped device address reaches its new memory.
///
/// hipMemUnmap followed by hipMemMap with another handle returns at once,
/// but the device keeps the translation it cached for an address it has
/// used: kernels, runtime copies and copy-engine copies through that address
/// go on reading and writing the OLD page, with no error. The cached
/// translation goes only when the process's device work has been idle for a
/// moment; nothing else drops it (measured on gfx1151, ROCm 10.1, Linux
/// 7.3.3 with the MES scheduler, November 2026: idle 150 us, every address
/// new in 3,000 remaps; no idle, 15% new; another thread running kernels
/// back to back, still old after 200 ms. A second unmap, an access change,
/// hipDeviceSynchronize, or freeing and reserving the address change
/// nothing. See docs/internal/state-memory-shared-pages.md).
///
/// So an address that lost a mapping is *dirty* until the device has been
/// idle, and mapping memory at a dirty address needs Settle() before the
/// device uses the address again. An address keeps the stamp of the moment
/// it lost its mapping; a Settle() that began after that moment makes it
/// clean, and a clean address can be mapped and used at once, as one that
/// was never mapped can.
class RemapBarrier {
public:
  /// The stamp for an address that loses its mapping now; never 0.
  [[nodiscard]] static std::uint64_t Stamp() noexcept;
  /// Whether an address with this stamp (0: never mapped) can be mapped and
  /// used without a Settle().
  [[nodiscard]] static bool Clean(std::uint64_t stamp) noexcept;
  /// Memory was mapped at an address that is not clean: Settle() must run
  /// before the device uses the address.
  static void NoteRemap() noexcept;
  /// Whether a NoteRemap() still waits for its Settle().
  [[nodiscard]] static bool Pending() noexcept;
  /// Waits for the device's work and keeps the device idle for kIdleMicros:
  /// every address stamped before the call is clean afterwards, and every
  /// NoteRemap() before it is answered. Call it on the thread that runs the
  /// model, between its launches: nothing it launched may be relied on to be
  /// still running. Threads that submit device work on their own bracket it
  /// with Enter() and Leave(); Settle() waits for them and holds new ones back,
  /// for a bounded time. Device work this cannot see (another model's thread in
  /// the same process) is detected only when it is still running at the end
  /// of the idle time.
  static void Settle() noexcept;
  /// Before a batch of unmaps on the thread that will Settle(): holds other
  /// threads' device work back from now on, after a short wait for what
  /// they have in flight. An unmap waits for device work in flight, and a
  /// disk write's copies running beside a request that replaced one page
  /// per buffer stalled it for 290 ms. Settle() or Release() on this thread
  /// ends the hold; without either it ends after kHoldLimit.
  static void Hold() noexcept;
  static void Release() noexcept;
  /// Around device work a thread submits and waits for on its own (a disk
  /// write's staging copy, a restore's copies beside decoders, a page clear
  /// on the page pool's thread). Enter() waits while a Settle() runs.
  static void Enter() noexcept;
  static void Leave() noexcept;

  struct Counts {
    /// Settle() calls, those that gave up waiting for a quiet device, and
    /// the time all of them took.
    std::uint64_t settles{0};
    std::uint64_t unquiet{0};
    std::uint64_t micros{0};
  };
  [[nodiscard]] static Counts Stats() noexcept;

  /// The device backend installs how to wait for all device work
  /// (hipDeviceSynchronize). Without one Settle() only keeps the idle time:
  /// HIP-free, like the rest of this library.
  static void SetDeviceDrain(void (*drain)() noexcept) noexcept;

  /// Idle time that drops the cached translations: the largest time any
  /// remap needed was under 200 us (copy engine: 2 of 30 old at 100 us, 0 of
  /// 400 at 200 us).
  static constexpr int kIdleMicros = 400;
  /// The longest a Hold() or a Settle() keeps other threads waiting.
  static constexpr std::chrono::milliseconds kHoldLimit{25};
};

/// Enter()/Leave() for a scope.
class DeviceWorkScope {
public:
  DeviceWorkScope() noexcept { RemapBarrier::Enter(); }
  ~DeviceWorkScope() { RemapBarrier::Leave(); }
  DeviceWorkScope(const DeviceWorkScope&) = delete;
  DeviceWorkScope& operator=(const DeviceWorkScope&) = delete;
};

}  // namespace gufo::mem

#endif  // GUFO_MEM_REMAP_BARRIER_HPP_
