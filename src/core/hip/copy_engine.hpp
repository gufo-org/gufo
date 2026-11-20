#ifndef GUFO_CORE_HIP_COPY_ENGINE_HPP_
#define GUFO_CORE_HIP_COPY_ENGINE_HPP_

#include <cstddef>
#include <cstdint>

namespace gufo::hip {

// Copies into device memory on one of the GPU's SDMA copy engines (HSA async
// copy), outside every HIP stream and compute queue. Beside running decode
// graphs, a copy kernel or copy on a stream that shares a hardware queue with
// them completes only after the whole graph, and one on a queue of its own
// costs a compute queue and doubles decode step time while it runs; the
// engine does neither (2.2 GB at 57-70 GB/s beside decode, steps unchanged).
// It reads and writes by GPU virtual address, so both sides must be device
// memory or host memory the device maps at its own address. A copy is not
// ordered against any stream: the caller makes sure nothing else writes its
// destination or source until its fence clears.
class CopyEngine {
public:
  // Counts copies in flight; one per lease, reused once it clears.
  using Fence = std::uint64_t;
  virtual ~CopyEngine() = default;
  // Throws when the runtime cannot create one.
  [[nodiscard]] virtual Fence CreateFence() = 0;
  virtual void DestroyFence(Fence) noexcept = 0;
  // Queues a copy that `fence` counts until the engine finishes it. False
  // when the runtime refused it; nothing was queued then.
  [[nodiscard]] virtual bool Copy(void* destination, const void* source,
                                  std::size_t bytes, Fence) noexcept = 0;
  // Whether every copy `fence` counts has finished.
  [[nodiscard]] virtual bool Done(Fence) const noexcept = 0;
  // Waits until it is, sleeping rather than spinning.
  virtual void Wait(Fence) noexcept = 0;
};

// The current HIP device's copy engine, created on first use; null where the
// platform or runtime has none (Windows, a runtime without HSA, no available
// SDMA engine). Lives for the process.
[[nodiscard]] CopyEngine* DeviceCopyEngine() noexcept;

}  // namespace gufo::hip
#endif
