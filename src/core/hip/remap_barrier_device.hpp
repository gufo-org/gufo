#ifndef GUFO_CORE_HIP_REMAP_BARRIER_DEVICE_HPP_
#define GUFO_CORE_HIP_REMAP_BARRIER_DEVICE_HPP_

namespace gufo::mem {

/// Lets mem::RemapBarrier wait for the whole device (hipDeviceSynchronize)
/// and flush its translations (FlushTranslations()) before a remapped
/// address is used. The page pool and the expert stream store install it
/// for the process.
void InstallRemapBarrierDevice() noexcept;

/// Makes the device drop the address translations it cached for this
/// process: the kernel driver ioctls behind hipFree and behind hipMalloc of
/// a 4 MiB block each flush the process's GPU TLB, whatever other processes
/// run. A 64 KiB block comes from the HIP runtime's sub-allocator, never
/// reaches the driver and flushes nothing (gfx1151, ROCm 10.1).
///
/// It frees the block the previous call allocated, then allocates the next
/// one, so a flush does not depend on memory being free at that moment. It
/// fails when the held block cannot be freed (it stays held, and nothing is
/// allocated), or when it holds none and cannot allocate one. The process
/// keeps one 4 MiB block.
///
/// hipMemUnmap alone does not: see mem::RemapBarrier for the driver gap and
/// docs/internal/state-memory-shared-pages.md for the measurements.
///
/// Call it outside any capture on this thread, as hipDeviceSynchronize:
/// hipFree synchronizes the device. False when neither the free nor the
/// allocation succeeded. True does not prove the flush, which depends on how
/// the runtime serves the block; mem_remap_barrier_device_test checks it.
bool FlushTranslations() noexcept;

}  // namespace gufo::mem

#endif  // GUFO_CORE_HIP_REMAP_BARRIER_DEVICE_HPP_
