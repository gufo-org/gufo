/* SSD-backed routed experts. The stage buffers preserve the original expert
 * indices, so decode, prefill and speculative verification use the same
 * kernels and arithmetic as resident weights. */
#include "src/core/hip/expert_stream/store.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "src/core/hip/managed_alloc.hpp"
#include "src/core/logging/logging.hpp"
#include "src/core/platform/fs.hpp"
#include "src/core/platform/mapped_file.hpp"
#include "src/core/platform/os_info.hpp"
#include "src/core/platform/pages.hpp"
#include "src/ext/host_budget.hpp"
#include "src/mem/ledger.hpp"
#include "src/mem/remap_barrier.hpp"

namespace gufo::expert_stream {
namespace {

constexpr uint64_t kGiB = 1ull << 30;

/* Host memory available now, as the rest of Gufo measures it. */
uint64_t AvailableMemory() {
  return gufo::ext::HostAvailableBytes();
}
/* Each slot size class grows by its own chunks, so a smaller chunk bounds
 * the capacity a class can strand in a part-used chunk. */
constexpr uint64_t kChunkBytes = 128ull << 20;
/* Cache chunks are physical device pages (hipMemCreate), the pages Gufo's
 * PagePool hands out: the host cannot swap them, and copies into them run at
 * device speed (75.6 GB/s from pinned staging, against 23.7 into managed
 * memory; kernel reads are the same). Each size class maps its pages at the
 * end of one reserved address range, so its slots run on across page and
 * chunk boundaries and only the class's last slot can leave a page part
 * used. A chunk is a run of pages and owns the slots that end in it.
 * Where the device heap is smaller than host memory, or the device has no
 * virtual memory management, the whole cache is managed memory instead,
 * one allocation per chunk. */
constexpr uint64_t kPageBytes = 2ull << 20;

struct Chunk {
  uint8_t* base = nullptr;
  uint64_t bytes = 0;                                  // mapped (or allocated)
  std::vector<hipMemGenericAllocationHandle_t> pages;  // empty: managed
  uint32_t size_class = 0;
  uint64_t class_used = 0;  // the class's slot bytes before this chunk's
};

/* A size class's reserved range: `mapped` bytes of pages from `base`, the
 * first `used` of them in slots. */
struct ClassRange {
  uint8_t* base = nullptr;
  uint64_t reserved = 0, mapped = 0, used = 0;
  /* The most the range has had mapped, and the RemapBarrier stamp of the
   * last time it lost pages: mapping below `high_water` reuses addresses
   * the device may still translate to the freed pages. */
  uint64_t high_water = 0;
  uint64_t released_at = 0;
};

hipMemAllocationProp DeviceProp(int device) {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  return prop;
}

bool DevicePagesAvailable(int device) {
  if (!platform::DevicePagesUseHostMemory())
    return false;
  int supported = 0;
  if (hipDeviceGetAttribute(&supported,
                            hipDeviceAttributeVirtualMemoryManagementSupported,
                            device) != hipSuccess ||
      !supported) {
    (void)hipGetLastError();
    return false;
  }
  const hipMemAllocationProp prop = DeviceProp(device);
  size_t granularity = 0;
  if (hipMemGetAllocationGranularity(&granularity, &prop,
                                     hipMemAllocationGranularityMinimum) !=
      hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  if (!granularity || kPageBytes % granularity != 0)
    return false;
  // As Gufo's PagePool lends: only where the heap covers host memory, so the
  // whole cache fits in one kind of memory.
  size_t heap_free = 0, heap_total = 0;
  const auto host = platform::QueryHostMemory();
  if (hipMemGetInfo(&heap_free, &heap_total) != hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  return mem::HeapCoversHost(heap_total, host ? host->total_bytes : 0);
}

/* Unmaps and releases `count` pages mapped from `at`. */
void UnmapPages(uint8_t* at, const hipMemGenericAllocationHandle_t* pages,
                size_t count) {
  if (count)
    (void)hipMemUnmap(at, count * kPageBytes);
  for (size_t i = 0; i < count; ++i)
    (void)hipMemRelease(pages[i]);
  (void)hipGetLastError();
}

/* Maps `bytes` (a multiple of kPageBytes) of new device pages at the end of
 * `range`, reserving the range first; false, with nothing held, when any
 * step fails. Pages mapped where the range lost pages before are reached
 * only after RemapBarrier::Settle, which this runs when needed: call it on
 * the thread that runs the model, between its launches. */
bool MapClassPages(int device, ClassRange& range, uint64_t bytes,
                   Chunk& chunk) {
  const hipMemAllocationProp prop = DeviceProp(device);
  if (!range.base) {
    const auto host = platform::QueryHostMemory();
    const uint64_t reserved =
        ((host ? host->total_bytes : 0) + kPageBytes - 1) / kPageBytes *
        kPageBytes;
    void* base = nullptr;
    if (!reserved || hipMemAddressReserve(&base, reserved, kPageBytes, nullptr,
                                          0) != hipSuccess) {
      (void)hipGetLastError();
      return false;
    }
    range.base = static_cast<uint8_t*>(base);
    range.reserved = reserved;
  }
  if (range.mapped + bytes > range.reserved)
    return false;
  uint8_t* const at = range.base + range.mapped;
  std::vector<hipMemGenericAllocationHandle_t> pages;
  for (uint64_t offset = 0; offset < bytes; offset += kPageBytes) {
    hipMemGenericAllocationHandle_t page{};
    if (hipMemCreate(&page, kPageBytes, &prop, 0) != hipSuccess)
      break;
    if (hipMemMap(at + offset, kPageBytes, 0, page, 0) != hipSuccess) {
      (void)hipMemRelease(page);
      break;
    }
    pages.push_back(page);
  }
  hipMemAccessDesc access{};
  access.location = prop.location;
  access.flags = hipMemAccessFlagsProtReadWrite;
  if (pages.size() * kPageBytes != bytes ||
      hipMemSetAccess(at, bytes, &access, 1) != hipSuccess) {
    UnmapPages(at, pages.data(), pages.size());
    return false;
  }
  chunk.base = at;
  chunk.bytes = bytes;
  chunk.pages = std::move(pages);
  // Until the device has been idle it reaches the pages these addresses
  // had before (RemapBarrier): wait for that before any table points here.
  if (range.mapped < range.high_water &&
      !mem::RemapBarrier::Clean(range.released_at))
    mem::RemapBarrier::Settle();
  range.mapped += bytes;
  range.high_water = std::max(range.high_water, range.mapped);
  return true;
}

/* A routing trace (Policy::trace_path) starts with kTraceMagic and the
 * format version; records follow. */
constexpr char kTraceMagic[8] = {'G', 'U', 'F', 'O', 'X', 'T', 'R', 'C'};
constexpr uint32_t kTraceVersion = 2;
constexpr uint64_t kReadBytes = 32ull << 20;
constexpr uint64_t kDirectAlign = 4096;
constexpr uint32_t kReadWorkers = 8;

constexpr uint32_t kNoSlot = std::numeric_limits<uint32_t>::max();
/* Eviction candidates kept per size class: the kVictims lowest keep values
 * of the slots an expert of that class could take, rebuilt by one
 * scan when the list runs dry or ages past kVictimsTicks prepares. Pops
 * skip slots freed, reoccupied or used since the build, so the lists need
 * no invalidation and the order among untouched slots is the scan's. */
constexpr size_t kVictims = 32;
constexpr uint64_t kVictimsTicks = 512;
constexpr uint32_t kNoLayer = std::numeric_limits<uint32_t>::max();

/* A prediction window: the layer the next prepare handles (low 32 bits) and
 * the count of windows opened before it (high 32 bits). Its parity picks the
 * prefetch half its jobs write, so consecutive windows use different halves
 * whatever the layer count: where an odd number of layers wraps to layer 0,
 * layer parity would give the last layer and the first the same half. */
constexpr uint64_t window_of(uint64_t sequence, uint32_t layer) {
  return sequence << 32 | layer;
}
constexpr uint32_t window_layer(uint64_t window) {
  return static_cast<uint32_t>(window);
}
constexpr uint32_t window_half(uint64_t window) {
  return static_cast<uint32_t>(window >> 32) & 1;
}
/* The window after `window`, for `layer`. */
constexpr uint64_t window_next(uint64_t window, uint32_t layer) {
  return window_of((window >> 32) + 1, layer);
}
/* Matches no job: jobs target layers below kNoLayer. */
constexpr uint64_t kNoWindow = ~0ull;

struct Entry {
  /* Atomic: the prefetch submission thread reads the target layer's slots
   * while the evaluation thread promotes or evicts entries. */
  std::atomic<uint32_t> slot{kNoSlot};
  uint64_t last = 0;  // tick of the last use; victim lists detect uses by it
  /* Eviction by frequency class, after llama.cpp's ClassLRU: `uses` counts uses
   * (at most use_cap per prepare) decaying with a half-life of use_half_life
   * tokens of the layer's clock; at each use the expert takes class
   * floor(log2(uses)), at most max_class, and `protect` = 2^class. The
   * victim is the expert with the largest age / protect: an expert used
   * twice as often may stay unused twice as long. A prompt chunk advances
   * its layer's clock by at most clock_rows, so one long prompt does not age
   * the whole decode working set at once. Seeded experts start with
   * seed_uses of their ranking weight: enough to serve a cold prompt, low
   * enough that decode's working set displaces them (Policy). */
  uint64_t used_at = 0;  // layer clock at the last use
  double uses = 0;
  double protect = 1;
};

double protect_of(double uses, double max_class) {
  if (uses < 1.0)
    return 1.0;
  return std::exp2(std::min(max_class, std::floor(std::log2(uses))));
}

struct Slot {
  uint8_t* data = nullptr;
  uint32_t size_class = 0;  // capacity is the class's bytes
  uint32_t layer = kNoSlot;
  uint32_t expert = kNoSlot;
};

/* `file` takes buffered reads, `direct` aligned unbuffered ones (invalid
 * where the platform has none). */
struct Source {
  platform::PlatformFile file = platform::kInvalidFile;
  platform::PlatformFile direct = platform::kInvalidFile;
  uint64_t offset = 0;
  uint64_t bytes = 0;  // one expert
};

/* An expert projection's tail in the optional MODEL.gguf.expert-tails file
 * (Store::AttachTails): bytes from `cut` to the end of each
 * expert's projection, at first + expert * stride. */
struct TailSource {
  uint64_t cut = 0;
  uint64_t first = 0;
  uint64_t stride = 0;
};

struct ReadTask {
  platform::PlatformFile file;
  platform::PlatformFile direct;
  uint64_t offset;
  uint64_t bytes;
  uint8_t* destination;
  /* Prefetch bookkeeping of the owning job; null for demand reads. */
  std::atomic<uint64_t>* job_bytes = nullptr;
  std::atomic<bool>* job_failed = nullptr;
};

/* A recorded router prediction: the ids live in caller-owned pinned memory
 * and are valid until the event is waited on and the job submitted. */
struct Prediction {
  hipEvent_t event = nullptr;
  uint32_t layer = 0;
  const int32_t* ids = nullptr;
  uint32_t count = 0;
  bool wide = false;
};

void PredictWork(Store::Impl* store);

/* A background read job submitted between demand phases. Workers run its
 * tasks whenever no demand phase is active; the next prepare of the same
 * layer adopts the still-unrun tasks of the experts it routes into its
 * phase, any other prepare drops the job after the in-flight tasks drain.
 * Tasks are grouped per expert so adoption can skip unrouted experts:
 * waiting for a wrong prediction would cost the phase its whole read. */
struct PrefetchJob {
  std::vector<ReadTask> tasks;
  std::vector<uint32_t> task_begin;  // first task per expert
  std::vector<uint32_t> task_count;  // tasks per expert
  std::vector<uint32_t> row_of;      // compact stage row per expert
  std::vector<uint32_t> experts;     // experts the completed job stages
  std::atomic<uint64_t> bytes{0};
  std::atomic<bool> failed{false};
  uint32_t layer = 0;
  uint64_t window = 0;  // the prediction window it was built for
  /* A prefill-scale prediction; only a batch-scale prepare adopts it, so a
   * decode step never inherits a whole-prompt expert set. */
  bool wide = false;
  size_t taken = 0;  // tasks already handed to workers
};

/* One positional read, retried when interrupted; 0 on failure or EOF. */
size_t read_at(platform::PlatformFile file, uint8_t* dst, uint64_t bytes,
               uint64_t offset) {
  std::error_code ec;
  for (;;) {
    const size_t got =
        platform::ReadAt(file, offset, dst, static_cast<size_t>(bytes), ec);
    if (got || ec != std::errc::interrupted)
      return got;
  }
}

bool read_all(platform::PlatformFile file, uint8_t* dst, uint64_t bytes,
              uint64_t offset) {
  while (bytes) {
    const size_t got = read_at(file, dst, bytes, offset);
    if (!got)
      return false;
    dst += got;
    offset += static_cast<uint64_t>(got);
    bytes -= static_cast<uint64_t>(got);
  }
  return true;
}

/* Direct reads need aligned buffers and ranges. The last read can end at an
 * unaligned EOF, provided it contains the entire requested tensor span. */
bool read_direct(platform::PlatformFile file, uint8_t* dst, uint64_t bytes,
                 uint64_t offset, uint64_t* skip) {
  if (!platform::FileValid(file))
    return false;
  const uint64_t aligned = offset / kDirectAlign * kDirectAlign;
  *skip = offset - aligned;
  const uint64_t need = *skip + bytes;
  const uint64_t length =
      (need + kDirectAlign - 1) / kDirectAlign * kDirectAlign;
  uint64_t done = 0;
  while (done < need) {
    const size_t got = read_at(file, dst + done, length - done, aligned + done);
    if (!got)
      return false;
    done += static_cast<uint64_t>(got);
    if (done < need && static_cast<uint64_t>(got) % kDirectAlign)
      return false;
  }
  return true;
}

/* The queue is sorted before dispatch. Workers read independent ranges into
 * their own pinned buffers and upload on their own streams, so the copies
 * overlap the legacy stream's kernels instead of queueing behind them; each
 * upload is awaited before its buffer is reused. Demand phases always
 * preempt the background prefetch sweep, which resumes between phases. No
 * work outlives Execute, so stage buffers and shard descriptors stay
 * valid. */
class ReadPool {
public:
  ReadPool(const std::array<uint8_t*, kReadWorkers>& buffers, int device)
      : buffers_(buffers), device_(device) {
    bool streams_ok = true;
    for (uint32_t i = 0; i < kReadWorkers && streams_ok; ++i)
      streams_ok = hipStreamCreateWithFlags(
                       &streams_[i], hipStreamNonBlocking) == hipSuccess &&
                   hipEventCreateWithFlags(&events_[i],
                                           hipEventDisableTiming) == hipSuccess;
    if (!streams_ok) {
      ReleaseResources();
      throw std::runtime_error("expert read pool streams");
    }
    try {
      for (uint32_t i = 0; i < kReadWorkers; ++i)
        threads_.emplace_back([this, i] { Work(i); });
    } catch (...) {
      Stop();
      throw;
    }
  }
  ~ReadPool() {
    Stop();
    ReleaseResources();
  }

  /* Runs one demand phase for `layer`: resolves any background job (adopting
   * a same-layer, healthy job whose reads the batch will consume, parking a
   * next-layer job for after the phase, dropping anything overtaken), lets
   * `build` assemble the task list around the adopted job, then keeps the
   * workers on it while the caller's overlap callback runs. `adopt_wide`
   * gates prefill-scale jobs, which a small batch must drop instead of
   * reading a whole prompt's experts. */
  /* Resolves the background job before a demand phase of `layer` is
   * planned: quiesces its tasks and adopts a healthy same-layer job (a wide
   * one only when adopt_wide), parks a next-layer job to resume after the
   * phase, and drops anything overtaken. Callers need the adopted job's row
   * assignment to build the weight table, which is why this is separate
   * from RunPhase. A job submitted between Resolve and RunPhase runs as
   * background after the phase and is simply double-read if it targeted
   * this layer — harmless, and rarer than the prediction it loses. Jobs are
   * matched by prediction window: `window` is the phase's own (kNoWindow
   * when nothing may be adopted), `next` the one its prepare opened. */
  std::shared_ptr<PrefetchJob> Resolve(uint64_t window, uint64_t next,
                                       bool adopt_wide,
                                       uint64_t* dropped_bytes) {
    std::unique_lock lock(mutex_);
    // One thread stops the background job at a time: a submission that is
    // replacing it finishes first. Two waiters would race on pf_idle_: if the
    // submission woke first it restarted the workers with pf_stop_ cleared,
    // so they never signalled this wait again (a hang); if this wait woke
    // first it reset pf_, which the submission then dereferenced.
    pf_idle_.wait(lock, [this] { return !resolving_; });
    if (!pf_run_)
      return nullptr;
    std::shared_ptr<PrefetchJob> job;
    pf_stop_ = true;
    resolving_ = true;
    pf_idle_.wait(lock, [this] { return pf_inflight_ == 0; });
    pf_run_ = false;
    pf_stop_ = false;
    resolving_ = false;
    pf_->taken = pf_next_;
    if (pf_->window == window && !pf_->failed.load() &&
        (adopt_wide || !pf_->wide))
      job = pf_;  // its un-run, routed tasks join the phase
    else if (pf_->window == next && !pf_->failed.load())
      parked_ = pf_;  // the next window's half is free; resume after
    else
      *dropped_bytes += pf_->bytes.load();
    pf_.reset();
    pf_next_ = 0;
    return job;
  }

  bool RunPhase(uint64_t next, const std::vector<ReadTask>& tasks,
                bool (*overlap)(void*, const Overlap*), void* overlap_context,
                const Overlap& info) {
    {
      std::lock_guard lock(mutex_);
      demand_ = &tasks;
      demand_next_ = 0;
      parked_workers_ = 0;
      phase_failed_.store(false);
      ++wake_;
      start_.notify_all();
    }
    const bool overlap_ok = !overlap || overlap(overlap_context, &info);
    {
      const auto wait_start = std::chrono::steady_clock::now();
      std::unique_lock lock(mutex_);
      done_.wait(lock, [this] { return parked_workers_ == threads_.size(); });
      last_wait_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - wait_start)
                          .count();
      demand_ = nullptr;
      parked_workers_ = 0;
      /* After the phase: a job submitted mid-phase runs next; otherwise the
       * parked next-window job resumes. Both target `next`; anything else
       * was overtaken by events and is not queued in the first place. */
      std::shared_ptr<PrefetchJob> resume;
      if (pending_ && pending_->window == next)
        resume = std::move(pending_);
      else if (parked_ && parked_->window == next)
        resume = std::move(parked_);
      pending_.reset();
      parked_.reset();
      if (resume) {
        pf_ = std::move(resume);
        pf_next_ = pf_->taken;  // a parked job resumes where it stopped
        pf_run_ = true;
        ++wake_;
      }
    }
    start_.notify_all();
    return overlap_ok && !phase_failed_.load();
  }

  /* Time the last phase blocked on its reads after the overlapped work. */
  uint64_t last_wait_ns() const { return last_wait_ns_; }

  /* `expected` is the prediction window of the next prepare. Checked under
   * the lock a prepare's Resolve takes after advancing it, so a job that an
   * evaluation overtook while it was built never starts: its half may hold
   * the rows the current layer's kernels are reading. */
  bool SubmitPrefetch(std::shared_ptr<PrefetchJob> job,
                      const std::atomic<uint64_t>& expected) {
    std::unique_lock lock(mutex_);
    // A submission racing a phase's job resolution is simply dropped: the
    // prediction is best effort, and adopting half-resolved state is unsound.
    if (stopping_ || resolving_ || !job || job->tasks.empty() ||
        job->window != expected.load(std::memory_order_acquire))
      return false;
    if (demand_) {
      // Submitted mid-phase: run right after it, replacing any queued job
      // (a queued job has not run, so nothing is wasted).
      pending_ = std::move(job);
      return true;
    }
    if (pf_run_) {
      pf_stop_ = true;
      resolving_ = true;
      pf_idle_.wait(lock, [this] { return pf_inflight_ == 0; });
      pf_stop_ = false;
      resolving_ = false;
      orphaned_.fetch_add(pf_->bytes.load());
    }
    pf_ = std::move(job);
    pf_next_ = 0;
    pf_run_ = true;
    ++wake_;
    start_.notify_all();
    pf_idle_.notify_all();  // a Resolve may wait for this replacement
    return true;
  }

  /* Stops and drops the background job, queued or running, so the caller
   * may move slots while no phase runs (Store::Release); prefetch is best
   * effort. Its bytes count as displaced (TakeOrphaned). No submission may
   * race it. */
  void Cancel() {
    std::unique_lock lock(mutex_);
    pf_idle_.wait(lock, [this] { return !resolving_; });
    if (pf_run_) {
      pf_stop_ = true;
      resolving_ = true;
      pf_idle_.wait(lock, [this] { return pf_inflight_ == 0; });
      pf_run_ = false;
      pf_stop_ = false;
      resolving_ = false;
    }
    for (const auto* job : {&pf_, &pending_, &parked_})
      if (*job)
        orphaned_.fetch_add((*job)->bytes.load());
    pf_.reset();
    pending_.reset();
    parked_.reset();
    pf_next_ = 0;
    pf_idle_.notify_all();
  }

  /* Bytes of jobs displaced before any phase consumed them; the store folds
   * them into its statistics from its own thread. */
  uint64_t TakeOrphaned() { return orphaned_.exchange(0); }

  /* Idle for the caller's fast path: no phase is running and no prefetch
   * task is queued or in flight. A job whose tasks all completed still
   * waits for the next phase to resolve it. */
  bool Idle() {
    std::lock_guard lock(mutex_);
    return !demand_ && !pending_ && pf_inflight_ == 0 &&
           (!pf_run_ || pf_next_ >= pf_->tasks.size());
  }

  uint64_t direct_reads() const { return direct_reads_.load(); }
  uint64_t buffered_reads() const { return buffered_reads_.load(); }

private:
  void ReleaseResources() {
    for (uint32_t i = 0; i < kReadWorkers; ++i) {
      if (events_[i])
        (void)hipEventDestroy(events_[i]);
      if (streams_[i])
        (void)hipStreamDestroy(streams_[i]);
      events_[i] = nullptr;
      streams_[i] = nullptr;
    }
  }
  void Stop() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    start_.notify_all();
    for (std::thread& thread : threads_)
      if (thread.joinable())
        thread.join();
  }

  bool RunTask(const ReadTask& task, uint32_t index) {
    const uintptr_t raw = reinterpret_cast<uintptr_t>(buffers_[index]);
    auto* buffer = reinterpret_cast<uint8_t*>((raw + kDirectAlign - 1) /
                                              kDirectAlign * kDirectAlign);
    uint64_t skip = 0;
    const bool direct =
        read_direct(task.direct, buffer, task.bytes, task.offset, &skip);
    if (direct)
      ++direct_reads_;
    else {
      skip = 0;
      ++buffered_reads_;
    }
    if (!direct && !read_all(task.file, buffer, task.bytes, task.offset))
      return false;
    {
      // Device work of this thread's own: a remap barrier elsewhere waits
      // for it and holds the next upload back (RemapBarrier).
      const mem::DeviceWorkScope device_work;
      if (hipMemcpyAsync(task.destination, buffer + skip, task.bytes,
                         hipMemcpyHostToDevice,
                         streams_[index]) != hipSuccess ||
          hipEventRecord(events_[index], streams_[index]) != hipSuccess ||
          hipEventSynchronize(events_[index]) != hipSuccess)
        return false;
    }
    if (task.job_bytes)
      task.job_bytes->fetch_add(task.bytes);
    return true;
  }

  void Work(uint32_t index) {
    const bool device_ok = hipSetDevice(device_) == hipSuccess;
    uint64_t seen = 0;
    for (;;) {
      std::unique_lock lock(mutex_);
      start_.wait(lock, [&] { return stopping_ || wake_ != seen; });
      if (stopping_)
        return;
      seen = wake_;
      lock.unlock();
      for (;;) {
        const ReadTask* task = nullptr;
        bool demand = false;
        {
          std::lock_guard lock(mutex_);
          if (stopping_)
            return;
          if (demand_) {
            if (demand_next_ < demand_->size()) {
              task = &(*demand_)[demand_next_++];
              demand = true;
            } else {
              /* Park once per phase: a worker that joined the phase from a
               * prefetch task still holds an older wake count and would
               * otherwise wake for this phase again and park twice, which
               * overshoots the count RunPhase waits for. */
              seen = wake_;
              if (++parked_workers_ == threads_.size())
                done_.notify_one();
              break;
            }
          } else if (pf_run_ && !pf_stop_ && pf_next_ < pf_->tasks.size()) {
            task = &pf_->tasks[pf_next_++];
            demand = false;
            ++pf_inflight_;
          } else {
            break;
          }
        }
        const bool ok = device_ok && RunTask(*task, index);
        std::lock_guard lock(mutex_);
        if (demand) {
          // Adopted prefetch tails carry job pointers but belong to the
          // phase, so their failure ends the evaluation like a demand read.
          if (!ok)
            phase_failed_.store(true);
        } else {
          --pf_inflight_;
          if (!ok && task->job_failed)
            task->job_failed->store(true);
          if (pf_stop_ && pf_inflight_ == 0)
            pf_idle_.notify_all();
        }
      }
    }
  }

  std::array<uint8_t*, kReadWorkers> buffers_;
  int device_;
  std::vector<std::thread> threads_;
  std::array<hipStream_t, kReadWorkers> streams_{};
  std::array<hipEvent_t, kReadWorkers> events_{};
  std::mutex mutex_;
  std::condition_variable start_, done_, pf_idle_;
  const std::vector<ReadTask>* demand_ = nullptr;
  size_t demand_next_ = 0;
  std::atomic<bool> phase_failed_{false};
  std::shared_ptr<PrefetchJob> pf_, pending_, parked_;
  bool pf_run_ = false, pf_stop_ = false, resolving_ = false;
  size_t pf_next_ = 0;
  uint32_t pf_inflight_ = 0;
  uint32_t parked_workers_ = 0;
  uint64_t last_wait_ns_ = 0;
  uint64_t wake_ = 0;
  bool stopping_ = false;
  std::atomic<uint64_t> direct_reads_{0}, buffered_reads_{0};
  std::atomic<uint64_t> orphaned_{0};
};

}  // namespace

struct Store::Impl {
  Policy policy;
  uint32_t n_layer = 0;
  uint32_t n_expert = 0;  // routed experts per layer
  uint32_t top_k = 0;
  uint32_t n_pieces = 0;
  /* Route rows of the largest split evaluation: prefill batches split their
   * resident rows from the rows their reads fill, so the resident experts'
   * kernels run while the reads stream (smaller batches are decode,
   * speculative verification and joint rows). */
  uint32_t split_rows = 0;
  std::vector<std::array<Source, kMaxPieces>> source;  // per layer
  std::unique_ptr<Entry[]> entry_table;  // [layer * n_expert + expert]
  std::vector<uint64_t> clock;           // per layer, see Policy::clock_rows
  Entry& entry(uint32_t l, uint32_t x) {
    return entry_table[uint64_t{l} * n_expert + x];
  }
  const Entry& entry(uint32_t l, uint32_t x) const {
    return entry_table[uint64_t{l} * n_expert + x];
  }
  /* How much an expert deserves its slot: −age / protect; the lowest is
   * evicted first. */
  double keep(uint32_t l, uint32_t x) const {
    const Entry& e = entry(l, x);
    return -static_cast<double>(clock[l] - e.used_at) / e.protect;
  }
  /* Prepare's per-expert working sets, sized once (Prepare calls are
   * serialized; Prefetch runs on another thread and keeps its own). */
  std::vector<uint32_t> counts, recent, stage_row;
  std::vector<uint8_t> missing, admit, hits, demand, placed;
  std::vector<uint8_t*> direct;
  /* The demand stage (whole-layer, rows indexed by expert id) and the
   * compact prefetch stage (two parity halves of job-relative rows).
   * Demand reads are ordered behind the previous layer's compute by the
   * blocking routes copy, so they safely share one buffer; background jobs
   * lack that ordering and write the compact stage instead, where the
   * parity halves keep a job from clobbering the rows the layer before
   * adopted. */
  std::array<std::array<uint8_t*, kMaxPieces>, 2> stage{};
  /* Compact prefetch row stride per projection: the largest expert of any
   * layer. Rows must not use the layer's own size, or a layer with large
   * experts overruns into the other parity half when the layer after has
   * smaller ones (the builds mix 50- to 82-byte block types). */
  std::array<uint64_t, kMaxPieces> stage_stride{};
  bool stage_pairs = false;
  /* The only prediction window a concurrent prefetch submission may target:
   * the next prepare call's (window_of). Set by that call before its reads
   * begin. */
  std::atomic<uint64_t> pf_window{window_of(0, kNoLayer)};
  /* Router-predicted prefetch: the caller records predictions with the
   * device event that signals their ids, and this thread submits each read
   * job when the event lands, so reads start during the layer the caller
   * is still computing instead of when its phase would. */
  std::thread predictor;
  std::mutex pred_mutex;
  std::condition_variable pred_cv, pred_done;
  std::vector<Prediction> pred_queue;
  bool pred_active = false;
  bool pred_stopping = false;
  std::vector<Chunk> chunks;
  std::vector<ClassRange> ranges;  // per size class, with device pages
  int device = 0;
  bool device_pages = false;          // DevicePagesAvailable at Create
  bool pages_refused = false;         // until chunks are freed
  std::vector<uint32_t> chunk_first;  // first slot index of each chunk
  /* Per-layer device tables of each expert's weight base (slot data or stage
   * row) with pinned host mirrors of what was uploaded, and the hit/miss
   * route rows of the current prepare call (hit rows, then miss rows) with
   * their pinned staging. A layer's table is uploaded only when it changes,
   * which a warm cache almost never does; the uploads are asynchronous, as
   * the next prepare's routes copy (blocking, or after a routes_ready event
   * recorded behind them) orders them before any mirror is rewritten. */
  std::array<const uint8_t**, kMaxPieces> table_host{};
  std::array<const uint8_t**, kMaxPieces> table{};
  uint64_t table_entries = 0;  // n_layer * n_expert
  uint32_t table_stride = 0;   // Layout::table_experts, or n_expert
  /* The tables of the last Prepare, as Overlap::table and its result. */
  std::array<const void*, kMaxPieces> prepared{};
  int32_t* split = nullptr;
  int32_t* split_host = nullptr;
  /* With a routes_ready event, prepare copies the routes here on its own
   * stream, so work queued after the router keeps the device busy. */
  hipStream_t routes_stream = nullptr;
  hipEvent_t routes_done = nullptr;  // the caller stream's routes copy
  int32_t* routes_host = nullptr;
  size_t routes_capacity = 0;
  std::vector<platform::PlatformFile> owned_files;  // handles the store opened
  /* With an expert-tails file on a second drive, every piece read is
   * split: the head from the model file, the tail from the tails file, so
   * both drives read at once. */
  std::vector<std::array<TailSource, kMaxPieces>> tail;  // per layer
  platform::PlatformFile tail_file = platform::kInvalidFile;
  platform::PlatformFile tail_direct = platform::kInvalidFile;
  std::vector<Slot> slots;
  /* Slot sizes: the distinct per-layer expert sizes, ascending. The builds
   * mix block types, so one largest-expert slot size wasted 14-16% of the
   * cache. A slot holds any expert up to its class's size. */
  std::vector<uint64_t> class_bytes;
  std::vector<uint32_t> layer_class;              // per layer
  std::vector<std::vector<uint32_t>> free_slots;  // per class
  /* Slots lent to misses past the demand stage's rows that the batch does
   * not admit: unoccupied, but outside free_slots, which any miss may fill.
   * A lent slot comes from here or evicts the lowest keep value, serves its
   * prepare call (`borrowed`), and returns here at the next call, when the
   * kernels that read it have finished. Admitted experts take spare slots
   * before they evict, so the slots a cold prompt set aside go back to the
   * cache as decode admits its misses. */
  std::vector<std::vector<uint32_t>> spare_slots;  // per class
  std::vector<uint32_t> borrowed;
  /* Rows for such misses when no slot can be lent (a cache smaller than a
   * batch's misses): allocated on demand, one expert each, kept until
   * Release or Fit (free_emergency_rows). */
  std::vector<uint8_t*> emergency_rows;
  uint64_t emergency_row_bytes = 0;
  uint64_t read_margin = 0;            // Layout::read_margin_bytes
  std::vector<uint64_t> grow_failure;  // tick, per class
  /* Eviction candidates, one list per size class: the kVictims lowest
   * keep values at build time of the occupied slots an expert of that
   * class could take, ascending. Pops skip slots freed, reoccupied or used
   * since the build, so the lists need no invalidation; one
   * full scan rebuilds a list when it runs dry or ages out. */
  struct Victim {
    double score;  // keep value at build
    uint32_t slot;
    uint32_t layer, expert;  // the occupant at build
    uint64_t last;           // its last use at build
  };
  struct VictimList {
    std::vector<Victim> entries;
    size_t next = 0;
    uint64_t built = 0;  // tick of the build; 0 never built
  };
  std::vector<VictimList> victims;  // per class, indexed like free_slots
  std::array<uint8_t*, kReadWorkers> read_buffers{};
  std::unique_ptr<ReadPool> readers;
  uint64_t cache_limit = 0;
  uint64_t requested_limit = 0;  // Policy::cache_limit_bytes; 0: the grant
  bool granted = false;          // Fit ran: the host sized the cache
  uint64_t granted_limit = 0;    // the cache's limit from Fit; Release lowers
                                 // cache_limit below it until Restore
  uint64_t released = 0;         // freed by Release since Fit or Restore
  uint64_t tick = 0;
  uint64_t logged_tokens = 0;
  /* Policy::trace_path: binary routing trace (Store::Tracing). The next
   * prepare writes a snapshot of the cache when trace_started is false: the
   * first one, and the first after Fit, Release or Restore changed the
   * cache outside a prepare. */
  std::FILE* trace = nullptr;
  bool trace_started = false;
  bool seeding = false;  // seed prepares stay out of the trace
  Stats stats;
  Stats logged_stats;

  ~Impl() {
    {
      std::lock_guard lock(pred_mutex);
      pred_stopping = true;
    }
    pred_cv.notify_all();
    if (predictor.joinable())
      predictor.join();
    readers.reset();
    if (trace)
      (void)std::fclose(trace);
    for (platform::PlatformFile file : owned_files)
      platform::CloseFile(file);
    (void)hipDeviceSynchronize();
    const bool had_pages =
        std::any_of(chunks.begin(), chunks.end(),
                    [](const Chunk& chunk) { return !chunk.pages.empty(); });
    while (!chunks.empty())
      free_last_chunk();
    // The next owner of these addresses may map them at once: leave them
    // clean (RemapBarrier). The device is already idle.
    if (had_pages)
      mem::RemapBarrier::Settle();
    for (ClassRange& range : ranges)
      if (range.base)
        (void)hipMemAddressFree(range.base, range.reserved);
    for (auto& set : stage)
      for (void* p : set)
        if (p)
          (void)hipFree(p);
    for (void* p : emergency_rows)
      (void)hipFree(p);
    for (void* p : read_buffers)
      if (p)
        (void)hipHostFree(p);
    for (uint32_t p = 0; p < kMaxPieces; ++p) {
      if (table[p])
        (void)hipFree(table[p]);
      if (table_host[p])
        (void)hipHostFree(table_host[p]);
    }
    if (split)
      (void)hipFree(split);
    if (split_host)
      (void)hipHostFree(split_host);
    if (routes_host)
      (void)hipHostFree(routes_host);
    if (routes_stream)
      (void)hipStreamDestroy(routes_stream);
    if (routes_done)
      (void)hipEventDestroy(routes_done);
  }

  /* Memory the cache may take now: host memory, and with device pages the
   * device heap too, which they are charged to. */
  uint64_t available_memory() const {
    const uint64_t host = AvailableMemory();
    if (!device_pages)
      return host;
    size_t heap_free = 0, heap_total = 0;
    if (hipMemGetInfo(&heap_free, &heap_total) != hipSuccess) {
      (void)hipGetLastError();
      return 0;
    }
    return std::min<uint64_t>(host, heap_free);
  }

  /* Frees the newest chunk. A device-page chunk returns its pages, and its
   * class's range ends where the chunk began. */
  void free_last_chunk() {
    Chunk& chunk = chunks.back();
    if (chunk.pages.empty()) {
      (void)hipFree(chunk.base);
      (void)hipGetLastError();
    } else {
      ClassRange& range = ranges[chunk.size_class];
      range.released_at = mem::RemapBarrier::Stamp();
      UnmapPages(chunk.base, chunk.pages.data(), chunk.pages.size());
      range.mapped = static_cast<uint64_t>(chunk.base - range.base);
      range.used = chunk.class_used;
    }
    chunks.pop_back();
  }

  // Free memory the cache never grows into at this moment.
  uint64_t live_floor() const { return policy.live_floor_bytes; }

  /* Frees the emergency rows; no prepare may be running and the device
   * must reach none of them again (Release, Fit). Returns their bytes. */
  uint64_t free_emergency_rows() {
    if (emergency_rows.empty())
      return 0;
    if (hipDeviceSynchronize() != hipSuccess) {
      (void)hipGetLastError();
      return 0;
    }
    for (void* row : emergency_rows)
      (void)hipFree(row);
    (void)hipGetLastError();
    const uint64_t bytes =
        emergency_rows.size() * (emergency_row_bytes + read_margin);
    emergency_rows.clear();
    return bytes;
  }

  bool grow(uint32_t size_class) {
    const uint64_t free_bytes = available_memory();
    const uint64_t floor = live_floor();
    if (free_bytes <= floor)
      return false;
    uint64_t available = free_bytes - floor;
    if (stats.cache_bytes >= cache_limit)
      return false;
    available = std::min(available, cache_limit - stats.cache_bytes);
    const uint64_t slot_bytes = class_bytes[size_class];
    // A chunk holds at least one slot, even of an expert above kChunkBytes.
    uint64_t count =
        std::min(available, std::max(kChunkBytes, slot_bytes)) / slot_bytes;
    if (!count)
      return false;
    Chunk chunk;
    chunk.size_class = size_class;
    uint8_t* first = nullptr;  // the first new slot
    if (device_pages) {
      if (pages_refused)
        return false;
      ClassRange& range = ranges[size_class];
      const uint64_t end = range.used + count * slot_bytes + read_margin;
      const uint64_t bytes =
          end > range.mapped
              ? (end - range.mapped + kPageBytes - 1) / kPageBytes * kPageBytes
              : kPageBytes;
      if (!MapClassPages(device, range, bytes, chunk)) {
        gufo::server::Logger::LogFormatted(
            gufo::server::LogLevel::kWarn, policy.log_tag.c_str(),
            "the device heap refused expert cache pages at %.2f GiB",
            static_cast<double>(stats.cache_bytes) / kGiB);
        pages_refused = true;
        return false;
      }
      // Slots fill the new pages, within the budget.
      count = std::min((range.mapped - read_margin - range.used) / slot_bytes,
                       available / slot_bytes);
      chunk.class_used = range.used;
      first = range.base + range.used;
      range.used += count * slot_bytes;
    } else {
      void* allocation = nullptr;
      if (gufo::hip::AllocateManaged(
              &allocation, count * slot_bytes + read_margin) != hipSuccess)
        return false;
      chunk.base = static_cast<uint8_t*>(allocation);
      chunk.bytes = count * slot_bytes + read_margin;
      first = chunk.base;
    }
    const uint64_t bytes = count * slot_bytes;
    chunks.push_back(std::move(chunk));
    chunk_first.push_back(static_cast<uint32_t>(slots.size()));
    stats.cache_bytes += bytes;
    for (uint64_t i = 0; i < count; ++i) {
      free_slots[size_class].push_back(static_cast<uint32_t>(slots.size()));
      slots.push_back({first + i * slot_bytes, size_class, kNoSlot, kNoSlot});
    }
    return true;
  }

  /* Refills the class's victim list with one pass over the slots: the
   * kVictims lowest keep values among occupied slots of class `own` or
   * larger, ascending. A batch's protected hits are handled at pop time
   * (the protection differs per prepare). */
  void rebuild_victims(uint32_t own) {
    VictimList& list = victims[own];
    list.entries.clear();
    list.next = 0;
    list.built = tick;
    const auto lower = [](const Victim& a, const Victim& b) {
      return a.score < b.score;  // max-heap of the kVictims smallest
    };
    for (uint32_t i = 0; i < slots.size(); ++i) {
      const Slot& s = slots[i];
      if (s.layer == kNoSlot || s.size_class < own)
        continue;
      const Entry& e = entry(s.layer, s.expert);
      const Victim v{keep(s.layer, s.expert), i, s.layer, s.expert, e.last};
      if (list.entries.size() < kVictims) {
        list.entries.push_back(v);
        std::push_heap(list.entries.begin(), list.entries.end(), lower);
      } else if (v.score < list.entries.front().score) {
        std::pop_heap(list.entries.begin(), list.entries.end(), lower);
        list.entries.back() = v;
        std::push_heap(list.entries.begin(), list.entries.end(), lower);
      }
    }
    std::sort(list.entries.begin(), list.entries.end(), lower);
  }

  /* `hits` marks the experts of `layer` the current batch found resident:
   * its weight table points at their slots and its kernels run after the
   * admission copies, so they are never victims. (A miss admitted earlier
   * in the same batch may be: its table entry is the stage row.) */
  uint32_t acquire(uint32_t layer, bool allow_evict,
                   const std::vector<uint8_t>& hits) {
    const uint32_t own = layer_class[layer];
    if (free_slots[own].empty() && grow_failure[own] != tick && !grow(own))
      grow_failure[own] = tick;
    // Once the cache stops growing, the smallest larger free slot serves.
    for (uint32_t c = own; c < free_slots.size(); ++c)
      if (!free_slots[c].empty()) {
        const uint32_t slot = free_slots[c].back();
        free_slots[c].pop_back();
        return slot;
      }
    if (!allow_evict)
      return kNoSlot;
    return take_spare_or_evict(layer, hits);
  }

  /* A slot for an expert of `layer` that may displace another: a spare
   * one, else the lowest keep value's. The slot is returned unoccupied. */
  uint32_t take_spare_or_evict(uint32_t layer,
                               const std::vector<uint8_t>& hits) {
    const uint32_t own = layer_class[layer];
    for (uint32_t c = own; c < spare_slots.size(); ++c)
      if (!spare_slots[c].empty()) {
        const uint32_t slot = spare_slots[c].back();
        spare_slots[c].pop_back();
        return slot;
      }
    bool rebuilt = false;
    for (;;) {
      VictimList& list = victims[own];
      if (!rebuilt && (list.built == 0 || tick - list.built >= kVictimsTicks ||
                       list.next >= list.entries.size())) {
        rebuild_victims(own);
        rebuilt = true;
      }
      while (list.next < list.entries.size()) {
        const Victim& v = list.entries[list.next++];
        const uint32_t victim = v.slot;
        const Slot& s = slots[victim];
        if (s.layer != v.layer || s.expert != v.expert || s.size_class < own ||
            entry(s.layer, s.expert).last != v.last ||
            (s.layer == layer && hits[s.expert]))
          continue;  // freed, reoccupied, used since the build, or protected
        // The incoming expert was just used (age 0), so it outranks any victim.
        entry(s.layer, s.expert).slot = kNoSlot;
        slots[victim].layer = slots[victim].expert = kNoSlot;
        return victim;
      }
      if (rebuilt)
        return kNoSlot;
      // A fresh list ran out of usable entries: one rebuild, then give up.
    }
  }

  /* Frees the newest chunks until the cache fits `limit`, keeping the
   * highest keep values of each size class: survivors in the freed chunks
   * move into the kept slots of the lowest, which are evicted.
   * No prepare or prefetch may be running (Fit, or Release). A survivor
   * whose copy fails is evicted instead, so the cache stays consistent and
   * fits `limit` even then; false reports the failure. */
  bool shrink_to(uint64_t limit) {
    if (stats.cache_bytes <= limit || chunks.empty())
      return true;
    size_t drop = chunks.size();
    uint64_t freed = 0;
    while (drop > 0 && stats.cache_bytes - freed > limit) {
      --drop;
      const uint32_t first = chunk_first[drop];
      const uint32_t end =
          drop + 1 < chunks.size() ? chunk_first[drop + 1] : slots.size();
      freed += uint64_t(end - first) * class_bytes[slots[first].size_class];
    }
    const uint32_t cut = chunk_first[drop];
    if (hipDeviceSynchronize() != hipSuccess) {
      (void)hipGetLastError();
      return false;
    }
    bool copied = true;
    for (uint32_t c = 0; c < class_bytes.size(); ++c) {
      // Occupied slots of this class, kept region and freed region.
      std::vector<uint32_t> kept, moving;
      uint32_t kept_slots = 0;
      for (uint32_t i = 0; i < slots.size(); ++i) {
        if (slots[i].size_class != c)
          continue;
        if (i < cut)
          ++kept_slots;
        if (slots[i].layer != kNoSlot)
          (i < cut ? kept : moving).push_back(i);
      }
      std::vector<uint32_t> all = kept;
      all.insert(all.end(), moving.begin(), moving.end());
      const auto score = [&](uint32_t i) {
        return keep(slots[i].layer, slots[i].expert);
      };
      std::stable_sort(all.begin(), all.end(), [&](uint32_t a, uint32_t b) {
        return score(a) > score(b);
      });
      std::vector<bool> keep(slots.size(), false);
      for (size_t k = 0; k < all.size() && k < kept_slots; ++k)
        keep[all[k]] = true;
      // Kept slots freed by eviction, plus kept slots that were empty.
      std::vector<uint32_t> open;
      for (uint32_t i = 0; i < cut; ++i) {
        if (slots[i].size_class != c)
          continue;
        if (slots[i].layer == kNoSlot) {
          open.push_back(i);
        } else if (!keep[i]) {
          entry(slots[i].layer, slots[i].expert).slot = kNoSlot;
          slots[i].layer = slots[i].expert = kNoSlot;
          open.push_back(i);
        }
      }
      size_t next = 0;
      for (uint32_t i : moving) {
        Slot& from = slots[i];
        Entry& moved = entry(from.layer, from.expert);
        if (!keep[i] || next >= open.size()) {
          moved.slot = kNoSlot;
        } else if (Slot& to = slots[open[next]];
                   hipMemcpy(to.data, from.data, class_bytes[c],
                             hipMemcpyDeviceToDevice) != hipSuccess) {
          // `to` stays open and rejoins the free list below.
          (void)hipGetLastError();
          copied = false;
          moved.slot = kNoSlot;
        } else {
          ++next;
          to.layer = from.layer;
          to.expert = from.expert;
          moved.slot = static_cast<uint32_t>(&to - slots.data());
        }
        from.layer = from.expert = kNoSlot;
      }
    }
    while (chunks.size() > drop)
      free_last_chunk();
    pages_refused = false;
    chunk_first.resize(drop);
    slots.resize(cut);
    stats.cache_bytes -= freed;
    for (auto& list : free_slots) {
      list.clear();
    }
    // Spare and lent slots are unoccupied: they rejoin the free lists.
    for (auto& list : spare_slots)
      list.clear();
    borrowed.clear();
    for (uint32_t i = 0; i < slots.size(); ++i)
      if (slots[i].layer == kNoSlot)
        free_slots[slots[i].size_class].push_back(i);
    for (auto& list : victims) {
      list.entries.clear();
      list.next = 0;
      list.built = 0;
    }
    // Uploaded tables may name freed slots: force the next uploads.
    for (uint32_t p = 0; p < n_pieces; ++p)
      std::fill_n(table_host[p], uint64_t{n_layer} * table_stride, nullptr);
    return copied;
  }

  void release(uint32_t slot_id) {
    Slot& slot = slots[slot_id];
    slot.layer = slot.expert = kNoSlot;
    free_slots[slot.size_class].push_back(slot_id);
  }
};

Store::Store(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Store::~Store() = default;

std::unique_ptr<Store> Store::Create(const Layout& layout,
                                     const Policy& policy) {
  const char* tag = policy.log_tag.c_str();
  const uint32_t n_layer = static_cast<uint32_t>(layout.layers.size());
  bool valid =
      n_layer && layout.pieces && layout.pieces <= kMaxPieces && layout.top_k &&
      layout.experts >= layout.top_k && layout.max_batch_tokens &&
      policy.demand_stage_rows &&
      (!layout.table_experts || layout.table_experts >= layout.experts) &&
      policy.prefetch_stage_rows >= 2;
  for (const auto& pieces : layout.layers) {
    valid = valid && pieces.size() == layout.pieces;
    for (const Piece& piece : pieces)
      valid = valid && platform::FileValid(piece.file) && piece.bytes;
  }
  if (!valid) {
    gufo::server::Logger::LogFormatted(gufo::server::LogLevel::kError, tag,
                                       "invalid streamed expert layout");
    return nullptr;
  }
  // The ranking names each layer's expert at most once.
  std::vector<uint8_t> ranked(uint64_t{n_layer} * layout.experts, 0);
  for (const Ranked& item : policy.ranking) {
    if (item.layer >= n_layer || item.expert >= layout.experts ||
        ranked[uint64_t{item.layer} * layout.experts + item.expert]++) {
      gufo::server::Logger::LogFormatted(
          gufo::server::LogLevel::kError, tag,
          "invalid expert ranking (%s): layer %u, expert %u",
          policy.ranking_source.c_str(), item.layer, item.expert);
      return nullptr;
    }
  }
  auto owner = std::unique_ptr<Impl>(new (std::nothrow) Impl);
  Impl* store = owner.get();
  if (!store)
    return nullptr;
  store->policy = policy;
  store->n_layer = n_layer;
  store->n_expert = layout.experts;
  store->top_k = layout.top_k;
  store->n_pieces = layout.pieces;
  store->split_rows = layout.max_batch_tokens * layout.top_k;
  store->source.resize(n_layer);
  store->tail.resize(n_layer);
  store->clock.assign(n_layer, 0);
  store->layer_class.assign(n_layer, 0);
  store->entry_table =
      std::make_unique<Entry[]>(uint64_t{n_layer} * layout.experts);
  store->table_entries = uint64_t{n_layer} * layout.experts;
  store->table_stride =
      layout.table_experts ? layout.table_experts : layout.experts;
  store->read_margin = layout.read_margin_bytes;
  for (auto* v : {&store->counts, &store->recent, &store->stage_row})
    v->assign(layout.experts, 0);
  for (auto* v : {&store->missing, &store->admit, &store->hits, &store->demand,
                  &store->placed})
    v->assign(layout.experts, 0);
  store->direct.assign(layout.experts, nullptr);
  /* One reopened handle per distinct file, from the caller's handle so it
   * names the very file the caller opened: unbuffered where the platform
   * reads that way (Linux), else a handle for concurrent buffered reads
   * (Windows). */
  std::vector<std::pair<platform::PlatformFile, Source>> reopened;
  const auto reopen = [&](platform::PlatformFile file) {
    for (const auto& [from, to] : reopened)
      if (from == file)
        return to;
    Source to;
    to.file = file;
    bool direct = false;
    const platform::PlatformFile handle =
        platform::ReopenRandomReadFile(file, direct);
    if (platform::FileValid(handle)) {
      (direct ? to.direct : to.file) = handle;
      store->owned_files.push_back(handle);
    }
    reopened.emplace_back(file, to);
    return to;
  };
  std::array<uint64_t, kMaxPieces> max_bytes{};
  for (uint32_t layer = 0; layer < n_layer; ++layer) {
    uint64_t combined = 0;
    for (uint32_t p = 0; p < store->n_pieces; ++p) {
      const Piece& piece = layout.layers[layer][p];
      Source& src = store->source[layer][p];
      const Source handles = reopen(piece.file);
      src.file = handles.file;
      src.direct = handles.direct;
      src.offset = piece.offset;
      src.bytes = piece.bytes;
      max_bytes[p] = std::max(max_bytes[p], src.bytes);
      combined += src.bytes;
    }
    // Slot bases stay 256-byte aligned, as the chunk allocations are.
    combined = (combined + 255) / 256 * 256;
    if (std::find(store->class_bytes.begin(), store->class_bytes.end(),
                  combined) == store->class_bytes.end())
      store->class_bytes.push_back(combined);
  }
  std::sort(store->class_bytes.begin(), store->class_bytes.end());
  for (uint32_t layer = 0; layer < n_layer; ++layer) {
    uint64_t combined = 0;
    for (uint32_t p = 0; p < store->n_pieces; ++p)
      combined += store->source[layer][p].bytes;
    combined = (combined + 255) / 256 * 256;
    store->layer_class[layer] = static_cast<uint32_t>(
        std::lower_bound(store->class_bytes.begin(), store->class_bytes.end(),
                         combined) -
        store->class_bytes.begin());
  }
  store->free_slots.resize(store->class_bytes.size());
  store->ranges.resize(store->class_bytes.size());
  store->spare_slots.resize(store->class_bytes.size());
  store->victims.resize(store->class_bytes.size());
  store->grow_failure.assign(store->class_bytes.size(), ~0ull);
  bool ok = true;
  for (uint32_t i = 0; i < kReadWorkers && ok; ++i)
    ok = hipHostMalloc(&store->read_buffers[i],
                       kReadBytes + 2 * kDirectAlign) == hipSuccess;
  for (uint32_t p = 0; p < store->n_pieces && ok; ++p)
    ok = hipMalloc(&store->stage[0][p],
                   max_bytes[p] *
                           std::min(store->n_expert, policy.demand_stage_rows) +
                       store->read_margin) == hipSuccess;
  for (uint32_t p = 0; p < store->n_pieces; ++p)
    store->emergency_row_bytes += max_bytes[p];
  /* The second stage set (prefetch) is allocated lazily, after the startup
   * seed has taken its memory, so it never shrinks the resident set. */
  for (uint32_t p = 0; p < store->n_pieces && ok; ++p)
    ok = hipHostMalloc(reinterpret_cast<void**>(&store->table_host[p]),
                       uint64_t{n_layer} * store->table_stride *
                           sizeof(const uint8_t*)) == hipSuccess &&
         hipMalloc(reinterpret_cast<void**>(&store->table[p]),
                   uint64_t{n_layer} * store->table_stride *
                       sizeof(const uint8_t*)) == hipSuccess &&
         // Prepare uploads `experts` entries: the rest stay null (Layout).
         hipMemset(store->table[p], 0,
                   uint64_t{n_layer} * store->table_stride *
                       sizeof(const uint8_t*)) == hipSuccess;
  /* Null mirrors never match a weight base, so each layer's first prepare
   * uploads its table. */
  for (uint32_t p = 0; p < store->n_pieces && ok; ++p)
    std::fill_n(store->table_host[p], uint64_t{n_layer} * store->table_stride,
                nullptr);
  if (ok)
    ok = hipMalloc(reinterpret_cast<void**>(&store->split),
                   2 * store->split_rows * sizeof(int32_t)) == hipSuccess &&
         hipHostMalloc(reinterpret_cast<void**>(&store->split_host),
                       2 * store->split_rows * sizeof(int32_t)) == hipSuccess;
  if (!ok) {
    gufo::server::Logger::LogFormatted(
        gufo::server::LogLevel::kError, tag,
        "unable to allocate streamed expert staging buffers");
    return nullptr;
  }
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess) {
    return nullptr;
  }
  store->device = device;
  store->device_pages = DevicePagesAvailable(device);
  // A cache that lost pages maps new ones at the same addresses, which must
  // wait for the device's work (RemapBarrier), as Gufo's page pool does.
  if (store->device_pages)
    mem::RemapBarrier::SetDeviceDrain(+[]() noexcept {
      (void)hipDeviceSynchronize();
      (void)hipGetLastError();
    });
  try {
    store->readers = std::make_unique<ReadPool>(store->read_buffers, device);
  } catch (const std::exception&) {
    return nullptr;
  }
  try {
    store->predictor = std::thread(PredictWork, store);
  } catch (const std::exception&) {
    return nullptr;
  }
  // Nothing is cached until Fit sizes the cache: misses read into the stage.
  store->cache_limit = 0;
  store->requested_limit = policy.cache_limit_bytes;
  if (!policy.trace_path.empty()) {
    store->trace = std::fopen(policy.trace_path.c_str(), "wb");
    if (!store->trace)
      gufo::server::Logger::LogFormatted(gufo::server::LogLevel::kWarn, tag,
                                         "cannot open expert trace %s",
                                         policy.trace_path.c_str());
    else if (std::fwrite(kTraceMagic, sizeof(kTraceMagic), 1, store->trace) !=
                 1 ||
             std::fwrite(&kTraceVersion, sizeof(kTraceVersion), 1,
                         store->trace) != 1)
      gufo::server::Logger::LogFormatted(gufo::server::LogLevel::kWarn, tag,
                                         "cannot write expert trace %s",
                                         policy.trace_path.c_str());
  }
  return std::unique_ptr<Store>(new Store(std::move(owner)));
}

namespace {

/* Tails file header (little endian); the deepseek41 extension's
 * tools/ds41/pack_expert_tails.py writes it. */
constexpr char kTailMagic[8] = {'G', 'U', 'F', 'O', 'E', 'T', 'L', '1'};
constexpr uint32_t kTailVersion = 1;
constexpr size_t kTailHeader = 8192;

template<typename T>
T read_le(const uint8_t* p) {
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;
}

}  // namespace

bool Store::AttachTails(const char* path) {
  Impl* store = impl_.get();
  if (!path)
    return false;
  const char* tag = store->policy.log_tag.c_str();
  const uint32_t n_layer = store->n_layer;
  const uint32_t n_pieces = store->n_pieces;
  const auto fail = [&](const char* why) {
    gufo::server::Logger::LogFormatted(gufo::server::LogLevel::kWarn, tag,
                                       "ignoring expert tails %s: %s", path,
                                       why);
    return false;
  };
  std::string error;
  const platform::PlatformFile tails = platform::OpenReadFile(path, &error);
  if (!platform::FileValid(tails))
    return fail(error.c_str());
  std::vector<uint8_t> header(kTailHeader);
  std::vector<std::array<TailSource, kMaxPieces>> tail(n_layer);
  bool ok = read_all(tails, header.data(), header.size(), 0);
  const uint8_t* h = header.data();
  if (!ok || std::memcmp(h, kTailMagic, sizeof(kTailMagic)) != 0 ||
      read_le<uint32_t>(h + 8) != kTailVersion ||
      read_le<uint32_t>(h + 12) != n_layer ||
      read_le<uint32_t>(h + 16) != store->n_expert) {
    platform::CloseFile(tails);
    return fail("not a tails file for this model layout");
  }
  const uint32_t head_ppm = read_le<uint32_t>(h + 20);
  const uint32_t n_files = read_le<uint32_t>(h + 24);
  size_t pos = 28;
  std::vector<uint64_t> file_size(n_files);
  for (uint32_t i = 0; i < n_files && pos + 8 <= kTailHeader; ++i, pos += 8)
    file_size[i] = read_le<uint64_t>(h + pos);
  /* Every piece must match the layout: size, offset in its file, and the
   * file's size. */
  for (uint32_t l = 0; l < n_layer && ok; ++l)
    for (uint32_t p = 0; p < n_pieces && ok; ++p, pos += 36) {
      if (pos + 36 > kTailHeader) {
        ok = false;
        break;
      }
      const uint64_t bytes = read_le<uint64_t>(h + pos),
                     cut = read_le<uint64_t>(h + pos + 8);
      const uint64_t padded = read_le<uint64_t>(h + pos + 16),
                     base = read_le<uint64_t>(h + pos + 24);
      const uint32_t file = read_le<uint32_t>(h + pos + 32);
      const Source& src = store->source[l][p];
      const std::optional<uint64_t> size = platform::FileSizeBytes(src.file);
      ok = bytes == src.bytes && base == src.offset && cut > 0 && cut < bytes &&
           padded >= bytes - cut && file < n_files && size &&
           *size == file_size[file];
      tail[l][p].cut = cut;
      tail[l][p].stride = padded;  // the piece size; the record stride follows
    }
  for (uint32_t l = 0; l < n_layer && ok; ++l, pos += 16) {
    if (pos + 16 > kTailHeader) {
      ok = false;
      break;
    }
    const uint64_t first = read_le<uint64_t>(h + pos),
                   record = read_le<uint64_t>(h + pos + 8);
    uint64_t piece = 0;
    for (uint32_t p = 0; p < n_pieces; ++p) {
      const uint64_t padded = tail[l][p].stride;
      tail[l][p].first = first + piece;
      tail[l][p].stride = record;
      piece += padded;
    }
    ok = piece == record && first % kDirectAlign == 0 &&
         record % kDirectAlign == 0;
  }
  const uint64_t end = tail[n_layer - 1][0].first +
                       (uint64_t)store->n_expert * tail[n_layer - 1][0].stride;
  const std::optional<uint64_t> tails_size = platform::FileSizeBytes(tails);
  if (ok && (!tails_size || *tails_size < end))
    ok = false;
  if (!ok) {
    platform::CloseFile(tails);
    return fail("it does not match this model's expert tensors; rebuild it");
  }
  /* Spot check: the tails of a few experts equal the model's bytes. */
  std::vector<uint8_t> a, b;
  for (const auto [l, e] : {std::pair<uint32_t, uint32_t>{0, 0},
                            {n_layer / 2, store->n_expert / 2},
                            {n_layer - 1, store->n_expert - 1}})
    for (uint32_t p = 0; p < n_pieces && ok; ++p) {
      const Source& src = store->source[l][p];
      const TailSource& t = tail[l][p];
      const uint64_t n = src.bytes - t.cut;
      a.resize(n);
      b.resize(n);
      ok = read_all(src.file, a.data(), n,
                    src.offset + (uint64_t)e * src.bytes + t.cut) &&
           read_all(tails, b.data(), n, t.first + (uint64_t)e * t.stride) &&
           a == b;
    }
  if (!ok) {
    platform::CloseFile(tails);
    return fail("its bytes differ from the model's; rebuild it");
  }
  bool direct = false;
  const platform::PlatformFile reopened =
      platform::ReopenRandomReadFile(tails, direct);
  store->tail_file = tails;
  store->owned_files.push_back(tails);
  if (platform::FileValid(reopened)) {
    (direct ? store->tail_direct : store->tail_file) = reopened;
    store->owned_files.push_back(reopened);
  }
  store->tail = tail;
  gufo::server::Logger::LogFormatted(gufo::server::LogLevel::kInfo, tag,
                                     "expert tails %s: %.2f GiB, %.0f%% of "
                                     "each expert read from the model file",
                                     path, static_cast<double>(end) / kGiB,
                                     head_ppm / 1e4);
  return true;
}

namespace {

bool prepare(Store::Impl* store, uint32_t layer, const int32_t* routes,
             uint32_t n_tokens, const void** tables,
             bool (*overlap)(void*, const Overlap*), void* overlap_context,
             hipEvent_t routes_ready, hipStream_t stream);
bool prefetch(Store::Impl* store, uint32_t layer, const int32_t* experts,
              uint32_t count, bool wide);

}  // namespace

/* Fills the room the grant leaves with Policy::ranking first, then every
 * expert it leaves out, interleaved across layers, as far as the cache's
 * limit and free memory reach. Startup work: it does not count as inference
 * in the statistics. */
static bool seed_cache(Store::Impl* store) {
  const uint32_t n_layer = store->n_layer;
  const uint32_t top_k = store->top_k;
  const char* source = store->policy.ranking_source.empty()
                           ? "no ranking"
                           : store->policy.ranking_source.c_str();
  const size_t ranked_count = store->policy.ranking.size();
  std::vector<Ranked> ranked = store->policy.ranking;
  std::vector<uint8_t> ranked_set(store->table_entries, 0);
  for (const Ranked& item : ranked)
    ranked_set[uint64_t{item.layer} * store->n_expert + item.expert] = 1;
  for (uint32_t e = 0; e < store->n_expert; ++e)
    for (uint32_t l = 0; l < n_layer; ++l)
      if (!ranked_set[uint64_t{l} * store->n_expert + e])
        ranked.push_back({l, e, 0});
  const uint64_t available = store->available_memory();
  const uint64_t floor = store->live_floor();
  const uint64_t free_budget = available > floor ? available - floor : 0;
  // The ranking prefix to hold: what is cached plus the room left.
  const uint64_t budget =
      std::min(store->cache_limit, store->stats.cache_bytes + free_budget);
  size_t target = 0;
  for (uint64_t used = 0; target < ranked.size(); ++target) {
    used += store->class_bytes[store->layer_class[ranked[target].layer]];
    if (used > budget)
      break;
  }
  if (!target)
    return true;
  int32_t* device_ids = nullptr;
  const uint32_t seed_slots = (store->n_expert + top_k - 1) / top_k * top_k;
  if (hipMalloc(&device_ids, seed_slots * sizeof(int32_t)) != hipSuccess)
    return false;
  const auto begin = std::chrono::steady_clock::now();
  const Stats previous = store->stats;
  std::vector<uint8_t> selected(store->table_entries, 0);
  std::vector<std::vector<Ranked>> by_layer(n_layer);
  for (size_t i = 0; i < target; ++i)
    by_layer[ranked[i].layer].push_back(ranked[i]);
  bool ok = true;
  store->seeding = true;
  /* Seeded experts start with exactly the Policy::seed_uses value: the seed
   * prepare's own uses and clock advance are undone after it. */
  const auto seed = [&](uint32_t l) {
    for (const Ranked& item : by_layer[l]) {
      Entry& seeded = store->entry(l, item.expert);
      seeded.uses =
          store->policy.seed_uses * std::min(8.0, std::log2(1.0 + item.count));
      seeded.protect = protect_of(seeded.uses, store->policy.max_class);
      seeded.used_at = store->clock[l];
    }
  };
  for (uint32_t l = 0; l < n_layer && ok; ++l) {
    std::vector<int32_t> routes;
    for (const Ranked& item : by_layer[l]) {
      routes.push_back(static_cast<int32_t>(item.expert));
      selected[uint64_t{l} * store->n_expert + item.expert] = 1;
    }
    seed(l);
    if (routes.empty())
      continue;
    if (store->available_memory() <= floor)
      break;
    // Whole tokens: dropped rows pad the last one and count no use.
    while (routes.size() % top_k)
      routes.push_back(-1);
    const uint64_t clock = store->clock[l];
    const void* tables[kMaxPieces] = {};
    ok = hipMemcpy(device_ids, routes.data(), routes.size() * sizeof(int32_t),
                   hipMemcpyHostToDevice) == hipSuccess &&
         prepare(store, l, device_ids,
                 static_cast<uint32_t>(routes.size() / top_k), tables, nullptr,
                 nullptr, nullptr, nullptr);
    store->clock[l] = clock;
    seed(l);
  }
  store->seeding = false;
  (void)hipFree(device_ids);
  const uint64_t bytes = store->stats.bytes_read - previous.bytes_read;
  const uint64_t allocated = store->stats.cache_bytes;
  store->stats = previous;
  store->stats.cache_bytes = allocated;
  if (ok) {
    uint32_t resident = 0;
    for (uint32_t l = 0; l < n_layer; ++l)
      for (uint32_t e = 0; e < store->n_expert; ++e)
        resident += selected[uint64_t{l} * store->n_expert + e] &&
                    store->entry(l, e).slot != kNoSlot;
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
            .count();
    gufo::server::Logger::LogFormatted(
        gufo::server::LogLevel::kInfo, store->policy.log_tag.c_str(),
        "expert seed (%s): cached %u of %zu experts (%zu ranked) from %.2f GiB "
        "in %.2f s",
        source, resident, target, std::min(target, ranked_count),
        static_cast<double>(bytes) / kGiB, seconds);
  }
  return ok;
}

namespace {

/* The layer's pieces in file order (by handle, then offset), so reads
 * of one layer run sequentially through each file. */
std::array<uint32_t, kMaxPieces> piece_order(const Store::Impl* store,
                                             uint32_t layer) {
  const auto& src = store->source[layer];
  const auto before = [&](uint32_t a, uint32_t b) {
    return src[a].file == src[b].file
               ? src[a].offset < src[b].offset
               : std::less<platform::PlatformFile>()(src[a].file, src[b].file);
  };
  // Insertion sort: at most kMaxPieces distinct pieces.
  std::array<uint32_t, kMaxPieces> order{};
  const uint32_t n = std::min(store->n_pieces, kMaxPieces);
  for (uint32_t p = 0; p < n; ++p) {
    uint32_t i = p;
    for (; i > 0 && before(p, order[i - 1]); --i)
      order[i] = order[i - 1];
    order[i] = p;
  }
  return order;
}

/* Reads of `bytes` from `offset` of a file into dst, each at most kReadBytes:
 * a worker reads a task whole into its pinned buffer of that size. */
void append_reads(platform::PlatformFile file, platform::PlatformFile direct,
                  uint64_t offset, uint64_t bytes, uint8_t* dst,
                  std::vector<ReadTask>& tasks) {
  for (uint64_t done = 0; done < bytes;) {
    const uint64_t n = std::min(bytes - done, kReadBytes);
    tasks.push_back({file, direct, offset + done, n, dst + done});
    done += n;
  }
}

/* Reads of one expert's piece p into dst: the whole span from the model
 * file, or with an expert-tails file its head from the model file and its
 * tail from the tails file, queued next to each other so the two drives read
 * together. */
void append_projection(const Store::Impl* store, uint32_t layer, uint32_t p,
                       uint32_t e, uint8_t* dst, std::vector<ReadTask>& tasks) {
  const Source& src = store->source[layer][p];
  const uint64_t offset = src.offset + (uint64_t)e * src.bytes;
  const TailSource& t = store->tail[layer][p];
  if (!platform::FileValid(store->tail_file) || !t.stride) {
    append_reads(src.file, src.direct, offset, src.bytes, dst, tasks);
    return;
  }
  append_reads(src.file, src.direct, offset, t.cut, dst, tasks);
  append_reads(store->tail_file, store->tail_direct,
               t.first + (uint64_t)e * t.stride, src.bytes - t.cut, dst + t.cut,
               tasks);
}

/* Demand read tasks. Experts with a stage row (`row_of`, assigned in
 * ascending expert order at the layer's own row size, so neighbours in the
 * file are neighbours in the stage) merge into chunks of at most kReadBytes
 * per piece, ordered by file and offset for sequential access. An expert
 * with a base instead is read into it, its pieces one after another as in a
 * cache slot. */
std::vector<ReadTask> build_read_tasks(const Store::Impl* store, uint32_t layer,
                                       const std::vector<uint32_t>& row_of,
                                       const std::vector<uint8_t*>& base_of) {
  const auto& src = store->source[layer];
  const uint32_t n_pieces = store->n_pieces;
  const uint32_t n_expert = store->n_expert;
  const std::array<uint32_t, kMaxPieces> order = piece_order(store, layer);
  std::array<uint64_t, kMaxPieces> within{};  // piece offsets in a slot
  for (uint32_t p = 1; p < n_pieces; ++p)
    within[p] = within[p - 1] + src[p - 1].bytes;
  std::vector<ReadTask> tasks;
  for (uint32_t i = 0; i < n_pieces; ++i) {
    const uint32_t p = order[i];
    for (uint32_t e = 0; e < n_expert;) {
      if (row_of[e] == kNoSlot) {
        ++e;
        continue;
      }
      const uint32_t first = e++;
      uint8_t* dst =
          store->stage[0][p] + (uint64_t)row_of[first] * src[p].bytes;
      if (platform::FileValid(store->tail_file)) {
        append_projection(store, layer, p, first, dst, tasks);
        continue;
      }
      while (e < n_expert && row_of[e] != kNoSlot)
        ++e;
      uint64_t offset = src[p].offset + (uint64_t)first * src[p].bytes;
      uint64_t left = (e - first) * src[p].bytes;
      while (left) {
        const uint64_t bytes = std::min(left, kReadBytes);
        tasks.push_back({src[p].file, src[p].direct, offset, bytes, dst});
        offset += bytes;
        dst += bytes;
        left -= bytes;
      }
    }
    for (uint32_t e = 0; e < n_expert; ++e)
      if (base_of[e])
        append_projection(store, layer, p, e, base_of[e] + within[p], tasks);
  }
  return tasks;
}

/* The compact prefetch stage is bought when prefetch first starts, or by
 * Fit. `stage_bytes`, when given, receives its size, bought now or before. */
bool ensure_stage_pairs(Store::Impl* store, uint64_t* stage_bytes = nullptr) {
  std::array<uint64_t, kMaxPieces> bytes{};
  for (uint32_t p = 0; p < store->n_pieces; ++p)
    for (uint32_t layer = 0; layer < store->n_layer; ++layer)
      bytes[p] = std::max(bytes[p], store->source[layer][p].bytes);
  if (stage_bytes) {
    *stage_bytes = 0;
    for (uint32_t p = 0; p < store->n_pieces; ++p)
      *stage_bytes +=
          bytes[p] * store->policy.prefetch_stage_rows + store->read_margin;
  }
  if (store->stage_pairs)
    return true;
  store->stage_stride = bytes;
  for (uint32_t p = 0; p < store->n_pieces; ++p)
    if (hipMalloc(&store->stage[1][p],
                  bytes[p] * store->policy.prefetch_stage_rows +
                      store->read_margin) != hipSuccess) {
      for (uint32_t q = 0; q <= p; ++q) {
        if (store->stage[1][q])
          (void)hipFree(store->stage[1][q]);
        store->stage[1][q] = nullptr;
      }
      return false;
    }
  store->stage_pairs = true;
  return true;
}

/* Prefetch tasks of one expert into a compact stage row: a chunked range
 * per piece, in the file/offset-sorted piece order. */
void append_expert_tasks(const Store::Impl* store, uint32_t layer, uint32_t row,
                         uint32_t e, std::vector<ReadTask>& tasks) {
  const std::array<uint32_t, kMaxPieces> order = piece_order(store, layer);
  for (uint32_t i = 0; i < store->n_pieces; ++i) {
    const uint32_t p = order[i];
    append_projection(
        store, layer, p, e,
        store->stage[1][p] + (uint64_t)row * store->stage_stride[p], tasks);
  }
}

std::shared_ptr<PrefetchJob> build_prefetch_job(
    const Store::Impl* store, uint64_t window,
    const std::vector<uint32_t>& routed, bool wide) {
  if (!store->stage_pairs)
    return nullptr;
  const uint32_t n_expert = store->n_expert;
  const uint32_t layer = window_layer(window);
  auto job = std::make_shared<PrefetchJob>();
  job->layer = layer;
  job->window = window;
  job->wide = wide;
  job->task_begin.assign(n_expert, 0);
  job->task_count.assign(n_expert, 0);
  job->row_of.assign(n_expert, 0);
  /* Tasks are grouped per expert at a compact row of the window's half, so
   * a later prepare can adopt exactly the routed experts' reads; experts
   * beyond the half simply stay demand reads. */
  const uint32_t half = store->policy.prefetch_stage_rows / 2;
  const uint32_t base = window_half(window) * half;
  /* A job wider than its half keeps the experts the most predicted routes
   * use; the kept experts still read in id order. */
  std::vector<uint8_t> wanted(n_expert, 0);
  uint32_t n_wanted = 0;
  for (uint32_t e = 0; e < n_expert; ++e)
    n_wanted += (wanted[e] = routed[e] != 0);
  if (n_wanted > half) {
    std::vector<uint32_t> order(n_expert);
    for (uint32_t e = 0; e < n_expert; ++e)
      order[e] = e;
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
      return routed[a] > routed[b];
    });
    for (uint32_t i = half; i < n_expert; ++i)
      wanted[order[i]] = 0;
  }
  uint32_t used = 0;
  for (uint32_t e = 0; e < n_expert && used < half; ++e) {
    if (!wanted[e])
      continue;
    job->row_of[e] = base + used++;
    job->task_begin[e] = static_cast<uint32_t>(job->tasks.size());
    append_expert_tasks(store, layer, job->row_of[e], e, job->tasks);
    job->task_count[e] =
        static_cast<uint32_t>(job->tasks.size()) - job->task_begin[e];
    job->experts.push_back(e);
  }
  if (job->tasks.empty())
    return nullptr;
  for (ReadTask& task : job->tasks) {
    task.job_bytes = &job->bytes;
    task.job_failed = &job->failed;
  }
  return job;
}

void PredictWork(Store::Impl* store) {
  for (;;) {
    Prediction prediction{};
    {
      std::unique_lock lock(store->pred_mutex);
      store->pred_cv.wait(lock, [&] {
        return store->pred_stopping || !store->pred_queue.empty();
      });
      if (store->pred_stopping)
        return;
      prediction = store->pred_queue.front();
      store->pred_queue.erase(store->pred_queue.begin());
      store->pred_active = true;
    }
    if (hipEventSynchronize(prediction.event) == hipSuccess)
      (void)prefetch(store, prediction.layer, prediction.ids, prediction.count,
                     prediction.wide);
    {
      std::lock_guard lock(store->pred_mutex);
      store->pred_active = false;
      store->pred_done.notify_all();
    }
  }
}

}  // namespace

namespace {

uint64_t trace_now() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

template<typename T>
void trace_put(std::FILE* f, const T& value) {
  (void)std::fwrite(&value, sizeof(T), 1, f);
}

/* 'S': the cache as seeded, so a replay starts from the same state, with
 * the layout and the policy values a replay mirrors. */
void trace_snapshot(Store::Impl* store) {
  std::FILE* f = store->trace;
  const Policy& policy = store->policy;
  trace_put(f, 'S');
  trace_put(f, store->n_layer);
  trace_put(f, store->n_expert);
  trace_put(f, store->top_k);
  trace_put(f, store->n_pieces);
  trace_put(f, store->read_margin);  // uint64
  trace_put(f, store->tick);
  trace_put(f, store->cache_limit);
  trace_put(f, store->stats.cache_bytes);
  trace_put(f, store->available_memory());
  trace_put(f, kChunkBytes);
  trace_put(f, policy.use_half_life);
  trace_put(f, policy.use_cap);
  trace_put(f, policy.max_class);
  trace_put(f, policy.clock_rows);
  trace_put(f, policy.seed_uses);
  trace_put(f, policy.recent_rows);
  trace_put(f, policy.sweep_tokens);
  trace_put(f, static_cast<uint32_t>(store->class_bytes.size()));
  for (uint64_t bytes : store->class_bytes)
    trace_put(f, bytes);
  for (uint32_t l = 0; l < store->n_layer; ++l) {
    trace_put(f, store->layer_class[l]);
    trace_put(f, store->clock[l]);
    for (uint32_t p = 0; p < store->n_pieces; ++p)
      trace_put(f, store->source[l][p].bytes);
  }
  trace_put(f, static_cast<uint32_t>(store->slots.size()));
  for (const Slot& slot : store->slots) {
    trace_put(f, slot.size_class);
    trace_put(f, slot.layer);
    trace_put(f, slot.expert);
  }
  for (uint32_t l = 0; l < store->n_layer; ++l)
    for (uint32_t e = 0; e < store->n_expert; ++e) {
      trace_put(f, store->entry(l, e).uses);
      trace_put(f, store->entry(l, e).last);
    }
}

bool prepare(Store::Impl* store, uint32_t layer, const int32_t* routes,
             uint32_t n_tokens, const void** tables,
             bool (*overlap)(void*, const Overlap*), void* overlap_context,
             hipEvent_t routes_ready, hipStream_t stream) {
  const uint32_t n_expert = store->n_expert;
  const uint32_t n_pieces = store->n_pieces;
  const uint32_t top_k = store->top_k;
  const Policy& policy = store->policy;
  const char* tag = policy.log_tag.c_str();
  if (layer >= store->n_layer || !routes || !n_tokens || !tables)
    return false;
  const auto start = std::chrono::steady_clock::now();
  std::FILE* const trace = store->seeding ? nullptr : store->trace;
  const uint64_t trace_start = trace ? trace_now() : 0;
  if (trace && !store->trace_started) {
    trace_snapshot(store);
    store->trace_started = true;
  }
  const size_t n_routes = static_cast<size_t>(n_tokens) * top_k;
  std::vector<int32_t> ids(n_routes);
  if (routes_ready) {
    if (!store->routes_stream &&
        hipStreamCreateWithFlags(&store->routes_stream, hipStreamNonBlocking) !=
            hipSuccess)
      return false;
    if (store->routes_capacity < n_routes) {
      if (store->routes_host)
        (void)hipHostFree(store->routes_host);
      store->routes_host = nullptr;
      store->routes_capacity = 0;
      if (hipHostMalloc(reinterpret_cast<void**>(&store->routes_host),
                        n_routes * sizeof(int32_t)) != hipSuccess)
        return false;
      store->routes_capacity = n_routes;
    }
    if (hipStreamWaitEvent(store->routes_stream, routes_ready, 0) !=
            hipSuccess ||
        hipMemcpyAsync(store->routes_host, routes, n_routes * sizeof(int32_t),
                       hipMemcpyDeviceToHost,
                       store->routes_stream) != hipSuccess ||
        hipStreamSynchronize(store->routes_stream) != hipSuccess)
      return false;
    std::copy_n(store->routes_host, n_routes, ids.begin());
  } else if (stream) {
    // Waiting for the copy also finishes the stream's earlier kernels, which
    // may still read the previous tables and stage rows. The routes land in
    // pinned memory and the wait spins: the host queues the experts as soon
    // as the routes exist.
    if (store->routes_capacity < n_routes) {
      if (store->routes_host)
        (void)hipHostFree(store->routes_host);
      store->routes_host = nullptr;
      store->routes_capacity = 0;
      if (hipHostMalloc(reinterpret_cast<void**>(&store->routes_host),
                        n_routes * sizeof(int32_t)) != hipSuccess)
        return false;
      store->routes_capacity = n_routes;
    }
    if (!store->routes_done &&
        hipEventCreateWithFlags(&store->routes_done, hipEventDisableTiming) !=
            hipSuccess)
      return false;
    if (hipMemcpyAsync(store->routes_host, routes, n_routes * sizeof(int32_t),
                       hipMemcpyDeviceToHost, stream) != hipSuccess ||
        hipEventRecord(store->routes_done, stream) != hipSuccess)
      return false;
    hipError_t state;
    while ((state = hipEventQuery(store->routes_done)) == hipErrorNotReady) {
    }
    if (state != hipSuccess)
      return false;
    std::copy_n(store->routes_host, n_routes, ids.begin());
  } else if (hipMemcpy(ids.data(), routes, n_routes * sizeof(int32_t),
                       hipMemcpyDeviceToHost) != hipSuccess) {
    return false;
  }
  store->stats.route_wait_ns +=
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start)
          .count();
  const uint64_t trace_routes = trace ? trace_now() : 0;
  std::vector<uint32_t>& counts = store->counts;
  std::vector<uint32_t>& recent = store->recent;
  std::fill(counts.begin(), counts.end(), 0u);
  std::fill(recent.begin(), recent.end(), 0u);
  for (size_t i = 0; i < ids.size(); ++i) {
    const int32_t id = ids[i];
    /* The router drops a slot (-1) when its scores are not finite; the
     * kernels zero that row, as on the resident path. */
    if (id == -1)
      continue;
    if (id < 0 || id >= static_cast<int32_t>(store->n_expert)) {
      gufo::server::Logger::LogFormatted(gufo::server::LogLevel::kError, tag,
                                         "invalid routed expert %d", id);
      return false;
    }
    ++counts[id];
    if (i >= (n_tokens - std::min(n_tokens, policy.recent_rows)) * top_k)
      ++recent[id];
  }
  ++store->tick;
  if (layer == 0)
    store->stats.tokens += n_tokens;
  const uint64_t now = store->clock[layer];
  std::vector<uint8_t>& missing = store->missing;
  std::vector<uint8_t>& admit = store->admit;
  std::vector<uint8_t>& hits = store->hits;
  for (auto* v : {&missing, &admit, &hits})
    std::fill(v->begin(), v->end(), uint8_t{0});
  for (uint32_t e = 0; e < n_expert; ++e) {
    if (!counts[e])
      continue;
    Entry& entry = store->entry(layer, e);
    store->stats.uses += counts[e];
    if (entry.slot != kNoSlot)
      store->stats.hits += counts[e];
    missing[e] = entry.slot == kNoSlot;
    hits[e] = !missing[e];
    const double idle = static_cast<double>(now - entry.used_at);
    entry.uses = entry.uses * std::exp2(-idle / policy.use_half_life) +
                 std::min(counts[e], policy.use_cap);
    entry.protect = protect_of(entry.uses, policy.max_class);
    entry.used_at = now;
    entry.last = store->tick;
    // A large prefill scans most experts. Admit its final rows and unusually
    // frequent experts, rather than displacing the decode working set.
    admit[e] = n_tokens <= policy.recent_rows || recent[e] ||
               counts[e] >= std::max(4u, n_tokens / 32);
  }
  /* Resolve the background job before planning the phase: the table and the
   * demand reads both need its compact row assignment. */
  uint32_t misses = 0;
  for (uint32_t e = 0; e < n_expert; ++e)
    misses += missing[e] ? 1u : 0u;
  /* From here predictions may target only the next window, whose half no
   * running kernel reads: this call's kernels read only the half of its own
   * window, and the earlier calls' kernels ran before these routes existed.
   * After the last layer the window wraps to layer 0. Jobs built for this
   * call's window are adoptable only when it names this layer. */
  const uint64_t own_window = store->pf_window.load(std::memory_order_relaxed);
  const uint64_t window =
      window_layer(own_window) == layer ? own_window : kNoWindow;
  const uint64_t next_window =
      window_next(own_window, layer + 1 < store->n_layer ? layer + 1 : 0);
  store->pf_window.store(next_window, std::memory_order_release);
  const uint64_t orphaned = store->readers->TakeOrphaned();
  store->stats.bytes_read += orphaned;
  store->stats.prefetch_dropped_bytes += orphaned;
  std::shared_ptr<PrefetchJob> adopted;
  uint64_t dropped = 0;
  uint64_t trace_demand = 0, trace_adopted = 0, trace_read_wait = 0;
  const bool phase = misses || !store->readers->Idle();
  if (phase)
    adopted = store->readers->Resolve(
        window, next_window, n_tokens >= policy.sweep_tokens, &dropped);
  store->stats.bytes_read += dropped;
  store->stats.prefetch_dropped_bytes += dropped;
  /* The slots lent to the previous call are free of readers now: every
   * kernel of that call ran before this call's routes existed. */
  for (uint32_t slot_id : store->borrowed)
    store->spare_slots[store->slots[slot_id].size_class].push_back(slot_id);
  store->borrowed.clear();
  /* Where each miss no adopted job covers is read. While they fit the demand
   * stage, all of them into its rows, in expert order. A batch with more
   * reads the experts it admits into their slots (no copy follows) and
   * gives the rows to the others; those past the rows fill free slots, as
   * a scan does, or take a lent one. A seed prepare runs no kernels, so its
   * misses past the rows that find no free slot are not read at all. */
  const uint32_t stage_rows = std::min(n_expert, policy.demand_stage_rows);
  std::vector<uint32_t>& stage_row = store->stage_row;
  std::vector<uint8_t*>& direct = store->direct;
  std::vector<uint8_t>& demand = store->demand;
  std::vector<uint8_t>& placed = store->placed;
  std::fill(stage_row.begin(), stage_row.end(), kNoSlot);
  std::fill(direct.begin(), direct.end(), nullptr);
  std::fill(demand.begin(), demand.end(), uint8_t{0});
  std::fill(placed.begin(), placed.end(), uint8_t{0});
  uint32_t demands = 0, rows_used = 0, emergency_used = 0;
  for (uint32_t e = 0; e < store->n_expert; ++e) {
    demand[e] = missing[e] && !(adopted && adopted->task_count[e]);
    demands += demand[e] ? 1u : 0u;
  }
  std::vector<uint32_t> claimed;  // slots this call's reads fill
  const auto take = [&](uint32_t e, bool evict) {
    const uint32_t slot_id = store->acquire(layer, evict, hits);
    if (slot_id == kNoSlot)
      return false;
    Slot& slot = store->slots[slot_id];
    slot.layer = layer;
    slot.expert = e;
    store->entry(layer, e).slot = slot_id;
    claimed.push_back(slot_id);
    hits[e] = true;  // the table points at it: never a victim of this call
    direct[e] = slot.data;
    placed[e] = true;
    return true;
  };
  /* A failed call caches nothing its reads may not have filled, and leaves
   * the layer's mirror null so the next call uploads its whole table. */
  const auto abandon = [&] {
    for (uint32_t slot_id : claimed) {
      const Slot& slot = store->slots[slot_id];
      store->entry(slot.layer, slot.expert).slot = kNoSlot;
      store->release(slot_id);
    }
    for (uint32_t p = 0; p < n_pieces; ++p)
      std::fill_n(store->table_host[p] + uint64_t{layer} * store->table_stride,
                  n_expert, nullptr);
    return false;
  };
  if (demands > stage_rows)
    for (uint32_t e = 0; e < store->n_expert; ++e)
      if (demand[e] && admit[e])
        (void)take(e, true);
  for (uint32_t e = 0; e < store->n_expert; ++e) {
    if (!demand[e] || placed[e])
      continue;
    if (rows_used < stage_rows) {
      stage_row[e] = rows_used++;
      continue;
    }
    if (take(e, false))
      continue;
    placed[e] = true;
    if (store->seeding)
      continue;
    const uint32_t slot_id = store->take_spare_or_evict(layer, hits);
    if (slot_id != kNoSlot) {
      store->borrowed.push_back(slot_id);
      direct[e] = store->slots[slot_id].data;
      continue;
    }
    if (emergency_used == store->emergency_rows.size()) {
      void* row = nullptr;
      if (hipMalloc(&row, store->emergency_row_bytes + store->read_margin) !=
          hipSuccess) {
        gufo::server::Logger::LogFormatted(gufo::server::LogLevel::kError, tag,
                                           "unable to stage a missed expert");
        return abandon();
      }
      store->emergency_rows.push_back(static_cast<uint8_t*>(row));
    }
    direct[e] = store->emergency_rows[emergency_used++];
  }
  /* Map every expert of the layer to its weights: the cache slot when
   * resident, the adopted job's compact stage row when it stages one, else
   * the demand stage row or slot the phase's reads fill. The kernels index
   * the table, so resident experts are read in place without a staging
   * copy. An expert the batch does not route gets a row nothing reads. */
  const auto& src = store->source[layer];
  std::array<const uint8_t**, kMaxPieces> mirror{};
  std::array<const uint8_t**, kMaxPieces> device_table{};
  for (uint32_t p = 0; p < n_pieces; ++p) {
    mirror[p] = store->table_host[p] + uint64_t{layer} * store->table_stride;
    device_table[p] = store->table[p] + uint64_t{layer} * store->table_stride;
  }
  std::array<bool, kMaxPieces> changed{};
  for (uint32_t e = 0; e < store->n_expert; ++e) {
    const uint32_t slot_id = store->entry(layer, e).slot;
    const uint8_t* base = direct[e]            ? direct[e]
                          : slot_id != kNoSlot ? store->slots[slot_id].data
                                               : nullptr;
    const bool covered = adopted && adopted->task_count[e] != 0;
    const uint32_t demand_row = stage_row[e] != kNoSlot ? stage_row[e] : 0;
    const uint32_t job_row = covered ? adopted->row_of[e] : 0;
    uint64_t staged = 0;
    for (uint32_t p = 0; p < n_pieces; ++p) {
      const uint8_t* row =
          covered
              ? store->stage[1][p] + (uint64_t)job_row * store->stage_stride[p]
              : store->stage[0][p] + (uint64_t)demand_row * src[p].bytes;
      const uint8_t* entry = base ? base + staged : row;
      changed[p] = changed[p] || mirror[p][e] != entry;
      mirror[p][e] = entry;
      staged += src[p].bytes;
    }
  }
  for (uint32_t p = 0; p < n_pieces; ++p)
    if (changed[p] &&
        hipMemcpyAsync(device_table[p], mirror[p],
                       store->n_expert * sizeof(uint8_t*),
                       hipMemcpyHostToDevice, stream) != hipSuccess)
      return abandon();
  /* The hit and miss rows keep their route positions, each array holding -1
   * for the other's experts, so one launch over either array leaves exactly
   * the other launch's rows to fill. */
  Overlap info{};
  info.layer = layer;
  info.rows = static_cast<uint32_t>(n_routes);
  for (uint32_t p = 0; p < n_pieces; ++p)
    store->prepared[p] = device_table[p];
  info.table = store->prepared.data();
  uint32_t miss_rows = 0;
  for (size_t i = 0; i < ids.size(); ++i)
    miss_rows += ids[i] >= 0 && missing[ids[i]];
  info.miss_rows = miss_rows;
  if (n_routes <= store->split_rows) {
    /* Only a batch with both hits and misses splits its launches; any other
     * batch keeps one pass over the routes, so its rows are not uploaded. */
    if (miss_rows && miss_rows < n_routes) {
      int32_t* hit = store->split_host;
      int32_t* miss = store->split_host + n_routes;
      for (size_t i = 0; i < ids.size(); ++i) {
        const int32_t id = ids[i];
        const bool is_miss = id >= 0 && missing[id];
        hit[i] = is_miss ? -1 : id;  // a dropped slot stays -1 in both
        miss[i] = is_miss ? id : -1;
      }
      info.hit_ids = store->split;
      info.miss_ids = store->split + n_routes;
      if (hipMemcpyAsync(store->split, store->split_host,
                         2 * n_routes * sizeof(int32_t), hipMemcpyHostToDevice,
                         stream) != hipSuccess)
        return abandon();
    }
  }
  if (phase) {
    // Demand reads of the misses no adopted job row covers; the job's
    // un-run tasks for the experts this batch routes join them.
    std::vector<ReadTask> tasks =
        build_read_tasks(store, layer, stage_row, direct);
    uint64_t demand_bytes = 0;
    for (const ReadTask& task : tasks)
      demand_bytes += task.bytes;
    if (adopted) {
      /* Adopt only the experts this batch routes: waiting for a wrong
       * prediction would cost the phase its whole read. Tasks the workers
       * already took (index < taken) are done; the rest run in-phase. */
      const uint32_t taken = static_cast<uint32_t>(adopted->taken);
      for (uint32_t e : adopted->experts) {
        if (!missing[e])
          continue;
        const uint32_t end = adopted->task_begin[e] + adopted->task_count[e];
        for (uint32_t i = std::max(adopted->task_begin[e], taken); i < end; ++i)
          tasks.push_back(adopted->tasks[i]);
      }
    }
    if (!store->readers->RunPhase(next_window, tasks, overlap, overlap_context,
                                  info)) {
      gufo::server::Logger::LogFormatted(
          gufo::server::LogLevel::kError, tag,
          "failed to stage experts or run overlapped work");
      return abandon();
    }
    store->stats.read_wait_ns += store->readers->last_wait_ns();
    store->stats.bytes_read += demand_bytes;
    trace_read_wait = store->readers->last_wait_ns();
    trace_demand = demand_bytes;
    if (adopted) {
      const uint64_t bytes = adopted->bytes.load();
      trace_adopted = bytes;
      store->stats.bytes_read += bytes;
      store->stats.prefetch_bytes += bytes;
    }
  } else if (overlap && !overlap(overlap_context, &info)) {
    return abandon();
  }
  for (uint32_t e = 0; e < n_expert; ++e) {
    if (!missing[e] || placed[e])
      continue;
    // Fill unused capacity during a scan. Once full, only promoted prefill
    // experts may displace the existing working set.
    const uint32_t slot_id = store->acquire(layer, admit[e], hits);
    if (slot_id == kNoSlot)
      continue;
    Slot& slot = store->slots[slot_id];
    uint8_t* dst = slot.data;
    for (uint32_t p = 0; p < n_pieces; ++p) {
      if (hipMemcpyAsync(dst, mirror[p][e], src[p].bytes,
                         hipMemcpyDeviceToDevice, stream) != hipSuccess) {
        store->release(slot_id);
        return false;
      }
      dst += src[p].bytes;
    }
    slot.layer = layer;
    slot.expert = e;
    store->entry(layer, e).slot = slot_id;
  }
  for (uint32_t p = 0; p < n_pieces; ++p)
    tables[p] = device_table[p];
  if (trace) {
    // 'P': one prepare, its timing and each route's hit before admission.
    std::FILE* f = trace;
    trace_put(f, 'P');
    trace_put(f, trace_start);
    trace_put(f, trace_routes);
    trace_put(f, trace_now());
    trace_put(f, layer);
    trace_put(f, n_tokens);
    trace_put(f, trace_read_wait);
    trace_put(f, trace_demand);
    trace_put(f, trace_adopted);
    (void)std::fwrite(ids.data(), sizeof(int32_t), n_routes, f);
    for (size_t i = 0; i < n_routes; ++i)
      trace_put(f, static_cast<uint8_t>(ids[i] >= 0 && !missing[ids[i]]));
  }
  store->stats.stall_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
  if (layer == store->n_layer - 1 &&
      store->stats.tokens >= store->logged_tokens + 64) {
    const Stats s = store->stats;
    const Stats old = store->logged_stats;
    const uint64_t uses = s.uses - old.uses, hits = s.hits - old.hits;
    const uint64_t tokens = s.tokens - old.tokens;
    const uint64_t read = s.bytes_read - old.bytes_read;
    gufo::server::Logger::LogFormatted(
        gufo::server::LogLevel::kInfo, tag,
        "expert cache interval: %.1f%% route hits, %.2f MiB read/token "
        "(%.1f%% prefetched), %.2f ms host wait/token (%.2f routes, %.2f "
        "reads), %.2f GiB allocated, %llu tokens",
        uses ? 100.0 * static_cast<double>(hits) / uses : 0.0,
        tokens ? static_cast<double>(read) / tokens / (1u << 20) : 0.0,
        read ? 100.0 *
                   static_cast<double>(s.prefetch_bytes - old.prefetch_bytes) /
                   read
             : 0.0,
        tokens ? static_cast<double>(s.stall_ns - old.stall_ns) / tokens / 1e6
               : 0.0,
        tokens ? static_cast<double>(s.route_wait_ns - old.route_wait_ns) /
                     tokens / 1e6
               : 0.0,
        tokens ? static_cast<double>(s.read_wait_ns - old.read_wait_ns) /
                     tokens / 1e6
               : 0.0,
        static_cast<double>(s.cache_bytes) / kGiB,
        static_cast<unsigned long long>(tokens));
    store->logged_stats = s;
    store->logged_tokens = store->stats.tokens;
  }
  // After admission: a victim's age counts this batch's tokens only from the
  // next prepare of its layer on, which measured best on DeepSeek V4.1 Flash.
  store->clock[layer] += std::min<uint64_t>(n_tokens, policy.clock_rows);
  return true;
}

bool prefetch(Store::Impl* store, uint32_t layer, const int32_t* experts,
              uint32_t count, bool wide) {
  if (layer >= store->n_layer || !experts || !count)
    return false;
  // Only the layer the next prepare handles is accepted: an overtaken
  // prediction would never be adopted, and its parity half may hold rows in
  // compute. SubmitPrefetch repeats the check under the pool's lock, since
  // the first job also allocates the prefetch stage.
  const uint64_t window = store->pf_window.load(std::memory_order_acquire);
  if (layer != window_layer(window))
    return false;
  // Predicted routes per uncached expert.
  std::vector<uint32_t> want(store->n_expert, 0);
  for (uint32_t i = 0; i < count; ++i) {
    const int32_t e = experts[i];
    if (e < 0 || e >= static_cast<int32_t>(store->n_expert))
      return false;
    if (store->entry(layer, e).slot == kNoSlot)
      ++want[e];
  }
  if (!ensure_stage_pairs(store))
    return false;
  std::shared_ptr<PrefetchJob> job =
      build_prefetch_job(store, window, want, wide);
  if (!job)
    return false;
  return store->readers->SubmitPrefetch(std::move(job), store->pf_window);
}

}  // namespace

bool Store::Prepare(uint32_t layer, const int32_t* routes, uint32_t n_tokens,
                    const void** tables, bool (*overlap)(void*, const Overlap*),
                    void* overlap_context, hipEvent_t routes_ready,
                    hipStream_t stream) {
  return prepare(impl_.get(), layer, routes, n_tokens, tables, overlap,
                 overlap_context, routes_ready, stream);
}

bool Store::Prefetch(uint32_t layer, const int32_t* experts, uint32_t count,
                     bool wide) {
  return prefetch(impl_.get(), layer, experts, count, wide);
}

bool Store::Predict(uint32_t layer, hipEvent_t done, const int32_t* ids,
                    uint32_t count, bool wide) {
  Impl* store = impl_.get();
  if (!done || !ids || !count || layer >= store->n_layer)
    return false;
  std::lock_guard lock(store->pred_mutex);
  if (store->pred_stopping || store->pred_queue.size() >= 4)
    return false;
  for (const Prediction& pending : store->pred_queue)
    if (pending.layer == layer)
      return false;
  store->pred_queue.push_back({done, layer, ids, count, wide});
  store->pred_cv.notify_one();
  return true;
}

void Store::Expect(uint32_t layer) {
  // The window keeps its half: the last prepare's kernels read the other.
  if (layer < impl_->n_layer)
    impl_->pf_window.store(
        window_of(impl_->pf_window.load(std::memory_order_relaxed) >> 32,
                  layer),
        std::memory_order_release);
}

void Store::PredictFlush() {
  Impl* store = impl_.get();
  std::unique_lock lock(store->pred_mutex);
  store->pred_done.wait(
      lock, [&] { return store->pred_queue.empty() && !store->pred_active; });
}

bool Store::Tracing() const {
  return impl_->trace != nullptr;
}

void Store::TracePrediction(uint32_t layer, uint32_t lead, const int32_t* ids,
                            uint32_t n_tokens, uint32_t k) {
  Impl* store = impl_.get();
  if (!store->trace || !store->trace_started)
    return;
  std::FILE* f = store->trace;
  trace_put(f, 'Q');
  trace_put(f, layer);
  trace_put(f, lead);
  trace_put(f, n_tokens);
  trace_put(f, k);
  (void)std::fwrite(ids, sizeof(int32_t), static_cast<size_t>(n_tokens) * k, f);
}

uint64_t Store::Release(uint64_t bytes) {
  Impl* store = impl_.get();
  if (!bytes || !store->granted)
    return 0;
  PredictFlush();
  // Held so no prediction starts a prefetch while slots move.
  std::lock_guard lock(store->pred_mutex);
  // Prefetch is best effort: drop what the flush started rather than give up
  // the release while it reads.
  store->readers->Cancel();
  const uint64_t before = store->stats.cache_bytes;
  // Consistent even when a move failed: the cache then holds less.
  (void)store->shrink_to(before > bytes ? before - bytes : 0);
  const uint64_t freed =
      before - store->stats.cache_bytes + store->free_emergency_rows();
  const uint64_t limit = store->cache_limit;
  store->cache_limit = std::min(limit, store->stats.cache_bytes);
  store->released += freed;
  // The trace's replay restarts from the cache as Release left it.
  if (freed || store->cache_limit != limit)
    store->trace_started = false;
  return freed;
}

uint64_t Store::Restore() {
  Impl* store = impl_.get();
  std::lock_guard lock(store->pred_mutex);
  if (store->cache_limit >= store->granted_limit)
    return 0;
  const uint64_t raised = store->granted_limit - store->cache_limit;
  store->cache_limit = store->granted_limit;
  store->released = 0;
  store->trace_started = false;
  return raised;
}

uint64_t Store::ReleasedBytes() const {
  return impl_->released;
}

uint64_t Store::FloorBytes() const {
  return impl_->live_floor();
}

bool Store::PrefetchIdle() const {
  return impl_->readers->Idle();
}

Stats Store::GetStats() const {
  return impl_->stats;
}

void Store::Log() const {
  const Impl* store = impl_.get();
  if (!store->stats.uses)
    return;
  const auto& s = store->stats;
  const char* tag = store->policy.log_tag.c_str();
  gufo::server::Logger::LogFormatted(
      gufo::server::LogLevel::kInfo, tag,
      "expert cache total: %.1f%% route hits, %.2f MiB read/token, %.2f ms "
      "host wait/token (%.2f routes, %.2f reads), %.2f GiB allocated, %.2f GiB "
      "prefetched (%.2f GiB "
      "dropped)",
      100.0 * static_cast<double>(s.hits) / s.uses,
      s.tokens ? static_cast<double>(s.bytes_read) / s.tokens / (1u << 20)
               : 0.0,
      s.tokens ? static_cast<double>(s.stall_ns) / s.tokens / 1e6 : 0.0,
      s.tokens ? static_cast<double>(s.route_wait_ns) / s.tokens / 1e6 : 0.0,
      s.tokens ? static_cast<double>(s.read_wait_ns) / s.tokens / 1e6 : 0.0,
      static_cast<double>(s.cache_bytes) / kGiB,
      static_cast<double>(s.prefetch_bytes) / kGiB,
      static_cast<double>(s.prefetch_dropped_bytes) / kGiB);
  gufo::server::Logger::LogFormatted(
      gufo::server::LogLevel::kInfo, tag,
      "expert reads: %llu direct, %llu buffered",
      static_cast<unsigned long long>(store->readers->direct_reads()),
      static_cast<unsigned long long>(store->readers->buffered_reads()));
}

bool Store::Fit(uint64_t grant_bytes) {
  Impl* store = impl_.get();
  /* The prefetch stage is bought on the first prompt anyway; buying it now
   * takes it from the grant instead of from under the cache. */
  uint64_t stage_bytes = 0;
  const bool staged = ensure_stage_pairs(store, &stage_bytes);
  store->granted = true;
  uint64_t limit = grant_bytes > stage_bytes ? grant_bytes - stage_bytes : 0;
  if (store->requested_limit)
    limit = std::min(limit, store->requested_limit);
  store->cache_limit = limit;
  store->granted_limit = limit;
  store->released = 0;
  (void)store->free_emergency_rows();
  const bool shrunk = store->shrink_to(limit);
  const bool ok = staged && shrunk && seed_cache(store);
  // The trace's replay starts from the cache as Fit left it.
  store->trace_started = false;
  return ok;
}

}  // namespace gufo::expert_stream
