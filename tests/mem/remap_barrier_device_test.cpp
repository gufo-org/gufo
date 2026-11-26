// GPU check of the remap barrier with the HIP backend
// (src/core/hip/remap_barrier_device.*): addresses remapped while another
// process keeps the GPU busy reach their new pages once
// RemapBarrier::Settle() returns. Before Linux 7.4 the driver flushes no
// translations after an unmap, and beside a busy process the idle time
// alone left most addresses on their old pages; the translation flush must
// settle every one, for kernels, runtime copies and the copy engine. The
// busy process is this binary run with --busy.
#include "src/core/hip/remap_barrier_device.hpp"

#include <hip/hip_runtime.h>
#include <signal.h>
#include <spawn.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "src/core/hip/copy_engine.hpp"
#include "src/mem/remap_barrier.hpp"

extern char** environ;

namespace {

using gufo::mem::RemapBarrier;

constexpr std::size_t kPage = std::size_t{2} << 20;
constexpr std::size_t kWords = kPage / 4;
// Pages per batch, and handle sets the address alternates between.
constexpr int kPages = 9;
constexpr int kSets = 2;
constexpr int kBatches = 50;
// The ways of reading a page that Reads() found on another handle.
constexpr int kKernel = 1;
constexpr int kRuntimeCopy = 2;
constexpr int kEngine = 4;

void Check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
    std::exit(1);
  }
}

__global__ void Fill(std::uint32_t* data, std::size_t words,
                     std::uint32_t value) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
       i += gridDim.x * blockDim.x)
    data[i] = value;
}

// Word 0 of each page.
__global__ void ReadPages(const std::uint32_t* data, std::uint32_t* out) {
  if (threadIdx.x < kPages)
    out[threadIdx.x] = data[threadIdx.x * kWords];
}

__global__ void Spin(std::uint32_t* data, int iterations) {
  std::uint32_t x = data[threadIdx.x];
  for (int i = 0; i < iterations; ++i)
    x = x * 1664525u + 1013904223u;
  data[threadIdx.x] = x;
}

// Short kernels back to back, with no host gaps, until killed or `seconds`
// have passed. One byte on stdout once they run.
int Busy(double seconds) {
  // Ends with the test, also when a failed check exits it.
  (void)prctl(PR_SET_PDEATHSIG, SIGKILL);
  std::uint32_t* data = nullptr;
  hipStream_t stream = nullptr;
  Check(hipMalloc(&data, 4096), "busy buffer");
  Check(hipStreamCreate(&stream), "busy stream");
  const auto start = std::chrono::steady_clock::now();
  bool announced = false;
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
             .count() < seconds) {
    for (int i = 0; i < 64; ++i)
      Spin<<<1, 256, 0, stream>>>(data, 2000);
    Check(hipStreamSynchronize(stream), "busy kernels");
    if (!announced) {
      announced = write(STDOUT_FILENO, "1", 1) == 1;
    }
  }
  return 0;
}

std::uint32_t Value(int set, int page) {
  return 0x1000u * static_cast<std::uint32_t>(set + 1) +
         static_cast<std::uint32_t>(page);
}

struct Old {
  int batches = 0;  // with a page read on another handle in any way
  int engine = 0;   // through the copy engine
};

struct Pages {
  hipMemAllocationProp prop{};
  hipMemAccessDesc access{};
  hipMemGenericAllocationHandle_t handles[kSets][kPages]{};
  void* address = nullptr;
  std::uint32_t* out = nullptr;
  // The copy engine restores and disk writes use; null without one.
  gufo::hip::CopyEngine* engine = gufo::hip::DeviceCopyEngine();
  gufo::hip::CopyEngine::Fence fence{};

  Pages() {
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = 0;
    access.location = prop.location;
    access.flags = hipMemAccessFlagsProtReadWrite;
    Check(hipMalloc(&out, 4096), "read buffer");
    // Each handle holds its own constant, written through a home address
    // that is never used again.
    for (int set = 0; set < kSets; ++set)
      for (int page = 0; page < kPages; ++page) {
        auto& handle = handles[set][page];
        Check(hipMemCreate(&handle, kPage, &prop, 0), "create");
        void* home = nullptr;
        Check(hipMemAddressReserve(&home, kPage, kPage, nullptr, 0), "home");
        Check(hipMemMap(home, kPage, 0, handle, 0), "map home");
        Check(hipMemSetAccess(home, kPage, &access, 1), "access home");
        Fill<<<64, 256>>>(static_cast<std::uint32_t*>(home), kWords,
                          Value(set, page));
        Check(hipDeviceSynchronize(), "fill");
        Check(hipMemUnmap(home, kPage), "unmap home");
      }
    Check(hipMemAddressReserve(&address, kPages * kPage, kPage, nullptr, 0),
          "reserve");
    if (engine)
      fence = engine->CreateFence();
    Map(0);
  }

  void* Page(int page) const {
    return static_cast<std::byte*>(address) + page * kPage;
  }

  void Map(int set) {
    for (int page = 0; page < kPages; ++page)
      Check(hipMemMap(Page(page), kPage, 0, handles[set][page], 0), "map");
    Check(hipMemSetAccess(address, kPages * kPage, &access, 1), "access");
  }

  static bool Match(const std::uint32_t (&got)[kPages], int set) {
    for (int page = 0; page < kPages; ++page)
      if (got[page] != Value(set, page))
        return false;
    return true;
  }

  // Which ways of reading word 0 of each page miss `set`'s constant: a
  // kernel, hipMemcpy and the copy engine.
  int Reads(int set) {
    int wrong = 0;
    std::uint32_t got[kPages];
    ReadPages<<<1, 64>>>(static_cast<const std::uint32_t*>(address), out);
    Check(hipDeviceSynchronize(), "read kernel");
    Check(hipMemcpy(got, out, sizeof(got), hipMemcpyDeviceToHost), "result");
    wrong |= Match(got, set) ? 0 : kKernel;
    for (int page = 0; page < kPages; ++page)
      Check(hipMemcpy(&got[page], Page(page), sizeof(got[page]),
                      hipMemcpyDeviceToHost),
            "copy read");
    wrong |= Match(got, set) ? 0 : kRuntimeCopy;
    if (engine) {
      // Zero is no page's constant: a copy that did not land reads wrong.
      Check(hipMemset(out, 0, sizeof(got)), "clear");
      Check(hipDeviceSynchronize(), "clear");
      for (int page = 0; page < kPages; ++page)
        if (!engine->Copy(out + page, Page(page), sizeof(got[page]), fence)) {
          std::fprintf(stderr, "the copy engine refused a copy\n");
          std::exit(1);
        }
      engine->Wait(fence);
      Check(hipMemcpy(got, out, sizeof(got), hipMemcpyDeviceToHost),
            "engine result");
      wrong |= Match(got, set) ? 0 : kEngine;
    }
    return wrong;
  }

  // Remaps the address between the handle sets `batches` times, settling
  // after each, and counts the batches with a page still on the old one.
  Old Remaps(int batches, int& set) {
    Old old;
    for (int batch = 0; batch < batches; ++batch) {
      set = (set + 1) % kSets;
      Check(hipMemUnmap(address, kPages * kPage), "unmap");
      Map(set);
      RemapBarrier::NoteRemap();
      // False only when an installed flush failed: counted in unflushed.
      (void)RemapBarrier::Settle();
      const int wrong = Reads(set);
      old.batches += wrong ? 1 : 0;
      old.engine += (wrong & kEngine) ? 1 : 0;
    }
    return old;
  }
};

// Starts this binary with --busy and waits until its kernels run.
pid_t StartBusy(const char* self) {
  int ready[2];
  if (pipe(ready) != 0) {
    std::perror("pipe");
    std::exit(1);
  }
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_adddup2(&actions, ready[1], STDOUT_FILENO);
  posix_spawn_file_actions_addclose(&actions, ready[0]);
  char busy[] = "--busy";
  char seconds[] = "60";
  char* argv[] = {const_cast<char*>(self), busy, seconds, nullptr};
  pid_t child = 0;
  const int spawned =
      posix_spawn(&child, self, &actions, nullptr, argv, environ);
  posix_spawn_file_actions_destroy(&actions);
  close(ready[1]);
  if (spawned != 0) {
    std::fprintf(stderr, "cannot start the busy process: %s\n",
                 std::strerror(spawned));
    std::exit(1);
  }
  char byte = 0;
  if (read(ready[0], &byte, 1) != 1) {
    std::fprintf(stderr, "the busy process did not start its kernels\n");
    std::exit(1);
  }
  close(ready[0]);
  return child;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 3 && std::strcmp(argv[1], "--busy") == 0)
    return Busy(std::atof(argv[2]));
  Check(hipSetDevice(0), "device");
  int vmm = 0;
  Check(hipDeviceGetAttribute(
            &vmm, hipDeviceAttributeVirtualMemoryManagementSupported, 0),
        "attribute");
  if (vmm == 0) {
    std::puts("mem_remap_barrier_device_test: no virtual memory; skipped");
    return 0;
  }
  gufo::mem::InstallRemapBarrierDevice();
  Pages pages;
  int set = 0;
  // The address's translations are in use.
  if (pages.Reads(set) != 0) {
    std::fprintf(stderr, "the first mapping reads wrong\n");
    return 1;
  }
  const pid_t busy = StartBusy("/proc/self/exe");
  // Without the flush the idle time decides; on a driver with the gap most
  // batches stay old beside the busy process. Reported, not checked: a
  // fixed driver settles them as well.
  RemapBarrier::SetTranslationFlush(nullptr);
  const Old idle_only = pages.Remaps(kBatches, set);
  gufo::mem::InstallRemapBarrierDevice();
  const auto before = RemapBarrier::Stats();
  const Old flushed = pages.Remaps(kBatches, set);
  const auto after = RemapBarrier::Stats();
  // Without it both phases ran on an otherwise idle GPU. A child that ended
  // is reaped here, and its PID is not signalled.
  pid_t reaped = 0;
  while ((reaped = waitpid(busy, nullptr, WNOHANG)) == -1 && errno == EINTR) {
  }
  const bool busy_ran = reaped == 0;
  if (busy_ran) {
    kill(busy, SIGTERM);
    waitpid(busy, nullptr, 0);
  }
  if (!busy_ran) {
    std::fprintf(stderr, "the busy process ended early\n");
    return 1;
  }
  std::printf(
      "mem_remap_barrier_device_test: beside a busy process, %d of %d "
      "batches old after the idle time alone (copy engine: %d), %d of %d "
      "with the flush (copy engine: %d)%s\n",
      idle_only.batches, kBatches, idle_only.engine, flushed.batches, kBatches,
      flushed.engine, pages.engine ? "" : "; no copy engine");
  assert(after.unflushed == before.unflushed);
  assert(flushed.batches == 0);
  std::puts("mem_remap_barrier_device_test: ok");
  return 0;
}
