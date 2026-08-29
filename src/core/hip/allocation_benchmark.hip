#include "src/core/diagnostics/allocation_benchmark.h"

#if defined(ENGINE_ENABLE_HIP)

#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif

#ifndef GUFO_VERSION
#define GUFO_VERSION "development"
#endif

namespace gufo::diagnostics {
namespace {

constexpr std::size_t kSafetyHeadroom = 4ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr std::uint8_t kPatternSeed = 37U;
constexpr std::array<std::string_view, 5> kPathNames = {
    "hip_malloc", "hip_malloc_managed", "hip_host_malloc_mapped",
    "file_mmap_host_register", "hip_malloc_async"};

enum class AllocationKind {
  kHipMalloc,
  kHipMallocManaged,
  kHipHostMallocMapped,
  kFileMmapHostRegister,
  kHipMallocAsync,
};

struct FaultSnapshot {
  std::uint64_t minor{0};
  std::uint64_t major{0};
};

FaultSnapshot ReadFaults() {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    return {};
  }
  return {
      .minor = static_cast<std::uint64_t>(usage.ru_minflt),
      .major = static_cast<std::uint64_t>(usage.ru_majflt),
  };
}

AllocationFaultDelta FaultDelta(FaultSnapshot before, FaultSnapshot after) {
  return {
      .minor = after.minor >= before.minor ? after.minor - before.minor : 0,
      .major = after.major >= before.major ? after.major - before.major : 0,
  };
}

std::uint64_t ReadMeminfoValue(std::string_view requested_key) {
  std::ifstream input{"/proc/meminfo"};
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream fields(line);
    std::string key;
    std::uint64_t value = 0;
    std::string unit;
    if (!(fields >> key >> value) || key != requested_key) {
      continue;
    }
    fields >> unit;
    return unit == "kB" ? value * 1024ULL : value;
  }
  return 0;
}

std::uint64_t AvailableMemoryBytes() {
  return ReadMeminfoValue("MemAvailable:");
}

std::uint64_t PageSizeBytes() {
  const long page_size = sysconf(_SC_PAGESIZE);
  return page_size > 0 ? static_cast<std::uint64_t>(page_size) : 0;
}

std::string HipReason(hipError_t status) {
  return std::string(hipGetErrorName(status)) + ": " +
         hipGetErrorString(status);
}

bool IsUnsupported(hipError_t status) {
  return status == hipErrorNotSupported || status == hipErrorInvalidValue;
}

std::string MemoryTypeName(hipMemoryType type) {
  switch (type) {
    case hipMemoryTypeUnregistered:
      return "unregistered";
    case hipMemoryTypeHost:
      return "host";
    case hipMemoryTypeDevice:
      return "device";
    case hipMemoryTypeManaged:
      return "managed";
    case hipMemoryTypeArray:
      return "array";
    case hipMemoryTypeUnified:
      return "unified";
    default:
      return "unknown";
  }
}

std::uint64_t ExpectedChecksum(std::size_t size, std::uint8_t seed) {
  std::uint64_t cycle_sum = 0;
  for (std::uint32_t index = 0; index < 256U; ++index) {
    cycle_sum += static_cast<std::uint8_t>(index + seed);
  }
  const std::size_t full_cycles = size / 256U;
  const std::size_t remainder = size % 256U;
  std::uint64_t total =
      static_cast<std::uint64_t>(full_cycles) * cycle_sum;
  for (std::size_t index = 0; index < remainder; ++index) {
    total += static_cast<std::uint8_t>(index + seed);
  }
  return total;
}

void FillCpu(void* pointer, std::size_t size, std::uint8_t seed,
             std::size_t global_offset = 0) {
  auto* bytes = static_cast<std::uint8_t*>(pointer);
  for (std::size_t index = 0; index < size; ++index) {
    bytes[index] = static_cast<std::uint8_t>(global_offset + index + seed);
  }
}

std::uint64_t ChecksumCpu(const void* pointer, std::size_t size) {
  const auto* bytes = static_cast<const std::uint8_t*>(pointer);
  std::uint64_t sum = 0;
  for (std::size_t index = 0; index < size; ++index) {
    sum += bytes[index];
  }
  return sum;
}

__global__ void FillPatternKernel(std::uint8_t* data, std::size_t size,
                                  std::uint8_t seed) {
  const std::size_t thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index = thread; index < size; index += stride) {
    data[index] = static_cast<std::uint8_t>(index + seed);
  }
}

__global__ void ChecksumKernel(const std::uint8_t* data, std::size_t size,
                               unsigned long long* checksum) {
  const std::size_t thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  unsigned long long local = 0;
  for (std::size_t index = thread; index < size; index += stride) {
    local += data[index];
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    local += __shfl_down(local, offset, 32);
  }
  __shared__ unsigned long long wave_sums[8];
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int wave = threadIdx.x >> 5U;
  if (lane == 0U) {
    wave_sums[wave] = local;
  }
  __syncthreads();
  if (wave == 0U) {
    local = lane < 8U ? wave_sums[lane] : 0ULL;
    for (int offset = 16; offset > 0; offset /= 2) {
      local += __shfl_down(local, offset, 32);
    }
    if (lane == 0U) {
      atomicAdd(checksum, local);
    }
  }
}

class Events {
public:
  Events() {
    valid_ = hipEventCreate(&start_) == hipSuccess &&
             hipEventCreate(&stop_) == hipSuccess;
  }
  ~Events() {
    if (start_ != nullptr) {
      (void)hipEventDestroy(start_);
    }
    if (stop_ != nullptr) {
      (void)hipEventDestroy(stop_);
    }
  }

  Events(const Events&) = delete;
  Events& operator=(const Events&) = delete;

  [[nodiscard]] bool valid() const noexcept { return valid_; }
  [[nodiscard]] hipEvent_t start() const noexcept { return start_; }
  [[nodiscard]] hipEvent_t stop() const noexcept { return stop_; }

private:
  hipEvent_t start_{nullptr};
  hipEvent_t stop_{nullptr};
  bool valid_{false};
};

class AllocationResource {
public:
  ~AllocationResource() { Cleanup(); }

  AllocationResource() = default;
  AllocationResource(const AllocationResource&) = delete;
  AllocationResource& operator=(const AllocationResource&) = delete;

  void Cleanup() noexcept {
    if (async_allocation_ && allocation_ != nullptr && stream_ != nullptr) {
      (void)hipFreeAsync(allocation_, stream_);
      (void)hipStreamSynchronize(stream_);
    } else if (hip_host_allocation_ && host_ != nullptr) {
      (void)hipHostFree(host_);
    } else if (owns_device_ && allocation_ != nullptr) {
      (void)hipFree(allocation_);
    }
    if (host_registered_ && host_ != nullptr) {
      (void)hipHostUnregister(host_);
    }
    if (mapping_ != nullptr && mapping_size_ != 0) {
      (void)munmap(mapping_, mapping_size_);
    }
    if (fd_ >= 0) {
      (void)close(fd_);
    }
    if (stream_ != nullptr) {
      (void)hipStreamDestroy(stream_);
    }
    host_ = nullptr;
    device_ = nullptr;
    allocation_ = nullptr;
    mapping_ = nullptr;
    mapping_size_ = 0;
    fd_ = -1;
    stream_ = nullptr;
    owns_device_ = false;
    hip_host_allocation_ = false;
    host_registered_ = false;
    async_allocation_ = false;
  }

  void* host_{nullptr};
  void* device_{nullptr};
  void* allocation_{nullptr};
  void* mapping_{nullptr};
  std::size_t mapping_size_{0};
  int fd_{-1};
  hipStream_t stream_{nullptr};
  bool cpu_accessible_{false};
  bool owns_device_{false};
  bool hip_host_allocation_{false};
  bool host_registered_{false};
  bool async_allocation_{false};
  bool managed_{false};
};

AllocationPhaseResult Phase(std::string name, std::string operation,
                            std::string unit = "ms") {
  AllocationPhaseResult phase;
  phase.name = std::move(name);
  phase.operation = std::move(operation);
  phase.unit = std::move(unit);
  return phase;
}

template<typename Function>
hipError_t MeasureHipCall(AllocationPhaseResult* phase, Function&& function) {
  const FaultSnapshot before_faults = ReadFaults();
  const auto before = std::chrono::steady_clock::now();
  const hipError_t status = function();
  const auto after = std::chrono::steady_clock::now();
  phase->raw_repetitions.push_back(
      std::chrono::duration<double, std::milli>(after - before).count());
  phase->faults = FaultDelta(before_faults, ReadFaults());
  if (status != hipSuccess) {
    phase->status = IsUnsupported(status) ? "unsupported" : "unavailable";
    phase->reason = HipReason(status);
  }
  ComputeAllocationPhaseStatistics(phase);
  return status;
}

template<typename Function>
void MeasureCpuCall(AllocationPhaseResult* phase, Function&& function) {
  const FaultSnapshot before_faults = ReadFaults();
  const auto before = std::chrono::steady_clock::now();
  function();
  const auto after = std::chrono::steady_clock::now();
  phase->raw_repetitions.push_back(
      std::chrono::duration<double, std::milli>(after - before).count());
  phase->faults = FaultDelta(before_faults, ReadFaults());
  ComputeAllocationPhaseStatistics(phase);
}

std::string MincoreEvidence(void* pointer, std::size_t size) {
  if (pointer == nullptr || size == 0) {
    return "mincore unavailable: no host mapping";
  }
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    return "mincore unavailable: page size query failed";
  }
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  const auto page = static_cast<std::uintptr_t>(page_size);
  const auto aligned = address & ~(page - 1U);
  const std::size_t prefix = address - aligned;
  if (size > std::numeric_limits<std::size_t>::max() - prefix) {
    return "mincore unavailable: range overflow";
  }
  const std::size_t page_count = (prefix + size + page - 1U) / page;
  constexpr std::size_t kChunkPages = 1ULL << 20U;
  std::vector<unsigned char> states(
      std::min(page_count, kChunkPages), 0U);
  std::size_t resident = 0;
  std::size_t completed = 0;
  while (completed < page_count) {
    const std::size_t pages =
        std::min(states.size(), page_count - completed);
    void* chunk = reinterpret_cast<void*>(aligned + completed * page);
    if (mincore(chunk, pages * page, states.data()) != 0) {
      return "mincore unavailable for this allocation";
    }
    resident += static_cast<std::size_t>(std::count_if(
        states.begin(), states.begin() + pages,
        [](unsigned char value) { return (value & 1U) != 0U; }));
    completed += pages;
  }
  return std::to_string(resident) + "/" + std::to_string(page_count) +
         " pages resident";
}

std::uint32_t KernelBlocks() {
  hipDeviceProp_t properties{};
  if (hipGetDeviceProperties(&properties, 0) != hipSuccess ||
      properties.multiProcessorCount <= 0) {
    return 160U;
  }
  return static_cast<std::uint32_t>(
      std::max(1, properties.multiProcessorCount * 8));
}

hipError_t LaunchFill(void* device, std::size_t size, hipStream_t stream) {
  hipLaunchKernelGGL(FillPatternKernel, dim3(KernelBlocks()), dim3(256), 0,
                     stream, static_cast<std::uint8_t*>(device), size,
                     kPatternSeed);
  return hipGetLastError();
}

hipError_t LaunchChecksum(const void* device, std::size_t size,
                          unsigned long long* output, hipStream_t stream) {
  const auto clear =
      hipMemsetAsync(output, 0, sizeof(unsigned long long), stream);
  if (clear != hipSuccess) {
    return clear;
  }
  hipLaunchKernelGGL(ChecksumKernel, dim3(KernelBlocks()), dim3(256), 0, stream,
                     static_cast<const std::uint8_t*>(device), size, output);
  return hipGetLastError();
}

bool MeasureGpuOperation(AllocationPhaseResult* throughput,
                         AllocationPhaseResult* synchronization,
                         std::size_t bytes, hipStream_t stream,
                         const std::function<hipError_t()>& launch) {
  Events events;
  if (!events.valid()) {
    throughput->status = "unavailable";
    throughput->reason = "hipEventCreate failed";
    synchronization->status = "unavailable";
    synchronization->reason = throughput->reason;
    return false;
  }
  const auto start_status = hipEventRecord(events.start(), stream);
  const auto launch_status =
      start_status == hipSuccess ? launch() : start_status;
  const auto stop_status =
      launch_status == hipSuccess ? hipEventRecord(events.stop(), stream)
                                  : launch_status;
  const FaultSnapshot before_faults = ReadFaults();
  const auto sync_begin = std::chrono::steady_clock::now();
  const auto sync_status =
      stop_status == hipSuccess ? hipEventSynchronize(events.stop())
                                : stop_status;
  const auto sync_end = std::chrono::steady_clock::now();
  synchronization->raw_repetitions.push_back(
      std::chrono::duration<double, std::milli>(sync_end - sync_begin).count());
  const auto synchronization_faults =
      FaultDelta(before_faults, ReadFaults());
  synchronization->faults.minor += synchronization_faults.minor;
  synchronization->faults.major += synchronization_faults.major;
  if (sync_status != hipSuccess) {
    throughput->status =
        IsUnsupported(sync_status) ? "unsupported" : "unavailable";
    throughput->reason = HipReason(sync_status);
    synchronization->status = throughput->status;
    synchronization->reason = throughput->reason;
    ComputeAllocationPhaseStatistics(synchronization);
    return false;
  }
  float milliseconds = 0.0F;
  const auto elapsed_status =
      hipEventElapsedTime(&milliseconds, events.start(), events.stop());
  if (elapsed_status != hipSuccess || milliseconds <= 0.0F) {
    throughput->status = "unavailable";
    throughput->reason = elapsed_status == hipSuccess
                             ? "HIP event reported a non-positive duration"
                             : HipReason(elapsed_status);
    ComputeAllocationPhaseStatistics(synchronization);
    return false;
  }
  const double seconds = static_cast<double>(milliseconds) / 1000.0;
  throughput->raw_repetitions.push_back(
      static_cast<double>(bytes) / 1.0e9 / seconds);
  ComputeAllocationPhaseStatistics(throughput);
  ComputeAllocationPhaseStatistics(synchronization);
  return true;
}

void DescribePointer(AllocationPathResult* result, void* device) {
  hipPointerAttribute_t attributes{};
  const auto status = hipPointerGetAttributes(&attributes, device);
  if (status != hipSuccess) {
    result->pointer_memory_type = "unavailable: " + HipReason(status);
    return;
  }
  result->pointer_memory_type = MemoryTypeName(attributes.type);
  result->managed = attributes.isManaged != 0;
}

void MarkCreationFailure(AllocationPathResult* result, hipError_t status) {
  result->status = IsUnsupported(status) ? "unsupported" : "unavailable";
  result->reason = HipReason(status);
  result->checksum_verified = false;
}

bool CreateResource(AllocationKind kind, std::size_t size,
                    AllocationResource* resource,
                    AllocationPathResult* result) {
  auto allocation = Phase("allocation", "allocate untouched working set");
  hipError_t status = hipSuccess;
  switch (kind) {
    case AllocationKind::kHipMalloc:
      result->allocation_api = "hipMalloc";
      result->flags = "none";
      result->placement_inference =
          "HIP reports device memory. On this integrated APU that does not "
          "identify a distinct physical memory pool or prove residency.";
      status = MeasureHipCall(&allocation, [&] {
        return hipMalloc(&resource->allocation_, size);
      });
      resource->device_ = resource->allocation_;
      resource->owns_device_ = status == hipSuccess;
      break;
    case AllocationKind::kHipMallocManaged:
      result->allocation_api = "hipMallocManaged";
      result->flags = "hipMemAttachGlobal";
      result->placement_inference =
          "HIP reports managed memory. Pointer attributes and successful "
          "prefetch do not reveal physical page placement.";
      status = MeasureHipCall(&allocation, [&] {
        return hipMallocManaged(&resource->allocation_, size,
                                hipMemAttachGlobal);
      });
      resource->host_ = resource->allocation_;
      resource->device_ = resource->allocation_;
      resource->owns_device_ = status == hipSuccess;
      resource->cpu_accessible_ = status == hipSuccess;
      resource->managed_ = status == hipSuccess;
      break;
    case AllocationKind::kHipHostMallocMapped:
      result->allocation_api = "hipHostMalloc + hipHostGetDevicePointer";
      result->flags = "hipHostMallocMapped";
      result->placement_inference =
          "HIP reports host memory mapped into GPU virtual address space. "
          "This does not expose DRAM bank or cache residency.";
      status = MeasureHipCall(&allocation, [&] {
        return hipHostMalloc(&resource->host_, size, hipHostMallocMapped);
      });
      resource->hip_host_allocation_ = status == hipSuccess;
      resource->cpu_accessible_ = status == hipSuccess;
      if (status == hipSuccess) {
        auto alias =
            Phase("device_alias", "hipHostGetDevicePointer", "ms");
        status = MeasureHipCall(&alias, [&] {
          return hipHostGetDevicePointer(&resource->device_, resource->host_,
                                         0);
        });
        result->phases.push_back(std::move(alias));
      }
      break;
    case AllocationKind::kFileMmapHostRegister: {
      result->allocation_api =
          "memfd_create + mmap(MAP_SHARED) + hipHostRegister";
      result->flags =
          "PROT_READ|PROT_WRITE, MAP_SHARED, hipHostRegisterMapped";
      result->placement_inference =
          "mincore reports Linux page-cache residency for the memfd mapping; "
          "HIP exposes only a mapped host alias and no physical GPU residency.";
      const FaultSnapshot before_faults = ReadFaults();
      const auto begin = std::chrono::steady_clock::now();
      resource->fd_ = static_cast<int>(
          syscall(SYS_memfd_create, "gufo-allocation-benchmark", MFD_CLOEXEC));
      if (resource->fd_ >= 0 &&
          ftruncate(resource->fd_, static_cast<off_t>(size)) == 0) {
        resource->mapping_ =
            mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                 resource->fd_, 0);
        if (resource->mapping_ == MAP_FAILED) {
          resource->mapping_ = nullptr;
        }
      }
      const auto end = std::chrono::steady_clock::now();
      allocation.raw_repetitions.push_back(
          std::chrono::duration<double, std::milli>(end - begin).count());
      allocation.faults = FaultDelta(before_faults, ReadFaults());
      if (resource->fd_ < 0 || resource->mapping_ == nullptr) {
        allocation.status = "unavailable";
        allocation.reason = "memfd_create/ftruncate/mmap failed";
        status = hipErrorOutOfMemory;
      } else {
        resource->host_ = resource->mapping_;
        resource->mapping_size_ = size;
        resource->cpu_accessible_ = true;
      }
      ComputeAllocationPhaseStatistics(&allocation);
      if (status == hipSuccess) {
        auto registration = Phase(
            "registration",
            "hipHostRegister(mapped) + hipHostGetDevicePointer");
        status = MeasureHipCall(&registration, [&] {
          const auto register_status =
              hipHostRegister(resource->host_, size, hipHostRegisterMapped);
          if (register_status != hipSuccess) {
            return register_status;
          }
          resource->host_registered_ = true;
          const auto pointer_status = hipHostGetDevicePointer(
              &resource->device_, resource->host_, 0);
          if (pointer_status != hipSuccess) {
            (void)hipHostUnregister(resource->host_);
            resource->host_registered_ = false;
          }
          return pointer_status;
        });
        result->phases.push_back(std::move(registration));
      }
      break;
    }
    case AllocationKind::kHipMallocAsync:
      result->allocation_api = "hipMallocAsync";
      result->flags = "default device memory pool";
      result->placement_inference =
          "hipMallocAsync is measured as allocation/reuse policy only; it is "
          "not claimed as a distinct physical placement from hipMalloc.";
      status = hipStreamCreateWithFlags(&resource->stream_, hipStreamNonBlocking);
      if (status == hipSuccess) {
        status = MeasureHipCall(&allocation, [&] {
          const auto allocate_status =
              hipMallocAsync(&resource->allocation_, size, resource->stream_);
          if (allocate_status != hipSuccess) {
            return allocate_status;
          }
          return hipStreamSynchronize(resource->stream_);
        });
      } else {
        allocation.status = "unavailable";
        allocation.reason = HipReason(status);
      }
      resource->device_ = resource->allocation_;
      resource->async_allocation_ = status == hipSuccess;
      break;
  }
  result->phases.insert(result->phases.begin(), std::move(allocation));
  if (status != hipSuccess || resource->device_ == nullptr) {
    MarkCreationFailure(result, status);
    return false;
  }
  if (resource->stream_ == nullptr &&
      hipStreamCreateWithFlags(&resource->stream_, hipStreamNonBlocking) !=
          hipSuccess) {
    result->status = "unavailable";
    result->reason = "hipStreamCreateWithFlags failed";
    return false;
  }
  DescribePointer(result, resource->device_);
  return true;
}

bool IsSetupPhase(std::string_view name) {
  return name == "allocation" || name == "device_alias" ||
         name == "registration" || name == "teardown";
}

void MergePhaseSample(AllocationPathResult* result,
                      const AllocationPhaseResult& sample) {
  const auto phase = std::ranges::find_if(
      result->phases, [&sample](const AllocationPhaseResult& candidate) {
        return candidate.name == sample.name;
      });
  if (phase == result->phases.end()) {
    result->phases.push_back(sample);
    return;
  }
  phase->raw_repetitions.insert(phase->raw_repetitions.end(),
                                sample.raw_repetitions.begin(),
                                sample.raw_repetitions.end());
  phase->faults.minor += sample.faults.minor;
  phase->faults.major += sample.faults.major;
  if (sample.status != "completed") {
    phase->status = sample.status;
    phase->reason = sample.reason;
  }
  ComputeAllocationPhaseStatistics(&*phase);
}

void AddUnsupportedCpuTouch(AllocationPathResult* result) {
  auto phase =
      Phase("first_cpu_touch", "direct CPU write to allocation", "ms");
  phase.status = "unsupported";
  phase.reason =
      "device allocation has no supported direct CPU pointer; explicit "
      "hipMemcpy phases are reported separately";
  result->phases.push_back(std::move(phase));
}

bool VerifyGpuChecksum(AllocationResource* resource, std::size_t size,
                       unsigned long long* device_checksum,
                       std::uint64_t* checksum) {
  if (LaunchChecksum(resource->device_, size, device_checksum,
                     resource->stream_) != hipSuccess ||
      hipStreamSynchronize(resource->stream_) != hipSuccess) {
    return false;
  }
  unsigned long long host_checksum = 0;
  if (hipMemcpy(&host_checksum, device_checksum, sizeof(host_checksum),
                hipMemcpyDeviceToHost) != hipSuccess) {
    return false;
  }
  *checksum = host_checksum;
  return true;
}

void AddManagedAdvicePhases(AllocationResource* resource, std::size_t size,
                            AllocationPathResult* result) {
  auto advice = Phase("memory_advice",
                      "hipMemAdviseSetPreferredLocation(device 0)");
  (void)MeasureHipCall(&advice, [&] {
    return hipMemAdvise(resource->device_, size,
                        hipMemAdviseSetPreferredLocation, 0);
  });
  result->phases.push_back(std::move(advice));

  auto prefetch = Phase("prefetch_to_gpu", "hipMemPrefetchAsync(device 0)");
  (void)MeasureHipCall(&prefetch, [&] {
    const auto status =
        hipMemPrefetchAsync(resource->device_, size, 0, resource->stream_);
    return status == hipSuccess ? hipStreamSynchronize(resource->stream_)
                                : status;
  });
  result->phases.push_back(std::move(prefetch));

  auto cpu_prefetch =
      Phase("prefetch_to_cpu", "hipMemPrefetchAsync(hipCpuDeviceId)");
  (void)MeasureHipCall(&cpu_prefetch, [&] {
    const auto status = hipMemPrefetchAsync(resource->device_, size,
                                            hipCpuDeviceId, resource->stream_);
    return status == hipSuccess ? hipStreamSynchronize(resource->stream_)
                                : status;
  });
  result->phases.push_back(std::move(cpu_prefetch));
}

void AddAsyncReusePhase(std::size_t size, AllocationResource* resource,
                        const AllocationBenchmarkOptions& options,
                        AllocationPathResult* result) {
  auto reuse =
      Phase("allocation_reuse", "hipMallocAsync + hipFreeAsync", "ms");
  for (std::uint32_t repetition = 0; repetition < options.repetitions;
       ++repetition) {
    void* replacement = nullptr;
    const auto begin = std::chrono::steady_clock::now();
    auto status =
        hipFreeAsync(resource->allocation_, resource->stream_);
    if (status == hipSuccess) {
      resource->allocation_ = nullptr;
      resource->device_ = nullptr;
      status =
          hipMallocAsync(&replacement, size, resource->stream_);
    }
    if (status == hipSuccess) {
      resource->allocation_ = replacement;
      resource->device_ = replacement;
    }
    if (status == hipSuccess) {
      status = hipStreamSynchronize(resource->stream_);
    }
    const auto end = std::chrono::steady_clock::now();
    if (status != hipSuccess) {
      reuse.status = IsUnsupported(status) ? "unsupported" : "unavailable";
      reuse.reason = HipReason(status);
      break;
    }
    reuse.raw_repetitions.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  ComputeAllocationPhaseStatistics(&reuse);
  result->phases.push_back(std::move(reuse));
}

void AddExplicitCopyPhases(AllocationResource* resource, std::size_t size,
                           const AllocationBenchmarkOptions& options,
                           AllocationPathResult* result) {
  const std::size_t chunk =
      std::max<std::size_t>(1, std::min(size, options.transfer_chunk_bytes));
  void* device_staging = nullptr;
  void* host_staging = nullptr;
  const auto device_status = hipMalloc(&device_staging, chunk);
  const auto host_status =
      hipHostMalloc(&host_staging, chunk, hipHostMallocDefault);
  if (device_status != hipSuccess || host_status != hipSuccess) {
    if (device_staging != nullptr) {
      (void)hipFree(device_staging);
    }
    if (host_staging != nullptr) {
      (void)hipHostFree(host_staging);
    }
    auto copy =
        Phase("explicit_copy", "bounded chunked hipMemcpyAsync", "GB/s");
    copy.status = "unavailable";
    copy.reason = "failed to allocate bounded transfer staging buffers";
    result->phases.push_back(std::move(copy));
    return;
  }
  std::memset(host_staging, 0, chunk);

  auto h2d = Phase("explicit_h2d_copy",
                   "chunked hipMemcpyAsync(host to device)", "GB/s");
  auto h2d_sync =
      Phase("explicit_h2d_sync", "hipEventSynchronize after H2D", "ms");
  for (std::uint32_t repetition = 0; repetition < options.repetitions;
       ++repetition) {
    (void)MeasureGpuOperation(
        &h2d, &h2d_sync, size, resource->stream_, [&] {
          for (std::size_t offset = 0; offset < size; offset += chunk) {
            const std::size_t bytes = std::min(chunk, size - offset);
            const void* source =
                resource->cpu_accessible_
                    ? static_cast<const std::uint8_t*>(resource->host_) + offset
                    : host_staging;
            const auto status =
                hipMemcpyAsync(device_staging, source, bytes,
                               hipMemcpyHostToDevice, resource->stream_);
            if (status != hipSuccess) {
              return status;
            }
          }
          return hipSuccess;
        });
  }
  result->phases.push_back(std::move(h2d));
  result->phases.push_back(std::move(h2d_sync));

  auto d2h = Phase("explicit_d2h_copy",
                   "chunked hipMemcpyAsync(device to host)", "GB/s");
  auto d2h_sync =
      Phase("explicit_d2h_sync", "hipEventSynchronize after D2H", "ms");
  for (std::uint32_t repetition = 0; repetition < options.repetitions;
       ++repetition) {
    (void)MeasureGpuOperation(
        &d2h, &d2h_sync, size, resource->stream_, [&] {
          for (std::size_t offset = 0; offset < size; offset += chunk) {
            const std::size_t bytes = std::min(chunk, size - offset);
            const void* source =
                static_cast<const std::uint8_t*>(resource->device_) + offset;
            const auto status =
                hipMemcpyAsync(host_staging, source, bytes,
                               hipMemcpyDeviceToHost, resource->stream_);
            if (status != hipSuccess) {
              return status;
            }
          }
          return hipSuccess;
        });
  }
  result->phases.push_back(std::move(d2h));
  result->phases.push_back(std::move(d2h_sync));

  (void)hipHostFree(host_staging);
  (void)hipFree(device_staging);
}

void AddCpuViaCopyPhase(AllocationResource* resource, std::size_t size,
                        const AllocationBenchmarkOptions& options,
                        AllocationPathResult* result) {
  const std::size_t chunk =
      std::max<std::size_t>(1, std::min(size, options.transfer_chunk_bytes));
  void* staging = nullptr;
  const auto status =
      hipHostMalloc(&staging, chunk, hipHostMallocDefault);
  auto phase = Phase("warm_cpu_read_via_d2h",
                     "hipMemcpy D2H plus CPU checksum", "GB/s");
  if (status != hipSuccess) {
    phase.status = "unavailable";
    phase.reason = HipReason(status);
    result->phases.push_back(std::move(phase));
    return;
  }
  for (std::uint32_t repetition = 0; repetition < options.repetitions;
       ++repetition) {
    const FaultSnapshot before_faults = ReadFaults();
    const auto begin = std::chrono::steady_clock::now();
    std::uint64_t checksum = 0;
    for (std::size_t offset = 0; offset < size; offset += chunk) {
      const std::size_t bytes = std::min(chunk, size - offset);
      const auto copy_status = hipMemcpy(
          staging, static_cast<const std::uint8_t*>(resource->device_) + offset,
          bytes, hipMemcpyDeviceToHost);
      if (copy_status != hipSuccess) {
        phase.status = "unavailable";
        phase.reason = HipReason(copy_status);
        break;
      }
      checksum += ChecksumCpu(staging, bytes);
    }
    const auto end = std::chrono::steady_clock::now();
    const auto faults = FaultDelta(before_faults, ReadFaults());
    phase.faults.minor += faults.minor;
    phase.faults.major += faults.major;
    if (phase.status != "completed") {
      break;
    }
    result->cpu_checksum = checksum;
    const double seconds =
        std::chrono::duration<double>(end - begin).count();
    phase.raw_repetitions.push_back(
        static_cast<double>(size) / 1.0e9 / seconds);
  }
  ComputeAllocationPhaseStatistics(&phase);
  result->phases.push_back(std::move(phase));
  (void)hipHostFree(staging);
}

AllocationPathResult RunPath(AllocationKind kind, std::string_view path_name,
                             std::size_t size,
                             const AllocationBenchmarkOptions& options) {
  AllocationPathResult result;
  result.path_name = std::string(path_name);
  result.working_set_bytes = size;
  result.expected_checksum = ExpectedChecksum(size, kPatternSeed);

  const std::uint64_t available = AvailableMemoryBytes();
  if (available != 0 &&
      (size > std::numeric_limits<std::size_t>::max() - kSafetyHeadroom ||
       available < size + kSafetyHeadroom)) {
    result.status = "unavailable";
    result.reason = "insufficient MemAvailable for requested working set plus "
                    "4 GiB safety headroom; size was not reduced";
    result.allocation_api = std::string(path_name);
    result.placement_inference =
        "No allocation was attempted after the bounded memory safety check.";
    return result;
  }

  std::vector<AllocationPhaseResult> setup_samples;
  for (std::uint32_t repetition = 1; repetition < options.repetitions;
       ++repetition) {
    AllocationResource sample_resource;
    AllocationPathResult sample_result;
    if (!CreateResource(kind, size, &sample_resource, &sample_result)) {
      sample_result.path_name = result.path_name;
      sample_result.working_set_bytes = size;
      sample_result.expected_checksum = result.expected_checksum;
      return sample_result;
    }
    auto sample_teardown =
        Phase("teardown", "release allocation and mapping", "ms");
    MeasureCpuCall(&sample_teardown,
                   [&] { sample_resource.Cleanup(); });
    sample_result.phases.push_back(std::move(sample_teardown));
    for (auto& phase : sample_result.phases) {
      if (IsSetupPhase(phase.name)) {
        setup_samples.push_back(std::move(phase));
      }
    }
  }

  AllocationResource resource;
  if (!CreateResource(kind, size, &resource, &result)) {
    return result;
  }
  for (const auto& phase : setup_samples) {
    if (phase.name != "teardown") {
      MergePhaseSample(&result, phase);
    }
  }
  const std::string before_residency =
      MincoreEvidence(resource.host_, size);

  unsigned long long* device_checksum = nullptr;
  if (hipMalloc(&device_checksum, sizeof(unsigned long long)) != hipSuccess) {
    result.status = "unavailable";
    result.reason = "failed to allocate GPU checksum scalar";
    return result;
  }

  if (resource.cpu_accessible_) {
    auto first_cpu =
        Phase("first_cpu_touch", "seed every byte from CPU", "ms");
    MeasureCpuCall(&first_cpu,
                   [&] { FillCpu(resource.host_, size, kPatternSeed); });
    result.phases.push_back(std::move(first_cpu));
    result.cpu_checksum = ChecksumCpu(resource.host_, size);
  } else {
    AddUnsupportedCpuTouch(&result);
    auto first_gpu_write =
        Phase("first_gpu_touch_write", "seed every byte from GPU", "GB/s");
    auto first_gpu_sync =
        Phase("first_gpu_touch_sync", "hipEventSynchronize", "ms");
    (void)MeasureGpuOperation(
        &first_gpu_write, &first_gpu_sync, size, resource.stream_,
        [&] { return LaunchFill(resource.device_, size, resource.stream_); });
    result.phases.push_back(std::move(first_gpu_write));
    result.phases.push_back(std::move(first_gpu_sync));
  }

  auto first_gpu_read =
      Phase("first_gpu_touch_read", "GPU checksum over every byte", "GB/s");
  auto first_gpu_read_sync =
      Phase("first_gpu_touch_read_sync", "hipEventSynchronize", "ms");
  (void)MeasureGpuOperation(
      &first_gpu_read, &first_gpu_read_sync, size, resource.stream_, [&] {
        return LaunchChecksum(resource.device_, size, device_checksum,
                              resource.stream_);
      });
  result.phases.push_back(std::move(first_gpu_read));
  result.phases.push_back(std::move(first_gpu_read_sync));
  (void)VerifyGpuChecksum(&resource, size, device_checksum,
                          &result.gpu_checksum);

  if (resource.cpu_accessible_) {
    auto warm_cpu_read =
        Phase("warm_cpu_read", "CPU checksum over every byte", "GB/s");
    for (std::uint32_t warmup = 0; warmup < options.warmup; ++warmup) {
      result.cpu_checksum = ChecksumCpu(resource.host_, size);
    }
    for (std::uint32_t repetition = 0; repetition < options.repetitions;
         ++repetition) {
      const FaultSnapshot before_faults = ReadFaults();
      const auto begin = std::chrono::steady_clock::now();
      result.cpu_checksum = ChecksumCpu(resource.host_, size);
      const auto end = std::chrono::steady_clock::now();
      const auto faults = FaultDelta(before_faults, ReadFaults());
      warm_cpu_read.faults.minor += faults.minor;
      warm_cpu_read.faults.major += faults.major;
      const double seconds =
          std::chrono::duration<double>(end - begin).count();
      warm_cpu_read.raw_repetitions.push_back(
          static_cast<double>(size) / 1.0e9 / seconds);
    }
    ComputeAllocationPhaseStatistics(&warm_cpu_read);
    result.phases.push_back(std::move(warm_cpu_read));

    auto warm_cpu_write =
        Phase("warm_cpu_write", "CPU seed write over every byte", "GB/s");
    for (std::uint32_t repetition = 0; repetition < options.repetitions;
         ++repetition) {
      const FaultSnapshot before_faults = ReadFaults();
      const auto begin = std::chrono::steady_clock::now();
      FillCpu(resource.host_, size, kPatternSeed);
      const auto end = std::chrono::steady_clock::now();
      const auto faults = FaultDelta(before_faults, ReadFaults());
      warm_cpu_write.faults.minor += faults.minor;
      warm_cpu_write.faults.major += faults.major;
      const double seconds =
          std::chrono::duration<double>(end - begin).count();
      warm_cpu_write.raw_repetitions.push_back(
          static_cast<double>(size) / 1.0e9 / seconds);
    }
    ComputeAllocationPhaseStatistics(&warm_cpu_write);
    result.phases.push_back(std::move(warm_cpu_write));
  } else {
    AddCpuViaCopyPhase(&resource, size, options, &result);
  }

  auto warm_gpu_read =
      Phase("warm_gpu_read", "GPU checksum over every byte", "GB/s");
  auto warm_gpu_read_sync =
      Phase("warm_gpu_read_sync", "hipEventSynchronize", "ms");
  for (std::uint32_t warmup = 0; warmup < options.warmup; ++warmup) {
    (void)LaunchChecksum(resource.device_, size, device_checksum,
                         resource.stream_);
    (void)hipStreamSynchronize(resource.stream_);
  }
  for (std::uint32_t repetition = 0; repetition < options.repetitions;
       ++repetition) {
    (void)MeasureGpuOperation(
        &warm_gpu_read, &warm_gpu_read_sync, size, resource.stream_, [&] {
          return LaunchChecksum(resource.device_, size, device_checksum,
                                resource.stream_);
        });
  }
  result.phases.push_back(std::move(warm_gpu_read));
  result.phases.push_back(std::move(warm_gpu_read_sync));

  auto warm_gpu_write =
      Phase("warm_gpu_write", "GPU seed write over every byte", "GB/s");
  auto warm_gpu_write_sync =
      Phase("warm_gpu_write_sync", "hipEventSynchronize", "ms");
  for (std::uint32_t repetition = 0; repetition < options.repetitions;
       ++repetition) {
    (void)MeasureGpuOperation(
        &warm_gpu_write, &warm_gpu_write_sync, size, resource.stream_,
        [&] { return LaunchFill(resource.device_, size, resource.stream_); });
  }
  result.phases.push_back(std::move(warm_gpu_write));
  result.phases.push_back(std::move(warm_gpu_write_sync));

  AddExplicitCopyPhases(&resource, size, options, &result);
  if (resource.managed_) {
    AddManagedAdvicePhases(&resource, size, &result);
  }
  if (kind == AllocationKind::kHipMallocAsync) {
    AddAsyncReusePhase(size, &resource, options, &result);
  }

  (void)LaunchFill(resource.device_, size, resource.stream_);
  (void)hipStreamSynchronize(resource.stream_);
  if (resource.cpu_accessible_) {
    result.cpu_checksum = ChecksumCpu(resource.host_, size);
  } else {
    AddCpuViaCopyPhase(&resource, size,
                       AllocationBenchmarkOptions{
                           .working_set_bytes = {size},
                           .warmup = 0,
                           .repetitions = 1,
                           .transfer_chunk_bytes =
                               options.transfer_chunk_bytes,
                       },
                       &result);
  }
  (void)VerifyGpuChecksum(&resource, size, device_checksum,
                          &result.gpu_checksum);
  result.checksum_verified =
      result.cpu_checksum == result.expected_checksum &&
      result.gpu_checksum == result.expected_checksum;
  if (!result.checksum_verified) {
    result.status = "unavailable";
    result.reason = "CPU/GPU deterministic checksum mismatch";
  }

  result.residency_evidence =
      "before first touch: " + before_residency +
      "; after measured phases: " + MincoreEvidence(resource.host_, size);

  auto teardown = Phase("teardown", "release allocation and mapping", "ms");
  MeasureCpuCall(&teardown, [&] { resource.Cleanup(); });
  result.phases.push_back(std::move(teardown));
  for (const auto& phase : setup_samples) {
    if (phase.name == "teardown") {
      MergePhaseSample(&result, phase);
    }
  }
  (void)hipFree(device_checksum);
  return result;
}

std::string CurrentIso8601Utc() {
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
  gmtime_r(&time, &utc);
  std::ostringstream output;
  output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return output.str();
}

}  // namespace

AllocationBenchmarkReport RunHipAllocationBenchmark(
    const AllocationBenchmarkOptions& options,
    const SystemInventory& inventory,
    const MachineFingerprint& fingerprint) {
  AllocationBenchmarkReport report;
  report.timestamp = CurrentIso8601Utc();
  report.engine_revision = GUFO_VERSION;
  report.options = options;
  report.inventory = inventory;
  report.fingerprint = fingerprint;
  report.fingerprint_id = fingerprint.fingerprint_id;
  report.page_size_bytes = PageSizeBytes();
  report.huge_pages_total = ReadMeminfoValue("HugePages_Total:");
  report.huge_page_size_bytes = ReadMeminfoValue("Hugepagesize:");
  report.limitations = {
      "HIP pointer attributes identify API-visible memory types, not physical "
      "DRAM banks or cache residency on an integrated APU.",
      "Linux mincore reports whether host-mapped pages are resident in the "
      "process mapping; it does not identify GPU cache residency.",
      "Minor and major faults are process-level getrusage deltas and may "
      "include unrelated runtime activity.",
      "No privileged performance counters, page migration counters, persistent "
      "huge-page settings, NUMA changes, or power-policy changes are used.",
      "The file-backed path uses an unnamed memfd mapping so artifacts contain "
      "no host path; end-to-end #125 separately measures the real GGUF files.",
  };

  int device_count = 0;
  const auto device_status = hipGetDeviceCount(&device_count);
  if (device_status != hipSuccess || device_count == 0) {
    for (const auto size : options.working_set_bytes) {
      for (const auto path_name : kPathNames) {
        AllocationPathResult path;
        path.path_name = path_name;
        path.allocation_api = "HIP runtime";
        path.status = "unavailable";
        path.reason = device_status == hipSuccess
                          ? "no HIP device found"
                          : HipReason(device_status);
        path.working_set_bytes = size;
        path.placement_inference = "No placement evidence is available.";
        report.paths.push_back(std::move(path));
      }
    }
  } else {
    for (const auto size : options.working_set_bytes) {
      report.paths.push_back(RunPath(AllocationKind::kHipMalloc, kPathNames[0],
                                     size, options));
      report.paths.push_back(RunPath(AllocationKind::kHipMallocManaged,
                                     kPathNames[1], size, options));
      report.paths.push_back(RunPath(AllocationKind::kHipHostMallocMapped,
                                     kPathNames[2], size, options));
      report.paths.push_back(RunPath(AllocationKind::kFileMmapHostRegister,
                                     kPathNames[3], size, options));
      report.paths.push_back(RunPath(AllocationKind::kHipMallocAsync,
                                     kPathNames[4], size, options));
    }
  }

  const std::size_t expected_count =
      options.working_set_bytes.size() * kPathNames.size();
  report.all_requested_paths_reported = report.paths.size() == expected_count;
  report.checksums_verified =
      std::ranges::all_of(report.paths, [](const AllocationPathResult& path) {
        return path.status == "completed"
                   ? path.checksum_verified
                   : !path.reason.empty();
      });
  report.phases_separated =
      std::ranges::all_of(report.paths, [](const AllocationPathResult& path) {
        if (path.status != "completed") {
          return !path.reason.empty();
        }
        const auto has_phase = [&path](std::string_view name) {
          return std::ranges::any_of(
              path.phases, [name](const AllocationPhaseResult& phase) {
                return phase.name == name;
              });
        };
        return has_phase("allocation") && has_phase("first_gpu_touch_read") &&
               has_phase("warm_gpu_read") && has_phase("warm_gpu_write") &&
               has_phase("teardown");
      });
  return report;
}

}  // namespace gufo::diagnostics

#endif  // defined(ENGINE_ENABLE_HIP)
