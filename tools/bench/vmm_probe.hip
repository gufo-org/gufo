// Probes HIP virtual memory management on gfx1151: reserve a large VA range,
// back it with physical chunks on demand, and check that kernels and captured
// graphs read chunks mapped after capture, that a chunk can be unmapped and
// remapped under a live graph, how long mapping takes, what the device heap
// reports, and that VMM-backed rows read as fast as hipMalloc memory.
//
//   tools/bench/build.sh vmm_probe && /tmp/vmm_probe [chunk_mib] [reserve_gib]
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess)
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

double Now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

__global__ void Fill(std::uint32_t* p, std::size_t n, std::uint32_t salt) {
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
       i < n; i += std::size_t{gridDim.x} * blockDim.x)
    p[i] = static_cast<std::uint32_t>(i) * 2654435761u ^ salt;
}

// Sums the words of [begin, end) as read through a device-side extent, the
// way captured attention reads only the rows its control block names.
__global__ void Sum(const std::uint32_t* p, const std::uint64_t* extent,
                    unsigned long long* out) {
  const std::size_t n = *extent;
  unsigned long long acc = 0;
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
       i < n; i += std::size_t{gridDim.x} * blockDim.x)
    acc += p[i];
  atomicAdd(out, acc);
}

__global__ void Stream(const uint4* p, std::size_t n, uint4* sink) {
  uint4 acc{};
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
       i < n; i += std::size_t{gridDim.x} * blockDim.x) {
    const uint4 v = p[i];
    acc.x ^= v.x;
    acc.y ^= v.y;
    acc.z ^= v.z;
    acc.w ^= v.w;
  }
  if (acc.x == 0x12345678u)
    sink[0] = acc;
}

unsigned long long Expected(std::size_t n, std::uint32_t salt) {
  unsigned long long s = 0;
  for (std::size_t i = 0; i < n; ++i)
    s += static_cast<std::uint32_t>(i) * 2654435761u ^ salt;
  return s;
}

double Bandwidth(const void* p, std::size_t bytes, uint4* sink) {
  const std::size_t n = bytes / sizeof(uint4);
  hipLaunchKernelGGL(Stream, dim3(1024), dim3(256), 0, nullptr,
                     static_cast<const uint4*>(p), n, sink);
  Check(hipDeviceSynchronize(), "warm");
  const double t0 = Now();
  constexpr int kReps = 10;
  for (int r = 0; r < kReps; ++r)
    hipLaunchKernelGGL(Stream, dim3(1024), dim3(256), 0, nullptr,
                       static_cast<const uint4*>(p), n, sink);
  Check(hipDeviceSynchronize(), "stream");
  return kReps * double(bytes) / (Now() - t0) / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::size_t chunk_mib = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 64;
    const std::size_t reserve_gib =
        argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 64;
    int device = 0;
    Check(hipSetDevice(device), "device");
    int vmm = 0;
    Check(hipDeviceGetAttribute(
              &vmm, hipDeviceAttributeVirtualMemoryManagementSupported, device),
          "vmm attribute");
    std::printf("VirtualMemoryManagementSupported=%d\n", vmm);
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;
    std::size_t minimum = 0, recommended = 0;
    Check(hipMemGetAllocationGranularity(&minimum, &prop,
                                         hipMemAllocationGranularityMinimum),
          "granularity");
    Check(hipMemGetAllocationGranularity(
              &recommended, &prop, hipMemAllocationGranularityRecommended),
          "recommended granularity");
    std::printf("granularity minimum=%zu recommended=%zu\n", minimum,
                recommended);
    const std::size_t chunk = chunk_mib << 20;
    if (chunk % minimum != 0)
      throw std::runtime_error("chunk is not a granularity multiple");
    const std::size_t reserve = reserve_gib << 30;

    std::size_t free0 = 0, total = 0;
    Check(hipMemGetInfo(&free0, &total), "mem info");
    void* base = nullptr;
    double t0 = Now();
    Check(hipMemAddressReserve(&base, reserve, 0, nullptr, 0), "reserve");
    std::printf("reserved %zu GiB at %p in %.3f ms\n", reserve_gib, base,
                (Now() - t0) * 1e3);

    hipMemAccessDesc access{};
    access.location = prop.location;
    access.flags = hipMemAccessFlagsProtReadWrite;
    std::vector<hipMemGenericAllocationHandle_t> handles;
    const auto map = [&](std::size_t index) {
      hipMemGenericAllocationHandle_t h{};
      const double s = Now();
      Check(hipMemCreate(&h, chunk, &prop, 0), "create");
      auto* at = static_cast<char*>(base) + index * chunk;
      Check(hipMemMap(at, chunk, 0, h, 0), "map");
      Check(hipMemSetAccess(at, chunk, &access, 1), "access");
      handles.push_back(h);
      return (Now() - s) * 1e3;
    };
    double worst = 0;
    for (std::size_t i = 0; i < 4; ++i)
      worst = std::max(worst, map(i));
    std::size_t free1 = 0;
    Check(hipMemGetInfo(&free1, &total), "mem info");
    std::printf("4 chunks mapped, worst %.3f ms; heap free %zu -> %zu MiB "
                "(%zd MiB charged)\n",
                worst, free0 >> 20, free1 >> 20,
                (static_cast<std::ptrdiff_t>(free0) - free1) >> 20);

    auto* words = static_cast<std::uint32_t*>(base);
    const std::size_t per_chunk = chunk / 4;
    hipLaunchKernelGGL(Fill, dim3(1024), dim3(256), 0, nullptr, words,
                       4 * per_chunk, 7u);
    Check(hipDeviceSynchronize(), "fill");

    // Capture a graph that reads through a device extent, then grow.
    std::uint64_t* extent = nullptr;
    unsigned long long* out = nullptr;
    Check(hipMalloc(&extent, 8), "extent");
    Check(hipMalloc(&out, 8), "out");
    hipStream_t stream;
    Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    hipGraph_t graph;
    hipGraphExec_t exec;
    Check(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), "capture");
    Check(hipMemsetAsync(out, 0, 8, stream), "zero");
    hipLaunchKernelGGL(Sum, dim3(1024), dim3(256), 0, stream, words, extent,
                       out);
    Check(hipStreamEndCapture(stream, &graph), "end capture");
    Check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
    const auto replay = [&](std::size_t n) {
      const std::uint64_t e = n;
      Check(hipMemcpy(extent, &e, 8, hipMemcpyHostToDevice), "extent");
      Check(hipGraphLaunch(exec, stream), "launch");
      Check(hipStreamSynchronize(stream), "replay");
      unsigned long long got = 0;
      Check(hipMemcpy(&got, out, 8, hipMemcpyDeviceToHost), "out");
      return got;
    };
    bool ok = replay(4 * per_chunk) == Expected(4 * per_chunk, 7u);
    std::printf("graph over 4 chunks: %s\n", ok ? "ok" : "MISMATCH");
    double map_ms = 0;
    for (std::size_t i = 4; i < 16; ++i) {
      const double ms = map(i);
      map_ms += ms;
      worst = std::max(worst, ms);
    }
    std::printf("steady-state create+map+access: mean %.3f ms per chunk\n",
                map_ms / 12);
    hipLaunchKernelGGL(Fill, dim3(1024), dim3(256), 0, nullptr, words,
                       16 * per_chunk, 7u);
    Check(hipDeviceSynchronize(), "fill");
    const bool grown = replay(16 * per_chunk) == Expected(16 * per_chunk, 7u);
    std::printf("same graph after growing to 16 chunks: %s (worst map %.3f ms)\n",
                grown ? "ok" : "MISMATCH", worst);
    ok = ok && grown;

    // Unmap the last 8 chunks, remap fresh memory, refill, replay.
    t0 = Now();
    for (std::size_t i = 8; i < 16; ++i) {
      auto* at = static_cast<char*>(base) + i * chunk;
      Check(hipMemUnmap(at, chunk), "unmap");
      Check(hipMemRelease(handles[i]), "release");
    }
    handles.resize(8);
    std::size_t free2 = 0;
    Check(hipMemGetInfo(&free2, &total), "mem info");
    std::printf("unmapped 8 chunks in %.3f ms, heap free %zu MiB\n",
                (Now() - t0) * 1e3, free2 >> 20);
    for (std::size_t i = 8; i < 16; ++i)
      map(i);
    hipLaunchKernelGGL(Fill, dim3(1024), dim3(256), 0, nullptr, words,
                       16 * per_chunk, 9u);
    Check(hipDeviceSynchronize(), "refill");
    const bool remapped = replay(16 * per_chunk) == Expected(16 * per_chunk, 9u);
    std::printf("graph after unmap/remap: %s\n", remapped ? "ok" : "MISMATCH");
    ok = ok && remapped;

    // hipMemcpy and hipMemsetAsync across chunk boundaries.
    {
      std::vector<std::uint32_t> host(per_chunk * 2);
      Check(hipMemcpy(host.data(), words + per_chunk * 3, host.size() * 4,
                      hipMemcpyDeviceToHost),
            "copy across chunks");
      bool same = true;
      for (std::size_t i = 0; i < host.size(); ++i)
        same = same && host[i] == (static_cast<std::uint32_t>(i + per_chunk * 3) *
                                       2654435761u ^ 9u);
      void* plain = nullptr;
      Check(hipMalloc(&plain, chunk * 2), "plain");
      Check(hipMemcpy(plain, words + per_chunk * 3, chunk * 2,
                      hipMemcpyDeviceToDevice),
            "d2d across chunks");
      Check(hipMemsetAsync(words + per_chunk * 5 - 3, 0, 4096, stream),
            "memset across chunks");
      Check(hipStreamSynchronize(stream), "memset");
      std::printf("host/device copies and memset across chunks: %s\n",
                  same ? "ok" : "MISMATCH");
      ok = ok && same;
      // Read bandwidth, VMM vs hipMalloc, over 8 chunks (beyond MALL).
      uint4* sink = nullptr;
      Check(hipMalloc(&sink, 16), "sink");
      void* big = nullptr;
      Check(hipMalloc(&big, chunk * 8), "big");
      Check(hipMemset(big, 1, chunk * 8), "big fill");
      const double vmm_bw = Bandwidth(words, chunk * 8, sink);
      const double malloc_bw = Bandwidth(big, chunk * 8, sink);
      std::printf("read bandwidth over %zu MiB: VMM %.1f GB/s, hipMalloc %.1f "
                  "GB/s\n",
                  chunk_mib * 8, vmm_bw, malloc_bw);
      (void)hipFree(big);
      (void)hipFree(plain);
      (void)hipFree(sink);
    }

    // Exhaustion: create chunks until creation fails, then release them.
    {
      std::vector<hipMemGenericAllocationHandle_t> extra;
      const std::size_t limit = std::strtoull(
          std::getenv("VMM_EXHAUST_GIB") ? std::getenv("VMM_EXHAUST_GIB") : "0",
          nullptr, 10);
      std::size_t created = 0;
      while (created * chunk < (limit << 30)) {
        hipMemGenericAllocationHandle_t h{};
        const hipError_t e = hipMemCreate(&h, chunk, &prop, 0);
        if (e != hipSuccess) {
          std::printf("hipMemCreate failed cleanly after %zu MiB: %s\n",
                      (created * chunk) >> 20, hipGetErrorString(e));
          (void)hipGetLastError();
          break;
        }
        extra.push_back(h);
        ++created;
      }
      for (auto h : extra)
        Check(hipMemRelease(h), "release extra");
      if (limit)
        std::printf("exhaustion probe created %zu MiB\n", (created * chunk) >> 20);
    }

    (void)hipGraphExecDestroy(exec);
    (void)hipGraphDestroy(graph);
    for (std::size_t i = 0; i < handles.size(); ++i) {
      Check(hipMemUnmap(static_cast<char*>(base) + i * chunk, chunk), "unmap");
      Check(hipMemRelease(handles[i]), "release");
    }
    Check(hipMemAddressFree(base, reserve), "free reserve");
    std::printf("%s\n", ok ? "VMM probe: ok" : "VMM probe: FAILED");
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
