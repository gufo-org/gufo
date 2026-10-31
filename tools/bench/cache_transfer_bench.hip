// Copy-path bandwidth on gfx1151 for component-cache chunk sizes.
// Host buffers mimic CommittedBackingPool: mmap + MADV_NOHUGEPAGE + populate
// + hipHostRegister. Modes: hipMemcpyAsync H2D / D2H / D2D over 1, 4 and 16
// streams, and one gather kernel launch device->device, from mapped host
// memory and to mapped host memory. GB/s over 1 GiB, median of five.
//
//   nix develop -c tools/bench/build.sh cache_transfer_bench && /tmp/cache_transfer_bench
#include <hip/hip_runtime.h>
#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x)                                                              \
  do {                                                                        \
    hipError_t e = (x);                                                       \
    if (e != hipSuccess) {                                                    \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,                  \
                   hipGetErrorString(e));                                     \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

struct Run {
  const uint4* src;
  uint4* dst;
  unsigned long long n16;
};

__global__ void Gather(const Run* runs) {
  const Run r = runs[blockIdx.y];
  for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < r.n16;
       i += gridDim.x * blockDim.x)
    r.dst[i] = r.src[i];
}

using Clock = std::chrono::steady_clock;

int main() {
  constexpr size_t kTotal = size_t{1} << 30;  // 1 GiB per measurement
  void* host = mmap(nullptr, kTotal, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (host == MAP_FAILED) return 1;
  madvise(host, kTotal, MADV_NOHUGEPAGE);
  std::memset(host, 1, kTotal);
  CHECK(hipHostRegister(host, kTotal, hipHostRegisterDefault));
  void* host_dev = nullptr;
  CHECK(hipHostGetDevicePointer(&host_dev, host, 0));
  void *dev_a = nullptr, *dev_b = nullptr;
  CHECK(hipMalloc(&dev_a, kTotal));
  CHECK(hipMalloc(&dev_b, kTotal));
  CHECK(hipMemset(dev_a, 2, kTotal));
  CHECK(hipMemset(dev_b, 3, kTotal));
  std::vector<hipStream_t> streams(16);
  for (auto& s : streams) CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  Run* runs_dev = nullptr;
  CHECK(hipMalloc(&runs_dev, sizeof(Run) * 65536));

  const size_t pieces[] = {128 << 10, 256 << 10, 1 << 20, 2 << 20, 8 << 20};
  std::printf("%-6s %8s %3s %9s\n", "mode", "piece", "S", "GB/s");
  auto report = [](const char* mode, size_t piece, int s, double sec) {
    std::printf("%-6s %7zuK %3d %9.1f\n", mode, piece >> 10, s, kTotal / sec / 1e9);
  };
  for (size_t piece : pieces) {
    const size_t n = kTotal / piece;
    struct M { const char* name; void* dst; const void* src; hipMemcpyKind kind; };
    const M modes[] = {{"H2D", dev_a, host, hipMemcpyHostToDevice},
                       {"D2H", host, dev_a, hipMemcpyDeviceToHost},
                       {"D2D", dev_b, dev_a, hipMemcpyDeviceToDevice}};
    for (const auto& m : modes) {
      for (int s : {1, 4, 16}) {
        std::vector<double> t;
        for (int rep = 0; rep < 5; ++rep) {
          CHECK(hipDeviceSynchronize());
          const auto a = Clock::now();
          for (size_t i = 0; i < n; ++i)
            CHECK(hipMemcpyAsync(static_cast<char*>(m.dst) + i * piece,
                                 static_cast<const char*>(m.src) + i * piece,
                                 piece, m.kind, streams[i % s]));
          for (int k = 0; k < s; ++k) CHECK(hipStreamSynchronize(streams[k]));
          t.push_back(std::chrono::duration<double>(Clock::now() - a).count());
        }
        std::sort(t.begin(), t.end());
        report(m.name, piece, s, t[2]);
      }
    }
    // One gather launch over n runs: device->device, then host-mapped->device.
    for (int variant = 0; variant < 3; ++variant) {
      const bool src_host = variant == 1, dst_host = variant == 2;
      std::vector<Run> runs(n);
      for (size_t i = 0; i < n; ++i)
        runs[i] = {reinterpret_cast<const uint4*>(
                       static_cast<const char*>(src_host ? host_dev : dev_a) + i * piece),
                   reinterpret_cast<uint4*>(static_cast<char*>(dst_host ? host_dev : dev_b) + i * piece),
                   piece / 16};
      CHECK(hipMemcpy(runs_dev, runs.data(), sizeof(Run) * n, hipMemcpyHostToDevice));
      const unsigned blocks_x = std::max<size_t>(1, std::min<size_t>(64, piece / (256 * 16 * 4)));
      std::vector<double> t;
      for (int rep = 0; rep < 5; ++rep) {
        CHECK(hipDeviceSynchronize());
        const auto a = Clock::now();
        Gather<<<dim3(blocks_x, n), 256, 0, streams[0]>>>(runs_dev);
        CHECK(hipStreamSynchronize(streams[0]));
        t.push_back(std::chrono::duration<double>(Clock::now() - a).count());
      }
      std::sort(t.begin(), t.end());
      report(dst_host ? "KD2H" : src_host ? "KH2D" : "KD2D", piece, 1, t[2]);
    }
  }
  return 0;
}
