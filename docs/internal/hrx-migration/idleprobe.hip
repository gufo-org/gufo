// Scratch probe: GPU busy % and power while a process sits idle in three
// states: after runtime init, after one finished kernel, and with a stream
// blocked on a host-signalled wait value. Prints the sysfs readings itself.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <glob.h>
#include <string>
#include <thread>

#define OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1);} } while (0)

__global__ void Touch(unsigned* p) { p[threadIdx.x] = threadIdx.x; }

static std::string Find(const char* pattern) {
  glob_t g; glob(pattern, 0, nullptr, &g);
  std::string out = g.gl_pathc ? g.gl_pathv[0] : "";
  globfree(&g); return out;
}

static long Read(const std::string& path) {
  std::ifstream f(path); long v = 0; f >> v; return v;
}

static void Observe(const char* label) {
  static const std::string power = Find("/sys/class/drm/card*/device/hwmon/hwmon*/power1_average");
  static const std::string busy = Find("/sys/class/drm/card*/device/gpu_busy_percent");
  static const std::string sclk = Find("/sys/class/drm/card*/device/hwmon/hwmon*/freq1_input");
  std::this_thread::sleep_for(std::chrono::seconds(3));
  long p = 0, b = 0, f = 0; const int n = 70;
  for (int i = 0; i < n; ++i) {
    p += Read(power); b += Read(busy); f += Read(sclk);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  std::printf("%-34s power %4.1f W  busy %3ld%%  sclk %4ld MHz\n", label,
              p / double(n) / 1e6, b / n, f / n / 1000000);
  std::fflush(stdout);
}

int main() {
  OK(hipSetDevice(0));
  unsigned* buf; OK(hipMalloc(&buf, 4096));
  Observe("after init + hipMalloc");
  hipStream_t s; OK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  Touch<<<1, 64, 0, s>>>(buf); OK(hipStreamSynchronize(s));
  Observe("after one finished kernel");
  // Model-load style uploads: pageable 64 MiB, pinned 64 MiB async.
  {
    const size_t bytes = 64u << 20;
    void* host = std::malloc(bytes); std::memset(host, 1, bytes);
    void* dev; OK(hipMalloc(&dev, bytes));
    OK(hipMemcpy(dev, host, bytes, hipMemcpyHostToDevice));
    Observe("after 64 MiB pageable hipMemcpy");
    void* pinned; OK(hipHostMalloc(&pinned, bytes, hipHostMallocDefault));
    OK(hipMemcpyAsync(dev, pinned, bytes, hipMemcpyHostToDevice, s));
    OK(hipStreamSynchronize(s));
    Observe("after pinned hipMemcpyAsync");
    void* reg = std::malloc(bytes); std::memset(reg, 2, bytes);
    OK(hipHostRegister(reg, bytes, hipHostRegisterDefault));
    OK(hipMemcpyAsync(dev, reg, bytes, hipMemcpyHostToDevice, s));
    OK(hipStreamSynchronize(s));
    Observe("after registered hipMemcpyAsync");
  }
  // Fork/join across two streams with events, then idle.
  {
    hipStream_t side; OK(hipStreamCreate(&side));
    hipEvent_t e1, e2; OK(hipEventCreateWithFlags(&e1, hipEventDisableTiming));
    OK(hipEventCreateWithFlags(&e2, hipEventDisableTiming));
    OK(hipEventRecord(e1, s)); OK(hipStreamWaitEvent(side, e1, 0));
    Touch<<<1, 64, 0, side>>>(buf);
    OK(hipEventRecord(e2, side)); OK(hipStreamWaitEvent(s, e2, 0));
    OK(hipStreamSynchronize(s));
    Observe("after cross-stream fork/join");
    // A wait on an event that is recorded much later (pending forever while idle).
    hipEvent_t later; OK(hipEventCreateWithFlags(&later, hipEventDisableTiming));
    OK(hipEventRecord(later, side));
    OK(hipStreamWaitEvent(s, later, 0));
    Observe("after wait on completed event");
  }
  // A stream waiting on a value the host writes later (unsupported on HRX).
  {
    unsigned* flag = nullptr;
    if (hipHostMalloc(&flag, 64, hipHostMallocMapped) == hipSuccess) {
      *flag = 0;
      unsigned* dflag = nullptr;
      (void)hipHostGetDevicePointer((void**)&dflag, flag, 0);
      if (hipStreamWaitValue32(s, dflag, 1, hipStreamWaitValueEq, 0xffffffffu) == hipSuccess) {
        Observe("stream blocked on wait value");
        *flag = 1; OK(hipStreamSynchronize(s));
      } else {
        std::printf("hipStreamWaitValue32 unsupported\n");
      }
    }
  }
  // A captured graph launched once, its executable kept alive.
  {
    hipGraph_t graph; hipGraphExec_t exec;
    OK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
    for (int i = 0; i < 64; ++i) Touch<<<1, 64, 0, s>>>(buf);
    OK(hipStreamEndCapture(s, &graph));
    OK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    Observe("graph instantiated, not launched");
    OK(hipGraphLaunch(exec, s)); OK(hipStreamSynchronize(s));
    Observe("graph launched once, exec alive");
    for (int i = 0; i < 100; ++i) OK(hipGraphLaunch(exec, s));
    OK(hipStreamSynchronize(s));
    Observe("graph launched 100x, exec alive");
    OK(hipGraphExecDestroy(exec)); OK(hipGraphDestroy(graph));
    Observe("graph destroyed");
  }
  Observe("final idle");
  return 0;
}
