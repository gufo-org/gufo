// The stale-translation question for the SDMA copy engine
// (hsa_amd_memory_async_copy_on_engine, as src/core/hip/copy_engine.cpp calls
// it): after an address is unmapped and mapped with another handle, does a copy
// through the engine reach the new handle?
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>

#include "common.hpp"
static const size_t P = 2u << 20, W = P / 4;
__global__ void Fill(uint32_t* d, size_t words, uint32_t v) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
       i += gridDim.x * blockDim.x)
    d[i] = v;
}
__global__ void Read1(const uint32_t* d, uint32_t* out) {
  out[0] = d[0];
}
__global__ void Spin(uint32_t* d, int iters) {
  uint32_t x = d[threadIdx.x];
  for (int i = 0; i < iters; ++i)
    x = x * 1664525u + 1013904223u;
  d[threadIdx.x] = x;
}
static Vmm v;
static uint32_t* dout;
static hsa_agent_t gpu{}, cpu{};
static hsa_amd_sdma_engine_id_t engine{};
static hsa_status_t visit(hsa_agent_t a, void*) {
  hsa_device_type_t t{};
  hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
  if (t == HSA_DEVICE_TYPE_CPU && !cpu.handle)
    cpu = a;
  if (t == HSA_DEVICE_TYPE_GPU && !gpu.handle)
    gpu = a;
  return HSA_STATUS_SUCCESS;
}
static void* reserve(size_t n = 1) {
  hipDeviceptr_t p{};
  C(hipMemAddressReserve(&p, n * P, P, nullptr, 0));
  return p;
}
static hipMemGenericAllocationHandle_t make() {
  hipMemGenericAllocationHandle_t h;
  C(hipMemCreate(&h, P, &v.prop, 0));
  return h;
}
static void map(void* a, hipMemGenericAllocationHandle_t h) {
  C(hipMemMap(a, P, 0, h, 0));
  C(hipMemSetAccess(a, P, &v.acc, 1));
}
static void spin_us(double us) {
  const double t = now_us();
  while (now_us() - t < us) {
  }
}
static uint32_t witness(hipMemGenericAllocationHandle_t h) {
  void* w = reserve();
  map(w, h);
  Read1<<<1, 1>>>((const uint32_t*)w, dout);
  C(hipDeviceSynchronize());
  uint32_t o;
  C(hipMemcpy(&o, dout, 4, hipMemcpyDeviceToHost));
  return o;
}
static bool sdma(void* dst, const void* src, size_t n) {
  hsa_signal_t sig{};
  if (hsa_signal_create(1, 0, nullptr, &sig) != HSA_STATUS_SUCCESS)
    return false;
  if (hsa_amd_memory_async_copy_on_engine(dst, gpu, src, cpu, n, 0, nullptr,
                                          sig, engine,
                                          true) != HSA_STATUS_SUCCESS) {
    printf("copy refused\n");
    return false;
  }
  while (hsa_signal_load_scacquire(sig) != 0)
    hsa_signal_wait_scacquire(sig, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                              HSA_WAIT_STATE_BLOCKED);
  hsa_signal_destroy(sig);
  return true;
}
int main(int argc, char** argv) {
  const int c = argc > 1 ? atoi(argv[1]) : 1;
  const int reps = argc > 2 ? atoi(argv[2]) : 20;
  const double idle = argc > 3 ? atof(argv[3]) : 0;
  C(hipMalloc(&dout, 64));
  if (hsa_init() != HSA_STATUS_SUCCESS) {
    printf("hsa_init failed\n");
    return 1;
  }
  hsa_iterate_agents(visit, nullptr);
  uint32_t engines = 0;
  if (hsa_amd_memory_copy_engine_status(gpu, cpu, &engines) !=
          HSA_STATUS_SUCCESS ||
      !engines) {
    printf("no engine\n");
    return 1;
  }
  engine = (hsa_amd_sdma_engine_id_t)(engines & -engines);
  uint32_t* host[2];
  for (int i = 0; i < 2; ++i) {
    C(hipHostMalloc((void**)&host[i], P, 0));
    for (size_t w = 0; w < W; ++w)
      host[i][w] = 0xA0 + i;
  }
  std::atomic<bool> stop{false};
  std::thread bg;
  if (c == 6)
    bg = std::thread([&] {
      hipStream_t b;
      C(hipStreamCreate(&b));
      uint32_t* buf;
      C(hipMalloc(&buf, 4096));
      while (!stop) {
        Spin<<<1, 256, 0, b>>>(buf, 2000);
        C(hipStreamSynchronize(b));
      }
    });
  int ok = 0, stale = 0, other = 0;
  for (int r = 0; r < reps; ++r) {
    void* X = reserve();
    auto h0 = make(), h1 = make();
    void* home = reserve();
    map(home, h1);
    Fill<<<64, 256>>>((uint32_t*)home, W, 0x11);
    C(hipDeviceSynchronize());
    C(hipMemUnmap(home, P));
    map(X, h0);
    // Touch X: 1,2,4,6 through the engine; 3 with a compute kernel; 5 through
    // the engine, then read with a kernel.
    if (c == 3) {
      Fill<<<64, 256>>>((uint32_t*)X, W, 0xA0);
      C(hipDeviceSynchronize());
    } else if (!sdma(X, host[0], P))
      return 1;
    C(hipMemUnmap(X, P));
    map(X, h1);
    if (c == 2 || c == 4 || c == 6)
      spin_us(idle);
    uint32_t seen0 = 0, seen1 = 0;
    if (c == 5) {
      Read1<<<1, 1>>>((const uint32_t*)X, dout);
      C(hipDeviceSynchronize());
      uint32_t o;
      C(hipMemcpy(&o, dout, 4, hipMemcpyDeviceToHost));
      ok += o == 0x11;
      stale += o == 0xA0;
      other += o != 0x11 && o != 0xA0;
      continue;
    }
    if (!sdma(X, host[1], P))
      return 1;
    seen0 = witness(h0);
    seen1 = witness(h1);
    ok += seen0 == 0xA0 && seen1 == 0xA1;
    stale += seen0 == 0xA1;
    other += !(seen0 == 0xA0 && seen1 == 0xA1) && seen0 != 0xA1;
  }
  stop = true;
  if (bg.joinable())
    bg.join();
  const char* names[] = {
      "",
      "engine touch, engine copy at once",
      "engine touch, idle, engine copy",
      "kernel touch, engine copy at once",
      "engine touch, idle, engine copy",
      "engine touch, kernel read at once",
      "engine touch, idle with kernels running, engine copy"};
  printf("case %d (%s) idle %.0f us: new %d, old %d, other %d of %d\n", c,
         names[c], idle, ok, stale, other, reps);
  return 0;
}
