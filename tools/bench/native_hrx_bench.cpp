#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "hrx-system/libhrx/include/hrx_runtime.h"

#define HRX_CHECK(expr)                                                        \
  do {                                                                         \
    hrx_status_t status = (expr);                                              \
    if (!hrx_status_is_ok(status)) {                                           \
      char* msg = nullptr;                                                     \
      size_t len = 0;                                                          \
      hrx_status_to_string(status, &msg, &len);                                \
      std::fprintf(stderr, "HRX error at %s:%d: %s\n", __FILE__, __LINE__,     \
                   msg ? msg : "unknown");                                     \
      hrx_status_free_message(msg);                                            \
      hrx_status_ignore(status);                                               \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

static inline float Bf16ToFloat(uint16_t val) {
  union {
    std::uint32_t u;
    float f;
  } converter;
  converter.u = static_cast<std::uint32_t>(val) << 16;
  return converter.f;
}

static inline uint16_t FloatToBf16(float val) {
  union {
    float f;
    std::uint32_t u;
  } converter;
  converter.f = val;
  return static_cast<std::uint16_t>(converter.u >> 16);
}

int main() {
  const std::size_t M = 17408;
  const std::size_t K = 5120;
  const int iters = 50;

  int major = 0, minor = 0, patch = 0;
  hrx_runtime_version(&major, &minor, &patch);
  std::printf("Native libhrx version: %d.%d.%d\n", major, minor, patch);

  HRX_CHECK(hrx_gpu_initialize(0));
  std::printf("Initialized HRX GPU runtime!\n");

  hrx_device_t device = nullptr;
  HRX_CHECK(hrx_gpu_device_get(0, &device));

  hrx_stream_t stream = nullptr;
  HRX_CHECK(hrx_stream_create(device, 0, &stream));

  // Load the Loom-compiled executable artifact
  hrx_executable_t executable = nullptr;
  HRX_CHECK(hrx_executable_load_file(
      device, "/tmp/qwen_swiglu.fb", "amdgpu", "gfx1151", &executable));
  std::printf("Loaded Loom executable artifact /tmp/qwen_swiglu.fb!\n");

  // Host data
  std::vector<float> h_x(K);
  std::vector<uint16_t> h_gate(M * K);
  std::vector<uint16_t> h_up(M * K);
  std::vector<float> h_out_ref(M, 0.0F);
  std::vector<float> h_out_hrx(M, 0.0F);

  for (std::size_t i = 0; i < K; ++i) {
    h_x[i] = 0.01F * static_cast<float>((i % 11) - 5);
  }
  for (std::size_t i = 0; i < M * K; ++i) {
    h_gate[i] = FloatToBf16(0.0005F * static_cast<float>((i % 13) - 6));
    h_up[i] = FloatToBf16(0.0005F * static_cast<float>((i % 17) - 8));
  }

  // Compute CPU oracle
  for (std::size_t m = 0; m < M; ++m) {
    float dot_g = 0.0F;
    float dot_u = 0.0F;
    for (std::size_t k = 0; k < K; ++k) {
      dot_g += Bf16ToFloat(h_gate[m * K + k]) * h_x[k];
      dot_u += Bf16ToFloat(h_up[m * K + k]) * h_x[k];
    }
    float silu_g = dot_g / (1.0F + std::exp(-dot_g));
    h_out_ref[m] = silu_g * dot_u;
  }

  // Allocate native HRX device buffers
  hrx_buffer_t buf_x = nullptr, buf_gate = nullptr, buf_up = nullptr, buf_out = nullptr;
  HRX_CHECK(hrx_buffer_allocate(stream, K * sizeof(float), HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &buf_x));
  HRX_CHECK(hrx_buffer_allocate(stream, M * K * sizeof(uint16_t), HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &buf_gate));
  HRX_CHECK(hrx_buffer_allocate(stream, M * K * sizeof(uint16_t), HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &buf_up));
  HRX_CHECK(hrx_buffer_allocate(stream, M * sizeof(float), HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &buf_out));

  // Copy data to device
  HRX_CHECK(hrx_synchronous_h2d(device, h_x.data(), buf_x, 0, K * sizeof(float)));
  HRX_CHECK(hrx_synchronous_h2d(device, h_gate.data(), buf_gate, 0, M * K * sizeof(uint16_t)));
  HRX_CHECK(hrx_synchronous_h2d(device, h_up.data(), buf_up, 0, M * K * sizeof(uint16_t)));

  // Setup dispatch config: 2 rows per workgroup tile -> M/2 workgroups of 160 threads
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = static_cast<uint32_t>(M / 2);
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 160;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;

  // Setup buffer bindings (input, gate_weight, up_weight, output)
  hrx_buffer_ref_t bindings[4];
  bindings[0].buffer = buf_x;
  bindings[0].offset = 0;
  bindings[0].length = K * sizeof(float);

  bindings[1].buffer = buf_gate;
  bindings[1].offset = 0;
  bindings[1].length = M * K * sizeof(uint16_t);

  bindings[2].buffer = buf_up;
  bindings[2].offset = 0;
  bindings[2].length = M * K * sizeof(uint16_t);

  bindings[3].buffer = buf_out;
  bindings[3].offset = 0;
  bindings[3].length = M * sizeof(float);

  // Kernel arguments (constants): rows parameter (uint32_t, 4 bytes)
  uint32_t rows_param = static_cast<uint32_t>(M);

  // Warmup
  for (int i = 0; i < 5; ++i) {
    HRX_CHECK(hrx_stream_dispatch(stream, executable, 0, &config, &rows_param, sizeof(rows_param), bindings, 4, 0));
  }
  HRX_CHECK(hrx_stream_synchronize(stream));

  // Benchmark native libhrx dispatch
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    HRX_CHECK(hrx_stream_dispatch(stream, executable, 0, &config, &rows_param, sizeof(rows_param), bindings, 4, 0));
  }
  HRX_CHECK(hrx_stream_synchronize(stream));
  auto t1 = std::chrono::high_resolution_clock::now();

  double lat_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

  // Readback and verify correctness
  HRX_CHECK(hrx_synchronous_d2h(device, buf_out, 0, h_out_hrx.data(), M * sizeof(float)));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < M; ++i) {
    float diff = std::fabs(h_out_hrx[i] - h_out_ref[i]) / (std::fabs(h_out_ref[i]) + 1e-4F);
    if (diff > max_diff) max_diff = diff;
  }

  double total_bytes = static_cast<double>(2 * M * K * sizeof(uint16_t) + K * sizeof(float) + M * sizeof(float));
  double bw_gbps = (total_bytes / (lat_us * 1e-6)) / 1e9;

  std::printf("\n======================================================================\n");
  std::printf("  Native libhrx Direct Dispatch Benchmark (N=%zu, K=%zu, %d iters)\n", M, K, iters);
  std::printf("======================================================================\n");
  std::printf("Native libhrx JIT Dispatch : %7.2f us | Bandwidth: %6.2f GB/s | Max Diff: %.2e\n",
              lat_us, bw_gbps, max_diff);

  hrx_buffer_release(buf_x);
  hrx_buffer_release(buf_gate);
  hrx_buffer_release(buf_up);
  hrx_buffer_release(buf_out);
  hrx_executable_release(executable);
  hrx_stream_release(stream);
  hrx_device_release(device);

  std::printf("\nAll native libhrx operations completed successfully!\n");
  return 0;
}
