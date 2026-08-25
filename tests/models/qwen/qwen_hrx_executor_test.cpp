#include "src/models/qwen/hrx/qwen_hrx_executor.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

#ifndef GUFO_HRX_KERNEL_DIR
#error "HRX executor tests require Nix-built AOT kernel artifacts"
#endif

constexpr std::string_view kHrxKernelDir = GUFO_HRX_KERNEL_DIR;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << "\n";
    std::exit(1);
  }
}

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

void TestQwenHrxExecutorLifecycleAndDispatch() {
  gufo::hrx::QwenHrxExecutor executor(0);
  const std::string artifact =
      std::string(kHrxKernelDir) + "/qwen_fused_swiglu_bf16.fb";
  bool ready = executor.Initialize(artifact);
  Expect(ready, "QwenHrxExecutor loads the Nix-built SwiGLU artifact");

  Expect(executor.IsReady(), "QwenHrxExecutor is ready");
  auto& backend = executor.Backend();
  hrx_stream_t stream = backend.Stream();
  hrx_device_t device = backend.Device();

  const uint32_t M = 128;   // Test with 128 rows
  const uint32_t K = 5120;  // Qwen hidden dim

  std::vector<float> h_x(K);
  std::vector<uint16_t> h_gate(M * K);
  std::vector<uint16_t> h_up(M * K);
  std::vector<float> h_out_ref(M, 0.0F);
  std::vector<float> h_out_hrx(M, 0.0F);

  for (std::size_t i = 0; i < K; ++i) {
    h_x[i] = 0.01F * static_cast<float>((i % 7) - 3);
  }
  for (std::size_t i = 0; i < M * K; ++i) {
    h_gate[i] = FloatToBf16(0.001F * static_cast<float>((i % 11) - 5));
    h_up[i] = FloatToBf16(0.001F * static_cast<float>((i % 13) - 6));
  }

  // CPU Oracle
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

  // Allocate HRX buffers
  hrx_buffer_t buf_x = nullptr, buf_gate = nullptr, buf_up = nullptr,
               buf_out = nullptr;
  HRX_CHECK(hrx_buffer_allocate(stream, K * sizeof(float),
                                HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                HRX_BUFFER_USAGE_DEFAULT, &buf_x));
  HRX_CHECK(hrx_buffer_allocate(stream, M * K * sizeof(uint16_t),
                                HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                HRX_BUFFER_USAGE_DEFAULT, &buf_gate));
  HRX_CHECK(hrx_buffer_allocate(stream, M * K * sizeof(uint16_t),
                                HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                HRX_BUFFER_USAGE_DEFAULT, &buf_up));
  HRX_CHECK(hrx_buffer_allocate(stream, M * sizeof(float),
                                HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                HRX_BUFFER_USAGE_DEFAULT, &buf_out));

  HRX_CHECK(
      hrx_synchronous_h2d(device, h_x.data(), buf_x, 0, K * sizeof(float)));
  HRX_CHECK(hrx_synchronous_h2d(device, h_gate.data(), buf_gate, 0,
                                M * K * sizeof(uint16_t)));
  HRX_CHECK(hrx_synchronous_h2d(device, h_up.data(), buf_up, 0,
                                M * K * sizeof(uint16_t)));

  bool ok = executor.DispatchSwiGLU(buf_x, buf_gate, buf_up, buf_out, M, K);
  Expect(ok, "DispatchSwiGLU executes successfully");

  HRX_CHECK(hrx_stream_synchronize(stream));
  HRX_CHECK(hrx_synchronous_d2h(device, buf_out, 0, h_out_hrx.data(),
                                M * sizeof(float)));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < M; ++i) {
    float diff = std::fabs(h_out_hrx[i] - h_out_ref[i]) /
                 (std::fabs(h_out_ref[i]) + 1e-4F);
    if (diff > max_diff)
      max_diff = diff;
  }

  Expect(max_diff < 1e-3F, "Numerical output matches CPU reference oracle");

  hrx_buffer_release(buf_x);
  hrx_buffer_release(buf_gate);
  hrx_buffer_release(buf_up);
  hrx_buffer_release(buf_out);
}

void TestQwenHrxExecutorMultiKernelAndGraph() {
  gufo::hrx::QwenHrxExecutor executor(0);
  bool ready = executor.InitializeAllKernels(std::string(kHrxKernelDir));
  Expect(ready, "InitializeAllKernels loads the Nix-built artifacts");

  Expect(executor.IsReady(), "InitializeAllKernels marks executor ready");
  auto& backend = executor.Backend();
  hrx_stream_t stream = backend.Stream();

  const uint32_t hidden_dim = 5120;
  const uint32_t intermediate_dim = 17408;

  // Allocate buffers for graph execution
  hrx_buffer_t buf_hidden = nullptr, buf_attn_gamma = nullptr,
               buf_w_qkv = nullptr;
  hrx_buffer_t buf_cos = nullptr, buf_sin = nullptr, buf_q_out = nullptr;
  hrx_buffer_t buf_k_cache = nullptr, buf_v_cache = nullptr;
  hrx_buffer_t buf_ffn_gamma = nullptr, buf_w_down = nullptr, buf_out = nullptr;

  HRX_CHECK(hrx_buffer_allocate(stream, hidden_dim * sizeof(float),
                                HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                HRX_BUFFER_USAGE_DEFAULT, &buf_hidden));
  HRX_CHECK(hrx_buffer_allocate(stream, hidden_dim * sizeof(float),
                                HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                HRX_BUFFER_USAGE_DEFAULT, &buf_attn_gamma));
  HRX_CHECK(hrx_buffer_allocate(stream, 8192 * hidden_dim * sizeof(uint16_t),
                                HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                HRX_BUFFER_USAGE_DEFAULT, &buf_w_qkv));
  HRX_CHECK(hrx_buffer_allocate(stream, 64 * sizeof(float),
                                HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                HRX_BUFFER_USAGE_DEFAULT, &buf_cos));
  HRX_CHECK(hrx_buffer_allocate(stream, 64 * sizeof(float),
                                HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                HRX_BUFFER_USAGE_DEFAULT, &buf_sin));
  HRX_CHECK(hrx_buffer_allocate(stream, 17408 * sizeof(float),
                                HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                HRX_BUFFER_USAGE_DEFAULT, &buf_q_out));
  HRX_CHECK(hrx_buffer_allocate(stream, 8192 * sizeof(float),
                                HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                HRX_BUFFER_USAGE_DEFAULT, &buf_k_cache));
  HRX_CHECK(hrx_buffer_allocate(stream, 8192 * sizeof(float),
                                HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                HRX_BUFFER_USAGE_DEFAULT, &buf_v_cache));
  HRX_CHECK(hrx_buffer_allocate(stream, intermediate_dim * sizeof(float),
                                HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                HRX_BUFFER_USAGE_DEFAULT, &buf_ffn_gamma));
  HRX_CHECK(hrx_buffer_allocate(
      stream, hidden_dim * intermediate_dim * sizeof(uint16_t),
      HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &buf_w_down));
  HRX_CHECK(hrx_buffer_allocate(stream, hidden_dim * sizeof(float),
                                HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                HRX_BUFFER_USAGE_DEFAULT, &buf_out));

  bool graph_built = executor.BuildAndInstantiateDecodeGraph(
      buf_hidden, buf_attn_gamma, buf_w_qkv, buf_cos, buf_sin, buf_q_out,
      buf_k_cache, buf_v_cache, buf_ffn_gamma, buf_w_down, buf_out);
  Expect(graph_built, "BuildAndInstantiateDecodeGraph builds decode DAG");
  bool executed = executor.ExecuteDecodeGraph();
  Expect(executed, "ExecuteDecodeGraph executes instantiated decode DAG");
  HRX_CHECK(hrx_stream_synchronize(stream));

  hrx_buffer_release(buf_hidden);
  hrx_buffer_release(buf_attn_gamma);
  hrx_buffer_release(buf_w_qkv);
  hrx_buffer_release(buf_cos);
  hrx_buffer_release(buf_sin);
  hrx_buffer_release(buf_q_out);
  hrx_buffer_release(buf_k_cache);
  hrx_buffer_release(buf_v_cache);
  hrx_buffer_release(buf_ffn_gamma);
  hrx_buffer_release(buf_w_down);
  hrx_buffer_release(buf_out);
}

}  // namespace

int main() {
  std::cout << "Running qwen_hrx_executor_test...\n";
  TestQwenHrxExecutorLifecycleAndDispatch();
  TestQwenHrxExecutorMultiKernelAndGraph();
  std::cout << "All qwen_hrx_executor_test assertions passed!\n";
  return 0;
}
