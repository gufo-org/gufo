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

gufo::hrx::QwenHrxArtifactContract Qwen38Contract() {
  gufo::core::ModelConfig config;
  config.num_layers = 64;
  config.hidden_size = 5120;
  config.intermediate_size = 17408;
  config.num_attention_heads = 24;
  config.num_key_value_heads = 4;
  config.head_dim = 256;
  config.rotary_dim = 64;
  config.ssm_group_count = 16;
  config.ssm_state_size = 128;
  config.ssm_time_step_rank = 48;
  config.ssm_inner_size = 6144;
  config.vocab_size = 248320;
  const auto contract = gufo::hrx::QwenHrxArtifactContract::FromConfig(config);
  Expect(contract.has_value(), "test Qwen3.8 contract is valid");
  return *contract;
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
  gufo::hrx::QwenHrxExecutor executor(Qwen38Contract(), 0);
  const std::string artifact =
      std::string(kHrxKernelDir) + "/qwen_fused_swiglu_bf16.fb";
  bool ready = executor.Initialize(artifact);
  Expect(ready, "QwenHrxExecutor loads the Nix-built SwiGLU artifact");

  Expect(executor.IsSwiGLUReady(), "SwiGLU capability is ready");
  Expect(!executor.IsReady(),
         "a single artifact does not mark the complete set ready");
  Expect(executor.MissingKernelArtifacts().size() == 7,
         "single-artifact initialization reports missing capabilities");
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
    const auto sample = static_cast<int>(i % 7) - 3;
    h_x[i] = 0.01F * static_cast<float>(sample);
  }
  for (std::size_t i = 0; i < M * K; ++i) {
    const auto gate_sample = static_cast<int>(i % 11) - 5;
    const auto up_sample = static_cast<int>(i % 13) - 6;
    h_gate[i] = FloatToBf16(0.001F * static_cast<float>(gate_sample));
    h_up[i] = FloatToBf16(0.001F * static_cast<float>(up_sample));
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

  bool ok = executor.DispatchSwiGLU(buf_x, buf_gate, buf_up, buf_out, M);
  Expect(ok, "DispatchSwiGLU executes successfully");

  HRX_CHECK(hrx_stream_synchronize(stream));
  HRX_CHECK(hrx_synchronous_d2h(device, buf_out, 0, h_out_hrx.data(),
                                M * sizeof(float)));

  float max_diff = 0.0F;
  bool outputs_are_finite = true;
  for (std::size_t i = 0; i < M; ++i) {
    if (!std::isfinite(h_out_ref[i]) || !std::isfinite(h_out_hrx[i])) {
      outputs_are_finite = false;
      continue;
    }
    const float diff = std::fabs(h_out_hrx[i] - h_out_ref[i]) /
                       (std::fabs(h_out_ref[i]) + 1e-4F);
    outputs_are_finite = outputs_are_finite && std::isfinite(diff);
    if (diff > max_diff) {
      max_diff = diff;
    }
  }

  Expect(outputs_are_finite, "SwiGLU outputs are finite");
  Expect(max_diff < 1e-3F, "Numerical output matches CPU reference oracle");

  hrx_buffer_release(buf_x);
  hrx_buffer_release(buf_gate);
  hrx_buffer_release(buf_up);
  hrx_buffer_release(buf_out);
}

void TestRMSNormQKVParity() {
  gufo::hrx::QwenHrxExecutor executor(Qwen38Contract(), 0);
  Expect(executor.InitializeAllKernels(std::string(kHrxKernelDir)),
         "QwenHrxExecutor loads the Qwen3.8 artifact set");

  auto& backend = executor.Backend();
  constexpr uint32_t kHiddenSize = 5120;
  constexpr uint32_t kRows = 2;
  constexpr float kEpsilon = 0.000001F;
  const std::vector<float> input(kHiddenSize, 1.0F);
  const std::vector<float> gamma(kHiddenSize, 1.0F);
  const std::vector<uint16_t> weights(kRows * kHiddenSize, 0x3F80U);
  std::vector<float> output(kRows, 0.0F);

  hrx_buffer_t input_buffer = nullptr;
  hrx_buffer_t gamma_buffer = nullptr;
  hrx_buffer_t weight_buffer = nullptr;
  hrx_buffer_t output_buffer = nullptr;
  HRX_CHECK(hrx_buffer_allocate(backend.Stream(), input.size() * sizeof(float),
                               HRX_MEMORY_TYPE_DEVICE_LOCAL,
                               HRX_BUFFER_USAGE_STORAGE | HRX_BUFFER_USAGE_TRANSFER,
                               &input_buffer));
  HRX_CHECK(hrx_buffer_allocate(backend.Stream(), gamma.size() * sizeof(float),
                               HRX_MEMORY_TYPE_DEVICE_LOCAL,
                               HRX_BUFFER_USAGE_STORAGE | HRX_BUFFER_USAGE_TRANSFER,
                               &gamma_buffer));
  HRX_CHECK(hrx_buffer_allocate(backend.Stream(),
                               weights.size() * sizeof(uint16_t),
                               HRX_MEMORY_TYPE_DEVICE_LOCAL,
                               HRX_BUFFER_USAGE_STORAGE | HRX_BUFFER_USAGE_TRANSFER,
                               &weight_buffer));
  HRX_CHECK(hrx_buffer_allocate(backend.Stream(), output.size() * sizeof(float),
                               HRX_MEMORY_TYPE_DEVICE_LOCAL,
                               HRX_BUFFER_USAGE_STORAGE | HRX_BUFFER_USAGE_TRANSFER,
                               &output_buffer));
  HRX_CHECK(hrx_synchronous_h2d(backend.Device(), input.data(), input_buffer,
                                0, input.size() * sizeof(float)));
  HRX_CHECK(hrx_synchronous_h2d(backend.Device(), gamma.data(), gamma_buffer,
                                0, gamma.size() * sizeof(float)));
  HRX_CHECK(hrx_synchronous_h2d(backend.Device(), weights.data(),
                                weight_buffer, 0,
                                weights.size() * sizeof(uint16_t)));

  Expect(executor.DispatchRMSNormQKV(input_buffer, gamma_buffer, weight_buffer,
                                     output_buffer, kRows),
         "native HRX RMSNorm+SSM-QKV dispatch succeeds");
  HRX_CHECK(hrx_stream_synchronize(backend.Stream()));
  HRX_CHECK(hrx_synchronous_d2h(backend.Device(), output_buffer, 0,
                                output.data(), output.size() * sizeof(float)));

  const float expected =
      static_cast<float>(kHiddenSize) / std::sqrt(1.0F + kEpsilon);
  for (const float value : output) {
    Expect(std::abs(value - expected) < 0.02F,
           "native HRX RMSNorm+SSM-QKV matches the CPU oracle");
  }

  hrx_buffer_release(input_buffer);
  hrx_buffer_release(gamma_buffer);
  hrx_buffer_release(weight_buffer);
  hrx_buffer_release(output_buffer);
}

void TestQwenHrxExecutorMultiKernelAndGraph() {
  gufo::hrx::QwenHrxExecutor executor(Qwen38Contract(), 0);
  bool ready = executor.InitializeAllKernels(std::string(kHrxKernelDir));
  Expect(ready, "InitializeAllKernels loads the Nix-built artifacts");

  Expect(executor.IsReady(),
         "InitializeAllKernels marks the complete artifact set ready");
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
  TestRMSNormQKVParity();
  TestQwenHrxExecutorMultiKernelAndGraph();
  std::cout << "All qwen_hrx_executor_test assertions passed!\n";
  return 0;
}
