#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "src/models/qwen/hrx/qwen_hrx_executor.hpp"

int main() {
  std::printf("======================================================================\n");
  std::printf("  HRX Graph DAG Replay vs Sequential Stream Dispatch Benchmark\n");
  std::printf("  Target Silicon: AMD Radeon 8060S Graphics (gfx1151)\n");
  std::printf("======================================================================\n");

  strix::core::ModelConfig model_config;
  model_config.num_layers = 64;
  model_config.hidden_size = 5120;
  model_config.intermediate_size = 17408;
  model_config.num_attention_heads = 24;
  model_config.num_key_value_heads = 4;
  model_config.head_dim = 256;
  model_config.rotary_dim = 64;
  model_config.ssm_group_count = 16;
  model_config.ssm_state_size = 128;
  model_config.ssm_time_step_rank = 48;
  model_config.ssm_inner_size = 6144;
  model_config.vocab_size = 248320;
  const auto contract =
      strix::hrx::QwenHrxArtifactContract::FromConfig(model_config);
  if (!contract.has_value()) {
    std::fprintf(stderr, "Invalid Qwen3.8 HRX artifact contract!\n");
    return 1;
  }

  strix::hrx::QwenHrxExecutor executor(*contract, 0);
  bool ready =
      executor.InitializeAllKernels("build/hrx-test/share/strix/kernels");
  if (!ready) {
    std::fprintf(stderr, "Failed to initialize HRX executor and Loom kernels!\n");
    return 1;
  }

  auto& backend = executor.Backend();
  hrx_stream_t stream = backend.Stream();

  const uint32_t hidden_dim = 5120;
  const uint32_t intermediate_dim = 17408;
  const int warmups = 50;
  const int iters = 500;

  // Allocate device buffers
  hrx_buffer_t buf_hidden = nullptr, buf_attn_gamma = nullptr, buf_w_qkv = nullptr;
  hrx_buffer_t buf_cos = nullptr, buf_sin = nullptr, buf_q_out = nullptr;
  hrx_buffer_t buf_k_cache = nullptr, buf_v_cache = nullptr;
  hrx_buffer_t buf_ffn_gamma = nullptr, buf_w_down = nullptr, buf_out = nullptr;

  HRX_CHECK(hrx_buffer_allocate(stream, hidden_dim * sizeof(float), HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &buf_hidden));
  HRX_CHECK(hrx_buffer_allocate(stream, hidden_dim * sizeof(float), HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &buf_attn_gamma));
  HRX_CHECK(hrx_buffer_allocate(stream, contract->QkvWidth() * hidden_dim * sizeof(uint16_t), HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &buf_w_qkv));
  HRX_CHECK(hrx_buffer_allocate(stream, 64 * sizeof(float), HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &buf_cos));
  HRX_CHECK(hrx_buffer_allocate(stream, 64 * sizeof(float), HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &buf_sin));
  HRX_CHECK(hrx_buffer_allocate(stream, intermediate_dim * sizeof(float), HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &buf_q_out));
  HRX_CHECK(hrx_buffer_allocate(stream, contract->KWidth() * sizeof(float), HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &buf_k_cache));
  HRX_CHECK(hrx_buffer_allocate(stream, contract->VWidth() * sizeof(float), HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &buf_v_cache));
  HRX_CHECK(hrx_buffer_allocate(stream, intermediate_dim * sizeof(float), HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &buf_ffn_gamma));
  HRX_CHECK(hrx_buffer_allocate(stream, hidden_dim * intermediate_dim * sizeof(uint16_t), HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &buf_w_down));
  HRX_CHECK(hrx_buffer_allocate(stream, hidden_dim * sizeof(float), HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &buf_out));

  // Build and instantiate the Graph DAG
  bool graph_built = executor.BuildAndInstantiateDecodeGraph(
      buf_hidden, buf_attn_gamma, buf_w_qkv, buf_cos, buf_sin, buf_q_out,
      buf_k_cache, buf_v_cache, buf_ffn_gamma, buf_w_down, buf_out);

  if (!graph_built) {
    std::fprintf(stderr, "Failed to build and instantiate HRX Decode Graph DAG!\n");
    return 1;
  }

  // Warmup sequential stream dispatches
  for (int i = 0; i < warmups; ++i) {
    executor.DispatchLayerAttention(buf_hidden, buf_attn_gamma, buf_w_qkv, buf_cos, buf_sin, buf_q_out, buf_k_cache, buf_v_cache);
    executor.DispatchLayerFFN(buf_q_out, buf_ffn_gamma, buf_w_down, buf_hidden, buf_out);
  }
  HRX_CHECK(hrx_stream_synchronize(stream));

  // Benchmark sequential stream dispatches
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    executor.DispatchLayerAttention(buf_hidden, buf_attn_gamma, buf_w_qkv, buf_cos, buf_sin, buf_q_out, buf_k_cache, buf_v_cache);
    executor.DispatchLayerFFN(buf_q_out, buf_ffn_gamma, buf_w_down, buf_hidden, buf_out);
  }
  HRX_CHECK(hrx_stream_synchronize(stream));
  auto t1 = std::chrono::high_resolution_clock::now();
  double seq_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

  // Warmup HRX Graph execution
  for (int i = 0; i < warmups; ++i) {
    executor.ExecuteDecodeGraph();
  }
  HRX_CHECK(hrx_stream_synchronize(stream));

  // Benchmark HRX Graph execution
  t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    executor.ExecuteDecodeGraph();
  }
  HRX_CHECK(hrx_stream_synchronize(stream));
  t1 = std::chrono::high_resolution_clock::now();
  double graph_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

  double delta_us = seq_us - graph_us;
  double speedup_pct = (delta_us / seq_us) * 100.0;

  std::printf("| Mode                     | Latency / Step | Full Layer Delta (64L) | Speedup |\n");
  std::printf("|:-------------------------|:--------------:|:----------------------:|:-------:|\n");
  std::printf("| Sequential Dispatch (HRX)| %11.2f us |            -           |    -    |\n", seq_us);
  std::printf("| Graph DAG Exec (libhrx)  | %11.2f us |      %8.2f ms       | +%.1f%%  |\n",
              graph_us, (delta_us * 64.0) / 1000.0, speedup_pct);
  std::printf("======================================================================\n");

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

  return 0;
}
