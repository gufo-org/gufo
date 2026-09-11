#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/detail/hip_graph_decode_executor.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/modules/ffn.hpp"
#include "src/models/qwen/modules/layer_view.hpp"
#include "src/models/qwen/modules/module_ctx.hpp"
#include "src/models/qwen/modules/norm.hpp"
#include "src/models/qwen/modules/quant_gemm.hpp"
#include "src/models/qwen/modules/residual.hpp"
#include "tests/models/qwen/hip/support/bfloat16.hpp"
#include "tests/models/qwen/hip/support/comparisons.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

void TestHipGraphDecodeStep() {
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  constexpr std::size_t hidden_size = 256;
  constexpr std::size_t vocab_size = 1024;
  std::vector<float> h_embd(vocab_size * hidden_size, 1.0F);
  for (std::size_t i = 0; i < vocab_size * hidden_size; ++i) {
    h_embd[i] = 0.01F * static_cast<float>(i % 37);
  }

  void* d_embd = nullptr;
  float* d_hidden = nullptr;
  float* d_normed = nullptr;
  float* d_weight = nullptr;
  std::uint32_t* d_params = nullptr;

  HIP_CHECK(hipMalloc(&d_embd, vocab_size * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_hidden, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_normed, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_weight, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_params, 4 * sizeof(std::uint32_t)));

  std::vector<float> h_w(hidden_size, 1.5F);
  HIP_CHECK(hipMemcpy(d_embd, h_embd.data(),
                      vocab_size * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_weight, h_w.data(), hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));

  gufo::hip::detail::HipGraphDecodeExecutor executor;
  constexpr gufo::hip::detail::HipGraphCaptureKey graph_key{
      .execution_identity = 11,
      .workload_identity = 29,
  };
  constexpr gufo::hip::detail::HipGraphCaptureKey mismatched_key{
      .execution_identity = 11,
      .workload_identity = 31,
  };
  gufo::test::Expect(executor.IsEnabled(), "HIP graph executor disabled");
  gufo::test::Expect(!executor.IsCaptured(),
                     "new HIP graph executor is already captured");

  // Test across 5 consecutive tokens with graph replay
  for (std::uint32_t step = 0; step < 5; ++step) {
    const std::uint32_t token_id = step * 10 + 3;
    const std::uint32_t pos = step;
    const std::uint32_t in_params[2] = {token_id, pos};
    HIP_CHECK(hipMemcpyAsync(d_params, in_params, sizeof(in_params),
                             hipMemcpyHostToDevice, stream));

    auto StepOps = [&]() {
      gufo::hip::LaunchEmbeddingLookup(d_embd, gufo::core::GgmlType::kF32,
                                       d_params + 0, d_hidden, hidden_size,
                                       stream);
      gufo::hip::LaunchRMSNorm(d_hidden, d_weight, d_normed, hidden_size, 1e-6F,
                               stream);
    };

    if (!executor.IsCaptured()) {
      const bool ok = executor.TryCapture(stream, graph_key, StepOps);
      gufo::test::Expect(ok, "HIP graph capture failed");
      gufo::test::Expect(executor.IsCapturedFor(graph_key),
                         "HIP graph capture key was not retained");
      gufo::test::Expect(!executor.IsCapturedFor(mismatched_key),
                         "mismatched HIP graph key was accepted");
      gufo::test::Expect(!executor.Launch(stream, mismatched_key),
                         "HIP graph replay accepted a mismatched key");
      const bool launch_ok = executor.Launch(stream, graph_key);
      gufo::test::Expect(launch_ok, "HIP graph launch failed");
    } else {
      const bool ok = executor.Launch(stream, graph_key);
      gufo::test::Expect(ok, "HIP graph replay failed");
    }

    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<float> h_out(hidden_size);
    HIP_CHECK(hipMemcpy(h_out.data(), d_normed, hidden_size * sizeof(float),
                        hipMemcpyDeviceToHost));

    // Verify RMSNorm output
    float sum_sq = 0.0F;
    for (std::size_t i = 0; i < hidden_size; ++i) {
      const float val = h_embd[token_id * hidden_size + i];
      sum_sq += val * val;
    }
    const float rms =
        std::sqrt(sum_sq / static_cast<float>(hidden_size) + 1e-6F);
    for (std::size_t i = 0; i < hidden_size; ++i) {
      const float expected = (h_embd[token_id * hidden_size + i] / rms) * 1.5F;
      gufo::test::ExpectNear(expected, h_out[i], 1e-4F,
                             "captured RMSNorm result mismatch");
    }
  }

  executor.Reset();
  gufo::test::Expect(!executor.IsCaptured(),
                     "HIP graph reset retained capture state");
  gufo::test::Expect(!executor.IsCapturedFor(graph_key),
                     "HIP graph reset retained capture identity");

  HIP_CHECK(hipFree(d_embd));
  HIP_CHECK(hipFree(d_hidden));
  HIP_CHECK(hipFree(d_normed));
  HIP_CHECK(hipFree(d_weight));
  HIP_CHECK(hipFree(d_params));
  HIP_CHECK(hipStreamDestroy(stream));
}

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int gate = gufo::test::GateHipDevice(
      gufo::test::HipDeviceRequirement::kOptional, "Qwen graph ops test");
  if (gate != gufo::test::kHipTestSuccess) {
    return gate;
  }

  TestHipGraphDecodeStep();
  std::cout << "Qwen graph ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen graph ops test.\n";
  return 77;
#endif
}
