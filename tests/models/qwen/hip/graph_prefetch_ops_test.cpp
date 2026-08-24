#include <algorithm>
#include <cassert>
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

  strix::hip::detail::HipGraphDecodeExecutor executor;
  assert(executor.IsEnabled());
  assert(!executor.IsCaptured());

  // Test across 5 consecutive tokens with graph replay
  for (std::uint32_t step = 0; step < 5; ++step) {
    const std::uint32_t token_id = step * 10 + 3;
    const std::uint32_t pos = step;
    const std::uint32_t in_params[2] = {token_id, pos};
    HIP_CHECK(hipMemcpyAsync(d_params, in_params, sizeof(in_params),
                             hipMemcpyHostToDevice, stream));

    auto StepOps = [&]() {
      strix::hip::LaunchEmbeddingLookup(d_embd, strix::core::GgmlType::kF32,
                                        d_params + 0, d_hidden, hidden_size,
                                        stream);
      strix::hip::LaunchRMSNorm(d_hidden, d_weight, d_normed, hidden_size,
                                1e-6F, stream);
    };

    if (!executor.IsCaptured()) {
      const bool ok = executor.TryCapture(stream, StepOps);
      assert(ok);
      assert(executor.IsCaptured());
      const bool l_ok = executor.Launch(stream);
      assert(l_ok);
    } else {
      const bool ok = executor.Launch(stream);
      assert(ok);
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
      assert(std::abs(h_out[i] - expected) < 1e-4F);
    }
  }

  executor.Reset();
  assert(!executor.IsCaptured());

  HIP_CHECK(hipFree(d_embd));
  HIP_CHECK(hipFree(d_hidden));
  HIP_CHECK(hipFree(d_normed));
  HIP_CHECK(hipFree(d_weight));
  HIP_CHECK(hipFree(d_params));
  HIP_CHECK(hipStreamDestroy(stream));
}

void TestLayerWeightPrefetch() {
  // opt-c014-layer-prefetch: the async page-touch must never modify the buffer
  // and must be safe on a trailing partial page (bounds-clamped reads).
  constexpr std::size_t kBytes = 4096U * 3U + 123U;
  std::vector<unsigned char> h_buf(kBytes);
  for (std::size_t i = 0; i < kBytes; ++i) {
    h_buf[i] = static_cast<unsigned char>((i * 31U) & 0xFFU);
  }
  void* d_buf = nullptr;
  HIP_CHECK(hipMalloc(&d_buf, kBytes));
  HIP_CHECK(hipMemcpy(d_buf, h_buf.data(), kBytes, hipMemcpyHostToDevice));

  strix::hip::LaunchLayerWeightPrefetch(d_buf, kBytes);
  strix::hip::LaunchLayerWeightPrefetch(nullptr, kBytes);
  strix::hip::LaunchLayerWeightPrefetch(d_buf, 0);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<unsigned char> h_out(kBytes);
  HIP_CHECK(hipMemcpy(h_out.data(), d_buf, kBytes, hipMemcpyDeviceToHost));
  assert(h_out == h_buf);

  HIP_CHECK(hipFree(d_buf));
}

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cout
        << "No HIP device found, skipping Qwen graph prefetch ops test.\n";
    return 0;
  }

  TestHipGraphDecodeStep();
  TestLayerWeightPrefetch();
  std::cout << "Qwen graph prefetch ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen graph prefetch ops test.\n";
  return 0;
#endif
}
