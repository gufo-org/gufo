#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
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
#include "src/core/hip/snapshot_transfer.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/families/qwen/hip_common/attention_policy.hpp"
#include "src/families/qwen/hip_common/execution_policy.hpp"
#include "src/families/qwen/kernels/ops.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/modules/ffn.hpp"
#include "src/models/qwen/modules/layer_view.hpp"
#include "src/models/qwen/modules/module_ctx.hpp"
#include "src/models/qwen/modules/norm.hpp"
#include "src/models/qwen/modules/quant_gemm.hpp"
#include "src/models/qwen/modules/residual.hpp"
#include "tests/models/qwen/hip/support/bfloat16.hpp"
#include "tests/models/qwen/hip/support/comparisons.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/models/qwen/hip/support/device_buffer.hpp"
#include "tests/models/qwen/support/synthetic_weights.hpp"

void TestHipGraphDecodeStep() {
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

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

  bool threw = false;
  try {
    executor.TryCapture(stream, graph_key,
                        [] { HIP_CHECK(hipErrorOutOfMemory); });
  } catch (const std::runtime_error&) {
    threw = true;
  }
  gufo::test::Expect(threw, "capture failure must propagate");
  hipStreamCaptureStatus capture = hipStreamCaptureStatusActive;
  HIP_CHECK(hipStreamIsCapturing(stream, &capture));
  gufo::test::Expect(capture == hipStreamCaptureStatusNone,
                     "failed capture must release the stream");

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
      const bool ok = executor.TryCapture(stream, graph_key, [&] {
        StepOps();
        // A different request may allocate/copy a snapshot during capture.
        // Global capture or a blocking stream makes these independent copies
        // invalidate the decode graph (previously hidden by HIP_CHECK).
        std::async(std::launch::async, [&] {
          void* independent = nullptr;
          HIP_CHECK(hipMalloc(&independent, hidden_size * sizeof(float)));
          try {
            gufo::hip::SnapshotTransfer transfer;
            transfer.Copy(independent, h_w.data(), hidden_size * sizeof(float),
                          hipMemcpyHostToDevice);
            std::vector<float> restored(hidden_size);
            transfer.Copy2D(restored.data(), 64 * sizeof(float), independent,
                            64 * sizeof(float), 64 * sizeof(float),
                            hidden_size / 64);
            gufo::test::Expect(restored == h_w,
                               "snapshot copy during capture changed values");
          } catch (...) {
            (void)hipFree(independent);
            throw;
          }
          HIP_CHECK(hipFree(independent));
        }).get();
      });
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

// A captured decode step bakes the recurrent-state pointers of the pass it
// recorded, so the graph key carries QwenGpuArena::RecurrentIoIdentity. The
// pass after a rollback restore uses the same buffer pair as the pass after
// the pin, so the graph captured then replays with valid pointers and must
// reproduce the eager result bit for bit; the in-place pass between them has a
// different identity and must miss.
void TestHipGraphRecurrentFork() {
  using gufo::hip::QwenGpuArena;
  using gufo::test::DeviceBuffer;
  using gufo::test::Expect;
  const auto config = gufo::models::qwen::make_small_qwen_config();
  const std::uint32_t heads = config.ssm_time_step_rank;
  const std::uint32_t key_heads = config.ssm_group_count;
  const std::uint32_t key_dim = config.ssm_state_size;
  const std::uint32_t val_dim = config.SsmValueSize();
  const std::size_t qkv = config.SsmQkvSize();
  const std::size_t inner = config.ssm_inner_size;
  const std::size_t layers = config.SsmLayerCount();
  const std::size_t conv_elements = layers * qkv * 4;
  const std::size_t delta_elements = layers * heads * key_dim * val_dim;
  const auto values = [](std::size_t size, float scale, float phase) {
    std::vector<float> result(size);
    for (std::size_t i = 0; i < size; ++i)
      result[i] = scale * std::sin(phase + static_cast<float>(i) * 0.013F);
    return result;
  };
  DeviceBuffer<float> input(values(qkv, 1.0F, 1)),
      weights(values(qkv * 4, 0.3F, 2)), alpha(values(heads, 1.0F, 3)),
      beta(values(heads, 1.0F, 4)), decay(std::vector<float>(heads, -0.15F)),
      dt(values(heads, 0.1F, 5)), norm(std::vector<float>(val_dim, 1.0F)),
      gate(values(inner, 0.5F, 6)), conv_output(qkv), output(inner);
  const auto conv_seed = values(conv_elements, 0.3F, 7);
  const auto delta_seed = values(delta_elements, 0.02F, 8);
  const auto seed = [&](QwenGpuArena& arena) {
    HIP_CHECK(hipMemcpyAsync(arena.d_ssm_conv_state, conv_seed.data(),
                             conv_elements * sizeof(float),
                             hipMemcpyHostToDevice, arena.stream));
    HIP_CHECK(hipMemcpyAsync(arena.d_ssm_deltanet_state, delta_seed.data(),
                             delta_elements * sizeof(float),
                             hipMemcpyHostToDevice, arena.stream));
    HIP_CHECK(hipStreamSynchronize(arena.stream));
  };
  const auto state_bytes = [&](const QwenGpuArena& arena) {
    HIP_CHECK(hipStreamSynchronize(arena.stream));
    std::vector<float> state(conv_elements + delta_elements);
    HIP_CHECK(hipMemcpy(state.data(), arena.d_ssm_conv_state,
                        conv_elements * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(state.data() + conv_elements,
                        arena.d_ssm_deltanet_state,
                        delta_elements * sizeof(float), hipMemcpyDeviceToHost));
    return state;
  };
  // One decode row through every recurrent layer; the graph-captured part of
  // the decode step that touches the state.
  const auto step = [&](QwenGpuArena& arena,
                        gufo::hip::QwenRecurrentStateIo state) {
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
      gufo::hip::LaunchSSMConvRecurrence(
          input.data(), weights.data(), state, conv_output.data(), alpha.data(),
          beta.data(), decay.data(), dt.data(), norm.data(), gate.data(),
          output.data(), layer, qkv, key_heads, heads, key_dim, val_dim,
          arena.stream);
    }
  };
  const auto key_for = [](const QwenGpuArena& arena) {
    return gufo::hip::detail::HipGraphCaptureKey{
        .execution_identity = 11,
        .workload_identity = gufo::hip::ExtendQwenGraphWorkloadIdentity(
            29, arena.RecurrentIoIdentity()),
    };
  };

  QwenGpuArena reference(config, 16);
  seed(reference);
  step(reference, reference.BeginRecurrentWrite());
  const auto expected = state_bytes(reference);

  QwenGpuArena arena(config, 16);
  gufo::hip::detail::HipGraphDecodeExecutor executor;
  seed(arena);
  const auto seeded = state_bytes(arena);
  arena.SaveState(1);
  const auto pinned_pass = arena.BeginRecurrentWrite();
  const auto pinned_key = key_for(arena);
  Expect(pinned_pass.Diverges(), "pass after the pin must diverge");
  Expect(executor.TryCapture(arena.stream, pinned_key,
                             [&] { step(arena, pinned_pass); }),
         "HIP graph capture of the diverging pass failed");
  Expect(executor.Launch(arena.stream, pinned_key),
         "HIP graph launch of the diverging pass failed");
  Expect(state_bytes(arena) == expected,
         "captured diverging pass changed the recurrence");

  // The next pass is in place on the other buffer: a different key.
  const auto in_place_pass = arena.BeginRecurrentWrite();
  const auto in_place_key = key_for(arena);
  Expect(!in_place_pass.Diverges(), "second pass after a pin must be in place");
  Expect(in_place_key != pinned_key && !executor.IsCapturedFor(in_place_key),
         "in-place pass must not reuse the diverging graph");

  // After a restore the pair is the one the graph recorded: replay it.
  arena.RestoreState();
  Expect(state_bytes(arena) == seeded, "restore did not return the pin");
  const auto replay_pass = arena.BeginRecurrentWrite();
  Expect(replay_pass.conv_in == pinned_pass.conv_in &&
             replay_pass.conv_out == pinned_pass.conv_out &&
             replay_pass.recurrent_in == pinned_pass.recurrent_in &&
             replay_pass.recurrent_out == pinned_pass.recurrent_out,
         "pass after a restore must reuse the post-pin buffer pair");
  Expect(executor.IsCapturedFor(key_for(arena)),
         "graph key must match across the restore");
  // The destination still holds the first pass's result; poison it so the
  // replay has to produce the bytes again.
  HIP_CHECK(hipMemsetAsync(replay_pass.conv_out, 0x5A,
                           conv_elements * sizeof(float), arena.stream));
  HIP_CHECK(hipMemsetAsync(replay_pass.recurrent_out, 0x5A,
                           delta_elements * sizeof(float), arena.stream));
  Expect(executor.Launch(arena.stream, key_for(arena)),
         "HIP graph replay after the restore failed");
  Expect(state_bytes(arena) == expected,
         "graph replay after the restore changed the recurrence");
  arena.RestoreState();
  Expect(state_bytes(arena) == seeded,
         "graph replay wrote into the pinned buffer");
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
  TestHipGraphRecurrentFork();
  std::cout << "Qwen graph ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen graph ops test.\n";
  return 77;
#endif
}
