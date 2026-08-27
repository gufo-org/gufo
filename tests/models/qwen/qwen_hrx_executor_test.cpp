#include "src/models/qwen/hrx/qwen_hrx_executor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hrx/qwen_hrx_arena.hpp"

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

gufo::hrx::HrxBufferBinding Binding(hrx_buffer_t buffer, std::size_t length,
                                    std::size_t offset = 0) {
  return {.buffer = buffer, .offset = offset, .length = length};
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
  Expect(!executor.PrototypeArtifactsReady(),
         "a single artifact does not mark the retained prototype set ready");
  Expect(!executor.MissingKernelArtifacts().empty(),
         "single-artifact initialization reports missing prototypes");
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

  // Use non-zero offsets to exercise imported-shard subrange dispatch.
  constexpr std::size_t kPrefixBytes = 256;
  const std::size_t input_bytes = K * sizeof(float);
  const std::size_t weight_bytes = M * K * sizeof(uint16_t);
  const std::size_t output_bytes = M * sizeof(float);
  hrx_buffer_t buf_x = nullptr, buf_gate = nullptr, buf_up = nullptr,
               buf_out = nullptr;
  HRX_CHECK(hrx_buffer_allocate(stream, kPrefixBytes + input_bytes,
                                HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                HRX_BUFFER_USAGE_DEFAULT, &buf_x));
  HRX_CHECK(hrx_buffer_allocate(stream, kPrefixBytes + weight_bytes,
                                HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                HRX_BUFFER_USAGE_DEFAULT, &buf_gate));
  HRX_CHECK(hrx_buffer_allocate(stream, kPrefixBytes + weight_bytes,
                                HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                HRX_BUFFER_USAGE_DEFAULT, &buf_up));
  HRX_CHECK(hrx_buffer_allocate(stream, kPrefixBytes + output_bytes,
                                HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                HRX_BUFFER_USAGE_DEFAULT, &buf_out));

  HRX_CHECK(hrx_synchronous_h2d(device, h_x.data(), buf_x, kPrefixBytes,
                                input_bytes));
  HRX_CHECK(hrx_synchronous_h2d(device, h_gate.data(), buf_gate, kPrefixBytes,
                                weight_bytes));
  HRX_CHECK(hrx_synchronous_h2d(device, h_up.data(), buf_up, kPrefixBytes,
                                weight_bytes));

  const auto input_binding = Binding(buf_x, input_bytes, kPrefixBytes);
  const auto gate_binding = Binding(buf_gate, weight_bytes, kPrefixBytes);
  const auto up_binding = Binding(buf_up, weight_bytes, kPrefixBytes);
  const auto output_binding = Binding(buf_out, output_bytes, kPrefixBytes);
  const auto undersized_output =
      Binding(buf_out, output_bytes - 1, kPrefixBytes);
  Expect(!executor.DispatchSwiGLU(input_binding, gate_binding, up_binding,
                                  undersized_output, M),
         "DispatchSwiGLU rejects an undersized binding before dispatch");
  Expect(executor.DispatchSwiGLU(input_binding, gate_binding, up_binding,
                                 output_binding, M),
         "DispatchSwiGLU preserves non-zero operand offsets");

  HRX_CHECK(hrx_stream_synchronize(stream));
  HRX_CHECK(hrx_synchronous_d2h(device, buf_out, kPrefixBytes, h_out_hrx.data(),
                                output_bytes));

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

void TestRMSNormSsmQkvParity() {
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
  HRX_CHECK(hrx_buffer_allocate(
      backend.Stream(), input.size() * sizeof(float),
      HRX_MEMORY_TYPE_DEVICE_LOCAL,
      HRX_BUFFER_USAGE_STORAGE | HRX_BUFFER_USAGE_TRANSFER, &input_buffer));
  HRX_CHECK(hrx_buffer_allocate(
      backend.Stream(), gamma.size() * sizeof(float),
      HRX_MEMORY_TYPE_DEVICE_LOCAL,
      HRX_BUFFER_USAGE_STORAGE | HRX_BUFFER_USAGE_TRANSFER, &gamma_buffer));
  HRX_CHECK(hrx_buffer_allocate(
      backend.Stream(), weights.size() * sizeof(uint16_t),
      HRX_MEMORY_TYPE_DEVICE_LOCAL,
      HRX_BUFFER_USAGE_STORAGE | HRX_BUFFER_USAGE_TRANSFER, &weight_buffer));
  HRX_CHECK(hrx_buffer_allocate(
      backend.Stream(), output.size() * sizeof(float),
      HRX_MEMORY_TYPE_DEVICE_LOCAL,
      HRX_BUFFER_USAGE_STORAGE | HRX_BUFFER_USAGE_TRANSFER, &output_buffer));
  HRX_CHECK(hrx_synchronous_h2d(backend.Device(), input.data(), input_buffer, 0,
                                input.size() * sizeof(float)));
  HRX_CHECK(hrx_synchronous_h2d(backend.Device(), gamma.data(), gamma_buffer, 0,
                                gamma.size() * sizeof(float)));
  HRX_CHECK(hrx_synchronous_h2d(backend.Device(), weights.data(), weight_buffer,
                                0, weights.size() * sizeof(uint16_t)));

  Expect(executor.DispatchRMSNormSsmQKV(
             Binding(input_buffer, input.size() * sizeof(float)),
             Binding(gamma_buffer, gamma.size() * sizeof(float)),
             Binding(weight_buffer, weights.size() * sizeof(uint16_t)),
             Binding(output_buffer, output.size() * sizeof(float)), kRows),
         "native HRX RMSNorm plus SSM-QKV dispatch succeeds");
  HRX_CHECK(hrx_stream_synchronize(backend.Stream()));
  HRX_CHECK(hrx_synchronous_d2h(backend.Device(), output_buffer, 0,
                                output.data(), output.size() * sizeof(float)));

  const float expected =
      static_cast<float>(kHiddenSize) / std::sqrt(1.0F + kEpsilon);
  for (const float value : output) {
    Expect(std::abs(value - expected) < 0.02F,
           "native HRX RMSNorm plus SSM-QKV matches the CPU oracle");
  }

  hrx_buffer_release(input_buffer);
  hrx_buffer_release(gamma_buffer);
  hrx_buffer_release(weight_buffer);
  hrx_buffer_release(output_buffer);
}

void TestNativeVectorPrimitives() {
  gufo::hrx::QwenHrxExecutor executor(Qwen38Contract(), 0);
  Expect(executor.InitializeAllKernels(std::string(kHrxKernelDir)),
         "native vector primitive artifacts load");
  auto& backend = executor.Backend();
  std::string error;
  constexpr std::size_t kFfn = 17408;
  const std::size_t bytes = kFfn * sizeof(float);
  auto a = gufo::hrx::HrxOwnedBuffer::Allocate(backend.Stream(), bytes, &error);
  auto b = gufo::hrx::HrxOwnedBuffer::Allocate(backend.Stream(), bytes, &error);
  auto c = gufo::hrx::HrxOwnedBuffer::Allocate(backend.Stream(), bytes, &error);

  Expect(a.has_value() && b.has_value() && c.has_value(),
         "native vector test buffers allocate");

  constexpr std::size_t kHidden = 5120;
  std::vector<float> host_a(kFfn, 1.0F);
  std::vector<float> host_b(kFfn, 2.0F);
  std::vector<float> host_out(kFfn, 0.0F);
  Expect(gufo::hrx::HrxCopyFromHost(backend.Device(), host_a.data(),
                                    a->Binding(), bytes, &error),
         "native vector left input uploads");
  Expect(gufo::hrx::HrxCopyFromHost(backend.Device(), host_b.data(),
                                    b->Binding(), bytes, &error),
         "native vector right input uploads");
  Expect(executor.DispatchResidualAdd(a->Binding(), b->Binding(), c->Binding(),
                                      kHidden),
         "native residual add dispatches");
  HRX_CHECK(hrx_stream_synchronize(backend.Stream()));
  Expect(
      gufo::hrx::HrxCopyToHost(backend.Device(), c->Binding(), host_out.data(),
                               kHidden * sizeof(float), &error),
      "native residual output downloads");

  for (std::size_t i = 0; i < kHidden; ++i) {
    Expect(std::fabs(host_out[i] - 3.0F) < 1e-5F,
           "native residual output matches nonzero oracle");
  }

  std::fill(host_a.begin(), host_a.end(), 0.0F);
  std::fill(host_b.begin(), host_b.end(), 3.0F);
  Expect(gufo::hrx::HrxCopyFromHost(backend.Device(), host_a.data(),
                                    a->Binding(), bytes, &error) &&
             gufo::hrx::HrxCopyFromHost(backend.Device(), host_b.data(),
                                        b->Binding(), bytes, &error),
         "native SwiGLU inputs upload");
  Expect(executor.DispatchSwiGLUPointwise(a->Binding(), b->Binding(),
                                          c->Binding(), kFfn),
         "native pointwise SwiGLU dispatches");
  HRX_CHECK(hrx_stream_synchronize(backend.Stream()));
  Expect(gufo::hrx::HrxCopyToHost(backend.Device(), c->Binding(),
                                  host_out.data(), bytes, &error),
         "native SwiGLU output downloads");
  for (const float value : host_out) {
    Expect(std::fabs(value) < 1e-6F,
           "native pointwise SwiGLU matches nonzero-up oracle");
  }

  std::fill(host_a.begin(), host_a.end(), 1.0F);
  std::fill(host_b.begin(), host_b.end(), 2.0F);
  Expect(
      gufo::hrx::HrxCopyFromHost(backend.Device(), host_a.data(), a->Binding(),
                                 kHidden * sizeof(float), &error) &&
          gufo::hrx::HrxCopyFromHost(backend.Device(), host_b.data(),
                                     b->Binding(), kHidden * sizeof(float),
                                     &error),
      "native RMSNorm inputs upload");

  Expect(executor.DispatchRMSNorm(a->Binding(), b->Binding(), c->Binding()),
         "native RMSNorm dispatches");
  HRX_CHECK(hrx_stream_synchronize(backend.Stream()));
  Expect(
      gufo::hrx::HrxCopyToHost(backend.Device(), c->Binding(), host_out.data(),
                               kHidden * sizeof(float), &error),
      "native RMSNorm output downloads");

  for (std::size_t i = 0; i < kHidden; ++i) {
    Expect(std::fabs(host_out[i] - 1.999999F) < 1e-4F,
           "native RMSNorm matches nonzero oracle");
  }
}

void TestQ8MathPrimitives() {
  gufo::hrx::QwenHrxExecutor executor(Qwen38Contract(), 0);
  Expect(executor.InitializeAllKernels(std::string(kHrxKernelDir)),
         "native Q8_0 artifacts load");
  Expect(executor.Q8MathReady(), "native Q8_0 capability is ready");
  Expect(executor.FfnStageReady(), "native Q8_0 FFN capability is ready");
  Expect(executor.AttentionStageReady(),
         "native causal attention capability is ready");
  Expect(executor.SsmStageReady(),
         "complete native DeltaNet capability is ready");
  Expect(executor.FinalStageReady(),
         "native final norm/logits/argmax capability is ready");

  auto& backend = executor.Backend();
  constexpr std::size_t kRows = 2;
  constexpr std::size_t kColumns = 5120;
  constexpr std::size_t kBlocksPerRow = kColumns / 32;
  constexpr std::size_t kPrefix = 256;
  using Q8Block = gufo::quant::block_q8_0;
  static_assert(sizeof(Q8Block) == 34);

  std::vector<Q8Block> weights(kRows * kBlocksPerRow);
  std::vector<float> input(kColumns);
  std::vector<float> expected(kRows, 0.0F);
  std::vector<float> actual(kRows, 0.0F);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = 0.125F * static_cast<float>(static_cast<int>(index % 7) - 3);
  }
  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t block = 0; block < kBlocksPerRow; ++block) {
      auto& value = weights[row * kBlocksPerRow + block];
      value.d = row == 0 ? 0x3800U : 0xB400U;  // +0.5 and -0.25.
      const float scale = gufo::quant::Fp16ToFloat(value.d);
      for (std::size_t lane = 0; lane < 32; ++lane) {
        value.qs[lane] = static_cast<std::int8_t>(
            static_cast<int>((row * 5 + block + lane) % 15) - 7);
        expected[row] += scale * static_cast<float>(value.qs[lane]) *
                         input[block * 32 + lane];
      }
    }
  }

  const std::size_t weight_bytes = weights.size() * sizeof(Q8Block);
  const std::size_t input_bytes = input.size() * sizeof(float);
  const std::size_t output_bytes = actual.size() * sizeof(float);
  hrx_buffer_t weight_buffer = nullptr;
  hrx_buffer_t input_buffer = nullptr;
  hrx_buffer_t output_buffer = nullptr;
  HRX_CHECK(hrx_buffer_allocate(backend.Stream(), kPrefix + weight_bytes,
                                HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                HRX_BUFFER_USAGE_DEFAULT, &weight_buffer));
  HRX_CHECK(hrx_buffer_allocate(backend.Stream(), kPrefix + input_bytes,
                                HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                HRX_BUFFER_USAGE_DEFAULT, &input_buffer));
  HRX_CHECK(hrx_buffer_allocate(
      backend.Stream(), kPrefix + kColumns * sizeof(float),
      HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &output_buffer));
  HRX_CHECK(hrx_synchronous_h2d(backend.Device(), weights.data(), weight_buffer,
                                kPrefix, weight_bytes));
  HRX_CHECK(hrx_synchronous_h2d(backend.Device(), input.data(), input_buffer,
                                kPrefix, input_bytes));

  const auto weight_binding = Binding(weight_buffer, weight_bytes, kPrefix);
  const auto input_binding = Binding(input_buffer, input_bytes, kPrefix);
  const auto gemv_output = Binding(output_buffer, output_bytes, kPrefix);
  Expect(executor.DispatchQ8Gemv(weight_binding, input_binding, gemv_output,
                                 kRows, kColumns),
         "native Q8_0 GEMV dispatches with nonzero offsets");
  HRX_CHECK(hrx_stream_synchronize(backend.Stream()));
  HRX_CHECK(hrx_synchronous_d2h(backend.Device(), output_buffer, kPrefix,
                                actual.data(), output_bytes));
  for (std::size_t row = 0; row < kRows; ++row) {
    Expect(std::isfinite(actual[row]) &&
               std::fabs(actual[row] - expected[row]) < 0.02F,
           "native Q8_0 GEMV matches signed non-unit-scale CPU oracle");
  }

  std::vector<float> embedding(kColumns, 0.0F);
  const auto embedding_output =
      Binding(output_buffer, kColumns * sizeof(float), kPrefix);
  Expect(executor.DispatchQ8Embedding(weight_binding, 1, embedding_output),
         "native Q8_0 embedding dispatches with a nonzero offset");
  HRX_CHECK(hrx_stream_synchronize(backend.Stream()));
  HRX_CHECK(hrx_synchronous_d2h(backend.Device(), output_buffer, kPrefix,
                                embedding.data(), kColumns * sizeof(float)));
  for (std::size_t index = 0; index < kColumns; ++index) {
    const auto& block = weights[kBlocksPerRow + index / 32];
    const float reference = gufo::quant::Fp16ToFloat(block.d) *
                            static_cast<float>(block.qs[index % 32]);
    Expect(std::fabs(embedding[index] - reference) < 1e-6F,
           "native Q8_0 embedding matches the CPU oracle");
  }

  const auto short_weight = Binding(weight_buffer, weight_bytes - 1, kPrefix);
  Expect(!executor.DispatchQ8Gemv(short_weight, input_binding, gemv_output,
                                  kRows, kColumns),
         "native Q8_0 GEMV rejects an undersized packed binding");

  hrx_buffer_release(weight_buffer);
  hrx_buffer_release(input_buffer);
  hrx_buffer_release(output_buffer);
}

void TestFullFfnParityWhenModelIsProvided() {
  const char* model_path = std::getenv("GUFO_HRX_Q8_MODEL");
  if (model_path == nullptr || *model_path == '\0') {
    std::cout << "Skipping full native FFN parity; set GUFO_HRX_Q8_MODEL\n";
    return;
  }

  std::string error;
  auto reader_owner = gufo::core::GgufReader::OpenFile(model_path, &error);
  Expect(reader_owner != nullptr, "FFN parity model opens");

      reader, 1, std::string(kHrxKernelDir), &error);
      Expect(executor != nullptr, "strict Q8_0 native model creates");
      Expect(executor->FfnStageReady(), "complete FFN artifacts are ready");

      constexpr std::size_t kHidden = 5120;
      constexpr std::size_t kFfn = 17408;
      std::vector<float> input(kHidden);
      for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] =
            0.01F * static_cast<float>(static_cast<int>(index % 17) - 8);
      }
      const auto hidden =
          executor->GetArenaBinding(gufo::hrx::QwenHrxArenaBuffer::kHidden);

      Expect(hidden.has_value(), "FFN parity exposes arena hidden binding");
      Expect(gufo::hrx::HrxCopyFromHost(executor->Backend().Device(),
                                        input.data(), *hidden,
                                        input.size() * sizeof(float), &error),

             "FFN parity hidden input uploads");
      Expect(executor->DispatchFfnQ8(0, &error),
             "complete native Q8_0 FFN dispatches");
      HRX_CHECK(hrx_stream_synchronize(executor->Backend().Stream()));
      std::vector<float> actual(kHidden);
      Expect(gufo::hrx::HrxCopyToHost(executor->Backend().Device(), *hidden,
                                      actual.data(),
                                      actual.size() * sizeof(float), &error),
             "FFN parity hidden output downloads");

      const auto weights =
          gufo::models::QwenModelWeights::LoadFromGguf(*reader, &error);

      Expect(weights.has_value() && !weights->layers.empty(),
             "FFN parity CPU weights load");
      const auto& layer_weights = weights->layers[0];
      float square_sum = 0.0F;
      for (const float value : input) {
        square_sum += value * value;
      }
      const float inverse_rms =
          1.0F / std::sqrt(square_sum / static_cast<float>(kHidden) + 1e-6F);
      std::vector<float> normed(kHidden);
      for (std::size_t column = 0; column < kHidden; ++column) {
        normed[column] =
            input[column] * inverse_rms * layer_weights.ffn_norm.Get(column);
      }
      std::vector<float> activation(kFfn);
      for (std::size_t row = 0; row < kFfn; ++row) {
        float gate = 0.0F;
        float up = 0.0F;
        const std::size_t base = row * kHidden;
        for (std::size_t column = 0; column < kHidden; ++column) {
          gate += layer_weights.ffn_gate.Get(base + column) * normed[column];
          up += layer_weights.ffn_up.Get(base + column) * normed[column];
        }
        activation[row] = (gate / (1.0F + std::exp(-gate))) * up;
      }
      std::vector<float> expected(kHidden);
      for (std::size_t row = 0; row < kHidden; ++row) {
        float projected = 0.0F;
        const std::size_t base = row * kFfn;
        for (std::size_t column = 0; column < kFfn; ++column) {
          projected +=
              layer_weights.ffn_down.Get(base + column) * activation[column];
        }
        expected[row] = input[row] + projected;
      }
      float max_relative_error = 0.0F;
      for (std::size_t index = 0; index < kHidden; ++index) {
        Expect(std::isfinite(actual[index]),
               "native FFN output remains finite");
        const float relative = std::fabs(actual[index] - expected[index]) /
                               (std::fabs(expected[index]) + 1e-4F);
        max_relative_error = std::max(max_relative_error, relative);
      }
      Expect(max_relative_error < 0.01F,
             "complete native FFN matches the CPU Q8_0 oracle");
}

void TestComposedNativeStagesWhenModelIsProvided() {
  const char* model_path = std::getenv("GUFO_HRX_Q8_MODEL");
  if (model_path == nullptr || *model_path == '\0') {
    std::cout << "Skipping composed native stage checks; set "
                 "GUFO_HRX_Q8_MODEL\n";
    return;
  }

  std::string error;
  auto reader_owner = gufo::core::GgufReader::OpenFile(model_path, &error);
  Expect(reader_owner != nullptr, "composed stage model opens");

      reader, 2, std::string(kHrxKernelDir), &error);
      Expect(executor != nullptr, "composed native stage executor creates");
      Expect(executor->ModelExecutionReady(),
             "strict Q8_0 model is ready when every native artifact loads");

      constexpr std::size_t kHidden = 5120;
      std::vector<float> input(kHidden);
      for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] =
            0.002F * static_cast<float>(static_cast<int>(index % 31) - 15);
      }
      const auto hidden =
          executor->GetArenaBinding(gufo::hrx::QwenHrxArenaBuffer::kHidden);

      Expect(hidden.has_value(), "composed stage hidden binding exists");

      const auto& weights = executor->GetConfig();
      std::size_t attention_layer = weights.num_layers;
      std::size_t ssm_layer = weights.num_layers;
      const auto loaded =
          gufo::models::QwenModelWeights::LoadFromGguf(*reader, &error);

      Expect(loaded.has_value(), "composed stage CPU metadata loads");
      for (std::size_t index = 0; index < loaded->layers.size(); ++index) {
        if (loaded->layers[index].is_full_attention &&
            attention_layer == weights.num_layers) {
          attention_layer = index;
        }
        if (!loaded->layers[index].is_full_attention &&
            ssm_layer == weights.num_layers) {
          ssm_layer = index;
        }
      }
      Expect(attention_layer < weights.num_layers &&
                 ssm_layer < weights.num_layers,
             "model exposes attention and SSM layers");

      Expect(gufo::hrx::HrxCopyFromHost(executor->Backend().Device(),
                                        input.data(), *hidden,
                                        input.size() * sizeof(float), &error),

             "attention stage hidden uploads");
      Expect(executor->DispatchAttentionQ8(attention_layer, 0, &error),
             "complete native attention stage dispatches at position zero");
      HRX_CHECK(hrx_stream_synchronize(executor->Backend().Stream()));
      std::vector<float> actual(kHidden);
      Expect(gufo::hrx::HrxCopyToHost(executor->Backend().Device(), *hidden,
                                      actual.data(),
                                      actual.size() * sizeof(float), &error),
             "attention stage output downloads");
      Expect(std::ranges::all_of(
                 actual, [](float value) { return std::isfinite(value); }),
             "attention stage output is finite");

      Expect(gufo::hrx::HrxCopyFromHost(executor->Backend().Device(),
                                        input.data(), *hidden,
                                        input.size() * sizeof(float), &error),

             "DeltaNet input uploads");
      Expect(executor->DispatchSsmQ8(ssm_layer, &error),
             "complete native DeltaNet stage dispatches and mutates state");
      HRX_CHECK(hrx_stream_synchronize(executor->Backend().Stream()));
      Expect(gufo::hrx::HrxCopyToHost(executor->Backend().Device(), *hidden,
                                      actual.data(),
                                      actual.size() * sizeof(float), &error),
             "DeltaNet output downloads");
      Expect(std::ranges::all_of(
                 actual, [](float value) { return std::isfinite(value); }),
             "DeltaNet output is finite");

      Expect(gufo::hrx::HrxCopyFromHost(executor->Backend().Device(),
                                        input.data(), *hidden,
                                        input.size() * sizeof(float), &error),

             "final stage hidden uploads");
      gufo::tokenization::TokenId token = weights.vocab_size;
      Expect(executor->DispatchFinalQ8(&token, &error),
             "native final norm, vocabulary GEMV, and argmax dispatch");
      Expect(token < weights.vocab_size,
             "native greedy argmax returns an in-vocabulary token");

      Expect(executor->Reset(&error), "native token sequencing reset succeeds");
      const auto next = executor->ForwardToken(1, 0, true, &error);
      Expect(next.has_value() && *next < weights.vocab_size,
             "bare-minimum native path produces one greedy token");
      const auto repeated = executor->ForwardToken(1, 0, false, &error);
      Expect(!repeated.has_value(),
             "native token path rejects a repeated sequence position");
      const auto no_logits = executor->ForwardToken(1, 1, false, &error);
      Expect(
          no_logits.has_value() && *no_logits == 0,
          "sequential native token without logits advances and returns zero");
}

void TestQwenHrxArenaLifecycle() {
  auto& backend = gufo::hrx::HrxBackend::Instance();
  Expect(backend.Initialize(0), "Backend initialize before HRX arena test");

  std::string error;
  auto arena = gufo::hrx::QwenHrxArena::Create(
      backend.Device(), backend.Stream(), Qwen38Contract(), 1, &error);
  Expect(arena.has_value(), "single-token HRX arena allocates");
  Expect(error.empty(), "successful HRX arena creation clears the error");
  Expect(arena->Layout().max_context == 1,
         "HRX arena retains requested context");
  Expect(arena->Binding(gufo::hrx::QwenHrxArenaBuffer::kHidden).length ==
             arena->Layout().hidden_bytes,
         "hidden arena binding exposes exact allocation size");
  Expect(arena->Binding(gufo::hrx::QwenHrxArenaBuffer::kKvCache).length ==
             arena->Layout().kv_cache_bytes,
         "KV arena binding exposes requested-context allocation");

                 32 * sizeof(float),
         "arena allocates one rotary coefficient row");

         "arena allocates 48-head recurrent state");
         Expect(arena->Reset(&error),
                "arena reset uses native graph fill without a host roundtrip");
         Expect(error.empty(), "successful arena reset clears the error");

         gufo::hrx::HrxModuleLoader loader;
         HRX_CHECK(loader.LoadFromFile(
             backend.Device(), "qwen_copy",
             std::string(kHrxKernelDir) + "/qwen_copy_f32.fb", "amdgpu",
             "gfx1151"));
         const auto copy_executable = loader.GetExecutable("qwen_copy");
         Expect(copy_executable != nullptr, "native state-copy artifact loads");
         Expect(arena->SaveState(copy_executable, &error),
                "arena snapshot uses native device copy");
         Expect(
             gufo::hrx::HrxFillBuffer(
                 backend.Device(), backend.Stream(),
                 arena->Binding(gufo::hrx::QwenHrxArenaBuffer::kSsmConvState),
                 0x3f800000U, &error),
             "live convolution state can be changed after snapshot");
         Expect(arena->RestoreState(copy_executable, &error),
                "arena restore uses native device copy");
         std::uint32_t restored_edge = 1;
         const auto recurrent =
             arena->Binding(gufo::hrx::QwenHrxArenaBuffer::kSsmConvState);
         const gufo::hrx::HrxBufferBinding first_word{
             .buffer = recurrent.buffer,
             .offset = recurrent.offset,
             .length = sizeof(restored_edge),
         };
         Expect(gufo::hrx::HrxCopyToHost(backend.Device(), first_word,
                                         &restored_edge, sizeof(restored_edge),
                                         &error),

                "restored state edge can be read back");
         Expect(restored_edge == 0, "restored state matches the zero snapshot");

         loader.UnloadAll();
         arena.reset();
         backend.Shutdown();
}

void TestPrototypeArtifactReadiness() {
  gufo::hrx::QwenHrxExecutor executor(Qwen38Contract(), 0);
  Expect(executor.InitializeAllKernels(std::string(kHrxKernelDir)),
         "retained HRX prototype artifacts load");
  Expect(executor.PrototypeArtifactsReady(),
         "retained prototype artifact set is ready");
  Expect(executor.MissingKernelArtifacts().empty(),
         "retained prototype artifact set reports no missing files");
  Expect(!executor.ModelExecutionReady(),
         "artifacts without strict model bindings do not enable execution");
}

}  // namespace

int main() {
  std::cout << "Running qwen_hrx_executor_test...\n";
  TestQwenHrxExecutorLifecycleAndDispatch();
  TestRMSNormSsmQkvParity();
  TestNativeVectorPrimitives();
  TestQ8MathPrimitives();
  TestFullFfnParityWhenModelIsProvided();
  TestComposedNativeStagesWhenModelIsProvided();
  TestQwenHrxArenaLifecycle();
  TestPrototypeArtifactReadiness();
  std::cout << "All qwen_hrx_executor_test assertions passed!\n";
  return 0;
}
