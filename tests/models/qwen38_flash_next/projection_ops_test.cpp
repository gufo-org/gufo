#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

float Uniform(std::uint32_t* state, float scale) {
  return scale *
         static_cast<float>(static_cast<int>(NextRandom(state) & 0xFFFFU) -
                            32768) /
         32768.0F;
}

/// Row-major Q8_0 weights [m][k] and their dequantized values.
struct Q8Weights {
  std::vector<std::uint8_t> blocks;
  std::vector<float> values;
};

Q8Weights MakeWeights(std::size_t m, std::size_t k, std::uint32_t seed,
                      bool reference_values = true) {
  Q8Weights w;
  w.blocks.resize(m * (k / 32) * 34);
  if (reference_values)
    w.values.resize(m * k);
  for (std::size_t r = 0; r < m; ++r) {
    for (std::size_t b = 0; b < k / 32; ++b) {
      float raw[32];
      float max_abs = 0.0F;
      for (float& v : raw) {
        v = Uniform(&seed, 1.0F);
        max_abs = std::max(max_abs, std::abs(v));
      }
      const __half d = __float2half(max_abs / 127.0F);
      const float df = __half2float(d);
      std::uint8_t* block = w.blocks.data() + (r * (k / 32) + b) * 34;
      std::memcpy(block, &d, 2);
      for (std::size_t i = 0; i < 32; ++i) {
        const auto qv = static_cast<std::int8_t>(
            std::lround(df != 0.0F ? raw[i] / df : 0.0F));
        block[2 + i] = static_cast<std::uint8_t>(qv);
        if (reference_values)
          w.values[r * k + b * 32 + i] = df * static_cast<float>(qv);
      }
    }
  }
  return w;
}

// Reuse the wide projection fixture to check the fused convolution through
// its actual consumer, including the state carried into the next chunk.
void CheckSsmProjection(const void* w, const __half* x, float* projected,
                        const std::vector<float>& reference,
                        std::uint32_t batch) {
  constexpr std::uint32_t m = 16384, k = 2560, channels = 10240;
  constexpr std::uint32_t kh = 16, vh = 48, d = 128, value_dim = vh * d;
  const std::size_t conv_count = std::size_t(batch) * channels;
  const std::size_t state_count = std::size_t(vh) * d * d;
  const std::size_t out_count = std::size_t(batch) * value_dim;
  struct Buffers {
    std::vector<float*> pointers;
    ~Buffers() {
      for (float* p : pointers) {
        (void)hipFree(p);
      }
    }
    float* Make(std::size_t count) {
      float* p = nullptr;
      CheckHip(hipMalloc(&p, count * sizeof(float)), "SSM allocation");
      pointers.push_back(p);
      return p;
    }
  } buffers;
  std::uint32_t seed = 0x349B71U;
  auto values = [&](std::size_t count, float scale, float offset = 0.0F) {
    std::vector<float> v(count);
    for (float& x : v) {
      x = offset + Uniform(&seed, scale);
    }
    return v;
  };
  auto upload = [](float* p, const std::vector<float>& v) {
    CheckHip(
        hipMemcpy(p, v.data(), v.size() * sizeof(float), hipMemcpyHostToDevice),
        "SSM upload");
  };
  auto input = [&](std::size_t count, float scale, float offset = 0.0F) {
    float* p = buffers.Make(count);
    upload(p, values(count, scale, offset));
    return p;
  };
  auto download = [](const float* p, std::size_t count) {
    std::vector<float> v(count);
    CheckHip(
        hipMemcpy(v.data(), p, count * sizeof(float), hipMemcpyDeviceToHost),
        "SSM download");
    return v;
  };
  auto exact = [&](const std::vector<float>& expected, const float* actual,
                   const char* name) {
    const auto result = download(actual, expected.size());
    if (std::memcmp(expected.data(), result.data(),
                    expected.size() * sizeof(float)) != 0) {
      throw std::runtime_error(std::string("fused SSM changed ") + name);
    }
    for (float value : result) {
      if (!std::isfinite(value)) {
        throw std::runtime_error(std::string("nonfinite SSM ") + name);
      }
    }
  };
  float* conv_w = input(channels * 4, 0.05F);
  float* ab = input(std::size_t(batch) * 2 * vh, 1.0F);
  float* a = input(vh, 0.5F, -1.5F);
  float* dt = input(vh, 1.0F);
  float* norm = input(d, 0.5F, 1.0F);
  const auto history = values(channels * 3, 0.5F);
  const auto initial_state = values(state_count, 0.05F);
  float* conv_state = buffers.Make(history.size());
  float* state = buffers.Make(state_count);
  float* scratch = buffers.Make(conv_count + channels * 4);
  float* qn = buffers.Make(std::size_t(batch) * kh * d);
  float* kn = buffers.Make(std::size_t(batch) * kh * d);
  float* raw = buffers.Make(out_count);
  float* out = buffers.Make(out_count);
  float* fused = buffers.Make(reference.size() + 8);
  auto consume = [&](float* qkvz, bool convolved) {
    q::GatedDeltaNet(qkvz, m, qkvz + channels, m, ab, conv_w, a, dt, norm,
                     conv_state, scratch, qn, kn, raw, state, out, nullptr, {},
                     {}, batch, kh, vh, d, 4, true, convolved, 1e-6F, nullptr);
  };
  upload(conv_state, history);
  upload(state, initial_state);
  consume(projected, false);
  const auto expected_conv = download(scratch, conv_count);
  const auto expected_history = download(conv_state, history.size());
  const auto expected_state = download(state, state_count);
  const auto expected_out = download(out, out_count);
  for (int replay = 0; replay < 2; ++replay) {
    upload(conv_state, history);
    upload(state, initial_state);
    CheckHip(hipMemset(fused, 0xFF, (reference.size() + 8) * sizeof(float)),
             "poison fused projection");
    CheckHip(hipMemset(scratch, 0xFF, conv_count * sizeof(float)),
             "poison fused convolution");
    if (!q::DenseF16SsmGemm(w, x, conv_w, conv_state, fused, scratch, batch, m,
                            k, channels, 4, nullptr)) {
      throw std::runtime_error("SSM projection rejected the shape");
    }
    exact(expected_conv, scratch, "convolution");
    const auto projected = download(fused, reference.size() + 8);
    for (std::size_t t = 0; t < batch; ++t) {
      const std::size_t first = t + 3 >= batch ? 0 : channels;
      if (std::memcmp(projected.data() + t * m + first,
                      reference.data() + t * m + first,
                      (m - first) * sizeof(float)) != 0) {
        throw std::runtime_error("SSM projection changed Z or final QKV");
      }
    }
    for (std::size_t i = reference.size(); i < projected.size(); ++i) {
      if (std::bit_cast<std::uint32_t>(projected[i]) != 0xFFFFFFFFU) {
        throw std::runtime_error("SSM projection overwrote its guard");
      }
    }
    consume(fused, true);
    exact(expected_history, conv_state, "history");
    exact(expected_state, state, "recurrent state");
    exact(expected_out, out, "output");
  }
  // A rejected geometry must not touch any pointer.
  if (q::DenseF16SsmGemm(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         1023, m, k, channels, 4, nullptr) ||
      q::DenseF16SsmGemm(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         batch, m, k, channels, 3, nullptr)) {
    throw std::runtime_error("SSM projection accepted an unsupported shape");
  }
  std::cout << "fused SSM convolution, output, state and replay are exact\n";
}

void CheckAttentionProjection(const void* weights, const __half* input,
                              const float* projected, std::uint32_t batch) {
  constexpr std::uint32_t start = 131069;
  const std::size_t query_bytes = (std::size_t(batch) * 6144 + 16) * 4;
  const std::size_t cache_bytes =
      (std::size_t(start + batch) * 512 + 16) * sizeof(__half);
  struct Buffers {
    std::vector<void*> pointers;
    ~Buffers() {
      for (void* p : pointers)
        (void)hipFree(p);
    }
    void* Make(std::size_t bytes) {
      void* p = nullptr;
      CheckHip(hipMalloc(&p, bytes), "attention allocation");
      pointers.push_back(p);
      CheckHip(hipMemset(p, 0xA5, bytes), "attention guards");
      return p;
    }
  } buffers;
  void* expected[4];
  void* actual[4];
  for (unsigned i = 0; i < 4; ++i) {
    const auto bytes = i < 2 ? query_bytes : cache_bytes;
    expected[i] = buffers.Make(bytes);
    actual[i] = buffers.Make(bytes);
  }
  auto* position = static_cast<std::uint32_t*>(buffers.Make(4));
  CheckHip(hipMemcpy(position, &start, 4, hipMemcpyHostToDevice),
           "attention position");
  std::vector<float> gamma(512);
  std::uint32_t seed = 0x34185A9U;
  for (float& value : gamma)
    value = 1.0F + Uniform(&seed, 0.5F);
  auto* device_gamma = static_cast<float*>(buffers.Make(gamma.size() * 4));
  CheckHip(hipMemcpy(device_gamma, gamma.data(), gamma.size() * 4,
                     hipMemcpyHostToDevice),
           "attention norm weights");
  if (!q::PrepareAttention(projected, 13312, device_gamma, device_gamma + 256,
                           static_cast<float*>(expected[0]) + 8,
                           static_cast<float*>(expected[1]) + 8,
                           static_cast<__half*>(expected[2]) + 8,
                           static_cast<__half*>(expected[3]) + 8, batch, 24, 2,
                           256, 64, position, 1e7F, 1e-6F, nullptr))
    throw std::runtime_error("attention preparation rejected the model shape");
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t replay = nullptr;
  CheckHip(hipStreamCreate(&stream), "attention stream");
  CheckHip(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
           "attention capture");
  if (!q::AttentionF16Gemm(weights, input, device_gamma, device_gamma + 256,
                           static_cast<float*>(actual[0]) + 8,
                           static_cast<float*>(actual[1]) + 8,
                           static_cast<__half*>(actual[2]) + 8,
                           static_cast<__half*>(actual[3]) + 8, batch, position,
                           1e7F, 1e-6F, stream))
    throw std::runtime_error("attention fusion rejected the model shape");
  CheckHip(hipStreamEndCapture(stream, &graph), "attention capture end");
  CheckHip(hipGraphInstantiate(&replay, graph, nullptr, nullptr, 0),
           "attention graph");
  for (unsigned repetition = 0; repetition < 2; ++repetition) {
    for (unsigned i = 0; i < 4; ++i)
      CheckHip(hipMemsetAsync(actual[i], 0xA5,
                              i < 2 ? query_bytes : cache_bytes, stream),
               "attention replay guards");
    CheckHip(hipGraphLaunch(replay, stream), "attention replay");
    CheckHip(hipStreamSynchronize(stream), "attention wait");
    for (unsigned i = 0; i < 4; ++i) {
      const auto bytes = i < 2 ? query_bytes : cache_bytes;
      std::vector<std::byte> a(bytes), b(bytes);
      CheckHip(hipMemcpy(a.data(), expected[i], bytes, hipMemcpyDeviceToHost),
               "attention reference download");
      CheckHip(hipMemcpy(b.data(), actual[i], bytes, hipMemcpyDeviceToHost),
               "attention output download");
      if (a != b)
        throw std::runtime_error(
            "attention fusion changed rounding, cache boundaries or replay");
    }
  }
  if (q::AttentionF16Gemm(weights, input, device_gamma, device_gamma + 256,
                          static_cast<float*>(actual[0]) + 8,
                          static_cast<float*>(actual[1]) + 8,
                          static_cast<__half*>(actual[2]) + 8,
                          static_cast<__half*>(actual[3]) + 8, 1023, position,
                          1e7F, 1e-6F, nullptr))
    throw std::runtime_error("attention fusion accepted a short batch");
  CheckHip(hipGraphExecDestroy(replay), "attention graph free");
  CheckHip(hipGraphDestroy(graph), "attention graph definition free");
  CheckHip(hipStreamDestroy(stream), "attention stream free");
}

void CheckHcDownProjection(const void* w, const void* tiled,
                           const float* projected, std::uint32_t batch) {
  const std::size_t count = std::size_t(batch) * 320;
  float* activated = nullptr;
  __half* reference = nullptr;
  __half* output = nullptr;
  CheckHip(hipMalloc(&activated, count * sizeof(float)), "HC allocation");
  CheckHip(hipMalloc(&reference, (count + 16) * sizeof(__half)),
           "HC reference");
  CheckHip(hipMalloc(&output, (count + 16) * sizeof(__half)), "HC output");
  CheckHip(hipMemset(reference, 0xA5, (count + 16) * sizeof(__half)),
           "HC guards");
  CheckHip(hipMemcpy(activated, projected, count * sizeof(float),
                     hipMemcpyDeviceToDevice),
           "HC projection copy");
  q::SiluScale(activated, 0.25F, count, nullptr);
  q::NarrowActivations(activated, reference + 8, false, count, nullptr);
  std::vector<__half> expected(count + 16), actual(count + 16);
  CheckHip(hipMemcpy(expected.data(), reference,
                     expected.size() * sizeof(__half), hipMemcpyDeviceToHost),
           "HC reference download");
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t replay = nullptr;
  CheckHip(hipStreamCreate(&stream), "HC stream");
  CheckHip(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
           "HC capture");
  if (!q::HcDownF16Gemm(w, tiled, output + 8, batch, stream))
    throw std::runtime_error("fused HC down rejected the model shape");
  CheckHip(hipStreamEndCapture(stream, &graph), "HC capture end");
  CheckHip(hipGraphInstantiate(&replay, graph, nullptr, nullptr, 0),
           "HC graph");
  for (unsigned repetition = 0; repetition < 2; ++repetition) {
    CheckHip(
        hipMemsetAsync(output, 0xA5, actual.size() * sizeof(__half), stream),
        "HC output guards");
    CheckHip(hipGraphLaunch(replay, stream), "HC replay");
    CheckHip(
        hipMemcpyAsync(actual.data(), output, actual.size() * sizeof(__half),
                       hipMemcpyDeviceToHost, stream),
        "HC download");
    CheckHip(hipStreamSynchronize(stream), "HC wait");
    if (std::memcmp(actual.data(), expected.data(),
                    actual.size() * sizeof(__half)) != 0)
      throw std::runtime_error(
          "fused HC down changed rounding, guards or replay");
  }
  if (q::HcDownF16Gemm(w, tiled, output + 8, 95, nullptr))
    throw std::runtime_error("fused HC down accepted a short batch");
  CheckHip(hipGraphExecDestroy(replay), "HC graph free");
  CheckHip(hipGraphDestroy(graph), "HC graph definition free");
  CheckHip(hipStreamDestroy(stream), "HC stream free");
  CheckHip(hipFree(output), "HC output free");
  CheckHip(hipFree(reference), "HC reference free");
  CheckHip(hipFree(activated), "HC activation free");
}

void CheckRoutedQ8Placement() {
  constexpr int rows = 64, cols = 512, tokens = 65, experts = 2;
  const auto weights = MakeWeights(rows * experts, cols, 31415, false);
  std::vector<float> input(tokens * cols), output(tokens * experts * rows);
  std::vector<std::int32_t> ids(tokens * experts);
  std::uint32_t seed = 271828;
  // Isolate the dot product and row layout from quantizer tie semantics.
  // Non-power-of-two scales still exercise FP32 product contraction.
  for (int b = 0; b < cols / 32; ++b) {
    const float scale = 0.0012345F * (1 + b % 7);
    for (int j = 0; j < 32; ++j)
      input[b * 32 + j] =
          (j == 0 ? 127 : int(NextRandom(&seed) % 255) - 127) * scale;
  }
  for (int t = 0; t < tokens; ++t) {
    if (t)
      std::copy_n(input.data(), cols, input.data() + t * cols);
    for (int e = 0; e < experts; ++e)
      ids[t * experts + e] = (t + e) % experts;
  }
  void* dw = nullptr;
  float *dx = nullptr, *dy = nullptr;
  std::int32_t* di = nullptr;
  CheckHip(hipMalloc(&dw, weights.blocks.size()), "routed Q8 weights");
  CheckHip(hipMalloc(&dx, input.size() * sizeof(float)), "routed Q8 input");
  CheckHip(hipMalloc(&dy, output.size() * sizeof(float)), "routed Q8 output");
  CheckHip(hipMalloc(&di, ids.size() * sizeof(std::int32_t)), "routed Q8 IDs");
  CheckHip(hipMemcpy(dw, weights.blocks.data(), weights.blocks.size(),
                     hipMemcpyHostToDevice),
           "routed Q8 upload");
  CheckHip(hipMemcpy(dx, input.data(), input.size() * sizeof(float),
                     hipMemcpyHostToDevice),
           "routed Q8 upload");
  CheckHip(hipMemcpy(di, ids.data(), ids.size() * sizeof(std::int32_t),
                     hipMemcpyHostToDevice),
           "routed Q8 upload");
  qfn_mmq_set_routed_max_expert_rows(tokens);
  qfn_mmq_set_routed_tile_cols(32);
  if (qfn_mmq_q8_0_moe_raw(dw, dx, di, dy, rows, cols, tokens, experts, experts,
                           nullptr) != 0)
    throw std::runtime_error("routed Q8 placement projection failed");
  CheckHip(hipMemcpy(output.data(), dy, output.size() * sizeof(float),
                     hipMemcpyDeviceToHost),
           "routed Q8 download");
  for (int row = 0; row < experts * rows; ++row) {
    double expected = 0;
    for (int b = 0; b < cols / 32; ++b) {
      float maximum = 0;
      for (int j = 0; j < 32; ++j)
        maximum = std::max(maximum, std::abs(input[b * 32 + j]));
      const float inverse = 127.0F / maximum;
      const float scale = 1.0F / inverse;
      const auto* block = weights.blocks.data() + (row * (cols / 32) + b) * 34;
      __half weight_scale;
      std::memcpy(&weight_scale, block, sizeof(weight_scale));
      int dot = 0;
      for (int j = 0; j < 32; ++j)
        dot += static_cast<std::int8_t>(block[2 + j]) *
               static_cast<int>(std::round(input[b * 32 + j] * inverse));
      expected += double(__half2float(weight_scale)) * scale * dot;
    }
    if (!std::isfinite(output[row]) ||
        std::abs(double(output[row]) - expected) > 1e-4)
      throw std::runtime_error(
          "routed Q8 differs from FP64 dot oracle at row " +
          std::to_string(row) + ": actual=" + std::to_string(output[row]) +
          " expected=" + std::to_string(expected));
  }
  // Identical activations must not depend on expert packing, column minitile
  // or the ragged final tile. Fast-math formerly contracted these differently.
  for (std::size_t slot = 0; slot < ids.size(); ++slot) {
    if (std::memcmp(output.data() + slot * rows,
                    output.data() + ids[slot] * rows,
                    rows * sizeof(float)) != 0)
      throw std::runtime_error("routed Q8 output depends on row placement");
  }
  qfn_mmq_set_routed_max_expert_rows(0);
  qfn_mmq_set_routed_tile_cols(0);
  for (void* p : {dw, static_cast<void*>(dx), static_cast<void*>(dy),
                  static_cast<void*>(di)})
    CheckHip(hipFree(p), "routed Q8 free");
}

double Run(std::size_t batch, std::size_t m, std::size_t k, std::uint32_t seed,
           std::size_t reference_tokens = 0) {
  const Q8Weights w = MakeWeights(m, k, seed);
  std::vector<float> x(batch * k);
  std::uint32_t state = seed ^ 0xABCDEF01U;
  for (float& v : x) {
    v = Uniform(&state, 2.0F);
  }
  void* d_w = nullptr;
  float* d_x = nullptr;
  float* d_mmq = nullptr;
  float* d_w8 = nullptr;
  float* d_f16 = nullptr;
  __half* d_x_half = nullptr;
  void* d_tiled = nullptr;
  CheckHip(hipMalloc(&d_w, w.blocks.size() + 4096), "hipMalloc");
  CheckHip(hipMalloc(&d_x, x.size() * 4), "hipMalloc");
  CheckHip(hipMalloc(&d_mmq, batch * m * 4), "hipMalloc");
  CheckHip(hipMalloc(&d_w8, batch * m * 4), "hipMalloc");
  CheckHip(hipMalloc(&d_tiled, q::Q8TiledBytes(batch, k)), "hipMalloc");
  CheckHip(hipMalloc(&d_f16, batch * m * 4), "hipMalloc");
  CheckHip(hipMalloc(&d_x_half, x.size() * 2), "hipMalloc");
  CheckHip(
      hipMemcpy(d_w, w.blocks.data(), w.blocks.size(), hipMemcpyHostToDevice),
      "upload");
  CheckHip(hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice),
           "upload");
  CheckHip(hipMemset(d_w8, 0, batch * m * 4), "memset");
  CheckHip(hipMemset(d_f16, 0, batch * m * 4), "memset");
  if (qfn_mmq_q8_0_dense(d_w, d_x, d_mmq, static_cast<int>(m),
                         static_cast<int>(batch), static_cast<int>(k),
                         nullptr) != 0) {
    throw std::runtime_error("MMQ dense failed");
  }
  q::QuantizeQ8Tiled(d_x, d_tiled, batch, k, nullptr);
  if (!q::W8A8Gemm(d_w, d_tiled, d_w8, batch, m, k, nullptr)) {
    throw std::runtime_error("W8A8 GEMM rejected the shape");
  }
  q::NarrowActivations(d_x, d_x_half, false, x.size(), nullptr);
  if (!q::DenseF16Gemm(d_w, d_x_half, d_f16, batch, m, k, nullptr)) {
    throw std::runtime_error("dense F16 GEMM rejected the shape");
  }
  CheckHip(hipDeviceSynchronize(), "GEMMs");
  std::vector<float> mmq(batch * m);
  std::vector<float> w8(batch * m);
  std::vector<float> f16(batch * m);
  CheckHip(hipMemcpy(f16.data(), d_f16, f16.size() * 4, hipMemcpyDeviceToHost),
           "download");
  CheckHip(hipMemcpy(mmq.data(), d_mmq, mmq.size() * 4, hipMemcpyDeviceToHost),
           "download");
  CheckHip(hipMemcpy(w8.data(), d_w8, w8.size() * 4, hipMemcpyDeviceToHost),
           "download");
  // Both routes quantize the activations per 32-wide block, so they agree
  // to accumulation order; the F64 reference over the dequantized weights
  // bounds the activation quantization itself.
  double worst_vs_mmq = 0.0;
  double worst_f16_vs_mmq = 0.0;
  double worst_vs_ref = 0.0;
  double worst_f16 = 0.0;
  double ref_scale = 0.0;
  for (std::size_t i = 0; i < w8.size(); ++i) {
    if (!std::isfinite(w8[i]) || !std::isfinite(f16[i])) {
      throw std::runtime_error("projection output is not finite");
    }
    worst_vs_mmq =
        std::max(worst_vs_mmq, std::abs(static_cast<double>(mmq[i] - w8[i])));
    worst_f16_vs_mmq = std::max(worst_f16_vs_mmq,
                                std::abs(static_cast<double>(mmq[i] - f16[i])));
  }
  if (!q::W8A8Gemm(d_w, d_tiled, d_w8, batch, m, k, nullptr)) {
    throw std::runtime_error("W8A8 replay rejected the shape");
  }
  std::vector<float> replay(w8.size());
  CheckHip(hipMemcpy(replay.data(), d_w8, replay.size() * sizeof(float),
                     hipMemcpyDeviceToHost),
           "replay download");
  if (replay != w8) {
    throw std::runtime_error("W8A8 replay changed the output");
  }
  if (!q::DenseF16Gemm(d_w, d_x_half, d_f16, batch, m, k, nullptr)) {
    throw std::runtime_error("F16 replay rejected the shape");
  }
  CheckHip(hipMemcpy(replay.data(), d_f16, replay.size() * sizeof(float),
                     hipMemcpyDeviceToHost),
           "F16 replay download");
  if (std::memcmp(replay.data(), f16.data(), replay.size() * sizeof(float)) !=
      0) {
    throw std::runtime_error("F16 replay changed the output");
  }
  if (batch >= 1024 && m == 2560 && k == 6144) {
    const std::size_t stride = k + 32;
    __half* padded = nullptr;
    // The ragged final row ends at the allocation boundary. NaN padding
    // exposes accidental reads without masking them with extra valid rows.
    const auto elements = (batch - 1) * stride + k;
    CheckHip(hipMalloc(&padded, elements * sizeof(__half)), "padded input");
    CheckHip(hipMemset(padded, 0xFF, elements * sizeof(__half)), "padding");
    CheckHip(hipMemcpy2D(padded, stride * sizeof(__half), d_x_half,
                         k * sizeof(__half), k * sizeof(__half), batch,
                         hipMemcpyDeviceToDevice),
             "padded rows");
    if (!q::DenseF16Gemm(d_w, padded, d_f16, batch, m, k, nullptr, stride))
      throw std::runtime_error("strided F16 projection rejected the shape");
    CheckHip(hipMemcpy(replay.data(), d_f16, replay.size() * sizeof(float),
                       hipMemcpyDeviceToHost),
             "strided output");
    CheckHip(hipFree(padded), "free padded input");
    if (std::memcmp(replay.data(), f16.data(), replay.size() * sizeof(float)))
      throw std::runtime_error("strided F16 projection changed the output");
  }

  // Large production shapes still compare every output against MMQ. Sample
  // evenly spaced tokens for the more expensive independent F64 reference.
  const std::size_t samples =
      reference_tokens == 0 ? batch : std::min(batch, reference_tokens);
  for (std::size_t sample = 0; sample < samples; ++sample) {
    const std::size_t t =
        samples > 1 ? sample * (batch - 1) / (samples - 1) : 0;
    for (std::size_t r = 0; r < m; ++r) {
      double ref = 0.0;
      for (std::size_t i = 0; i < k; ++i) {
        ref += static_cast<double>(w.values[r * k + i]) * x[t * k + i];
      }
      const std::size_t idx = t * m + r;
      worst_vs_ref = std::max(worst_vs_ref, std::abs(ref - w8[idx]));
      worst_f16 = std::max(worst_f16, std::abs(ref - f16[idx]));
      ref_scale = std::max(ref_scale, std::abs(ref));
    }
  }
  std::cout << "W8A8 batch=" << batch << " m=" << m << " k=" << k
            << ": worst |W8A8 - MMQ| " << worst_vs_mmq << ", worst |F16 - MMQ| "
            << worst_f16_vs_mmq << ", worst |W8A8 - F64| " << worst_vs_ref
            << ", worst |F16 - F64| " << worst_f16 << " (reference scale "
            << ref_scale << ")\n";
  if (m == 16384 && k == 2560 && batch >= 1024) {
    CheckSsmProjection(d_w, d_x_half, d_f16, f16,
                       static_cast<std::uint32_t>(batch));
  }
  if (m == 320 && k == 10240 && batch >= 96) {
    CheckHcDownProjection(d_w, d_tiled, d_w8,
                          static_cast<std::uint32_t>(batch));
  }
  if (m == 13312 && k == 2560 && batch >= 1024) {
    CheckAttentionProjection(d_w, d_x_half, d_f16,
                             static_cast<std::uint32_t>(batch));
  }
  (void)hipFree(d_w);
  (void)hipFree(d_x);
  (void)hipFree(d_mmq);
  (void)hipFree(d_w8);
  (void)hipFree(d_tiled);
  (void)hipFree(d_f16);
  (void)hipFree(d_x_half);
  // The MMQ agreement is accumulation order; the F64 gap is the shared
  // 8-bit activation quantization, well under 1% of the output scale. The
  // F16 route's gap is its F16 activation rounding, a few ulps smaller.
  return worst_vs_mmq < 1e-3
             ? std::max({worst_vs_ref, worst_f16, worst_f16_vs_mmq}) / ref_scale
             : 1.0;
}

// Unquantized router, alpha/beta and indexer projections must keep the same
// result when a token moves between decode and any verification batch width.
void CheckSmallProjection(q::WeightType type, unsigned rows, unsigned cols) {
  constexpr unsigned tokens = 64;
  const unsigned element_bytes = type == q::WeightType::kF32 ? 4 : 2;
  std::vector<std::uint8_t> weights(std::size_t(rows) * cols * element_bytes);
  std::vector<float> reference_weights(std::size_t(rows) * cols);
  std::vector<float> input(std::size_t(tokens) * cols);
  std::uint32_t seed = 0x319F42U;
  for (std::size_t i = 0; i < reference_weights.size(); ++i) {
    float value = Uniform(&seed, 0.125F);
    if (type == q::WeightType::kF32) {
      std::memcpy(weights.data() + i * 4, &value, 4);
    } else if (type == q::WeightType::kBF16) {
      const auto packed =
          static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16);
      std::memcpy(weights.data() + i * 2, &packed, 2);
      value = std::bit_cast<float>(std::uint32_t(packed) << 16);
    } else {
      const __half packed = __float2half(value);
      std::memcpy(weights.data() + i * 2, &packed, 2);
      value = __half2float(packed);
    }
    reference_weights[i] = value;
  }
  for (float& value : input)
    value = Uniform(&seed, 1.7F);
  void* dw = nullptr;
  float* dx = nullptr;
  float* dy = nullptr;
  const std::size_t count = std::size_t(tokens) * rows;
  CheckHip(hipMalloc(&dw, weights.size()), "small weights allocation");
  CheckHip(hipMalloc(&dx, input.size() * sizeof(float)),
           "small input allocation");
  CheckHip(hipMalloc(&dy, (count + 8) * sizeof(float)),
           "small output allocation");
  CheckHip(hipMemcpy(dw, weights.data(), weights.size(), hipMemcpyHostToDevice),
           "small weights upload");
  CheckHip(hipMemcpy(dx, input.data(), input.size() * sizeof(float),
                     hipMemcpyHostToDevice),
           "small input upload");
  for (unsigned token = 0; token < tokens; ++token)
    q::SmallGemm(dw, type, dx + token * cols, dy + token * rows, 1, rows, cols,
                 nullptr);
  std::vector<float> scalar(count), batch(count + 8), replay(count);
  CheckHip(hipMemcpy(scalar.data(), dy, count * sizeof(float),
                     hipMemcpyDeviceToHost),
           "small scalar output");
  for (unsigned token = 0; token < tokens; ++token) {
    for (unsigned row = 0; row < rows; ++row) {
      double expected = 0.0;
      for (unsigned col = 0; col < cols; ++col)
        expected += double(reference_weights[std::size_t(row) * cols + col]) *
                    input[std::size_t(token) * cols + col];
      const float actual = scalar[std::size_t(token) * rows + row];
      if (!std::isfinite(actual) || std::abs(double(actual) - expected) >
                                        2e-5 * (1.0 + std::abs(expected)))
        throw std::runtime_error(
            "small projection differs from FP64 reference");
    }
  }
  for (const unsigned n :
       {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 16U, 31U, 32U, 33U, tokens}) {
    CheckHip(hipMemset(dy, 0xA5, batch.size() * sizeof(float)),
             "small output poison");
    q::SmallGemm(dw, type, dx, dy, n, rows, cols, nullptr);
    CheckHip(hipMemcpy(batch.data(), dy, batch.size() * sizeof(float),
                       hipMemcpyDeviceToHost),
             "small batch output");
    if (std::memcmp(scalar.data(), batch.data(),
                    std::size_t(n) * rows * sizeof(float)))
      throw std::runtime_error("small projection changes with batch width");
    for (std::size_t i = std::size_t(n) * rows; i < batch.size(); ++i)
      if (std::bit_cast<std::uint32_t>(batch[i]) != 0xA5A5A5A5U)
        throw std::runtime_error("small projection overwrote its output guard");
  }
  q::SmallGemm(dw, type, dx, dy, tokens, rows, cols, nullptr);
  CheckHip(hipMemcpy(replay.data(), dy, count * sizeof(float),
                     hipMemcpyDeviceToHost),
           "small replay output");
  if (std::memcmp(scalar.data(), replay.data(), count * sizeof(float)))
    throw std::runtime_error("small projection replay differs");
  CheckHip(hipFree(dy), "small output free");
  CheckHip(hipFree(dx), "small input free");
  CheckHip(hipFree(dw), "small weights free");
}

void CheckDecodeGrouping(int rows, int cols, int grouped_tokens = 0) {
  const int tokens = grouped_tokens != 0 ? grouped_tokens
                     : rows == 320 && cols == 10240
                         ? 64
                         : (rows >= 8192 && cols == 2560 ? 48 : 32);
  const auto w = MakeWeights(rows, cols, 11, false);
  // The large fixtures exercise matrix dispatch. Gated vectors already have
  // independent-weight coverage in the small fixtures.
  const auto gate =
      rows <= 320 ? MakeWeights(rows, cols, 17, false) : Q8Weights{};
  std::vector<float> x(tokens * cols);
  std::uint32_t seed = 37;
  for (auto& v : x)
    v = Uniform(&seed, 2.0F);
  void* dw = nullptr;
  void* dg = nullptr;
  float* dx = nullptr;
  void* dq = nullptr;
  float* out = nullptr;
  CheckHip(hipMalloc(&dw, w.blocks.size() + 4096), "decode weights");
  if (!gate.blocks.empty())
    CheckHip(hipMalloc(&dg, gate.blocks.size() + 4096), "decode gate");
  CheckHip(hipMalloc(&dx, x.size() * sizeof(float)), "decode inputs");
  CheckHip(hipMalloc(&dq, qfn_mmq_q8_1_bytes(tokens, cols)),
           "decode quantized inputs");
  CheckHip(hipMalloc(&out, (rows * tokens + 4) * sizeof(float)),
           "decode output");
  CheckHip(
      hipMemcpy(dw, w.blocks.data(), w.blocks.size(), hipMemcpyHostToDevice),
      "weights upload");
  if (dg != nullptr)
    CheckHip(hipMemcpy(dg, gate.blocks.data(), gate.blocks.size(),
                       hipMemcpyHostToDevice),
             "gate upload");
  CheckHip(
      hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice),
      "input upload");
  if (qfn_mmq_quantize_q8_1(dx, dq, tokens, cols, nullptr))
    throw std::runtime_error("decode input quantization failed");
  for (const bool gated : {false, true}) {
    if (gated && dg == nullptr)
      continue;
    for (int t = 0; t < tokens; ++t) {
      const auto* qrow = static_cast<const std::uint8_t*>(dq) +
                         t * qfn_mmq_q8_1_bytes(1, cols);
      if (qfn_mmq_q8_0_dense_vec_preq(dw, gated ? dg : nullptr, qrow,
                                      out + t * rows, rows, 1, cols, nullptr))
        throw std::runtime_error("scalar dense projection failed");
    }
    std::vector<float> scalar(tokens * rows), batch(tokens * rows + 4);
    CheckHip(hipMemcpy(scalar.data(), out, scalar.size() * sizeof(float),
                       hipMemcpyDeviceToHost),
             "scalar output");
    for (int n = 2; n <= (gated ? 8 : tokens); ++n) {
      if (grouped_tokens != 0 && n > 32 && n % 32 != 0)
        continue;
      if (rows == 320 && cols == 10240 && n > 32 && n % 8 != 0)
        continue;
      CheckHip(hipMemset(out, 0xA5, batch.size() * sizeof(float)),
               "decode output guard");
      // End every shape at the allocation boundary. Using a full shared
      // input buffer can hide overreads by a partially populated final wave.
      const auto* input = static_cast<const std::uint8_t*>(dq) +
                          qfn_mmq_q8_1_bytes(tokens - n, cols);
      if (qfn_mmq_q8_0_dense_vec_preq(dw, gated ? dg : nullptr, input, out,
                                      rows, n, cols, nullptr))
        throw std::runtime_error("batched dense projection failed");
      CheckHip(hipMemcpy(batch.data(), out, batch.size() * sizeof(float),
                         hipMemcpyDeviceToHost),
               "batch output");
      if (std::memcmp(scalar.data() + (tokens - n) * rows, batch.data(),
                      n * rows * sizeof(float)) != 0)
        throw std::runtime_error(
            "Q8 dense grouping differs: M=" + std::to_string(rows) +
            " K=" + std::to_string(cols) + " N=" + std::to_string(n) +
            " gated=" + std::to_string(gated));
      for (std::size_t i = std::size_t(n) * rows; i < batch.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(batch[i]) != 0xA5A5A5A5U)
          throw std::runtime_error("Q8 batch projection overwrote its guard");
      }
    }
  }
  for (void* ptr :
       {dw, dg, static_cast<void*>(dx), dq, static_cast<void*>(out)})
    CheckHip(hipFree(ptr), "decode test free");
}

void CheckMtpOutputHead(float input_scale) {
  constexpr int rows = 65, cols = 2560, blocks = cols / 32;
  auto weights = MakeWeights(rows, cols, 73);
  std::vector<float> input(cols), output(rows + 8, -1234567.0F);
  std::uint32_t seed = 31;
  for (auto& value : input)
    value = Uniform(&seed, 2.0F * input_scale);
  std::vector<std::uint8_t> quantized(qfn_mmq_q8_1_bytes(1, cols));
  void *dw = nullptr, *dq = nullptr;
  float *dx = nullptr, *dy = nullptr;
  CheckHip(hipMalloc(&dw, weights.blocks.size()), "MTP Q8 weights");
  CheckHip(hipMalloc(&dq, quantized.size()), "MTP Q8 input");
  CheckHip(hipMalloc(&dx, input.size() * 4), "MTP input");
  CheckHip(hipMalloc(&dy, output.size() * 4), "MTP output");
  CheckHip(hipMemcpy(dw, weights.blocks.data(), weights.blocks.size(),
                     hipMemcpyHostToDevice),
           "weights");
  CheckHip(hipMemcpy(dx, input.data(), input.size() * 4, hipMemcpyHostToDevice),
           "input");
  CheckHip(
      hipMemcpy(dy, output.data(), output.size() * 4, hipMemcpyHostToDevice),
      "guards");
  if (qfn_mmq_quantize_q8_1(dx, dq, 1, cols, nullptr))
    throw std::runtime_error("MTP input quantization failed");
  hipStream_t stream;
  hipGraph_t graph;
  hipGraphExec_t replay;
  CheckHip(hipStreamCreate(&stream), "MTP stream");
  CheckHip(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
           "MTP capture");
  if (qfn_mmq_q8_0_dense_vec_preq(dw, nullptr, dq, dy + 4, rows, 1, cols,
                                  stream))
    throw std::runtime_error("MTP Q8 head launch failed");
  CheckHip(hipStreamEndCapture(stream, &graph), "MTP capture end");
  CheckHip(hipGraphInstantiate(&replay, graph, nullptr, nullptr, 0),
           "MTP graph");
  std::vector<float> first;
  for (unsigned repetition = 0; repetition < 2; ++repetition) {
    CheckHip(hipGraphLaunch(replay, stream), "MTP replay");
    CheckHip(hipStreamSynchronize(stream), "MTP wait");
    CheckHip(
        hipMemcpy(output.data(), dy, output.size() * 4, hipMemcpyDeviceToHost),
        "MTP output");
    if (repetition == 0)
      first = output;
    else if (output != first)
      throw std::runtime_error("MTP Q8 replay changed");
  }
  CheckHip(
      hipMemcpy(quantized.data(), dq, quantized.size(), hipMemcpyDeviceToHost),
      "MTP quantized input");
  for (int i = 0; i < rows + 8; ++i)
    if ((i < 4 || i >= rows + 4) && output[i] != -1234567.0F)
      throw std::runtime_error("MTP Q8 output guard changed");
  for (int row = 0; row < rows; ++row) {
    double expected = 0;
    for (int b = 0; b < blocks; ++b) {
      const auto* w = weights.blocks.data() + (row * blocks + b) * 34;
      const auto* x = quantized.data() + b * 36;
      __half wd, xd;
      std::memcpy(&wd, w, 2);
      std::memcpy(&xd, x, 2);
      int dot = 0;
      for (int i = 0; i < 32; ++i)
        dot += static_cast<std::int8_t>(w[2 + i]) *
               static_cast<std::int8_t>(x[4 + i]);
      expected += double(__half2float(wd)) * __half2float(xd) * dot;
    }
    if (!std::isfinite(output[row + 4]) ||
        std::abs(output[row + 4] - expected) > 0.001 * input_scale)
      throw std::runtime_error("MTP Q8 projection differs from F64 dot oracle");
  }
  CheckHip(hipGraphExecDestroy(replay), "MTP graph free");
  CheckHip(hipGraphDestroy(graph), "MTP source graph free");
  CheckHip(hipStreamDestroy(stream), "MTP stream free");
  for (void* p : {dw, dq, static_cast<void*>(dx), static_cast<void*>(dy)})
    CheckHip(hipFree(p), "MTP free");
}

}  // namespace

int main() {
  try {
    CheckRoutedQ8Placement();
    CheckSmallProjection(q::WeightType::kF32, 513, 2560);
    CheckSmallProjection(q::WeightType::kF32, 96, 2560);
    CheckSmallProjection(q::WeightType::kBF16, 129, 2560);
    CheckSmallProjection(q::WeightType::kF16, 7, 131);
    CheckDecodeGrouping(64, 2560);
    CheckDecodeGrouping(320, 10240);
    CheckDecodeGrouping(1024, 2560, 256);
    CheckDecodeGrouping(10240, 320, 256);
    CheckDecodeGrouping(2561, 2560, 64);
    CheckDecodeGrouping(12289, 2560);
    CheckDecodeGrouping(65537, 2560);
    CheckMtpOutputHead(1.0F);
    // Small activations must retain products below F16's normal range.
    CheckMtpOutputHead(0.0001F);
    bool ok = true;
    // Ragged batch and rows against the 128-wide macro tiles, the 64-token
    // tile below 96, and the model's ssm_out / shexp_down widths.
    ok = Run(100, 320, 2560, 0x1234ABCDU) < 1e-2 && ok;
    ok = Run(37, 640, 2560, 0x0BADF00DU) < 1e-2 && ok;
    ok = Run(200, 200, 6144, 0xDEADBEEFU) < 1e-2 && ok;
    ok = Run(1025, 2560, 6144, 0x51A17U, 2) < 1e-2 && ok;
    ok = Run(2049, 2560, 6144, 0x25606144U, 2) < 1e-2 && ok;
    // Wide SSM projection: two K blocks per stage, with a partial token tile.
    ok = Run(2049, 16384, 2560, 0x16384256U, 2) < 1e-2 && ok;
    ok = Run(2049, 13312, 2560, 0x13312256U, 2) < 1e-2 && ok;
    // HC up: the grouped grid, including the last partial token tile.
    ok = Run(2049, 10240, 320, 0x8A8A320U, 2) < 1e-2 && ok;
    // HC down fusion: its first supported batch and a partial final tile.
    ok = Run(96, 320, 10240, 0x320A96U, 2) < 1e-2 && ok;
    ok = Run(2049, 320, 10240, 0x320A2049U, 2) < 1e-2 && ok;
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
