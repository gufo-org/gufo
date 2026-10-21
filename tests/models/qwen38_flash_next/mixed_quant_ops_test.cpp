// Independent host dequantization checks the packed mixed Q4_K_M paths.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/cpu_ops.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace model = gufo::models::qwen38_flash_next;
namespace gpu = model::rocm;
using gufo::core::GgmlType;

namespace {
void CheckHip(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}
struct Buffer {
  void* data{};
  explicit Buffer(std::size_t bytes) { CheckHip(hipMalloc(&data, bytes)); }
  ~Buffer() { (void)hipFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void Upload(const void* src, std::size_t bytes) {
    CheckHip(hipMemcpy(data, src, bytes, hipMemcpyHostToDevice));
  }
};
void Close(float got, double expected) {
  if (!std::isfinite(got) ||
      std::abs(got - expected) > 2e-4 + 2e-5 * std::abs(expected))
    throw std::runtime_error("mixed-quant mismatch: " + std::to_string(got) +
                             " != " + std::to_string(expected));
}

struct Weights {
  model::TensorRef ref;
  std::vector<std::uint8_t> bytes;
  std::vector<float> values;
  Weights(GgmlType type, unsigned rows, unsigned cols, unsigned experts = 1) {
    ref.type = type;
    ref.rows = rows;
    ref.cols = cols;
    ref.experts = experts;
    bytes.resize(ref.SizeBytes());
    values.resize(std::size_t(rows) * cols * experts);
    const unsigned block_elems = type == GgmlType::kQ5_0 ? 32 : 256;
    const unsigned block_bytes = type == GgmlType::kQ5_0   ? 22
                                 : type == GgmlType::kQ4_K ? 144
                                                           : 210;
    for (std::size_t b = 0; b < bytes.size() / block_bytes; ++b) {
      auto* p = bytes.data() + b * block_bytes;
      for (unsigned j = 0; j < block_bytes; ++j)
        p[j] = static_cast<std::uint8_t>(j * 73 + b * 37 + 19);
      const __half d = __float2half(b % 2 ? -0.03137F : 0.01517F);
      const unsigned scale_offset = type == GgmlType::kQ6_K ? 208 : 0;
      std::memcpy(p + scale_offset, &d, 2);
      if (type == GgmlType::kQ4_K) {
        const __half dm = __float2half(0.00791F);
        std::memcpy(p + 2, &dm, 2);
      }
    }
    ref.data = bytes.data();
    for (unsigned e = 0; e < experts; ++e)
      for (unsigned r = 0; r < rows; ++r)
        model::cpu::DequantizeRow(
            ref, e, r, values.data() + (std::size_t(e) * rows + r) * cols);
    if (bytes.size() !=
        std::size_t(rows) * experts * (cols / block_elems) * block_bytes)
      throw std::runtime_error("bad storage geometry");
  }
};

void CheckDense(GgmlType type, gpu::WeightType device_type, unsigned k) {
  constexpr unsigned m = 13;
  Weights w(type, m, k);
  Buffer weights(w.bytes.size());
  weights.Upload(w.bytes.data(), w.bytes.size());
  for (unsigned n : {1U, 2U, 7U, 8U, 9U, 32U, 33U}) {
    std::vector<float> x(std::size_t(n) * k);
    for (std::size_t i = 0; i < x.size(); ++i)
      x[i] = std::sin(float(i + 1) * 0.19F) * 0.07F;
    Buffer input(x.size() * 4), output((std::size_t(n) * m + 1) * 4);
    input.Upload(x.data(), x.size() * 4);
    std::vector<float> actual(std::size_t(n) * m + 1, -123.0F);
    output.Upload(actual.data(), actual.size() * 4);
    gpu::SmallGemm(weights.data, device_type, static_cast<float*>(input.data),
                   static_cast<float*>(output.data), n, m, k, nullptr);
    CheckHip(hipGetLastError());
    CheckHip(hipMemcpy(actual.data(), output.data, actual.size() * 4,
                       hipMemcpyDeviceToHost));
    for (unsigned t = 0; t < n; ++t)
      for (unsigned r = 0; r < m; ++r) {
        double sum = 0;
        for (unsigned j = 0; j < k; ++j)
          sum += double(w.values[r * k + j]) * x[t * k + j];
        Close(actual[t * m + r], sum);
      }
    if (actual.back() != -123.0F)
      throw std::runtime_error("dense output guard overwritten");
    Buffer scalar(std::size_t(n) * m * sizeof(float));
    for (unsigned t = 0; t < n; ++t)
      gpu::SmallGemm(
          weights.data, device_type, static_cast<float*>(input.data) + t * k,
          static_cast<float*>(scalar.data) + t * m, 1, m, k, nullptr);
    std::vector<float> isolated(std::size_t(n) * m);
    CheckHip(hipMemcpy(isolated.data(), scalar.data,
                       isolated.size() * sizeof(float), hipMemcpyDeviceToHost));
    if (std::memcmp(actual.data(), isolated.data(),
                    isolated.size() * sizeof(float)) != 0)
      throw std::runtime_error("packed projection depends on batch width");
  }
  const std::int32_t ids[] = {12, 0, 7};
  Buffer tokens(sizeof(ids)), output(3 * 4 * k * sizeof(float));
  tokens.Upload(ids, sizeof(ids));
  gpu::EmbedTokens(weights.data, device_type,
                   static_cast<std::int32_t*>(tokens.data),
                   static_cast<float*>(output.data), 3, k, 4, nullptr);
  std::vector<float> actual(3 * 4 * k);
  CheckHip(hipMemcpy(actual.data(), output.data, actual.size() * 4,
                     hipMemcpyDeviceToHost));
  for (unsigned t = 0; t < 3; ++t)
    for (unsigned s = 0; s < 4; ++s)
      for (unsigned j = 0; j < k; ++j)
        Close(actual[(t * 4 + s) * k + j], w.values[ids[t] * k + j]);
}

void CheckExperts(unsigned k, unsigned used) {
  constexpr unsigned m = 13, experts = 3;
  Weights w(GgmlType::kQ5_0, m, k, experts);
  // MMQ may read a full final K tile. Match production's zeroed weight tail.
  Buffer weights(w.bytes.size() + 4096);
  CheckHip(hipMemset(weights.data, 0, w.bytes.size() + 4096));
  weights.Upload(w.bytes.data(), w.bytes.size());
  for (unsigned n : {1U, 8U, 9U, 33U}) {
    std::vector<float> x(std::size_t(n) * k);
    std::vector<std::int32_t> ids(n * used);
    for (unsigned t = 0; t < n; ++t) {
      for (unsigned j = 0; j < k; ++j) {
        // Exactly representable Q8_1 input: no reference quantizer needed.
        const int code = j % 32 == 0   ? 127
                         : j % 32 == 1 ? -127
                                       : int((j + t) % 31) - 15;
        x[t * k + j] = code / 128.0F;
      }
      for (unsigned u = 0; u < used; ++u)
        ids[t * used + u] = (t + u) % experts;
    }
    Buffer input(x.size() * 4), routing(ids.size() * 4),
        output((n * used * m + 1) * 4);
    input.Upload(x.data(), x.size() * 4);
    routing.Upload(ids.data(), ids.size() * 4);
    for (bool tiled : {false, true}) {
      std::vector<float> actual(n * used * m + 1, -123.0F);
      output.Upload(actual.data(), actual.size() * 4);
      qfn_mmq_set_routed_max_expert_rows(n * used);
      const int rc =
          tiled ? qfn_mmq_q5_0_moe_raw(weights.data,
                                       static_cast<float*>(input.data),
                                       static_cast<std::int32_t*>(routing.data),
                                       static_cast<float*>(output.data), m, k,
                                       n, experts, used, nullptr)
                : qfn_mmq_moe_vec(6, weights.data,
                                  static_cast<float*>(input.data),
                                  static_cast<std::int32_t*>(routing.data),
                                  static_cast<float*>(output.data), m, k, n,
                                  experts, used, nullptr);
      if (rc != 0)
        throw std::runtime_error("Q5_0 expert dispatch failed");
      CheckHip(hipMemcpy(actual.data(), output.data, actual.size() * 4,
                         hipMemcpyDeviceToHost));
      for (unsigned t = 0; t < n; ++t)
        for (unsigned u = 0; u < used; ++u)
          for (unsigned r = 0; r < m; ++r) {
            double sum = 0;
            for (unsigned j = 0; j < k; ++j)
              sum += double(w.values[(ids[t * used + u] * m + r) * k + j]) *
                     x[t * k + j];
            Close(actual[(t * used + u) * m + r], sum);
          }
      if (actual.back() != -123.0F)
        throw std::runtime_error("expert output guard overwritten");
    }
  }
}
// Independent F64 dots bound both F16 operand rounding and accumulation.
// The original scalar checks above retain their tighter tolerances unchanged.
void CheckPrefill(GgmlType type, gpu::WeightType device_type, unsigned m,
                  unsigned k, float input_scale = 1.0F) {
  Weights w(type, m, k);
  Buffer weights(w.bytes.size());
  weights.Upload(w.bytes.data(), w.bytes.size());
  std::vector<float> first;
  for (unsigned n :
       {1U, 8U, 9U, 32U, 33U, 127U, 128U, 129U, 2047U, 2048U, 2049U}) {
    std::vector<float> x(std::size_t(n) * k);
    for (std::size_t i = 0; i < x.size(); ++i)
      x[i] = std::sin(float(i + 1) * 0.19F) * 0.07F * input_scale;
    Buffer input(x.size() * sizeof(float)),
        output((std::size_t(n) * m + 1) * 4);
    input.Upload(x.data(), x.size() * sizeof(float));
    std::vector<float> actual(std::size_t(n) * m + 1, -123.0F);
    output.Upload(actual.data(), actual.size() * 4);
    if (!gpu::DensePackedF16Gemm(
            weights.data, device_type, static_cast<const float*>(input.data),
            static_cast<float*>(output.data), n, m, k, nullptr))
      throw std::runtime_error("packed prefill rejected valid shape");
    CheckHip(hipMemcpy(actual.data(), output.data, actual.size() * 4,
                       hipMemcpyDeviceToHost));
    if (actual.back() != -123.0F ||
        !std::all_of(actual.begin(), actual.end(),
                     [](float v) { return std::isfinite(v); }))
      throw std::runtime_error("prefill nonfinite result or overwritten guard");
    if (first.empty())
      first.assign(actual.begin(), actual.begin() + m);
    else if (std::memcmp(first.data(), actual.data(), m * sizeof(float)) != 0)
      throw std::runtime_error("prefill token depends on batch width");
    double error = 0, norm = 0, weight_error = 0, input_error = 0;
    const unsigned stride = std::max(1U, m / 129);
    for (unsigned t : {0U, n / 2, n - 1})
      for (unsigned r = 0; r < m; r += stride) {
        double expected = 0, rounded_w = 0, rounded_x = 0;
        for (unsigned j = 0; j < k; ++j) {
          const float wv = w.values[r * k + j], xv = x[t * k + j];
          expected += double(wv) * xv;
          rounded_w += double(__half2float(__float2half_rn(wv))) * xv;
          rounded_x += double(wv) * __half2float(__float2half_rn(xv));
        }
        const double delta = actual[t * m + r] - expected;
        error += delta * delta;
        weight_error += (rounded_w - expected) * (rounded_w - expected);
        input_error += (rounded_x - expected) * (rounded_x - expected);
        norm += expected * expected;
      }
    const double rms = std::sqrt(error / std::max(norm, 1e-30));
    std::cout << "packed_prefill type=" << int(type) << " n=" << n << " m=" << m
              << " k=" << k << " relative_rms=" << rms
              << " weight_rounding=" << std::sqrt(weight_error / norm)
              << " input_rounding=" << std::sqrt(input_error / norm) << '\n';
    if (rms > 0.002)
      throw std::runtime_error("prefill exceeds independent F64 error bound");
    if (n == 129) {
      for (unsigned offset : {1U, 7U, 15U, 31U, 63U, 127U}) {
        if (!gpu::DensePackedF16Gemm(
                weights.data, device_type,
                static_cast<const float*>(input.data) + std::size_t(offset) * k,
                static_cast<float*>(output.data), n - offset, m, k, nullptr))
          throw std::runtime_error("packed prefill slice rejected");
        std::vector<float> sliced(std::size_t(n - offset) * m);
        CheckHip(hipMemcpy(sliced.data(), output.data,
                           sliced.size() * sizeof(float),
                           hipMemcpyDeviceToHost));
        if (std::memcmp(sliced.data(), actual.data() + std::size_t(offset) * m,
                        sliced.size() * sizeof(float)) != 0)
          throw std::runtime_error(
              "prefill slice changes bits: type=" + std::to_string(int(type)) +
              " m=" + std::to_string(m) + " offset=" + std::to_string(offset));
      }
    }
  }
  if (gpu::DensePackedF16Gemm(nullptr, device_type, nullptr, nullptr, 1, m, k,
                              nullptr) ||
      gpu::DensePackedF16Gemm(weights.data, gpu::WeightType::kF32, nullptr,
                              nullptr, 1, m, k, nullptr))
    throw std::runtime_error("invalid prefill arguments accepted");
}

// One-hot activation rows expose every aligned K16 code group separately.
// Use the existing independent host-decode tolerance. No dot-product
// cancellation can hide a bad code or a misplaced scale/min group.
void CheckPrefillBasis(GgmlType type, gpu::WeightType device_type, unsigned k) {
  constexpr unsigned m = 5, n = 2048;
  Weights w(type, m, k);
  std::vector<float> x(std::size_t(n) * k, 0.0F);
  for (unsigned i = 0; i < n; ++i)
    x[std::size_t(i) * k + i % k] = 1.0F;
  Buffer weights(w.bytes.size()), input(x.size() * sizeof(float)),
      output((std::size_t(n) * m + 1) * sizeof(float));
  weights.Upload(w.bytes.data(), w.bytes.size());
  input.Upload(x.data(), x.size() * sizeof(float));
  std::vector<float> actual(std::size_t(n) * m + 1, -123.0F);
  output.Upload(actual.data(), actual.size() * sizeof(float));
  if (!gpu::DensePackedF16Gemm(
          weights.data, device_type, static_cast<const float*>(input.data),
          static_cast<float*>(output.data), n, m, k, nullptr))
    throw std::runtime_error("basis projection rejected");
  CheckHip(hipMemcpy(actual.data(), output.data, actual.size() * sizeof(float),
                     hipMemcpyDeviceToHost));
  for (unsigned token = 0; token < n; ++token) {
    for (unsigned row = 0; row < m; ++row) {
      const float value = w.values[std::size_t(row) * k + token % k];
      Close(actual[std::size_t(token) * m + row], value);
    }
  }
  if (actual.back() != -123.0F)
    throw std::runtime_error("basis output guard overwritten");
}

void CheckRoutedPrefill() {
  constexpr unsigned experts = 3, m = 129, k = 640, used = 2;
  Weights w(GgmlType::kQ5_0, m, k, experts);
  Buffer weights(w.bytes.size());
  weights.Upload(w.bytes.data(), w.bytes.size());
  for (unsigned n : {1U, 9U, 33U, 129U}) {
    std::vector<__half> x(n * k);
    for (unsigned i = 0; i < x.size(); ++i)
      x[i] = __float2half_rn(std::sin(float(i) * 0.13F) * 0.03F);
    std::vector<std::vector<int>> tokens(experts), slots(experts);
    for (unsigned t = 0; t < n; ++t)
      for (unsigned u = 0; u < used; ++u) {
        const unsigned e = (t + u) % experts;
        tokens[e].push_back(t);
        slots[e].push_back(t * used + u);
      }
    std::vector<int> bounds(1, 0), rows_in, rows_out;
    for (unsigned e = 0; e < experts; ++e) {
      const auto count = (tokens[e].size() + 15) / 16 * 16;
      rows_in.insert(rows_in.end(), tokens[e].begin(), tokens[e].end());
      rows_out.insert(rows_out.end(), slots[e].begin(), slots[e].end());
      rows_in.resize(bounds.back() + count, -1);
      rows_out.resize(bounds.back() + count, -1);
      bounds.push_back(rows_in.size());
    }
    Buffer input(x.size() * sizeof(__half)), db(bounds.size() * 4),
        di(rows_in.size() * 4), dout(rows_out.size() * 4),
        output((n * used * m + 1) * 4),
        output_half((n * used * m + 1) * sizeof(__half));
    input.Upload(x.data(), x.size() * sizeof(__half));
    db.Upload(bounds.data(), bounds.size() * 4);
    di.Upload(rows_in.data(), rows_in.size() * 4);
    dout.Upload(rows_out.data(), rows_out.size() * 4);
    for (unsigned width : {16U, 48U, 64U}) {
      std::vector<int> tiles;
      for (unsigned e = 0; e < experts; ++e)
        for (unsigned t = 0; t < unsigned(bounds[e + 1] - bounds[e]);
             t += width)
          tiles.push_back(e | ((t / width) << 16));
      Buffer dt(tiles.size() * 4);
      dt.Upload(tiles.data(), tiles.size() * 4);
      std::vector<float> actual(n * used * m + 1, -123.0F);
      output.Upload(actual.data(), actual.size() * 4);
      const auto launch = [&](float* f32, __half* f16) {
        return gpu::RoutedF16Gemm(
            weights.data, gpu::WeightType::kQ5_0,
            static_cast<const __half*>(input.data),
            static_cast<const int*>(dt.data), tiles.size(), width,
            static_cast<const int*>(db.data), static_cast<const int*>(di.data),
            static_cast<const int*>(dout.data), nullptr, f32, f16, m, k,
            nullptr);
      };
      if (!launch(static_cast<float*>(output.data), nullptr))
        throw std::runtime_error("Q5_0 routed prefill dispatch rejected");
      CheckHip(hipMemcpy(actual.data(), output.data, actual.size() * 4,
                         hipMemcpyDeviceToHost));
      double error = 0, norm = 0;
      for (unsigned t = 0; t < n; ++t)
        for (unsigned u = 0; u < used; ++u)
          for (unsigned r = 0; r < m; ++r) {
            double expected = 0;
            for (unsigned j = 0; j < k; ++j)
              expected +=
                  double(w.values[(((t + u) % experts) * m + r) * k + j]) *
                  __half2float(x[t * k + j]);
            const float v = actual[(t * used + u) * m + r];
            if (!std::isfinite(v))
              throw std::runtime_error("nonfinite routed prefill");
            error += (v - expected) * (v - expected);
            norm += expected * expected;
          }
      if (std::sqrt(error / norm) > 0.002 || actual.back() != -123.0F)
        throw std::runtime_error(
            "routed prefill numerical error or guard failure");
      std::vector<__half> narrowed(actual.size(), __float2half_rn(-123.0F));
      output_half.Upload(narrowed.data(), narrowed.size() * sizeof(__half));
      if (!launch(nullptr, static_cast<__half*>(output_half.data)))
        throw std::runtime_error("Q5_0 routed half-output dispatch rejected");
      CheckHip(hipMemcpy(narrowed.data(), output_half.data,
                         narrowed.size() * sizeof(__half),
                         hipMemcpyDeviceToHost));
      for (unsigned i = 0; i < narrowed.size(); ++i)
        if (__half2float(narrowed[i]) !=
            __half2float(__float2half_rn(actual[i])))
          throw std::runtime_error(
              "routed half epilogue changed rounding or guard");
    }
  }
}
}  // namespace

int main() {
  try {
    CheckDense(GgmlType::kQ4_K, gpu::WeightType::kQ4_K, 512);
    CheckDense(GgmlType::kQ4_K, gpu::WeightType::kQ4_K, 2560);
    CheckDense(GgmlType::kQ4_K, gpu::WeightType::kQ4_K, 10240);
    CheckDense(GgmlType::kQ5_0, gpu::WeightType::kQ5_0, 320);
    CheckDense(GgmlType::kQ5_0, gpu::WeightType::kQ5_0, 640);
    CheckDense(GgmlType::kQ6_K, gpu::WeightType::kQ6_K, 512);
    CheckDense(GgmlType::kQ6_K, gpu::WeightType::kQ6_K, 2560);
    CheckDense(GgmlType::kQ6_K, gpu::WeightType::kQ6_K, 10240);
    CheckExperts(640, 1);
    CheckExperts(640, 2);
    CheckExperts(2560, 2);
    CheckPrefill(GgmlType::kQ4_K, gpu::WeightType::kQ4_K, 13, 512);
    CheckPrefill(GgmlType::kQ4_K, gpu::WeightType::kQ4_K, 13, 512, 0.0001F);
    CheckPrefill(GgmlType::kQ5_0, gpu::WeightType::kQ5_0, 17, 96);
    CheckPrefill(GgmlType::kQ4_K, gpu::WeightType::kQ4_K, 16384, 2560);
    CheckPrefill(GgmlType::kQ5_0, gpu::WeightType::kQ5_0, 10240, 320);
    CheckPrefill(GgmlType::kQ5_0, gpu::WeightType::kQ5_0, 2560, 640);
    CheckPrefill(GgmlType::kQ6_K, gpu::WeightType::kQ6_K, 320, 10240);
    CheckPrefill(GgmlType::kQ6_K, gpu::WeightType::kQ6_K, 640, 2560);
    CheckPrefillBasis(GgmlType::kQ4_K, gpu::WeightType::kQ4_K, 512);
    CheckPrefillBasis(GgmlType::kQ5_0, gpu::WeightType::kQ5_0, 96);
    CheckPrefillBasis(GgmlType::kQ6_K, gpu::WeightType::kQ6_K, 512);
    CheckRoutedPrefill();
    std::cout << "PASS: mixed Q4_K/Q5_0/Q6_K embeddings, projections and "
                 "routed experts\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
