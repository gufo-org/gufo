#include "src/models/qwen3_tts/hip/encoder_support.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <gufo/ext/experimental/rocblas_gemm.hpp>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

namespace gufo::models::qwen3_tts::hip {
namespace {

float RoundBfloat16(float value) {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  if ((bits & 0x7F800000U) != 0x7F800000U) {
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
  }
  return std::bit_cast<float>(bits & 0xFFFF0000U);
}

std::vector<float> ReadEncoderTensor(const Tensor& tensor,
                                     std::size_t elements) {
  std::vector<float> host(elements);
  if (tensor.dtype == DType::kF32) {
    std::memcpy(host.data(), tensor.data, host.size() * sizeof(float));
  } else {
    const auto* source = reinterpret_cast<const std::uint16_t*>(tensor.data);
    for (std::size_t index = 0; index < host.size(); ++index) {
      host[index] = std::bit_cast<float>(
          static_cast<std::uint32_t>(source[index]) << 16U);
    }
  }
  return host;
}

}  // namespace

void RequireEncoderHip(hipError_t status, std::string_view operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

void RequireEncoderRocblas(rocblas_status status, std::string_view operation) {
  if (status != rocblas_status_success) {
    throw std::runtime_error(std::string(operation) + ": status " +
                             std::to_string(static_cast<int>(status)));
  }
}

const float* EncoderWeights::Load(std::string_view name,
                                  std::span<const std::uint64_t> shape) {
  const Tensor* tensor = store_.Find(name);
  if (tensor == nullptr || !std::ranges::equal(tensor->shape, shape) ||
      (tensor->dtype != DType::kF32 && tensor->dtype != DType::kBF16)) {
    throw std::runtime_error("unexpected Qwen3-TTS encoder tensor: " +
                             std::string(name));
  }
  const auto elements = tensor->NumElements();
  if (!elements.has_value()) {
    throw std::runtime_error("invalid Qwen3-TTS encoder tensor size: " +
                             std::string(name));
  }
  std::vector<float> host = ReadEncoderTensor(*tensor, *elements);
  for (float& value : host) {
    value = RoundBfloat16(value);
  }
  auto allocation = std::make_unique<EncoderDeviceBuffer<float>>(host.size());
  RequireEncoderHip(
      hipMemcpy(allocation->get(), host.data(), host.size() * sizeof(float),
                hipMemcpyHostToDevice),
      "copy Qwen3-TTS encoder weight");
  const float* pointer = allocation->get();
  allocations_.push_back(std::move(allocation));
  return pointer;
}

const float* EncoderWeights::LoadBfloat16CodebookEmbedding(
    std::string_view embedding_sum_name, std::string_view cluster_usage_name,
    std::size_t codebook_size, std::size_t dimension) {
  const Tensor* embedding_sum = store_.Find(embedding_sum_name);
  const Tensor* cluster_usage = store_.Find(cluster_usage_name);
  const std::array<std::uint64_t, 2> embedding_shape{codebook_size, dimension};
  const std::array<std::uint64_t, 1> usage_shape{codebook_size};
  if (embedding_sum == nullptr || cluster_usage == nullptr ||
      !std::ranges::equal(embedding_sum->shape, embedding_shape) ||
      !std::ranges::equal(cluster_usage->shape, usage_shape) ||
      (embedding_sum->dtype != DType::kF32 &&
       embedding_sum->dtype != DType::kBF16) ||
      (cluster_usage->dtype != DType::kF32 &&
       cluster_usage->dtype != DType::kBF16)) {
    throw std::runtime_error("unexpected Qwen3-TTS encoder codebook tensors");
  }
  std::vector<float> sums =
      ReadEncoderTensor(*embedding_sum, codebook_size * dimension);
  std::vector<float> usage = ReadEncoderTensor(*cluster_usage, codebook_size);
  std::vector<float> embedding(sums.size());
  for (std::size_t code = 0; code < codebook_size; ++code) {
    const float denominator = std::max(RoundBfloat16(usage[code]), 1.0e-5F);
    for (std::size_t column = 0; column < dimension; ++column) {
      const std::size_t index = code * dimension + column;
      embedding[column * codebook_size + code] =
          RoundBfloat16(RoundBfloat16(sums[index]) / denominator);
    }
  }
  auto allocation =
      std::make_unique<EncoderDeviceBuffer<float>>(embedding.size());
  RequireEncoderHip(
      hipMemcpy(allocation->get(), embedding.data(),
                embedding.size() * sizeof(float), hipMemcpyHostToDevice),
      "copy Qwen3-TTS encoder codebook embedding");
  const float* pointer = allocation->get();
  allocations_.push_back(std::move(allocation));
  return pointer;
}

EncoderF32Gemm::EncoderF32Gemm() {
  RequireEncoderRocblas(rocblas_create_handle(&handle_),
                        "rocblas_create_handle");
  RequireEncoderRocblas(
      rocblas_set_atomics_mode(handle_, rocblas_atomics_not_allowed),
      "rocblas_set_atomics_mode");
}

EncoderF32Gemm::~EncoderF32Gemm() {
  if (handle_ != nullptr) {
    (void)rocblas_destroy_handle(handle_);
  }
}

void EncoderF32Gemm::Run(const float* weight, const float* input, float* output,
                         std::size_t rows, std::size_t output_columns,
                         std::size_t reduction, hipStream_t stream) {
  constexpr std::size_t maximum =
      static_cast<std::size_t>(std::numeric_limits<rocblas_int>::max());
  if (rows == 0 || output_columns == 0 || reduction == 0 || rows > maximum ||
      output_columns > maximum || reduction > maximum) {
    throw std::length_error("invalid Qwen3-TTS encoder GEMM shape");
  }
  RequireEncoderRocblas(rocblas_set_stream(handle_, stream),
                        "rocblas_set_stream");
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  RequireEncoderRocblas(
      rocblas_gemm_ex(
          handle_, rocblas_operation_transpose, rocblas_operation_none,
          static_cast<rocblas_int>(output_columns),
          static_cast<rocblas_int>(rows), static_cast<rocblas_int>(reduction),
          &alpha, weight, rocblas_datatype_f32_r,
          static_cast<rocblas_int>(reduction), input, rocblas_datatype_f32_r,
          static_cast<rocblas_int>(reduction), &beta,
          gufo::hip::BetaZeroGemmC(output, output_columns, rows),
          rocblas_datatype_f32_r, static_cast<rocblas_int>(output_columns),
          output, rocblas_datatype_f32_r,
          static_cast<rocblas_int>(output_columns), rocblas_datatype_f32_r,
          rocblas_gemm_algo_standard, 0, 0),
      "Qwen3-TTS encoder GEMM");
}

}  // namespace gufo::models::qwen3_tts::hip

#endif
