#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/diagnostics/system_inventory.h"
#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/qwen_mtp_reference.hpp"
#include "src/core/xdna2/device.h"
#include "src/models/qwen/xdna2/qwen_mtp_eh_proj.h"

namespace {

constexpr std::size_t kBlockElements = 256;
constexpr std::size_t kGroupElements = 32;
constexpr std::size_t kBlocksPerRow =
    strix::xdna2::kQwenMtpEhProjInputElements / kBlockElements;

using BlockQ4K = strix::quant::block_q4_K;

static_assert(sizeof(BlockQ4K) == 144);

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

void SetScaleMin(BlockQ4K& block, std::size_t group, std::uint8_t scale,
                 std::uint8_t minimum) {
  if (group < 4) {
    block.scales[group] =
        static_cast<std::uint8_t>((block.scales[group] & 0xC0U) | scale);
    block.scales[group + 4] =
        static_cast<std::uint8_t>((block.scales[group + 4] & 0xC0U) | minimum);
    return;
  }
  block.scales[group + 4] = static_cast<std::uint8_t>(
      (block.scales[group + 4] & 0xF0U) | (scale & 0x0FU));
  block.scales[group - 4] = static_cast<std::uint8_t>(
      (block.scales[group - 4] & 0x3FU) | ((scale >> 4U) << 6U));
  block.scales[group + 4] = static_cast<std::uint8_t>(
      (block.scales[group + 4] & 0x0FU) | ((minimum & 0x0FU) << 4U));
  block.scales[group] = static_cast<std::uint8_t>(
      (block.scales[group] & 0x3FU) | ((minimum >> 4U) << 6U));
}

void SetQuant(BlockQ4K& block, std::size_t index, std::uint8_t value) {
  const std::size_t group = index / kGroupElements;
  const std::size_t pair = group / 2;
  const std::size_t lane = index % kGroupElements;
  auto& packed = block.qs[(pair * kGroupElements) + lane];
  if ((group & 1U) == 0U) {
    packed = static_cast<std::uint8_t>((packed & 0xF0U) | value);
  } else {
    packed = static_cast<std::uint8_t>((packed & 0x0FU) | (value << 4U));
  }
}

std::vector<BlockQ4K> MakeIdentityWeights() {
  std::vector<BlockQ4K> blocks(strix::xdna2::kQwenMtpEhProjOutputElements *
                               kBlocksPerRow);
  for (std::size_t row = 0; row < strix::xdna2::kQwenMtpEhProjOutputElements;
       ++row) {
    const std::size_t input_index =
        row % strix::xdna2::kQwenMtpEhProjInputElements;
    const std::size_t block_index = input_index / kBlockElements;
    const std::size_t within_block = input_index % kBlockElements;
    auto& block = blocks[(row * kBlocksPerRow) + block_index];
    block.d = 0x3C00U;
    SetScaleMin(block, within_block / kGroupElements, 1, 0);
    SetQuant(block, within_block, 1);
  }
  return blocks;
}

std::vector<float> MakeRepresentableInput() {
  std::vector<float> input(strix::xdna2::kQwenMtpEhProjInputElements);
  for (std::size_t group_start = 0; group_start < input.size();
       group_start += kGroupElements) {
    for (std::size_t lane = 0; lane < kGroupElements; ++lane) {
      const int quantized =
          lane == kGroupElements - 1 ? 127 : static_cast<int>(lane) - 16;
      input[group_start + lane] = static_cast<float>(quantized) / 64.0F;
    }
  }
  return input;
}

struct Comparison {
  double rmse{0.0};
  double cosine{0.0};
  float max_abs{0.0F};
};

Comparison Compare(std::span<const float> actual,
                   std::span<const float> expected) {
  Expect(actual.size() == expected.size(), "comparison size mismatch");
  double squared_error = 0.0;
  double actual_squared = 0.0;
  double expected_squared = 0.0;
  double dot = 0.0;
  float max_abs = 0.0F;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double lhs = actual[index];
    const double rhs = expected[index];
    const double difference = lhs - rhs;
    squared_error += difference * difference;
    actual_squared += lhs * lhs;
    expected_squared += rhs * rhs;
    dot += lhs * rhs;
    max_abs = std::max(max_abs, static_cast<float>(std::abs(difference)));
  }
  return {
      .rmse = std::sqrt(squared_error / static_cast<double>(actual.size())),
      .cosine = dot / std::sqrt(actual_squared * expected_squared),
      .max_abs = max_abs,
  };
}

std::unique_ptr<strix::xdna2::QwenMtpEhProjSession> CreateSession(
    const strix::models::QwenTensorRef& weights,
    const strix::xdna2::XrtDeviceInfo& device, std::string_view context) {
  strix::xdna2::QwenMtpEhProjFailure failure;
  auto session = strix::xdna2::QwenMtpEhProjSession::Create(
      {.program_dir = STRIX_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR}, device, weights,
      &failure);
  Expect(session != nullptr, std::string(context) + ": " + failure.category +
                                 ": " + failure.message);
  return session;
}

void TestSynthetic(const strix::xdna2::XrtDeviceInfo& device) {
  const auto session_count =
      strix::xdna2::QwenMtpEhProjSession::ActiveSessionCountForDiagnostics();
  const auto bo_count =
      strix::xdna2::QwenMtpEhProjSession::ActiveBoCountForDiagnostics();
  const auto blocks = MakeIdentityWeights();
  const strix::models::QwenTensorRef weights{
      .data = blocks.data(),
      .type = strix::core::GgmlType::kQ4_K,
      .num_elements = strix::xdna2::kQwenMtpEhProjOutputElements *
                      strix::xdna2::kQwenMtpEhProjInputElements,
  };
  auto session = CreateSession(weights, device, "synthetic session");
  Expect(
      strix::xdna2::QwenMtpEhProjSession::ActiveSessionCountForDiagnostics() ==
          session_count + 1,
      "session counter increments");
  Expect(strix::xdna2::QwenMtpEhProjSession::ActiveBoCountForDiagnostics() ==
             bo_count + 3,
         "BO counter increments");

  const auto input = MakeRepresentableInput();
  std::vector<float> output(strix::xdna2::kQwenMtpEhProjOutputElements);
  strix::xdna2::QwenMtpEhProjRunMetrics metrics;
  strix::xdna2::QwenMtpEhProjFailure failure;
  for (std::size_t iteration = 0; iteration < 3; ++iteration) {
    Expect(session->Run(input, output, &metrics, &failure),
           failure.category + ": " + failure.message);
  }
  float max_absolute = 0.0F;
  for (std::size_t row = 0; row < output.size(); ++row) {
    max_absolute = std::max(max_absolute, std::abs(output[row] - input[row]));
  }
  Expect(max_absolute <= 1.0e-3F, "synthetic Q4_K identity output is exact");
  Expect(metrics.command_us > 0.0, "command timing is positive");
  Expect(metrics.end_to_end_us >= metrics.command_us,
         "end-to-end timing includes command");
  const auto& info = session->ProgramInfo();
  Expect(info.partition_columns == 8, "program uses all eight NPU2 columns");
  Expect(info.packed_weight_bytes == 52'428'800, "packed weight view size");

  std::cout << "qwen_mtp_eh_proj_synthetic: setup_ms=" << info.setup_ms
            << " weight_pack_ms=" << info.weight_pack_ms
            << " weight_upload_ms=" << info.weight_upload_ms
            << " activation_pack_us=" << metrics.activation_pack_us
            << " command_us=" << metrics.command_us
            << " end_to_end_us=" << metrics.end_to_end_us
            << " max_abs=" << max_absolute << '\n';
  session.reset();
  Expect(
      strix::xdna2::QwenMtpEhProjSession::ActiveSessionCountForDiagnostics() ==
          session_count,
      "session counter returns to baseline");
  Expect(strix::xdna2::QwenMtpEhProjSession::ActiveBoCountForDiagnostics() ==
             bo_count,
         "BO counter returns to baseline");
}

void TestRealModel(const strix::xdna2::XrtDeviceInfo& device) {
  const char* model_path = std::getenv("STRIX_MTP_MODEL");
  if (model_path == nullptr || std::string_view(model_path).empty()) {
    std::cout << "qwen_mtp_eh_proj_real: skipped "
                 "(STRIX_MTP_MODEL not set)\n";
    return;
  }
  std::string error;
  auto reader_owner = strix::core::GgufReader::OpenFile(model_path, &error);
  Expect(reader_owner != nullptr, error);
  std::shared_ptr<const strix::core::GgufReader> reader(
      std::move(reader_owner));
  const auto weights =
      strix::speculative::QwenMtpWeights::LoadFromGguf(*reader, &error);
  Expect(weights.has_value(), error);
  auto session =
      CreateSession(weights->fusion_projection, device, "real model session");

  const auto input = MakeRepresentableInput();
  std::vector<float> output(strix::xdna2::kQwenMtpEhProjOutputElements);
  strix::xdna2::QwenMtpEhProjRunMetrics metrics;
  strix::xdna2::QwenMtpEhProjFailure failure;
  Expect(session->Run(input, output, &metrics, &failure),
         failure.category + ": " + failure.message);

  std::vector<float> expected(output.size());
  const auto* rows =
      static_cast<const std::uint8_t*>(weights->fusion_projection.data);
  const std::size_t row_bytes = strix::quant::QuantizedRowBytes(
      weights->fusion_projection.type,
      strix::xdna2::kQwenMtpEhProjInputElements);
  for (std::size_t row = 0; row < expected.size(); ++row) {
    expected[row] =
        strix::quant::DotProductQ4_K(rows + (row * row_bytes), input,
                                     strix::xdna2::kQwenMtpEhProjInputElements);
  }
  std::size_t nonfinite_output = 0;
  std::size_t nonfinite_expected = 0;
  for (std::size_t row = 0; row < output.size(); ++row) {
    if (!std::isfinite(output[row])) {
      ++nonfinite_output;
    }
    if (!std::isfinite(expected[row])) {
      ++nonfinite_expected;
    }
  }
  Expect(nonfinite_output == 0, "real model NPU output is finite");
  Expect(nonfinite_expected == 0, "real model CPU reference is finite");
  const auto comparison = Compare(output, expected);
  Expect(std::isfinite(comparison.rmse), "real model RMSE is finite");
  Expect(comparison.rmse < 0.1, "real model RMSE");
  Expect(comparison.cosine > 0.9999, "real model cosine");
  Expect(comparison.max_abs < 1.0F, "real model max absolute error");
  std::cout << "qwen_mtp_eh_proj_real: rmse=" << comparison.rmse
            << " cosine=" << comparison.cosine
            << " max_abs=" << comparison.max_abs
            << " command_us=" << metrics.command_us
            << " end_to_end_us=" << metrics.end_to_end_us << '\n';
}

}  // namespace

int main() {
  try {
    const auto inventory = strix::diagnostics::CollectSystemInventory();
    const auto device = strix::xdna2::DiscoverXrtDevice(0, inventory);
    Expect(device.available, device.error_category +
                                 ": detected=" + device.detected +
                                 " required=" + device.required);
    TestRealModel(device);
    TestSynthetic(device);
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "qwen_mtp_xdna2_eh_proj_test failed: " << exception.what()
              << '\n';
    return 1;
  }
}
