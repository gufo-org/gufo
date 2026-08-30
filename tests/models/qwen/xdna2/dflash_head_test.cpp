#include "src/models/qwen/xdna2/dflash_head.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/diagnostics/system_inventory.h"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/xdna2/device.h"

namespace {

constexpr std::size_t kBlocksPerRow =
    gufo::xdna2::kQwenDFlashHeadInputElements / 32;

using BlockQ8 = gufo::quant::block_q8_0;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

std::vector<BlockQ8> MakeIdentityWeights() {
  std::vector<BlockQ8> blocks(gufo::xdna2::kQwenDFlashHeadOutputElements *
                              kBlocksPerRow);
  for (std::size_t row = 0; row < gufo::xdna2::kQwenDFlashHeadOutputElements;
       ++row) {
    const std::size_t input_index =
        row % gufo::xdna2::kQwenDFlashHeadInputElements;
    auto& block = blocks[(row * kBlocksPerRow) + (input_index / 32)];
    block.d = 0x3C00U;
    block.qs[input_index % 32] = 1;
  }
  return blocks;
}

std::vector<float> MakeRepresentableInput(std::size_t row_count) {
  std::vector<float> input(row_count *
                           gufo::xdna2::kQwenDFlashHeadInputElements);
  for (std::size_t row = 0; row < row_count; ++row) {
    for (std::size_t group_start = 0;
         group_start < gufo::xdna2::kQwenDFlashHeadInputElements;
         group_start += 32) {
      for (std::size_t lane = 0; lane < 32; ++lane) {
        const int quantized = lane == 31 ? 127 : static_cast<int>(lane) - 16;
        input[(row * gufo::xdna2::kQwenDFlashHeadInputElements) + group_start +
              lane] = static_cast<float>(quantized) / 64.0F;
      }
    }
  }
  return input;
}

float MaximumIdentityError(std::span<const float> input,
                           std::span<const float> output,
                           std::size_t row_count) {
  float maximum = 0.0F;
  for (std::size_t batch_row = 0; batch_row < row_count; ++batch_row) {
    for (std::size_t output_row = 0;
         output_row < gufo::xdna2::kQwenDFlashHeadOutputElements;
         ++output_row) {
      const float expected =
          input[(batch_row * gufo::xdna2::kQwenDFlashHeadInputElements) +
                (output_row % gufo::xdna2::kQwenDFlashHeadInputElements)];
      const float actual =
          output[(batch_row * gufo::xdna2::kQwenDFlashHeadOutputElements) +
                 output_row];
      maximum = std::max(maximum, std::abs(actual - expected));
    }
  }
  return maximum;
}

std::unique_ptr<gufo::xdna2::QwenDFlashHeadSession> CreateSession(
    const gufo::models::QwenTensorRef& weights,
    const gufo::xdna2::XrtDeviceInfo& device) {
  gufo::xdna2::QwenDFlashHeadFailure failure;
  auto session = gufo::xdna2::QwenDFlashHeadSession::Create(
      {.program_dir = GUFO_AIE_QWEN_DFLASH_HEAD_PROGRAM_DIR}, device, weights,
      &failure);
  Expect(session != nullptr, failure.category + ": " + failure.message);
  return session;
}

void TestSynthetic(const gufo::xdna2::XrtDeviceInfo& device) {
  const auto session_count =
      gufo::xdna2::QwenDFlashHeadSession::ActiveSessionCountForDiagnostics();
  const auto bo_count =
      gufo::xdna2::QwenDFlashHeadSession::ActiveBoCountForDiagnostics();
  const auto blocks = MakeIdentityWeights();
  const gufo::models::QwenTensorRef weights{
      .data = blocks.data(),
      .type = gufo::core::GgmlType::kQ8_0,
      .num_elements = gufo::xdna2::kQwenDFlashHeadOutputElements *
                      gufo::xdna2::kQwenDFlashHeadInputElements,
  };
  auto session = CreateSession(weights, device);
  Expect(
      gufo::xdna2::QwenDFlashHeadSession::ActiveSessionCountForDiagnostics() ==
          session_count + 1,
      "session counter increments");
  Expect(gufo::xdna2::QwenDFlashHeadSession::ActiveBoCountForDiagnostics() ==
             bo_count + 3,
         "BO counter increments");

  gufo::xdna2::QwenDFlashHeadRunMetrics metrics;
  gufo::xdna2::QwenDFlashHeadFailure failure;
  float maximum = 0.0F;
  for (const std::size_t row_count : {std::size_t{8}, std::size_t{3}}) {
    const auto input = MakeRepresentableInput(row_count);
    std::vector<float> output(row_count *
                              gufo::xdna2::kQwenDFlashHeadOutputElements);
    Expect(session->Run(input, row_count, output, &metrics, &failure),
           failure.category + ": " + failure.message);
    maximum = std::max(maximum, MaximumIdentityError(input, output, row_count));
  }
  Expect(maximum <= 1.0e-3F, "synthetic Q8_0 identity output is exact");
  Expect(metrics.command_us > 0.0, "command timing is positive");
  Expect(metrics.end_to_end_us >= metrics.command_us,
         "end-to-end timing includes command");
  const auto& info = session->ProgramInfo();
  Expect(info.partition_columns == 8, "program uses all eight NPU2 columns");
  Expect(info.packed_weight_bytes == 47'185'920, "packed weight view size");

  std::cout << "qwen_dflash_head_synthetic: setup_ms=" << info.setup_ms
            << " weight_pack_ms=" << info.weight_pack_ms
            << " weight_upload_ms=" << info.weight_upload_ms
            << " activation_pack_us=" << metrics.activation_pack_us
            << " command_us=" << metrics.command_us
            << " output_unpack_us=" << metrics.output_unpack_us
            << " end_to_end_us=" << metrics.end_to_end_us
            << " max_abs=" << maximum << '\n';

  session.reset();
  Expect(
      gufo::xdna2::QwenDFlashHeadSession::ActiveSessionCountForDiagnostics() ==
          session_count,
      "session counter returns to baseline");
  Expect(gufo::xdna2::QwenDFlashHeadSession::ActiveBoCountForDiagnostics() ==
             bo_count,
         "BO counter returns to baseline");
}

}  // namespace

int main() {
  try {
    const auto inventory = gufo::diagnostics::CollectSystemInventory();
    const auto device = gufo::xdna2::DiscoverXrtDevice(0, inventory);
    Expect(device.available, device.error_category +
                                 ": detected=" + device.detected +
                                 " required=" + device.required);
    TestSynthetic(device);
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "qwen_dflash_head_test failed: " << exception.what() << '\n';
    return 1;
  }
}
