// Copyright (C) 2026 Gufo Engine contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gufo/aie_ds4_q2k_down_manifest.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/deepseek_v4_flash/xdna2/q2k_down_contract.hpp"

namespace {

namespace q2k = gufo::models::deepseek_v4_flash::xdna2::q2k;

constexpr std::size_t kInputElements = 2048;
constexpr std::size_t kOutputElements = 4096;
constexpr std::size_t kInputBlocks = kInputElements / q2k::kBlockElements;
constexpr std::size_t kOutputTiles = kOutputElements / q2k::kMmulN;
constexpr std::size_t kArrayColumns = 8;
constexpr std::size_t kArrayRows = 4;
constexpr std::size_t kOutputTilesPerCore =
    kOutputTiles / (kArrayColumns * kArrayRows);
constexpr std::size_t kWeightRowChunkBytes =
    q2k::kWeightRecordBytes / q2k::kMmulN;
constexpr std::size_t kWeightRowBytes = kInputBlocks * kWeightRowChunkBytes;
constexpr std::size_t kPackedWeightBytes = kOutputElements * kWeightRowBytes;
constexpr std::size_t kPackedInputBytes = kInputBlocks * q2k::kInputRecordBytes;
constexpr std::size_t kPackedOutputElements = q2k::kMmulRows * kOutputElements;
constexpr std::size_t kOutputElementsPerColumn =
    kPackedOutputElements / kArrayColumns;
constexpr std::size_t kOutputBytes = kPackedOutputElements * sizeof(float);
constexpr std::uint32_t kDefaultRepetitions = 5;

static_assert(kWeightRowChunkBytes == 384);
static_assert(kPackedWeightBytes == 12'582'912);
static_assert(kPackedInputBytes == 12'288);
static_assert(kOutputElementsPerColumn == 2048);

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

void SetSourceCode(q2k::BlockQ2K& block, std::uint32_t index,
                   std::uint8_t value) {
  const std::uint32_t half = index / 128U;
  const std::uint32_t within_half = index % 128U;
  const std::uint32_t shift = (within_half / 32U) * 2U;
  const std::uint32_t byte = half * 32U + within_half % 32U;
  const std::uint8_t mask = static_cast<std::uint8_t>(0x03U << shift);
  block.codes[byte] = static_cast<std::uint8_t>((block.codes[byte] & ~mask) |
                                                ((value & 0x03U) << shift));
}

q2k::BlockQ2K MakeBlock(std::size_t output_row, std::size_t input_block) {
  q2k::BlockQ2K block{};
  block.d = 0x3000U;     // 0.125
  block.dmin = 0x2800U;  // 0.03125
  for (std::uint32_t group = 0; group < q2k::kGroupsPerBlock; ++group) {
    const auto scale = static_cast<std::uint8_t>(
        1U + (output_row + input_block + group) % 15U);
    const auto minimum =
        static_cast<std::uint8_t>((output_row + 3U * input_block + group) % 4U);
    block.scales[group] = static_cast<std::uint8_t>((minimum << 4U) | scale);
  }
  for (std::uint32_t index = 0; index < q2k::kBlockElements; ++index) {
    SetSourceCode(
        block, index,
        static_cast<std::uint8_t>(
            (output_row + 5U * input_block + index + index / 7U) & 0x03U));
  }
  return block;
}

std::vector<float> MakeInput() {
  std::vector<float> input(q2k::kMmulRows * kInputElements);
  for (std::size_t row = 0; row < q2k::kMmulRows; ++row) {
    for (std::size_t group_start = 0; group_start < kInputElements;
         group_start += q2k::kGroupElements) {
      for (std::size_t lane = 0; lane < q2k::kGroupElements; ++lane) {
        const int code =
            lane == q2k::kGroupElements - 1
                ? 127
                : static_cast<int>((row * 29U + group_start + lane * 11U) %
                                   191U) -
                      95;
        input[row * kInputElements + group_start + lane] =
            static_cast<float>(code) / 64.0F;
      }
    }
  }
  return input;
}

struct PackedExperiment {
  std::vector<std::uint8_t> weights;
  std::vector<std::uint8_t> input;
  std::vector<float> expected;
};

PackedExperiment BuildExperiment() {
  PackedExperiment experiment{
      .weights = std::vector<std::uint8_t>(kPackedWeightBytes),
      .input = std::vector<std::uint8_t>(kPackedInputBytes),
      .expected = std::vector<float>(kPackedOutputElements),
  };
  const std::vector<float> input = MakeInput();
  for (std::size_t input_block = 0; input_block < kInputBlocks; ++input_block) {
    std::span<std::uint8_t, q2k::kInputRecordBytes> record(
        experiment.input.data() + input_block * q2k::kInputRecordBytes,
        q2k::kInputRecordBytes);
    q2k::PackInputRecord(input, q2k::kMmulRows, kInputElements, input_block,
                         record);
  }

  for (std::size_t output_tile = 0; output_tile < kOutputTiles; ++output_tile) {
    std::array<float, q2k::kMmulRows * q2k::kMmulN> tile_expected{};
    for (std::size_t input_block = 0; input_block < kInputBlocks;
         ++input_block) {
      std::array<q2k::BlockQ2K, q2k::kMmulN> blocks{};
      std::array<const q2k::BlockQ2K*, q2k::kMmulN> source_rows{};
      for (std::size_t lane = 0; lane < q2k::kMmulN; ++lane) {
        blocks[lane] = MakeBlock(output_tile * q2k::kMmulN + lane, input_block);
        source_rows[lane] = &blocks[lane];
      }

      std::array<std::uint8_t, q2k::kWeightRecordBytes> record{};
      q2k::PackWeightRecord(source_rows, record);
      for (std::size_t local_row = 0; local_row < q2k::kMmulN; ++local_row) {
        const std::size_t output_row = output_tile * q2k::kMmulN + local_row;
        std::memcpy(experiment.weights.data() + output_row * kWeightRowBytes +
                        input_block * kWeightRowChunkBytes,
                    record.data() + local_row * kWeightRowChunkBytes,
                    kWeightRowChunkBytes);
      }

      const std::span<const std::uint8_t, q2k::kInputRecordBytes> input_record(
          experiment.input.data() + input_block * q2k::kInputRecordBytes,
          q2k::kInputRecordBytes);
      q2k::AccumulateSourceRecord(source_rows, input_record, tile_expected);
    }
    for (std::size_t row = 0; row < q2k::kMmulRows; ++row) {
      for (std::size_t lane = 0; lane < q2k::kMmulN; ++lane) {
        experiment.expected[row * kOutputElements + output_tile * q2k::kMmulN +
                            lane] = tile_expected[row * q2k::kMmulN + lane];
      }
    }
  }
  return experiment;
}

std::filesystem::path ProgramDir() {
#ifdef GUFO_AIE_DS4_Q2K_DOWN_PROGRAM_DIR
  return GUFO_AIE_DS4_Q2K_DOWN_PROGRAM_DIR;
#else
  return {};
#endif
}

std::string SelectKernelName(const xrt::xclbin& xclbin) {
  const auto kernels = xclbin.get_kernels();
  const auto match =
      std::ranges::find_if(kernels, [](const xrt::xclbin::kernel& kernel) {
        return kernel.get_name().starts_with("MLIR_AIE");
      });
  if (match != kernels.end()) {
    return match->get_name();
  }
  return kernels.size() == 1 ? kernels.front().get_name() : std::string{};
}

std::uint32_t Repetitions() {
  const char* value = std::getenv("GUFO_DS4_Q2K_DOWN_REPETITIONS");
  if (value == nullptr || value[0] == '\0') {
    return kDefaultRepetitions;
  }
  const unsigned long parsed = std::stoul(value);
  Expect(parsed > 0 && parsed <= 100, "repetitions must be in [1, 100]");
  return static_cast<std::uint32_t>(parsed);
}

double Median(std::vector<double> samples) {
  Expect(!samples.empty(), "median requires samples");
  std::ranges::sort(samples);
  const std::size_t middle = samples.size() / 2;
  if ((samples.size() & 1U) != 0U) {
    return samples[middle];
  }
  return 0.5 * (samples[middle - 1] + samples[middle]);
}

struct Comparison {
  double rmse{0.0};
  double cosine{0.0};
  float max_abs{0.0F};
};

Comparison CompareTiledOutput(std::span<const float> actual,
                              std::span<const float> expected) {
  double squared_error = 0.0;
  double actual_squared = 0.0;
  double expected_squared = 0.0;
  double dot = 0.0;
  float max_abs = 0.0F;
  for (std::size_t column = 0; column < kArrayColumns; ++column) {
    for (std::size_t iteration = 0; iteration < kOutputTilesPerCore;
         ++iteration) {
      for (std::size_t worker_row = 0; worker_row < kArrayRows; ++worker_row) {
        for (std::size_t activation_row = 0; activation_row < q2k::kMmulRows;
             ++activation_row) {
          for (std::size_t lane = 0; lane < q2k::kMmulN; ++lane) {
            const std::size_t actual_index =
                column * kOutputElementsPerColumn +
                iteration * kArrayRows * q2k::kMmulRows * q2k::kMmulN +
                worker_row * q2k::kMmulRows * q2k::kMmulN +
                activation_row * q2k::kMmulN + lane;
            const std::size_t output_column =
                column * kOutputElements / kArrayColumns +
                iteration * kArrayRows * q2k::kMmulN +
                worker_row * q2k::kMmulN + lane;
            const std::size_t expected_index =
                activation_row * kOutputElements + output_column;
            const double lhs = actual[actual_index];
            const double rhs = expected[expected_index];
            const double difference = lhs - rhs;
            squared_error += difference * difference;
            actual_squared += lhs * lhs;
            expected_squared += rhs * rhs;
            dot += lhs * rhs;
            max_abs =
                std::max(max_abs, static_cast<float>(std::abs(difference)));
          }
        }
      }
    }
  }
  return {
      .rmse = std::sqrt(squared_error / static_cast<double>(expected.size())),
      .cosine = dot / std::sqrt(actual_squared * expected_squared),
      .max_abs = max_abs,
  };
}

void Run() {
  Expect(gufo::xdna2::generated::kDs4Q2kDownTarget == "npu2",
         "unexpected AIE target");
  Expect(gufo::xdna2::generated::kDs4Q2kDownTensorContract ==
             "q2_k-u8-dyn-i8-g16-int32-fp32-m4-n4096-k2048",
         "unexpected AIE tensor contract");
  Expect(gufo::xdna2::generated::kDs4Q2kDownPartitionColumns == 8,
         "experiment requires all eight NPU columns");

  const auto setup_start = std::chrono::steady_clock::now();
  PackedExperiment experiment = BuildExperiment();
  const auto packed_at = std::chrono::steady_clock::now();

  const std::filesystem::path program_dir = ProgramDir();
  Expect(!program_dir.empty(), "DS4 Q2_K AIE program directory is unset");
  xrt::device device(0);
  xrt::xclbin xclbin((program_dir / "ds4_q2k_down.xclbin").string());
  const std::string kernel_name = SelectKernelName(xclbin);
  Expect(!kernel_name.empty(), "AIE XCLBIN has no unique MLIR_AIE kernel");
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  xrt::elf elf((program_dir / "ds4_q2k_down.insts.elf").string());
  xrt::module module(elf);
  xrt::ext::kernel kernel(context, module, kernel_name);
  xrt::bo weight_bo = xrt::ext::bo(device, kPackedWeightBytes);
  xrt::bo input_bo = xrt::ext::bo(device, kPackedInputBytes);
  xrt::bo output_bo = xrt::ext::bo(device, kOutputBytes);
  auto* weight = weight_bo.map<std::uint8_t*>();
  auto* input = input_bo.map<std::uint8_t*>();
  auto* output = output_bo.map<float*>();
  Expect(weight != nullptr && input != nullptr && output != nullptr,
         "XRT returned a null BO mapping");
  std::ranges::copy(experiment.weights, weight);
  std::ranges::copy(experiment.input, input);
  std::fill_n(output, kPackedOutputElements, 0.0F);
  weight_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  const auto ready_at = std::chrono::steady_clock::now();

  std::vector<double> command_samples;
  std::vector<double> end_to_end_samples;
  const std::uint32_t repetitions = Repetitions();
  for (std::uint32_t iteration = 0; iteration <= repetitions; ++iteration) {
    const auto end_to_end_start = std::chrono::steady_clock::now();
    input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    const auto command_start = std::chrono::steady_clock::now();
    auto run = kernel(3U, 0U, 0U, weight_bo, input_bo, output_bo);
    const auto status = run.wait2(std::chrono::seconds(30));
    const auto command_end = std::chrono::steady_clock::now();
    Expect(status != std::cv_status::timeout, "AIE command timed out");
    Expect(run.state() == ERT_CMD_STATE_COMPLETED,
           "AIE command did not complete");
    output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const auto end_to_end_end = std::chrono::steady_clock::now();
    if (iteration != 0) {
      command_samples.push_back(
          std::chrono::duration<double, std::micro>(command_end - command_start)
              .count());
      end_to_end_samples.push_back(std::chrono::duration<double, std::micro>(
                                       end_to_end_end - end_to_end_start)
                                       .count());
    }
  }

  const Comparison comparison =
      CompareTiledOutput(std::span<const float>(output, kPackedOutputElements),
                         experiment.expected);
  Expect(std::isfinite(comparison.rmse), "NPU RMSE is finite");
  Expect(comparison.rmse < 0.01, "NPU RMSE exceeds the experiment gate");
  Expect(comparison.cosine > 0.99999,
         "NPU cosine is below the experiment gate");
  Expect(comparison.max_abs < 0.1F,
         "NPU max absolute error exceeds the experiment gate");

  const double command_us = Median(command_samples);
  const double end_to_end_us = Median(end_to_end_samples);
  constexpr double operations =
      2.0 * q2k::kMmulRows * kOutputElements * kInputElements;
  const double effective_tops = operations / (command_us * 1.0e6);
  std::cout
      << "ds4_q2k_down_xdna2: command_us=" << command_us
      << " end_to_end_us=" << end_to_end_us
      << " effective_tops=" << effective_tops << " rmse=" << comparison.rmse
      << " cosine=" << comparison.cosine << " max_abs=" << comparison.max_abs
      << " packed_weight_mib="
      << static_cast<double>(kPackedWeightBytes) / (1024.0 * 1024.0)
      << " pack_ms="
      << std::chrono::duration<double, std::milli>(packed_at - setup_start)
             .count()
      << " xrt_setup_upload_ms="
      << std::chrono::duration<double, std::milli>(ready_at - packed_at).count()
      << " repetitions=" << repetitions << '\n';
}

}  // namespace

int main() {
  try {
    Run();
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "ds4_q2k_down_xdna2 failed: " << exception.what() << '\n';
    return 1;
  }
}
