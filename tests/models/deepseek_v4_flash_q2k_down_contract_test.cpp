#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/deepseek_v4_flash/xdna2/q2k_down_contract.hpp"

namespace {

namespace q2k = gufo::models::deepseek_v4_flash::xdna2::q2k;

constexpr std::uint32_t kInputColumns = 2048;
constexpr std::uint32_t kBlocks = kInputColumns / q2k::kBlockElements;

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

using ExpertRows = std::array<std::array<q2k::BlockQ2K, kBlocks>, q2k::kMmulN>;

ExpertRows MakeWeights() {
  ExpertRows weights{};
  for (std::uint32_t lane = 0; lane < q2k::kMmulN; ++lane) {
    for (std::uint32_t block_index = 0; block_index < kBlocks; ++block_index) {
      auto& block = weights[lane][block_index];
      block.d = 0x3C00U;     // 1.0
      block.dmin = 0x3400U;  // 0.25
      for (std::uint32_t group = 0; group < q2k::kGroupsPerBlock; ++group) {
        const std::uint8_t scale =
            static_cast<std::uint8_t>(1U + (lane + group) % 15U);
        const std::uint8_t minimum =
            static_cast<std::uint8_t>((lane + 2U * group) % 4U);
        block.scales[group] =
            static_cast<std::uint8_t>((minimum << 4U) | scale);
      }
      for (std::uint32_t index = 0; index < q2k::kBlockElements; ++index) {
        SetSourceCode(block, index,
                      static_cast<std::uint8_t>(
                          (lane + block_index + index / 7U + index) & 0x03U));
      }
    }
  }
  return weights;
}

std::vector<float> MakeInputs() {
  std::vector<float> input(q2k::kMmulRows * kInputColumns);
  for (std::uint32_t row = 0; row < q2k::kMmulRows; ++row) {
    for (std::uint32_t group_start = 0; group_start < kInputColumns;
         group_start += q2k::kGroupElements) {
      for (std::uint32_t lane = 0; lane < q2k::kGroupElements; ++lane) {
        const int code =
            lane == q2k::kGroupElements - 1
                ? 127
                : static_cast<int>((row * 17U + group_start + lane) % 191U) -
                      95;
        input[static_cast<std::size_t>(row) * kInputColumns + group_start +
              lane] = static_cast<float>(code) / 64.0F;
      }
    }
  }
  return input;
}

void TestSourceCodeLayout() {
  q2k::BlockQ2K block{};
  for (std::uint32_t index = 0; index < q2k::kBlockElements; ++index) {
    SetSourceCode(block, index, static_cast<std::uint8_t>(index & 0x03U));
  }
  for (std::uint32_t index = 0; index < q2k::kBlockElements; ++index) {
    Expect(q2k::SourceCode(block, index) == (index & 0x03U),
           "Q2_K source code layout");
  }
}

void TestPackedContract() {
  const ExpertRows weights = MakeWeights();
  const std::vector<float> input = MakeInputs();
  std::array<float, q2k::kMmulRows * q2k::kMmulN> packed_output{};
  std::array<float, q2k::kMmulRows * q2k::kMmulN> source_output{};

  for (std::uint32_t block_index = 0; block_index < kBlocks; ++block_index) {
    std::array<const q2k::BlockQ2K*, q2k::kMmulN> source_rows{};
    for (std::uint32_t lane = 0; lane < q2k::kMmulN; ++lane) {
      source_rows[lane] = &weights[lane][block_index];
    }

    std::array<std::uint8_t, q2k::kWeightRecordBytes> packed_weights{};
    std::array<std::uint8_t, q2k::kInputRecordBytes> packed_input{};
    q2k::PackWeightRecord(source_rows, packed_weights);
    q2k::PackInputRecord(input, q2k::kMmulRows, kInputColumns, block_index,
                         packed_input);

    for (std::uint32_t lane = 0; lane < q2k::kMmulN; ++lane) {
      for (std::uint32_t group = 0; group < q2k::kGroupsPerBlock; ++group) {
        Expect(
            q2k::LoadScalar<float>(packed_weights.data(),
                                   q2k::WeightParameterOffset(
                                       q2k::kWeightScaleOffset, group, lane)) ==
                q2k::SourceScale(*source_rows[lane], group),
            "Q2_K packed scale");
        Expect(q2k::LoadScalar<float>(
                   packed_weights.data(),
                   q2k::WeightParameterOffset(q2k::kWeightMinimumOffset, group,
                                              lane)) ==
                   q2k::SourceMinimum(*source_rows[lane], group),
               "Q2_K packed minimum");
        for (std::uint32_t k = 0; k < q2k::kMmulK; ++k) {
          Expect(packed_weights[q2k::WeightCodeOffset(group, k, lane)] ==
                     q2k::SourceCode(*source_rows[lane],
                                     group * q2k::kGroupElements + k),
                 "Q2_K packed code");
        }
      }
    }

    q2k::AccumulatePackedRecord(packed_weights, packed_input, packed_output);
    q2k::AccumulateSourceRecord(source_rows, packed_input, source_output);
  }

  float maximum_error = 0.0F;
  for (std::size_t index = 0; index < packed_output.size(); ++index) {
    Expect(std::isfinite(packed_output[index]), "packed output is finite");
    maximum_error = std::max(
        maximum_error, std::abs(packed_output[index] - source_output[index]));
  }
  Expect(maximum_error == 0.0F,
         "packed Q2_K record is numerically identical to source contract");
  std::cout << "q2k_down_contract: max_abs=" << maximum_error
            << " weight_record_bytes=" << q2k::kWeightRecordBytes
            << " input_record_bytes=" << q2k::kInputRecordBytes << '\n';
}

void TestActiveRowMask() {
  const ExpertRows weights = MakeWeights();
  const std::vector<float> input = MakeInputs();
  std::array<const q2k::BlockQ2K*, q2k::kMmulN> source_rows{};
  for (std::uint32_t lane = 0; lane < q2k::kMmulN; ++lane) {
    source_rows[lane] = &weights[lane][0];
  }
  std::array<std::uint8_t, q2k::kWeightRecordBytes> packed_weights{};
  std::array<std::uint8_t, q2k::kInputRecordBytes> packed_input{};
  std::array<float, q2k::kMmulRows * q2k::kMmulN> output{};
  q2k::PackWeightRecord(source_rows, packed_weights);
  q2k::PackInputRecord(input, 3, kInputColumns, 0, packed_input);
  q2k::AccumulatePackedRecord(packed_weights, packed_input, output);
  for (std::uint32_t lane = 0; lane < q2k::kMmulN; ++lane) {
    Expect(output[3U * q2k::kMmulN + lane] == 0.0F,
           "inactive physical row remains zero");
  }
}

}  // namespace

int main() {
  try {
    TestSourceCodeLayout();
    TestPackedContract();
    TestActiveRowMask();
    std::cout << "DeepSeek V4 Flash Q2_K down contract: PASS\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "deepseek_v4_flash_q2k_down_contract_test failed: "
              << exception.what() << '\n';
    return 1;
  }
}
