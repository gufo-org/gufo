#include "src/core/quant/ggml_dequant.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace {

constexpr std::size_t kBlockElements = 256;

void ExpectNear(float actual, float expected, std::string_view message,
                float tolerance = 1.0e-5F) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << "Assertion failed: " << message << " actual=" << actual
              << " expected=" << expected << '\n';
    std::exit(1);
  }
}

std::array<std::uint8_t, 144> MakeQ4Block(
    const std::array<std::uint8_t, 8>& scales,
    const std::array<std::uint8_t, 8>& minima) {
  std::array<std::uint8_t, 144> block{};
  block[0] = 0x00;
  block[1] = 0x3C;
  block[2] = 0x00;
  block[3] = 0x38;
  for (std::size_t index = 0; index < 4; ++index) {
    block[4 + index] = static_cast<std::uint8_t>(
        (scales[index] & 0x3FU) | ((scales[index + 4] >> 4U) << 6U));
    block[8 + index] = static_cast<std::uint8_t>(
        (minima[index] & 0x3FU) | ((minima[index + 4] >> 4U) << 6U));
    block[12 + index] = static_cast<std::uint8_t>(
        (scales[index + 4] & 0x0FU) | ((minima[index + 4] & 0x0FU) << 4U));
  }
  for (std::size_t pair = 0; pair < 4; ++pair) {
    for (std::size_t lane = 0; lane < 32; ++lane) {
      const auto low = static_cast<std::uint8_t>((lane + pair) & 0x0FU);
      const auto high = static_cast<std::uint8_t>((15U - lane + pair) & 0x0FU);
      block[16 + (pair * 32) + lane] =
          static_cast<std::uint8_t>(low | (high << 4U));
    }
  }
  return block;
}

std::array<std::uint8_t, 210> MakeQ6Block(
    const std::array<std::int8_t, 16>& scales) {
  std::array<std::uint8_t, 210> block{};
  const auto logical_quant = [](std::size_t index) {
    return static_cast<std::int8_t>(static_cast<int>((index * 7U) % 64U) - 32);
  };
  for (std::size_t half = 0; half < 2; ++half) {
    for (std::size_t lane = 0; lane < 32; ++lane) {
      std::array<std::uint8_t, 4> code{};
      for (std::size_t segment = 0; segment < code.size(); ++segment) {
        code[segment] = static_cast<std::uint8_t>(
            logical_quant((half * 128) + (segment * 32) + lane) + 32);
      }
      block[(half * 64) + lane] = static_cast<std::uint8_t>(
          (code[0] & 0x0FU) | ((code[2] & 0x0FU) << 4U));
      block[(half * 64) + 32 + lane] = static_cast<std::uint8_t>(
          (code[1] & 0x0FU) | ((code[3] & 0x0FU) << 4U));
      block[128 + (half * 32) + lane] = static_cast<std::uint8_t>(
          ((code[0] >> 4U) & 0x03U) | (((code[1] >> 4U) & 0x03U) << 2U) |
          (((code[2] >> 4U) & 0x03U) << 4U) |
          (((code[3] >> 4U) & 0x03U) << 6U));
    }
  }
  std::ranges::copy(scales, block.begin() + 192);
  block[208] = 0x00;
  block[209] = 0x3C;
  return block;
}

std::array<std::uint8_t, 110> MakeQ3Block(
    const std::array<std::int8_t, 16>& scales) {
  std::array<std::uint8_t, 110> block{};
  for (std::size_t index = 0; index < scales.size(); ++index) {
    const auto encoded = static_cast<std::uint8_t>(scales[index] + 32);
    if (index < 8) {
      block[96 + index] |= encoded & 0x0FU;
    } else {
      block[96 + index - 8] |=
          static_cast<std::uint8_t>((encoded & 0x0FU) << 4U);
    }
    block[104 + (index % 4)] |= static_cast<std::uint8_t>(
        ((encoded >> 4U) & 0x03U) << (2U * (index / 4)));
  }

  for (std::size_t index = 0; index < kBlockElements; ++index) {
    const std::size_t half = index / 128;
    const std::size_t within_half = index % 128;
    const std::size_t pair = within_half / 32;
    const std::size_t lane = within_half % 32;
    const auto quant =
        static_cast<std::int8_t>(static_cast<int>((index * 3U) % 8U) - 4);
    const auto low = static_cast<std::uint8_t>(quant < 0 ? quant + 4 : quant);
    block[32 + (half * 32) + lane] |=
        static_cast<std::uint8_t>(low << (2U * pair));
    if (quant >= 0) {
      block[lane] |= static_cast<std::uint8_t>(1U << ((half * 4) + pair));
    }
  }
  block[108] = 0x00;
  block[109] = 0x3C;
  return block;
}

void TestQ4() {
  const std::array<std::uint8_t, 8> scales{1, 7, 15, 31, 32, 47, 55, 63};
  const std::array<std::uint8_t, 8> minima{0, 1, 2, 3, 4, 5, 6, 7};
  const auto block = MakeQ4Block(scales, minima);
  std::array<float, kBlockElements> actual{};
  gufo::quant::DequantizeQ4_K(block.data(), actual.data(), actual.size());

  std::array<float, kBlockElements> expected{};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const std::size_t group = index / 32;
    const std::size_t pair = group / 2;
    const std::size_t lane = index % 32;
    const auto quant =
        group % 2 == 0 ? static_cast<std::uint8_t>((lane + pair) & 0x0FU)
                       : static_cast<std::uint8_t>((15U - lane + pair) & 0x0FU);
    expected[index] = static_cast<float>(scales[group] * quant) -
                      (0.5F * static_cast<float>(minima[group]));
    ExpectNear(actual[index], expected[index], "Q4_K dequant value");
  }

  std::array<float, kBlockElements> vector{};
  std::iota(vector.begin(), vector.end(), -0.5F);
  const float expected_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(gufo::quant::DotProductQ4_K(block.data(), vector, vector.size()),
             expected_dot, "Q4_K dot product");
}

void TestQ6() {
  const std::array<std::int8_t, 16> scales{-8, -7, -6, -5, -4, -3, -2, -1,
                                           1,  2,  3,  4,  5,  6,  7,  8};
  const auto block = MakeQ6Block(scales);
  std::array<float, kBlockElements> actual{};
  gufo::quant::DequantizeQ6_K(block.data(), actual.data(), actual.size());

  std::array<float, kBlockElements> expected{};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const std::size_t half = index / 128;
    const std::size_t within_half = index % 128;
    const std::size_t segment = within_half / 32;
    const std::size_t lane = within_half % 32;
    const std::size_t scale_index = (half * 8) + (lane / 16) + (segment * 2);
    const auto quant =
        static_cast<std::int8_t>(static_cast<int>((index * 7U) % 64U) - 32);
    expected[index] =
        static_cast<float>(scales[scale_index]) * static_cast<float>(quant);
    ExpectNear(actual[index], expected[index], "Q6_K dequant value");
  }

  std::array<float, kBlockElements> vector{};
  std::iota(vector.begin(), vector.end(), -0.25F);
  const float expected_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(gufo::quant::DotProductQ6_K(block.data(), vector, vector.size()),
             expected_dot, "Q6_K dot product");
}

void TestQ3() {
  const std::array<std::int8_t, 16> scales{-32, -27, -18, -9, -1, 0,  1,  7,
                                           12,  18,  23,  27, 29, 30, 31, -16};
  const auto block = MakeQ3Block(scales);
  std::array<float, kBlockElements> actual{};
  gufo::quant::DequantizeQ3_K(block.data(), actual.data(), actual.size());

  std::array<float, kBlockElements> expected{};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const std::size_t half = index / 128;
    const std::size_t within_half = index % 128;
    const std::size_t scale_index =
        (half * 8) + ((within_half / 32) * 2) + ((within_half % 32) / 16);
    const auto quant =
        static_cast<std::int8_t>(static_cast<int>((index * 3U) % 8U) - 4);
    expected[index] =
        static_cast<float>(scales[scale_index]) * static_cast<float>(quant);
    ExpectNear(actual[index], expected[index], "Q3_K dequant value");
  }

  std::array<float, kBlockElements> vector{};
  std::iota(vector.begin(), vector.end(), 0.125F);
  const float expected_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(gufo::quant::DotProductQ3_K(block.data(), vector, vector.size()),
             expected_dot, "Q3_K dot product");
}

void TestQ8() {
  struct Q8Block {
    float d;
    std::int8_t qs[256];
    std::int16_t bsums[16];
  };
  Q8Block block{};
  block.d = 0.00390625F;
  for (std::size_t i = 0; i < kBlockElements; ++i) {
    block.qs[i] = static_cast<std::int8_t>(static_cast<int>(i % 65) - 32);
    block.bsums[i / 16] += block.qs[i];
  }

  std::array<float, kBlockElements> actual{};
  gufo::quant::DequantizeQ8_K(&block, actual.data(), actual.size());

  std::array<float, kBlockElements> expected{};
  for (std::size_t index = 0; index < kBlockElements; ++index) {
    expected[index] = block.d * static_cast<float>(block.qs[index]);
    ExpectNear(actual[index], expected[index], "Q8_K dequant value");
  }

  std::array<float, kBlockElements> vector{};
  std::iota(vector.begin(), vector.end(), -0.75F);
  const float expected_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(gufo::quant::DotProductQ8_K(&block, vector, vector.size()),
             expected_dot, "Q8_K dot product");
}

void TestQ8_0() {
  constexpr std::size_t kQ8_0Elements = 32;
  struct Q8_0Block {
    std::uint16_t d;
    std::int8_t qs[kQ8_0Elements];
  };
  static_assert(sizeof(Q8_0Block) == 34);
  Q8_0Block block{};
  block.d = 0x3800;  // fp16 0.5
  for (std::size_t i = 0; i < kQ8_0Elements; ++i) {
    block.qs[i] = static_cast<std::int8_t>(static_cast<int>(i % 65) - 32);
  }

  std::array<float, kQ8_0Elements> actual{};
  gufo::quant::DequantizeQ8_0(&block, actual.data(), actual.size());

  std::array<float, kQ8_0Elements> expected{};
  for (std::size_t index = 0; index < kQ8_0Elements; ++index) {
    expected[index] =
        gufo::quant::Fp16ToFloat(block.d) * static_cast<float>(block.qs[index]);
    ExpectNear(actual[index], expected[index], "Q8_0 dequant value");
  }

  std::array<float, kQ8_0Elements> vector{};
  std::iota(vector.begin(), vector.end(), -0.75F);
  const float expected_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(gufo::quant::DotProductQ8_0(&block, vector, vector.size()),
             expected_dot, "Q8_0 dot product");
}

void TestQ5_K() {
  constexpr std::size_t kQ5Elements = 256;
  struct Q5Block {
    std::uint16_t d;
    std::uint16_t dmin;
    std::uint8_t scales[12];
    std::uint8_t qh[32];
    std::uint8_t qs[128];
  };
  static_assert(sizeof(Q5Block) == 176);

  // QuantizedRowBytes covers every packed projection/embedding format used by
  // Qwen, including both Q8 layouts.
  if (gufo::quant::QuantizedRowBytes(gufo::core::GgmlType::kQ5_K,
                                     kQ5Elements) != 176 ||
      gufo::quant::QuantizedRowBytes(gufo::core::GgmlType::kQ8_K, 256) != 292 ||
      gufo::quant::QuantizedRowBytes(gufo::core::GgmlType::kQ8_0, 32) != 34 ||
      gufo::quant::EncodedSizeBytes(gufo::core::GgmlType::kF32, 8) != 32 ||
      gufo::quant::EncodedSizeBytes(gufo::core::GgmlType::kBF16, 8) != 16 ||
      gufo::quant::EncodedSizeBytes(gufo::core::GgmlType::kQ8_K, 256) != 292) {
    std::cerr << "Assertion failed: QuantizedRowBytes contract\n";
    std::exit(1);
  }

  // Build a handcrafted block. scales[12] is chosen so GetQ4ScaleMin(0..7)
  // yields (scale=4, min=1) for every index: packed[0..3]=4, packed[4..7]=1,
  // packed[8..11]=0x14 (high-half scale/min encodings) -> sc=4, m=1 all 8.
  const auto make_block = [](bool high_bits) {
    Q5Block block{};
    block.d = 0x3400;     // fp16 0.25
    block.dmin = 0x3000;  // fp16 0.125
    for (std::size_t index = 0; index < 4; ++index) {
      block.scales[index] = 4;
      block.scales[index + 4] = 1;
      block.scales[index + 8] = 0x14;
    }
    // qs[gg*32+lane] nibbles: group 0 low nibble=1, group 1 -> 2, etc.
    for (std::size_t gg = 0; gg < 4; ++gg) {
      const auto nibble = static_cast<std::uint8_t>(gg + 1);
      const auto val = static_cast<std::uint8_t>(nibble | (nibble << 4U));
      for (std::size_t lane = 0; lane < 32; ++lane) {
        block.qs[(gg * 32) + lane] = val;
      }
    }
    if (high_bits) {
      for (std::size_t lane = 0; lane < 32; ++lane) {
        block.qh[lane] = 0xFF;
      }
    }
    return block;
  };

  const auto check_block = [](const Q5Block& block, bool high_bits) {
    std::array<float, kQ5Elements> actual{};
    gufo::quant::DequantizeQ5_K(&block, actual.data(), actual.size());

    std::array<float, kQ5Elements> expected{};
    for (std::size_t index = 0; index < kQ5Elements; ++index) {
      const std::size_t gg = index / 64;
      const auto quant = static_cast<float>(gg + 1 + (high_bits ? 16 : 0));
      // Fp16ToFloat(0x3400)=0.25, Fp16ToFloat(0x3000)=0.125, sc=4, m=1.
      expected[index] = quant - 0.125F;
      ExpectNear(actual[index], expected[index],
                 high_bits ? "Q5_K dequant (high bits)" : "Q5_K dequant value");
    }

    std::array<float, kQ5Elements> unit{};
    unit.fill(1.0F);
    const float expected_dot =
        std::accumulate(expected.begin(), expected.end(), 0.0F);
    ExpectNear(gufo::quant::DotProductQ5_K(&block, unit, unit.size()),
               expected_dot, "Q5_K dot product");
  };

  check_block(make_block(false), false);  // no high bits
  check_block(make_block(true), true);    // all high bits set
}

struct ModelSample {
  std::size_t index;
  float value;
};

void TestModelRow(const gufo::core::GgufReader& reader, std::string_view name,
                  gufo::core::GgmlType expected_type, std::size_t expected_k,
                  std::span<const ModelSample> samples, double expected_dot) {
  const auto* tensor = reader.FindTensor(name);
  if (tensor == nullptr || tensor->data == nullptr ||
      tensor->dimensions.empty()) {
    std::cerr << "Assertion failed: missing model tensor " << name << '\n';
    std::exit(1);
  }
  if (tensor->type != expected_type ||
      tensor->dimensions.front() != expected_k) {
    std::cerr << "Assertion failed: model tensor contract " << name << '\n';
    std::exit(1);
  }

  std::vector<float> row(expected_k);
  switch (tensor->type) {
    case gufo::core::GgmlType::kQ4_K:
      gufo::quant::DequantizeQ4_K(tensor->data, row.data(), row.size());
      break;
    case gufo::core::GgmlType::kQ6_K:
      gufo::quant::DequantizeQ6_K(tensor->data, row.data(), row.size());
      break;
    default:
      std::cerr << "Assertion failed: unsupported model tensor type " << name
                << '\n';
      std::exit(1);
  }

  for (const auto& sample : samples) {
    ExpectNear(row[sample.index], sample.value, "real GGUF dequant value",
               1.0e-8F);
  }

  std::vector<float> input(expected_k);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] =
        static_cast<float>(static_cast<int>(index % 257) - 128) / 64.0F;
  }
  const float actual_dot =
      tensor->type == gufo::core::GgmlType::kQ4_K
          ? gufo::quant::DotProductQ4_K(tensor->data, input, input.size())
          : gufo::quant::DotProductQ6_K(tensor->data, input, input.size());
  ExpectNear(actual_dot, static_cast<float>(expected_dot),
             "real GGUF dot product", 5.0e-4F);
}

void TestModelRows(const std::filesystem::path& model_path) {
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (reader == nullptr) {
    std::cerr << "Assertion failed: MTP GGUF opens: " << error << '\n';
    std::exit(1);
  }

  constexpr std::array<ModelSample, 16> kQ4Samples{{
      {0, 0.008269786834716797F},
      {1, -0.019530773162841797F},
      {15, 0.023714542388916016F},
      {31, -0.01335287094116211F},
      {32, 0.009994983673095703F},
      {63, 0.005248546600341797F},
      {64, -0.002658843994140625F},
      {127, 0.013089179992675781F},
      {128, -0.006126880645751953F},
      {255, -0.0026617050170898438F},
      {256, -0.002628028392791748F},
      {511, 0.013987362384796143F},
      {1023, -0.0033451318740844727F},
      {2047, 0.0010213851928710938F},
      {4095, -0.00015413761138916016F},
      {5119, -0.0033974647521972656F},
  }};
  TestModelRow(*reader, "blk.64.nextn.eh_proj.weight",
               gufo::core::GgmlType::kQ4_K, 10240, kQ4Samples,
               1.1511284159496427);

  constexpr std::array<ModelSample, 16> kQ6Samples{{
      {0, -0.0027894973754882812F},
      {1, -0.013947486877441406F},
      {15, -0.024175643920898438F},
      {31, 0.009633064270019531F},
      {32, -0.003979682922363281F},
      {63, 0.008864402770996094F},
      {64, 0.0030994415283203125F},
      {127, -0.006855964660644531F},
      {128, -0.008070945739746094F},
      {255, 0.011157989501953125F},
      {256, -0.03636360168457031F},
      {511, 0.024633407592773438F},
      {1023, -0.0008414983749389648F},
      {2047, -0.012941122055053711F},
      {4095, 0.02767181396484375F},
      {5119, -0.012220144271850586F},
  }};
  TestModelRow(*reader, "blk.64.attn_q.weight", gufo::core::GgmlType::kQ6_K,
               5120, kQ6Samples, 0.5780322588980198);
}

/// Deterministic pseudo-random blocks of every block format, with sane F16
/// scales patched in; the expected FNV-1a hashes of the dequantized F32 bits
/// come from gguf-py's independent NumPy dequantizers
/// (gguf.quants.dequantize) over the same bytes, so this pins every format
/// to ggml's reference values bit for bit.
void TestDequantizeRowMatchesGgufPy() {
  struct Expected {
    gufo::core::GgmlType type;
    std::uint64_t hash;
  };
  constexpr std::array<Expected, 18> kExpected = {{
      {gufo::core::GgmlType::kQ4_0, 0xD281B0239A243459ULL},
      {gufo::core::GgmlType::kQ4_1, 0x6D7BD9EB30002C9FULL},
      {gufo::core::GgmlType::kQ5_0, 0x05A04BF224732323ULL},
      {gufo::core::GgmlType::kQ5_1, 0x4CA02E1925FA5604ULL},
      {gufo::core::GgmlType::kQ2_K, 0x4BA82C45B8873FA0ULL},
      {gufo::core::GgmlType::kQ3_K, 0xA428B48204D3D498ULL},
      {gufo::core::GgmlType::kQ4_K, 0xDF9F1270E319F0FDULL},
      {gufo::core::GgmlType::kQ5_K, 0xB76045F1545ED5EFULL},
      {gufo::core::GgmlType::kQ6_K, 0x182CD8E2D5F5C037ULL},
      {gufo::core::GgmlType::kIQ2_XXS, 0x29D0E4333BCAA5A7ULL},
      {gufo::core::GgmlType::kIQ2_XS, 0xF6326BD4300437A2ULL},
      {gufo::core::GgmlType::kIQ2_S, 0xEF001A0F5E65084BULL},
      {gufo::core::GgmlType::kIQ3_XXS, 0xDA348F74BC187DD8ULL},
      {gufo::core::GgmlType::kIQ3_S, 0x72ECFE3E950A55D8ULL},
      {gufo::core::GgmlType::kIQ1_S, 0x6249EDACD06371EDULL},
      {gufo::core::GgmlType::kIQ1_M, 0xB17975BE5B3D4FD8ULL},
      {gufo::core::GgmlType::kIQ4_NL, 0x5FB0BFED6A869927ULL},
      {gufo::core::GgmlType::kIQ4_XS, 0x1AC72565AD1320A1ULL},
  }};
  constexpr std::size_t kBlocks = 64;
  for (const auto& expected : kExpected) {
    const auto type = expected.type;
    const std::size_t elements = gufo::quant::QuantizedBlockElements(type);
    const std::size_t bytes = gufo::quant::QuantizedRowBytes(type, elements);
    if (elements == 0 || bytes == 0) {
      std::cerr << "unsized format " << gufo::core::ToString(type) << '\n';
      std::exit(1);
    }
    std::uint32_t state = 0x9E3779B9U ^ static_cast<std::uint32_t>(bytes);
    std::vector<std::uint8_t> blocks(kBlocks * bytes);
    for (auto& b : blocks) {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      b = static_cast<std::uint8_t>(state & 0xFFU);
    }
    const std::string_view name = gufo::core::ToString(type);
    for (std::size_t k = 0; k < kBlocks; ++k) {
      std::uint8_t* block = blocks.data() + k * bytes;
      const auto put = [&](std::size_t offset) {
        block[offset] = 0x66;  // F16 0x2E66, about 0.1
        block[offset + 1] = 0x2E;
      };
      if (name == "Q2_K") {
        put(80);
        put(82);
      } else if (name == "Q3_K") {
        put(108);
      } else if (name == "Q6_K") {
        put(208);
      } else if (name == "IQ1_M") {
        // The F16 scale is the top nibbles of the four scale words.
        constexpr std::uint8_t kNibbles[4] = {0x6, 0x6, 0xE, 0x2};
        for (std::size_t i = 0; i < 4; ++i) {
          std::uint8_t& high = block[48 + 2 * i + 1];
          high = static_cast<std::uint8_t>((high & 0x0FU) | (kNibbles[i] << 4));
        }
      } else {
        put(0);
        if (name == "Q4_1" || name == "Q5_1" || name == "Q4_K" ||
            name == "Q5_K") {
          put(2);
        }
      }
    }
    std::vector<float> values(kBlocks * elements);
    if (!gufo::quant::DequantizeRow(type, blocks.data(), values.data(),
                                    values.size())) {
      std::cerr << "DequantizeRow rejected " << name << '\n';
      std::exit(1);
    }
    std::uint64_t hash = 0xCBF29CE484222325ULL;
    for (const float v : values) {
      std::uint32_t bits = 0;
      std::memcpy(&bits, &v, sizeof(bits));
      for (int i = 0; i < 4; ++i) {
        hash ^= (bits >> (8 * i)) & 0xFFU;
        hash *= 0x100000001B3ULL;
      }
    }
    if (hash != expected.hash) {
      std::cerr << "dequantized " << name << " differs from gguf-py\n";
      std::exit(1);
    }
    if (gufo::quant::DequantizeRow(type, blocks.data(), values.data(),
                                   elements + 1)) {
      std::cerr << "DequantizeRow accepted a ragged " << name << " row\n";
      std::exit(1);
    }
  }
}

}  // namespace

int main() {
  TestQ4();
  TestQ6();
  TestQ3();
  TestQ8();
  TestQ8_0();
  TestQ5_K();
  TestDequantizeRowMatchesGgufPy();
  if (const char* model = std::getenv("GUFO_MTP_MODEL");
      model != nullptr && std::string_view(model).size() > 0) {
    TestModelRows(model);
  }
  return 0;
}
