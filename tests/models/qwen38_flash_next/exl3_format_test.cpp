// EXL3 format contracts on the host, against values computed by independent
// references: Crucible's NumPy EXL3 tile reference (ref/exl3_ref.py, itself
// checked against ExLlamaV3) for the mul1 codebook and tile decoding, and
// ExLlamaV3's n-gram codec (exllamav3/modules/quant/exl3_lib/ngram_codec.py)
// for n-gram rows. Inputs are xorshift streams both sides generate alike.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/exl3/format.hpp"
#include "src/models/qwen38_flash_next/exl3/reference.hpp"

namespace fmt = gufo::models::qwen38_flash_next::exl3;

namespace {

int failures = 0;

void Expect(bool ok, const std::string& what) {
  if (!ok) {
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  }
}

std::vector<std::uint16_t> Xorshift(std::uint32_t seed, std::size_t n) {
  std::vector<std::uint16_t> out(n);
  std::uint32_t s = seed;
  for (auto& v : out) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    v = static_cast<std::uint16_t>(s & 0xFFFFU);
  }
  return out;
}

void Codebook() {
  const std::uint32_t states[] = {0, 1, 255, 0x1234, 0x8000, 0xFFFF};
  const std::uint16_t bits[] = {0xC2E8, 0x3921, 0x3AB3, 0xB89E, 0xBAF2, 0xB936};
  for (std::size_t i = 0; i < std::size(states); ++i) {
    Expect(fmt::FloatToHalf(fmt::Mul1Value(states[i])) == bits[i],
           "mul1 codebook state " + std::to_string(states[i]));
    Expect(fmt::Mul1Table()[states[i]] == fmt::Mul1Value(states[i]),
           "mul1 table state " + std::to_string(states[i]));
  }
}

void Tiles() {
  struct Case {
    int k;
    std::uint16_t picks[6];
    double sum;
  };
  const Case cases[] = {
      {2, {0xB711, 0x3C37, 0xB9EA, 0xB4AF, 0x3CD0, 0xBFFE}, -9.136428833007812},
      {3, {0x38CE, 0xBE81, 0x386D, 0x3958, 0xBC9C, 0xBCAA}, 23.392372131347656},
      {6, {0xBD26, 0xBE81, 0xB1E8, 0x3F2B, 0x3C07, 0xAC72}, 3.1268081665039062},
  };
  const int positions[] = {0, 1, 17, 100, 200, 255};
  for (const auto& c : cases) {
    const auto words = Xorshift(1000 + c.k, 16 * static_cast<std::size_t>(c.k));
    float tile[256];
    fmt::DecodeTile(words.data(), c.k, tile);
    for (int i = 0; i < 6; ++i)
      Expect(fmt::FloatToHalf(tile[positions[i]]) == c.picks[i],
             "tile K" + std::to_string(c.k) + " element " +
                 std::to_string(positions[i]));
    double sum = 0.0;
    for (const float v : tile)
      sum += v;
    Expect(std::fabs(sum - c.sum) < 1e-6,
           "tile K" + std::to_string(c.k) + " sum");
  }
}

void NgramRows() {
  struct Case {
    int k;
    double picks[6];
    double sum;
  };
  const Case cases[] = {
      {4,
       {-1.21661376953125, -0.84912109375, -1.29193115234375, -0.2122802734375,
        1.0660400390625, 0.980224609375},
       0.09775924682617188},
      {5,
       {-1.270751953125, -1.659423828125, -0.781005859375, 0.208984375,
        1.2436676025390625, 1.653564453125},
       -0.9999027252197266},
      {6,
       {-1.2487621307373047, -1.175628662109375, -1.231048583984375,
        0.1209716796875, 1.33843994140625, 1.3743896484375},
       4.537515640258789},
      {8,
       {-1.368896484375, -1.0081787109375, -1.0771484375, 0.178466796875,
        1.13201904296875, 1.73828125},
       -3.8205833435058594},
  };
  const int positions[] = {0, 1, 2, 79, 158, 159};
  std::vector<std::uint16_t> bias(fmt::kNgramRowDim);
  for (std::uint32_t i = 0; i < fmt::kNgramRowDim; ++i)
    bias[i] = fmt::FloatToHalf((static_cast<float>(i) - 80.0F) / 64.0F);
  for (const auto& c : cases) {
    auto row = Xorshift(2000 + c.k, fmt::NgramRowWords(c.k));
    row[0] = fmt::FloatToHalf(0.25F);
    std::vector<float> out(fmt::kNgramRowDim);
    fmt::DecodeNgramRow(row.data(), c.k, bias.data(), out.data());
    for (int i = 0; i < 6; ++i)
      Expect(std::fabs(out[positions[i]] - c.picks[i]) < 1e-6,
             "n-gram K" + std::to_string(c.k) + " element " +
                 std::to_string(positions[i]));
    double sum = 0.0;
    for (const float v : out)
      sum += v;
    Expect(std::fabs(sum - c.sum) < 1e-4,
           "n-gram K" + std::to_string(c.k) + " sum");
  }
}

void Hadamard() {
  // H is orthonormal and its own inverse.
  std::vector<float> v(256);
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = static_cast<float>(i % 7) - 3.0F;
  auto w = v;
  fmt::Hadamard128(w);
  fmt::Hadamard128(w);
  double err = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i)
    err = std::max(err, static_cast<double>(std::fabs(w[i] - v[i])));
  Expect(err < 1e-5, "Hadamard involution");
}

}  // namespace

int main() {
  Codebook();
  Tiles();
  NgramRows();
  Hadamard();
  if (failures != 0) {
    std::fprintf(stderr, "%d failures\n", failures);
    return 1;
  }
  std::printf("exl3 format: ok\n");
  return 0;
}
