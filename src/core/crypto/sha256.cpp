#include "src/core/crypto/sha256.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace gufo::crypto {
namespace {

constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428A2F98U, 0x71374491U, 0xB5C0FBCFU, 0xE9B5DBA5U, 0x3956C25BU,
    0x59F111F1U, 0x923F82A4U, 0xAB1C5ED5U, 0xD807AA98U, 0x12835B01U,
    0x243185BEU, 0x550C7DC3U, 0x72BE5D74U, 0x80DEB1FEU, 0x9BDC06A7U,
    0xC19BF174U, 0xE49B69C1U, 0xEFBE4786U, 0x0FC19DC6U, 0x240CA1CCU,
    0x2DE92C6FU, 0x4A7484AAU, 0x5CB0A9DCU, 0x76F988DAU, 0x983E5152U,
    0xA831C66DU, 0xB00327C8U, 0xBF597FC7U, 0xC6E00BF3U, 0xD5A79147U,
    0x06CA6351U, 0x14292967U, 0x27B70A85U, 0x2E1B2138U, 0x4D2C6DFCU,
    0x53380D13U, 0x650A7354U, 0x766A0ABBU, 0x81C2C92EU, 0x92722C85U,
    0xA2BFE8A1U, 0xA81A664BU, 0xC24B8B70U, 0xC76C51A3U, 0xD192E819U,
    0xD6990624U, 0xF40E3585U, 0x106AA070U, 0x19A4C116U, 0x1E376C08U,
    0x2748774CU, 0x34B0BCB5U, 0x391C0CB3U, 0x4ED8AA4AU, 0x5B9CCA4FU,
    0x682E6FF3U, 0x748F82EEU, 0x78A5636FU, 0x84C87814U, 0x8CC70208U,
    0x90BEFFFAU, 0xA4506CEBU, 0xBEF9A3F7U, 0xC67178F2U};

constexpr std::uint32_t RotateRight(std::uint32_t value,
                                    unsigned int bits) noexcept {
  return (value >> bits) | (value << (32U - bits));
}

class Sha256Context {
public:
  void Update(std::span<const std::uint8_t> bytes) noexcept {
    for (const std::uint8_t byte : bytes) {
      block_[block_size_++] = byte;
      ++total_bytes_;
      if (block_size_ == block_.size()) {
        Transform();
        block_size_ = 0;
      }
    }
  }

  [[nodiscard]] std::array<std::uint8_t, 32> Finish() noexcept {
    const std::uint64_t total_bits = total_bytes_ * 8U;
    block_[block_size_++] = 0x80U;
    if (block_size_ > 56) {
      while (block_size_ < block_.size()) {
        block_[block_size_++] = 0;
      }
      Transform();
      block_size_ = 0;
    }
    while (block_size_ < 56) {
      block_[block_size_++] = 0;
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
      block_[block_size_++] =
          static_cast<std::uint8_t>((total_bits >> shift) & 0xFFU);
    }
    Transform();

    std::array<std::uint8_t, 32> result{};
    for (std::size_t index = 0; index < state_.size(); ++index) {
      result[index * 4] = static_cast<std::uint8_t>(state_[index] >> 24U);
      result[index * 4 + 1] = static_cast<std::uint8_t>(state_[index] >> 16U);
      result[index * 4 + 2] = static_cast<std::uint8_t>(state_[index] >> 8U);
      result[index * 4 + 3] = static_cast<std::uint8_t>(state_[index]);
    }
    return result;
  }

private:
  void Transform() noexcept {
    std::array<std::uint32_t, 64> schedule{};
    for (std::size_t index = 0; index < 16; ++index) {
      schedule[index] =
          (static_cast<std::uint32_t>(block_[index * 4]) << 24U) |
          (static_cast<std::uint32_t>(block_[index * 4 + 1]) << 16U) |
          (static_cast<std::uint32_t>(block_[index * 4 + 2]) << 8U) |
          static_cast<std::uint32_t>(block_[index * 4 + 3]);
    }
    for (std::size_t index = 16; index < schedule.size(); ++index) {
      const std::uint32_t sigma_0 = RotateRight(schedule[index - 15], 7) ^
                                    RotateRight(schedule[index - 15], 18) ^
                                    (schedule[index - 15] >> 3U);
      const std::uint32_t sigma_1 = RotateRight(schedule[index - 2], 17) ^
                                    RotateRight(schedule[index - 2], 19) ^
                                    (schedule[index - 2] >> 10U);
      schedule[index] =
          schedule[index - 16] + sigma_0 + schedule[index - 7] + sigma_1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t index = 0; index < schedule.size(); ++index) {
      const std::uint32_t sigma_1 =
          RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
      const std::uint32_t choice = (e & f) ^ ((~e) & g);
      const std::uint32_t temporary_1 =
          h + sigma_1 + choice + kRoundConstants[index] + schedule[index];
      const std::uint32_t sigma_0 =
          RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary_2 = sigma_0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary_1;
      d = c;
      c = b;
      b = a;
      a = temporary_1 + temporary_2;
    }

    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_ = {0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U,
                                         0xA54FF53AU, 0x510E527FU, 0x9B05688CU,
                                         0x1F83D9ABU, 0x5BE0CD19U};
  std::array<std::uint8_t, 64> block_{};
  std::size_t block_size_{0};
  std::uint64_t total_bytes_{0};
};

}  // namespace

std::string Sha256Hex(std::span<const std::uint8_t> bytes) {
  Sha256Context context;
  context.Update(bytes);
  const auto digest = context.Finish();

  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const std::uint8_t byte : digest) {
    output << std::setw(2) << static_cast<unsigned int>(byte);
  }
  return output.str();
}

std::string Sha256FileHex(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("cannot open file for SHA-256");
  }
  Sha256Context context;
  std::vector<std::uint8_t> buffer(std::size_t{8} * 1024U * 1024U);
  while (input) {
    input.read(reinterpret_cast<char*>(buffer.data()),
               static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0) {
      context.Update(std::span(buffer.data(), static_cast<std::size_t>(count)));
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("cannot read file for SHA-256");
  }

  const auto digest = context.Finish();
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const std::uint8_t byte : digest) {
    output << std::setw(2) << static_cast<unsigned int>(byte);
  }
  return output.str();
}

}  // namespace gufo::crypto
