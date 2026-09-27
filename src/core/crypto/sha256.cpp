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

std::string DigestHex(const std::array<std::uint8_t, 32>& digest) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const std::uint8_t byte : digest) {
    output << std::setw(2) << static_cast<unsigned int>(byte);
  }
  return output.str();
}

}  // namespace

Sha256Hasher::Sha256Hasher() {
  if (!context_ ||
      EVP_DigestInit_ex(context_.get(), EVP_sha256(), nullptr) != 1) {
    throw std::runtime_error("cannot initialize SHA-256");
  }
}

void Sha256Hasher::Update(std::span<const std::uint8_t> bytes) {
  if (EVP_DigestUpdate(context_.get(), bytes.data(), bytes.size()) != 1) {
    throw std::runtime_error("cannot update SHA-256");
  }
}

std::array<std::uint8_t, 32> Sha256Hasher::Finish() {
  std::array<std::uint8_t, 32> result{};
  unsigned int size = 0;
  if (EVP_DigestFinal_ex(context_.get(), result.data(), &size) != 1 ||
      size != result.size()) {
    throw std::runtime_error("cannot finish SHA-256");
  }
  return result;
}

std::array<std::uint8_t, 32> Sha256Hasher::Digest() const {
  Sha256Hasher copy;
  if (EVP_MD_CTX_copy_ex(copy.context_.get(), context_.get()) != 1)
    throw std::runtime_error("cannot copy SHA-256 state");
  return copy.Finish();
}

std::string Sha256Hasher::FinishHex() {
  return DigestHex(Finish());
}

std::string Sha256Hex(std::span<const std::uint8_t> bytes) {
  Sha256Hasher hasher;
  hasher.Update(bytes);
  return hasher.FinishHex();
}

std::string Sha256FileHex(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("cannot open file for SHA-256");
  }
  Sha256Hasher hasher;
  std::vector<std::uint8_t> buffer(std::size_t{8} * 1024U * 1024U);
  while (input) {
    input.read(reinterpret_cast<char*>(buffer.data()),
               static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0) {
      hasher.Update(std::span(buffer.data(), static_cast<std::size_t>(count)));
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("cannot read file for SHA-256");
  }

  return hasher.FinishHex();
}

}  // namespace gufo::crypto
