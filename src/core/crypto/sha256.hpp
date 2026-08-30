#ifndef GUFO_CORE_CRYPTO_SHA256_HPP_
#define GUFO_CORE_CRYPTO_SHA256_HPP_

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>

namespace gufo::crypto {

[[nodiscard]] std::string Sha256Hex(std::span<const std::uint8_t> bytes);
[[nodiscard]] std::string Sha256FileHex(const std::filesystem::path& path);

}  // namespace gufo::crypto

#endif  // GUFO_CORE_CRYPTO_SHA256_HPP_
