#ifndef GUFO_CORE_CRYPTO_SHA256_HPP_
#define GUFO_CORE_CRYPTO_SHA256_HPP_

#include <openssl/evp.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

namespace gufo::crypto {

/// Incremental SHA-256 over a byte stream.
class Sha256Hasher {
public:
  Sha256Hasher();
  void Update(std::span<const std::uint8_t> bytes);
  /// Digest the current prefix without consuming the incremental state.
  [[nodiscard]] std::array<std::uint8_t, 32> Digest() const;
  [[nodiscard]] std::array<std::uint8_t, 32> Finish();
  /// Finishes the digest and renders it as 64 lowercase hex characters.
  [[nodiscard]] std::string FinishHex();

private:
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context_{
      EVP_MD_CTX_new(), EVP_MD_CTX_free};
};

[[nodiscard]] std::string Sha256Hex(std::span<const std::uint8_t> bytes);
[[nodiscard]] std::string Sha256FileHex(const std::filesystem::path& path);

}  // namespace gufo::crypto

#endif  // GUFO_CORE_CRYPTO_SHA256_HPP_
