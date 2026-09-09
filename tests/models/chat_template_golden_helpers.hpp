#pragma once
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/json.hpp"
#include "src/core/crypto/sha256.hpp"
namespace gufo::testing::chat_goldens {
namespace json = gufo::server::json;
inline void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

inline const json::Value& Required(const json::Value& object,
                                   std::string_view key) {
  const json::Value* value = object.find(std::string(key));
  Expect(value != nullptr, "Golden fixture is missing a required field");
  return *value;
}

inline const json::Value& GoldenCase(const json::Value& fixture,
                                     std::string_view model,
                                     std::string_view case_name) {
  return Required(Required(Required(fixture, model), "cases"), case_name);
}

inline std::string Sha256(std::string_view value) {
  const auto* data = reinterpret_cast<const std::uint8_t*>(value.data());
  return gufo::crypto::Sha256Hex({data, value.size()});
}

template<typename Token>
std::string TokenSha256(std::span<const Token> tokens) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(tokens.size() * sizeof(std::uint32_t));
  for (const Token token : tokens) {
    const auto value = static_cast<std::uint32_t>(token);
    bytes.push_back(static_cast<std::uint8_t>(value));
    bytes.push_back(static_cast<std::uint8_t>(value >> 8U));
    bytes.push_back(static_cast<std::uint8_t>(value >> 16U));
    bytes.push_back(static_cast<std::uint8_t>(value >> 24U));
  }
  return gufo::crypto::Sha256Hex(bytes);
}

}  // namespace gufo::testing::chat_goldens
