#ifndef GUFO_CORE_JSON_SCHEMA_REGEX_HPP_
#define GUFO_CORE_JSON_SCHEMA_REGEX_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace gufo::sampling {

// An immutable Unicode DFA for the intersection of string predicates.
// Every admitted prefix has an accepting continuation within the length bounds.
class JsonSchemaRegex {
public:
  static std::shared_ptr<const JsonSchemaRegex> Compile(
      std::span<const std::string> patterns, std::uint32_t maximum);
  [[nodiscard]] std::uint32_t Start() const { return 0; }
  [[nodiscard]] bool Accepting(std::uint32_t state) const;
  [[nodiscard]] std::uint32_t MaximumSuffix() const;
  [[nodiscard]] std::uint32_t Advance(std::uint32_t state,
                                      std::uint32_t codepoint) const;
  [[nodiscard]] bool CanFinish(std::uint32_t state, std::uint32_t minimum,
                               std::uint32_t maximum) const;
  [[nodiscard]] bool CanAdvance(std::uint32_t state, std::uint32_t first,
                                std::uint32_t last, std::uint32_t minimum,
                                std::uint32_t maximum) const;
  static constexpr std::uint32_t kDead = UINT32_MAX;

private:
  struct Impl;
  explicit JsonSchemaRegex(std::shared_ptr<Impl> impl);
  std::shared_ptr<const Impl> impl_;
};

}  // namespace gufo::sampling
#endif
