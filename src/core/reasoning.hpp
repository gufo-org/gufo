#ifndef GUFO_CORE_REASONING_HPP_
#define GUFO_CORE_REASONING_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace gufo {

/// Provider-neutral reasoning levels accepted by the HTTP and CLI surfaces.
///
/// Model formatters map these levels onto their native, smaller effort sets.
enum class ReasoningEffort : std::uint8_t {
  kMinimal,
  kLow,
  kMedium,
  kHigh,
  kXHigh,
  kMax,
};

struct ReasoningOptions {
  std::optional<bool> enabled;
  std::optional<ReasoningEffort> effort;
  std::optional<bool> preserve_thinking;
  /// Reasoning tokens before the end of thinking is forced (#266).
  std::optional<std::size_t> budget_tokens;
  /// Text forced before the reasoning end once the budget is spent (#693).
  std::optional<std::string> budget_message;
};

}  // namespace gufo

#endif  // GUFO_CORE_REASONING_HPP_
