#ifndef GUFO_CORE_REASONING_HPP_
#define GUFO_CORE_REASONING_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

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
};

/// Chat-template capabilities, reported on `/props` as `chat_template_caps`
/// with the same fields as llama.cpp's `jinja::caps`. Engines declare what
/// their renderer actually consumes: a level is a `reasoning_effort` name the
/// template renders distinctly from its fallback (aliases collapse onto the
/// highest name, ascending order); `reasoning_effort_rejected` lists names
/// the renderer errors on. Defaults match llama.cpp: a plain ChatML-style
/// template keeps the basic booleans and advertises no reasoning features.
struct ChatTemplateCaps {
  bool supports_string_content{true};
  bool supports_typed_content{false};
  bool supports_tools{true};
  bool supports_tool_calls{true};
  bool supports_parallel_tool_calls{true};
  bool supports_system_role{true};
  bool supports_preserve_reasoning{false};
  bool supports_reasoning_effort{false};
  bool supports_object_arguments{false};
  std::vector<std::string> reasoning_effort_levels;
  std::vector<std::string> reasoning_effort_rejected;
};

}  // namespace gufo

#endif  // GUFO_CORE_REASONING_HPP_
