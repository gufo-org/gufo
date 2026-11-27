#ifndef GUFO_CORE_PROMPT_TOO_LONG_HPP_
#define GUFO_CORE_PROMPT_TOO_LONG_HPP_

#include <cstddef>
#include <stdexcept>
#include <string>

namespace gufo::core {

/// A prompt that does not fit the context. `tokens` is the prompt length, or
/// a lower bound when tokenization stopped at the limit; `maximum` is the
/// longest prompt the context accepts. Surfaces that name the sizes, such as
/// Anthropic's "prompt is too long: N tokens > M maximum", read them here;
/// every other caller sees an ordinary length_error.
class PromptTooLong : public std::length_error {
public:
  PromptTooLong(std::size_t tokens, std::size_t maximum,
                const std::string& message)
      : std::length_error(message), tokens_(tokens), maximum_(maximum) {}

  [[nodiscard]] std::size_t tokens() const noexcept { return tokens_; }
  [[nodiscard]] std::size_t maximum() const noexcept { return maximum_; }

private:
  std::size_t tokens_;
  std::size_t maximum_;
};

}  // namespace gufo::core

#endif  // GUFO_CORE_PROMPT_TOO_LONG_HPP_
