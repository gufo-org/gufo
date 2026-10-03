#ifndef GUFO_SERVER_QUOTE_CONTEXT_HPP_
#define GUFO_SERVER_QUOTE_CONTEXT_HPP_

#include <cstddef>
#include <string>
#include <string_view>

namespace gufo::server {

/// Quoted regions of model output: fenced blocks and inline-code spans.
///
/// The parser reads quoted output to tell framing from prose: a tag the model
/// writes inside a backtick span or a code fence is the model *naming* the
/// syntax, not writing it (#383).
///
/// The scanner is incremental so a streaming consumer can feed output as it
/// arrives. Only completed lines toggle a fence, and inline-code parity is
/// counted within the current line, exactly as the parser reads a line prefix.
class QuoteScanner {
public:
  void Push(std::string_view text) {
    for (const char byte : text) {
      PushByte(byte);
    }
  }

  /// True when the current output position sits inside a quoted region.
  [[nodiscard]] bool in_quote() const noexcept {
    return fenced_ || inline_code();
  }

private:
  /// Odd number of backticks on the current line: an open inline-code span.
  [[nodiscard]] bool inline_code() const noexcept {
    return line_backticks_ % 2 != 0;
  }

  void PushByte(char byte) {
    if (byte == '\n') {
      const auto first = line_.find_first_not_of(" \t\r");
      if (first != std::string::npos) {
        const std::string_view trimmed(line_.data() + first,
                                       line_.size() - first);
        if (trimmed.starts_with("```") || trimmed.starts_with("~~~")) {
          fenced_ = !fenced_;
        }
      }
      line_.clear();
      line_backticks_ = 0;
      return;
    }
    if (byte == '`') {
      ++line_backticks_;
    }
    line_.push_back(byte);
  }

  std::string line_;
  std::size_t line_backticks_{0};
  bool fenced_{false};
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_QUOTE_CONTEXT_HPP_
