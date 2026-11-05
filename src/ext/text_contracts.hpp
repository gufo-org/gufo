#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace gufo::ext {

/// Reasoning tags for a token budget, as the model's chat format defines them:
/// `forced_end` is emitted once the budget is spent and any of `ends` closes
/// reasoning early.
struct ReasoningTags {
  std::string_view start;
  std::string_view forced_end;
  std::vector<std::string_view> ends;
};

/// Reasoning delimiters of the provider's generated text, as the response
/// parsers and the usage accounting read them. Budget enforcement works on
/// tokens and keeps `ReasoningTags`: its start must be one control token, while
/// `open` is the complete literal prefix the model emits, which may span tokens
/// and end in whitespace (for example `<|think|>\n`). It must not begin with
/// whitespace, because parsers skip whitespace before it. Reasoning that the
/// prompt already opened (`InitialOutputState::kThought`) may still repeat
/// `open`; the parsers drop it. `open` may be empty when the provider has no
/// opener. `close` must be non-empty.
struct ReasoningMarkup {
  std::string open;
  std::string close;

  friend bool operator==(const ReasoningMarkup&,
                         const ReasoningMarkup&) = default;
};

// Interim, by design: the request-level chat and tool contract is still the
// serve-owned `ChatRequest`, and the built-in tool formats are still
// `sampling::JsonConstraint::ToolFormat` (provider dialects register a
// `ToolDialect` by identity). RFC step 10 moves the runner header; step 7
// publishes both under `include/gufo/ext/`.

}  // namespace gufo::ext
