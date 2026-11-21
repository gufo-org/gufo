#pragma once

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

/// Thought framing as the output dialect defines it: the opener and closer
/// tokens. When `turn_end` is non-empty the model has no explicit closer, so
/// a thought runs to that next-turn control token, which ends the turn.
struct ThoughtMarkers {
  std::string_view open{"<think>"};
  std::string_view close{"</think>"};
  std::string_view turn_end{};
};

// Interim, by design: the request-level chat and tool contract is still the
// serve-owned `ChatRequest`, and the tool dialect is still
// `sampling::JsonConstraint::ToolFormat`. RFC step 10 moves the runner header;
// step 7 publishes both under `include/gufo/ext/`.

}  // namespace gufo::ext
