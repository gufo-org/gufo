#pragma once

// The C++ generation parser as it was before its Rust port (#1352), behind
// the events the Rust parser reports. generation_parse_test compares the two.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/responses_logprobs.hpp"
#include "src/cli/serve/text_generation_backend.hpp"

namespace gufo::server::frozen {

enum class EventKind : std::uint8_t {
  kReasoningDelta,
  kContentDelta,
  kToolDelta,
  kFinalReasoning,
  kFinalContent,
  kFinalCall,
  kFinalEnd,
};

struct Event {
  EventKind kind{EventKind::kFinalEnd};
  std::string text;
  std::string name;
  std::string arguments;
  std::vector<TextSourceSpan> spans;
  std::size_t index{0};
  bool begin{false};
  bool complete{false};
  bool hide_tool_markup{false};
};

/// How a parse ended: completely, with tool_choice required unmet (no call
/// produced, or a constrained call not completed), or with a streamed call
/// the final parse did not accept.
enum class Outcome : std::uint8_t {
  kComplete,
  kNotProduced,
  kNotCompleted,
  kUnsafeStream,
};

/// The request as the parsers read it.
struct Setup {
  std::optional<std::string> dialect;
  ext::ReasoningMarkup reasoning;
  TextGenerationBackend::InitialOutputState initial{
      TextGenerationBackend::InitialOutputState::kContent};
  bool response_format{false};
  bool constrained_tools{false};
  ChatRequest::ToolChoice choice{ChatRequest::ToolChoice::kAuto};
  std::vector<tokenization::ChatTool> tools;
  bool track_spans{false};
  /// Fed pieces are only buffered: tool deltas come from the final parse.
  bool buffered{false};
};

/// Streams `pieces` as the Chat, Responses and Messages handlers did, then
/// finishes with `enforce_required` = !interrupted. With `piece_ns`, records
/// the nanoseconds each piece took.
Outcome Stream(const Setup& setup, const std::vector<std::string_view>& pieces,
               bool interrupted, std::vector<Event>* events,
               std::vector<std::uint64_t>* piece_ns = nullptr);

/// Parses a whole response as the buffered Chat handler did.
Outcome Parse(const Setup& setup, std::string_view bytes, bool enforce_required,
              std::vector<Event>* events);

}  // namespace gufo::server::frozen
