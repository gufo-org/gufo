#ifndef GUFO_SERVER_GENERATION_PARSE_HPP_
#define GUFO_SERVER_GENERATION_PARSE_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/responses_logprobs.hpp"
#include "src/cli/serve/text_generation_backend.hpp"

struct GufoGeneration;

namespace gufo::server {

/// The call markup of one admitted request's tool dialect. The strings live
/// for the process: they are constants or a registered dialect's.
struct GenerationMarkup {
  /// Openers that start a call; nothing else does.
  std::span<const std::string_view> markers;
  /// Complete native call headers which end a constrained reasoning phase.
  std::span<const std::string_view> reasoning_markers;
  std::span<const std::string_view> closers;
  /// The registered parser of calls the host does not parse itself.
  const ext::ToolDialect* dialect{nullptr};
  /// The host parses calls opened by the Qwen marker.
  bool qwen_calls{false};
  /// A native grammar frames arguments before a value completes, so partial
  /// calls stream.
  bool native{false};
  /// Whitespace after an explicit end of reasoning that is framing.
  std::string_view reasoning_separator{"\r\n"};
};

/// The markup of the dialect the request admitted; the endpoints
/// (openai_chat.cpp) define it.
GenerationMarkup SelectGenerationMarkup(const std::optional<std::string>& id);

/// What one response is parsed with. `tools` must outlive the parser.
struct GenerationSetup {
  TextGenerationBackend::InitialOutputState initial{
      TextGenerationBackend::InitialOutputState::kContent};
  ext::ReasoningMarkup reasoning;
  GenerationMarkup markup;
  bool response_format{false};
  bool constrained_tools{false};
  ChatRequest::ToolChoice tool_choice{ChatRequest::ToolChoice::kAuto};
  std::span<const tokenization::ChatTool> tools;
  /// Final content reports its source spans.
  bool track_spans{false};
  /// Fed pieces are only buffered: tool deltas come from the final parse.
  bool buffered{false};
};

/// One event of the Rust parser (include/gufo/generation_parse.h); the views
/// are valid during the callback.
struct GenerationEvent {
  enum class Kind : std::uint8_t {
    kReasoningDelta,
    kContentDelta,
    kToolDelta,
    kFinalReasoning,
    kFinalContent,
    kFinalCall,
    kFinalEnd,
  };
  Kind kind{Kind::kFinalEnd};
  std::size_t index{0};
  std::string_view text;
  std::string_view name;
  std::string_view arguments;
  std::span<const TextSourceSpan> spans;
  bool begin{false};
  bool complete{false};
  bool hide_tool_markup{false};
};

/// Returns false to stop.
using GenerationEventFn = std::function<bool(const GenerationEvent&)>;

/// A streamed response. Feed and Finish return false when `on_event` asked
/// to stop, and throw TextGenerationError for an unmet tool_choice required,
/// std::runtime_error for a streamed call the final parse did not accept,
/// and std::bad_alloc or std::logic_error for internal failures; the stream
/// then takes no more input.
class GenerationStream {
public:
  explicit GenerationStream(const GenerationSetup& setup);
  ~GenerationStream();
  GenerationStream(const GenerationStream&) = delete;
  GenerationStream& operator=(const GenerationStream&) = delete;

  bool Feed(std::string_view bytes, const GenerationEventFn& on_event);
  bool Finish(bool enforce_required, bool interrupted,
              const GenerationEventFn& on_event);

  struct Host;

private:
  std::unique_ptr<Host> host_;
  GufoGeneration* handle_{nullptr};
};

struct GeneratedCall {
  std::string name;
  /// The arguments object as JSON, numbers as the model spelled them.
  std::string arguments;
};

struct GeneratedText {
  std::string text;
  std::string reasoning;
  std::vector<GeneratedCall> calls;
  bool hide_tool_markup{false};
  std::vector<TextSourceSpan> spans;
};

/// Parses a whole response, which may hold malformed UTF-8; throws as
/// GenerationStream does.
GeneratedText ParseGeneratedText(const GenerationSetup& setup,
                                 std::string_view bytes, bool enforce_required);

}  // namespace gufo::server

#endif  // GUFO_SERVER_GENERATION_PARSE_HPP_
