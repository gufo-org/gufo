// FROZEN: a verbatim copy of the C++ generation parser before its Rust
// port (#1352), taken from c887222c: src/cli/serve/quote_tracker.hpp and
// src/cli/serve/openai_chat.cpp lines 75-226, 1098-2218 and 2354-2791.
// generation_parse_test compares the Rust parser against it event by
// event. Do not edit it except to keep it compiling; replace both with
// expected-result tests once the port has shipped (src/core/rust/AGENTS.md).
// clang-format off
#include "tests/cli/generation_parse_frozen.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <utility>

#include "src/core/json.hpp"
#include "src/core/json_constraint.hpp"
#include "src/core/utf8.hpp"
#include "src/families/qwen/text/chat_template.hpp"

namespace gufo::server::frozen {
namespace {


/// Request-owned quoting state. Completed inline spans across nonblank lines
/// and completed fences protect documentation. Markers in unfinished spans are
/// held during streaming until a closing delimiter or the final parse decides.
/// Own the scanned bytes rather than identifying a buffer by its address.
class QuoteTracker {
public:
  void Reset(std::string_view text = {}, std::size_t origin = 0) {
    origin = std::min(origin, text.size());
    text_.assign(origin, ' ');
    spans_.clear();
    unfinished_inline_.clear();
    first_nonblank_ = std::string_view::npos;
    run_begin_ = 0;
    run_size_ = 0;
    run_char_ = 0;
    inline_size_ = 0;
    fence_size_ = 0;
    fence_char_ = 0;
    closing_fence_ = false;
    Append(text.substr(origin));
  }

  void Append(std::string_view piece) {
    const auto begin = text_.size();
    text_.append(piece);
    for (auto cursor = begin; cursor < text_.size(); ++cursor) {
      Read(cursor);
    }
    bytes_scanned_ += piece.size();
  }

  [[nodiscard]] bool QuotedAt(std::size_t position) const {
    const auto next = std::upper_bound(
        spans_.begin(), spans_.end(), position,
        [](auto at, const Span& span) { return at < span.begin; });
    if (next != spans_.begin()) {
      const auto& span = *std::prev(next);
      if (position < span.end)
        return true;
    }
    // A closing delimiter can be the final bytes, without a following newline
    // or ordinary character to flush the pending run. This is provisional:
    // appending more text can make the last line cease to be a closing fence.
    if (fence_size_ > 0) {
      if (closing_fence_ && position >= fence_begin_ && position < close_begin_)
        return true;
      if (run_size_ >= fence_size_ && run_char_ == fence_char_ &&
          AtLineStart(run_begin_) && position >= fence_begin_ &&
          position < run_begin_)
        return true;
    } else if (inline_size_ > 0 && run_char_ == '`' &&
               run_size_ == inline_size_ && position >= inline_begin_ &&
               position < run_begin_) {
      return true;
    }
    return false;
  }

  [[nodiscard]] bool UnclosedAt(std::size_t position) const {
    if (QuotedAt(position))
      return false;
    if ((fence_size_ > 0 && position >= fence_begin_) ||
        (inline_size_ > 0 && position >= inline_begin_))
      return true;
    const auto next = std::upper_bound(
        unfinished_inline_.begin(), unfinished_inline_.end(), position,
        [](auto at, const Span& span) { return at < span.begin; });
    return next != unfinished_inline_.begin() &&
           position < std::prev(next)->end;
  }

  [[nodiscard]] std::size_t bytes_scanned() const noexcept {
    return bytes_scanned_;
  }

private:
  struct Span {
    std::size_t begin;
    std::size_t end;
  };

  [[nodiscard]] bool AtLineStart(std::size_t position) const {
    return first_nonblank_ == position;
  }

  void FlushRun() {
    if (run_size_ == 0)
      return;
    if (fence_size_ > 0) {
      if (run_char_ == fence_char_ && run_size_ >= fence_size_ &&
          AtLineStart(run_begin_)) {
        closing_fence_ = true;
        close_begin_ = run_begin_;
      }
    } else if (inline_size_ == 0 && run_size_ >= 3 && AtLineStart(run_begin_)) {
      fence_char_ = run_char_;
      fence_size_ = run_size_;
      fence_begin_ = run_begin_ + run_size_;
    } else if (run_char_ == '`') {
      if (inline_size_ == 0) {
        inline_size_ = run_size_;
        inline_begin_ = run_begin_ + run_size_;
      } else if (run_size_ == inline_size_) {
        spans_.push_back({inline_begin_, run_begin_});
        inline_size_ = 0;
      }
    }
    run_size_ = 0;
  }

  void Read(std::size_t cursor) {
    const auto byte = text_[cursor];
    if (first_nonblank_ == std::string_view::npos && byte != ' ' &&
        byte != '\t' && byte != '\r' && byte != '\n') {
      first_nonblank_ = cursor;
    }
    if (byte == '`' || byte == '~') {
      if (run_size_ != 0 && run_char_ != byte)
        FlushRun();
      if (run_size_ == 0) {
        run_begin_ = cursor;
        run_char_ = byte;
      }
      ++run_size_;
      return;
    }
    FlushRun();
    if (byte == '\n') {
      if (fence_size_ > 0 && closing_fence_) {
        spans_.push_back({fence_begin_, close_begin_});
        fence_size_ = 0;
        fence_char_ = 0;
      }
      if (inline_size_ > 0 && first_nonblank_ == std::string_view::npos) {
        // A blank line ends an unfinished inline span. Retain its range so
        // possible calls still require the schema-checked fallback.
        unfinished_inline_.push_back({inline_begin_, cursor});
        inline_size_ = 0;
      }
      closing_fence_ = false;
      first_nonblank_ = std::string_view::npos;
    } else if (byte != ' ' && byte != '\t' && byte != '\r') {
      closing_fence_ = false;
    }
  }

  std::string text_;
  std::vector<Span> spans_;
  std::vector<Span> unfinished_inline_;
  std::size_t bytes_scanned_{0};
  std::size_t first_nonblank_{std::string_view::npos};
  std::size_t run_begin_{0};
  std::size_t run_size_{0};
  char run_char_{0};
  std::size_t inline_begin_{0};
  std::size_t inline_size_{0};
  std::size_t fence_begin_{0};
  std::size_t fence_size_{0};
  char fence_char_{0};
  std::size_t close_begin_{0};
  bool closing_fence_{false};
};

std::string RandomId(std::string_view prefix) {
  return std::string(prefix) + "frozen";
}

struct ParsedToolCall {
  std::string id;
  std::string name;
  std::vector<tokenization::ChatMessage::ToolArgument> arguments;
  // An append-only JSON prefix; absent on complete calls.
  std::optional<std::string> partial_arguments;
};

struct ParsedGeneration {
  std::string text;
  std::string reasoning_content;
  std::vector<ParsedToolCall> tool_calls;
  bool hide_tool_markup{false};
  std::vector<TextSourceSpan> text_spans;
};

void AppendTextSpans(std::vector<TextSourceSpan>& output,
                     std::span<const TextSourceSpan> source, std::size_t offset,
                     std::size_t size) {
  for (const auto& span : source) {
    if (offset >= span.size) {
      offset -= span.size;
      continue;
    }
    const auto count = std::min(size, span.size - offset);
    if (count)
      output.push_back({span.begin + offset, count});
    size -= count;
    if (!size)
      break;
    offset = 0;
  }
}

void SetParsedText(ParsedGeneration& parsed, std::string_view text,
                   std::string_view source, bool track) {
  parsed.text = std::string(text);
  parsed.text_spans.clear();
  if (track && !text.empty())
    parsed.text_spans.push_back(
        {static_cast<std::size_t>(text.data() - source.data()), text.size()});
}

void RebaseTextSpans(ParsedGeneration& parsed, std::size_t offset) {
  for (auto& span : parsed.text_spans)
    span.begin += offset;
}

constexpr std::array<std::string_view, 7> kToolMarkers{
    tokenization::QwenToolSyntax::kCallOpen,
    "<｜DSML｜tool_calls｜>",
    "<｜DSML｜tool_calls>",
    "<DSML｜tool_calls｜>",
    "<DSML｜tool_calls>",
    "<tool_calls｜>",
    "<tool_calls>",
};
using ReasoningMarkup = ext::ReasoningMarkup;
using ToolMarkerSet = std::span<const std::string_view>;
using ToolCloserSet = std::span<const std::string_view>;

// Closers eligible for removal immediately after an accepted call. Standalone
// closers and XML remain literal; the declared-tool envelope rule is separate.
constexpr std::array<std::string_view, 4> kQwenClosers{
    tokenization::QwenToolSyntax::kParameterClose,
    tokenization::QwenToolSyntax::kFunctionClose,
    tokenization::QwenToolSyntax::kCallClose,
    // The client's envelope, whose block rules already strip it whole.
    "</invoke>",
};
constexpr std::array<std::string_view, 9> kToolClosers{
    // Qwen family.
    tokenization::QwenToolSyntax::kParameterClose,
    tokenization::QwenToolSyntax::kFunctionClose,
    tokenization::QwenToolSyntax::kCallClose,
    // Envelope dialects the clients and harness prompts render; Qwen models
    // echo these back verbatim, so they are framing here too.
    "</invoke>",
    "</function_results>",
    "</tool_calls>",
    // Model-native dialects, which this fallback admits like any other opener.
    "</｜DSML｜parameter>",
    "</｜DSML｜invoke>",
    "</｜DSML｜tool_calls>",
};

// The call markup of one admitted request's tool dialect.
struct ToolMarkup {
  // Openers that start a call; nothing else does.
  ToolMarkerSet markers;
  // Complete native call headers which end a constrained reasoning phase.
  ToolMarkerSet reasoning_markers;
  ToolCloserSet closers;
  // The registered parser of calls the host does not parse itself.
  const ext::ToolDialect* dialect{nullptr};
  // The host parses calls opened by the Qwen marker.
  bool qwen_calls{false};
  // A native grammar frames arguments before a value completes, so partial
  // calls stream. Recovery from JSON fallback text needs a full call.
  bool native{false};
  // Whitespace after an explicit end of reasoning that is framing.
  std::string_view reasoning_separator{"\r\n"};
};

// The markup of the dialect the request admitted. Other dialects' markers and
// closers are ordinary prose under its grammar, not additional calls.
ToolMarkup SelectToolMarkup(const std::optional<std::string>& id) {
  using Format = sampling::JsonConstraint::ToolFormat;
  // The JSON fallback has no native counterpart in llama.cpp. Its grammar
  // leaves text before <tool_call> unconstrained, where models still write
  // their native call syntax; keep recognizing every opener there.
  if (!id || *id == ext::BuiltinToolDialectId(Format::kJson))
    return {
        .markers = kToolMarkers,
        // Mirror the markers: stripping follows the dialect the request
        // admitted, the client's envelope excepted, which no dialect admits
        // and every one of them parses.
        .closers =
            id ? ToolCloserSet(kQwenClosers) : ToolCloserSet(kToolClosers),
        .dialect = ext::FindToolDialect(Format::kDeepSeek),
        .qwen_calls = true,
    };
  if (*id == ext::BuiltinToolDialectId(Format::kQwen)) {
    static constexpr std::array<std::string_view, 1> qwen{
        tokenization::QwenToolSyntax::kCallOpen};
    // A native function header, rather than a bare marker mentioned in prose,
    // permits the implicit reasoning boundary used by llama.cpp's native
    // parsers.
    static constexpr std::array<std::string_view, 1> header{
        tokenization::QwenToolSyntax::kNativeCallOpen};
    return {
        .markers = qwen,
        .reasoning_markers = header,
        .closers = kQwenClosers,
        .qwen_calls = true,
        .native = true,
        // Qwen's llama.cpp PEG uses `reasoning << content`: `<<` consumes
        // space().
        .reasoning_separator = " \t\r\n\f\v",
    };
  }
  const auto* dialect = ext::FindToolDialect(*id);
  if (dialect == nullptr)
    return {};
  return {
      .markers = dialect->markers,
      .reasoning_markers = dialect->reasoning_markers,
      .closers = dialect->closers,
      .dialect = dialect,
      .native = true,
  };
}

// `value` without its trailing whitespace.
std::string_view TrimTrailing(std::string_view value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back())) != 0) {
    value.remove_suffix(1);
  }
  return value;
}

std::string_view Trim(std::string_view value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front())) != 0) {
    value.remove_prefix(1);
  }
  return TrimTrailing(value);
}

std::string_view AfterReasoningSeparator(std::string_view value,
                                         const ToolMarkup& markup) {
  // Only the boundary after an explicit reasoning close is framing; answer and
  // argument interiors are untouched.
  const auto first = value.find_first_not_of(markup.reasoning_separator);
  return first == std::string_view::npos ? std::string_view{}
                                         : value.substr(first);
}

// Qwen's XML-like calls look like Python keyword arguments, and the models
// sometimes write Python's True, False and None where the schema asks for a
// JSON boolean or null. Rewrites those words outside JSON string literals.
std::string PythonLiteralsToJson(std::string_view value) {
  std::string out;
  out.reserve(value.size());
  bool in_string = false;
  for (std::size_t i = 0; i < value.size();) {
    const char c = value[i];
    if (in_string) {
      const std::size_t length = c == '\\' && i + 1 < value.size() ? 2 : 1;
      out += value.substr(i, length);
      in_string = c != '"';
      i += length;
      continue;
    }
    if (std::isalpha(static_cast<unsigned char>(c)) == 0) {
      out += c;
      in_string = c == '"';
      ++i;
      continue;
    }
    std::size_t word_end = i;
    while (word_end < value.size() &&
           (std::isalnum(static_cast<unsigned char>(value[word_end])) != 0 ||
            value[word_end] == '_')) {
      ++word_end;
    }
    const auto word = value.substr(i, word_end - i);
    out += word == "True"    ? std::string_view{"true"}
           : word == "False" ? std::string_view{"false"}
           : word == "None"  ? std::string_view{"null"}
                             : word;
    i = word_end;
  }
  return out;
}

std::optional<json::Value> TryParseJson(std::string_view value) noexcept {
  try {
    return json::parse(value);
  } catch (...) {
    return std::nullopt;
  }
}

// Earliest call marker at or after `from` that is not quoted prose. A marker
// the model writes inside a backtick span or a code fence is the model
// *explaining* the format; an answer that names its tags must not be parsed
// or cut at them (#383). `full` is the whole response so the surrounding
// lines are visible.
std::size_t EarliestMarker(std::string_view full, ToolMarkerSet markers,
                           QuoteTracker& quotes, std::size_t from = 0) {
  for (std::size_t position = from; position < full.size();) {
    const std::size_t next = full.find('<', position);
    if (next == std::string_view::npos) {
      break;
    }
    // The cheap question first: only a position that carries a marker is worth
    // asking about quoting, which is what keeps a line full of '<' linear.
    for (const auto marker : markers) {
      const auto tag_offset = marker.find('<');
      if (tag_offset != std::string_view::npos && next - from >= tag_offset &&
          full.substr(next - tag_offset).starts_with(marker) &&
          !quotes.QuotedAt(next)) {
        return next - tag_offset;
      }
    }
    position = next + 1;
  }
  return std::string_view::npos;
}

std::size_t ReasoningToolMarker(std::string_view full, ToolMarkerSet markers,
                                QuoteTracker& quotes, std::size_t from = 0) {
  auto found = EarliestMarker(full, markers, quotes, from);
  while (found != std::string_view::npos && quotes.UnclosedAt(found))
    found = EarliestMarker(full, markers, quotes, found + 1);
  return found;
}

// Bytes at the end of [0, end) that could still open a call marker. A marker
// that would open inside quoted prose is prose, so it is not held back.
std::size_t HeldMarkerPrefix(std::string_view full, std::size_t end,
                             ToolMarkerSet markers, QuoteTracker& quotes) {
  std::size_t maximum_marker = 0;
  for (const auto marker : markers) {
    maximum_marker = std::max(maximum_marker, marker.size());
  }
  const std::size_t maximum =
      std::min(end, maximum_marker > 0 ? maximum_marker - 1 : 0);
  for (std::size_t length = maximum; length > 0; --length) {
    const std::string_view suffix = full.substr(end - length, length);
    if (std::ranges::any_of(markers,
                            [&](std::string_view marker) {
                              return marker.starts_with(suffix);
                            }) &&
        !quotes.QuotedAt(end - length)) {
      return length;
    }
  }
  return 0;
}

bool ToolMarkerPrefix(std::string_view text, ToolMarkerSet markers) {
  return std::ranges::any_of(
      markers, [&](auto marker) { return marker.starts_with(text); });
}

// Client envelope syntax is removed only when it names a declared tool, or
// immediately echoes a call the parser accepted. Other XML stays content.
constexpr std::array<std::string_view, 1> kEnvelopeHeads{"<invoke name="};
constexpr std::array<std::string_view, 1> kEnvelopeParameters{
    "<parameter name="};
constexpr std::array<std::string_view, 2> kEnvelopeClosers{"</invoke>",
                                                           "</parameter>"};
// The last of `tags` at or before `stop`, or npos when there is none or the
// last one sits in quoted prose.
template<std::size_t N>
std::size_t LastUnquotedTag(std::string_view full, std::size_t stop,
                            std::size_t floor,
                            const std::array<std::string_view, N>& tags,
                            QuoteTracker& quotes) {
  if (stop <= floor) {
    return std::string_view::npos;
  }
  // Search only what the caller can use: a tag starting before `floor` is
  // discarded there, and these tags hold no `<`, so none can straddle a marker
  // or a closer and be missed.
  const std::string_view window = full.substr(floor, stop - floor);
  std::size_t last = std::string_view::npos;
  for (const auto tag : tags) {
    const auto found = window.rfind(tag);
    if (found == std::string_view::npos) {
      continue;
    }
    if (last == std::string_view::npos || found > last) {
      last = found;
    }
  }
  if (last == std::string_view::npos || quotes.QuotedAt(floor + last)) {
    return std::string_view::npos;
  }
  return floor + last;
}

bool DeclaredEnvelope(std::string_view tag,
                      std::span<const tokenization::ChatTool> tools) {
  const auto value = tag.substr(kEnvelopeHeads.front().size());
  if (value.empty() || (value.front() != '\"' && value.front() != '\''))
    return false;
  const auto end = value.find(value.front(), 1);
  return end != std::string_view::npos &&
         std::ranges::any_of(tools, [&](const auto& tool) {
           return tool.name == value.substr(1, end - 1);
         });
}

// Find a closed or truncated client envelope whose request context identifies
// a tool attempt. A bare parameter block is not enough evidence.
std::size_t EnvelopeBlockStart(std::string_view full, std::size_t stop,
                               std::size_t floor, ToolCloserSet closers,
                               std::span<const tokenization::ChatTool> tools,
                               bool after_call, QuoteTracker& quotes) {
  const auto opening =
      LastUnquotedTag(full, stop, floor, kEnvelopeHeads, quotes);
  if (opening == std::string_view::npos) {
    return std::string_view::npos;
  }
  const auto tag_end = full.find('>', opening);
  if (!after_call) {
    if (tag_end == std::string_view::npos || tag_end >= stop) {
      return std::string_view::npos;
    }
    if (!DeclaredEnvelope(full.substr(opening, tag_end - opening), tools)) {
      return std::string_view::npos;
    }
  }
  const auto region = TrimTrailing(full.substr(opening, stop - opening));
  for (const auto closer : kEnvelopeClosers) {
    if (region.ends_with(closer)) {
      return opening;
    }
  }
  if (region.find(kEnvelopeParameters.front()) == std::string_view::npos) {
    return std::string_view::npos;
  }
  for (const auto closer : closers) {
    if (region.find(closer) != std::string_view::npos) {
      return std::string_view::npos;
    }
  }
  return opening;
}

// Remove framing only with request context: a closer echo directly after a
// parsed call, or a client envelope naming a declared tool. Other XML is data.
// A dialect without closers removes nothing.
std::string_view ContentBefore(std::string_view full, std::size_t begin,
                               std::size_t end, ToolCloserSet closers,
                               QuoteTracker& quotes,
                               std::span<const tokenization::ChatTool> tools,
                               bool after_call = false) {
  const auto stop = std::min(end, full.size());
  if (stop <= begin)
    return {};
  if (closers.empty())
    return full.substr(begin, stop - begin);
  if (after_call) {
    auto cursor = begin;
    bool removed = false;
    while (cursor < stop) {
      while (cursor < stop &&
             std::isspace(static_cast<unsigned char>(full[cursor])))
        ++cursor;
      const auto tag = std::ranges::find_if(closers, [&](auto closer) {
        return full.substr(cursor, stop - cursor).starts_with(closer);
      });
      if (tag == closers.end() || quotes.QuotedAt(cursor))
        break;
      cursor += tag->size();
      removed = true;
    }
    if (removed)
      begin = cursor;
  }
  const auto slice = full.substr(begin, stop - begin);
  // The newline a model writes between or after its calls is layout, not a
  // reply: a turn of parallel calls would otherwise carry content "\n" (#673).
  if (after_call && Trim(slice).empty())
    return {};
  const bool adjacent_envelope =
      after_call && Trim(slice).starts_with(kEnvelopeHeads.front());
  if (const auto block = EnvelopeBlockStart(full, stop, begin, closers, tools,
                                            adjacent_envelope, quotes);
      block != std::string_view::npos) {
    return TrimTrailing(full.substr(begin, block - begin));
  }
  return slice;
}

// Hold only a possible client envelope. A declared-tool head must stay pending
// until final parsing decides whether it is quoted documentation or framing.
std::size_t FramingHold(std::string_view full, std::size_t end,
                        std::size_t floor, ToolCloserSet closers,
                        QuoteTracker& quotes,
                        std::span<const tokenization::ChatTool> tools) {
  if (closers.empty())
    return 0;
  std::size_t hold = 0;
  if (end > 0) {
    // Hold only a suffix that can still become admitted framing. A bare
    // comparison such as "3 < 5" is already prose once the space arrives.
    const auto open = full.find_last_of('<', end - 1);
    if (open != std::string_view::npos && open >= floor &&
        !quotes.QuotedAt(open)) {
      const auto suffix = full.substr(open, end - open);
      const auto prefix = [&](auto tags) {
        return std::ranges::any_of(
            tags, [&](auto tag) { return tag.starts_with(suffix); });
      };
      if (prefix(kEnvelopeHeads)) {
        hold = suffix.size();
      }
    }
  }
  if (const auto opening =
          LastUnquotedTag(full, end, floor, kEnvelopeHeads, quotes);
      opening != std::string_view::npos) {
    const auto tag_end = full.find('>', opening);
    if (tag_end == std::string_view::npos || tag_end >= end ||
        DeclaredEnvelope(full.substr(opening, tag_end - opening), tools)) {
      hold = std::max(hold, end - opening);
    }
  }
  // The separator before framing is removed with it. Keep trailing whitespace
  // undecided until the next piece establishes whether it introduces prose or
  // a held tag, rather than streaming a separator we cannot retract later.
  const auto ready = full.substr(floor, end - hold - floor);
  hold += ready.size() - TrimTrailing(ready).size();
  return hold;
}

// Numbers keep the spelling the model generated, as llama.cpp returns them:
// above 2^53, and 1.0 or 1e5, a double would change the client's arguments
// and the replayed turn. Schema checks pass kCanonical, which the native
// number grammar reads.
std::string ArgumentsJson(
    std::span<const tokenization::ChatMessage::ToolArgument> arguments,
    json::Value::Numbers numbers = json::Value::Numbers::kAsParsed) {
  json::Value object = json::Value::object();
  for (const auto& argument : arguments) {
    if (argument.is_string) {
      object[argument.name] = argument.value;
      continue;
    }
    try {
      object[argument.name] = json::parse(argument.value);
    } catch (...) {
      object[argument.name] = argument.value;
    }
  }
  return object.dump(numbers);
}

// Qwen's XML-like arguments carry no type marker. The advertised schema is
// needed to distinguish a string such as 42 from the JSON number 42.
const json::Value* ResolveToolSchema(const json::Value& root,
                                     const json::Value& schema) {
  const auto* target = &schema;
  std::vector<const json::Value*> seen;
  while (const auto* reference = target->find("$ref")) {
    if (std::ranges::find(seen, target) != seen.end())
      return nullptr;
    seen.push_back(target);
    try {
      target = sampling::JsonConstraint::ResolveReference(root, *reference);
      // A parameter pointing to its containing tool schema is recursive.
      // Native best-effort grammar cannot enforce that recursion; do not
      // infer a definite scalar type from the outer parameter container.
      if (target == &root)
        return nullptr;
    } catch (const std::invalid_argument&) {
      return nullptr;  // Non-strict schemas may not be compilable.
    }
  }
  return target;
}

bool SchemaAccepts(const json::Value& root, const json::Value& original,
                   const json::Value& value, std::size_t depth = 0) {
  const auto* resolved = ResolveToolSchema(root, original);
  if (!resolved || depth > 16)
    return true;  // Retain text for unknown non-strict argument types.
  const auto& schema = *resolved;
  // Match value kinds, as llama.cpp's value_types(), rather than guessing
  // from the lexeme. {"const":"42"} is text even without an explicit type.
  const auto same_kind = [&](const json::Value& candidate) {
    return (candidate.is_string() && value.is_string()) ||
           (candidate.is_number() && value.is_number()) ||
           (candidate.is_bool() && value.is_bool()) ||
           (candidate.is_null() && value.is_null()) ||
           (candidate.is_array() && value.is_array()) ||
           (candidate.is_object() && value.is_object());
  };
  if (const auto* constant = schema.find("const"))
    return same_kind(*constant);
  if (const auto* values = schema.find("enum"); values && values->is_array())
    return std::ranges::any_of(values->items(), same_kind);
  const auto matches = [&](std::string_view type) {
    return (type == "string" && value.is_string()) ||
           (type == "number" && value.is_number()) ||
           (type == "integer" && value.is_number() &&
            std::floor(value.as_double()) == value.as_double()) ||
           (type == "boolean" && value.is_bool()) ||
           (type == "null" && value.is_null()) ||
           (type == "array" && value.is_array()) ||
           (type == "object" && value.is_object());
  };
  if (const auto* type = schema.find("type")) {
    if (type->is_string())
      return matches(type->get_str());
    if (type->is_array())
      return std::ranges::any_of(type->items(), [&](const auto& item) {
        return item.is_string() && matches(item.get_str());
      });
    return false;
  }
  for (const auto* name : {"anyOf", "oneOf"}) {
    if (const auto* choices = schema.find(name); choices && choices->is_array())
      return std::ranges::any_of(choices->items(), [&](const auto& item) {
        return SchemaAccepts(root, item, value, depth + 1);
      });
  }
  if (schema.contains("properties") ||
      (schema.contains("additionalProperties") &&
       !(schema.find("additionalProperties")->is_bool() &&
         schema.find("additionalProperties")->as_bool())))
    return value.is_object();
  if (const auto* parts = schema.find("allOf"); parts && parts->is_array())
    return std::ranges::all_of(parts->items(), [&](const auto& item) {
      return SchemaAccepts(root, item, value, depth + 1);
    });
  if (schema.contains("items") || schema.contains("prefixItems"))
    return value.is_array();
  if (schema.contains("pattern") || schema.contains("minLength") ||
      schema.contains("maxLength"))
    return value.is_string();
  return true;
}

bool StringOnly(const json::Value& root, const json::Value& schema,
                unsigned depth = 0) {
  if (depth > 32)
    return false;
  const auto* resolved = ResolveToolSchema(root, schema);
  if (!resolved)
    return false;
  if (resolved != &schema)
    return StringOnly(root, *resolved, depth + 1);
  if (const auto* type = schema.find("type"); type && type->is_string())
    return type->get_str() == "string";
  if (const auto* value = schema.find("const"))
    return value->is_string();
  if (const auto* values = schema.find("enum"); values && values->is_array())
    return !values->items().empty() &&
           std::ranges::all_of(values->items(), [](const auto& value) {
             return value.is_string();
           });
  for (const auto* key : {"anyOf", "oneOf"})
    if (const auto* values = schema.find(key); values && values->is_array())
      return !values->items().empty() &&
             std::ranges::all_of(values->items(), [&](const auto& value) {
               return StringOnly(root, value, depth + 1);
             });
  return false;
}

std::string ArgumentsPrefix(const ParsedToolCall& call) {
  auto prefix = ArgumentsJson(call.arguments);
  prefix.pop_back();
  return prefix;
}

void PartialStringArgument(ParsedToolCall& call, std::string_view name,
                           std::string_view value) {
  if (std::ranges::any_of(call.arguments, [&](const auto& argument) {
        return argument.name == name;
      }))
    return;  // Identical repeated parameters are discarded by the full parser.
  auto prefix = ArgumentsPrefix(call);
  if (!call.arguments.empty())
    prefix += ',';
  prefix += json::Value(std::string(name)).dump() + ":";
  auto quoted = json::Value(std::string(value)).dump();
  quoted.pop_back();
  call.partial_arguments = prefix + quoted;
}

// Resolves the schema's exact key. A declared key with surrounding spaces is
// preserved; otherwise the spelled name is trimmed, as llama.cpp's mapper
// does.
std::string QwenParameterName(const json::Value* properties,
                              std::string_view spelled) {
  return properties && properties->is_object() &&
                 properties->contains(std::string(spelled))
             ? std::string(spelled)
             : std::string(Trim(spelled));
}

// Interprets one scanned Qwen parameter against the advertised schema. Qwen's
// XML-like arguments carry no type marker, so the schema decides whether the
// value is a string or JSON; without one the raw text is kept.
bool InterpretQwenParameter(
    const std::optional<json::Value>& schema, const json::Value* properties,
    const tokenization::QwenToolCallParameter& parameter,
    tokenization::ChatMessage::ToolArgument* argument) {
  const std::string name = QwenParameterName(properties, parameter.name);
  if (name.empty())
    return false;
  const auto* property = properties ? properties->find(name) : nullptr;
  const bool string_allowed =
      !property || SchemaAccepts(*schema, *property,
                                 json::Value(std::string(parameter.value)));
  // Prefer text if the schema permits it; parsing ambiguous scalars as JSON
  // would silently change a caller's declared string type. A union also
  // admitting other types tries those first, as llama.cpp's qwen3-coder
  // parser does.
  std::optional<json::Value> typed;
  if (string_allowed && property) {
    typed = TryParseJson(Trim(parameter.value));
    if (!typed || typed->is_string() ||
        !SchemaAccepts(*schema, *property, *typed))
      typed.reset();
  }
  const bool is_string = string_allowed && !typed;
  std::string raw(is_string ? parameter.value : Trim(parameter.value));
  if (!is_string && !typed) {
    auto parsed = TryParseJson(raw);
    if (parsed && !SchemaAccepts(*schema, *property, *parsed))
      parsed.reset();
    if (!parsed) {
      auto converted = PythonLiteralsToJson(raw);
      auto alternative = TryParseJson(converted);
      if (alternative && SchemaAccepts(*schema, *property, *alternative)) {
        raw = std::move(converted);
        parsed = std::move(alternative);
      }
    }
    // As in llama.cpp, a value the grammar admitted is kept when its schema
    // cannot be enforced natively (e.g. a recursive or empty one). Recovery
    // without a grammar checks the whole call below.
    if (!parsed)
      return false;
  }
  *argument = {.name = name, .value = std::move(raw), .is_string = is_string};
  return true;
}

void ParseQwenCalls(
    std::string_view text, std::span<const tokenization::ChatTool> tools,
    std::vector<ParsedToolCall>* calls,
    std::vector<std::pair<std::size_t, std::size_t>>* spans = nullptr,
    bool require_schema = false, bool partial = false) {
  QuoteTracker call_quotes;
  call_quotes.Reset(text);
  std::size_t cursor = 0;
  while ((cursor = text.find(tokenization::QwenToolSyntax::kCallOpen,
                             cursor)) != std::string_view::npos) {
    if (call_quotes.QuotedAt(cursor) ||
        (partial && call_quotes.UnclosedAt(cursor))) {
      cursor += tokenization::QwenToolSyntax::kCallOpen.size();
      continue;
    }
    const auto marker_begin = cursor;
    const auto scan = tokenization::ScanQwenToolCall(text.substr(marker_begin));
    if (!scan.matched || !scan.call) {
      cursor = marker_begin + scan.progress;
      if (cursor == marker_begin)
        cursor += tokenization::QwenToolSyntax::kCallOpen.size();
      continue;
    }
    cursor = marker_begin + scan.progress;
    const auto& syntax = *scan.call;
    ParsedToolCall call;
    call.name = syntax.name;
    std::optional<json::Value> schema;
    for (const auto& tool : tools) {
      if (tool.name == call.name) {
        schema = TryParseJson(tool.parameters_json);
        break;
      }
    }
    const auto* object = schema ? ResolveToolSchema(*schema, *schema) : nullptr;
    const auto* properties = object ? object->find("properties") : nullptr;
    bool valid = true;
    bool complete = false;
    // An implicit end of reasoning is only tentative while the call is
    // unfinished: a declared call under the native grammar may still complete.
    const bool tentative = partial && syntax.canonical_header && schema &&
                           !require_schema &&
                           !call_quotes.UnclosedAt(marker_begin);
    if (syntax.native) {
      for (const auto& parameter : syntax.parameters) {
        // Recovery resumes after the rejected parameter, not after the call.
        tokenization::ChatMessage::ToolArgument argument;
        if (!InterpretQwenParameter(schema, properties, parameter, &argument)) {
          cursor = marker_begin + parameter.progress;
          valid = false;
          break;
        }
        // Models sometimes repeat a parameter. An identical copy is harmless;
        // conflicting copies leave no safe choice.
        const auto previous =
            std::ranges::find(call.arguments, argument.name,
                              [](const auto& arg) { return arg.name; });
        if (previous == call.arguments.end()) {
          call.arguments.push_back(std::move(argument));
        } else if (previous->value != argument.value ||
                   previous->is_string != argument.is_string) {
          cursor = marker_begin + parameter.progress;
          valid = false;
          break;
        }
      }
      if (valid && syntax.complete)
        complete = true;
      // Arguments after a rejected parameter are never streamed.
      if (!complete && tentative && valid && syntax.unfinished) {
        const std::string name =
            QwenParameterName(properties, syntax.unfinished->name);
        const auto* property = properties ? properties->find(name) : nullptr;
        if (syntax.unfinished->framed && !name.empty() &&
            (!property || StringOnly(*schema, *property)))
          PartialStringArgument(call, name, syntax.unfinished->value);
      }
    } else if (syntax.complete) {
      const auto parsed = TryParseJson(syntax.json_payload);
      if (parsed && parsed->is_object()) {
        call.name = parsed->member_str("name");
        const auto* arguments = parsed->find("arguments");
        if (!call.name.empty() && arguments && arguments->is_object()) {
          for (const auto& [name, value] : arguments->members())
            call.arguments.push_back(
                {.name = name,
                 .value = value.is_string()
                              ? value.get_str()
                              : value.dump(json::Value::Numbers::kAsParsed),
                 .is_string = value.is_string()});
          complete = true;
        }
      }
    }
    if (complete && !tools.empty() &&
        std::ranges::none_of(
            tools, [&](const auto& tool) { return tool.name == call.name; }))
      complete = false;
    if (complete && (require_schema || call_quotes.UnclosedAt(marker_begin))) {
      // An unfinished quotation is ambiguous. Restore the legacy fallback only
      // for a declared call whose complete arguments satisfy its schema.
      const auto tool =
          std::ranges::find(tools, call.name, &tokenization::ChatTool::name);
      if (tool == tools.end()) {
        complete = false;
      } else {
        try {
          const auto grammar = sampling::JsonConstraint::Compile(
              json::parse(tool->parameters_json), false);
          auto state = grammar->Start();
          for (const unsigned char byte :
               ArgumentsJson(call.arguments, json::Value::Numbers::kCanonical))
            state = grammar->Advance(state, byte);
          complete = grammar->Complete(state);
        } catch (const std::exception&) {
          complete = false;
        }
      }
    }
    if (complete || tentative) {
      call.id = RandomId("call_");
      if (!complete && !call.partial_arguments)
        call.partial_arguments = ArgumentsPrefix(call);
      calls->push_back(std::move(call));
      if (!complete)
        break;
      cursor = marker_begin + scan.end;
      // Parameter bytes belong to the call, not to prose quotation state.
      // An unmatched backtick in a file write must not quote the next call.
      call_quotes.Reset(text, cursor);
      if (spans)
        spans->emplace_back(marker_begin, cursor);
    }
  }
}

// Counts the argument prefix accepted by the advertised schema. A
// `string="false"` value that fails its declared type rejects the call, as for
// Qwen. Streaming keeps only the arguments before that rejection; strings and
// unknown parameters stay as the dialect wrote them.
std::size_t DeepSeekAcceptedArgumentCount(
    std::span<const tokenization::ChatTool> tools, std::string_view name,
    std::span<const ext::ScannedToolArgument> arguments) {
  const auto tool =
      std::ranges::find(tools, name, &tokenization::ChatTool::name);
  if (tool == tools.end())
    return arguments.size();
  const auto schema = TryParseJson(tool->parameters_json);
  if (!schema)
    return arguments.size();
  const auto* object = ResolveToolSchema(*schema, *schema);
  const auto* properties = object ? object->find("properties") : nullptr;
  if (!properties || !properties->is_object())
    return arguments.size();
  for (std::size_t index = 0; index < arguments.size(); ++index) {
    const auto& argument = arguments[index];
    if (argument.is_string)
      continue;
    const auto* property = properties->find(argument.name);
    if (!property)
      continue;
    const auto parsed = TryParseJson(Trim(argument.value));
    if (!parsed || !SchemaAccepts(*schema, *property, *parsed))
      return index;
  }
  return arguments.size();
}

// Converts a scanned DeepSeek argument list to the shared API form.
std::vector<tokenization::ChatMessage::ToolArgument> DeepSeekArguments(
    std::span<const ext::ScannedToolArgument> arguments) {
  std::vector<tokenization::ChatMessage::ToolArgument> converted;
  converted.reserve(arguments.size());
  for (const auto& argument : arguments)
    converted.push_back({.name = argument.name,
                         .value = argument.value,
                         .is_string = argument.is_string});
  return converted;
}

// Calls of a registered dialect, read by its scanner and checked against the
// declared schemas.
void ParseScannedCalls(
    std::string_view text, const ToolMarkup& markup,
    std::span<const tokenization::ChatTool> tools,
    std::vector<ParsedToolCall>* calls,
    std::vector<std::pair<std::size_t, std::size_t>>* spans = nullptr,
    bool partial = false) {
  const auto* dialect = markup.dialect;
  if (dialect == nullptr || dialect->scan_tool_calls == nullptr)
    return;
  // The Qwen opener is not an envelope for this dialect even when the JSON
  // fallback grammar admits both.
  std::vector<std::string_view> dialect_markers;
  dialect_markers.reserve(markup.markers.size());
  for (const auto marker : markup.markers) {
    if (!markup.qwen_calls || marker != tokenization::QwenToolSyntax::kCallOpen)
      dialect_markers.push_back(marker);
  }
  std::vector<ext::ScannedToolCall> scanned;
  dialect->scan_tool_calls(text, dialect_markers, &scanned, spans, partial);
  for (auto& invocation : scanned) {
    const auto accepted = DeepSeekAcceptedArgumentCount(tools, invocation.name,
                                                        invocation.arguments);
    const auto arguments = std::span(invocation.arguments).first(accepted);
    const bool schema_valid = accepted == invocation.arguments.size();
    const bool valid = invocation.valid && schema_valid;
    if (valid && invocation.complete) {
      ParsedToolCall call;
      call.id = RandomId("call_");
      call.name = std::move(invocation.name);
      call.arguments = DeepSeekArguments(arguments);
      calls->push_back(std::move(call));
      continue;
    }
    if (!partial || invocation.name.empty())
      continue;
    ParsedToolCall call;
    call.id = RandomId("call_");
    call.name = std::move(invocation.name);
    call.arguments = DeepSeekArguments(arguments);
    if (invocation.unfinished && schema_valid)
      PartialStringArgument(call, invocation.unfinished->name,
                            invocation.unfinished->value);
    if (!call.partial_arguments)
      call.partial_arguments = ArgumentsPrefix(call);
    calls->push_back(std::move(call));
    break;
  }
}
ParsedGeneration ParseGeneration(
    std::string_view raw,
    TextGenerationBackend::InitialOutputState initial_output_state,
    const ReasoningMarkup& reasoning,
    std::span<const tokenization::ChatTool> tools,
    ChatRequest::ToolChoice choice, bool enforce_required,
    const ToolMarkup& markup, QuoteTracker& quotes, bool partial = false,
    bool track_text = false) {
  const auto markers = markup.markers;
  const auto closers = markup.closers;
  quotes.Reset(raw);
  ParsedGeneration parsed;
  std::string_view content = raw;
  std::string_view reasoning_call_context;
  bool unfinished_reasoning_quote = false;
  const bool recognize_tools =
      !tools.empty() && choice != ChatRequest::ToolChoice::kNone;
  const auto tool_marker = [recognize_tools, markers,
                            &quotes](std::string_view text) {
    quotes.Reset(text);
    return recognize_tools ? EarliestMarker(text, markers, quotes)
                           : std::string_view::npos;
  };

  if (initial_output_state ==
      TextGenerationBackend::InitialOutputState::kReasoning) {
    if (content.starts_with(reasoning.open)) {
      content.remove_prefix(reasoning.open.size());
    }
    std::size_t think_end = content.find(reasoning.close);
    if (tool_marker(content) < think_end)
      think_end = std::string_view::npos;
    if (think_end == std::string_view::npos) {
      const auto marker = tool_marker(content);
      parsed.reasoning_content = std::string(Trim(content.substr(0, marker)));
      if (marker == std::string_view::npos) {
        if (enforce_required && choice == ChatRequest::ToolChoice::kRequired)
          throw TextGenerationError(
              TextGenerationErrorCode::kToolChoiceUnsatisfied,
              "model did not produce a declared tool call");
        return parsed;
      }
      reasoning_call_context = content;
      unfinished_reasoning_quote = quotes.UnclosedAt(marker);
      SetParsedText(parsed, content.substr(marker), raw, track_text);
    } else {
      parsed.reasoning_content =
          std::string(Trim(content.substr(0, think_end)));
      content.remove_prefix(think_end + reasoning.close.size());
      SetParsedText(parsed, AfterReasoningSeparator(content, markup), raw,
                    track_text);
    }
  } else if (initial_output_state ==
             TextGenerationBackend::InitialOutputState::kAuto) {
    // Only the initial phase has reasoning markup semantics. A literal tag
    // inside an argument (or quoted ordinary text) must never be stripped.
    const auto leading = content.find_first_not_of(" \t\r\n");
    const std::size_t think_start =
        !reasoning.open.empty() && leading != std::string_view::npos &&
                content.substr(leading).starts_with(reasoning.open)
            ? leading
            : std::string_view::npos;
    if (think_start != std::string_view::npos) {
      const std::size_t think_content_start =
          think_start + reasoning.open.size();
      std::size_t think_end =
          content.find(reasoning.close, think_content_start);
      const auto marker = tool_marker(content.substr(think_content_start));
      if (marker != std::string_view::npos &&
          think_content_start + marker < think_end)
        think_end = std::string_view::npos;
      if (think_end != std::string_view::npos) {
        parsed.reasoning_content = std::string(Trim(content.substr(
            think_content_start, think_end - think_content_start)));
        std::string_view remaining =
            content.substr(think_end + reasoning.close.size());
        remaining = AfterReasoningSeparator(remaining, markup);
        if (think_start > 0) {
          SetParsedText(parsed, content.substr(0, think_start), raw,
                        track_text);
          parsed.text += remaining;
          if (track_text && !remaining.empty())
            parsed.text_spans.push_back(
                {static_cast<std::size_t>(remaining.data() - raw.data()),
                 remaining.size()});
        } else {
          SetParsedText(parsed, remaining, raw, track_text);
        }
      } else {
        const auto remaining = content.substr(think_content_start);
        const auto marker = tool_marker(remaining);
        parsed.reasoning_content =
            std::string(Trim(remaining.substr(0, marker)));
        SetParsedText(parsed, content.substr(0, think_start), raw, track_text);
        if (marker != std::string_view::npos) {
          reasoning_call_context = remaining;
          unfinished_reasoning_quote = quotes.UnclosedAt(marker);
          parsed.text += remaining.substr(marker);
          if (track_text)
            parsed.text_spans.push_back(
                {static_cast<std::size_t>(remaining.data() - raw.data()) +
                     marker,
                 remaining.size() - marker});
        }
      }
    } else {
      SetParsedText(parsed, content, raw, track_text);
    }
  } else {
    // After reasoning the prompt closed, the model writes the separator.
    SetParsedText(parsed,
                  initial_output_state ==
                          TextGenerationBackend::InitialOutputState::
                              kContentAfterReasoning
                      ? AfterReasoningSeparator(content, markup)
                      : content,
                  raw, track_text);
  }

  quotes.Reset(parsed.text);
  const std::size_t marker = EarliestMarker(parsed.text, markers, quotes);
  if (choice != ChatRequest::ToolChoice::kNone && !tools.empty() &&
      marker != std::string_view::npos) {
    if (partial && (unfinished_reasoning_quote || quotes.UnclosedAt(marker)))
      return parsed;
    const auto text_from_tools = std::string_view(parsed.text).substr(marker);
    std::vector<std::pair<std::size_t, std::size_t>> spans;
    // The outer envelope selects the format, as in llama.cpp's model parsers.
    // Scanning both dialects would turn a literal call inside an argument into
    // an additional API invocation.
    if (markup.qwen_calls &&
        text_from_tools.starts_with(tokenization::QwenToolSyntax::kCallOpen))
      ParseQwenCalls(parsed.text, tools, &parsed.tool_calls, nullptr,
                     unfinished_reasoning_quote, partial);
    else
      ParseScannedCalls(
          text_from_tools, markup, tools, &parsed.tool_calls, &spans,
          partial && !unfinished_reasoning_quote && !quotes.UnclosedAt(marker));
    std::erase_if(parsed.tool_calls, [&](const auto& call) {
      return std::ranges::none_of(
          tools, [&](const auto& tool) { return tool.name == call.name; });
    });
    if (parsed.tool_calls.empty() && unfinished_reasoning_quote &&
        choice != ChatRequest::ToolChoice::kRequired) {
      // The tentative implicit end of reasoning was an invalid quoted example.
      // Preserve it in its original phase, including a later explicit boundary.
      const auto end = reasoning_call_context.find(reasoning.close);
      if (end != std::string_view::npos) {
        parsed = ParseGeneration(
            reasoning_call_context.substr(end + reasoning.close.size()),
            TextGenerationBackend::InitialOutputState::kContent, reasoning,
            tools, choice, enforce_required, markup, quotes, partial,
            track_text);
        RebaseTextSpans(parsed,
                        static_cast<std::size_t>(reasoning_call_context.data() -
                                                 raw.data()) +
                            end + reasoning.close.size());
      } else {
        parsed.text.clear();
        parsed.text_spans.clear();
      }
      parsed.reasoning_content =
          std::string(reasoning_call_context.substr(0, end));
      return parsed;
    }
    if (parsed.tool_calls.empty() && quotes.UnclosedAt(marker) &&
        choice != ChatRequest::ToolChoice::kRequired) {
      return parsed;
    }
    // An explicit stop can interrupt a call before its closing tags. Keep
    // complete calls, but do not expose an unfinished call as ordinary text.
    if (!parsed.tool_calls.empty() || !enforce_required) {
      // Text outside complete envelopes is content. Framing echoed directly
      // after a call is removed, as for Qwen calls (#383).
      const std::string full = std::move(parsed.text);
      const auto old_spans = std::move(parsed.text_spans);
      std::string text;
      const auto append = [&](std::string_view part) {
        text += part;
        if (!part.empty())
          AppendTextSpans(parsed.text_spans, old_spans,
                          part.data() - full.data(), part.size());
      };
      append(ContentBefore(full, 0, marker, closers, quotes, tools));
      std::size_t cursor = marker;
      for (const auto& [begin, end] : spans) {
        append(ContentBefore(full, cursor, marker + begin, closers, quotes,
                             tools, cursor > marker));
        cursor = marker + end;
      }
      if (!spans.empty()) {
        auto unfinished = EarliestMarker(full, markers, quotes, cursor);
        if (unfinished == std::string_view::npos && !enforce_required &&
            choice == ChatRequest::ToolChoice::kRequired)
          if (const auto held =
                  HeldMarkerPrefix(full, full.size(), markers, quotes))
            unfinished = full.size() - held;
        append(ContentBefore(full, cursor, unfinished, closers, quotes, tools,
                             true));
      }
      parsed.text = std::move(text);
      parsed.hide_tool_markup = true;
    }
  } else if (recognize_tools) {
    // No marker: the text is content, but the framing it carries still goes.
    // The streamed response drops it as the turn ends, and both transports
    // must report the same content.
    const auto part = ContentBefore(parsed.text, 0, parsed.text.size(), closers,
                                    quotes, tools);
    std::vector<TextSourceSpan> spans;
    if (!part.empty())
      AppendTextSpans(spans, parsed.text_spans,
                      part.data() - parsed.text.data(), part.size());
    parsed.text = std::string(part);
    parsed.text_spans = std::move(spans);
  }
  if (enforce_required && choice == ChatRequest::ToolChoice::kRequired &&
      parsed.tool_calls.empty())
    throw TextGenerationError(TextGenerationErrorCode::kToolChoiceUnsatisfied,
                              "model did not produce a declared tool call");
  return parsed;
}

ParsedGeneration ParseStructuredGeneration(
    std::string_view raw, TextGenerationBackend::InitialOutputState initial,
    const ReasoningMarkup& reasoning,
    std::span<const tokenization::ChatTool> tools,
    ChatRequest::ToolChoice choice, bool enforce_required, bool tool_only,
    const ToolMarkup& markup, QuoteTracker& quotes, bool partial = false,
    bool track_text = false) {
  const auto markers = markup.markers;
  const auto closers = markup.closers;
  ParsedGeneration parsed;
  const auto source = raw;
  if (initial == TextGenerationBackend::InitialOutputState::kAuto) {
    const auto leading = raw.find_first_not_of(" \t\r\n");
    if (!reasoning.open.empty() && leading != std::string_view::npos &&
        raw.substr(leading).starts_with(reasoning.open)) {
      raw.remove_prefix(leading);
      initial = TextGenerationBackend::InitialOutputState::kReasoning;
    }
  }
  if (initial == TextGenerationBackend::InitialOutputState::kReasoning) {
    if (raw.starts_with(reasoning.open))
      raw.remove_prefix(reasoning.open.size());
    const auto end = raw.find(reasoning.close);
    quotes.Reset(raw);
    const auto call =
        !tools.empty() && choice != ChatRequest::ToolChoice::kNone
            ? ReasoningToolMarker(raw, markup.reasoning_markers, quotes)
            : std::string_view::npos;
    if (call < end) {
      if (partial && quotes.UnclosedAt(call))
        return parsed;
      parsed.reasoning_content = std::string(raw.substr(0, call));
      raw.remove_prefix(call);
    } else {
      parsed.reasoning_content = std::string(raw.substr(0, end));
      if (end == std::string_view::npos)
        return parsed;
      raw.remove_prefix(end + reasoning.close.size());
      raw = AfterReasoningSeparator(raw, markup);
    }
  } else if (initial == TextGenerationBackend::InitialOutputState::
                            kContentAfterReasoning) {
    raw = AfterReasoningSeparator(raw, markup);
  }
  const auto trimmed = Trim(raw);
  const bool held_opener = choice == ChatRequest::ToolChoice::kRequired &&
                           trimmed.empty() && !raw.empty() &&
                           ToolMarkerPrefix(raw, markers);
  const auto content = tool_only || held_opener ? raw : trimmed;
  quotes.Reset(content);
  if (const auto marker = EarliestMarker(content, markers, quotes);
      !tools.empty() && choice != ChatRequest::ToolChoice::kNone &&
      (tool_only || marker == 0) && marker != std::string_view::npos &&
      !(markup.qwen_calls && content.substr(marker).starts_with(
                                 tokenization::QwenToolSyntax::kCallOpen))) {
    auto native = ParseGeneration(
        content, TextGenerationBackend::InitialOutputState::kContent, reasoning,
        tools, choice, enforce_required, markup, quotes, partial, track_text);
    RebaseTextSpans(native, content.data() - source.data());
    native.reasoning_content = std::move(parsed.reasoning_content);
    return native;
  }
  constexpr std::string_view tool_start =
      tokenization::QwenToolSyntax::kCallOpen;
  auto marker = tool_only ? EarliestMarker(content, markers, quotes) : 0;
  if (tool_only && choice == ChatRequest::ToolChoice::kRequired &&
      marker == std::string_view::npos) {
    if (const auto held =
            HeldMarkerPrefix(content, content.size(), markers, quotes)) {
      marker = content.size() - held;
    }
  }
  if (!tools.empty() && choice != ChatRequest::ToolChoice::kNone &&
      !content.empty() && marker != std::string_view::npos &&
      (tool_only || (markup.qwen_calls && content.starts_with(tool_start)) ||
       ToolMarkerPrefix(content, markers))) {
    ParsedGeneration call;
    std::vector<std::pair<std::size_t, std::size_t>> spans;
    ParseQwenCalls(content, tools, &call.tool_calls,
                   tool_only ? &spans : nullptr, false, partial);
    if (tool_only) {
      const auto append = [&](std::string_view part) {
        call.text += part;
        if (track_text && !part.empty())
          call.text_spans.push_back(
              {static_cast<std::size_t>(part.data() - source.data()),
               part.size()});
      };
      std::size_t cursor = 0;
      for (const auto& [begin, end] : spans) {
        append(ContentBefore(content, cursor, begin, closers, quotes, tools,
                             cursor > 0));
        cursor = end;
      }
      const auto tail = content.substr(cursor);
      auto unfinished = EarliestMarker(content, markers, quotes, cursor);
      if (unfinished != std::string_view::npos) {
        unfinished -= cursor;
      } else if (!enforce_required &&
                 choice == ChatRequest::ToolChoice::kRequired) {
        if (const auto held =
                HeldMarkerPrefix(content, content.size(), markers, quotes)) {
          unfinished = tail.size() - held;
        }
      }
      const std::size_t limit = spans.empty() ? marker : unfinished;
      const std::size_t stop = limit == std::string_view::npos
                                   ? content.size()
                                   : std::min(cursor + limit, content.size());
      // The model repeats the framing of the call it just made; that echo is
      // markup, not prose the client should replay on the next turn (#383).
      append(ContentBefore(content, cursor, stop, closers, quotes, tools,
                           cursor > 0));
    }
    if (call.tool_calls.empty() && marker != std::string_view::npos &&
        quotes.UnclosedAt(marker) &&
        choice != ChatRequest::ToolChoice::kRequired) {
      SetParsedText(parsed, raw, source, track_text);
      return parsed;
    }
    call.hide_tool_markup = true;
    if (call.tool_calls.empty() && enforce_required &&
        choice == ChatRequest::ToolChoice::kRequired)
      throw TextGenerationError(TextGenerationErrorCode::kToolChoiceUnsatisfied,
                                "model did not complete a declared tool call");
    call.reasoning_content = std::move(parsed.reasoning_content);
    return call;
  }
  // Markers in JSON strings are data. Only the initial reasoning phase has
  // markup semantics; ordinary tool/reasoning parsing must not run here.
  if (enforce_required && choice == ChatRequest::ToolChoice::kRequired)
    throw TextGenerationError(TextGenerationErrorCode::kToolChoiceUnsatisfied,
                              "model did not complete a declared tool call");
  // A constrained turn that produced no call is still reported without
  // framing: the streamed response drops it as the turn ends, and both
  // transports must report the same content.
  SetParsedText(parsed,
                tool_only
                    ? ContentBefore(raw, 0, raw.size(), closers, quotes, tools)
                    : raw,
                source, track_text);
  return parsed;
}

class StreamingTextFilter {
public:
  using EmitCallback =
      std::function<bool(std::string_view piece, bool is_reasoning,
                         std::span<const TextSourceSpan> source)>;

  StreamingTextFilter(
      TextGenerationBackend::InitialOutputState initial_output_state,
      ReasoningMarkup reasoning, EmitCallback emit_piece, bool structured,
      bool tool_only, bool recognize_tools, const ToolMarkup& markup,
      std::span<const tokenization::ChatTool> tools)
      : reasoning_(std::move(reasoning)),
        emit_piece_(std::move(emit_piece)),
        raw_content_(structured || tool_only),
        tool_only_(tool_only),
        recognize_tools_(recognize_tools),
        markup_(markup),
        markers_(markup.markers),
        closers_(markup.closers),
        tools_(tools) {
    if (initial_output_state ==
        TextGenerationBackend::InitialOutputState::kReasoning) {
      state_ = State::kThinking;
      reasoning_start_ = true;
    } else if (TextGenerationBackend::IsContent(initial_output_state)) {
      state_ = State::kContent;
      // The prompt closed reasoning: the model writes the separator.
      trim_reasoning_separator_ =
          initial_output_state ==
          TextGenerationBackend::InitialOutputState::kContentAfterReasoning;
    }
  }

  bool Push(std::string_view bytes, bool final = false) {
    const auto piece = decoder_.Push(bytes, final);
    raw_.append(piece);
    quotes_.Append(piece);
    if (tool_mode_) {
      hidden_.append(piece);
      return true;
    }
    pending_.append(piece);
    if (reasoning_start_) {
      // A prompt-opened reasoning phase may still emit its opening tag.
      // Match the buffered parser without exposing a split opener as thought.
      if (!final && reasoning_.open.starts_with(pending_) &&
          pending_.size() < reasoning_.open.size())
        return true;
      if (pending_.starts_with(reasoning_.open))
        pending_.erase(0, reasoning_.open.size());
      reasoning_start_ = false;
    }
    if (raw_content_ && state_ == State::kContent && !trim_reasoning_separator_)
      return StructuredContent();

    if (state_ == State::kInitial) {
      const std::string_view open = reasoning_.open;
      std::string_view view = pending_;
      while (!view.empty() &&
             std::isspace(static_cast<unsigned char>(view.front())) != 0) {
        view.remove_prefix(1);
      }
      if (view.empty()) {
        return true;
      }
      if (!open.empty() && open.starts_with(view)) {
        if (view == open) {
          state_ = State::kThinking;
          pending_.clear();
        }
        return true;
      }
      if (!open.empty() && view.starts_with(open)) {
        // A buffered result arrives as one piece: the opening tag together
        // with the reasoning after it.
        state_ = State::kThinking;
        pending_.erase(0, pending_.size() - view.size() + open.size());
      } else {
        state_ = State::kContent;
      }
    }

    if (state_ == State::kThinking) {
      const std::string_view close = reasoning_.close;
      const auto offset = pending_offset();
      const auto think_found = raw_.find(close, offset);
      const std::size_t end_pos = think_found == std::string::npos
                                      ? std::string::npos
                                      : think_found - offset;
      const auto thinking_markers =
          raw_content_ ? markup_.reasoning_markers : markers_;
      const auto found =
          !recognize_tools_ ? std::string::npos
          : raw_content_
              ? ReasoningToolMarker(raw_, thinking_markers, quotes_, offset)
              : EarliestMarker(raw_, thinking_markers, quotes_, offset);
      const std::size_t marker =
          found == std::string::npos ? std::string::npos : found - offset;
      if (marker < end_pos) {
        if (marker > 0 && !Emit(pending_.substr(0, marker), true))
          return false;
        hidden_ = pending_.substr(marker);
        pending_.clear();
        tool_mode_ = true;
        return true;
      }
      if (end_pos != std::string::npos) {
        if (end_pos > 0 && !Emit(pending_.substr(0, end_pos), true)) {
          return false;
        }
        std::string_view remaining = pending_;
        remaining.remove_prefix(end_pos + close.size());
        pending_ = std::string(remaining);
        state_ = State::kContent;
        quotes_.Reset(raw_, offset + end_pos + close.size());
        trim_reasoning_separator_ = true;
      } else {
        std::size_t held = !recognize_tools_
                               ? 0
                               : HeldMarkerPrefix(raw_, raw_.size(),
                                                  thinking_markers, quotes_);
        held = std::min(held, pending_.size());
        for (std::size_t len = std::min(pending_.size(), close.size() - 1);
             len > 0; --len) {
          if (close.starts_with(pending_.substr(pending_.size() - len))) {
            held = std::max(held, len);
            break;
          }
        }
        const std::size_t ready = pending_.size() - held;
        if (ready > 0 && !Emit(pending_.substr(0, ready), true)) {
          return false;
        }
        pending_.erase(0, ready);
        return true;
      }
    }

    if (state_ == State::kContent) {
      if (trim_reasoning_separator_) {
        const auto content = AfterReasoningSeparator(pending_, markup_);
        if (content.empty()) {
          pending_.clear();
          return true;
        }
        pending_.erase(0, pending_.size() - content.size());
        trim_reasoning_separator_ = false;
      }
      if (raw_content_) {
        return StructuredContent();
      }
      const auto offset = pending_offset();
      const auto found = recognize_tools_
                             ? EarliestMarker(raw_, markers_, quotes_, offset)
                             : std::string_view::npos;
      const std::size_t marker =
          found == std::string::npos ? std::string::npos : found - offset;
      if (marker != std::string::npos) {
        const auto prose =
            recognize_tools_
                ? ContentBefore(raw_, offset, found, closers_, quotes_, tools_)
                : std::string_view(pending_).substr(0, marker);
        if (!prose.empty() && !Emit(prose, false, prose.data() - raw_.data())) {
          return false;
        }
        hidden_ = pending_.substr(marker);
        pending_.clear();
        tool_mode_ = true;
        return true;
      }

      const std::size_t held =
          recognize_tools_
              ? std::min(std::max(HeldMarkerPrefix(raw_, raw_.size(), markers_,
                                                   quotes_),
                                  FramingHold(raw_, raw_.size(), offset,
                                              closers_, quotes_, tools_)),
                         pending_.size())
              : 0;
      const std::size_t ready = pending_.size() - held;
      if (ready > 0 && !Emit(pending_.substr(0, ready), false, offset)) {
        return false;
      }
      pending_.erase(0, ready);
      return true;
    }

    return true;
  }

  bool Finish(bool hide_tool_markup, std::string_view content = {},
              std::string_view reasoning = {},
              std::span<const TextSourceSpan> content_spans = {}) {
    const auto emit_content = [&](std::size_t offset) {
      std::vector<TextSourceSpan> spans;
      AppendTextSpans(spans, content_spans, offset, content.size() - offset);
      return EmitMapped(content.substr(offset), false, spans);
    };
    if (tool_mode_ && state_ == State::kThinking) {
      const auto tail = reasoning.substr(
          std::min(emitted_reasoning_bytes_, reasoning.size()));
      return (tail.empty() || Emit(tail, true)) &&
             (content.empty() || emit_content(0));
    }
    if (tool_only_ && tool_mode_ && hide_tool_markup) {
      // The prefix was streamed before the first call. Preserve prose between
      // or after complete calls without exposing their markup or partial calls.
      const auto offset = std::min(emitted_content_bytes_, content.size());
      const auto tail = content.substr(offset);
      return tail.empty() || emit_content(offset);
    }
    if (raw_content_ && !structured_started_ && hide_tool_markup) {
      pending_.clear();
      return true;
    }
    if (!tool_mode_) {
      if (!pending_.empty()) {
        // With tools declared, the closing tags the model repeats after a call
        // are markup; a client must never store them as assistant prose (#383).
        const auto offset = pending_offset();
        const auto text = recognize_tools_ && state_ == State::kContent
                              ? ContentBefore(raw_, offset, raw_.size(),
                                              closers_, quotes_, tools_)
                              : std::string_view(pending_);
        const bool is_reasoning = (state_ == State::kThinking);
        const bool emitted = Emit(text, is_reasoning, offset);
        pending_.clear();
        if (!emitted) {
          return false;
        }
      }
      return true;
    }
    if (!hide_tool_markup && !hidden_.empty()) {
      return Emit(hidden_, false, raw_.size() - hidden_.size());
    }
    return true;
  }

  [[nodiscard]] std::string_view raw() const noexcept { return raw_; }
  [[nodiscard]] bool in_tools() const noexcept { return tool_mode_; }

  // `pending_` always holds a suffix of what the model generated, so the
  // context of a marker inside it is decided against the whole response.
  [[nodiscard]] std::size_t pending_offset() const noexcept {
    return raw_.size() - pending_.size();
  }

private:
  bool Emit(std::string_view piece, bool reasoning,
            std::size_t offset = std::string_view::npos) {
    const TextSourceSpan span{offset, piece.size()};
    return EmitMapped(piece, reasoning,
                      offset == std::string_view::npos
                          ? std::span<const TextSourceSpan>{}
                          : std::span(&span, 1));
  }

  bool EmitMapped(std::string_view piece, bool reasoning,
                  std::span<const TextSourceSpan> source) {
    if (!emit_piece_(piece, reasoning, source))
      return false;
    if (reasoning)
      emitted_reasoning_bytes_ += piece.size();
    return true;
  }

  bool StructuredContent() {
    if (tool_only_) {
      const auto offset = pending_offset();
      const auto found = EarliestMarker(raw_, markers_, quotes_, offset);
      const auto start =
          found == std::string::npos ? std::string::npos : found - offset;
      if (start != std::string::npos) {
        const auto prose =
            ContentBefore(raw_, offset, found, closers_, quotes_, tools_);
        if (!prose.empty() && !Emit(prose, false, prose.data() - raw_.data())) {
          return false;
        }
        emitted_content_bytes_ += prose.size();
        hidden_ = pending_.substr(start);
        pending_.clear();
        tool_mode_ = true;
        return true;
      }
      const auto held = std::min(
          std::max(HeldMarkerPrefix(raw_, raw_.size(), markers_, quotes_),
                   FramingHold(raw_, raw_.size(), offset, closers_, quotes_,
                               tools_)),
          pending_.size());
      const auto ready = pending_.size() - held;
      if (ready) {
        const auto text = ContentBefore(raw_, offset, offset + ready, closers_,
                                        quotes_, tools_);
        if (!text.empty() && !Emit(text, false, text.data() - raw_.data())) {
          return false;
        }
        emitted_content_bytes_ += text.size();
      }
      pending_.erase(0, ready);
      return true;
    }
    if (!structured_started_) {
      const auto content = Trim(pending_);
      if (content.empty() || (ToolMarkerPrefix(content, markers_) &&
                              EarliestMarker(content, markers_, quotes_) != 0))
        return true;
      structured_started_ = true;
      if (EarliestMarker(content, markers_, quotes_) == 0) {
        tool_mode_ = true;
        hidden_ = std::exchange(pending_, {});
        return true;
      }
    }
    const auto offset = pending_offset();
    auto text = std::exchange(pending_, {});
    return text.empty() || Emit(text, false, offset);
  }
  enum class State : std::uint8_t {
    kInitial,
    kThinking,
    kContent,
  };

  ReasoningMarkup reasoning_;
  EmitCallback emit_piece_;
  bool structured_started_{false};
  core::Utf8Decoder decoder_;
  std::string raw_;
  std::string pending_;
  std::string hidden_;
  State state_{State::kInitial};
  bool tool_mode_{false};
  bool raw_content_{false};
  bool tool_only_{false};
  bool recognize_tools_{false};
  ToolMarkup markup_;
  ToolMarkerSet markers_;
  QuoteTracker quotes_;
  ToolCloserSet closers_;
  std::span<const tokenization::ChatTool> tools_;
  std::size_t emitted_content_bytes_{0};
  std::size_t emitted_reasoning_bytes_{0};
  bool trim_reasoning_separator_{false};
  bool reasoning_start_{false};
};

struct ToolDelta {
  std::size_t index;
  std::string id;
  std::string name;
  std::string arguments;
  bool complete{false};
};

// Like llama.cpp's task_result_state, retain IDs by call index and publish
// only the newly parsed argument suffix. Final parsing remains authoritative.
class StreamingToolCalls {
public:
  using Emit = std::function<bool(const ToolDelta&)>;
  explicit StreamingToolCalls(Emit emit) : emit_(std::move(emit)) {}

  bool Update(const std::vector<ParsedToolCall>& calls) {
    for (std::size_t index = 0; index < calls.size(); ++index) {
      const auto& call = calls[index];
      const auto arguments =
          call.partial_arguments.value_or(ArgumentsJson(call.arguments));
      const bool added = index == calls_.size();
      if (added)
        calls_.push_back({call.id, call.name, {}, false});
      auto& previous = calls_[index];
      if (previous.name != call.name ||
          !arguments.starts_with(previous.arguments)) {
        // A still-unfinished quotation or value can make a partial parse
        // temporarily disappear. Only the final parse may accept the turn.
        continue;
      }
      const bool complete = !call.partial_arguments.has_value();
      const auto delta = arguments.substr(previous.arguments.size());
      if (added || !delta.empty() || (complete && !previous.complete)) {
        if (!emit_({index, added ? previous.id : "", added ? previous.name : "",
                    delta, complete}))
          return false;
      }
      previous.arguments = arguments;
      previous.complete = complete;
    }
    return true;
  }

  bool Finish(const std::vector<ParsedToolCall>& calls, bool interrupted) {
    if (!Update(calls))
      return false;
    for (std::size_t index = 0; index < calls_.size(); ++index) {
      const auto& sent = calls_[index];
      if (index >= calls.size()) {
        if (interrupted && !sent.complete)
          continue;
      } else if (sent.name == calls[index].name &&
                 sent.arguments == ArgumentsJson(calls[index].arguments) &&
                 sent.complete) {
        continue;
      }
      throw std::runtime_error("streamed tool call did not complete safely");
    }
    return true;
  }

private:
  struct Call {
    std::string id;
    std::string name;
    std::string arguments;
    bool complete;
  };
  Emit emit_;
  std::vector<Call> calls_;
};

struct FrozenChat {
  bool response_format;
  bool constrained_tools;
  std::span<const tokenization::ChatTool> tools;
  ChatRequest::ToolChoice tool_choice;
};

ParsedGeneration StreamingGeneration(
    const StreamingTextFilter& filter, const FrozenChat& chat,
    TextGenerationBackend::InitialOutputState initial,
    const ReasoningMarkup& reasoning, const ToolMarkup& markup,
    QuoteTracker& quotes, bool enforce_required, bool partial = false,
    bool track_text = false) {
  // Native grammars establish argument framing before the value completes.
  // Recovery from unconstrained/JSON fallback text still requires a full call.
  partial = partial && markup.native;
  return chat.response_format || chat.constrained_tools
             ? ParseStructuredGeneration(
                   filter.raw(), initial, reasoning, chat.tools,
                   chat.tool_choice, enforce_required,
                   chat.constrained_tools && !chat.response_format, markup,
                   quotes, partial, track_text)
             : ParseGeneration(filter.raw(), initial, reasoning, chat.tools,
                               chat.tool_choice, enforce_required, markup,
                               quotes, partial, track_text);
}

}  // namespace

namespace {

Outcome Unsatisfied(const TextGenerationError& error) {
  return std::string_view(error.what()).find("complete") !=
                 std::string_view::npos
             ? Outcome::kNotCompleted
             : Outcome::kNotProduced;
}

FrozenChat Chat(const Setup& setup) {
  return {.response_format = setup.response_format,
          .constrained_tools = setup.constrained_tools,
          .tools = setup.tools,
          .tool_choice = setup.choice};
}

void Final(const ParsedGeneration& parsed, std::vector<Event>* events) {
  events->push_back(
      {.kind = EventKind::kFinalReasoning, .text = parsed.reasoning_content});
  events->push_back({.kind = EventKind::kFinalContent,
                     .text = parsed.text,
                     .spans = parsed.text_spans});
  for (std::size_t index = 0; index < parsed.tool_calls.size(); ++index)
    events->push_back({.kind = EventKind::kFinalCall,
                       .name = parsed.tool_calls[index].name,
                       .arguments = ArgumentsJson(parsed.tool_calls[index].arguments),
                       .index = index});
  events->push_back({.kind = EventKind::kFinalEnd,
                     .hide_tool_markup = parsed.hide_tool_markup});
}

}  // namespace

Outcome Stream(const Setup& setup, const std::vector<std::string_view>& pieces,
               bool interrupted, std::vector<Event>* events,
               std::vector<std::uint64_t>* piece_ns) {
  const auto markup = SelectToolMarkup(setup.dialect);
  const auto chat = Chat(setup);
  QuoteTracker quotes;
  StreamingTextFilter filter(
      setup.initial, setup.reasoning,
      [&](std::string_view text, bool reasoning,
          std::span<const TextSourceSpan> source) {
        events->push_back({.kind = reasoning ? EventKind::kReasoningDelta
                                             : EventKind::kContentDelta,
                           .text = std::string(text),
                           .spans = {source.begin(), source.end()}});
        return true;
      },
      setup.response_format,
      setup.constrained_tools && !setup.response_format,
      !setup.tools.empty() && setup.choice != ChatRequest::ToolChoice::kNone,
      markup, setup.tools);
  StreamingToolCalls tools([&](const ToolDelta& call) {
    events->push_back({.kind = EventKind::kToolDelta,
                       .name = call.name,
                       .arguments = call.arguments,
                       .index = call.index,
                       .begin = !call.id.empty(),
                       .complete = call.complete});
    return true;
  });
  for (const auto piece : pieces) {
    const auto started = std::chrono::steady_clock::now();
    filter.Push(piece);
    if (!setup.buffered && filter.in_tools() && markup.native)
      tools.Update(StreamingGeneration(filter, chat, setup.initial,
                                       setup.reasoning, markup, quotes, false,
                                       true)
                       .tool_calls);
    if (piece_ns)
      piece_ns->push_back(static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - started)
              .count()));
  }
  filter.Push({}, true);
  ParsedGeneration generated;
  try {
    generated = StreamingGeneration(filter, chat, setup.initial,
                                    setup.reasoning, markup, quotes,
                                    !interrupted, false, setup.track_spans);
  } catch (const TextGenerationError& error) {
    return Unsatisfied(error);
  }
  try {
    tools.Finish(generated.tool_calls, interrupted);
  } catch (const TextGenerationError&) {
    throw;
  } catch (const std::runtime_error&) {
    return Outcome::kUnsafeStream;
  }
  filter.Finish(generated.hide_tool_markup, generated.text,
                generated.reasoning_content, generated.text_spans);
  Final(generated, events);
  return Outcome::kComplete;
}

Outcome Parse(const Setup& setup, std::string_view bytes, bool enforce_required,
              std::vector<Event>* events) {
  const auto markup = SelectToolMarkup(setup.dialect);
  core::Utf8Decoder decoder;
  const auto decoded = decoder.Push(bytes, true);
  QuoteTracker quotes;
  ParsedGeneration generated;
  try {
    generated =
        setup.response_format || setup.constrained_tools
            ? ParseStructuredGeneration(
                  decoded, setup.initial, setup.reasoning, setup.tools,
                  setup.choice, enforce_required,
                  setup.constrained_tools && !setup.response_format, markup,
                  quotes, false, setup.track_spans)
            : ParseGeneration(decoded, setup.initial, setup.reasoning,
                              setup.tools, setup.choice, enforce_required,
                              markup, quotes, false, setup.track_spans);
  } catch (const TextGenerationError& error) {
    return Unsatisfied(error);
  }
  Final(generated, events);
  return Outcome::kComplete;
}

}  // namespace gufo::server::frozen
