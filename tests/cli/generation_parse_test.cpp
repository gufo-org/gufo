// Expected results of the generation parser (src/core/generation_parse,
// through the C ABI and src/cli/serve/generation_parse.hpp) over the corpus
// in tests/fixtures/generation_parse, with the real Qwen and DeepSeek
// scanners and tool grammar and a registered extension dialect.
//
// Each case's events at its recorded token boundaries, buffered, and from
// the whole-response parse must equal tests/fixtures/generation_parse/
// expected/<corpus>.json. Split one byte at a time and at two-piece splits,
// and under deterministic mutations (truncation, quoting, malformed UTF-8,
// Python literals), what a client assembles must not depend on the split
// except in the documented ways counted in the expected file, and a
// stream's final events must equal the whole-response parse.
//
// Usage: generation_parse_test CORPUS_FILE_OR_DIR [--exhaustive | --write]
// --exhaustive splits and truncates long responses at every byte too, not
// only at a stride, and does not check the split-dependence counts. --write
// rewrites the expected files from the current parser; review the diff.

#include "src/cli/serve/generation_parse.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/json_constraint.hpp"

namespace gufo::models::deepseek_v4_flash {
void RegisterToolDialect();
}

namespace {

using Initial = gufo::server::TextGenerationBackend::InitialOutputState;
using Choice = gufo::server::ChatRequest::ToolChoice;
using EventKind = gufo::server::GenerationEvent::Kind;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << "\n";
    std::exit(1);
  }
}

// An invented provider dialect, absent from ToolFormat and distinct from the
// Qwen, DeepSeek and Gemma framings, with non-JSON value framing:
// `<<invoke:NAME>>{key:«text»,n:1,o:{k:true},a:[1]}<</invoke>>`.
constexpr std::string_view kLedgerOpen = "<<invoke:";
constexpr std::string_view kLedgerClose = "<</invoke>>";

// Reads ledger values into JSON. kShort: the text ends inside the value.
class LedgerReader {
public:
  enum class Read { kOk, kBad, kShort };
  explicit LedgerReader(std::string_view text, std::size_t pos)
      : text_(text), pos_(pos) {}
  [[nodiscard]] std::size_t pos() const { return pos_; }
  [[nodiscard]] bool AtEnd() const { return pos_ >= text_.size(); }
  [[nodiscard]] char Peek() const { return text_[pos_]; }
  void Skip() { ++pos_; }
  [[nodiscard]] std::string_view Rest() const { return text_.substr(pos_); }

  // A string's text, or its prefix so far when it is still open.
  Read String(std::string* out) {
    constexpr std::string_view open = "«", close = "»";
    if (!Rest().starts_with(open))
      return Read::kBad;
    const auto end = text_.find(close, pos_ + open.size());
    *out = std::string(
        text_.substr(pos_ + open.size(), end == std::string_view::npos
                                             ? std::string_view::npos
                                             : end - pos_ - open.size()));
    if (end == std::string_view::npos)
      return Read::kShort;
    pos_ = end + close.size();
    return Read::kOk;
  }
  Read Key(std::string* out) {
    if (AtEnd())
      return Read::kShort;
    if (Rest().starts_with("«"))
      return String(out);
    const auto end = text_.find(':', pos_);
    if (end == std::string_view::npos)
      return Read::kShort;
    *out = std::string(text_.substr(pos_, end - pos_));
    if (out->empty() || out->find_first_of("{}[],") != std::string::npos)
      return Read::kBad;
    pos_ = end;
    return Read::kOk;
  }
  Read Value(gufo::json::Value* out, int depth = 0) {
    if (AtEnd())
      return Read::kShort;
    if (depth > 16)
      return Read::kBad;
    if (Rest().starts_with("«")) {
      std::string text;
      const auto read = String(&text);
      *out = gufo::json::Value(std::move(text));
      return read;
    }
    if (Peek() == '{' || Peek() == '[') {
      const bool object = Peek() == '{';
      const char end = object ? '}' : ']';
      *out = object ? gufo::json::Value::object() : gufo::json::Value::array();
      Skip();
      if (AtEnd())
        return Read::kShort;
      if (Peek() == end) {
        Skip();
        return Read::kOk;
      }
      while (true) {
        std::string key;
        if (object) {
          if (const auto read = Key(&key); read != Read::kOk)
            return read;
          if (AtEnd())
            return Read::kShort;
          if (Peek() != ':')
            return Read::kBad;
          Skip();
        }
        gufo::json::Value item;
        if (const auto read = Value(&item, depth + 1); read != Read::kOk)
          return read;
        if (object)
          (*out)[key] = std::move(item);
        else
          out->push_back(std::move(item));
        if (AtEnd())
          return Read::kShort;
        if (Peek() == end) {
          Skip();
          return Read::kOk;
        }
        if (Peek() != ',')
          return Read::kBad;
        Skip();
      }
    }
    // A scalar ends at the next structural byte.
    const auto end = text_.find_first_of(",}]", pos_);
    if (end == std::string_view::npos)
      return Read::kShort;
    try {
      *out = gufo::json::parse(text_.substr(pos_, end - pos_));
    } catch (...) {
      return Read::kBad;
    }
    if (out->is_string() || out->is_object() || out->is_array())
      return Read::kBad;
    pos_ = end;
    return Read::kOk;
  }

private:
  std::string_view text_;
  std::size_t pos_;
};

void ScanLedgerToolCalls(
    std::string_view text, std::span<const std::string_view> markers,
    std::vector<gufo::ext::ScannedToolCall>* calls,
    std::vector<std::pair<std::size_t, std::size_t>>* spans, bool partial) {
  Expect(markers.size() == 1 && markers.front() == kLedgerOpen,
         "the scanner receives its own markers");
  using Read = LedgerReader::Read;
  std::size_t cursor = 0;
  while ((cursor = text.find(kLedgerOpen, cursor)) != std::string_view::npos) {
    const auto begin = cursor;
    gufo::ext::ScannedToolCall call;
    const auto name_end = text.find(">>", begin + kLedgerOpen.size());
    if (name_end == std::string_view::npos)
      return;  // Nothing is known until the name ends.
    call.name = std::string(text.substr(begin + kLedgerOpen.size(),
                                        name_end - begin - kLedgerOpen.size()));
    LedgerReader reader(text, name_end + 2);
    // Top-level members are read one at a time so a partial call reports the
    // members already closed and a string still open.
    const auto members = [&]() -> Read {
      if (reader.AtEnd())
        return Read::kShort;
      if (reader.Peek() != '{')
        return Read::kBad;
      reader.Skip();
      if (!reader.AtEnd() && reader.Peek() == '}') {
        reader.Skip();
        return Read::kOk;
      }
      while (true) {
        std::string key;
        if (const auto read = reader.Key(&key); read != Read::kOk)
          return read;
        if (reader.AtEnd())
          return Read::kShort;
        if (reader.Peek() != ':')
          return Read::kBad;
        reader.Skip();
        gufo::json::Value value;
        const auto read = reader.Value(&value);
        if (read == Read::kShort && value.is_string() && partial)
          call.unfinished = gufo::ext::ScannedToolArgument{
              .name = key, .value = value.get_str(), .is_string = true};
        if (read != Read::kOk)
          return read;
        if (reader.AtEnd())
          return Read::kShort;
        if (reader.Peek() != ',' && reader.Peek() != '}')
          return Read::kBad;
        call.arguments.push_back(
            {.name = key,
             .value = value.is_string() ? value.get_str() : value.dump(),
             .is_string = value.is_string()});
        reader.Skip();
        if (text[reader.pos() - 1] == '}')
          return Read::kOk;
      }
    };
    auto read = members();
    if (read == Read::kOk) {
      const auto rest = reader.Rest();
      if (rest.starts_with(kLedgerClose)) {
        call.valid = call.complete = true;
        cursor = reader.pos() + kLedgerClose.size();
        if (spans)
          spans->emplace_back(begin, cursor);
        calls->push_back(std::move(call));
        continue;
      }
      read = kLedgerClose.starts_with(rest) ? Read::kShort : Read::kBad;
    }
    // Only a call later text could still complete is reported unfinished; a
    // malformed one is rejected outright.
    if (read == Read::kShort && partial)
      calls->push_back(std::move(call));
    return;
  }
}

gufo::sampling::JsonConstraint::ToolSyntax LedgerSyntax() {
  return {
      .open = std::string(kLedgerOpen),
      .name_suffix = ">>",
      .close = std::string(kLedgerClose),
      .name_delimiters = ">",
      .values = {.string_open = "«",
                 .string_close = "»",
                 .bare_keys = true,
                 .whitespace = false,
                 .sorted_keys = true},
      .reasoning_header = true,
  };
}

void RegisterLedgerToolDialect() {
  gufo::ext::RegisterToolDialect({
      .markers = {kLedgerOpen},
      .reasoning_markers = {kLedgerOpen},
      .closers = {kLedgerClose},
      .scan_tool_calls = &ScanLedgerToolCalls,
      .id = "ledger",
      .syntax = LedgerSyntax(),
  });
}

/// The request as the parser reads it.
struct Setup {
  std::optional<std::string> dialect;
  gufo::ext::ReasoningMarkup reasoning;
  Initial initial{Initial::kContent};
  bool response_format{false};
  bool constrained_tools{false};
  Choice choice{Choice::kAuto};
  std::vector<gufo::tokenization::ChatTool> tools;
  bool track_spans{false};
  bool buffered{false};
};

struct Case {
  std::string name;
  Setup setup;
  bool interrupted{false};
  std::vector<std::string> pieces;
};

/// One parser event, owning its text.
struct Event {
  EventKind kind{EventKind::kFinalEnd};
  std::string text;
  std::string name;
  std::string arguments;
  std::vector<gufo::server::TextSourceSpan> spans;
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

struct Run {
  Outcome outcome{Outcome::kComplete};
  std::vector<Event> events;
};

std::string Hex(std::string_view text) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string out;
  for (const unsigned char byte : text) {
    out += kDigits[byte >> 4];
    out += kDigits[byte & 15];
  }
  return out;
}

std::string Unhex(std::string_view hex) {
  const auto digit = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
  std::string out;
  for (std::size_t i = 0; i + 1 < hex.size(); i += 2)
    out += static_cast<char>(digit(hex[i]) * 16 + digit(hex[i + 1]));
  return out;
}

bool ValidUtf8(std::string_view text) {
  for (std::size_t i = 0; i < text.size();) {
    const auto byte = static_cast<unsigned char>(text[i]);
    std::size_t length = 1;
    std::uint32_t point = byte;
    if (byte >= 0xf0 && byte <= 0xf4) {
      length = 4;
      point = byte & 0x07;
    } else if (byte >= 0xe0) {
      length = byte <= 0xef ? 3 : 0;
      point = byte & 0x0f;
    } else if (byte >= 0xc2) {
      length = 2;
      point = byte & 0x1f;
    } else if (byte >= 0x80) {
      length = 0;
    }
    if (length == 0 || i + length > text.size())
      return false;
    for (std::size_t k = 1; k < length; ++k) {
      const auto next = static_cast<unsigned char>(text[i + k]);
      if ((next & 0xc0) != 0x80)
        return false;
      point = (point << 6) | (next & 0x3f);
    }
    if ((length == 3 &&
         (point < 0x800 || (point >= 0xd800 && point <= 0xdfff))) ||
        (length == 4 && (point < 0x10000 || point > 0x10ffff)))
      return false;
    i += length;
  }
  return true;
}

// Text as a JSON string, or {"hex": ...} when it is not UTF-8, as pieces are
// stored in the corpus.
gufo::json::Value Text(std::string_view text) {
  if (ValidUtf8(text))
    return gufo::json::Value(std::string(text));
  auto hex = gufo::json::Value::object();
  hex["hex"] = Hex(text);
  return hex;
}

std::string_view Name(EventKind kind) {
  static constexpr std::array<std::string_view, 7> kNames{
      "reasoning",     "content",    "tool",     "final_reasoning",
      "final_content", "final_call", "final_end"};
  return kNames[static_cast<std::size_t>(kind)];
}

std::string_view Name(Outcome outcome) {
  switch (outcome) {
    case Outcome::kComplete:
      return "complete";
    case Outcome::kNotProduced:
      return "tool call not produced";
    case Outcome::kNotCompleted:
      return "tool call not completed";
    case Outcome::kUnsafeStream:
      return "unsafe stream";
  }
  return "?";
}

// One event as the expected files spell it: [kind, fields..., spans].
// Reasoning and content: [kind, text]; tool deltas: ["tool", index, name,
// arguments, "begin"?, "complete"?]; final calls: ["final_call", index, name,
// arguments]; ["final_end", "hide" | "show"]. Source spans follow as
// [[begin, size], ...] when there are any.
gufo::json::Value Json(const Event& event) {
  auto out = gufo::json::Value::array();
  out.push_back(std::string(Name(event.kind)));
  switch (event.kind) {
    case EventKind::kToolDelta:
    case EventKind::kFinalCall:
      out.push_back(event.index);
      out.push_back(Text(event.name));
      out.push_back(Text(event.arguments));
      if (event.begin)
        out.push_back("begin");
      if (event.complete)
        out.push_back("complete");
      break;
    case EventKind::kFinalEnd:
      out.push_back(event.hide_tool_markup ? "hide" : "show");
      break;
    default:
      out.push_back(Text(event.text));
  }
  if (!event.spans.empty()) {
    auto spans = gufo::json::Value::array();
    for (const auto& span : event.spans) {
      auto pair = gufo::json::Value::array();
      pair.push_back(span.begin);
      pair.push_back(span.size);
      spans.push_back(std::move(pair));
    }
    out.push_back(std::move(spans));
  }
  return out;
}

// A run's events, then ["error", outcome] unless it completed.
gufo::json::Value Json(const Run& run) {
  auto out = gufo::json::Value::array();
  for (const auto& event : run.events)
    out.push_back(Json(event));
  if (run.outcome != Outcome::kComplete) {
    auto error = gufo::json::Value::array();
    error.push_back("error");
    error.push_back(std::string(Name(run.outcome)));
    out.push_back(std::move(error));
  }
  return out;
}

std::string Show(const Event& event) {
  return Json(event).dump();
}

gufo::server::GenerationSetup Production(const Setup& setup) {
  return {
      .initial = setup.initial,
      .reasoning = setup.reasoning,
      .markup = gufo::server::SelectGenerationMarkup(setup.dialect),
      .response_format = setup.response_format,
      .constrained_tools = setup.constrained_tools,
      .tool_choice = setup.choice,
      .tools = setup.tools,
      .track_spans = setup.track_spans,
      .buffered = setup.buffered,
  };
}

Outcome Unsatisfied(const gufo::server::TextGenerationError& error) {
  return std::string_view(error.what()).find("complete") !=
                 std::string_view::npos
             ? Outcome::kNotCompleted
             : Outcome::kNotProduced;
}

// Streams `pieces` as the Chat, Responses and Messages handlers do, then
// finishes with enforce_required = !interrupted.
Run Stream(const Setup& setup, const std::vector<std::string_view>& pieces,
           bool interrupted) {
  Run run;
  gufo::server::GenerationStream parser(Production(setup));
  const gufo::server::GenerationEventFn record =
      [&](const gufo::server::GenerationEvent& event) {
        run.events.push_back({
            .kind = event.kind,
            .text = std::string(event.text),
            .name = std::string(event.name),
            .arguments = std::string(event.arguments),
            .spans = {event.spans.begin(), event.spans.end()},
            .index = event.index,
            .begin = event.begin,
            .complete = event.complete,
            .hide_tool_markup = event.hide_tool_markup,
        });
        return true;
      };
  try {
    for (const auto piece : pieces)
      Expect(parser.Feed(piece, record), "the parser streams every piece");
    Expect(parser.Finish(!interrupted, interrupted, record),
           "the parser finishes");
  } catch (const gufo::server::TextGenerationError& error) {
    run.outcome = Unsatisfied(error);
  } catch (const std::runtime_error& error) {
    Expect(std::string_view(error.what()) ==
               "streamed tool call did not complete safely",
           error.what());
    run.outcome = Outcome::kUnsafeStream;
  }
  return run;
}

// Parses a whole response as the buffered Chat handler does, in the order
// of a stream's final events.
Run Parse(const Setup& setup, std::string_view text, bool enforce_required) {
  Run run;
  try {
    const auto parsed = gufo::server::ParseGeneratedText(
        Production(setup), text, enforce_required);
    run.events.push_back(
        {.kind = EventKind::kFinalReasoning, .text = parsed.reasoning});
    run.events.push_back({.kind = EventKind::kFinalContent,
                          .text = parsed.text,
                          .spans = parsed.spans});
    for (std::size_t index = 0; index < parsed.calls.size(); ++index)
      run.events.push_back({.kind = EventKind::kFinalCall,
                            .name = parsed.calls[index].name,
                            .arguments = parsed.calls[index].arguments,
                            .index = index});
    run.events.push_back({.kind = EventKind::kFinalEnd,
                          .hide_tool_markup = parsed.hide_tool_markup});
  } catch (const gufo::server::TextGenerationError& error) {
    run.outcome = Unsatisfied(error);
  }
  return run;
}

// What Assembled leaves out, for the explanations below.
struct Leave {
  bool reasoning_whitespace{false};
  bool content{false};
  std::set<std::size_t> tools;
};

// The content deltas joined.
std::string StreamedContent(const Run& run) {
  std::string content;
  for (const auto& event : run.events)
    if (event.kind == EventKind::kContentDelta)
      content += event.text;
  return content;
}

// What a client assembles from a stream: the joined deltas and the final
// events. Equal across splits of one response.
std::string Assembled(const Run& run, const Leave& leave = {}) {
  std::string reasoning;
  std::string content;
  struct Tool {
    std::string name;
    std::string arguments;
    bool complete{false};
  };
  std::map<std::size_t, Tool> tools;
  std::ostringstream out;
  out << Name(run.outcome) << "\n";
  for (const auto& event : run.events) {
    switch (event.kind) {
      case EventKind::kReasoningDelta:
        reasoning += event.text;
        break;
      case EventKind::kContentDelta:
        content += event.text;
        break;
      case EventKind::kToolDelta: {
        auto& tool = tools[event.index];
        if (event.begin)
          tool.name = event.name;
        tool.arguments += event.arguments;
        tool.complete = event.complete;
        break;
      }
      default: {
        auto copy = event;
        copy.spans.clear();
        out << Show(copy) << "\n";
      }
    }
  }
  if (leave.reasoning_whitespace)
    reasoning.erase(reasoning.find_last_not_of(" \t\r\n") + 1);
  out << "reasoning " << Text(reasoning).dump() << "\n";
  if (!leave.content)
    out << "content " << Text(content).dump() << "\n";
  for (const auto& [index, tool] : tools)
    if (!leave.tools.contains(index))
      out << "tool #" << index << " " << Text(tool.name).dump() << " "
          << Text(tool.arguments).dump() << (tool.complete ? " complete" : "")
          << "\n";
  return out.str();
}

// The final events of a stream, which must equal the whole-response parse.
Run Finals(const Run& run) {
  Run finals{.outcome = run.outcome};
  for (const auto& event : run.events)
    if (event.kind >= EventKind::kFinalReasoning)
      finals.events.push_back(event);
  return finals;
}

// Responses longer than this are split and truncated at every `Stride`-th
// byte rather than at all of them, unless --exhaustive is given.
constexpr std::size_t kExhaustiveBytes = 96;
bool exhaustive = false;

std::size_t Stride(std::size_t size) {
  return exhaustive || size <= kExhaustiveBytes ? 1 : size / 48;
}

struct Totals {
  std::size_t cases{0};
  std::size_t runs{0};
  std::vector<std::string> failures;
};

// Why a stream assembles differently from the whole response, when that is
// by design; nothing when it is unexplained.
std::optional<std::string> Explained(const Case& item, std::string_view variant,
                                     const Run& whole, const Run& run) {
  // Partial calls stream only from multi-piece input; one the final parse
  // rejects ends the stream with an error.
  if (run.outcome == Outcome::kUnsafeStream &&
      whole.outcome != Outcome::kUnsafeStream)
    return "a streamed call was retracted";
  // A stream that ends in an error is not a response; only the error must
  // agree.
  if (run.outcome != Outcome::kComplete && run.outcome == whole.outcome)
    return "failed as the whole response did";
  // An interrupted turn drops a call it had begun streaming: everything a
  // client assembled agrees once the calls missing from the final events
  // are left out.
  if (item.interrupted && run.outcome == Outcome::kComplete) {
    const auto dropped = [](const Run& stream) {
      std::set<std::size_t> finals;
      std::set<std::size_t> begun;
      for (const auto& event : stream.events) {
        if (event.kind == EventKind::kFinalCall)
          finals.insert(event.index);
        if (event.kind == EventKind::kToolDelta && event.begin)
          begun.insert(event.index);
      }
      Leave leave;
      std::ranges::set_difference(
          begun, finals, std::inserter(leave.tools, leave.tools.end()));
      return leave;
    };
    const auto leave = dropped(run);
    if (!leave.tools.empty() &&
        Assembled(run, leave) == Assembled(whole, dropped(whole)))
      return "an interrupted turn dropped a begun call";
  }
  // A marker inside a fence still open when it arrives switches the stream
  // to tool mode; the rest of the reasoning then streams from the final,
  // trimmed reasoning.
  if (Assembled(run, {.reasoning_whitespace = true}) ==
      Assembled(whole, {.reasoning_whitespace = true}))
    return "streamed reasoning lost trailing whitespace";
  // A provisional inline quote (the defect in issue-backtick.md): a backtick
  // run that ends a piece closes the span around the call markup before it,
  // which then streams as content; later backticks reopen the span and the
  // final parse accepts the call. Everything else agrees, the streamed
  // content is P + X + T with X running from the first call marker to the
  // last '>', and the whole stream's content is P + T': T and T' only
  // backticks.
  if (variant.starts_with("inline") &&
      Assembled(run, {.content = true}) ==
          Assembled(whole, {.content = true})) {
    const auto streamed = StreamedContent(run);
    const auto expected = StreamedContent(whole);
    const auto markup =
        gufo::server::SelectGenerationMarkup(item.setup.dialect);
    auto first = std::string::npos;
    for (const auto marker : markup.markers)
      first = std::min(first, streamed.find(marker));
    const auto last = streamed.rfind('>');
    const auto ticks = [](std::string_view text) {
      return text.find_first_not_of('`') == std::string_view::npos;
    };
    if (first != std::string::npos && last != std::string::npos &&
        first < last && expected.starts_with(streamed.substr(0, first)) &&
        ticks(std::string_view(expected).substr(first)) &&
        ticks(std::string_view(streamed).substr(last + 1)))
      return "a provisional inline quote streamed call markup";
  }
  return std::nullopt;
}

std::vector<std::string_view> Views(const std::vector<std::string>& pieces) {
  return {pieces.begin(), pieces.end()};
}

std::string Joined(const std::vector<std::string>& pieces) {
  std::string text;
  for (const auto& piece : pieces)
    text += piece;
  return text;
}

// A stream's final events equal the parse of the same bytes, unless the
// stream retracted a call it had streamed.
void SameFinals(const std::string& label, const Run& stream, const Run& parse,
                Totals& totals) {
  ++totals.runs;
  if (stream.outcome == Outcome::kUnsafeStream)
    return;
  const auto finals = Json(Finals(stream)).dump();
  const auto parsed = Json(parse).dump();
  if (finals != parsed)
    totals.failures.push_back(label + ": final events differ from the parse" +
                              "\n    stream " + finals + "\n    parse  " +
                              parsed);
}

using Counts = std::map<std::string, std::size_t>;

// Every split of one response: what a client assembles does not depend on
// the split except in the explained ways, counted in `explained`, and every
// stream's final events equal the parse.
void Splits(const Case& item, const std::string& prefix,
            const std::string& text, Counts& explained, Totals& totals) {
  const auto parse = Parse(item.setup, text, !item.interrupted);
  const auto whole =
      Stream(item.setup, {std::string_view(text)}, item.interrupted);
  SameFinals(item.name + " [" + prefix + "whole]", whole, parse, totals);
  const auto reference = Assembled(whole);
  std::vector<std::pair<std::string, std::vector<std::string_view>>> splits;
  if (prefix.empty())
    splits.emplace_back("recorded", Views(item.pieces));
  std::vector<std::string_view> bytes;
  for (std::size_t i = 0; i < text.size(); ++i)
    bytes.push_back(std::string_view(text).substr(i, 1));
  splits.emplace_back("bytes", std::move(bytes));
  for (std::size_t at = 1; at < text.size(); at += Stride(text.size())) {
    const std::string_view view(text);
    splits.emplace_back("split " + std::to_string(at),
                        std::vector{view.substr(0, at), view.substr(at)});
  }
  for (const auto& [name, pieces] : splits) {
    const auto label = item.name + " [" + prefix + name + "]";
    const auto run = Stream(item.setup, pieces, item.interrupted);
    SameFinals(label, run, parse, totals);
    const auto assembled = Assembled(run);
    if (assembled == reference)
      continue;
    if (const auto reason = Explained(item, prefix, whole, run)) {
      ++explained[*reason];
      continue;
    }
    totals.failures.push_back(label + ": split-dependent\n  whole:\n" +
                              reference + "  this split:\n" + assembled);
  }
}

std::string Replaced(std::string text, std::string_view from,
                     std::string_view to) {
  for (auto at = text.find(from); at != std::string::npos;
       at = text.find(from, at + to.size()))
    text.replace(at, from.size(), to);
  return text;
}

// Deterministic mutations of every case: truncation at every byte, the
// response quoted inline and in a fence, malformed UTF-8 in the middle, and
// Python literals for JSON ones.
void Mutations(const Case& item, const std::string& text, Counts& explained,
               Totals& totals) {
  auto truncated = item;
  truncated.interrupted = true;
  for (std::size_t at = 0; at < text.size(); at += Stride(text.size())) {
    const std::string_view prefix = std::string_view(text).substr(0, at);
    SameFinals(item.name + " [truncated " + std::to_string(at) + "]",
               Stream(truncated.setup, {prefix}, true),
               Parse(truncated.setup, prefix, false), totals);
  }
  const auto middle = text.size() / 2;
  const std::vector<std::pair<std::string, std::string>> variants{
      {"fenced", "```\n" + text + "\n```\n"},
      {"inline", "`" + text + "`"},
      {"malformed", text.substr(0, middle) + "\xff" + text.substr(middle)},
      {"split scalar",
       text.substr(0, middle) + "\xe2\x82" + text.substr(middle)},
      {"python literals",
       Replaced(Replaced(Replaced(text, "true", "True"), "false", "False"),
                "null", "None")},
  };
  for (const auto& [name, mutated] : variants)
    if (mutated != text)
      Splits(item, name + " ", mutated, explained, totals);
}

Initial ParseInitial(std::string_view name) {
  if (name == "auto")
    return Initial::kAuto;
  if (name == "reasoning")
    return Initial::kReasoning;
  if (name == "content")
    return Initial::kContent;
  Expect(name == "content_after_reasoning", "known initial output state");
  return Initial::kContentAfterReasoning;
}

Choice ParseChoice(std::string_view name) {
  if (name == "none")
    return Choice::kNone;
  if (name == "required")
    return Choice::kRequired;
  Expect(name == "auto", "known tool choice");
  return Choice::kAuto;
}

std::string Piece(const gufo::json::Value& value) {
  return value.is_string() ? value.get_str() : Unhex(value.member_str("hex"));
}

gufo::json::Value ReadJson(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  Expect(file.good(), "readable " + path.string());
  std::stringstream text;
  text << file.rdbuf();
  return gufo::json::parse(text.str());
}

std::vector<Case> Load(const std::filesystem::path& path) {
  const auto corpus = ReadJson(path);
  std::vector<Case> cases;
  for (const auto& entry : corpus.find("cases")->items()) {
    Case item;
    item.name = entry.member_str("name");
    auto& setup = item.setup;
    if (const auto* dialect = entry.find("dialect");
        dialect && dialect->is_string())
      setup.dialect = dialect->get_str();
    const auto& reasoning = entry.find("reasoning")->items();
    setup.reasoning = {reasoning.at(0).get_str(), reasoning.at(1).get_str()};
    setup.initial = ParseInitial(entry.member_str("initial"));
    setup.response_format = entry.find("response_format")->as_bool();
    setup.constrained_tools = entry.find("constrained_tools")->as_bool();
    setup.choice = ParseChoice(entry.member_str("tool_choice"));
    const auto* tools = entry.find("tools");
    if (tools->is_number())
      tools = &corpus.find("tool_sets")->items().at(tools->as_size());
    for (const auto& tool : tools->items())
      setup.tools.push_back(
          {.name = tool.member_str("name"),
           .parameters_json = tool.member_str("parameters_json")});
    setup.track_spans = entry.find("track_spans")->as_bool();
    setup.buffered = entry.find("buffered")->as_bool();
    item.interrupted = entry.find("interrupted")->as_bool();
    for (const auto& piece : entry.find("pieces")->items())
      item.pieces.push_back(Piece(piece));
    cases.push_back(std::move(item));
  }
  return cases;
}

// The expected results of one case. "stream": its events at the recorded
// boundaries; "buffered": the same with the pieces only buffered, absent
// for a buffered case; "parse": the whole-response parse; "split_dependence":
// how many splits of it and its mutations assemble differently from the
// whole response, by documented reason.
gufo::json::Value Results(const Case& item, Totals& totals) {
  ++totals.cases;
  const auto text = Joined(item.pieces);
  auto results = gufo::json::Value::object();
  results["name"] = item.name;
  results["stream"] =
      Json(Stream(item.setup, Views(item.pieces), item.interrupted));
  if (!item.setup.buffered) {
    auto buffered = item.setup;
    buffered.buffered = true;
    results["buffered"] =
        Json(Stream(buffered, Views(item.pieces), item.interrupted));
  }
  results["parse"] = Json(Parse(item.setup, text, !item.interrupted));
  Counts explained;
  Splits(item, "", text, explained, totals);
  Mutations(item, text, explained, totals);
  auto counts = gufo::json::Value::object();
  for (const auto& [reason, count] : explained)
    counts[reason] = count;
  results["split_dependence"] = std::move(counts);
  return results;
}

std::filesystem::path ExpectedPath(const std::filesystem::path& corpus) {
  return corpus.parent_path() / "expected" / corpus.filename();
}

// One case per line, members in a fixed order.
void WriteExpected(const std::filesystem::path& path,
                   const std::vector<gufo::json::Value>& results) {
  std::ofstream out(path, std::ios::binary);
  out << "{\"cases\": [\n";
  for (std::size_t i = 0; i < results.size(); ++i) {
    out << (i ? ",\n " : " ") << "{";
    bool first = true;
    for (const auto& [key, value] : results[i].members()) {
      out << (first ? "" : ",\n  ") << gufo::json::Value(key).dump() << ": "
          << value.dump();
      first = false;
    }
    out << "}";
  }
  out << "\n]}\n";
  Expect(out.good(), "wrote " + path.string());
}

void Check(const gufo::json::Value& results, const gufo::json::Value* expected,
           Totals& totals) {
  const auto name = results.member_str("name");
  if (!expected) {
    totals.failures.push_back(name + ": no expected results");
    return;
  }
  for (const auto& [key, value] : results.members()) {
    if (exhaustive && key == "split_dependence")
      continue;
    const auto* stored = expected->find(key);
    if (stored && stored->dump() == value.dump())
      continue;
    // The first differing event of a run, or the whole member.
    std::string want = stored ? stored->dump() : "(absent)";
    std::string got = value.dump();
    std::string where;
    if (stored && stored->is_array() && value.is_array()) {
      const auto& a = stored->items();
      const auto& b = value.items();
      std::size_t i = 0;
      while (i < a.size() && i < b.size() && a[i].dump() == b[i].dump())
        ++i;
      where = " event " + std::to_string(i);
      want = i < a.size() ? a[i].dump() : "(end)";
      got = i < b.size() ? b[i].dump() : "(end)";
    }
    totals.failures.push_back(name + " [" + key + "]" + where +
                              "\n    expected " + want + "\n    actual   " +
                              got);
  }
  if (expected->members().size() != results.members().size())
    totals.failures.push_back(name + ": expected results have other members");
}

}  // namespace

int main(int argc, char** argv) {
  Expect(argc >= 2,
         "usage: generation_parse_test CORPUS [--exhaustive | --write]");
  gufo::models::deepseek_v4_flash::RegisterToolDialect();
  RegisterLedgerToolDialect();
  const std::string_view mode = argc > 2 ? argv[2] : "";
  exhaustive = mode == "--exhaustive";
  const bool write = mode == "--write";
  std::vector<std::filesystem::path> files;
  if (std::filesystem::is_directory(argv[1])) {
    for (const auto& entry : std::filesystem::directory_iterator(argv[1]))
      if (entry.path().extension() == ".json")
        files.push_back(entry.path());
  } else {
    files.push_back(argv[1]);
  }
  std::ranges::sort(files);
  Expect(!files.empty(), "the corpus has files");
  Totals totals;
  for (const auto& file : files) {
    const auto cases = Load(file);
    Expect(!cases.empty(), "the corpus file has cases");
    std::vector<gufo::json::Value> results;
    for (const auto& item : cases)
      results.push_back(Results(item, totals));
    if (write) {
      WriteExpected(ExpectedPath(file), results);
      continue;
    }
    const auto expected = ReadJson(ExpectedPath(file));
    std::map<std::string, const gufo::json::Value*> by_name;
    for (const auto& entry : expected.find("cases")->items())
      by_name[entry.member_str("name")] = &entry;
    Expect(by_name.size() == cases.size(),
           "one expected entry per case in " + file.string());
    for (const auto& item : results) {
      const auto found = by_name.find(item.member_str("name"));
      Check(item, found == by_name.end() ? nullptr : found->second, totals);
    }
  }
  std::cout << "generation_parse_test: " << totals.cases << " cases, "
            << totals.runs << " runs, " << totals.failures.size()
            << " failures\n";
  for (std::size_t i = 0; i < std::min<std::size_t>(totals.failures.size(), 20);
       ++i)
    std::cerr << "FAILURE " << totals.failures[i] << "\n";
  return totals.failures.empty() ? 0 : 1;
}
