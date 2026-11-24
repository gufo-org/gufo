// Compares the Rust generation parser (src/core/generation_parse, through
// src/cli/serve/generation_parse.hpp) with a frozen copy of the C++ parser it
// replaced, event by event, over the corpus in tests/fixtures/generation_parse:
// each case whole, at its recorded token boundaries, one byte at a time and at
// every two-piece split, buffered, and under deterministic mutations. Both use
// the real Qwen and DeepSeek scanners and tool grammar.
//
// Usage: generation_parse_test CORPUS_FILE_OR_DIR [--bench | --exhaustive]
// --bench replays every case at its recorded boundaries and prints the cost
// per token (p50 and p99) of each parser. --exhaustive splits and truncates
// long responses at every byte too, not only at a stride.

#include "src/cli/serve/generation_parse.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
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
#include "tests/cli/generation_parse_frozen.hpp"

namespace gufo::models::deepseek_v4_flash {
void RegisterToolDialect();
}

namespace {

using gufo::server::frozen::Event;
using gufo::server::frozen::EventKind;
using gufo::server::frozen::Outcome;
using gufo::server::frozen::Setup;
using Initial = gufo::server::TextGenerationBackend::InitialOutputState;
using Choice = gufo::server::ChatRequest::ToolChoice;

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
struct Case {
  std::string name;
  Setup setup;
  bool interrupted{false};
  std::vector<std::string> pieces;
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

std::string Show(std::string_view text) {
  std::string out = "\"";
  for (const unsigned char byte : text) {
    if (byte == '\n')
      out += "\\n";
    else if (byte == '"' || byte == '\\')
      out += {'\\', static_cast<char>(byte)};
    else if (byte < 0x20 || byte == 0x7f)
      out += "\\x" + Hex(std::string(1, static_cast<char>(byte)));
    else
      out += static_cast<char>(byte);
  }
  return out + "\"";
}

std::string Show(const Event& event) {
  static constexpr std::array<std::string_view, 7> kNames{
      "reasoning",     "content",    "tool",     "final_reasoning",
      "final_content", "final_call", "final_end"};
  std::ostringstream out;
  out << kNames[static_cast<std::size_t>(event.kind)] << " "
      << Show(event.text);
  if (event.kind == EventKind::kToolDelta ||
      event.kind == EventKind::kFinalCall)
    out << " #" << event.index << " " << Show(event.name) << " "
        << Show(event.arguments) << (event.begin ? " begin" : "")
        << (event.complete ? " complete" : "");
  if (event.kind == EventKind::kFinalEnd)
    out << (event.hide_tool_markup ? " hide" : " show");
  for (const auto& span : event.spans)
    out << " [" << span.begin << "+" << span.size << "]";
  return out.str();
}

std::string Show(Outcome outcome) {
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

bool Same(const Event& a, const Event& b) {
  return a.kind == b.kind && a.text == b.text && a.name == b.name &&
         a.arguments == b.arguments && a.index == b.index &&
         a.begin == b.begin && a.complete == b.complete &&
         a.hide_tool_markup == b.hide_tool_markup &&
         a.spans.size() == b.spans.size() &&
         std::equal(a.spans.begin(), a.spans.end(), b.spans.begin(),
                    [](const auto& x, const auto& y) {
                      return x.begin == y.begin && x.size == y.size;
                    });
}

// The first difference between two runs, or nothing.
std::optional<std::string> Difference(const Run& expected, const Run& actual) {
  if (expected.outcome != actual.outcome)
    return "outcome: frozen " + Show(expected.outcome) + ", rust " +
           Show(actual.outcome);
  const auto count = std::min(expected.events.size(), actual.events.size());
  for (std::size_t i = 0; i < count; ++i)
    if (!Same(expected.events[i], actual.events[i]))
      return "event " + std::to_string(i) + ": frozen " +
             Show(expected.events[i]) + "\n    rust   " +
             Show(actual.events[i]);
  if (expected.events.size() != actual.events.size()) {
    const auto& longer = expected.events.size() > count ? expected : actual;
    return "event count: frozen " + std::to_string(expected.events.size()) +
           ", rust " + std::to_string(actual.events.size()) + "; next " +
           Show(longer.events[count]);
  }
  return std::nullopt;
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

Run RustStream(const Setup& setup, const std::vector<std::string_view>& pieces,
               bool interrupted,
               std::vector<std::uint64_t>* piece_ns = nullptr) {
  Run run;
  gufo::server::GenerationStream parser(Production(setup));
  const gufo::server::GenerationEventFn record =
      [&](const gufo::server::GenerationEvent& event) {
        run.events.push_back({
            .kind = static_cast<EventKind>(event.kind),
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
    for (const auto piece : pieces) {
      const auto started = std::chrono::steady_clock::now();
      Expect(parser.Feed(piece, record), "the parser streams every piece");
      if (piece_ns)
        piece_ns->push_back(static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started)
                .count()));
    }
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

Run RustParse(const Setup& setup, std::string_view text,
              bool enforce_required) {
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

Run FrozenStream(const Setup& setup,
                 const std::vector<std::string_view>& pieces, bool interrupted,
                 std::vector<std::uint64_t>* piece_ns = nullptr) {
  Run run;
  run.outcome = gufo::server::frozen::Stream(setup, pieces, interrupted,
                                             &run.events, piece_ns);
  return run;
}

Run FrozenParse(const Setup& setup, std::string_view text,
                bool enforce_required) {
  Run run;
  run.outcome =
      gufo::server::frozen::Parse(setup, text, enforce_required, &run.events);
  return run;
}

// What a client assembles from a stream: the joined deltas and the final
// events. Equal across splits of one response.
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
  out << Show(run.outcome) << "\n";
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
  out << "reasoning " << Show(reasoning) << "\n";
  if (!leave.content)
    out << "content " << Show(content) << "\n";
  for (const auto& [index, tool] : tools)
    if (!leave.tools.contains(index))
      out << "tool #" << index << " " << Show(tool.name) << " "
          << Show(tool.arguments) << (tool.complete ? " complete" : "") << "\n";
  return out.str();
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
  std::size_t mutations{0};
  std::vector<std::string> failures;
  std::vector<std::string> split_dependent;
  std::map<std::string, std::size_t> explained;
};

// The final events, without spans.
std::string Finals(const Run& run) {
  std::string out;
  for (auto event : run.events)
    if (event.kind >= EventKind::kFinalReasoning) {
      event.spans.clear();
      out += Show(event) + "\n";
    }
  return out;
}

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

void Compare(const std::string& label, const Run& frozen, const Run& rust,
             Totals& totals) {
  ++totals.runs;
  if (const auto difference = Difference(frozen, rust))
    totals.failures.push_back(label + "\n    " + *difference);
}

// Streams `pieces` through both parsers and returns the frozen run.
Run Streams(const Case& item, const std::string& variant,
            const std::vector<std::string_view>& pieces, Totals& totals) {
  auto frozen = FrozenStream(item.setup, pieces, item.interrupted);
  const auto rust = RustStream(item.setup, pieces, item.interrupted);
  Compare(item.name + " [" + variant + "]", frozen, rust, totals);
  return frozen;
}

void Parses(const Case& item, const std::string& variant, std::string_view text,
            Totals& totals) {
  Compare(item.name + " [" + variant + "]",
          FrozenParse(item.setup, text, !item.interrupted),
          RustParse(item.setup, text, !item.interrupted), totals);
}

// Every split of one response: the parsers agree at each, and what a client
// assembles does not depend on the split.
void Splits(const Case& item, const std::string& prefix,
            const std::string& text, const std::vector<std::string>* recorded,
            Totals& totals) {
  Parses(item, prefix + "buffered parse", text, totals);
  const auto whole =
      Streams(item, prefix + "whole", {std::string_view(text)}, totals);
  const auto reference = Assembled(whole);
  std::vector<std::pair<std::string, Run>> runs;
  if (recorded)
    runs.emplace_back(prefix + "recorded", Streams(item, prefix + "recorded",
                                                   Views(*recorded), totals));
  std::vector<std::string_view> bytes;
  for (std::size_t i = 0; i < text.size(); ++i)
    bytes.push_back(std::string_view(text).substr(i, 1));
  runs.emplace_back(prefix + "bytes",
                    Streams(item, prefix + "bytes", bytes, totals));
  for (std::size_t at = 1; at < text.size(); at += Stride(text.size())) {
    const std::string_view view(text);
    const auto label = prefix + "split " + std::to_string(at);
    runs.emplace_back(
        label,
        Streams(item, label, {view.substr(0, at), view.substr(at)}, totals));
  }
  for (const auto& [label, run] : runs) {
    const auto assembled = Assembled(run);
    if (assembled == reference)
      continue;
    if (const auto reason = Explained(item, prefix, whole, run)) {
      ++totals.explained[*reason];
      continue;
    }
    totals.split_dependent.push_back(item.name + " [" + label +
                                     "]\n  whole:\n" + reference +
                                     "  this split:\n" + assembled);
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
void Mutations(const Case& item, const std::string& text, Totals& totals) {
  auto truncated = item;
  truncated.interrupted = true;
  for (std::size_t at = 0; at < text.size(); at += Stride(text.size())) {
    const std::string_view prefix = std::string_view(text).substr(0, at);
    const auto label = "truncated " + std::to_string(at);
    Parses(truncated, label, prefix, totals);
    Streams(truncated, label, {prefix}, totals);
    ++totals.mutations;
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
  for (const auto& [name, mutated] : variants) {
    if (mutated == text)
      continue;
    ++totals.mutations;
    if (exhaustive) {
      Splits(item, name + " ", mutated, nullptr, totals);
      continue;
    }
    Parses(item, name + " buffered parse", mutated, totals);
    Streams(item, name + " whole", {std::string_view(mutated)}, totals);
    std::vector<std::string_view> bytes;
    for (std::size_t i = 0; i < mutated.size(); ++i)
      bytes.push_back(std::string_view(mutated).substr(i, 1));
    Streams(item, name + " bytes", bytes, totals);
  }
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

std::vector<Case> Load(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  std::stringstream text;
  text << file.rdbuf();
  const auto corpus = gufo::json::parse(text.str());
  std::vector<Case> cases;
  for (const auto& entry : corpus.find("cases")->items()) {
    Case item;
    item.name = path.stem().string() + "/" + entry.member_str("name");
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

std::uint64_t Percentile(std::vector<std::uint64_t> values, double fraction) {
  if (values.empty())
    return 0;
  std::ranges::sort(values);
  const auto index = static_cast<std::size_t>(
      fraction * static_cast<double>(values.size() - 1) + 0.5);
  return values[index];
}

// Replays every case at its recorded boundaries, best of several rounds per
// piece, and prints the cost per token of each parser.
void Bench(const std::vector<Case>& cases) {
  constexpr int kRounds = 7;
  std::vector<std::uint64_t> frozen_ns;
  std::vector<std::uint64_t> rust_ns;
  for (const auto& item : cases) {
    const auto pieces = Views(item.pieces);
    std::vector<std::uint64_t> frozen_best(pieces.size(), UINT64_MAX);
    std::vector<std::uint64_t> rust_best(pieces.size(), UINT64_MAX);
    for (int round = 0; round < kRounds; ++round) {
      std::vector<std::uint64_t> frozen;
      std::vector<std::uint64_t> rust;
      FrozenStream(item.setup, pieces, item.interrupted, &frozen);
      RustStream(item.setup, pieces, item.interrupted, &rust);
      for (std::size_t i = 0; i < pieces.size(); ++i) {
        frozen_best[i] = std::min(frozen_best[i], frozen[i]);
        rust_best[i] = std::min(rust_best[i], rust[i]);
      }
    }
    frozen_ns.insert(frozen_ns.end(), frozen_best.begin(), frozen_best.end());
    rust_ns.insert(rust_ns.end(), rust_best.begin(), rust_best.end());
  }
  std::cout << "tokens " << frozen_ns.size() << "\n"
            << "frozen ns/token p50 " << Percentile(frozen_ns, 0.5) << " p99 "
            << Percentile(frozen_ns, 0.99) << "\n"
            << "rust   ns/token p50 " << Percentile(rust_ns, 0.5) << " p99 "
            << Percentile(rust_ns, 0.99) << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  Expect(argc >= 2,
         "usage: generation_parse_test CORPUS [--bench | --exhaustive]");
  gufo::models::deepseek_v4_flash::RegisterToolDialect();
  RegisterLedgerToolDialect();
  std::vector<std::filesystem::path> files;
  if (std::filesystem::is_directory(argv[1])) {
    for (const auto& entry : std::filesystem::directory_iterator(argv[1]))
      if (entry.path().extension() == ".json")
        files.push_back(entry.path());
  } else {
    files.push_back(argv[1]);
  }
  std::ranges::sort(files);
  std::vector<Case> cases;
  for (const auto& file : files)
    for (auto& item : Load(file))
      cases.push_back(std::move(item));
  Expect(!cases.empty(), "the corpus has cases");
  if (argc > 2 && std::string_view(argv[2]) == "--bench") {
    Bench(cases);
    return 0;
  }
  exhaustive = argc > 2 && std::string_view(argv[2]) == "--exhaustive";
  Totals totals;
  for (const auto& item : cases) {
    ++totals.cases;
    const auto text = Joined(item.pieces);
    Splits(item, "", text, &item.pieces, totals);
    auto buffered = item;
    buffered.setup.buffered = true;
    Streams(buffered, "buffered", Views(item.pieces), totals);
    Mutations(item, text, totals);
  }
  std::cout << "generation_parse_test: " << totals.cases << " cases, "
            << totals.mutations << " mutations, " << totals.runs
            << " comparisons, " << totals.failures.size() << " differences, "
            << totals.split_dependent.size()
            << " unexplained split-dependent runs\n";
  for (const auto& [reason, count] : totals.explained)
    std::cout << "  explained split dependence: " << count << " " << reason
              << "\n";
  for (std::size_t i = 0; i < std::min<std::size_t>(totals.failures.size(), 20);
       ++i)
    std::cerr << "DIFFERENCE " << totals.failures[i] << "\n";
  for (std::size_t i = 0;
       i < std::min<std::size_t>(totals.split_dependent.size(), 20); ++i)
    std::cerr << "SPLIT-DEPENDENT " << totals.split_dependent[i] << "\n";
  return totals.failures.empty() && totals.split_dependent.empty() ? 0 : 1;
}
