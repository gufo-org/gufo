#include "src/core/json_schema_lexeme.hpp"

#include <gufo/json_schema.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/json_number_limits.hpp"
#include "src/core/json_schema_call.hpp"
#include "src/core/json_schema_regex.hpp"

namespace gufo::sampling {
namespace {

class WhitespaceLexeme final : public JsonSchemaLexeme {
public:
  WhitespaceLexeme() : JsonSchemaLexeme(" \t\r\n") {}
  // Bound formatting work, never whitespace within a JSON string. Every JSON
  // value still has an admitted (compact) serialization.
  Match Check(std::string_view bytes) const override {
    std::string state;
    Match result{true, false};
    for (unsigned char byte : bytes) {
      result = Advance(state, byte);
      if (!result.prefix && !result.complete)
        return {};
    }
    return result;
  }
  Match Advance(std::string& state, unsigned char byte) const override {
    const auto count = state.empty() ? 0 : static_cast<unsigned char>(state[0]);
    if (count >= 32 || !AllowsByte(byte))
      return {};
    state.assign(1, static_cast<char>(count + 1));
    return {count + 1 < 32, true};
  }
  bool CacheTransitions() const override { return true; }
};

// The Rust compiler checks numbers exactly (include/gufo/json_schema.h): the
// bounds and multipleOf are decimal, never floating point, so 0.3 is a
// multiple of 0.1.
class NumberLexeme final : public JsonSchemaLexeme {
public:
  NumberLexeme(const json::Value& schema, bool integer)
      : JsonSchemaLexeme("0123456789-.") {
    const detail::JsonSchemaKeywords<5> keywords(
        schema, {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                 "multipleOf"});
    detail::JsonSchemaCall call;
    GufoJsonSchemaNumber* number = nullptr;
    const auto status = gufo_json_schema_number_new(
        keywords.data(), integer ? 1 : 0, kMaxJsonNumberDigits,
        detail::JsonSchemaCall::Reason, &call, &number);
    number_.reset(number);
    call.Check(status);
  }
  bool AcceptValue(const json::Value& value) const override {
    if (!value.is_number())
      return false;
    const auto text = value.dump();
    detail::JsonSchemaCall call;
    std::uint8_t accepted = 0;
    call.Check(gufo_json_schema_number_accept(
        number_.get(), reinterpret_cast<const std::uint8_t*>(text.data()),
        text.size(), &accepted, detail::JsonSchemaCall::Reason, &call));
    return accepted != 0;
  }
  Match Check(std::string_view bytes) const override {
    std::uint8_t found = 0;
    if (gufo_json_schema_number_check(
            number_.get(), reinterpret_cast<const std::uint8_t*>(bytes.data()),
            bytes.size(), &found) != GUFO_JSON_SCHEMA_COMPLETE)
      throw std::logic_error("JSON number matcher failed");
    return {(found & GUFO_JSON_SCHEMA_PREFIX) != 0,
            (found & GUFO_JSON_SCHEMA_MATCH) != 0};
  }

private:
  struct Free {
    void operator()(GufoJsonSchemaNumber* number) const {
      gufo_json_schema_number_free(number);
    }
  };
  std::unique_ptr<GufoJsonSchemaNumber, Free> number_;
};

class StringLexeme final : public JsonSchemaLexeme {
  enum Phase : std::uint32_t {
    Body,
    Escape,
    Hex,
    LowSlash,
    LowU,
    LowHex,
    Utf8
  };
  // Canonical request-local state. Completed characters do not retain their
  // spelling, so literal/escaped Unicode shares cached vocabulary transitions.
  enum Field { Dfa, Count, Value, Extra, Mode, Fields };
  using State = std::array<std::uint32_t, Fields>;

public:
  explicit StringLexeme(const json::Value& schema) {
    // Lengths, pattern and format, checked by the Rust compiler. Formats
    // follow JSON Schema's RFC-backed string formats. The grammar validates
    // decoded strings, so an escaped character has exactly the same meaning
    // as its UTF-8 spelling.
    const detail::JsonSchemaKeywords<4> keywords(
        schema, {"minLength", "maxLength", "pattern", "format"});
    detail::JsonSchemaCall call;
    call.Check(gufo_json_schema_string(keywords.data(), &minimum_, &maximum_,
                                       detail::JsonSchemaCall::Output,
                                       detail::JsonSchemaCall::Reason, &call));
    const auto& patterns = call.outputs;
    if (patterns.empty()) {
      scalar_only_ = true;
      static const auto unconstrained =
          JsonSchemaRegex::Compile({}, UINT32_MAX);
      regex_ = unconstrained;
    } else {
      regex_ = JsonSchemaRegex::Compile(patterns, maximum_);
    }
    if (!regex_->CanFinish(regex_->Start(), minimum_, maximum_))
      throw JsonSchemaEmpty(
          "JSON Schema: string predicates and lengths have no matching value");
  }

  bool CacheTransitions() const override { return true; }
  void CanonicalMaskState(std::string& encoded,
                          std::size_t token_bytes) const override {
    if (encoded.empty())
      return;
    State state{};
    if (encoded.size() != sizeof(state))
      throw std::logic_error("invalid JSON string matcher state");
    std::memcpy(state.data(), encoded.data(), sizeof(state));
    // A token can complete at most one Unicode character per byte. If even
    // its longest accepting suffix fits, the upper bound cannot affect its
    // mask. Only canonicalize the cache key; the request keeps its true count.
    const auto window = token_bytes + regex_->MaximumSuffix();
    if (scalar_only_ && state[Count] < minimum_ &&
        minimum_ - state[Count] > token_bytes) {
      // No vocabulary token can reach the minimum or close this string yet.
      // With no pattern, every valid Unicode continuation can still finish.
      state[Count] = 0;
    } else if (state[Count] >= minimum_ && maximum_ - state[Count] >= window) {
      state[Count] = minimum_;
    }
    encoded.assign(reinterpret_cast<const char*>(state.data()), sizeof(state));
  }
  Match Check(std::string_view bytes) const override {
    std::string state;
    Match result{true, false};
    for (unsigned char byte : bytes) {
      if (!result.prefix)
        return {};
      result = Advance(state, byte);
    }
    return result;
  }
  Match Advance(std::string& encoded, unsigned char byte) const override {
    State state{};
    if (encoded.empty()) {
      if (byte != '"')
        return {};
      state[Dfa] = regex_->Start();
    } else {
      if (encoded.size() != sizeof(state))
        throw std::logic_error("invalid JSON string matcher state");
      std::memcpy(state.data(), encoded.data(), sizeof(state));
      switch (state[Mode]) {
        case Body:
          if (byte == '"')
            return {false,
                    state[Count] >= minimum_ && regex_->Accepting(state[Dfa])};
          if (byte == '\\') {
            state[Mode] = Escape;
          } else if (byte < 0x20) {
            return {};
          } else if (byte < 0x80) {
            if (!Character(state, byte))
              return {};
          } else {
            const unsigned width = byte >= 0xc2 && byte <= 0xdf   ? 2
                                   : byte >= 0xe0 && byte <= 0xef ? 3
                                   : byte >= 0xf0 && byte <= 0xf4 ? 4
                                                                  : 0;
            if (!width)
              return {};
            state[Mode] = Utf8;
            if (scalar_only_) {
              const auto low = byte == 0xe0 ? 0xa0 : byte == 0xf0 ? 0x90 : 0x80;
              const auto high = byte == 0xed   ? 0x9f
                                : byte == 0xf4 ? 0x8f
                                               : 0xbf;
              state[Value] = (low << 8) | high;
              state[Extra] = width - 1;
            } else {
              state[Value] = byte & ((1U << (7 - width)) - 1);
              state[Extra] = (width << 8) | (width - 1);
            }
          }
          break;
        case Escape:
          if (byte == 'u') {
            state[Mode] = Hex;
            state[Extra] = 4;
          } else {
            const auto index = std::string_view("\"\\/bfnrt").find(byte);
            if (index == std::string_view::npos)
              return {};
            constexpr std::array<unsigned char, 8> values{
                '"', '\\', '/', '\b', '\f', '\n', '\r', '\t'};
            if (!Character(state, values[index]))
              return {};
          }
          break;
        case LowSlash:
          if (byte != '\\')
            return {};
          state[Mode] = LowU;
          break;
        case LowU:
          if (byte != 'u')
            return {};
          state[Mode] = LowHex;
          state[Extra] = (state[Value] << 8) | 4;
          state[Value] = 0;
          break;
        case Hex:
        case LowHex: {
          const int digit = byte >= '0' && byte <= '9'   ? byte - '0'
                            : byte >= 'a' && byte <= 'f' ? byte - 'a' + 10
                            : byte >= 'A' && byte <= 'F' ? byte - 'A' + 10
                                                         : -1;
          if (digit < 0)
            return {};
          if (scalar_only_) {
            const auto remaining = state[Extra] & 0xff;
            if (state[Mode] == LowHex) {
              if ((remaining == 4 && digit != 13) ||
                  (remaining == 3 && digit < 12))
                return {};
            } else if (remaining == 4) {
              state[Value] = digit == 13 ? 1 : 0;
            } else if (remaining == 3 && state[Value] == 1) {
              if (digit >= 12)
                return {};
              state[Value] = digit >= 8 ? 2 : 0;
            }
            --state[Extra];
            if (!(state[Extra] & 0xff)) {
              if (state[Mode] == Hex && state[Value] == 2) {
                state[Value] = 0xd800;
                state[Mode] = LowSlash;
              } else if (!Character(state, 0)) {
                return {};
              }
            }
            break;
          }
          state[Value] = (state[Value] << 4) | digit;
          --state[Extra];
          if (!(state[Extra] & 0xff)) {
            auto cp = state[Value];
            if (state[Mode] == LowHex) {
              if (cp < 0xdc00 || cp > 0xdfff)
                return {};
              cp =
                  0x10000 + ((state[Extra] >> 8) - 0xd800) * 1024 + cp - 0xdc00;
            } else if (cp >= 0xd800 && cp <= 0xdbff) {
              state[Mode] = LowSlash;
              state[Extra] = 0;
              break;
            }
            if (!Character(state, cp))
              return {};
          }
          break;
        }
        case Utf8:
          if (scalar_only_) {
            if (byte < (state[Value] >> 8) || byte > (state[Value] & 0xff))
              return {};
            if (!--state[Extra]) {
              if (!Character(state, 0))
                return {};
            } else {
              state[Value] = (0x80 << 8) | 0xbf;
            }
            break;
          }
          if (byte < 0x80 || byte > 0xbf)
            return {};
          state[Value] = (state[Value] << 6) | (byte & 0x3f);
          --state[Extra];
          if (!(state[Extra] & 0xff)) {
            const auto width = state[Extra] >> 8;
            const auto minimum = width == 2   ? 0x80U
                                 : width == 3 ? 0x800U
                                              : 0x10000U;
            if (state[Value] < minimum || !Character(state, state[Value]))
              return {};
          }
          break;
        default:
          throw std::logic_error("invalid JSON string matcher phase");
      }
    }
    if (state[Mode] != Body && !Pending(state))
      return {};
    encoded.assign(reinterpret_cast<const char*>(state.data()), sizeof(state));
    return {true, false};
  }

private:
  std::uint32_t RemainingMinimum(std::uint32_t count) const {
    return count >= minimum_ ? 0 : minimum_ - count;
  }
  bool Character(State& state, std::uint32_t cp) const {
    if (state[Count] == maximum_ || cp > 0x10ffff ||
        (cp >= 0xd800 && cp <= 0xdfff))
      return false;
    const auto count = state[Count] + 1;
    if (!scalar_only_) {
      const auto next = regex_->Advance(state[Dfa], cp);
      if (!regex_->CanFinish(next, RemainingMinimum(count), maximum_ - count))
        return false;
      state[Dfa] = next;
    }
    state[Count] = maximum_ == UINT32_MAX ? std::min(count, minimum_) : count;
    state[Value] = state[Extra] = 0;
    state[Mode] = Body;
    return true;
  }
  bool Range(const State& state, std::uint32_t first,
             std::uint32_t last) const {
    if (first > last || state[Count] == maximum_)
      return false;
    const auto count = state[Count] + 1;
    return regex_->CanAdvance(state[Dfa], first, last, RemainingMinimum(count),
                              maximum_ - count);
  }
  bool Pending(const State& state) const {
    if (scalar_only_)
      return state[Count] < maximum_;
    if (state[Mode] == Escape)
      return Range(state, 0, 0x10ffff);
    if (state[Mode] == Utf8) {
      const auto shift = 6 * (state[Extra] & 0xff);
      const auto width = state[Extra] >> 8;
      const auto minimum = width == 2 ? 0x80U : width == 3 ? 0x800U : 0x10000U;
      return Range(state, std::max(minimum, state[Value] << shift),
                   std::min(0x10ffffU, ((state[Value] + 1) << shift) - 1));
    }
    if (state[Mode] == LowSlash || state[Mode] == LowU) {
      const auto first = 0x10000 + (state[Value] - 0xd800) * 1024;
      return Range(state, first, first + 1023);
    }
    const auto shift = 4 * (state[Extra] & 0xff);
    const auto first = state[Value] << shift;
    const auto last = ((state[Value] + 1) << shift) - 1;
    if (state[Mode] == LowHex) {
      const auto low = std::max(first, 0xdc00U), high = std::min(last, 0xdfffU);
      if (low > high)
        return false;
      const auto base = 0x10000 + ((state[Extra] >> 8) - 0xd800) * 1024;
      return Range(state, base + low - 0xdc00, base + high - 0xdc00);
    }
    if (first <= 0xd7ff && Range(state, first, std::min(last, 0xd7ffU)))
      return true;
    if (last >= 0xe000 && Range(state, std::max(first, 0xe000U), last))
      return true;
    if (first <= 0xdbff && last >= 0xd800)
      return Range(state, 0x10000 + (std::max(first, 0xd800U) - 0xd800) * 1024,
                   0x10000 + (std::min(last, 0xdbffU) - 0xd800 + 1) * 1024 - 1);
    return false;
  }
  std::uint32_t minimum_{0}, maximum_{UINT32_MAX};
  bool scalar_only_{false};
  std::shared_ptr<const JsonSchemaRegex> regex_;
};

// Native tool arguments use raw UTF-8, not a JSON string literal. Reuse the
// exact Unicode/pattern/length matcher and its bounded canonical state, feeding
// escaped bytes only at this adapter boundary. Quotes and backslashes remain
// literal data. The enclosing grammar supplies the parameter delimiter.
class RawStringLexeme final : public JsonSchemaLexeme {
public:
  RawStringLexeme(const json::Value& schema, std::string delimiter)
      : string_(JsonSchemaLexeme::String(schema)),
        delimiter_(std::move(delimiter)),
        can_close_(delimiter_.size(), true) {
    for (std::size_t matched = 1; matched < delimiter_.size(); ++matched) {
      const auto suffix = delimiter_.substr(0, matched) + delimiter_;
      can_close_[matched] = suffix.find(delimiter_) == matched;
    }
  }
  bool CacheTransitions() const override { return true; }
  void CanonicalMaskState(std::string& state,
                          std::size_t token_bytes) const override {
    if (state.empty())
      return;
    auto inner = state.substr(1);
    string_->CanonicalMaskState(inner, token_bytes);
    state.replace(1, std::string::npos, inner);
  }
  Match Check(std::string_view bytes) const override {
    std::string state;
    Match result{true, string_->Check("\"\"").complete};
    for (const unsigned char byte : bytes) {
      result = Advance(state, byte);
      if (!result.prefix)
        break;
    }
    return result;
  }
  Match Advance(std::string& encoded, unsigned char byte) const override {
    auto state = encoded.empty() ? std::string() : encoded.substr(1);
    std::string suffix(delimiter_.substr(
        0, encoded.empty() ? 0 : static_cast<unsigned char>(encoded.front())));
    suffix += static_cast<char>(byte);
    auto matched = std::min(suffix.size(), delimiter_.size());
    while (matched && !suffix.ends_with(delimiter_.substr(0, matched)))
      --matched;
    if (matched == delimiter_.size())
      return {};
    if (state.empty())
      string_->Advance(state, '"');
    Match result;
    if (byte == '"' || byte == '\\') {
      result = string_->Advance(state, '\\');
      if (result.prefix)
        result = string_->Advance(state, byte);
    } else if (byte < 0x20) {
      constexpr char hex[] = "0123456789abcdef";
      for (const unsigned char part :
           std::string{'\\', 'u', '0', '0', hex[byte >> 4], hex[byte & 15]}) {
        result = string_->Advance(state, part);
        if (!result.prefix)
          return {};
      }
    } else {
      result = string_->Advance(state, byte);
    }
    if (!result.prefix)
      return {};
    auto closing = state;
    encoded.assign(1, static_cast<char>(matched));
    encoded += state;
    // The first delimiter must begin after the value, not inside a trailing
    // prefix. For "\n</parameter>\n", a value ending in "\n</parameter>"
    // would otherwise admit duplicated closers and disagree with the parser.
    return {true,
            can_close_[matched] && string_->Advance(closing, '"').complete};
  }

private:
  std::shared_ptr<const JsonSchemaLexeme> string_;
  std::string delimiter_;
  std::vector<bool> can_close_;
};

// Native calls can start before the reasoning close delimiter. Keep literal
// marker mentions and Markdown code quoted; a complete native function header
// starts constrained tool decoding. Only a small scan state belongs to each
// speculative branch.
class ReasoningLexeme final : public JsonSchemaLexeme {
public:
  ReasoningLexeme(std::string tool_prefix, ReasoningEnd end, std::string close)
      : tool_prefix_(std::move(tool_prefix)),
        close_(std::move(close)),
        end_(end),
        tool_failure_(Failure(tool_prefix_)),
        close_failure_(Failure(close_)) {}
  bool CacheTransitions() const override { return true; }
  bool PlainReasoning(std::string_view encoded) const override {
    Scan state{};
    if (!encoded.empty())
      std::memcpy(&state, encoded.data(), sizeof(state));
    return state.explicit_prefix == 0 && state.tool_prefix == 0;
  }
  Match Check(std::string_view bytes) const override {
    std::string state;
    Match result{true, end_ == ReasoningEnd::kUnfinished};
    for (const unsigned char byte : bytes) {
      result = Advance(state, byte);
      if (!result.prefix)
        break;
    }
    return result;
  }
  Match Advance(std::string& encoded, unsigned char byte) const override {
    Scan state{};
    if (!encoded.empty())
      std::memcpy(&state, encoded.data(), sizeof(state));
    const bool line_start = !state.nonblank;
    if (byte != ' ' && byte != '\t' && byte != '\r' && byte != '\n')
      state.nonblank = 1;
    if (state.run && (byte != state.run_char)) {
      if (state.fence) {
        if (state.run_char == state.fence_char && state.run >= state.fence &&
            state.run_at_start)
          state.closing_fence = 1;
      } else if (!state.inline_quote && state.run >= 3 && state.run_at_start) {
        state.fence = state.run;
        state.fence_char = state.run_char;
      } else if (state.run_char == '`') {
        if (!state.inline_quote)
          state.inline_quote = state.run;
        else if (state.run == state.inline_quote)
          state.inline_quote = 0;
      }
      state.run = 0;
      state.run_char = state.run_at_start = 0;
    }
    if (byte == '`' || byte == '~') {
      if (!state.run) {
        state.run_char = byte;
        state.run_at_start = line_start;
      }
      if (state.run < UINT32_MAX)
        ++state.run;
    } else if (byte == '\n') {
      if (state.fence && state.closing_fence)
        state.fence = state.fence_char = 0;
      if (state.inline_quote && !state.nonblank)
        state.inline_quote = 0;
      state.closing_fence = state.nonblank = 0;
    } else if (byte != ' ' && byte != '\t' && byte != '\r') {
      state.closing_fence = 0;
    }
    const auto advance = [&](std::uint32_t prefix, std::string_view marker,
                             const auto& failure) {
      // Preserve overlapping marker prefixes without allocating per byte.
      while (prefix && byte != static_cast<unsigned char>(marker[prefix]))
        prefix = failure[prefix - 1];
      if (byte == static_cast<unsigned char>(marker[prefix]))
        return prefix + 1;
      return std::uint32_t{0};
    };
    state.explicit_prefix =
        advance(state.explicit_prefix, close_, close_failure_);
    state.tool_prefix =
        state.inline_quote || state.fence
            ? 0
            : advance(state.tool_prefix, tool_prefix_, tool_failure_);
    if (state.explicit_prefix == close_.size())
      return {false, end_ == ReasoningEnd::kExplicit};
    if (state.tool_prefix == tool_prefix_.size())
      return {false, end_ == ReasoningEnd::kTool};
    const Scan empty{};
    if (std::memcmp(&state, &empty, sizeof(state)) == 0)
      encoded.clear();
    else
      encoded.assign(reinterpret_cast<const char*>(&state), sizeof(state));
    return {true, end_ == ReasoningEnd::kUnfinished};
  }

private:
  struct Scan {
    std::uint32_t explicit_prefix{}, tool_prefix{}, inline_quote{}, fence{};
    std::uint32_t run{}, run_char{}, run_at_start{}, fence_char{};
    std::uint32_t nonblank{}, closing_fence{};
  };
  // Knuth-Morris-Pratt failure links, so overlapping marker prefixes survive.
  static std::vector<std::uint32_t> Failure(std::string_view marker) {
    std::vector<std::uint32_t> failure(marker.size());
    for (std::size_t i = 1, matched = 0; i < marker.size(); ++i) {
      while (matched && marker[i] != marker[matched])
        matched = failure[matched - 1];
      if (marker[i] == marker[matched])
        ++matched;
      failure[i] = static_cast<std::uint32_t>(matched);
    }
    return failure;
  }
  std::string tool_prefix_;
  std::string close_;
  ReasoningEnd end_;
  std::vector<std::uint32_t> tool_failure_;
  std::vector<std::uint32_t> close_failure_;
};

}  // namespace

std::shared_ptr<const JsonSchemaLexeme> JsonSchemaLexeme::String(
    const json::Value& schema) {
  return std::make_shared<StringLexeme>(schema);
}
std::shared_ptr<const JsonSchemaLexeme> JsonSchemaLexeme::RawString(
    const json::Value& schema, std::string delimiter) {
  return std::make_shared<RawStringLexeme>(schema, std::move(delimiter));
}
std::shared_ptr<const JsonSchemaLexeme> JsonSchemaLexeme::Whitespace() {
  static const auto whitespace = std::make_shared<WhitespaceLexeme>();
  return whitespace;
}
std::shared_ptr<const JsonSchemaLexeme> JsonSchemaLexeme::Reasoning(
    std::string tool_prefix, ReasoningEnd end, std::string close) {
  return std::make_shared<ReasoningLexeme>(std::move(tool_prefix), end,
                                           std::move(close));
}

json::Value JsonSchemaLexeme::IntersectMultipleOf(const json::Value& a,
                                                  const json::Value& b) {
  const detail::JsonSchemaKeywords<2> keywords({&a, &b});
  detail::JsonSchemaCall call;
  call.Check(gufo_json_schema_multiple_of(
      keywords.data(), keywords.data() + 1, detail::JsonSchemaCall::Output,
      detail::JsonSchemaCall::Reason, &call));
  const auto& common = call.outputs.at(0);
  const auto result = json::parse(common);
  // A double must hold the exact least common multiple.
  std::uint8_t equal = 0;
  if (result.is_number() && std::isfinite(result.as_double())) {
    const auto text = result.dump();
    call.Check(gufo_json_schema_number_equal(
        reinterpret_cast<const std::uint8_t*>(text.data()), text.size(),
        reinterpret_cast<const std::uint8_t*>(common.data()), common.size(),
        &equal, detail::JsonSchemaCall::Reason, &call));
  }
  if (!equal)
    throw std::invalid_argument(
        "JSON Schema: combined multipleOf exceeds the exact schema-number "
        "range");
  return result;
}

json::Value JsonSchemaLexeme::Format(std::string_view format) {
  detail::JsonSchemaCall call;
  std::uint32_t cap = UINT32_MAX;
  call.Check(gufo_json_schema_format(
      reinterpret_cast<const std::uint8_t*>(format.data()), format.size(), &cap,
      detail::JsonSchemaCall::Output, detail::JsonSchemaCall::Reason, &call));
  auto schema = json::Value::object();
  schema["pattern"] = call.outputs.at(0);
  if (cap != UINT32_MAX)
    schema["maxLength"] = static_cast<int>(cap);
  return schema;
}
std::shared_ptr<const JsonSchemaLexeme> JsonSchemaLexeme::Number(
    const json::Value& schema, bool integer) {
  return std::make_shared<NumberLexeme>(schema, integer);
}
}  // namespace gufo::sampling
