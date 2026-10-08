#include "src/models/gemma4/tool_syntax.hpp"

#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <utility>

namespace gufo::models::gemma4 {
namespace {

constexpr std::string_view kQuote = "<|\"|>";
constexpr std::size_t kMaxDepth = 64;

struct Parser {
  std::string_view text;
  std::size_t pos{0};
  std::string error;

  bool Fail(std::string message) {
    if (error.empty()) {
      error = std::move(message) + " at offset " + std::to_string(pos);
    }
    return false;
  }

  void SkipSpace() {
    while (pos < text.size() &&
           std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
      ++pos;
    }
  }

  bool Consume(std::string_view token) {
    if (text.substr(pos, token.size()) == token) {
      pos += token.size();
      return true;
    }
    return false;
  }

  bool QuotedString(std::string* out) {
    if (!Consume(kQuote)) {
      return Fail("expected string delimiter");
    }
    const std::size_t end = text.find(kQuote, pos);
    if (end == std::string_view::npos) {
      return Fail("unterminated string");
    }
    *out = std::string(text.substr(pos, end - pos));
    pos = end + kQuote.size();
    return true;
  }

  bool Key(std::string* out) {
    SkipSpace();
    if (text.substr(pos, kQuote.size()) == kQuote) {
      return QuotedString(out);
    }
    const std::size_t colon = text.find(':', pos);
    if (colon == std::string_view::npos) {
      return Fail("expected ':' after key");
    }
    std::string_view key = text.substr(pos, colon - pos);
    while (!key.empty() &&
           std::isspace(static_cast<unsigned char>(key.back())) != 0) {
      key.remove_suffix(1);
    }
    if (key.empty() || key.find_first_of("{}[],") != std::string_view::npos) {
      return Fail("invalid key");
    }
    *out = std::string(key);
    pos = colon;
    return true;
  }

  /// A bare scalar ends at the next structural delimiter.
  json::Value Bare() {
    const std::size_t start = pos;
    while (pos < text.size() && text[pos] != ',' && text[pos] != '}' &&
           text[pos] != ']') {
      ++pos;
    }
    std::string_view raw = text.substr(start, pos - start);
    while (!raw.empty() &&
           std::isspace(static_cast<unsigned char>(raw.back())) != 0) {
      raw.remove_suffix(1);
    }
    if (raw == "true") {
      return json::Value(true);
    }
    if (raw == "false") {
      return json::Value(false);
    }
    if (raw == "null") {
      return json::Value(nullptr);
    }
    double number = 0.0;
    const auto [end, ec] =
        std::from_chars(raw.data(), raw.data() + raw.size(), number);
    if (ec == std::errc{} && end == raw.data() + raw.size() && !raw.empty() &&
        std::isfinite(number)) {
      return json::Value(number);
    }
    return json::Value(std::string(raw));
  }

  bool Value(json::Value* out, std::size_t depth) {
    if (depth > kMaxDepth) {
      return Fail("arguments nest too deeply");
    }
    SkipSpace();
    if (pos >= text.size()) {
      return Fail("expected a value");
    }
    if (text.substr(pos, kQuote.size()) == kQuote) {
      std::string s;
      if (!QuotedString(&s)) {
        return false;
      }
      *out = json::Value(std::move(s));
      return true;
    }
    if (text[pos] == '{') {
      return Object(out, depth + 1);
    }
    if (text[pos] == '[') {
      ++pos;
      *out = json::Value::array();
      SkipSpace();
      if (Consume("]")) {
        return true;
      }
      while (true) {
        json::Value item;
        if (!Value(&item, depth + 1)) {
          return false;
        }
        out->push_back(std::move(item));
        SkipSpace();
        if (Consume("]")) {
          return true;
        }
        if (!Consume(",")) {
          return Fail("expected ',' or ']'");
        }
      }
    }
    *out = Bare();
    return true;
  }

  bool Object(json::Value* out, std::size_t depth) {
    if (!Consume("{")) {
      return Fail("expected '{'");
    }
    *out = json::Value::object();
    SkipSpace();
    if (Consume("}")) {
      return true;
    }
    while (true) {
      std::string key;
      if (!Key(&key)) {
        return false;
      }
      SkipSpace();
      if (!Consume(":")) {
        return Fail("expected ':'");
      }
      json::Value value;
      if (!Value(&value, depth)) {
        return false;
      }
      (*out)[key] = std::move(value);
      SkipSpace();
      if (Consume("}")) {
        return true;
      }
      if (!Consume(",")) {
        return Fail("expected ',' or '}'");
      }
    }
  }
};

}  // namespace

std::optional<ToolCall> ParseToolCall(std::string_view body,
                                      std::string* error_msg) {
  Parser p{body};
  p.SkipSpace();
  ToolCall call;
  bool ok = p.Consume("call:");
  if (!ok) {
    p.Fail("expected 'call:'");
  } else {
    const std::size_t brace = body.find('{', p.pos);
    if (brace == std::string_view::npos || brace == p.pos) {
      ok = p.Fail("expected a function name followed by '{'");
    } else {
      call.name = std::string(body.substr(p.pos, brace - p.pos));
      p.pos = brace;
      ok = p.Object(&call.arguments, 0);
      p.SkipSpace();
      if (ok && p.pos != body.size()) {
        ok = p.Fail("unexpected text after arguments");
      }
    }
  }
  if (!ok) {
    if (error_msg != nullptr) {
      *error_msg = p.error;
    }
    return std::nullopt;
  }
  return call;
}

}  // namespace gufo::models::gemma4
