#pragma once

// The JSON parser of Gufo 0.35.0 (src/core/json.hpp), frozen as the oracle
// for json_parse_test: everything from `namespace detail` on is the original.
// Only Value is replaced by the fields and constructors the parser uses.

#include <charconv>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gufo::json_reference {

class Value {
public:
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };

  Value() = default;
  Value(bool v) : type_(Type::kBool), bool_(v) {}
  Value(double v) : type_(Type::kNumber), num_(v) {}
  Value(std::string&& v) : type_(Type::kString), str_(std::move(v)) {}

  static Value object() {
    Value v;
    v.type_ = Type::kObject;
    return v;
  }
  static Value array() {
    Value v;
    v.type_ = Type::kArray;
    return v;
  }
  void append_member(std::string key, Value value) {
    obj_.emplace_back(std::move(key), std::move(value));
  }
  void push_back(Value v) { arr_.push_back(std::move(v)); }

  Type type_ = Type::kNull;
  bool bool_ = false;
  double num_ = 0.0;
  std::string str_;
  std::vector<Value> arr_;
  std::vector<std::pair<std::string, Value>> obj_;
};

namespace detail {
struct Parser {
  std::string_view s;
  std::size_t i = 0;

  [[noreturn]] void fail(const char* msg) const {
    throw std::runtime_error(std::string("JSON parse error: ") + msg);
  }
  void skip_ws() {
    while (i < s.size()) {
      const char c = s[i];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++i;
      } else {
        break;
      }
    }
  }
  // NOLINTNEXTLINE(misc-no-recursion)
  Value parse_value(std::size_t depth = 0) {
    skip_ws();
    if (i >= s.size())
      fail("unexpected end of input");
    const char c = s[i];
    if ((c == '{' || c == '[') && depth >= 128) {
      fail("maximum nesting depth exceeded");
    }
    if (c == '{')
      return parse_object(depth + 1);
    if (c == '[')
      return parse_array(depth + 1);
    if (c == '"')
      return Value(parse_string());
    if (c == 't' || c == 'f')
      return parse_bool();
    if (c == 'n')
      return parse_null();
    return parse_number();
  }
  // NOLINTNEXTLINE(misc-no-recursion)
  Value parse_object(std::size_t depth) {
    Value v = Value::object();
    std::unordered_set<std::string> keys;
    ++i;  // {
    skip_ws();
    if (i < s.size() && s[i] == '}') {
      ++i;
      return v;
    }
    while (true) {
      skip_ws();
      if (i >= s.size() || s[i] != '"')
        fail("expected object key string");
      const std::string key = parse_string();
      skip_ws();
      if (i >= s.size() || s[i] != ':')
        fail("expected ':' after key");
      ++i;
      if (!keys.emplace(key).second)
        fail("duplicate object key");
      v.append_member(key, parse_value(depth));
      skip_ws();
      if (i >= s.size())
        fail("unterminated object");
      if (s[i] == ',') {
        ++i;
        continue;
      }
      if (s[i] == '}') {
        ++i;
        return v;
      }
      fail("expected ',' or '}' in object");
    }
  }
  // NOLINTNEXTLINE(misc-no-recursion)
  Value parse_array(std::size_t depth) {
    Value v = Value::array();
    ++i;  // [
    skip_ws();
    if (i < s.size() && s[i] == ']') {
      ++i;
      return v;
    }
    while (true) {
      v.push_back(parse_value(depth));
      skip_ws();
      if (i >= s.size())
        fail("unterminated array");
      if (s[i] == ',') {
        ++i;
        continue;
      }
      if (s[i] == ']') {
        ++i;
        return v;
      }
      fail("expected ',' or ']' in array");
    }
  }
  unsigned parse_hex_quad() {
    if (i + 4 > s.size())
      fail("bad \\u");
    unsigned code = 0;
    for (int k = 0; k < 4; ++k) {
      const char h = s[i++];
      code <<= 4;
      if (h >= '0' && h <= '9')
        code |= h - '0';
      else if (h >= 'a' && h <= 'f')
        code |= h - 'a' + 10;
      else if (h >= 'A' && h <= 'F')
        code |= h - 'A' + 10;
      else
        fail("bad hex");
    }
    return code;
  }
  std::string parse_string() {
    ++i;  // opening quote
    std::string out;
    while (i < s.size()) {
      const char c = s[i++];
      if (c == '"')
        return out;
      if (c == '\\') {
        if (i >= s.size())
          fail("bad escape");
        const char e = s[i++];
        switch (e) {
          case '"':
            out += '"';
            break;
          case '\\':
            out += '\\';
            break;
          case '/':
            out += '/';
            break;
          case 'b':
            out += '\b';
            break;
          case 'f':
            out += '\f';
            break;
          case 'n':
            out += '\n';
            break;
          case 'r':
            out += '\r';
            break;
          case 't':
            out += '\t';
            break;
          case 'u': {
            unsigned code = parse_hex_quad();
            if (code >= 0xD800 && code <= 0xDBFF) {
              if (i + 2 > s.size() || s[i] != '\\' || s[i + 1] != 'u')
                fail("unpaired high surrogate");
              i += 2;
              const unsigned low = parse_hex_quad();
              if (low < 0xDC00 || low > 0xDFFF)
                fail("invalid low surrogate");
              code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
            } else if (code >= 0xDC00 && code <= 0xDFFF) {
              fail("unpaired low surrogate");
            }
            if (code < 0x80) {
              out += static_cast<char>(code);
            } else if (code < 0x800) {
              out += static_cast<char>(0xC0 | (code >> 6));
              out += static_cast<char>(0x80 | (code & 0x3F));
            } else if (code < 0x10000) {
              out += static_cast<char>(0xE0 | (code >> 12));
              out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
              out += static_cast<char>(0x80 | (code & 0x3F));
            } else {
              out += static_cast<char>(0xF0 | (code >> 18));
              out += static_cast<char>(0x80 | ((code >> 12) & 0x3F));
              out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
              out += static_cast<char>(0x80 | (code & 0x3F));
            }
            break;
          }
          default:
            fail("bad escape");
        }
      } else {
        const auto lead = static_cast<unsigned char>(c);
        if (lead < 0x20)
          fail("unescaped control character");
        if (lead < 0x80) {
          out += c;
          continue;
        }
        const unsigned length = lead >= 0xC2 && lead <= 0xDF   ? 2
                                : lead >= 0xE0 && lead <= 0xEF ? 3
                                : lead >= 0xF0 && lead <= 0xF4 ? 4
                                                               : 0;
        if (length == 0 || length - 1 > s.size() - i)
          fail("invalid UTF-8");
        for (unsigned j = 1; j < length; ++j) {
          const auto byte = static_cast<unsigned char>(s[i + j - 1]);
          if (byte < 0x80 || byte > 0xBF ||
              (j == 1 && ((lead == 0xE0 && byte < 0xA0) ||
                          (lead == 0xED && byte > 0x9F) ||
                          (lead == 0xF0 && byte < 0x90) ||
                          (lead == 0xF4 && byte > 0x8F))))
            fail("invalid UTF-8");
        }
        out.append(s.substr(i - 1, length));
        i += length - 1;
      }
    }
    fail("unterminated string");
  }
  Value parse_bool() {
    if (s.compare(i, 4, "true") == 0) {
      i += 4;
      return Value(true);
    }
    if (s.compare(i, 5, "false") == 0) {
      i += 5;
      return Value(false);
    }
    fail("bad literal");
  }
  Value parse_null() {
    if (s.compare(i, 4, "null") == 0) {
      i += 4;
      return Value();
    }
    fail("bad literal");
  }
  Value parse_number() {
    const std::size_t start = i;
    if (i < s.size() && s[i] == '-')
      ++i;
    if (i >= s.size())
      fail("bad number");
    if (s[i] == '0') {
      ++i;
      if (i < s.size() && s[i] >= '0' && s[i] <= '9')
        fail("bad number");
    } else if (s[i] >= '1' && s[i] <= '9') {
      while (i < s.size() && s[i] >= '0' && s[i] <= '9')
        ++i;
    } else {
      fail("bad number");
    }
    if (i < s.size() && s[i] == '.') {
      ++i;
      if (i >= s.size() || s[i] < '0' || s[i] > '9')
        fail("bad number");
      while (i < s.size() && s[i] >= '0' && s[i] <= '9')
        ++i;
    }
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
      ++i;
      if (i < s.size() && (s[i] == '-' || s[i] == '+'))
        ++i;
      if (i >= s.size() || s[i] < '0' || s[i] > '9')
        fail("bad number");
      while (i < s.size() && s[i] >= '0' && s[i] <= '9')
        ++i;
    }
    const std::string tok(s.substr(start, i - start));
    double value = 0.0;
    const auto [end, error] =
        std::from_chars(tok.data(), tok.data() + tok.size(), value);
    if (error != std::errc{} || end != tok.data() + tok.size() ||
        !std::isfinite(value)) {
      fail("bad number");
    }
    Value number(value);
    number.str_ = tok;
    return number;
  }
};
}  // namespace detail

/// Parses `text` into a Value. Throws std::runtime_error on malformed input.
[[nodiscard]] inline Value parse(std::string_view text) {
  detail::Parser p;
  p.s = text;
  Value value = p.parse_value();
  p.skip_ws();
  if (p.i != text.size())
    p.fail("trailing input");
  return value;
}

}  // namespace gufo::json_reference
