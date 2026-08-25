#ifndef STRIX_SERVER_JSON_HPP_
#define STRIX_SERVER_JSON_HPP_

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace strix::server::json {

/// Minimal, ordered JSON value with a recursive-descent parser and serializer.
/// Object member order is preserved via a vector of (key, value) pairs.
class Value {
public:
  enum class Type : std::uint8_t {
    kNull,
    kBool,
    kNumber,
    kString,
    kArray,
    kObject
  };

  using Array = std::vector<Value>;
  using Member = std::pair<std::string, Value>;
  using Object = std::vector<Member>;

  Value() = default;
  Value(std::nullptr_t) {}
  Value(bool v) : type_(Type::kBool), bool_(v) {}
  Value(double v) : type_(Type::kNumber), num_(v) {}
  Value(int v) : type_(Type::kNumber), num_(static_cast<double>(v)) {}
  Value(long v) : type_(Type::kNumber), num_(static_cast<double>(v)) {}
  Value(long long v) : type_(Type::kNumber), num_(static_cast<double>(v)) {}
  Value(unsigned long long v)
      : type_(Type::kNumber), num_(static_cast<double>(v)) {}
  Value(std::size_t v) : type_(Type::kNumber), num_(static_cast<double>(v)) {}
  Value(const char* v) : type_(Type::kString), str_(v) {}
  Value(const std::string& v) : type_(Type::kString), str_(v) {}
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

  Type type() const noexcept { return type_; }
  bool is_null() const noexcept { return type_ == Type::kNull; }
  bool is_bool() const noexcept { return type_ == Type::kBool; }
  bool is_number() const noexcept { return type_ == Type::kNumber; }
  bool is_string() const noexcept { return type_ == Type::kString; }
  bool is_array() const noexcept { return type_ == Type::kArray; }
  bool is_object() const noexcept { return type_ == Type::kObject; }

  // ---- Object access ----
  /// Returns the member for `key`, inserting a null value if absent.
  Value& operator[](const std::string& key) {
    if (type_ != Type::kObject) {
      type_ = Type::kObject;
      obj_.clear();
    }
    for (auto& kv : obj_) {
      if (kv.first == key)
        return kv.second;
    }
    obj_.emplace_back(key, Value());
    return obj_.back().second;
  }
  const Value* find(const std::string& key) const noexcept {
    if (type_ != Type::kObject)
      return nullptr;
    for (const auto& kv : obj_) {
      if (kv.first == key)
        return &kv.second;
    }
    return nullptr;
  }
  bool contains(const std::string& key) const noexcept {
    return find(key) != nullptr;
  }
  const Object& members() const noexcept { return obj_; }
  /// String value of member `key` (or `def` if missing / not a string).
  std::string member_str(const std::string& key,
                         const std::string& def = "") const {
    const Value* v = find(key);
    return (v && v->is_string()) ? v->str() : def;
  }
  /// Numeric value of member `key` as size_t.
  std::size_t member_size(const std::string& key, std::size_t def = 0) const {
    const Value* v = find(key);
    return (v && v->is_number()) ? v->as_size(def) : def;
  }
  /// Numeric value of member `key` as double.
  double member_double(const std::string& key, double def = 0.0) const {
    const Value* v = find(key);
    return (v && v->is_number()) ? v->as_double(def) : def;
  }

  // ---- Array access ----
  void push_back(Value v) {
    if (type_ != Type::kArray) {
      type_ = Type::kArray;
      arr_.clear();
    }
    arr_.push_back(std::move(v));
  }
  void push_back() {
    if (type_ != Type::kArray) {
      type_ = Type::kArray;
      arr_.clear();
    }
    arr_.emplace_back();
  }
  const Array& items() const noexcept { return arr_; }
  std::size_t size() const noexcept {
    if (type_ == Type::kArray)
      return arr_.size();
    if (type_ == Type::kObject)
      return obj_.size();
    return 0;
  }
  bool empty() const noexcept { return size() == 0; }

  // ---- Typed getters (on the value itself) ----
  bool as_bool(bool def = false) const noexcept {
    return is_bool() ? bool_ : def;
  }
  double as_double(double def = 0.0) const noexcept {
    return is_number() ? num_ : def;
  }
  long long as_int(long long def = 0) const noexcept {
    return is_number() ? static_cast<long long>(std::llround(num_)) : def;
  }
  std::size_t as_size(std::size_t def = 0) const noexcept {
    if (!is_number() || num_ < 0.0)
      return def;
    return static_cast<std::size_t>(num_);
  }
  /// The string payload (empty string if this is not a string).
  const std::string& str() const noexcept {
    static const std::string kEmpty;
    return is_string() ? str_ : kEmpty;
  }
  /// The string payload, or `def` if not a string.
  std::string get_str(const std::string& def = "") const noexcept {
    return is_string() ? str_ : def;
  }

  std::string dump() const {
    std::ostringstream o;
    dump_to(o);
    return o.str();
  }

private:
  // NOLINTNEXTLINE(misc-no-recursion)
  void dump_to(std::ostream& os) const {
    switch (type_) {
      case Type::kNull:
        os << "null";
        break;
      case Type::kBool:
        os << (bool_ ? "true" : "false");
        break;
      case Type::kNumber:
        if (std::isfinite(num_) && std::floor(num_) == num_ &&
            std::fabs(num_) < 1e15) {
          os << static_cast<long long>(std::llround(num_));
        } else {
          os << num_;
        }
        break;
      case Type::kString:
        os << '"' << escape(str_) << '"';
        break;
      case Type::kArray: {
        os << '[';
        for (std::size_t i = 0; i < arr_.size(); ++i) {
          if (i)
            os << ',';
          arr_[i].dump_to(os);
        }
        os << ']';
        break;
      }
      case Type::kObject: {
        os << '{';
        for (std::size_t i = 0; i < obj_.size(); ++i) {
          if (i)
            os << ',';
          os << '"' << escape(obj_[i].first) << "\":";
          obj_[i].second.dump_to(os);
        }
        os << '}';
        break;
      }
    }
  }

  static std::string escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (const unsigned char c : s) {
      switch (c) {
        case '"':
          out += "\\\"";
          break;
        case '\\':
          out += "\\\\";
          break;
        case '\b':
          out += "\\b";
          break;
        case '\f':
          out += "\\f";
          break;
        case '\n':
          out += "\\n";
          break;
        case '\r':
          out += "\\r";
          break;
        case '\t':
          out += "\\t";
          break;
        default:
          if (c < 0x20) {
            char buf[8];
            (void)std::snprintf(buf, sizeof(buf), "\\u%04x", c);
            out += buf;
          } else {
            out += static_cast<char>(c);
          }
      }
    }
    return out;
  }

  Type type_ = Type::kNull;
  bool bool_ = false;
  double num_ = 0.0;
  std::string str_;
  Array arr_;
  Object obj_;
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
  Value parse_value() {
    skip_ws();
    if (i >= s.size())
      fail("unexpected end of input");
    const char c = s[i];
    if (c == '{')
      return parse_object();
    if (c == '[')
      return parse_array();
    if (c == '"')
      return Value(parse_string());
    if (c == 't' || c == 'f')
      return parse_bool();
    if (c == 'n')
      return parse_null();
    return parse_number();
  }
  // NOLINTNEXTLINE(misc-no-recursion)
  Value parse_object() {
    Value v = Value::object();
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
      if (v.contains(key))
        fail("duplicate object key");
      v[key] = parse_value();
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
  Value parse_array() {
    Value v = Value::array();
    ++i;  // [
    skip_ws();
    if (i < s.size() && s[i] == ']') {
      ++i;
      return v;
    }
    while (true) {
      v.push_back(parse_value());
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
            if (code < 0x80) {
              out += static_cast<char>(code);
            } else if (code < 0x800) {
              out += static_cast<char>(0xC0 | (code >> 6));
              out += static_cast<char>(0x80 | (code & 0x3F));
            } else {
              out += static_cast<char>(0xE0 | (code >> 12));
              out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
              out += static_cast<char>(0x80 | (code & 0x3F));
            }
            break;
          }
          default:
            fail("bad escape");
        }
      } else {
        out += c;
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
    return Value(value);
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

}  // namespace strix::server::json

#endif  // STRIX_SERVER_JSON_HPP_
