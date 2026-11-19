#ifndef GUFO_CORE_JSON_HPP_
#define GUFO_CORE_JSON_HPP_

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gufo::json {

namespace detail {
struct Parser;
}  // namespace detail

/// Minimal, ordered JSON value with a parser and serializer.
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

  /// How dump() and tojson() write numbers read by parse(). kCanonical
  /// spells every number from its double; kAsParsed writes the token parse()
  /// read, as llama.cpp returns tool-call arguments. Numbers built from C++
  /// values are always canonical.
  enum class Numbers : std::uint8_t { kCanonical, kAsParsed };

  Value() = default;
  Value(std::nullptr_t) {}
  Value(bool v) : type_(Type::kBool), bool_(v) {}
  Value(double v) : type_(Type::kNumber), num_(v) {}
  Value(int v) : type_(Type::kNumber), num_(static_cast<double>(v)) {}
  Value(long v) : type_(Type::kNumber), num_(static_cast<double>(v)) {}
  Value(long long v) : type_(Type::kNumber), num_(static_cast<double>(v)) {}
  Value(unsigned long long v)
      : type_(Type::kNumber), num_(static_cast<double>(v)) {}
  // Integral constructor for the widths the explicit overloads above do not
  // cover (size_t, plus narrower types such as char and short, which
  // previously converted through int). The constraints keep it from
  // duplicating those overloads on platforms where the widths coincide.
  template<
      typename T,
      typename = std::enable_if_t<
          std::is_integral_v<T> && !std::is_same_v<T, int> &&
          !std::is_same_v<T, unsigned int> && !std::is_same_v<T, long long> &&
          !std::is_same_v<T, unsigned long long> && !std::is_same_v<T, bool>>>
  Value(T v) : type_(Type::kNumber), num_(static_cast<double>(v)) {}
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
  void append_member(std::string key, Value value) {
    if (type_ != Type::kObject) {
      type_ = Type::kObject;
      obj_.clear();
    }
    obj_.emplace_back(std::move(key), std::move(value));
  }
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
  std::size_t as_size(std::size_t def = 0) const noexcept {
    // The largest size_t rounds up when converted to double on x86-64.
    // Compare against the exclusive power-of-two bound before conversion.
    if (!is_number() || !std::isfinite(num_) || num_ < 0.0 ||
        std::floor(num_) != num_ ||
        num_ >= std::ldexp(1.0, std::numeric_limits<std::size_t>::digits)) {
      return def;
    }
    return static_cast<std::size_t>(num_);
  }
  /// The string payload (empty string if this is not a string).
  const std::string& str() const noexcept {
    static const std::string kEmpty;
    return is_string() ? str_ : kEmpty;
  }
  /// The string payload, or `def` if not a string.
  std::string get_str(const std::string& def = "") const {
    return is_string() ? str_ : def;
  }

  std::string dump(Numbers numbers = Numbers::kCanonical) const {
    std::ostringstream o;
    dump_to(o, false, numbers);
    return o.str();
  }

  /// Jinja's `tojson` spelling, as Python's json.dumps(ensure_ascii=False)
  /// writes it: ", " between items and ": " after keys. Chat templates render
  /// non-string tool arguments this way, so models generate it.
  std::string tojson(Numbers numbers = Numbers::kCanonical) const {
    std::ostringstream o;
    dump_to(o, true, numbers);
    return o.str();
  }

private:
  friend struct detail::Parser;

  // NOLINTNEXTLINE(misc-no-recursion)
  void dump_to(std::ostream& os, bool spaced, Numbers numbers) const {
    switch (type_) {
      case Type::kNull:
        os << "null";
        break;
      case Type::kBool:
        os << (bool_ ? "true" : "false");
        break;
      case Type::kNumber: {
        if (numbers == Numbers::kAsParsed && !str_.empty()) {
          os << str_;
          break;
        }
        if (!std::isfinite(num_)) {
          throw std::invalid_argument("JSON numbers must be finite");
        }
        // Integers keep their decimal spelling, as Python's json.dumps and
        // chat templates write them. The shortest round-trip form would spell
        // 100000 as 1e+05: clients that declared an integer get a float, and
        // replayed tool calls no longer match the tokens the model generated.
        constexpr double kExactIntegers = 9007199254740992.0;  // 2^53
        const bool integral = std::trunc(num_) == num_ &&
                              std::fabs(num_) <= kExactIntegers &&
                              !(num_ == 0.0 && std::signbit(num_));
        char buffer[64];
        const auto [end, error] =
            integral ? std::to_chars(buffer, buffer + sizeof(buffer),
                                     static_cast<std::int64_t>(num_))
                     : std::to_chars(buffer, buffer + sizeof(buffer), num_);
        if (error != std::errc{}) {
          throw std::runtime_error("JSON number serialization failed");
        }
        os.write(buffer, end - buffer);
        break;
      }
      case Type::kString:
        os << '"' << escape(str_) << '"';
        break;
      case Type::kArray: {
        os << '[';
        for (std::size_t i = 0; i < arr_.size(); ++i) {
          if (i)
            os << (spaced ? ", " : ",");
          arr_[i].dump_to(os, spaced, numbers);
        }
        os << ']';
        break;
      }
      case Type::kObject: {
        os << '{';
        for (std::size_t i = 0; i < obj_.size(); ++i) {
          if (i)
            os << (spaced ? ", " : ",");
          os << '"' << escape(obj_[i].first) << (spaced ? "\": " : "\":");
          obj_[i].second.dump_to(os, spaced, numbers);
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
  // A string's payload, or the validated token of a number read by parse().
  std::string str_;
  Array arr_;
  Object obj_;
};

/// Parses `text` into a Value. Throws std::runtime_error on malformed input,
/// std::bad_alloc when memory runs out and std::logic_error on a parser bug.
/// src/core/json.cpp builds it from the events of the Rust parser in
/// src/core/json.
[[nodiscard]] Value parse(std::string_view text);

}  // namespace gufo::json

#endif  // GUFO_CORE_JSON_HPP_
