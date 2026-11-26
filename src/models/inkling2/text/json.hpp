// Small insertion-ordered JSON value with Python's json semantics: integers
// of any size, floats, and a serializer that writes json.dumps' spelling
// (ensure_ascii=False, repr() floats, optional sort_keys, custom separators),
// which the chat template relies on. Parsing is Gufo's JSON parser.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace inkling2::text::json {

class Value {
public:
  enum class Kind : std::uint8_t {
    kNull,
    kBool,
    kInt,
    kDouble,
    kString,
    kArray,
    kObject
  };
  using Member = std::pair<std::string, Value>;

  Value() = default;
  static Value Null() { return {}; }
  static Value Bool(bool b) {
    Value v;
    v.kind_ = Kind::kBool;
    v.b_ = b;
    return v;
  }
  /// `digits` is a canonical integer lexeme (arbitrary size).
  static Value Int(std::string digits) {
    Value v;
    v.kind_ = Kind::kInt;
    v.s_ = std::move(digits);
    return v;
  }
  static Value Int(std::int64_t n) { return Int(std::to_string(n)); }
  static Value Double(double d) {
    Value v;
    v.kind_ = Kind::kDouble;
    v.d_ = d;
    return v;
  }
  static Value String(std::string s) {
    Value v;
    v.kind_ = Kind::kString;
    v.s_ = std::move(s);
    return v;
  }
  static Value Array() {
    Value v;
    v.kind_ = Kind::kArray;
    return v;
  }
  static Value Object() {
    Value v;
    v.kind_ = Kind::kObject;
    return v;
  }

  [[nodiscard]] Kind kind() const { return kind_; }
  [[nodiscard]] bool IsNull() const { return kind_ == Kind::kNull; }
  [[nodiscard]] bool IsString() const { return kind_ == Kind::kString; }
  [[nodiscard]] bool IsObject() const { return kind_ == Kind::kObject; }
  [[nodiscard]] bool IsArray() const { return kind_ == Kind::kArray; }
  [[nodiscard]] bool AsBool() const { return b_; }
  [[nodiscard]] double AsDouble() const;
  [[nodiscard]] const std::string& str() const {
    return s_;
  }  // string or int lexeme
  [[nodiscard]] const std::vector<Value>& items() const { return arr_; }
  [[nodiscard]] std::vector<Value>& items() { return arr_; }
  [[nodiscard]] const std::vector<Member>& members() const { return obj_; }
  [[nodiscard]] std::vector<Member>& members() { return obj_; }

  [[nodiscard]] const Value* Find(std::string_view key) const;
  /// Python dict assignment: replaces in place, or appends.
  void Set(std::string key, Value v);
  void Push(Value v) { arr_.push_back(std::move(v)); }

  /// Python truthiness (None, False, 0, 0.0, "", [], {} are falsy).
  [[nodiscard]] bool Truthy() const;

private:
  Kind kind_{Kind::kNull};
  bool b_{false};
  double d_{0};
  std::string s_;
  std::vector<Value> arr_;
  std::vector<Member> obj_;
};

/// Parses `text` with Gufo's parser (strict RFC 8259); duplicate keys keep the
/// first position and the last value, as Python's json.loads does. Returns
/// false (and sets *error) on malformed input.
[[nodiscard]] bool Parse(std::string_view text, Value* out,
                         std::string* error = nullptr);

/// How a float is spelled. Python's repr writes positional notation for
/// decimal exponents in [-4, 16) and a sign and two digits in exponents
/// (1e-05, 1e+20). Thinking Machines' renderer (tml-renderers) writes
/// positional notation for exponents in [-5, 16) and unpadded exponents: tool
/// declarations keep the sign of positive ones (1e+20), call arguments drop
/// it (1e20).
enum class FloatStyle : std::uint8_t { kPython, kTml, kTmlUnsigned };

struct DumpOptions {
  bool sort_keys{false};
  std::string item_separator{", "};  // Python json.dumps default
  std::string key_separator{": "};
  FloatStyle floats{FloatStyle::kPython};
};
/// json.dumps(value, ensure_ascii=False, ...).
[[nodiscard]] std::string Dump(const Value& v, const DumpOptions& options = {});
void DumpTo(std::string* out, const Value& v, const DumpOptions& options);

/// A finite float in `style`; PythonFloatRepr is kPython.
[[nodiscard]] std::string FloatRepr(double d, FloatStyle style);
/// Python repr(float) / str(float).
[[nodiscard]] std::string PythonFloatRepr(double d);
/// JSON string literal as json.dumps(s, ensure_ascii=False).
[[nodiscard]] std::string QuoteString(std::string_view s);

}  // namespace inkling2::text::json
