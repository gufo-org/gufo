#include "src/models/inkling2/text/json.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <gufo/ext/json.hpp>
#include <new>

namespace inkling2::text::json {

double Value::AsDouble() const {
  if (kind_ == Kind::kDouble)
    return d_;
  if (kind_ == Kind::kInt)
    return std::strtod(s_.c_str(), nullptr);
  return 0;
}

const Value* Value::Find(std::string_view key) const {
  for (const auto& m : obj_)
    if (m.first == key)
      return &m.second;
  return nullptr;
}

void Value::Set(std::string key, Value v) {
  for (auto& m : obj_) {
    if (m.first == key) {
      m.second = std::move(v);
      return;
    }
  }
  obj_.emplace_back(std::move(key), std::move(v));
}

bool Value::Truthy() const {
  switch (kind_) {
    case Kind::kNull:
      return false;
    case Kind::kBool:
      return b_;
    case Kind::kInt:
      return s_ != "0";
    case Kind::kDouble:
      return d_ != 0.0;
    case Kind::kString:
      return !s_.empty();
    case Kind::kArray:
      return !arr_.empty();
    case Kind::kObject:
      return !obj_.empty();
  }
  return false;
}

namespace {

// NOLINTNEXTLINE(misc-no-recursion)
Value FromCore(const ::gufo::json::Value& v) {
  using T = ::gufo::json::Value::Type;
  switch (v.type()) {
    case T::kNull:
      return Value::Null();
    case T::kBool:
      return Value::Bool(v.as_bool());
    case T::kNumber: {
      // json.loads: an integer token stays an int of any size, anything with
      // a fraction or exponent is a float.
      const std::string lexeme =
          v.dump(::gufo::json::Value::Numbers::kAsParsed);
      if (lexeme.find_first_of(".eE") != std::string::npos)
        return Value::Double(v.as_double());
      return Value::Int(lexeme == "-0" ? std::string("0") : lexeme);
    }
    case T::kString:
      return Value::String(v.str());
    case T::kArray: {
      Value out = Value::Array();
      for (const auto& item : v.items())
        out.Push(FromCore(item));
      return out;
    }
    case T::kObject: {
      Value out = Value::Object();
      for (const auto& [key, item] : v.members())
        out.Set(key, FromCore(item));
      return out;
    }
  }
  return Value::Null();
}

}  // namespace

bool Parse(std::string_view text, Value* out, std::string* error) {
  if (error != nullptr)
    error->clear();
  try {
    *out = FromCore(::gufo::json::parse(text));
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const std::exception& e) {
    if (error != nullptr)
      *error = e.what();
    return false;
  }
  return true;
}

std::string QuoteString(std::string_view s) {
  std::string out;
  out.reserve(s.size() + 2);
  out.push_back('"');
  for (const char ch : s) {
    const auto c = static_cast<unsigned char>(ch);
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
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
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back(ch);
        }
    }
  }
  out.push_back('"');
  return out;
}

std::string PythonFloatRepr(double d) {
  return FloatRepr(d, FloatStyle::kPython);
}

std::string FloatRepr(double d, FloatStyle style) {
  if (std::isnan(d))
    return "nan";
  if (std::isinf(d))
    return d < 0 ? "-inf" : "inf";
  if (d == 0)
    return std::signbit(d) ? "-0.0" : "0.0";
  // Shortest round-trip digits, then the style's layout (Python's is
  // float_repr_style 'short').
  char buf[64];
  const auto r =
      std::to_chars(buf, buf + sizeof(buf), d, std::chars_format::scientific);
  std::string sci(buf, r.ptr);  // d.ddde[+-]XX
  const bool neg = sci[0] == '-';
  if (neg)
    sci.erase(0, 1);
  const auto epos = sci.find('e');
  std::string mant = sci.substr(0, epos);
  const int exp10 = std::atoi(sci.c_str() + epos + 1);
  std::string digits;
  for (const char c : mant)
    if (c != '.')
      digits.push_back(c);
  const int decpt =
      exp10 + 1;  // position of the decimal point relative to digits
  const bool python = style == FloatStyle::kPython;
  const int min_positional = python ? -4 : -5;
  std::string out;
  if (exp10 >= 16 || exp10 < min_positional) {
    out = digits.substr(0, 1);
    if (digits.size() > 1)
      out += "." + digits.substr(1);
    char eb[16];
    if (python)
      std::snprintf(eb, sizeof(eb), "e%c%02d", exp10 < 0 ? '-' : '+',
                    std::abs(exp10));
    else
      std::snprintf(eb, sizeof(eb), "e%s%d",
                    exp10 >= 0 && style == FloatStyle::kTml ? "+" : "", exp10);
    out += eb;
  } else if (decpt <= 0) {
    out = "0." + std::string(static_cast<std::size_t>(-decpt), '0') + digits;
  } else if (static_cast<std::size_t>(decpt) >= digits.size()) {
    out = digits + std::string(decpt - digits.size(), '0') + ".0";
  } else {
    out = digits.substr(0, decpt) + "." + digits.substr(decpt);
  }
  return neg ? "-" + out : out;
}

void DumpTo(std::string* out, const Value& v, const DumpOptions& o) {
  switch (v.kind()) {
    case Value::Kind::kNull:
      *out += "null";
      break;
    case Value::Kind::kBool:
      *out += v.AsBool() ? "true" : "false";
      break;
    case Value::Kind::kInt:
      *out += v.str();
      break;
    case Value::Kind::kDouble: {
      const double d = v.AsDouble();
      if (std::isnan(d))
        *out += "NaN";
      else if (std::isinf(d))
        *out += d < 0 ? "-Infinity" : "Infinity";
      else
        *out += FloatRepr(d, o.floats);
      break;
    }
    case Value::Kind::kString:
      *out += QuoteString(v.str());
      break;
    case Value::Kind::kArray: {
      out->push_back('[');
      bool first = true;
      for (const auto& e : v.items()) {
        if (!first)
          *out += o.item_separator;
        first = false;
        DumpTo(out, e, o);
      }
      out->push_back(']');
      break;
    }
    case Value::Kind::kObject: {
      out->push_back('{');
      std::vector<const Value::Member*> ms;
      for (const auto& m : v.members())
        ms.push_back(&m);
      if (o.sort_keys)
        std::stable_sort(
            ms.begin(), ms.end(),
            [](const auto* a, const auto* b) { return a->first < b->first; });
      bool first = true;
      for (const auto* m : ms) {
        if (!first)
          *out += o.item_separator;
        first = false;
        *out += QuoteString(m->first);
        *out += o.key_separator;
        DumpTo(out, m->second, o);
      }
      out->push_back('}');
      break;
    }
  }
}

std::string Dump(const Value& v, const DumpOptions& options) {
  std::string out;
  DumpTo(&out, v, options);
  return out;
}

}  // namespace inkling2::text::json
