#include "src/models/inkling2/text/json.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>

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

class Parser {
public:
  Parser(std::string_view t, std::string* e) : t_(t), err_(e) {}

  bool ParseDocument(Value* out) {
    SkipWs();
    if (!ParseValue(out, 0))
      return false;
    SkipWs();
    if (p_ != t_.size())
      return Fail("trailing characters");
    return true;
  }

private:
  bool Fail(const char* m) {
    if (err_ != nullptr && err_->empty())
      *err_ = std::string(m) + " at offset " + std::to_string(p_);
    return false;
  }
  void SkipWs() {
    while (p_ < t_.size() && (t_[p_] == ' ' || t_[p_] == '\t' ||
                              t_[p_] == '\n' || t_[p_] == '\r'))
      ++p_;
  }
  bool Lit(std::string_view w) {
    if (t_.substr(p_, w.size()) == w) {
      p_ += w.size();
      return true;
    }
    return false;
  }
  static void AppendUtf8(std::string* s, std::uint32_t cp) {
    if (cp < 0x80)
      s->push_back(static_cast<char>(cp));
    else if (cp < 0x800) {
      s->push_back(static_cast<char>(0xC0 | (cp >> 6)));
      s->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      s->push_back(static_cast<char>(0xE0 | (cp >> 12)));
      s->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      s->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      s->push_back(static_cast<char>(0xF0 | (cp >> 18)));
      s->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      s->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      s->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }
  bool Hex4(std::uint32_t* v) {
    if (p_ + 4 > t_.size())
      return false;
    std::uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = t_[p_ + i];
      r <<= 4;
      if (c >= '0' && c <= '9')
        r |= c - '0';
      else if (c >= 'a' && c <= 'f')
        r |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F')
        r |= c - 'A' + 10;
      else
        return false;
    }
    p_ += 4;
    *v = r;
    return true;
  }
  bool ParseString(std::string* out) {
    ++p_;  // opening quote
    while (p_ < t_.size()) {
      const char c = t_[p_++];
      if (c == '"')
        return true;
      if (static_cast<unsigned char>(c) < 0x20)
        return Fail("control character in string");
      if (c != '\\') {
        out->push_back(c);
        continue;
      }
      if (p_ >= t_.size())
        break;
      const char e = t_[p_++];
      switch (e) {
        case '"':
          out->push_back('"');
          break;
        case '\\':
          out->push_back('\\');
          break;
        case '/':
          out->push_back('/');
          break;
        case 'b':
          out->push_back('\b');
          break;
        case 'f':
          out->push_back('\f');
          break;
        case 'n':
          out->push_back('\n');
          break;
        case 'r':
          out->push_back('\r');
          break;
        case 't':
          out->push_back('\t');
          break;
        case 'u': {
          std::uint32_t u;
          if (!Hex4(&u))
            return Fail("bad \\u escape");
          if (u >= 0xD800 && u <= 0xDBFF && t_.substr(p_, 2) == "\\u") {
            const auto save = p_;
            p_ += 2;
            std::uint32_t lo;
            if (Hex4(&lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
              u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
            } else {
              p_ = save;
              u = 0xFFFD;
            }
          } else if (u >= 0xD800 && u <= 0xDFFF) {
            u = 0xFFFD;
          }
          AppendUtf8(out, u);
          break;
        }
        default:
          return Fail("bad escape");
      }
    }
    return Fail("unterminated string");
  }
  bool ParseNumber(Value* out) {
    const std::size_t start = p_;
    bool is_float = false;
    if (p_ < t_.size() && t_[p_] == '-')
      ++p_;
    if (p_ >= t_.size())
      return Fail("bad number");
    if (t_[p_] == '0')
      ++p_;
    else if (t_[p_] >= '1' && t_[p_] <= '9') {
      while (p_ < t_.size() && t_[p_] >= '0' && t_[p_] <= '9')
        ++p_;
    } else
      return Fail("bad number");
    if (p_ < t_.size() && t_[p_] == '.') {
      is_float = true;
      ++p_;
      const auto d0 = p_;
      while (p_ < t_.size() && t_[p_] >= '0' && t_[p_] <= '9')
        ++p_;
      if (p_ == d0)
        return Fail("bad number");
    }
    if (p_ < t_.size() && (t_[p_] == 'e' || t_[p_] == 'E')) {
      is_float = true;
      ++p_;
      if (p_ < t_.size() && (t_[p_] == '+' || t_[p_] == '-'))
        ++p_;
      const auto d0 = p_;
      while (p_ < t_.size() && t_[p_] >= '0' && t_[p_] <= '9')
        ++p_;
      if (p_ == d0)
        return Fail("bad number");
    }
    const std::string lex(t_.substr(start, p_ - start));
    if (is_float) {
      *out = Value::Double(std::strtod(lex.c_str(), nullptr));
    } else {
      *out = Value::Int(lex == "-0" ? std::string("0") : lex);
    }
    return true;
  }
  bool ParseValue(Value* out, int depth) {
    if (depth > 256)
      return Fail("nesting too deep");
    if (p_ >= t_.size())
      return Fail("unexpected end");
    const char c = t_[p_];
    if (c == '{') {
      ++p_;
      *out = Value::Object();
      SkipWs();
      if (p_ < t_.size() && t_[p_] == '}') {
        ++p_;
        return true;
      }
      for (;;) {
        SkipWs();
        if (p_ >= t_.size() || t_[p_] != '"')
          return Fail("expected key");
        std::string key;
        if (!ParseString(&key))
          return false;
        SkipWs();
        if (p_ >= t_.size() || t_[p_] != ':')
          return Fail("expected ':'");
        ++p_;
        SkipWs();
        Value v;
        if (!ParseValue(&v, depth + 1))
          return false;
        out->Set(std::move(key), std::move(v));
        SkipWs();
        if (p_ < t_.size() && t_[p_] == ',') {
          ++p_;
          continue;
        }
        if (p_ < t_.size() && t_[p_] == '}') {
          ++p_;
          return true;
        }
        return Fail("expected ',' or '}'");
      }
    }
    if (c == '[') {
      ++p_;
      *out = Value::Array();
      SkipWs();
      if (p_ < t_.size() && t_[p_] == ']') {
        ++p_;
        return true;
      }
      for (;;) {
        SkipWs();
        Value v;
        if (!ParseValue(&v, depth + 1))
          return false;
        out->Push(std::move(v));
        SkipWs();
        if (p_ < t_.size() && t_[p_] == ',') {
          ++p_;
          continue;
        }
        if (p_ < t_.size() && t_[p_] == ']') {
          ++p_;
          return true;
        }
        return Fail("expected ',' or ']'");
      }
    }
    if (c == '"') {
      std::string s;
      if (!ParseString(&s))
        return false;
      *out = Value::String(std::move(s));
      return true;
    }
    if (Lit("true")) {
      *out = Value::Bool(true);
      return true;
    }
    if (Lit("false")) {
      *out = Value::Bool(false);
      return true;
    }
    if (Lit("null")) {
      *out = Value::Null();
      return true;
    }
    return ParseNumber(out);
  }

  std::string_view t_;
  std::string* err_;
  std::size_t p_{0};
};

}  // namespace

bool Parse(std::string_view text, Value* out, std::string* error) {
  if (error != nullptr)
    error->clear();
  return Parser(text, error).ParseDocument(out);
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
  if (std::isnan(d))
    return "nan";
  if (std::isinf(d))
    return d < 0 ? "-inf" : "inf";
  if (d == 0)
    return std::signbit(d) ? "-0.0" : "0.0";
  // Shortest round-trip digits, then Python's repr layout (float_repr_style
  // 'short').
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
  std::string out;
  if (decpt > 16 || decpt < -3) {
    out = digits.substr(0, 1);
    if (digits.size() > 1)
      out += "." + digits.substr(1);
    char eb[16];
    std::snprintf(eb, sizeof(eb), "e%c%02d", exp10 < 0 ? '-' : '+',
                  std::abs(exp10));
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
        *out += PythonFloatRepr(d);
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
