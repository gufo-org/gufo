#include "src/core/json_schema_lexeme.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/json_schema_regex.hpp"

namespace gufo::sampling {
namespace {

[[noreturn]] void Invalid(std::string_view reason) {
  throw std::invalid_argument("JSON Schema: " + std::string(reason));
}

// Exact finite-decimal arithmetic. JSON Schema multipleOf must not rely on a
// floating-point remainder (e.g. 0.3 is a multiple of 0.1).
struct Decimal {
  std::string digits{"0"};
  int scale{0};
  bool negative{false};

  static Decimal Parse(std::string_view text) {
    Decimal value;
    value.digits.clear();
    if (text.starts_with('-')) {
      value.negative = true;
      text.remove_prefix(1);
    }
    bool fraction = false;
    std::size_t i = 0;
    for (; i < text.size() && text[i] != 'e' && text[i] != 'E'; ++i) {
      if (text[i] == '.') {
        fraction = true;
      } else {
        value.digits += text[i];
        value.scale += fraction;
      }
    }
    if (i < text.size()) {
      int exponent = 0;
      ++i;
      if (i < text.size() && text[i] == '+')
        ++i;
      const auto result =
          std::from_chars(text.data() + i, text.data() + text.size(), exponent);
      if (result.ec != std::errc{})
        Invalid("numeric exponent is outside the supported range");
      value.scale -= exponent;
    }
    value.Normalize();
    return value;
  }
  void Normalize() {
    const auto first = digits.find_first_not_of('0');
    if (first == std::string::npos) {
      digits = "0";
      scale = 0;
      negative = false;
      return;
    }
    digits.erase(0, first);
    while (digits.size() > 1 && digits.back() == '0') {
      digits.pop_back();
      --scale;
    }
  }
};

int Magnitude(const Decimal& a, const Decimal& b) {
  if (a.digits == "0" || b.digits == "0")
    return (a.digits != "0") - (b.digits != "0");
  const auto ae = static_cast<int>(a.digits.size()) - a.scale;
  const auto be = static_cast<int>(b.digits.size()) - b.scale;
  if (ae != be)
    return ae < be ? -1 : 1;
  for (std::size_t i = 0; i < std::max(a.digits.size(), b.digits.size()); ++i) {
    const char x = i < a.digits.size() ? a.digits[i] : '0';
    const char y = i < b.digits.size() ? b.digits[i] : '0';
    if (x != y)
      return x < y ? -1 : 1;
  }
  return 0;
}

int Compare(const Decimal& a, const Decimal& b) {
  if (a.negative != b.negative)
    return a.negative ? -1 : 1;
  return (a.negative ? -1 : 1) * Magnitude(a, b);
}

int NaturalCompare(std::string_view a, std::string_view b) {
  if (a.size() != b.size())
    return a.size() < b.size() ? -1 : 1;
  return a.compare(b);
}

void Subtract(std::string& a, std::string_view b) {
  int borrow = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    int digit = a[a.size() - 1 - i] - '0' - borrow;
    if (i < b.size())
      digit -= b[b.size() - 1 - i] - '0';
    borrow = digit < 0;
    a[a.size() - 1 - i] = '0' + digit + 10 * borrow;
  }
  const auto first = a.find_first_not_of('0');
  a = first == std::string::npos ? "0" : a.substr(first);
}

std::pair<std::string, std::string> Divide(const Decimal& value,
                                           const Decimal& divisor) {
  const int scale = std::max(value.scale, divisor.scale);
  std::string numerator = value.digits + std::string(scale - value.scale, '0');
  std::string denominator =
      divisor.digits + std::string(scale - divisor.scale, '0');
  std::string remainder{"0"};
  std::string quotient;
  for (const char digit : numerator) {
    if (remainder == "0")
      remainder.clear();
    remainder += digit;
    unsigned count = 0;
    while (NaturalCompare(remainder, denominator) >= 0)
      Subtract(remainder, denominator), ++count;
    if (count || !quotient.empty())
      quotient += static_cast<char>('0' + count);
  }
  return {quotient.empty() ? "0" : quotient, remainder};
}

bool Multiple(const Decimal& value, const Decimal& divisor) {
  return value.digits == "0" || Divide(value, divisor).second == "0";
}

std::string Increment(std::string value) {
  for (std::size_t i = value.size(); i > 0; --i) {
    if (value[i - 1] != '9') {
      ++value[i - 1];
      return value;
    }
    value[i - 1] = '0';
  }
  return "1" + value;
}

Decimal Product(const Decimal& value, std::string_view factor) {
  std::vector<unsigned> digits(value.digits.size() + factor.size());
  for (std::size_t i = 0; i < value.digits.size(); ++i)
    for (std::size_t j = 0; j < factor.size(); ++j)
      digits[i + j + 1] += (value.digits[i] - '0') * (factor[j] - '0');
  for (std::size_t i = digits.size() - 1; i > 0; --i) {
    digits[i - 1] += digits[i] / 10;
    digits[i] %= 10;
  }
  Decimal result{"", value.scale, false};
  for (unsigned digit : digits)
    result.digits += static_cast<char>('0' + digit);
  result.Normalize();
  return result;
}

Decimal AddUnit(const Decimal& value, int scale) {
  Decimal next = value;
  next.negative = false;
  if (next.digits == "0")
    next = Decimal{"1", scale, false};
  else {
    next.digits.append(scale - next.scale, '0');
    next.scale = scale;
    bool carry = true;
    for (std::size_t i = next.digits.size(); i && carry; --i) {
      carry = next.digits[i - 1] == '9';
      next.digits[i - 1] = carry ? '0' : next.digits[i - 1] + 1;
    }
    if (carry)
      next.digits.insert(0, 1, '1');
    next.Normalize();
  }
  return next;
}

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

class NumberLexeme final : public JsonSchemaLexeme {
public:
  NumberLexeme(const json::Value& schema, bool integer)
      : JsonSchemaLexeme("0123456789-."), integer_(integer) {
    for (const auto* key : {"minimum", "maximum", "exclusiveMinimum",
                            "exclusiveMaximum", "multipleOf"}) {
      const auto* entry = schema.find(key);
      if (!entry)
        continue;
      if (!entry->is_number() || !std::isfinite(entry->as_double()))
        Invalid(std::string(key) + " must be a finite number");
      const auto value = Decimal::Parse(entry->dump());
      if (std::string_view(key) == "multipleOf") {
        if (value.negative || value.digits == "0")
          Invalid("multipleOf must be positive");
        multiple_ = value;
        continue;
      }
      const bool lower = std::string_view(key) == "minimum" ||
                         std::string_view(key) == "exclusiveMinimum";
      auto& bound = lower ? lower_ : upper_;
      const bool exclusive = std::string_view(key).starts_with("exclusive");
      const int comparison = bound ? Compare(value, bound->value) : 0;
      if (!bound || (lower ? comparison > 0 : comparison < 0) ||
          (comparison == 0 && exclusive))
        bound = Bound{value, exclusive};
    }
    if (lower_ && upper_) {
      const int comparison = Compare(lower_->value, upper_->value);
      if (comparison > 0 ||
          (comparison == 0 &&
           (lower_->exclusive || upper_->exclusive || !Valid(lower_->value))))
        throw JsonSchemaEmpty(
            "JSON Schema: numeric constraints describe an empty interval");
    }
    if (integer_) {
      if (!multiple_)
        multiple_ = Decimal{"1", 0, false};
      if (multiple_->scale > 0) {
        const int scale = multiple_->scale;
        multiple_->scale = 0;
        for (const int factor : {2, 5}) {
          for (int count = 0; count < scale; ++count) {
            const auto division =
                Divide(*multiple_, Decimal{std::to_string(factor), 0, false});
            if (division.second != "0")
              break;
            multiple_->digits = division.first;
          }
        }
        multiple_->Normalize();
      }
    }
    if (lower_ && upper_ && multiple_ && !GridPoint(*lower_, *upper_))
      throw JsonSchemaEmpty(
          "JSON Schema: numeric constraints contain no multipleOf value");
  }
  bool AcceptValue(const json::Value& value) const override {
    return value.is_number() && Valid(Decimal::Parse(value.dump()));
  }

  Match Check(std::string_view bytes) const override {
    if (bytes.empty())
      return {true, false};
    if (bytes.size() > 4096)
      return {};
    const bool negative = bytes.starts_with('-');
    auto text = bytes;
    if (negative)
      text.remove_prefix(1);
    if (text.empty())
      return {!lower_ || lower_->value.negative ||
                  (lower_->value.digits == "0" && !lower_->exclusive),
              false};
    const auto dot = text.find('.');
    if (dot == 0 || (integer_ && dot != std::string_view::npos))
      return {};
    if (text.front() == '0' && text.size() > 1 && text[1] != '.')
      return {};
    for (std::size_t i = 0; i < text.size(); ++i)
      if ((text[i] < '0' || text[i] > '9') && (text[i] != '.' || i != dot))
        return {};
    const auto value = Decimal::Parse(bytes);
    const bool complete = text.back() != '.' && Valid(value);
    // At the scalar budget there is no room for a completing digit. Do not
    // admit an unfinished prefix which would fail on the next sampling step.
    if (bytes.size() == 4096)
      return {complete, complete};
    // Decimal prefixes cover an interval. Before the decimal point, appending
    // integer digits also shifts that interval by powers of ten.
    Decimal low = value;
    low.negative = false;
    const int places =
        dot == std::string_view::npos ? 0 : text.size() - dot - 1;
    Decimal high = AddUnit(low, places);
    const bool extend_integer =
        dot == std::string_view::npos && text.front() != '0';
    for (unsigned shift = 0; shift <= 1024; ++shift) {
      Decimal first = negative ? high : low;
      Decimal last = negative ? low : high;
      first.negative = negative && first.digits != "0";
      last.negative = negative && last.digits != "0";
      const bool intersects_lower =
          !lower_ || Compare(last, lower_->value) > 0 ||
          (negative && !lower_->exclusive && Compare(last, lower_->value) == 0);
      const bool intersects_upper = !upper_ ||
                                    Compare(first, upper_->value) < 0 ||
                                    (!negative && !upper_->exclusive &&
                                     Compare(first, upper_->value) == 0);
      if (intersects_lower && intersects_upper) {
        Bound lower{first, negative}, upper{last, !negative};
        if (lower_ &&
            (Compare(lower_->value, lower.value) > 0 ||
             (Compare(lower_->value, lower.value) == 0 && lower_->exclusive)))
          lower = *lower_;
        if (upper_ &&
            (Compare(upper_->value, upper.value) < 0 ||
             (Compare(upper_->value, upper.value) == 0 && upper_->exclusive)))
          upper = *upper_;
        if (!multiple_ || GridPoint(lower, upper))
          return {true, complete};
      }
      if (!extend_integer || (negative && !intersects_lower) ||
          (!negative && !intersects_upper))
        break;
      --low.scale;
      --high.scale;
    }
    return {complete, complete};
  }

private:
  struct Bound {
    Decimal value;
    bool exclusive;
  };
  bool GridPoint(Bound low, Bound high) const {
    if (low.value.negative && !high.value.negative && high.value.digits != "0")
      return true;  // Zero is inside.
    if (low.value.negative) {
      std::swap(low, high);
      low.value.negative = false;
      high.value.negative = false;
    }
    auto [quotient, remainder] = Divide(low.value, *multiple_);
    if (remainder != "0" || low.exclusive)
      quotient = Increment(std::move(quotient));
    const auto candidate = Product(*multiple_, quotient);
    const auto comparison = Compare(candidate, high.value);
    return comparison < 0 || (comparison == 0 && !high.exclusive);
  }
  bool Valid(const Decimal& value) const {
    if (integer_ && value.scale > 0)
      return false;
    if (lower_ &&
        Compare(value, lower_->value) < static_cast<int>(lower_->exclusive))
      return false;
    if (upper_ &&
        Compare(value, upper_->value) > -static_cast<int>(upper_->exclusive))
      return false;
    return !multiple_ || Multiple(value, *multiple_);
  }
  bool integer_;
  std::optional<Bound> lower_, upper_;
  std::optional<Decimal> multiple_;
};

// Formats follow JSON Schema's RFC-backed string formats. The grammar validates
// decoded strings, so an escaped character has exactly the same meaning as its
// UTF-8 spelling.
std::string FormatPattern(std::string_view format) {
  const std::string leap =
      R"((?:[0-9]{2}(?:0[48]|[2468][048]|[13579][26])|(?:[02468][048]|[13579][26])00))";
  const std::string date =
      R"((?:[0-9]{4}-(?:(?:0[13578]|1[02])-(?:0[1-9]|[12][0-9]|3[01])|(?:0[469]|11)-(?:0[1-9]|[12][0-9]|30)|02-(?:0[1-9]|1[0-9]|2[0-8]))|)" +
      leap + R"(-02-29))";
  const std::string time =
      R"((?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)(?:\.[0-9]+)?(?:[zZ]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9]))";
  if (format == "date")
    return "^" + date + "$";
  if (format == "time")
    return "^" + time + "$";
  if (format == "date-time")
    return "^" + date + "[tT]" + time + "$";
  if (format == "uuid")
    return R"(^[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$)";
  if (format == "ipv4")
    return R"(^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])$)";
  if (format == "hostname")
    return R"(^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*\.?$)";
  if (format == "email")
    return R"(^[^@\s]+@[^@\s]+$)";
  if (format == "duration")
    return R"(^P(?:[0-9]+W|(?=[0-9]|T[0-9])(?:[0-9]+Y)?(?:[0-9]+M)?(?:[0-9]+D)?(?:T(?=[0-9])(?:[0-9]+H)?(?:[0-9]+M)?(?:[0-9]+(?:\.[0-9]+)?S)?)?)$)";
  if (format == "ipv6") {
    const std::string group = "[0-9a-fA-F]{1,4}";
    auto groups = [&](int count) {
      std::string result;
      for (int i = 0; i < count; ++i)
        result += (i ? ":" : "") + group;
      return result;
    };
    std::string result = "^(?:" + groups(8);
    for (int left = 0; left < 8; ++left)
      for (int right = 0; left + right < 8; ++right)
        result += "|" + groups(left) + "::" + groups(right);
    auto ipv4 = FormatPattern("ipv4");
    ipv4 = ipv4.substr(1, ipv4.size() - 2);
    result += "|" + groups(6) + ":" + ipv4;
    for (int left = 0; left < 6; ++left)
      for (int right = 0; left + right < 6; ++right)
        result += "|" + groups(left) + "::" + groups(right) +
                  (right ? ":" : "") + ipv4;
    return result + ")$";
  }
  Invalid("unsupported string format");
}

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
    for (const auto* key : {"minLength", "maxLength"}) {
      if (const auto* value = schema.find(key)) {
        if (!value->is_number() || value->as_double() < 0 ||
            value->as_double() > 1048576 ||
            std::floor(value->as_double()) != value->as_double())
          Invalid("string lengths must be nonnegative integers up to 1048576");
        (std::string_view(key) == "minLength" ? minimum_ : maximum_) =
            value->as_size();
      }
    }
    if (minimum_ > maximum_)
      throw JsonSchemaEmpty("JSON Schema: minLength exceeds maxLength");
    std::vector<std::string> patterns;
    for (const auto* key : {"pattern", "format"}) {
      const auto* value = schema.find(key);
      if (!value)
        continue;
      if (!value->is_string())
        Invalid(std::string(key) + " must be a string");
      auto expression = std::string_view(key) == "format"
                            ? FormatPattern(value->str())
                            : value->str();
      if (std::string_view(key) == "format") {
        if (value->str() == "hostname")
          maximum_ = std::min(maximum_, 253U);
      }
      patterns.push_back(std::move(expression));
    }
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

// Native calls can start before </think>. Keep literal marker mentions and
// Markdown code quoted; a complete native function header starts constrained
// tool decoding. Only a small scan state belongs to each speculative branch.
class ReasoningLexeme final : public JsonSchemaLexeme {
public:
  ReasoningLexeme(std::string tool_prefix, ReasoningEnd end)
      : tool_prefix_(std::move(tool_prefix)),
        end_(end),
        tool_failure_(tool_prefix_.size()) {
    for (std::size_t i = 1, matched = 0; i < tool_prefix_.size(); ++i) {
      while (matched && tool_prefix_[i] != tool_prefix_[matched])
        matched = tool_failure_[matched - 1];
      if (tool_prefix_[i] == tool_prefix_[matched])
        ++matched;
      tool_failure_[i] = matched;
    }
  }
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
    constexpr std::string_view explicit_end = "</think>";
    constexpr std::array<std::uint32_t, 8> explicit_failure{};
    state.explicit_prefix =
        advance(state.explicit_prefix, explicit_end, explicit_failure);
    state.tool_prefix =
        state.inline_quote || state.fence
            ? 0
            : advance(state.tool_prefix, tool_prefix_, tool_failure_);
    if (state.explicit_prefix == explicit_end.size())
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
  std::string tool_prefix_;
  ReasoningEnd end_;
  std::vector<std::uint32_t> tool_failure_;
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
    std::string tool_prefix, ReasoningEnd end) {
  return std::make_shared<ReasoningLexeme>(std::move(tool_prefix), end);
}

json::Value JsonSchemaLexeme::IntersectMultipleOf(const json::Value& a,
                                                  const json::Value& b) {
  for (const auto* value : {&a, &b})
    if (!value->is_number() || !std::isfinite(value->as_double()) ||
        value->as_double() <= 0)
      Invalid("multipleOf must be a positive finite number");
  auto left = Decimal::Parse(a.dump()), right = Decimal::Parse(b.dump());
  // Put both steps on the same integer grid, then compute the exact LCM.
  const int scale = std::max(left.scale, right.scale);
  left.digits.append(scale - left.scale, '0');
  right.digits.append(scale - right.scale, '0');
  left.scale = right.scale = 0;
  auto x = left, y = right;
  while (y.digits != "0") {
    auto remainder = Divide(x, y).second;
    x = std::move(y);
    y = Decimal{std::move(remainder), 0, false};
  }
  auto common = Product(Decimal{Divide(left, x).first, 0, false}, right.digits);
  common.scale += scale;
  common.Normalize();
  const auto result =
      json::parse(common.digits + "e" + std::to_string(-common.scale));
  if (!result.is_number() || !std::isfinite(result.as_double()) ||
      Compare(Decimal::Parse(result.dump()), common) != 0)
    Invalid("combined multipleOf exceeds the exact schema-number range");
  return result;
}

json::Value JsonSchemaLexeme::Format(std::string_view format) {
  auto schema = json::Value::object();
  schema["pattern"] = FormatPattern(format);
  if (format == "hostname")
    schema["maxLength"] = 253;
  return schema;
}
std::shared_ptr<const JsonSchemaLexeme> JsonSchemaLexeme::Number(
    const json::Value& schema, bool integer) {
  return std::make_shared<NumberLexeme>(schema, integer);
}
}  // namespace gufo::sampling
