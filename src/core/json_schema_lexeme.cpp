#include "src/core/json_schema_lexeme.hpp"

#include <unicode/regex.h>
#include <unicode/uniset.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

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

class NumberLexeme final : public JsonSchemaLexeme {
public:
  NumberLexeme(const json::Value& schema, bool integer) : integer_(integer) {
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
        Invalid("numeric constraints describe an empty interval");
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
      Invalid("numeric constraints contain no multipleOf value");
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
    return R"(^(?=.{1,253}$)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*\.?$)";
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

// Safe alphabet over-approximation for fully anchored regular expressions.
// Unknown constructs keep the full alphabet. This makes escaped Unicode
// feasibility checks cheap for common identifier/date/UUID patterns.
icu::UnicodeSet PatternAlphabet(const icu::UnicodeString& pattern) {
  const icu::UnicodeSet all(0, 0x10ffff);
  int slashes = 0;
  for (int32_t i = pattern.length() - 2; i >= 0 && pattern.charAt(i) == '\\';
       --i)
    ++slashes;
  if (!pattern.startsWith("^") ||
      !((pattern.endsWith("$") && slashes % 2 == 0) ||
        (pattern.endsWith("\\z") && slashes % 2 == 1)))
    return all;
  icu::UnicodeSet alphabet;
  int depth = 0;
  for (int32_t i = 1; i < pattern.length();) {
    const UChar32 c = pattern.char32At(i);
    i += U16_LENGTH(c);
    if (c == '\\') {
      if (i == pattern.length())
        return all;
      const auto escaped = pattern.char32At(i);
      i += U16_LENGTH(escaped);
      if (escaped == 'z' && i == pattern.length())
        continue;
      if ((escaped >= 'a' && escaped <= 'z') ||
          (escaped >= 'A' && escaped <= 'Z') ||
          (escaped >= '0' && escaped <= '9'))
        return all;
      alphabet.add(escaped);
    } else if (c == '[') {
      const auto begin = i - 1;
      bool escaped = false;
      for (; i < pattern.length(); ++i) {
        const auto next = pattern.charAt(i);
        if (!escaped && next == ']')
          break;
        escaped = !escaped && next == '\\';
      }
      if (i == pattern.length())
        return all;
      UErrorCode status = U_ZERO_ERROR;
      icu::UnicodeSet characters(pattern.tempSubStringBetween(begin, ++i),
                                 status);
      if (U_FAILURE(status) || characters.hasStrings())
        return all;
      alphabet.addAll(characters);
    } else if (c == '.') {
      return all;
    } else if (c == '(') {
      if (pattern.charAt(i) == '?' && pattern.charAt(i + 1) != ':' &&
          pattern.charAt(i + 1) != '=' && pattern.charAt(i + 1) != '!')
        return all;
      ++depth;
    } else if (c == ')') {
      --depth;
    } else if (c == '|' && depth == 0) {
      return all;
    } else if (c != '^' && c != '$' && c != '*' && c != '+' && c != '?' &&
               c != '{' && c != '}' && c != '|') {
      alphabet.add(c);
    }
  }
  if (pattern.endsWith("$"))
    alphabet.add('\n').add('\r').add(0x85).add(0x2028).add(0x2029);
  return alphabet;
}

icu::UnicodeString PatternPrefix(const icu::UnicodeString& pattern) {
  icu::UnicodeString prefix;
  // A literal prefix is mandatory only when no alternative or case modifier
  // can bypass it. The full matcher handles the remaining constructs.
  if (!pattern.startsWith("^") || pattern.indexOf('|') >= 0 ||
      pattern.indexOf(icu::UnicodeString("(?i")) >= 0)
    return prefix;
  for (int32_t i = 1; i < pattern.length();) {
    UChar32 c = pattern.char32At(i);
    i += U16_LENGTH(c);
    if (c == '\\') {
      if (i == pattern.length())
        break;
      c = pattern.char32At(i);
      i += U16_LENGTH(c);
      if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
          (c >= '0' && c <= '9'))
        break;
    } else if (c < 128 && std::string_view(".[()|^$*+?{").find(c) !=
                              std::string_view::npos) {
      if (!prefix.isEmpty() && (c == '*' || c == '+' || c == '?' || c == '{'))
        prefix.truncate(prefix.moveIndex32(prefix.length(), -1));
      break;
    }
    prefix.append(c);
  }
  return prefix;
}

icu::UnicodeSet PendingCharacters(std::string_view bytes) {
  icu::UnicodeSet result;
  if (bytes == "\\")
    result.add(0, 0x10ffff);
  else if (bytes.starts_with("\\u")) {
    auto interval = [](std::string_view digits) {
      unsigned value = 0;
      if (!digits.empty())
        std::from_chars(digits.data(), digits.data() + digits.size(), value,
                        16);
      const unsigned shift = 4 * (4 - digits.size());
      return std::pair{value << shift, (value << shift) + (1U << shift) - 1};
    };
    const auto [low, high] =
        interval(bytes.substr(2, std::min<std::size_t>(4, bytes.size() - 2)));
    if (low <= 0xd7ff)
      result.add(low, std::min(high, 0xd7ffU));
    if (high >= 0xe000)
      result.add(std::max(low, 0xe000U), high);
    if (low <= 0xdbff && high >= 0xd800) {
      auto [tail_low, tail_high] = bytes.size() > 8
                                       ? interval(bytes.substr(8))
                                       : std::pair{0xdc00U, 0xdfffU};
      tail_low = std::max(tail_low, 0xdc00U);
      tail_high = std::min(tail_high, 0xdfffU);
      const unsigned first = std::max(low, 0xd800U);
      const unsigned last = std::min(high, 0xdbffU);
      result.add(0x10000 + ((first - 0xd800) << 10) + tail_low - 0xdc00,
                 0x10000 + ((last - 0xd800) << 10) + tail_high - 0xdc00);
    }
  } else {
    const auto first = static_cast<unsigned char>(bytes.front());
    const unsigned width = first < 0xe0 ? 2 : first < 0xf0 ? 3 : 4;
    unsigned value = first & ((1U << (7 - width)) - 1);
    for (std::size_t i = 1; i < bytes.size(); ++i)
      value = (value << 6) | (static_cast<unsigned char>(bytes[i]) & 0x3f);
    const unsigned shift = 6 * (width - bytes.size());
    const unsigned minimum = width == 2 ? 0x80 : width == 3 ? 0x800 : 0x10000;
    result.add(std::max(minimum, value << shift),
               std::min(0x10ffffU, (value << shift) + (1U << shift) - 1));
  }
  result.remove(0xd800, 0xdfff);
  return result;
}

class StringLexeme final : public JsonSchemaLexeme {
public:
  explicit StringLexeme(const json::Value& schema) {
    for (const auto* key : {"minLength", "maxLength"}) {
      if (const auto* value = schema.find(key)) {
        if (!value->is_number() || value->as_double() < 0 ||
            value->as_double() > 1048576 ||
            std::floor(value->as_double()) != value->as_double())
          Invalid(std::string(key) + " must be a nonnegative integer");
        (std::string_view(key) == "minLength" ? minimum_ : maximum_) =
            value->as_size();
      }
    }
    if (minimum_ > maximum_)
      Invalid("minLength exceeds maxLength");
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
        // Unlike a user pattern, a format must consume the whole string.
        // Regex '$' also matches immediately before a terminal newline.
        expression.pop_back();
        expression += "\\z";
      }
      UErrorCode status = U_ZERO_ERROR;
      std::unique_ptr<icu::RegexPattern> pattern(icu::RegexPattern::compile(
          icu::UnicodeString::fromUTF8(expression), 0, status));
      if (U_FAILURE(status))
        Invalid("invalid regular expression");
      alphabet_.retainAll(PatternAlphabet(pattern->pattern()));
      prefixes_.push_back(PatternPrefix(pattern->pattern()));
      patterns_.push_back(std::move(pattern));
    }
  }

  Match Check(std::string_view bytes) const override {
    if (bytes.empty())
      return {true, false};
    if (bytes.front() != '"' || bytes.size() > 65536)
      return {};
    bool complete = false;
    std::string decoded;
    // Reuse the strict JSON decoder. Partial escapes/UTF-8 are held until a
    // full code point is available, before applying Unicode regex/lengths.
    std::size_t cut = 1;
    for (std::size_t i = 1; i < bytes.size();) {
      const unsigned char c = bytes[i];
      if (c == '"') {
        if (i + 1 != bytes.size())
          return {};
        complete = true;
        break;
      }
      if (c < 0x20)
        return {};
      std::size_t width = c < 0x80                 ? 1
                          : c >= 0xc2 && c <= 0xdf ? 2
                          : c >= 0xe0 && c <= 0xef ? 3
                          : c >= 0xf0 && c <= 0xf4 ? 4
                                                   : 0;
      if (c == '\\') {
        if (i + 1 == bytes.size())
          break;
        const auto escaped = bytes[i + 1];
        if (escaped == 'u') {
          width = 6;
          for (std::size_t j = i + 2; j < std::min(i + width, bytes.size());
               ++j)
            if (std::string_view("0123456789abcdefABCDEF").find(bytes[j]) ==
                std::string_view::npos)
              return {};
          if (i + width <= bytes.size()) {
            unsigned code = 0;
            const auto result = std::from_chars(bytes.data() + i + 2,
                                                bytes.data() + i + 6, code, 16);
            if (result.ec != std::errc{} || result.ptr != bytes.data() + i + 6)
              return {};
            if (code >= 0xdc00 && code <= 0xdfff)
              return {};
            if (code >= 0xd800 && code <= 0xdbff) {
              width = 12;
              if (bytes.size() > i + 6 && bytes[i + 6] != '\\')
                return {};
              if (bytes.size() > i + 7 && bytes[i + 7] != 'u')
                return {};
              for (std::size_t j = i + 8; j < std::min(i + width, bytes.size());
                   ++j)
                if (std::string_view("0123456789abcdefABCDEF").find(bytes[j]) ==
                    std::string_view::npos)
                  return {};
              if (bytes.size() > i + 8 && bytes[i + 8] != 'd' &&
                  bytes[i + 8] != 'D')
                return {};
              if (bytes.size() > i + 9 &&
                  std::string_view("cdefCDEF").find(bytes[i + 9]) ==
                      std::string_view::npos)
                return {};
            }
          }
        } else if (std::string_view("\"\\/bfnrt").find(escaped) !=
                   std::string_view::npos)
          width = 2;
        else
          return {};
      }
      if (!width)
        return {};
      if (c >= 0x80) {
        for (std::size_t j = i + 1; j < std::min(i + width, bytes.size());
             ++j) {
          const auto continuation = static_cast<unsigned char>(bytes[j]);
          if (continuation < 0x80 || continuation > 0xbf)
            return {};
          if (j == i + 1 && ((c == 0xe0 && continuation < 0xa0) ||
                             (c == 0xed && continuation > 0x9f) ||
                             (c == 0xf0 && continuation < 0x90) ||
                             (c == 0xf4 && continuation > 0x8f)))
            return {};
        }
      }
      if (i + width > bytes.size())
        break;
      i += width;
      cut = i;
    }
    try {
      decoded = json::parse(std::string(bytes.substr(0, cut)) + "\"").str();
    } catch (const std::exception&) {
      return {};
    }
    const auto text = icu::UnicodeString::fromUTF8(decoded);
    const auto length = static_cast<std::size_t>(text.countChar32());
    if (length > maximum_)
      return {};
    bool matches = length >= minimum_;
    for (const auto& pattern : patterns_) {
      UErrorCode status = U_ZERO_ERROR;
      std::unique_ptr<icu::RegexMatcher> matcher(
          pattern->matcher(text, status));
      if (U_FAILURE(status))
        throw std::runtime_error("JSON pattern matcher allocation failed");
      matcher->setTimeLimit(10, status);
      matcher->setStackLimit(262144, status);
      const bool found = matcher->find(status);
      const bool hit_end = matcher->hitEnd();
      if (U_FAILURE(status))
        throw std::runtime_error("JSON pattern exceeded its evaluation budget");
      if (!found && (complete || length == maximum_ || !hit_end))
        return {};
      matches &= found;
    }
    if (complete)
      return {false, matches};
    if (cut < bytes.size()) {
      if (length == maximum_)
        return {};
      auto candidates = PendingCharacters(bytes.substr(cut));
      candidates.retainAll(alphabet_);
      for (const auto& prefix : prefixes_)
        if (text.length() < prefix.length() && prefix.startsWith(text))
          candidates.retain(prefix.char32At(text.length()));
      // A partial escape/code point must have at least one legal completion.
      // Otherwise a token ending in e.g. "\\uD800" could strand an ASCII
      // identifier grammar despite having passed the token mask.
      for (int32_t range = 0; range < candidates.getRangeCount(); ++range) {
        for (UChar32 cp = candidates.getRangeStart(range);
             cp <= candidates.getRangeEnd(range); ++cp) {
          auto candidate = text;
          candidate.append(cp);
          bool possible = true;
          for (const auto& pattern : patterns_) {
            UErrorCode status = U_ZERO_ERROR;
            std::unique_ptr<icu::RegexMatcher> matcher(
                pattern->matcher(candidate, status));
            if (U_FAILURE(status))
              throw std::runtime_error(
                  "JSON pattern matcher allocation failed");
            matcher->setTimeLimit(10, status);
            matcher->setStackLimit(262144, status);
            const bool found = matcher->find(status);
            if (U_FAILURE(status))
              throw std::runtime_error(
                  "JSON pattern exceeded its evaluation budget");
            if (!found && (length + 1 == maximum_ || !matcher->hitEnd())) {
              possible = false;
              break;
            }
          }
          if (possible)
            return {true, false};
        }
      }
      return {};
    }
    return {true, false};
  }

private:
  std::size_t minimum_{0}, maximum_{1048576};
  icu::UnicodeSet alphabet_{0, 0x10ffff};
  std::vector<icu::UnicodeString> prefixes_;
  std::vector<std::unique_ptr<icu::RegexPattern>> patterns_;
};

}  // namespace

std::shared_ptr<const JsonSchemaLexeme> JsonSchemaLexeme::String(
    const json::Value& schema) {
  return std::make_shared<StringLexeme>(schema);
}
std::shared_ptr<const JsonSchemaLexeme> JsonSchemaLexeme::Number(
    const json::Value& schema, bool integer) {
  return std::make_shared<NumberLexeme>(schema, integer);
}
}  // namespace gufo::sampling
