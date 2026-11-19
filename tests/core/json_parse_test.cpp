// Compares gufo::json::parse, which builds values from the Rust parser in
// src/core/json, with the frozen parser of Gufo 0.35.0
// (tests/core/json_reference.hpp): the same values, number tokens and member
// order, or the same error message, on grids, generated documents and their
// mutations and truncations.

#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <map>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"
#include "tests/core/json_reference.hpp"

namespace {
// Counts down the allocations until one fails; negative never fails.
thread_local std::int64_t fail_allocation_after = -1;
}  // namespace

// Inlined, GCC pairs the replacement new with free and warns.
#if defined(__GNUC__)
#define JSON_TEST_NOINLINE [[gnu::noinline]]
#else
#define JSON_TEST_NOINLINE
#endif

JSON_TEST_NOINLINE void* operator new(std::size_t bytes) {
  if (fail_allocation_after >= 0 && fail_allocation_after-- == 0)
    throw std::bad_alloc();
  if (void* memory = std::malloc(bytes ? bytes : 1))
    return memory;
  throw std::bad_alloc();
}
JSON_TEST_NOINLINE void operator delete(void* memory) noexcept {
  std::free(memory);
}
JSON_TEST_NOINLINE void operator delete(void* memory, std::size_t) noexcept {
  std::free(memory);
}
#undef JSON_TEST_NOINLINE

namespace {

namespace json = gufo::json;
namespace reference = gufo::json_reference;

using Random = std::mt19937_64;

[[noreturn]] void Fail(const std::string& message) {
  std::fprintf(stderr, "json_parse_test: %s\n", message.c_str());
  std::exit(1);
}

std::string Escape(std::string_view text) {
  std::string out;
  for (const unsigned char c : text.substr(0, 400)) {
    if (c >= 0x20 && c < 0x7f && c != '\\') {
      out += static_cast<char>(c);
    } else {
      char buffer[8];
      std::snprintf(buffer, sizeof(buffer), "\\x%02x", c);
      out += buffer;
    }
  }
  if (text.size() > 400)
    out += "... (" + std::to_string(text.size()) + " bytes)";
  return out;
}

bool Same(const json::Value& got, const reference::Value& want) {
  using Type = reference::Value::Type;
  switch (want.type_) {
    case Type::kNull:
      return got.is_null();
    case Type::kBool:
      return got.is_bool() && got.as_bool() == want.bool_;
    case Type::kNumber:
      return got.is_number() &&
             std::bit_cast<std::uint64_t>(got.as_double()) ==
                 std::bit_cast<std::uint64_t>(want.num_) &&
             got.dump(json::Value::Numbers::kAsParsed) == want.str_;
    case Type::kString:
      return got.is_string() && got.str() == want.str_;
    case Type::kArray: {
      if (!got.is_array() || got.items().size() != want.arr_.size())
        return false;
      for (std::size_t i = 0; i < want.arr_.size(); ++i)
        if (!Same(got.items()[i], want.arr_[i]))
          return false;
      return true;
    }
    case Type::kObject: {
      if (!got.is_object() || got.members().size() != want.obj_.size())
        return false;
      for (std::size_t i = 0; i < want.obj_.size(); ++i)
        if (got.members()[i].first != want.obj_[i].first ||
            !Same(got.members()[i].second, want.obj_[i].second))
          return false;
      return true;
    }
  }
  return false;
}

struct Totals {
  std::size_t compared = 0, accepted = 0;
  std::map<std::string, std::size_t> reasons;
};

void Compare(Totals& totals, std::string_view text) {
  ++totals.compared;
  std::string want_error, got_error;
  reference::Value want;
  json::Value got;
  try {
    want = reference::parse(text);
  } catch (const std::runtime_error& error) {
    want_error = error.what();
  }
  try {
    got = json::parse(text);
  } catch (const std::runtime_error& error) {
    got_error = error.what();
  }
  if (want_error != got_error)
    Fail("\"" + Escape(text) + "\": expected " +
         (want_error.empty() ? "a value" : "\"" + want_error + "\"") +
         ", got " + (got_error.empty() ? "a value" : "\"" + got_error + "\""));
  if (!want_error.empty()) {
    ++totals.reasons[want_error];
    return;
  }
  if (!Same(got, want))
    Fail("\"" + Escape(text) + "\": values differ; got " +
         Escape(got.dump(json::Value::Numbers::kAsParsed)));
  ++totals.accepted;
}

std::uint64_t Below(Random& random, std::uint64_t bound) {
  return random() % bound;
}

bool Chance(Random& random, std::uint64_t percent) {
  return Below(random, 100) < percent;
}

template<class T, std::size_t N>
const T& Pick(Random& random, const std::array<T, N>& items) {
  return items[Below(random, N)];
}

// ---- Grids ----

void Literals(Totals& totals) {
  for (const char* text : {"",
                           " ",
                           "\t\n\r ",
                           "null",
                           "true",
                           "false",
                           " null ",
                           "nul",
                           "tru",
                           "fals",
                           "nulll",
                           "truee",
                           "True",
                           "NULL",
                           "[]",
                           "{}",
                           "[ ]",
                           "{ }",
                           "[,]",
                           "[1,]",
                           "[,1]",
                           "{,}",
                           "{\"a\":1,}",
                           "{\"a\"}",
                           "{\"a\":}",
                           "{\"a\" : 1 , \"b\" : 2}",
                           "{1:2}",
                           "{null:1}",
                           "[1 2]",
                           "[1;2]",
                           "{\"a\":1 \"b\":2}",
                           "\"a\" \"b\"",
                           "1 1",
                           "[] []",
                           "\xef\xbb\xbf{}",
                           "\v1",
                           "\f1",
                           "[\"a\"]x",
                           "{\"a\":[}",
                           "{\"a\":{]}",
                           "]",
                           "}",
                           ":",
                           ",",
                           "[[[]]]",
                           "[[[]]",
                           "[[[]]]]",
                           "{\"\":0}",
                           "{\"\":0,\"\":1}",
                           "{\"a\":1,\"b\":2,\"a\":3}",
                           "{\"a\":{\"a\":{\"a\":1}}}",
                           "[{\"a\":1},{\"a\":1}]",
                           "{\"a\\u0000\":1,\"a\":2}",
                           "{\"\\u00e9\":1,\"\xc3\xa9\":2}",
                           "{\"/\":1,\"\\/\":2}"})
    Compare(totals, text);
  // A NUL byte alone, after a value and inside an array.
  using namespace std::string_view_literals;
  for (const std::string_view text : {"\0"sv, "1\0"sv, "[\0]"sv, "[1\0]"sv})
    Compare(totals, text);
}

void Numbers(Totals& totals) {
  const char* signs[] = {"", "-", "+", "--"};
  const char* integers[] = {"",
                            "0",
                            "00",
                            "01",
                            "1",
                            "9",
                            "10",
                            "123456789",
                            "-0",
                            "9007199254740993",
                            "123456789012345678901234567890"};
  const char* fractions[] = {
      "", ".", ".0", ".5", ".000001", ".12345678901234567890", ".e1", "..1"};
  const char* exponents[] = {
      "",     "e",     "E",     "e+",    "e-",    "e0",    "E1",    "e-1",
      "e+1",  "e308",  "e309",  "e-307", "e-308", "e-323", "e-324", "e-325",
      "e400", "e-400", "e0400", "e1e1",  "e+-1",  "e01"};
  for (const char* sign : signs)
    for (const char* integer : integers)
      for (const char* fraction : fractions)
        for (const char* exponent : exponents) {
          const std::string token =
              std::string(sign) + integer + fraction + exponent;
          Compare(totals, token);
          Compare(totals, "[" + token + "]");
          Compare(totals, "{\"n\":" + token + "}");
          Compare(totals, token + "x");
        }
  for (const char* token :
       {"1.7976931348623157e308", "1.7976931348623158e308",
        "1.7976931348623159e308", "-1.7976931348623159e308",
        "179769313486231580793728971405303415079934132710037826936173778980444"
        "968292764750946649017977587207096330286416692887910946555547851940402"
        "630657488671505820681908902000708383250623649306480209900233125378601"
        "0800000",
        "2.2250738585072011e-308", "2.2250738585072014e-308",
        "4.9406564584124654e-324", "2.4703282292062328e-324",
        "2.4703282292062327e-324",
        "0.000000000000000000000000000000000001e-300", "1e-310", "0e-99999",
        "0.0e99999", "-0.0", "-0e-400", "1E+2",
        "100000000000000000000000000000000000000000e-30",
        // 1 + 2^-53, halfway to the next double, and its neighbours.
        "1.00000000000000011102230246251565404236316680908203125",
        "1.00000000000000011102230246251565404236316680908203124",
        "1.00000000000000011102230246251565404236316680908203126",
        "9007199254740993.000000000000000000001",
        "9007199254740992.999999999999999999999", "Infinity", "NaN",
        "-Infinity", "0x10", ".5", "5.", "-.5", "1_000"})
    Compare(totals, token);
  // Halfway decided by a digit after the first 768, which both parsers keep
  // as a nonzero tail only.
  const std::string halfway =
      "1.00000000000000011102230246251565404236316680908203125";
  Compare(totals, halfway + std::string(800, '0') + "1");
  Compare(totals, halfway + std::string(800, '0'));
  Compare(totals,
          halfway.substr(0, halfway.size() - 1) + "4" + std::string(800, '9'));
}

void Strings(Totals& totals) {
  // Every byte alone and before each kind of end.
  for (int b = 0; b < 256; ++b) {
    const std::string byte(1, static_cast<char>(b));
    Compare(totals, "\"" + byte + "\"");
    Compare(totals, "\"" + byte);
    Compare(totals, "\"\\" + byte + "\"");
    Compare(totals, "{\"" + byte + "\":0}");
  }
  // Every pair of bytes from 0x80 on, and every lead byte with each second
  // byte before a third.
  for (int a = 0x80; a < 0x100; ++a)
    for (int b = 0; b < 0x100; ++b) {
      const std::string pair{static_cast<char>(a), static_cast<char>(b)};
      Compare(totals, "\"" + pair + "\"");
      if (a >= 0xe0)
        for (const char third : {'\x80', '\xbf', '"', 'a'})
          Compare(totals, "\"" + pair + third + "\"");
    }
  for (const char* tail : {"\x80\x80", "\x80\x80\x80", "\xbf\xbf\xbf",
                           "\x90\x80\x80", "\x8f\xbf\xbf", "\x80\"", "\x80"})
    for (int lead = 0xf0; lead < 0x100; ++lead)
      Compare(totals,
              "\"" + std::string(1, static_cast<char>(lead)) + tail + "\"");
  // \u escapes: hex digits, surrogates and what follows them.
  for (const char* escape :
       {"\\u", "\\u0", "\\u00", "\\u004", "\\u0041", "\\u004g", "\\uG041",
        "\\u0000", "\\u001f", "\\u007f", "\\u0080", "\\u07ff", "\\u0800",
        "\\uffff", "\\uFFFF", "\\uFfFf", "\\ud7ff", "\\ue000", "\\u20AC"})
    for (const char* end : {"\"", "", "x\"", "\\\"\""})
      Compare(totals, std::string("\"") + escape + end);
  for (const char* high : {"\\ud800", "\\udbff", "\\uDB40", "\\udc00",
                           "\\udfff", "\\ud7ff", "\\ue000"})
    for (const char* low :
         {"", "\\", "\\u", "\\udc0", "\\udc00", "\\udfff", "\\uDC00", "\\ud800",
          "\\udbff", "\\ue000", "\\u0041", "\\n", "x", "\\udc0g", "\\uZZZZ"})
      for (const char* end : {"\"", ""})
        Compare(totals, std::string("\"") + high + low + end);
}

std::string Nested(std::size_t depth, bool objects, std::string_view inner) {
  std::string text;
  for (std::size_t i = 0; i < depth; ++i)
    text += objects ? "{\"k\":" : "[";
  text += inner;
  for (std::size_t i = 0; i < depth; ++i)
    text += objects ? "}" : "]";
  return text;
}

void Depths(Totals& totals) {
  for (const std::size_t depth : {1, 2, 126, 127, 128, 129, 130, 1000})
    for (const bool objects : {false, true})
      for (const char* inner : {"1", "", "[]", "{}", "\"s\""}) {
        const auto text = Nested(depth, objects, inner);
        Compare(totals, text);
        Compare(totals, text.substr(0, text.size() - 1));
      }
  // Mixed arrays and objects.
  std::string mixed;
  for (int i = 0; i < 130; ++i) {
    mixed += i % 2 ? "[" : "{\"a\":";
    if (i == 127 || i == 128 || i == 129)
      Compare(totals, mixed + "0");
  }
}

// ---- Generated documents ----

struct Generator {
  Random& random;
  std::vector<std::string> keys;

  std::string Whitespace() {
    static constexpr std::array<std::string_view, 6> kSpace = {
        "", "", "", " ", "\n  ", "\t\r\n"};
    return std::string(Pick(random, kSpace));
  }

  std::string Number() {
    static constexpr std::array<std::string_view, 12> kTokens = {
        "0",      "-0",    "1",      "-17",
        "3.25",   "1e5",   "2.5E-3", "-0.0",
        "1e-400", "1e400", "1e-310", "12345678901234567890123"};
    if (Chance(random, 50))
      return std::to_string(static_cast<std::int64_t>(random() >> 20) -
                            (std::int64_t{1} << 42));
    if (Chance(random, 30)) {
      char buffer[64];
      std::snprintf(buffer, sizeof(buffer), "%.17g",
                    std::bit_cast<double>(random()));
      const std::string_view text(buffer);
      // NaN and infinities are not JSON; keep them as rejections sometimes.
      if (text.find_first_of("ni") == std::string_view::npos ||
          Chance(random, 10))
        return std::string(text);
    }
    return std::string(Pick(random, kTokens));
  }

  std::string String(std::size_t most) {
    static constexpr std::array<std::string_view, 24> kPieces = {
        "a",
        "Z",
        " ",
        "0",
        "\\n",
        "\\\"",
        "\\\\",
        "\\/",
        "\\t",
        "\\b",
        "\\f",
        "\\r",
        "\\u0041",
        "\\u00e9",
        "\\u20ac",
        "\\ud83d\\ude00",
        "\xc3\xa9",
        "\xe2\x82\xac",
        "\xf0\x9f\x98\x80",
        "\xe4\xb8\xad",
        "\\u0000",
        "\x7f",
        "{",
        "]"};
    std::string text = "\"";
    const auto length = Below(random, most + 1);
    for (std::uint64_t i = 0; i < length; ++i)
      text += Pick(random, kPieces);
    if (Chance(random, 3))
      text += std::string(Below(random, 300), 'x');
    return text + "\"";
  }

  std::string Key() {
    if (!keys.empty() && Chance(random, 3))
      return keys[Below(random, keys.size())];
    auto key = String(4);
    keys.push_back(key);
    return key;
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  std::string Value(int depth) {
    const auto kind = Below(random, depth > 6 ? 5 : 8);
    switch (kind) {
      case 0:
        return "null";
      case 1:
        return Chance(random, 50) ? "true" : "false";
      case 2:
      case 3:
        return Number();
      case 4:
        return String(12);
      case 5:
      case 6: {
        std::string text = "[";
        const auto count = Below(random, Chance(random, 10) ? 40 : 5);
        for (std::uint64_t i = 0; i < count; ++i)
          text +=
              (i ? "," : "") + Whitespace() + Value(depth + 1) + Whitespace();
        return text + "]";
      }
      default: {
        std::string text = "{";
        // Objects past 16 keys take the hash set path.
        const auto count = Below(random, Chance(random, 10) ? 40 : 5);
        const auto saved = keys;
        keys.clear();
        for (std::uint64_t i = 0; i < count; ++i)
          text += (i ? "," : "") + Whitespace() + Key() + Whitespace() + ":" +
                  Whitespace() + Value(depth + 1) + Whitespace();
        keys = saved;
        return text + "}";
      }
    }
  }
};

std::string Mutate(Random& random, std::string text) {
  static constexpr std::array<char, 24> kBytes = {
      '"',  '\\', '{',    '}',    '[',    ']',    ',',    ':',
      ' ',  '0',  '-',    'e',    '.',    'u',    'n',    't',
      '\0', '\n', '\x1f', '\x80', '\xc3', '\xed', '\xf4', '\xff'};
  const auto count = 1 + Below(random, 3);
  for (std::uint64_t m = 0; m < count && !text.empty(); ++m) {
    const auto at = Below(random, text.size());
    switch (Below(random, 5)) {
      case 0:
        text.erase(at, 1);
        break;
      case 1:
        text.insert(at, 1, Pick(random, kBytes));
        break;
      case 2:
        text[at] = Pick(random, kBytes);
        break;
      case 3:
        text.resize(at);
        break;
      default: {
        const auto length = Below(random, text.size() - at) + 1;
        text.insert(at, text.substr(at, length));
      }
    }
  }
  return text;
}

void Documents(Totals& totals) {
  Random random(0x6a736f6e);
  Generator generator{random, {}};
  for (int i = 0; i < 4000; ++i) {
    const auto document =
        generator.Whitespace() + generator.Value(0) + generator.Whitespace();
    Compare(totals, document);
    for (int m = 0; m < 20; ++m)
      Compare(totals, Mutate(random, document));
    if (document.size() < 600)
      for (std::size_t cut = 0; cut < document.size(); ++cut)
        Compare(totals, std::string_view(document).substr(0, cut));
  }
}

// Long strings and many keys take the parser's bulk paths.
void Large(Totals& totals) {
  Random random(7);
  std::string base64;
  constexpr std::string_view kAlphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  for (int i = 0; i < (1 << 20); ++i)
    base64 += kAlphabet[Below(random, kAlphabet.size())];
  const std::string prefix = "{\"image\":\"data:image/png;base64,";
  const std::string body = prefix + base64 + "\"}";
  Compare(totals, body);
  for (const std::size_t at :
       {std::size_t{100}, base64.size() / 2, base64.size() - 1}) {
    for (const char* bad :
         {"\x01", "\xc3", "\\", "\\q", "\"", "\xed\xa0\x80"}) {
      auto text = body;
      text.insert(prefix.size() + at, bad);
      Compare(totals, text);
    }
  }
  std::string keys = "{";
  for (int i = 0; i < 5000; ++i)
    keys += (i ? ",\"" : "\"") + std::to_string(i * 7919 % 100003) +
            "\":" + std::to_string(i);
  Compare(totals, keys + "}");
  Compare(totals, keys + ",\"7919\":1}");
  Compare(totals, keys + ",\"\\u0037919\":1}");
  // Long mantissas that offset exponents beyond 65,535 read as with
  // libstdc++'s std::from_chars, which 0.35.0 shipped with, on every
  // platform. How far std::from_chars reads an exponent differs between
  // standard libraries, so only libstdc++ also compares them with the
  // reference.
  const std::string zeros(700000, '0');
  struct Long {
    std::string number;
    double value;
    bool accepted;
  };
  for (const auto& [number, value, accepted] : std::initializer_list<Long>{
           {"1" + zeros + "e-700001", 0.1, true},
           {"-1" + zeros + "e-700001", -0.1, true},
           {"0." + zeros + "1e700001", 1.0, true},
           {"0." + zeros + "e999999", 0.0, true},
           {"1" + std::string(655300, '0') + "e-655360", 1e-60, true},
           {"1" + std::string(65500, '0') + "e-65536", 1e-36, true},
           {"0e4294967296", 0.0, true},
           {"1" + zeros + "e-1000000", 0.0, false},
           {"1" + zeros + "e-600000", 0.0, false},
           {"1e4294967296", 0.0, false},
           {"1e-4294967296", 0.0, false},
           {"1e268435456", 0.0, false},
           {"-1e-268435456", 0.0, false}}) {
    std::string error;
    json::Value got;
    try {
      got = json::parse(number);
    } catch (const std::runtime_error& rejection) {
      error = rejection.what();
    }
    const bool same = accepted
                          ? error.empty() && got.is_number() &&
                                std::bit_cast<std::uint64_t>(got.as_double()) ==
                                    std::bit_cast<std::uint64_t>(value)
                          : error == "JSON parse error: bad number";
    if (!same)
      Fail("a number of " + std::to_string(number.size()) +
           " characters: expected " +
           (accepted ? json::Value(value).dump() : "a rejection") + ", got " +
           (error.empty() ? got.dump() : "\"" + error + "\""));
#if defined(__GLIBCXX__)
    Compare(totals, "[" + number + "]");
#endif
  }
}

// An exception while building the values or the rejection stops the Rust
// parser and reaches the caller unchanged, whichever allocation throws it.
void BuilderFailures() {
  for (const std::string text :
       {R"({"a":[1,"two",{"three":3.5}],"b":"\u00e9x","c":[[],{}],"d":null})",
        R"({"a":[1,"two"],"b":{"c":"\u00e9"},"a":2})"}) {
    std::string expected;
    try {
      expected = json::parse(text).dump();
    } catch (const std::runtime_error& error) {
      expected = error.what();
    }
    for (std::int64_t after = 0;; ++after) {
      fail_allocation_after = after;
      bool failed = false;
      std::string got;
      try {
        const auto value = json::parse(text);
        failed = fail_allocation_after < 0;
        fail_allocation_after = -1;
        got = value.dump();
      } catch (const std::bad_alloc&) {
        fail_allocation_after = -1;
        continue;
      } catch (const std::runtime_error& error) {
        failed = fail_allocation_after < 0;
        fail_allocation_after = -1;
        got = error.what();
      } catch (const std::exception& error) {
        fail_allocation_after = -1;
        Fail(std::string("a failed allocation threw ") + error.what());
      }
      if (failed)
        Fail("a failed allocation gave " + got);
      if (got != expected)
        Fail("got " + got + " after failed allocations, expected " + expected);
      break;
    }
  }
}

}  // namespace

int main() {
  const auto start = std::chrono::steady_clock::now();
  Totals totals;
  Literals(totals);
  Numbers(totals);
  Strings(totals);
  Depths(totals);
  Documents(totals);
  Large(totals);
  BuilderFailures();
  // Every reason of the 0.35.0 parser was reached.
  const std::size_t reasons = 21;
  if (totals.reasons.size() != reasons) {
    for (const auto& [reason, count] : totals.reasons)
      std::fprintf(stderr, "  %zu x %s\n", count, reason.c_str());
    Fail("expected " + std::to_string(reasons) + " distinct errors, saw " +
         std::to_string(totals.reasons.size()));
  }
  const auto seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  std::printf(
      "json_parse_test: %zu texts match 0.35.0 (%zu values, %zu "
      "rejections over %zu reasons) in %.1f s\n",
      totals.compared, totals.accepted, totals.compared - totals.accepted,
      totals.reasons.size(), seconds);
  return 0;
}
