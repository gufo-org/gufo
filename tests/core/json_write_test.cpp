// Compares the Rust JSON writer in src/core/json (through gufo_json_dump)
// with json::Value::dump and tojson in both number modes: the same bytes for
// every text gufo::json::parse accepts, and the same rejection otherwise. The
// texts are grids of numbers and strings, generated documents and their
// seeded mutations. Any difference fails; the writers have no known ones for
// parsed values. (C++ values built in code may hold invalid UTF-8, which a
// Rust String cannot.)

#include <gufo/json.h>

#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"

namespace {

namespace json = gufo::json;
using Random = std::mt19937_64;

[[noreturn]] void Fail(const std::string& message) {
  std::fprintf(stderr, "json_write_test: %s\n", message.c_str());
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

struct Written {
  GufoJsonParse status = GUFO_JSON_INTERNAL;
  std::string text, error;
};

Written RustDump(std::string_view text, std::uint32_t flags) {
  Written written;
  const auto keep = [](void* context, const std::uint8_t* data,
                       std::size_t len) noexcept {
    static_cast<std::string*>(context)->assign(
        reinterpret_cast<const char*>(data), len);
  };
  // Both callbacks share the context; only one of them is called.
  std::string sink;
  written.status =
      gufo_json_dump(reinterpret_cast<const std::uint8_t*>(text.data()),
                     text.size(), flags, keep, keep, &sink);
  (written.status == GUFO_JSON_COMPLETE ? written.text : written.error) =
      std::move(sink);
  return written;
}

struct Totals {
  std::size_t compared = 0, accepted = 0;
};

constexpr std::array<std::uint32_t, 4> kFlags = {
    0, GUFO_JSON_DUMP_AS_PARSED, GUFO_JSON_DUMP_SPACED,
    GUFO_JSON_DUMP_AS_PARSED | GUFO_JSON_DUMP_SPACED};

void Compare(Totals& totals, std::string_view text) {
  ++totals.compared;
  json::Value value;
  std::string error;
  try {
    value = json::parse(text);
  } catch (const std::runtime_error& failure) {
    error = failure.what();
  }
  for (const auto flags : kFlags) {
    const auto got = RustDump(text, flags);
    if (!error.empty()) {
      if (got.status != GUFO_JSON_REJECTED ||
          "JSON parse error: " + got.error != error)
        Fail("\"" + Escape(text) + "\": C++ rejected it with \"" + error +
             "\", Rust gave status " + std::to_string(got.status) + " \"" +
             Escape(got.error) + "\"");
      return;
    }
    const auto numbers = (flags & GUFO_JSON_DUMP_AS_PARSED) != 0
                             ? json::Value::Numbers::kAsParsed
                             : json::Value::Numbers::kCanonical;
    const auto want = (flags & GUFO_JSON_DUMP_SPACED) != 0
                          ? value.tojson(numbers)
                          : value.dump(numbers);
    if (got.status != GUFO_JSON_COMPLETE || got.text != want)
      Fail("\"" + Escape(text) + "\" with flags " + std::to_string(flags) +
           ": expected \"" + Escape(want) + "\", got status " +
           std::to_string(got.status) + " \"" + Escape(got.text) + "\"");
  }
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

std::string Shortest(double value) {
  char buffer[64];
  const auto [end, error] =
      std::to_chars(buffer, buffer + sizeof(buffer), value);
  if (error != std::errc{})
    Fail("to_chars failed");
  return {buffer, end};
}

std::string Printf(const char* format, double value) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), format, value);
  return buffer;
}

// ---- Grids ----

// Integers near 2^53 and the int64 range, powers of ten across the double
// range, every digit count on both sides of the fixed/scientific choice,
// subnormals and the extremes, written in several spellings.
void Numbers(Totals& totals) {
  std::vector<std::string> tokens = {"0",
                                     "-0",
                                     "0.0",
                                     "-0.0",
                                     "0e5",
                                     "1",
                                     "-1",
                                     "1.0",
                                     "1E2",
                                     "1e+2",
                                     "1e-2",
                                     "10",
                                     "100000",
                                     "1e5",
                                     "0.1",
                                     "0.2",
                                     "0.3",
                                     "1e21",
                                     "1e+21",
                                     "1e-7",
                                     "1e16",
                                     "1e15",
                                     "123.456",
                                     "-2.5",
                                     "0.001",
                                     "0.0001",
                                     "1e308",
                                     "-1e308",
                                     "5e-324",
                                     "2.2250738585072014e-308",
                                     "1.7976931348623157e308",
                                     "4.9406564584124654e-324",
                                     "2.225073858507201e-308"};
  constexpr std::int64_t kTwo53 = std::int64_t{1} << 53;
  for (std::int64_t base :
       {kTwo53, kTwo53 * 2, kTwo53 * 4, std::int64_t{1} << 62}) {
    for (std::int64_t delta = -5; delta <= 5; ++delta) {
      tokens.push_back(std::to_string(base + delta));
      tokens.push_back(std::to_string(-(base + delta)));
      tokens.push_back(std::to_string(base + delta) + ".0");
    }
  }
  tokens.push_back("9223372036854775807");
  tokens.push_back("-9223372036854775808");
  tokens.push_back("18446744073709551616");
  for (int exponent = -325; exponent <= 309; ++exponent) {
    for (const char* mantissa :
         {"1", "-1", "1.5", "9.99", "123456789", "1.2345678901234567",
          "4.9406564584124654", "17976931348623157"}) {
      tokens.push_back(std::string(mantissa) + "e" + std::to_string(exponent));
    }
  }
  for (int digits = 1; digits <= 25; ++digits) {
    std::string whole(static_cast<std::size_t>(digits), '7');
    tokens.push_back(whole);
    tokens.push_back("0." + std::string(static_cast<std::size_t>(digits), '0') +
                     "7");
    tokens.push_back("1" + std::string(static_cast<std::size_t>(digits), '0'));
    tokens.push_back("12" + std::string(static_cast<std::size_t>(digits), '0'));
    tokens.push_back("0." + whole);
  }
  for (const auto& token : tokens) {
    Compare(totals, token);
    Compare(totals, "[" + token + "]");
  }
  // Powers of two, whose lower neighbour is nearer than the upper one, and
  // their neighbours.
  for (int exponent = -1074; exponent <= 1023; ++exponent) {
    const double power = std::ldexp(1.0, exponent);
    for (const double value :
         {power, std::nextafter(power, 0.0),
          std::nextafter(power, std::numeric_limits<double>::infinity()),
          -power}) {
      if (std::isfinite(value) && value != 0)
        Compare(totals, Shortest(value));
    }
  }
  // Random doubles: uniform bit patterns, integers, and short decimals, each
  // spelled shortest, with 17 digits and in fixed notation.
  Random random(0x6e756d73);
  for (int i = 0; i < 40000; ++i) {
    double value = 0;
    switch (Below(random, 4)) {
      case 0: {
        value = std::bit_cast<double>(random());
        break;
      }
      case 1:
        value = static_cast<double>(static_cast<std::int64_t>(random()) >>
                                    Below(random, 64));
        break;
      case 2:
        value = static_cast<double>(Below(random, 100000)) *
                std::pow(10.0, static_cast<double>(Below(random, 60)) - 30);
        break;
      default:
        value = std::ldexp(static_cast<double>(Below(random, 1 << 20)),
                           static_cast<int>(Below(random, 200)) - 100);
    }
    if (!std::isfinite(value))
      continue;
    Compare(totals, Shortest(value));
    Compare(totals, Printf("%.17g", value));
    if (std::fabs(value) < 1e25 && std::fabs(value) > 1e-25)
      Compare(totals, Printf("%.30f", value));
  }
}

// Every byte raw and escaped, and multi-byte characters.
void Strings(Totals& totals) {
  for (int c = 0; c < 0x80; ++c) {
    char escape[16];
    std::snprintf(escape, sizeof(escape), "\\u%04x", c);
    Compare(totals, std::string("\"a") + escape + "b\"");
    Compare(totals, std::string("{\"") + escape + "\":0}");
    Compare(totals, std::string("\"") + static_cast<char>(c) + "\"");
  }
  for (const char* text :
       {"\"\\\"\\\\\\/\\b\\f\\n\\r\\t\"", "\"\\u00e9\\u20ac\\ud83d\\ude00\"",
        "\"\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80\"", "\"\\u007f\x7f\"",
        "\"\\u0080\\u07ff\\u0800\\uffff\"", "\"\\udbff\\udfff\"", "\"\"",
        "{\"\":\"\"}", "\"\xed\x9f\xbf\"", "\"\xf4\x8f\xbf\xbf\""}) {
    Compare(totals, text);
  }
}

// ---- Generated documents ----

struct Generator {
  Random& random;

  std::string Whitespace() {
    static constexpr std::array<const char*, 6> kSpaces = {
        "", "", " ", "\n", "\t", " \r\n "};
    return Pick(random, kSpaces);
  }

  std::string Number() {
    static constexpr std::array<const char*, 16> kNumbers = {
        "0",
        "-0",
        "1",
        "-17",
        "2.50",
        "1e21",
        "1E-7",
        "0.1",
        "100000",
        "9007199254740993",
        "1.5e300",
        "-4.2e-310",
        "123456789012345678901234",
        "3.14159",
        "1e+2",
        "0.000"};
    if (Chance(random, 30)) {
      const auto value = std::bit_cast<double>(random());
      if (std::isfinite(value))
        return Shortest(value);
    }
    return Pick(random, kNumbers);
  }

  std::string String(std::uint64_t most) {
    static constexpr std::array<const char*, 18> kPieces = {"a",
                                                            "Z",
                                                            " ",
                                                            "\\\"",
                                                            "\\\\",
                                                            "\\/",
                                                            "\\n",
                                                            "\\t",
                                                            "\\b",
                                                            "\\u0001",
                                                            "\\u001f",
                                                            "\\u00e9",
                                                            "\\ud83d\\ude00",
                                                            "\xc3\xa9",
                                                            "\xe2\x82\xac",
                                                            "\x7f",
                                                            "\\u0000",
                                                            "{"};
    std::string text = "\"";
    const auto length = Below(random, most + 1);
    for (std::uint64_t i = 0; i < length; ++i)
      text += Pick(random, kPieces);
    return text + "\"";
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  std::string Value(int depth) {
    switch (Below(random, depth > 5 ? 5 : 8)) {
      case 0:
        return "null";
      case 1:
        return Chance(random, 50) ? "true" : "false";
      case 2:
      case 3:
        return Number();
      case 4:
        return String(10);
      case 5:
      case 6: {
        std::string text = "[";
        const auto count = Below(random, 6);
        for (std::uint64_t i = 0; i < count; ++i)
          text +=
              (i ? "," : "") + Whitespace() + Value(depth + 1) + Whitespace();
        return text + "]";
      }
      default: {
        std::string text = "{";
        const auto count = Below(random, 6);
        for (std::uint64_t i = 0; i < count; ++i)
          text += (i ? "," : "") + Whitespace() + "\"k" + std::to_string(i) +
                  "\"" + Whitespace() + ":" + Whitespace() + Value(depth + 1);
        return text + "}";
      }
    }
  }
};

std::string Mutate(Random& random, std::string text) {
  static constexpr std::array<char, 20> kBytes = {
      '"', '\\', '{', '}', '[', ']',    ',',    ':',    ' ',    '0',
      '-', 'e',  '.', '5', 'u', '\x01', '\x7f', '\xc3', '\xa9', '\xff'};
  const auto count = 1 + Below(random, 3);
  for (std::uint64_t m = 0; m < count && !text.empty(); ++m) {
    const auto at = Below(random, text.size());
    switch (Below(random, 4)) {
      case 0:
        text.erase(at, 1);
        break;
      case 1:
        text.insert(at, 1, Pick(random, kBytes));
        break;
      case 2:
        text[at] = Pick(random, kBytes);
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
  Random random(0x77726974);
  Generator generator{random};
  for (int i = 0; i < 3000; ++i) {
    const auto document =
        generator.Whitespace() + generator.Value(0) + generator.Whitespace();
    Compare(totals, document);
    for (int m = 0; m < 10; ++m)
      Compare(totals, Mutate(random, document));
  }
}

}  // namespace

int main() {
  const auto start = std::chrono::steady_clock::now();
  Totals totals;
  Numbers(totals);
  Strings(totals);
  Documents(totals);
  // The mutations must leave most documents valid and reject some.
  if (totals.accepted * 10 < totals.compared * 5 ||
      totals.accepted == totals.compared)
    Fail("unexpected mix: " + std::to_string(totals.accepted) + " of " +
         std::to_string(totals.compared) + " accepted");
  const auto seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  std::printf(
      "json_write_test: %zu texts, %zu written alike by Rust and C++ in 4 "
      "styles, the rest rejected alike, in %.1f s\n",
      totals.compared, totals.accepted, seconds);
  return 0;
}
