// Expected results of gufo::json::parse, which builds values from the Rust
// parser in src/core/json: values, number tokens and member order, or the
// message of each rejection, for edge cases of the grammar, numbers, strings,
// nesting and large inputs. Failed allocations in the C++ builder must reach
// the caller.

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "src/core/json.hpp"

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
using namespace std::string_view_literals;
using Pairs = std::initializer_list<std::pair<std::string_view, std::string>>;

int failures = 0;

std::string Escape(std::string_view text) {
  std::string out;
  for (const unsigned char c : text.substr(0, 200)) {
    if (c >= 0x20 && c < 0x7f && c != '\\') {
      out += static_cast<char>(c);
    } else {
      char buffer[8];
      std::snprintf(buffer, sizeof(buffer), "\\x%02x", c);
      out += buffer;
    }
  }
  if (text.size() > 200)
    out += "... (" + std::to_string(text.size()) + " bytes)";
  return out;
}

void Fail(std::string_view text, const std::string& what) {
  ++failures;
  std::fprintf(stderr, "json_parse_test: \"%s\": %s\n", Escape(text).c_str(),
               what.c_str());
}

// The value, or the message of the rejection.
bool Parse(std::string_view text, json::Value& value, std::string& error) {
  try {
    value = json::parse(text);
    return true;
  } catch (const std::runtime_error& rejection) {
    error = rejection.what();
    return false;
  }
}

// `text` parses to the value whose as-parsed dump is `dump`.
void Accepts(std::string_view text, const std::string& dump) {
  json::Value value;
  std::string error;
  if (!Parse(text, value, error))
    return Fail(text, "rejected with \"" + error + "\", expected " + dump);
  const auto got = value.dump(json::Value::Numbers::kAsParsed);
  if (got != dump)
    Fail(text, "gave " + Escape(got) + ", expected " + Escape(dump));
}

// `text` is rejected with "JSON parse error: <reason>".
void Rejects(std::string_view text, const std::string& reason) {
  json::Value value;
  std::string error;
  if (Parse(text, value, error))
    return Fail(text, "accepted, expected \"" + reason + "\"");
  if (error != "JSON parse error: " + reason)
    Fail(text, "gave \"" + error + "\", expected \"" + reason + "\"");
}

// `text` is one number with exactly this double and its own spelling.
void Number(std::string_view text, double expected) {
  json::Value value;
  std::string error;
  if (!Parse(text, value, error))
    return Fail(text, "rejected with \"" + error + "\"");
  if (!value.is_number() || std::bit_cast<std::uint64_t>(value.as_double()) !=
                                std::bit_cast<std::uint64_t>(expected))
    return Fail(text, "gave " + value.dump());
  if (value.dump(json::Value::Numbers::kAsParsed) != text)
    Fail(text, "kept the token " +
                   Escape(value.dump(json::Value::Numbers::kAsParsed)));
}

// `text` is one string decoding to `bytes`.
void Decodes(std::string_view text, std::string_view bytes) {
  json::Value value;
  std::string error;
  if (!Parse(text, value, error))
    return Fail(text, "rejected with \"" + error + "\"");
  if (!value.is_string() || value.str() != bytes)
    Fail(text, "decoded to " + Escape(value.is_string() ? value.str() : ""));
}

void Grammar() {
  for (const auto& [text, dump] : Pairs{
           {" null ", "null"},
           {"\t\n\r true", "true"},
           {"false", "false"},
           {"[ ]", "[]"},
           {"{ }", "{}"},
           {"[[[]]]", "[[[]]]"},
           {"{\"a\" : 1 , \"b\" : 2}", R"({"a":1,"b":2})"},
           {"{\"\":0}", R"({"":0})"},
           // Keys are unique per object only.
           {R"({"a":{"a":{"a":1}}})", R"({"a":{"a":{"a":1}}})"},
           {R"([{"a":1},{"a":1}])", R"([{"a":1},{"a":1}])"},
       })
    Accepts(text, dump);
  for (const auto& [text, reason] : Pairs{
           {"", "unexpected end of input"},
           {"\t\n\r ", "unexpected end of input"},
           {"[1,", "unexpected end of input"},
           {"{\"a\":", "unexpected end of input"},
           // Only space, tab, LF and CR are whitespace; a BOM, VT, FF or NUL
           // starts a number.
           {"\xef\xbb\xbf{}", "bad number"},
           {"\v1", "bad number"},
           {"\f1", "bad number"},
           {"\0"sv, "bad number"},
           {"[\0]"sv, "bad number"},
           {"1\0"sv, "trailing input"},
           {"[1\0]"sv, "expected ',' or ']' in array"},
           {"nul", "bad literal"},
           {"tru", "bad literal"},
           {"fals", "bad literal"},
           {"True", "bad number"},
           {"NULL", "bad number"},
           {"nulll", "trailing input"},
           {"truee", "trailing input"},
           {"1 1", "trailing input"},
           {"\"a\" \"b\"", "trailing input"},
           {"[] []", "trailing input"},
           {"[\"a\"]x", "trailing input"},
           {"[[[]]]]", "trailing input"},
           {"]", "bad number"},
           {"}", "bad number"},
           {":", "bad number"},
           {",", "bad number"},
           {"[,]", "bad number"},
           {"[1,]", "bad number"},
           {"[,1]", "bad number"},
           {"[1 2]", "expected ',' or ']' in array"},
           {"[1;2]", "expected ',' or ']' in array"},
           {"[[[]]", "unterminated array"},
           {"{,}", "expected object key string"},
           {"{\"a\":1,}", "expected object key string"},
           {"{1:2}", "expected object key string"},
           {"{null:1}", "expected object key string"},
           {"{\"a\":{]}", "expected object key string"},
           {"{\"a\"}", "expected ':' after key"},
           {"{\"a\":}", "bad number"},
           {"{\"a\":[}", "bad number"},
           {"{\"a\":1 \"b\":2}", "expected ',' or '}' in object"},
           {"{\"a\":1", "unterminated object"},
           // Keys compare decoded: escapes, an escaped solidus and UTF-8.
           {R"({"a":1,"b":2,"a":3})", "duplicate object key"},
           {R"({"":0,"":1})", "duplicate object key"},
           {R"({"/":1,"\/":2})", "duplicate object key"},
           {"{\"\\u00e9\":1,\"\xc3\xa9\":2}", "duplicate object key"},
       })
    Rejects(text, reason);
  // A NUL in a key keeps it distinct from its prefix.
  json::Value value;
  std::string error;
  if (!Parse(R"({"a\u0000":1,"a":2})", value, error) ||
      value.members().size() != 2 || value.members()[0].first != "a\0"sv ||
      value.members()[1].first != "a")
    Fail(R"({"a\u0000":1,"a":2})", "keys differ");
}

void Numbers() {
  const double max = std::numeric_limits<double>::max();
  const double above_one = std::nextafter(1.0, 2.0);
  for (const auto& [text, value] :
       std::initializer_list<std::pair<std::string_view, double>>{
           {"0", 0.0},
           {"-0", -0.0},
           {"-0.0", -0.0},
           {"1E+2", 100.0},
           {"2.50", 2.5},
           {"9007199254740993", 9007199254740992.0},
           {"123456789012345678901234567890", 123456789012345678901234567890.0},
           {"100000000000000000000000000000000000000000e-30", 1e11},
           // Zeros take any exponent; subnormals stay.
           {"0e-99999", 0.0},
           {"0.0e99999", 0.0},
           {"-0e-400", -0.0},
           {"1e-310", 1e-310},
           {"4.9406564584124654e-324", 4.9406564584124654e-324},
           {"2.4703282292062328e-324", 4.9406564584124654e-324},
           {"2.2250738585072011e-308", 2.2250738585072011e-308},
           {"1.7976931348623157e308", max},
           {"1.7976931348623158e308", max},
           // 1 + 2^-53 is halfway between 1 and the next double: ties to even.
           {"1.00000000000000011102230246251565404236316680908203125", 1.0},
           {"1.00000000000000011102230246251565404236316680908203124", 1.0},
           {"1.00000000000000011102230246251565404236316680908203126",
            above_one},
           {"9007199254740992.999999999999999999999", 9007199254740992.0},
       })
    Number(text, value);
  // A halfway case decided by a digit after the first 768.
  const std::string halfway =
      "1.00000000000000011102230246251565404236316680908203125";
  Number(halfway + std::string(800, '0') + "1", above_one);
  Number(halfway + std::string(800, '0'), 1.0);
  Number(halfway.substr(0, halfway.size() - 1) + "4" + std::string(800, '9'),
         1.0);
  Accepts(R"({"n":-0.0,"m":[1e5,2.50]})", R"({"n":-0.0,"m":[1e5,2.50]})");
  // Out of range, including results that round to zero, is a bad number.
  for (const std::string_view text :
       {"-"sv,
        "+1"sv,
        "--1"sv,
        "00"sv,
        "01"sv,
        "-01"sv,
        ".5"sv,
        "5."sv,
        "-.5"sv,
        "1.e1"sv,
        "..1"sv,
        "1e"sv,
        "1e+"sv,
        "1e+-1"sv,
        "Infinity"sv,
        "-Infinity"sv,
        "NaN"sv,
        "1e309"sv,
        "1.7976931348623159e308"sv,
        "-1.7976931348623159e308"sv,
        "1e400"sv,
        "1e-400"sv,
        "2.4703282292062327e-324"sv,
        "0.000000000000000000000000000000000001e-300"sv})
    Rejects(text, "bad number");
  for (const std::string_view text :
       {"0x10"sv, "1_000"sv, "1e1e1"sv, "1x"sv, "-0-"sv})
    Rejects(text, "trailing input");
}

void Strings() {
  Decodes(R"("")", "");
  Decodes(R"("\u0000")", "\0"sv);
  Decodes(R"("\"\\\/\b\f\n\r\t")", "\"\\/\b\f\n\r\t");
  Decodes("\"\x7f\"", "\x7f");
  Decodes(R"("\u007f\u0080\u07ff\u0800\uFfFf\ue000\u20AC")",
          "\x7f\xc2\x80\xdf\xbf\xe0\xa0\x80\xef\xbf\xbf\xee\x80\x80"
          "\xe2\x82\xac");
  Decodes(R"("\ud83d\ude00\uDB40\uDC00\udbff\udfff")",
          "\xf0\x9f\x98\x80\xf3\xa0\x80\x80\xf4\x8f\xbf\xbf");
  // The extremes of each UTF-8 length pass unchanged.
  const std::string utf8 =
      "\xc2\x80\xdf\xbf\xe0\xa0\x80\xed\x9f\xbf\xee\x80\x80\xef\xbf\xbf"
      "\xf0\x90\x80\x80\xf4\x8f\xbf\xbf";
  Decodes("\"" + utf8 + "\"", utf8);
  for (const auto& [text, reason] : Pairs{
           {"\"", "unterminated string"},
           {"\"abc", "unterminated string"},
           {"\"\\u0041", "unterminated string"},
           {"\"\\", "bad escape"},
           {R"("\q")", "bad escape"},
           {R"("\x41")", "bad escape"},
           {R"("\U0041")", "bad escape"},
           {R"("\u")", "bad \\u"},
           {R"("\u004")", "bad hex"},
           {"\"\\u004", "bad \\u"},
           {R"("\u004g")", "bad hex"},
           {R"("\uG041")", "bad hex"},
           {R"("\ud800")", "unpaired high surrogate"},
           {R"("\ud800x")", "unpaired high surrogate"},
           {R"("\ud800\n")", "unpaired high surrogate"},
           {R"("\ud800\u")", "bad \\u"},
           {R"("\ud800\udc0g")", "bad hex"},
           {R"("\ud800\ud800")", "invalid low surrogate"},
           {R"("\ud800\u0041")", "invalid low surrogate"},
           {R"("\udbff\ue000")", "invalid low surrogate"},
           {R"("\udc00")", "unpaired low surrogate"},
           {R"("\udfff\ud800")", "unpaired low surrogate"},
           {"\"\x01\"", "unescaped control character"},
           {"\"\x1f\"", "unescaped control character"},
           {"\"\t\"", "unescaped control character"},
           {"\"\0\""sv, "unescaped control character"},
           {"{\"\n\":0}", "unescaped control character"},
           // Stray and overlong leads, surrogates, past U+10FFFF, truncated
           // sequences, also at the end of the input.
           {"\"\x80\"", "invalid UTF-8"},
           {"\"\xbf\"", "invalid UTF-8"},
           {"\"\xc0\xaf\"", "invalid UTF-8"},
           {"\"\xc1\xbf\"", "invalid UTF-8"},
           {"\"\xe0\x9f\xbf\"", "invalid UTF-8"},
           {"\"\xed\xa0\x80\"", "invalid UTF-8"},
           {"\"\xf0\x8f\xbf\xbf\"", "invalid UTF-8"},
           {"\"\xf4\x90\x80\x80\"", "invalid UTF-8"},
           {"\"\xf5\x80\x80\x80\"", "invalid UTF-8"},
           {"\"\xff\"", "invalid UTF-8"},
           {"\"\xc3\"", "invalid UTF-8"},
           {"\"\xe2\x82\"", "invalid UTF-8"},
           {"\"\xc3", "invalid UTF-8"},
           {"\"\xf0\x9f\x98", "invalid UTF-8"},
           {"{\"\xc3\x28\":0}", "invalid UTF-8"},
       })
    Rejects(text, reason);
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

void Depths() {
  constexpr auto kDepth = "maximum nesting depth exceeded";
  for (const bool objects : {false, true}) {
    const auto deepest = Nested(128, objects, "1");
    Accepts(deepest, deepest);
    Accepts(Nested(127, objects, "[]"), Nested(127, objects, "[]"));
    Rejects(deepest.substr(0, deepest.size() - 1),
            objects ? "unterminated object" : "unterminated array");
    Rejects(Nested(128, objects, "[]"), kDepth);
    Rejects(Nested(128, objects, "{}"), kDepth);
    Rejects(Nested(129, objects, "1"), kDepth);
    // The limit is reached before the missing end.
    const auto deeper = Nested(1000, objects, "");
    Rejects(deeper.substr(0, deeper.size() - 1), kDepth);
  }
  // Mixed arrays and objects count together.
  std::string mixed;
  for (int i = 0; i < 128; ++i)
    mixed += i % 2 ? "[" : "{\"a\":";
  Rejects(mixed + "0", "unterminated array");
  Rejects(mixed + "{\"a\":0", kDepth);
}

// Long strings and many keys take the parser's bulk paths.
void Large() {
  constexpr std::string_view kAlphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string payload;
  for (std::size_t i = 0; i < (std::size_t{1} << 20); ++i)
    payload += kAlphabet[i * 37 % kAlphabet.size()];
  const std::string prefix = "{\"image\":\"data:image/png;base64,";
  const std::string body = prefix + payload + "\"}";
  json::Value value;
  std::string error;
  if (!Parse(body, value, error) ||
      value.member_str("image") != "data:image/png;base64," + payload)
    Fail(body, "did not decode the payload");
  for (const std::size_t at :
       {std::size_t{100}, payload.size() / 2, payload.size() - 1})
    for (const auto& [bad, reason] :
         Pairs{{"\x01", "unescaped control character"},
               {"\xc3", "invalid UTF-8"},
               {"\xed\xa0\x80", "invalid UTF-8"},
               {"\\q", "bad escape"},
               {"\"", "expected ',' or '}' in object"}}) {
      auto text = body;
      text.insert(prefix.size() + at, bad);
      Rejects(text, reason);
    }
  // Past the parser's switch to a key set.
  std::string keys = "{";
  for (int i = 0; i < 5000; ++i)
    keys += (i ? ",\"" : "\"") + std::to_string(i * 7919 % 100003) +
            "\":" + std::to_string(i);
  if (!Parse(keys + "}", value, error) || value.members().size() != 5000 ||
      value.members()[4999].first != std::to_string(4999 * 7919 % 100003) ||
      value.member_size("7919") != 1)
    Fail(keys + "}", "members differ");
  Rejects(keys + ",\"7919\":1}", "duplicate object key");
  Rejects(keys + ",\"\\u0037919\":1}", "duplicate object key");
}

// An exception while building the values or the rejection stops the Rust
// parser and reaches the caller unchanged, whichever allocation throws it.
void BuilderFailures() {
  for (
      const auto& [text, expected] : Pairs{
          {R"({"a":[1,"two",{"three":3.5}],"b":"\u00e9x","c":[[],{}],"d":null})",
           "{\"a\":[1,\"two\",{\"three\":3.5}],\"b\":\"\xc3\xa9x\",\"c\":[[],{}"
           "],\"d\":null}"},
          {R"({"a":[1,"two"],"b":{"c":"\u00e9"},"a":2})",
           "JSON parse error: duplicate object key"}}) {
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
        Fail(text, std::string("a failed allocation threw ") + error.what());
        break;
      }
      if (failed)
        Fail(text, "a failed allocation gave " + got);
      else if (got != expected)
        Fail(text, "gave " + Escape(got) + " after " + std::to_string(after) +
                       " allocations");
      break;
    }
  }
}

}  // namespace

int main() {
  Grammar();
  Numbers();
  Strings();
  Depths();
  Large();
  BuilderFailures();
  if (failures) {
    std::fprintf(stderr, "json_parse_test: %d failures\n", failures);
    return EXIT_FAILURE;
  }
  std::printf("json_parse_test: expected results match\n");
  return EXIT_SUCCESS;
}
