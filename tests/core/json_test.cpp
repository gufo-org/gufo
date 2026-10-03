#include "src/core/json.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <locale>
#include <string>

namespace {

using gufo::json::parse;
using gufo::json::Value;

void ExpectInvalid(std::string_view input) {
  try {
    (void)parse(input);
  } catch (const std::runtime_error&) {
    return;
  }
  std::cerr << "Accepted malformed JSON: " << input << '\n';
  std::abort();
}

void TestNumbers() {
  for (const double number :
       {0.0, -0.0, 0.12345678901234568, 9007199254740991.0, 1e-100, 1e100,
        std::numeric_limits<double>::max(), std::numeric_limits<double>::min(),
        std::numeric_limits<double>::denorm_min()}) {
    const auto serialized = Value(number).dump();
    assert(parse(serialized).as_double() == number);
  }
  for (const double number : {-1.0, 1.5, 1e100, std::ldexp(1.0, 64),
                              std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN()}) {
    assert(Value(number).as_size(777) == 777);
  }
  assert(Value(0).as_size(777) == 0);
  assert(Value(4096).as_size(777) == 4096);
  const double largest = std::nextafter(std::ldexp(1.0, 64), 0.0);
  assert(Value(largest).as_size(777) == static_cast<std::size_t>(largest));

  for (const double number : {std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN()}) {
    bool rejected = false;
    try {
      (void)Value(number).dump();
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
  struct CommaDecimal final : std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
  };
  const auto original = std::locale::global(
      std::locale(std::locale::classic(), new CommaDecimal));
  const auto serialized = Value(1.25).dump();
  std::locale::global(original);
  assert(serialized == "1.25");
}

void TestStructures() {
  for (char c = 0; c < 0x20; ++c)
    ExpectInvalid(std::string{'"', c, '"'});
  for (const auto bytes :
       {"\x80", "\xC0\xAF", "\xE0\x80\xAF", "\xED\xA0\x80", "\xF4\x90\x80\x80",
        "\xF5\x80\x80\x80", "\xC2", "\xE2\x82", "\xC2x"}) {
    ExpectInvalid(std::string("\"") + bytes + '"');
    ExpectInvalid(std::string("{\"") + bytes + "\":0}");
  }
  const std::string utf8 = "\xC2\x80\xE0\xA0\x80\xF4\x8F\xBF\xBF";
  assert(parse('"' + utf8 + '"').str() == utf8);
  for (const auto input :
       {"", "01", "1.", "+1", "1e999", "[1,]", "{\"x\":1,}", R"({"x":1,"x":2})",
        R"({"x":1,"\u0078":2})", R"("\uD800")", R"("\uDC00")",
        R"("\uD800\u0041")", R"("\uD800\uD800")", "true false"}) {
    ExpectInvalid(input);
  }
  const auto unicode = parse(R"({"text":"\uD83D\uDE00\n\t\\\""})");
  assert(unicode.member_str("text") == "\xF0\x9F\x98\x80\n\t\\\"");
  assert(parse(unicode.dump()).member_str("text") ==
         unicode.member_str("text"));
  const auto ordered = parse(R"({"z":1,"a":[true,null,{"n":2}]})");
  assert(ordered.members()[0].first == "z");
  assert(ordered.members()[1].first == "a");
  assert(ordered.dump() == R"({"z":1,"a":[true,null,{"n":2}]})");
  // Jinja tojson, as Python json.dumps(ensure_ascii=False) and llama.cpp's
  // Jinja runtime write it: spaced separators, raw UTF-8, ordered keys.
  assert(ordered.tojson() == R"({"z": 1, "a": [true, null, {"n": 2}]})");
  assert(parse(R"({"e":[],"o":{},"t":"\u00e9\n"})").tojson() ==
         "{\"e\": [], \"o\": {}, \"t\": \"\xC3\xA9\\n\"}");

  const auto bounded = std::string(128, '[') + "null" + std::string(128, ']');
  assert(parse(bounded).is_array());
  ExpectInvalid("[" + bounded + "]");
  std::string wide = "{";
  for (int i = 0; i < 4096; ++i) {
    if (i != 0)
      wide += ',';
    wide += '"' + std::to_string(i) + "\":" + std::to_string(i);
  }
  wide += '}';
  const auto metadata = parse(wide);
  assert(metadata.size() == 4096);
  assert(metadata.member_size("4095") == 4095);
}

}  // namespace

int main() {
  TestNumbers();
  TestStructures();
  std::cout << "Shared JSON checks passed.\n";
}
