#include "src/core/json_constraint.hpp"

#include <algorithm>
#include <array>
#include <barrier>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string_view>
#include <thread>

#include "src/core/sampling.hpp"

using gufo::json::parse;
using namespace gufo::sampling;

bool Accepts(const JsonConstraint& grammar, std::string_view text) {
  auto state = grammar.Start();
  for (unsigned char byte : text) {
    state = grammar.Advance(state, byte);
    if (state.empty())
      return false;
  }
  return grammar.Complete(state);
}

void TestJsonLanguage() {
  const auto grammar = JsonConstraint::Object();
  for (const char* valid :
       {"{}", R"({"a":[null,true,false,0,-2,1.25,2e-3]})",
        R"({"x":"\"\\\/\b\f\n\r\t\u00e9\uD83D\uDE00"})",
        "{\"utf8\":\"é 中 😀\"}", " \n { \"nested\" : {\"a\":1} } \t"})
    assert(Accepts(*grammar, valid));
  for (const char* invalid :
       {"[]", "null", "{", "{}{}", R"({"a":01})", R"({"a":1.})", R"({"a":+1})",
        R"({"a":NaN})", R"({"a":[1,]})", R"({"a":"\uDE00"})",
        R"({"a":"\uD83Dx"})", R"({"a":"\uD83D\u0000"})", "{\"x\":\"\n\"}",
        "{\"x\":\"\xc0\x80\"}", "{\"x\":\"\xed\xa0\x80\"}"})
    assert(!Accepts(*grammar, invalid));
  assert(Accepts(*grammar, std::string(32, '\n') + "{}"));
  assert(!Accepts(*grammar, std::string(33, '\n') + "{}"));
  assert(Accepts(*grammar, "{\"text\":\"" + std::string(100, ' ') + "\"}"));
}

void TestSchemaLanguage() {
  const auto schema = parse(R"({
    "type":"object","properties":{
      "name":{"type":"string","enum":["red","blue"]},
      "values":{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":2},
      "optional":{"anyOf":[{"type":"boolean"},{"type":"null"}]}
    },"required":["name","values","optional"],"additionalProperties":false
  })");
  const auto grammar = JsonConstraint::Compile(schema, true);
  assert(grammar == JsonConstraint::Compile(schema, true));
  for (const char* valid :
       {R"({"name":"red","values":[1],"optional":null})",
        R"({"name":"blue","values":[-1,2],"optional":true})"})
    assert(Accepts(*grammar, valid));
  for (const char* invalid :
       {R"({"name":"green","values":[1],"optional":null})",
        R"({"name":"red","values":[],"optional":null})",
        R"({"name":"red","values":[1,2,3],"optional":null})",
        R"({"name":"red","values":[1.2],"optional":null})",
        R"({"name":"red","values":[1]})",
        R"({"name":"red","values":[1],"optional":null,"extra":1})"})
    assert(!Accepts(*grammar, invalid));
  const auto optional = JsonConstraint::Compile(parse(R"({
    "type":"object","properties":{"a":{"type":"integer"},"b":{"type":"boolean"}},
    "additionalProperties":false})"),
                                                false);
  for (const char* text :
       {"{}", "{\"a\":1}", "{\"b\":true}", "{\"a\":1,\"b\":true}"})
    assert(Accepts(*optional, text));
  assert(!Accepts(*optional, "{\"b\":true,}"));
  const auto unbounded = JsonConstraint::Compile(parse(R"({
    "type":"object","properties":{"x":{"type":"array","items":{"type":"null"},"minItems":2}},
    "required":["x"],"additionalProperties":false})"),
                                                 true);
  assert(!Accepts(*unbounded, "{\"x\":[null]}"));
  std::string many = "{\"x\":[null";
  for (int i = 1; i < 300; ++i)
    many += ",null";
  many += "]}";
  assert(Accepts(*unbounded, many));
  const auto referenced = JsonConstraint::Compile(parse(R"({
    "type":"object","properties":{"x":{"$ref":"#/$defs/Label"}},
    "required":["x"],"additionalProperties":false,
    "$defs":{"Label":{"type":["string","null"],"enum":["a",null]}}
  })"),
                                                  true);
  assert(Accepts(*referenced, "{\"x\":null}"));
  assert(Accepts(*referenced, "{\"x\":\"a\"}"));
  assert(!Accepts(*referenced, "{\"x\":\"b\"}"));
}

void TestRejectedSchemas() {
  for (
      const char* input :
      {R"({"type":"array","items":{"type":"integer"}})",
       R"({"type":"object","properties":{},"additionalProperties":true})",
       R"({"type":"object","properties":{},"additionalProperties":false,"oneOf":[]})",
       R"({"type":"object","properties":{"x":{"type":"string","pattern":"["}},"additionalProperties":false,"required":["x"]})",
       R"({"type":"object","properties":{"x":{"type":"integer"}},"additionalProperties":false})",
       R"({"type":"object","properties":{"x":{"type":"string","anyOf":[true,{"type":"string"}]}},"required":["x"],"additionalProperties":false})",
       R"({"type":"object","properties":{"x":{"type":"string","anyOf":[{"anyOf":"invalid"},{"type":"string"}]}},"required":["x"],"additionalProperties":false})",
       R"({"type":"object","properties":{"x":{"type":"array","items":{"type":"null"},"minItems":2,"maxItems":1}},"required":["x"],"additionalProperties":false})",
       R"({"type":"object","properties":{"x":{"$ref":"https://example.invalid/schema"}},"required":["x"],"additionalProperties":false})",
       R"({"type":"object","properties":{},"additionalProperties":false,"$defs":{"x":{"$ref":"#/$defs/x"}}})",
       R"({"type":"object","properties":{},"additionalProperties":false,"$defs":{"x":{"type":"string","bad":1}}})"}) {
    bool rejected = false;
    try {
      (void)JsonConstraint::Compile(parse(input), true);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
}

void TestSchemaIntersections() {
  const auto grammar = JsonConstraint::Compile(parse(R"({
    "type":"object","properties":{
      "tag":{"$ref":"#/$defs/tag","pattern":"b","minLength":2,"maxLength":3},
      "n":{"$ref":"#/$defs/n","maximum":7},
      "record":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"boolean"}},
        "required":["a","b"],"additionalProperties":false,
        "enum":[{"b":true,"a":1},{"a":2,"b":false}]},
      "list":{"type":"array","items":{"type":"integer","minimum":0},
        "enum":[[1,2],[-1],[2]],"minItems":2}
    },"required":["tag","n","record","list"],"additionalProperties":false,
    "$defs":{"tag":{"type":"string","pattern":"^a","maxLength":8},
             "n":{"type":"integer","minimum":2,"maximum":10}}
  })"),
                                               true);
  assert(
      Accepts(*grammar,
              R"({"tag":"ab","n":7,"record":{"a":1,"b":true},"list":[1, 2]})"));
  for (const char* text :
       {R"({"tag":"ac","n":7,"record":{"a":1,"b":true},"list":[1,2]})",
        R"({"tag":"ab","n":8,"record":{"a":1,"b":true},"list":[1,2]})",
        R"({"tag":"ab","n":7,"record":{"a":1,"b":false},"list":[1,2]})",
        R"({"tag":"ab","n":7,"record":{"a":1,"b":true},"list":[2]})"})
    assert(!Accepts(*grammar, text));
  const auto boundaries = JsonConstraint::Compile(parse(R"({
    "type":"object","properties":{"x":{"type":"string","pattern":"\\bcat\\b"},
    "y":{"type":"string","pattern":"^\\u{1f600}\\cJ\\0$"}},
    "required":["x","y"],"additionalProperties":false})"),
                                                  true);
  assert(Accepts(*boundaries, R"({"x":"cat!","y":"😀\n\u0000"})"));
  assert(!Accepts(*boundaries, R"({"x":"cats","y":"😀\n\u0000"})"));
  const auto any = JsonConstraint::Compile(parse(R"({
    "type":"object","properties":{"x":{"anyOf":[
      {"type":"integer","minimum":1},{"type":"integer","minimum":4}],
      "maximum":5}},"required":["x"],"additionalProperties":false})"),
                                           true);
  assert(Accepts(*any, "{\"x\":5}"));
  assert(!Accepts(*any, "{\"x\":6}"));
  const auto partial = JsonConstraint::Compile(parse(R"({
    "type":"object","properties":{"x":{"anyOf":[
      {"type":"integer","minimum":6},{"type":"integer","minimum":1}],
      "maximum":5}},"required":["x"],"additionalProperties":false})"),
                                               true);
  assert(Accepts(*partial, "{\"x\":2}"));
  assert(!Accepts(*partial, "{\"x\":6}"));
  const auto dead = JsonConstraint::Compile(parse(R"({
    "type":"object","properties":{"x":{"anyOf":[
      {"type":"string","pattern":"^a$","minLength":2},
      {"type":"string","const":"b"}]}},
    "required":["x"],"additionalProperties":false})"),
                                            true);
  auto state = dead->Start();
  for (unsigned char byte : std::string_view("{\"x\":\"a"))
    state = dead->Advance(state, byte);
  assert(state.empty());
  assert(Accepts(*dead, "{\"x\":\"b\"}"));
  const auto multiples = JsonConstraint::Compile(parse(R"({
    "type":"object","properties":{"x":{"$ref":"#/$defs/N","multipleOf":0.3}},
    "required":["x"],"additionalProperties":false,
    "$defs":{"N":{"type":"number","multipleOf":0.2}}})"),
                                                 true);
  assert(Accepts(*multiples, R"({"x":0.6})"));
  assert(Accepts(*multiples, R"({"x":-1.20})"));
  assert(!Accepts(*multiples, R"({"x":0.3})"));
  assert(!Accepts(*multiples, R"({"x":0.4})"));
  const auto formats = JsonConstraint::Compile(parse(R"({
    "type":"object","properties":{"x":{"$ref":"#/$defs/S","format":"hostname",
      "pattern":"^127"}},
    "required":["x"],"additionalProperties":false,
    "$defs":{"S":{"type":"string","format":"ipv4"}}})"),
                                               true);
  assert(Accepts(*formats, R"({"x":"127.0.0.1"})"));
  assert(!Accepts(*formats, R"({"x":"192.168.1.1"})"));
  assert(!Accepts(*formats, R"({"x":"127.example"})"));
}

void TestPrimitiveConstraints() {
  auto compile = [](std::string_view property) {
    return JsonConstraint::Compile(
        parse(R"({"type":"object","properties":{"x":)" + std::string(property) +
              R"(},"required":["x"],"additionalProperties":false})"),
        true);
  };
  const auto pattern = compile(
      R"({"type":"string","pattern":"^@[a-zA-Z0-9_]+$","maxLength":6})");
  const auto positive = compile(R"({"type":"number","exclusiveMinimum":0})");
  auto number_state = positive->Start();
  for (unsigned char byte : std::string("{\"x\":0.") + std::string(4093, '0'))
    number_state = positive->Advance(number_state, byte);
  assert(!number_state.empty());
  assert(positive->Advance(number_state, '0').empty());
  const auto last_digit = positive->Advance(number_state, '1');
  assert(positive->Complete(positive->Advance(last_digit, '}')));
  for (const char* value : {R"("@abc")", R"("\u0040abc")", R"("@x_42")"})
    assert(Accepts(*pattern, std::string("{\"x\":") + value + "}"));
  for (const char* value :
       {R"("abc")", R"("@")", R"("@abcde!")", R"("@abcdef")", R"("@a-b")"})
    assert(!Accepts(*pattern, std::string("{\"x\":") + value + "}"));
  const auto unicode = compile(
      R"({"type":"string","pattern":"^é😀$","minLength":2,"maxLength":2})");
  assert(Accepts(*unicode, R"({"x":"é😀"})"));
  assert(Accepts(*unicode, R"({"x":"e\u0301😀"})") == false);
  assert(Accepts(*unicode, R"({"x":"\u00e9\uD83D\uDE00"})"));
  auto prefix_allowed = [](const JsonConstraint& grammar,
                           std::string_view text) {
    auto state = grammar.Start();
    for (unsigned char byte : text)
      state = grammar.Advance(state, byte);
    return !state.empty();
  };
  for (const auto prefix : {R"({"x":"\u)", R"({"x":"\u00e)",
                            R"({"x":"é\uD83D\uDE0)", "{\"x\":\"é\xf0\x9f"})
    assert(prefix_allowed(*unicode, prefix));
  for (const auto prefix : {R"({"x":"\uD800)", R"({"x":"é\uD800)",
                            R"({"x":"é\uD83D\uDF)", "{\"x\":\"é\xf0\x90"})
    assert(!prefix_allowed(*unicode, prefix));
  assert(!prefix_allowed(*pattern, R"({"x":"@\uD800)"));
  const auto unanchored = compile(R"({"type":"string","pattern":"cat"})");
  assert(Accepts(*unanchored, R"({"x":"\uD83D\uDE00 cat"})"));
  const auto alternatives = compile(R"({"type":"string","pattern":"^a|😀$"})");
  assert(Accepts(*alternatives, R"({"x":"\uD83D\uDE00"})"));
  const auto word = compile(R"({"type":"string","pattern":"^[\\w]+$"})");
  assert(Accepts(*word, R"({"x":"a_Z09"})"));
  assert(!Accepts(*word, R"({"x":"é"})"));
  assert(!Accepts(*word, R"({"x":"\u00e9"})"));
  const auto digit = compile(R"({"type":"string","pattern":"^\\d+$"})");
  assert(Accepts(*digit, R"({"x":"123"})"));
  assert(!Accepts(*digit, R"({"x":"\u0661"})"));
  const auto whitespace = compile(R"({"type":"string","pattern":"^\\s+$"})");
  assert(Accepts(*whitespace, R"({"x":"\ufeff\u2028\t"})"));
  assert(!Accepts(*whitespace, R"({"x":"\u0085"})"));
  const auto dot = compile(R"({"type":"string","pattern":"^.$"})");
  assert(Accepts(*dot, R"({"x":"\u000b"})"));
  assert(Accepts(*dot, R"({"x":"\u0085"})"));
  assert(!Accepts(*dot, R"({"x":"a\n"})"));
  assert(!Accepts(*dot, R"({"x":"\u2028"})"));
  const auto any = compile(R"({"type":"string","pattern":"^[^]$"})");
  assert(Accepts(*any, R"({"x":"\n"})"));
  const auto empty = compile(R"({"type":"string","pattern":"^[]*$"})");
  assert(Accepts(*empty, R"({"x":""})"));
  assert(!Accepts(*empty, R"({"x":"a"})"));
  const auto spaces = compile(R"({"type":"string","pattern":"^[a b]+$"})");
  assert(Accepts(*spaces, R"({"x":"a b"})"));
  assert(Accepts(*spaces, R"({"x":"a\u0020b"})"));
  const auto lengths =
      compile(R"({"type":"string","minLength":2,"maxLength":2})");
  for (auto text : {R"({"x":"ab"})", R"({"x":"é😀"})",
                    R"({"x":"\u00e9\uD83D\uDE00"})", R"({"x":"\\\""})"})
    assert(Accepts(*lengths, text));
  for (auto text : {R"({"x":""})", R"({"x":"a"})", R"({"x":"abc"})",
                    R"({"x":"\uD83D\uDE00"})", R"({"x":"e\u0301😀"})"})
    assert(!Accepts(*lengths, text));
  for (auto text : {R"({"x":"\uDC00a"})", R"({"x":"\uD800\u0000a"})",
                    "{\"x\":\"\xe0\x80\x80"
                    "a\"}",
                    "{\"x\":\"\xed\xa0\x80"
                    "a\"}",
                    "{\"x\":\"\xf4\x90\x80\x80"
                    "a\"}"})
    assert(!Accepts(*lengths, text));
  assert(Accepts(*lengths, R"({"x":"\uD800\uDFFFa"})"));
  const auto bounded_alternatives = compile(
      R"({"type":"string","pattern":"^(?:(?:ab){2}|c)$","maxLength":3})");
  assert(Accepts(*bounded_alternatives, R"({"x":"c"})"));
  assert(!prefix_allowed(*bounded_alternatives, R"({"x":"a)"));
  assert(!prefix_allowed(*bounded_alternatives, R"({"x":"\u0061)"));
  const auto lookahead = compile(
      R"({"type":"string","pattern":"^(?=a|c)(?:(?:ab){2}|c)$","maxLength":3})");
  assert(Accepts(*lookahead, R"({"x":"c"})"));
  assert(!prefix_allowed(*lookahead, R"({"x":"a)"));
  assert(!prefix_allowed(*lookahead, R"({"x":"\u0061)"));
  const auto intersection = compile(
      R"({"type":"string","pattern":"^(?:999|127)\\.0\\.0\\.1$","format":"ipv4"})");
  assert(Accepts(*intersection, R"({"x":"127.0.0.1"})"));
  assert(!prefix_allowed(*intersection, R"({"x":"9)"));
  assert(!prefix_allowed(*intersection, R"({"x":"\u0039)"));
  const auto negative =
      compile(R"({"type":"string","pattern":"^(?!.*bb)[ab]{1,4}$"})");
  assert(Accepts(*negative, R"({"x":"aba"})"));
  assert(!prefix_allowed(*negative, R"({"x":"abb)"));
  const auto letters =
      compile(R"({"type":"string","pattern":"^\\p{L}+$","maxLength":3})");
  assert(Accepts(*letters, R"({"x":"é中a"})"));
  assert(Accepts(*letters, R"({"x":"\u00e9\u4e2da"})"));
  assert(!Accepts(*letters, R"({"x":"123"})"));
  const auto large_repeat = compile(
      R"({"type":"string","pattern":"^(?:a{500}|b{1,500})$","maxLength":3})");
  assert(Accepts(*large_repeat, R"({"x":"bbb"})"));
  assert(!prefix_allowed(*large_repeat, R"({"x":"a)"));
  const auto bounded_repeat = compile(
      R"({"type":"string","pattern":"^(?:ab)+$","minLength":3,"maxLength":4})");
  assert(Accepts(*bounded_repeat, R"({"x":"abab"})"));
  assert(!prefix_allowed(*bounded_repeat, R"({"x":"abab\u0061)"));
  const auto bounded_class = compile(
      R"({"type":"string","pattern":"^(?:[^a-c]{4}|x)$","maxLength":3})");
  assert(Accepts(*bounded_class, R"({"x":"x"})"));
  assert(!prefix_allowed(*bounded_class, R"({"x":"y)"));
  const auto repeated_unicode = compile(
      R"({"type":"string","pattern":"^(?:é😀)+$","minLength":3,"maxLength":4})");
  assert(Accepts(*repeated_unicode, R"({"x":"\u00e9\uD83D\uDE00é😀"})"));
  const auto number = compile(
      R"({"type":"number","minimum":-0.4,"exclusiveMaximum":0.5,"multipleOf":0.1})");
  for (const auto value : {"-0.4", "-0.3", "0", "0.3", "0.40"})
    assert(Accepts(*number, std::string("{\"x\":") + value + "}"));
  for (const auto value : {"-0.5", "0.31", "0.5", "0.40000000000000001"})
    assert(!Accepts(*number, std::string("{\"x\":") + value + "}"));
  const auto integer = compile(
      R"({"type":"integer","multipleOf":1.5,"minimum":-12,"maximum":12})");
  for (int value = -14; value <= 14; ++value)
    assert(Accepts(*integer, "{\"x\":" + std::to_string(value) + "}") ==
           (value >= -12 && value <= 12 && value % 3 == 0));
  const std::array formats{
      std::array{"date", "2024-02-29", "2023-02-29"},
      std::array{"date-time", "2026-09-26T23:30:00+02:00",
                 "2026-09-31T23:30:00Z"},
      std::array{"time", "23:59:01.25Z", "24:00:00Z"},
      std::array{"uuid", "12345678-1234-1234-1234-123456789abc", "1234"},
      std::array{"ipv4", "192.168.1.89", "256.168.1.89"},
      std::array{"ipv6", "2001:db8::1", "1:::2"},
      std::array{"hostname", "example.com", "-example.com"},
      std::array{"email", "name@example.com", "name@bad domain"},
      std::array{"duration", "P1DT2H30M", "PT"}};
  for (const auto& [format, valid, invalid] : formats) {
    const auto grammar =
        compile(std::string(R"({"type":"string","format":")") + format + "\"}");
    assert(Accepts(*grammar, std::string("{\"x\":\"") + valid + "\"}"));
    assert(!Accepts(*grammar, std::string("{\"x\":\"") + invalid + "\"}"));
  }
}

void TestIntegerBoundsAndRecursion() {
  auto scalar = [](std::string_view constraints) {
    return JsonConstraint::Compile(
        parse(R"({"type":"object","properties":{"x":{"type":"integer",)" +
              std::string(constraints) +
              R"(}},"required":["x"],"additionalProperties":false})"),
        true);
  };
  for (int low = -12; low <= 12; ++low) {
    for (int high = low; high <= 12; ++high) {
      const auto grammar = scalar("\"minimum\":" + std::to_string(low) +
                                  ",\"maximum\":" + std::to_string(high));
      for (int number = -15; number <= 15; ++number)
        assert(Accepts(*grammar, "{\"x\":" + std::to_string(number) + "}") ==
               (low <= number && number <= high));
    }
  }
  const auto exclusive =
      scalar(R"("exclusiveMinimum":-2,"exclusiveMaximum":3)");
  for (int number = -4; number <= 4; ++number)
    assert(Accepts(*exclusive, "{\"x\":" + std::to_string(number) + "}") ==
           (-2 < number && number < 3));
  const auto fractional = scalar(R"("minimum":-1.2,"maximum":2.8)");
  assert(Accepts(*fractional, "{\"x\":-1}"));
  assert(Accepts(*fractional, "{\"x\":2}"));
  assert(!Accepts(*fractional, "{\"x\":-2}"));
  assert(!Accepts(*fractional, "{\"x\":3}"));
  const auto large = scalar(R"("exclusiveMinimum":9007199254740992)");
  assert(!Accepts(*large, "{\"x\":9007199254740992}"));
  assert(Accepts(*large, "{\"x\":9007199254740993}"));
  assert(Accepts(*large, "{\"x\":1000000000000000000000000000000000}"));
  const auto negative = scalar(R"("exclusiveMaximum":-9007199254740992)");
  assert(!Accepts(*negative, "{\"x\":-9007199254740992}"));
  assert(Accepts(*negative, "{\"x\":-9007199254740993}"));
  const auto tree = JsonConstraint::Compile(parse(R"({
    "type":"object","properties":{
      "value":{"type":"integer","minimum":1,"maximum":5},
      "children":{"type":"array","items":{"$ref":"#"}}
    },"required":["value","children"],"additionalProperties":false
  })"),
                                            true);
  assert(
      Accepts(*tree, R"({"value":1,"children":[{"value":5,"children":[]}]})"));
  assert(
      !Accepts(*tree, R"({"value":1,"children":[{"value":6,"children":[]}]})"));
  const auto list = JsonConstraint::Compile(parse(R"({
    "type":"object","properties":{"head":{"$ref":"#/$defs/node"}},
    "required":["head"],"additionalProperties":false,"$defs":{
      "node":{"anyOf":[{"type":"null"},{"type":"object",
        "properties":{"next":{"$ref":"#/$defs/node"}},
        "required":["next"],"additionalProperties":false}]}
    }
  })"),
                                            true);
  assert(Accepts(*list, R"({"head":{"next":{"next":null}}})"));
}

void TestTokensAndSampling() {
  bool invalid_config = false;
  try {
    SamplerState invalid({.constraint = std::make_shared<TokenConstraint>()});
  } catch (const std::invalid_argument&) {
    invalid_config = true;
  }
  assert(invalid_config);
  const std::vector<std::string> pieces{
      "{\"x\":",   "true",     "false", "}",   "INVALID",   "",
      "{\"x\":\"", "\xe2\x94", "\x8c",  "\"}", "false}BAD", "false}",
      "<think>",   "{\"x\":[", ",",     "]}"};
  auto vocabulary = std::make_shared<ConstraintVocabulary>(
      pieces.size(), [&](std::uint32_t i) {
        return ConstraintVocabulary::Piece{pieces[i], i == 5};
      });
  auto constraint = std::make_shared<TokenConstraint>();
  constraint->grammar = JsonConstraint::Object();
  constraint->vocabulary = vocabulary;
  auto state = constraint->grammar->Start();
  auto mask = constraint->Allowed(state);
  assert((*mask)[0] && (*mask)[6] && !(*mask)[4] && !(*mask)[5] &&
         !(*mask)[12]);
  state = vocabulary->Accept(*constraint->grammar, state, 6);
  assert((*constraint->Allowed(state))[12]);
  assert((*constraint->Allowed(state))[7]);
  state = vocabulary->Accept(*constraint->grammar, state, 7);
  assert((*constraint->Allowed(state))[8]);
  assert(!(*constraint->Allowed(state))[9]);
  state = vocabulary->Accept(*constraint->grammar, state, 8);
  state = vocabulary->Accept(*constraint->grammar, state, 9);
  mask = constraint->Allowed(state);
  for (std::size_t i = 0; i < mask->size(); ++i)
    assert((*mask)[i] == (i == 5));

  SamplerState sampler(
      {.temperature = .5F, .top_p = .8F, .seed = 42, .constraint = constraint});
  sampler.Accept(0);
  std::vector<float> logits(pieces.size(),
                            -std::numeric_limits<float>::infinity());
  logits[1] = 0;
  logits[2] = -1;
  logits[4] = 100;   // Invalid high-logit token must not affect top-p support.
  logits[10] = 101;  // Valid prefix but invalid suffix in the same token.
  const auto distribution = sampler.Distribution(logits);
  assert(distribution.entries().size() == 1);
  assert(distribution.probability(1) == 1);
  assert(distribution.probability(4) == 0);
  const auto before = sampler;
  auto tentative = sampler;
  tentative.Accept(1);
  tentative.Accept(3);
  logits[5] = 0;
  assert(tentative.Sample(logits) == 5);
  assert(before.Distribution(logits).probability(1) == 1);
  const std::array<TokenId, 1> proposal{4};
  const std::array<float, 1> probabilities{1};
  assert(sampler.SampleResidual(logits, proposal, probabilities) == 1);
  assert(!sampler.config().can_use_unmodified_argmax());
  SamplerState greedy({.constraint = constraint});
  greedy.Accept(0);
  assert(greedy.Sample(logits) == 1);
  assert(!greedy.config().can_use_unmodified_argmax());
  assert(greedy.WithoutConstraint().config().can_use_unmodified_argmax());
  // Independent pre-masked control for the allocation-free constrained argmax.
  // Include negative logits, ties, excluded nonfinite values and every penalty.
  for (const auto penalties : {false, true}) {
    SamplingConfig config{.repeat_penalty = penalties ? 1.2F : 1.0F,
                          .frequency_penalty = penalties ? .3F : 0.0F,
                          .presence_penalty = penalties ? .5F : 0.0F,
                          .constraint = constraint};
    SamplerState constrained(config, std::array<TokenId, 2>{1, 1});
    const std::array<TokenId, 3> generated{13, 1, 14};
    constrained.Accept(generated);
    config.constraint.reset();
    SamplerState control(config, std::array<TokenId, 2>{1, 1});
    control.Accept(generated);
    auto generated_state = constraint->grammar->Start();
    for (auto token : generated)
      generated_state =
          vocabulary->Accept(*constraint->grammar, generated_state, token);
    const auto allowed = constraint->Allowed(generated_state);
    for (const auto score : {-3.0F, 0.0F, 1.0F}) {
      std::fill(logits.begin(), logits.end(), INFINITY);
      logits[1] = logits[2] = score;
      logits[4] = NAN;
      auto masked = logits;
      for (std::size_t i = 0; i < masked.size(); ++i)
        if (!(*allowed)[i])
          masked[i] = -INFINITY;
      const auto expected = control.Sample(masked);
      assert(constrained.Sample(logits) == expected);
      assert(constrained.Distribution(logits).probability(expected) == 1);
    }
  }
  std::fill(logits.begin(), logits.end(),
            -std::numeric_limits<float>::infinity());
  bool rejected = false;
  try {
    (void)sampler.Sample(logits);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
  rejected = false;
  try {
    (void)greedy.Sample(logits);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
}

void TestStringMaskCache() {
  const std::vector<std::string> pieces{
      "{\"x\":\"",      "a",    "abcdef", "\"}",  "\\u0061",
      "\\uD83D\\uDE00", "\xc2", "\x80",   "a\"}", ""};
  auto vocabulary = std::make_shared<ConstraintVocabulary>(
      pieces.size(), [&](std::uint32_t i) {
        return ConstraintVocabulary::Piece{pieces[i], i == pieces.size() - 1};
      });
  for (bool pattern : {false, true}) {
    auto schema = parse(R"({
      "type":"object","properties":{"x":{
        "type":"string","minLength":1,"maxLength":48}},
      "required":["x"],"additionalProperties":false})");
    if (pattern)
      schema["properties"]["x"]["pattern"] = "^[a-z]+$";
    TokenConstraint constraint;
    constraint.grammar = JsonConstraint::Compile(schema, true);
    constraint.vocabulary = vocabulary;
    auto at = [&](std::size_t count) {
      auto state = vocabulary->Accept(*constraint.grammar,
                                      constraint.grammar->Start(), 0);
      for (std::size_t i = 0; i < count; ++i)
        state = vocabulary->Accept(*constraint.grammar, state, 1);
      return state;
    };
    const auto one = at(1), five = at(5);
    assert(one != five);
    assert(constraint.Allowed(one) == constraint.Allowed(five));
    const auto near = at(47);
    const auto mask = constraint.Allowed(near);
    assert((*mask)[1] && !(*mask)[2] && (*mask)[3] && (*mask)[4]);
    assert((*mask)[5] == !pattern);
    assert(!(*mask)[9]);
    auto last = vocabulary->Accept(*constraint.grammar, near, 4);
    const auto full = constraint.Allowed(last);
    assert(!(*full)[1] && !(*full)[2] && (*full)[3] && !(*full)[4]);
    if (!pattern) {
      auto partial = vocabulary->Accept(*constraint.grammar, near, 6);
      assert((*constraint.Allowed(partial))[7]);
      assert(!(*constraint.Allowed(partial))[3]);
      partial = vocabulary->Accept(*constraint.grammar, partial, 7);
      assert(*constraint.Allowed(partial) == *full);
    }
  }
  {
    const auto schema = parse(R"({
      "type":"object","properties":{"x":{"type":"string","minLength":48,
      "maxLength":96}},"required":["x"],"additionalProperties":false})");
    TokenConstraint constraint;
    constraint.grammar = JsonConstraint::Compile(schema, true);
    constraint.vocabulary = vocabulary;
    auto state =
        vocabulary->Accept(*constraint.grammar, constraint.grammar->Start(), 0);
    const auto empty = constraint.Allowed(state);
    for (int i = 0; i < 16; ++i)
      state = vocabulary->Accept(*constraint.grammar, state, 1);
    assert(constraint.Allowed(state) == empty);
    assert(!(*empty)[3] && !(*empty)[8]);
    for (int i = 16; i < 47; ++i)
      state = vocabulary->Accept(*constraint.grammar, state, 1);
    const auto near = constraint.Allowed(state);
    assert(near != empty && !(*near)[3] && (*near)[8]);
    state = vocabulary->Accept(*constraint.grammar, state, 1);
    assert((*constraint.Allowed(state))[3]);
  }
  // Character limits count Unicode scalars, not escaped bytes, and do not
  // impose the old 64 KiB serialized-prefix ceiling.
  auto schema = parse(R"({
    "type":"object","properties":{"x":{"type":"string","minLength":70000,
    "maxLength":70000}},"required":["x"],"additionalProperties":false})");
  auto grammar = JsonConstraint::Compile(schema, true);
  const auto content = std::string(69999, 'a') + "\\uD83D\\uDE00";
  assert(Accepts(*grammar, "{\"x\":\"" + content + "\"}"));
  assert(!Accepts(*grammar, "{\"x\":\"" + content + "a\"}"));
  // Large minLength must not require one allocated DFA frontier per character.
  schema["properties"]["x"]["pattern"] = "^(?:ab)+$";
  schema["properties"]["x"]["minLength"] = 1048575;
  schema["properties"]["x"]["maxLength"] = 1048576;
  grammar = JsonConstraint::Compile(schema, true);
  assert(!grammar->Start().empty());
}

void TestUnsupportedPatterns() {
  for (const auto pattern :
       {R"((a)\1)", "(?<=a)b", "(?i)a", "(?:(?=a)a)*", R"(\_)", R"(\01)"}) {
    auto schema = parse(R"({
      "type":"object","properties":{"x":{"type":"string"}},
      "required":["x"],"additionalProperties":false})");
    schema["properties"]["x"]["pattern"] = pattern;
    bool rejected = false;
    try {
      (void)JsonConstraint::Compile(schema, true);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
}

void TestNativeRequestMaskReuse() {
  const auto arguments =
      JsonConstraint::ToolParameters(parse(R"({
    "type":"object","properties":{"value":{"type":"string"}},
    "required":["value"],"additionalProperties":false})"),
                                     false, JsonConstraint::ToolFormat::kQwen);
  TokenConstraint binding;
  binding.grammar =
      JsonConstraint::WithTools(nullptr, {{"record", arguments}}, false, true,
                                JsonConstraint::ToolFormat::kQwen);
  binding.vocabulary =
      std::make_shared<ConstraintVocabulary>(129, [](std::uint32_t id) {
        return ConstraintVocabulary::Piece{
            id == 128 ? "" : std::string(1, static_cast<char>(id)), id == 128};
      });
  const auto initial = binding.grammar->Start();
  const auto first = binding.Allowed(initial);
  auto state = initial;
  for (const unsigned char byte : std::string_view(
           "<tool_call>\n<function=record>\n<parameter=value>\nalpha\n"
           "</parameter>\n</function>\n</tool_call>")) {
    state = binding.grammar->Advance(state, byte);
    assert(!state.empty());
    (void)binding.Allowed(state);
  }
  assert(binding.grammar->Complete(state));
  // Every new request starts here. Argument masks must not evict this mask
  // and force another full-vocabulary traversal on a warm cache hit.
  assert(binding.Allowed(initial) == first);
}

void TestReasoningConstraint() {
  const auto plain = JsonConstraint::Object();
  const auto grammar = JsonConstraint::WithReasoning(plain);
  assert(grammar == JsonConstraint::WithReasoning(plain));
  assert(Accepts(*grammar,
                 "Think freely <tool_call> and <think>.</think>{\"x\":1}"));
  assert(Accepts(*grammar, "</think>{\"x\":\"</think>\"}"));
  assert(!Accepts(*grammar, "thinking only"));
  assert(!Accepts(*grammar, "</think>not JSON"));
  const std::vector<std::string> pieces{
      "thinking", "</thi", "nk>{\"x\":", "true}", "nk>INVALID", ""};
  auto binding = std::make_shared<TokenConstraint>();
  binding->grammar = grammar;
  binding->vocabulary = std::make_shared<ConstraintVocabulary>(
      pieces.size(), [&](std::uint32_t i) {
        return ConstraintVocabulary::Piece{pieces[i], i == 5};
      });
  SamplerState sampler({.constraint = binding});
  sampler.Accept(0);
  sampler.Accept(1);
  auto tentative = sampler;
  tentative.Accept(2);
  tentative.Accept(3);
  const auto mask = binding->Allowed(grammar->Start());
  assert(!(*mask)[5]);
  std::vector<float> logits(pieces.size(), -INFINITY);
  logits[4] = 100;
  logits[2] = 0;
  assert(sampler.Sample(logits) == 2);
  logits[5] = 101;
  assert(tentative.Sample(logits) == 5);
  const auto tool = JsonConstraint::Compile(parse(R"({
    "type":"object","properties":{"value":{"type":"integer","minimum":1,"maximum":5}},
    "required":["value"],"additionalProperties":false})"),
                                            true);
  const auto mixed = JsonConstraint::WithTools(
      plain, {{"score", tool}, {"rating", tool}, {"metadata", plain}}, false);
  assert(Accepts(*mixed, "{\"final\":true}"));
  assert(Accepts(
      *mixed,
      R"(<tool_call>{"name":"score","arguments":{"value":3}}</tool_call>)"));
  assert(!Accepts(
      *mixed,
      R"(<tool_call>{"name":"score","arguments":{"value":6}}</tool_call>)"));
  assert(!Accepts(
      *mixed,
      R"(<tool_call>{"name":"unknown","arguments":{"value":3}}</tool_call>)"));
  assert(Accepts(
      *mixed,
      R"(<tool_call>{"name":"rating","arguments":{"value":4}}</tool_call>)"));
  assert(!Accepts(
      *mixed,
      R"(<tool_call>{"name":"rating","arguments":{"value":0}}</tool_call>)"));
  assert(Accepts(
      *mixed,
      R"(<tool_call>{"name":"metadata","arguments":{"extra":true}}</tool_call>)"));
  const auto required =
      JsonConstraint::WithTools(plain, {{"score", tool}}, true);
  assert(!Accepts(*required, "{\"final\":true}"));
  assert(Accepts(
      *JsonConstraint::WithReasoning(required),
      R"(Use the scoring tool.</think><tool_call>{"name":"score","arguments":{"value":3}}</tool_call>)"));
  const auto automatic =
      JsonConstraint::WithTools(nullptr, {{"score", tool}}, false, true);
  const std::string call =
      R"(<tool_call>{"name":"score","arguments":{"value":3}}</tool_call>)";
  assert(Accepts(*automatic, "Ordinary prose."));
  assert(Accepts(*automatic, "First: " + call + "\n" + call));
  assert(!Accepts(
      *automatic,
      R"(First: <tool_call>{"name":"score","arguments":{"value":6}}</tool_call>)"));
  assert(!Accepts(*automatic,
                  R"(<tool_call>{"name":"other","arguments":{}}</tool_call>)"));
  const auto single =
      JsonConstraint::WithTools(nullptr, {{"score", tool}}, true, false);
  assert(Accepts(*single, call));
  assert(!Accepts(*single, "Ordinary prose."));
  assert(!Accepts(*single, call + call));
  for (const auto format : {JsonConstraint::ToolFormat::kQwen,
                            JsonConstraint::ToolFormat::kDeepSeek}) {
    const bool qwen = format == JsonConstraint::ToolFormat::kQwen;
    const auto arguments = JsonConstraint::ToolParameters(parse(R"({
      "type":"object","properties":{"value":{"type":"string","const":"42"}},
      "required":["value"],"additionalProperties":false})"),
                                                          true, format);
    const std::string native =
        qwen ? "<tool_call>\n<function=record>\n<parameter=value>\n42\n"
               "</parameter>\n</function>\n</tool_call>"
             : "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"record\">\n"
               "<｜DSML｜parameter name=\"value\" string=\"true\">42"
               "</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    for (const bool required : {false, true}) {
      const auto native_grammar =
          JsonConstraint::WithReasoning(JsonConstraint::WithTools(
              nullptr, {{"record", arguments}}, required, false, format));
      assert(Accepts(*native_grammar, "Read the file.\n\n" + native));
      assert(Accepts(*native_grammar,
                     "A literal <tool_call> is data.</think>" + native));
      assert(Accepts(*native_grammar, "Example:\n```xml\n" + native +
                                          "\n```\nNow call it.</think>" +
                                          native));
      assert(Accepts(*native_grammar,
                     "Example: ``" + native + "``.</think>" + native));
      auto invalid = native;
      invalid.replace(invalid.find("42"), 2, "43");
      assert(!Accepts(*native_grammar, "Read the file.\n\n" + invalid));
      // Compare the fast reasoning mask against scalar grammar evaluation,
      // including partial/overlapping markers, quoted headers, and tokens that
      // cross from reasoning into invalid tool output.
      const std::vector<ConstraintVocabulary::Piece> pieces{
          {"word"},
          {"\n"},
          {"`"},
          {"~~~\n"},
          {"<"},
          {"</think>"},
          {"think>"},
          {"tool_call>\n<function=record>"},
          {native},
          {invalid},
          {"</think>" + native},
          {"</think>" + invalid},
          {"``" + native + "``"},
          {"<tool_call>\n<tool_call>\n<function="},
          {""},
          {"", true}};
      const ConstraintVocabulary vocabulary(
          pieces.size(), [&](auto id) { return pieces[id]; });
      for (const auto& prefix : std::vector<std::string>{
               "", "Read the file.\n\n", "Example: `literal ", "```xml\n", "<",
               "</", "<tool_call>\n<", "~~~\n", "``" + native, "</think>",
               "Read the file.\n\n" + native}) {
        auto checked = native_grammar->Start();
        for (const unsigned char byte : prefix)
          checked = native_grammar->Advance(checked, byte);
        assert(!checked.empty());
        const auto mask = vocabulary.Allowed(*native_grammar, checked);
        for (std::uint32_t id = 0; id < pieces.size(); ++id)
          assert(bool(mask[id]) ==
                 vocabulary.Allows(*native_grammar, checked, id));
      }
      assert(Accepts(*native_grammar, "Reasoning only.") == !required);
      auto state = native_grammar->Start();
      for (const auto byte : std::string("Read the file.\n\n"))
        state = native_grammar->Advance(state, byte);
      const auto checkpoint = state;
      for (const auto byte : native)
        state = native_grammar->Advance(state, byte);
      assert(native_grammar->Complete(state));
      state = checkpoint;
      for (const auto byte : "</think>" + native)
        state = native_grammar->Advance(state, byte);
      assert(native_grammar->Complete(state));
    }
  }
  const std::vector<std::string> tool_pieces{
      "Hello ",
      "<tool_",
      "call>{\"name\":\"score\",\"arguments\":{\"value\":",
      "3}}</tool_call>",
      "6}}</tool_call>",
      ""};
  auto tool_vocabulary = std::make_shared<ConstraintVocabulary>(
      tool_pieces.size(), [&](std::uint32_t i) {
        return ConstraintVocabulary::Piece{tool_pieces[i], i == 5};
      });
  auto state = automatic->Start();
  auto allowed = tool_vocabulary->Allowed(*automatic, state);
  assert(allowed[0] && allowed[1] && allowed[5]);
  for (const auto token : {0, 1, 2})
    state = tool_vocabulary->Accept(*automatic, state, token);
  allowed = tool_vocabulary->Allowed(*automatic, state);
  assert(allowed[3] && !allowed[4] && !allowed[5]);
  const auto abandoned = state;
  state = tool_vocabulary->Accept(*automatic, state, 3);
  assert(tool_vocabulary->Allowed(*automatic, state)[5]);
  assert(tool_vocabulary->Allowed(*automatic, abandoned) == allowed);

  std::barrier ready(8);
  std::array<std::shared_ptr<const JsonConstraint>, 8> concurrent;
  {
    std::vector<std::jthread> workers;
    for (std::size_t i = 0; i < concurrent.size(); ++i)
      workers.emplace_back([&, i] {
        ready.arrive_and_wait();
        concurrent[i] = JsonConstraint::WithReasoning(
            JsonConstraint::WithTools(plain, {{"concurrent", tool}}, true));
      });
  }
  for (const auto& grammar : concurrent) {
    assert(grammar == concurrent.front());
    assert(Accepts(
        *grammar,
        R"(Think.</think><tool_call>{"name":"concurrent","arguments":{"value":3}}</tool_call>)"));
  }
}

void TestAutomaticTools() {
  using Format = JsonConstraint::ToolFormat;
  const auto schema = parse(R"({"type":"object","properties":{
    "text":{"type":"string"},"optional":{"type":"integer"}},
    "required":["text"],"additionalProperties":false})");
  for (const auto format : {Format::kQwen, Format::kDeepSeek}) {
    const std::string opener =
        format == Format::kQwen ? "<tool_call>" : "<｜DSML｜tool_calls>";
    const std::string call =
        format == Format::kQwen
            ? "\n<function=f>\n<parameter=text>\nalpha\n</parameter>\n"
              "</function>\n</tool_call>"
            : "\n<｜DSML｜invoke name=\"f\">\n"
              "<｜DSML｜parameter name=\"text\" string=\"true\">alpha"
              "</｜DSML｜parameter>\n</｜DSML｜invoke>\n"
              "</｜DSML｜tool_calls>";
    const auto automatic = JsonConstraint::WithTools(
        nullptr, {{"f", JsonConstraint::ToolParameters(schema, false, format)}},
        false, true, format);
    for (const bool thinking : {false, true}) {
      const auto grammar =
          thinking ? JsonConstraint::WithReasoning(automatic) : automatic;
      auto binding = std::make_shared<TokenConstraint>();
      binding->grammar = grammar;
      binding->vocabulary =
          std::make_shared<ConstraintVocabulary>(259, [&](std::uint32_t i) {
            return ConstraintVocabulary::Piece{
                i < 256    ? std::string(1, static_cast<char>(i))
                : i == 258 ? opener
                           : "",
                i == 256};
          });
      for (const float temperature : {0.0F, 0.7F, 1.4F}) {
        gufo::sampling::SamplingConfig config{.temperature = temperature,
                                              .top_k = 20,
                                              .top_p = .8F,
                                              .min_p = .05F,
                                              .seed = 41,
                                              .frequency_penalty = .2F,
                                              .presence_penalty = .3F,
                                              .constraint = binding};
        SamplerState sampler(config);
        auto plain = sampler.WithoutConstraint();
        assert(!sampler.NeedsConstraintMask());
        assert(!sampler.CanSelectArgmax('a'));
        assert(sampler.CanSelectArgmax('a', /*penalties_applied=*/true) ==
               (temperature == 0.0F));
        std::vector<float> logits(259, -INFINITY);
        logits['a'] = 1;
        logits['b'] = .8F;
        for (int i = 0; i < 8; ++i) {
          const auto expected = plain.Sample(logits);
          const auto actual = sampler.Sample(logits);
          assert(actual == expected &&
                 sampler.rng_state() == plain.rng_state());
          sampler.Accept(actual);
          plain.Accept(expected);
          assert(!sampler.NeedsConstraintMask());
        }
        auto accept = [&](std::string_view text) {
          for (const unsigned char byte : text)
            sampler.Accept(byte);
        };
        if (thinking) {
          accept("</thi");
          auto saved = sampler;
          accept("nk>");
          assert(!sampler.NeedsConstraintMask());
          sampler = saved;
          accept("nk>");
        }
        auto before_call = sampler;
        sampler.Accept(258);
        assert(sampler.NeedsConstraintMask());
        assert(!sampler.CanSelectArgmax('!', /*penalties_applied=*/true));
        assert(!sampler.CanSelectArgmax(256, /*penalties_applied=*/true));
        accept(call);
        // As in llama.cpp, calls end the output: no text may follow them.
        // Parallel Qwen calls may follow each other; DeepSeek's share a block.
        assert(sampler.NeedsConstraintMask());
        assert(!sampler.CanSelectArgmax('a', /*penalties_applied=*/true));
        if (format == Format::kDeepSeek)
          assert(!sampler.CanSelectArgmax(258, /*penalties_applied=*/true));
        std::vector<float> after_call(259, -INFINITY);
        after_call['a'] = 1;
        after_call[256] = 0;
        assert(auto(sampler).Sample(after_call) == 256);
        sampler.Accept(257);  // Empty pieces preserve the ordinary path.
        sampler.Accept(256);  // Natural EOS is allowed without another call.
        sampler = before_call;
        assert(!sampler.NeedsConstraintMask());
        sampler.ResetHistory({});
        assert(!sampler.NeedsConstraintMask());
      }
    }
    // A token can span the trigger and an invalid name. Such a vocabulary
    // must never take the unrestricted sampling fast path.
    auto crossing = std::make_shared<TokenConstraint>();
    crossing->grammar = automatic;
    crossing->vocabulary =
        std::make_shared<ConstraintVocabulary>(2, [&](std::uint32_t i) {
          return ConstraintVocabulary::Piece{
              i == 0 ? "text" : opener + "\n<unknown>", false};
        });
    SamplerState sampler({.constraint = crossing});
    assert(sampler.NeedsConstraintMask());
    const std::vector<float> logits{0, 100};
    assert(sampler.Sample(logits) == 0);
  }
}

void TestNativeTools() {
  using Format = JsonConstraint::ToolFormat;
  const auto schema = parse(R"({
    "type":"object","properties":{
      "text":{"type":"string","minLength":1,"maxLength":256},
      "n":{"type":"integer","minimum":1,"maximum":5}},
    "required":["text","n"],"additionalProperties":false})");
  for (const auto format : {Format::kQwen, Format::kDeepSeek}) {
    const auto parameters =
        JsonConstraint::ToolParameters(schema, true, format);
    assert(parameters);
    const auto wrap = [&](std::string_view text, int n, bool reverse = false) {
      const bool qwen = format == Format::kQwen;
      auto argument = [&](std::string name, std::string value, bool string) {
        return qwen ? "<parameter=" + name + ">\n" + value + "\n</parameter>\n"
                    : "<｜DSML｜parameter name=\"" + name + "\" string=\"" +
                          (string ? "true" : "false") + "\">" + value +
                          "</｜DSML｜parameter>\n";
      };
      const auto a = argument("text", std::string(text), true);
      const auto b = argument("n", std::to_string(n), false);
      return (qwen ? "<tool_call>\n<function=f>\n"
                   : "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n") +
             (reverse ? b + a : a + b) +
             (qwen ? "</function>\n</tool_call>"
                   : "</｜DSML｜invoke>\n</｜DSML｜tool_calls>");
    };
    const auto grammar = JsonConstraint::WithTools(nullptr, {{"f", parameters}},
                                                   true, false, format);
    const auto call = wrap(" <think>literal</think> \"42\" \\\nπ🦉\n ", 3);
    assert(Accepts(*grammar, call));
    assert(Accepts(*grammar, wrap("</tool_call><tool_call>", 3, true)));
    assert(!Accepts(*grammar, wrap("", 3)));
    assert(!Accepts(*grammar, wrap("value", 6)));
    assert(!Accepts(*grammar, "ordinary text"));
    assert(!Accepts(*grammar, call + call));
    const auto parallel = JsonConstraint::WithTools(
        nullptr, {{"f", parameters}}, true, true, format);
    if (format == Format::kQwen) {
      // A raw value must not absorb a delimiter prefix and let the actual
      // closer complete it across the boundary. Pi's read call otherwise
      // generated two closers, which the native parser correctly rejects.
      assert(!Accepts(*grammar, wrap("archive.txt\n</parameter>", 3)));
      assert(!Accepts(*grammar, wrap("archive.txt\n</parameter>\n", 3)));
      assert(Accepts(*grammar, wrap("archive.txt\n</parameter> literal", 3)));
      assert(Accepts(*grammar, wrap("archive.txt\n", 3)));
      assert(Accepts(*parallel, call + call));
      for (const auto gap : {"", " ", "\n", "\n\n", "\n\t \t", "\n\n   "}) {
        assert(Accepts(*parallel, call + gap));
        assert(Accepts(*parallel, call + gap + call));
      }
      assert(Accepts(*parallel, call + "\n" + std::string(20, '\t')));
      for (const auto gap : {"  ", "\t", "\r\n", "\n\n\n"})
        assert(!Accepts(*parallel, call + gap));
      assert(!Accepts(*parallel, call + "\n" + std::string(21, ' ')));
    } else {
      // As in llama.cpp, parallel DeepSeek calls share one block, which ends
      // the output.
      constexpr std::string_view kOpen = "<｜DSML｜tool_calls>";
      constexpr std::string_view kClose = "\n</｜DSML｜tool_calls>";
      assert(Accepts(*parallel, call.substr(0, call.size() - kClose.size()) +
                                    call.substr(kOpen.size())));
      assert(!Accepts(*parallel, call + call));
    }
    assert(Accepts(*JsonConstraint::WithReasoning(grammar),
                   "Thinking.</think>" + call));
    // Vocabulary masks and speculative copies must agree with byte matching,
    // including at UTF-8, literal-markup and raw-string escape boundaries.
    std::vector<std::string> pieces{"\"42\"", "<think>",      "🦉", "\n",
                                    "\\",     "</parameter>", "",   "abc"};
    for (unsigned byte = 0; byte < 256; ++byte)
      pieces.emplace_back(1, static_cast<char>(byte));
    for (unsigned byte = 0; byte < 256; ++byte)
      pieces.push_back("shared-prefix" +
                       std::string(1, static_cast<char>(byte)));
    const gufo::sampling::ConstraintVocabulary vocabulary(
        pieces.size(), [&](std::uint32_t i) {
          return gufo::sampling::ConstraintVocabulary::Piece{pieces[i], i == 6};
        });
    auto binding = std::make_shared<gufo::sampling::TokenConstraint>();
    binding->grammar = grammar;
    binding->vocabulary =
        std::make_shared<gufo::sampling::ConstraintVocabulary>(vocabulary);
    auto state = grammar->Start();
    for (const unsigned char byte : call) {
      const auto mask = vocabulary.Allowed(*grammar, state);
      assert(*binding->Allowed(state) == mask);
      for (std::uint32_t token = 0; token < pieces.size(); ++token)
        assert(vocabulary.Allows(*grammar, state, token) == bool(mask[token]));
      state = grammar->Advance(state, byte);
    }
    assert(grammar->Complete(state));
  }
  const auto finite = parse(R"({"type":"object",
    "properties":{"text":{"type":"string","enum":["<think>literal</think>"]}},
    "required":["text"],"additionalProperties":false})");
  assert(JsonConstraint::ToolParameters(finite, true, Format::kQwen));
  const auto ambiguous = parse(R"({"type":"object",
    "properties":{"text":{"type":["string","null"]}},
    "required":["text"],"additionalProperties":false})");
  // As llama.cpp, a strict union admitting strings keeps native raw text
  // instead of moving every tool to a JSON envelope.
  for (const auto format : {Format::kQwen, Format::kDeepSeek}) {
    const auto parameters =
        JsonConstraint::ToolParameters(ambiguous, true, format);
    assert(parameters);
    const auto grammar = JsonConstraint::WithTools(nullptr, {{"f", parameters}},
                                                   true, false, format);
    assert(
        Accepts(*grammar, format == Format::kQwen
                              ? "<tool_call>\n<function=f>\n<parameter=text>"
                                "\nnull\n</parameter>\n</function>\n"
                                "</tool_call>"
                              : "<｜DSML｜tool_calls>\n<｜DSML｜invoke "
                                "name=\"f\">\n<｜DSML｜parameter name=\"text\" "
                                "string=\"false\">null</｜DSML｜parameter>\n"
                                "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"));
  }
  const auto delimiter_pattern = parse(R"({"type":"object",
    "properties":{"text":{"type":"string","pattern":"^\\n</parameter>$"}},
    "required":["text"],"additionalProperties":false})");
  assert(Accepts(*JsonConstraint::Compile(delimiter_pattern, true),
                 R"({"text":"\n</parameter>"})"));
  // The pattern needs the native delimiter, so it cannot be enforced; as in
  // llama.cpp the parameter stays raw native text.
  const auto delimited = JsonConstraint::WithTools(
      nullptr,
      {{"f", JsonConstraint::ToolParameters(delimiter_pattern, true,
                                            Format::kQwen)}},
      true, false, Format::kQwen);
  assert(Accepts(*delimited,
                 "<tool_call>\n<function=f>\n<parameter=text>\nanything\n"
                 "</parameter>\n</function>\n</tool_call>"));
  for (const std::string name : {" text", "text ", "\ttext", "text\f"}) {
    auto unusual = gufo::json::Value::object();
    unusual["type"] = "object";
    unusual["properties"][name]["type"] = "string";
    unusual["required"] = gufo::json::Value::array();
    unusual["required"].push_back(name);
    unusual["additionalProperties"] = false;
    auto value = gufo::json::Value::object();
    value[name] = "literal";
    assert(Accepts(*JsonConstraint::Compile(unusual, true), value.dump()));
    // As llama.cpp, the name is written literally in native syntax (the
    // parser matches declared names exactly) rather than a JSON envelope.
    const auto qwen = JsonConstraint::WithTools(
        nullptr,
        {{"f", JsonConstraint::ToolParameters(unusual, true, Format::kQwen)}},
        true, false, Format::kQwen);
    assert(Accepts(*qwen, "<tool_call>\n<function=f>\n<parameter=" + name +
                              ">\nliteral\n</parameter>\n</function>\n"
                              "</tool_call>"));
    assert(!Accepts(*qwen,
                    "<tool_call>\n<function=f>\n</function>\n</tool_call>"));
    assert(JsonConstraint::ToolParameters(unusual, true, Format::kDeepSeek));
  }
}

void TestOpenNativeTools() {
  using Format = JsonConstraint::ToolFormat;
  for (const auto format : {Format::kQwen, Format::kDeepSeek}) {
    const auto parameters = JsonConstraint::OpenToolParameters(format);
    assert(parameters == JsonConstraint::OpenToolParameters(format));
    const bool qwen = format == Format::kQwen;
    const std::string begin = qwen ? "<tool_call>\n<function=f>\n"
                                   : "<｜DSML｜tool_calls>\n"
                                     "<｜DSML｜invoke name=\"f\">\n";
    const std::string end = qwen ? "</function>\n</tool_call>"
                                 : "</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    const auto parameter = [&](std::string_view name, std::string_view value) {
      return qwen ? "<parameter=" + std::string(name) + ">\n" +
                        std::string(value) + "\n</parameter>\n"
                  : "<｜DSML｜parameter name=\"" + std::string(name) +
                        "\" string=\"true\">" + std::string(value) +
                        "</｜DSML｜parameter>\n";
    };
    const auto grammar = JsonConstraint::WithTools(nullptr, {{"f", parameters}},
                                                   false, true, format);
    assert(Accepts(*grammar, "No tool is necessary."));
    assert(Accepts(*grammar, begin + end));
    const auto call = begin + parameter("value", "42") +
                      parameter("city name", " é🦉\n\\path\n</tool_call> ") +
                      end;
    assert(Accepts(*grammar, call));
    // DeepSeek parallel calls share one block. As in llama.cpp, calls end
    // the output in both formats.
    assert(Accepts(*grammar, call + "\n" + call) == qwen);
    assert(!Accepts(*grammar, call + " Done."));
    if (!qwen)
      assert(Accepts(*grammar, begin + parameter("value", "42") +
                                   "</｜DSML｜invoke>" +
                                   begin.substr(begin.find('\n')) + end));
    assert(!Accepts(*grammar, begin + parameter(" value", "x") + end));
    assert(!Accepts(*grammar, begin + parameter("value ", "x") + end));
    assert(!Accepts(*grammar, begin + parameter("", "x") + end));
    assert(!Accepts(*grammar, begin + parameter("v>evil", "x") + end));
    assert(!Accepts(*grammar, begin + parameter("value", "\xff") + end));
    if (!qwen) {
      const std::string typed =
          "<｜DSML｜parameter name=\"value\" string=\"false\">";
      assert(Accepts(*grammar, begin + typed +
                                   "{\"count\":7,\"ok\":true,\"values\":[null]}"
                                   "</｜DSML｜parameter>\n" +
                                   end));
      assert(!Accepts(*grammar, begin + typed +
                                    "not JSON"
                                    "</｜DSML｜parameter>\n" +
                                    end));
    }
    auto state = grammar->Start();
    const auto prefix = begin + parameter("value", "alpha");
    for (const unsigned char byte : prefix)
      state = grammar->Advance(state, byte);
    auto replay = state;
    for (const unsigned char byte : end) {
      state = grammar->Advance(state, byte);
      replay = grammar->Advance(replay, byte);
    }
    assert(state == replay && grammar->Complete(state));
  }
}

void TestNonStrictAgentTools() {
  // Pi's ordinary edit schema leaves nested objects open. This must not switch
  // the entire tool set to JSON or alter the model's native chat template.
  const auto schema = parse(R"({"type":"object","properties":{
    "path":{"type":"string"},
    "edits":{"type":"array","items":{"type":"object","properties":{
      "oldText":{"type":"string"},"newText":{"type":"string"}},
      "required":["oldText","newText"]}}},
    "required":["path","edits"],"additionalProperties":false})");
  using Format = JsonConstraint::ToolFormat;
  for (const auto format : {Format::kQwen, Format::kDeepSeek}) {
    const auto parameters =
        JsonConstraint::ToolParameters(schema, false, format);
    assert(parameters);
    const std::string edits =
        R"([{"oldText":"return a - b","newText":"return a + b"}])";
    const std::string call =
        format == Format::kQwen
            ? "<tool_call>\n<function=edit>\n<parameter=path>\ncalc.py\n"
              "</parameter>\n<parameter=edits>\n" +
                  edits + "\n</parameter>\n</function>\n</tool_call>"
            : "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"edit\">\n"
              "<｜DSML｜parameter name=\"path\" string=\"true\">calc.py"
              "</｜DSML｜parameter>\n"
              "<｜DSML｜parameter name=\"edits\" string=\"false\">" +
                  edits +
                  "</｜DSML｜parameter>\n</｜DSML｜invoke>\n"
                  "</｜DSML｜tool_calls>";
    for (const bool required : {false, true}) {
      const auto grammar = JsonConstraint::WithTools(
          nullptr, {{"edit", parameters}}, required, false, format);
      assert(Accepts(*grammar, call));
      assert(!Accepts(*grammar, call.substr(0, call.size() - 1)));
      assert(Accepts(*grammar, "No edit is needed.") == !required);
      const std::string empty =
          format == Format::kQwen
              ? "<tool_call>\n<function=edit>\n</function>\n</tool_call>"
              : "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"edit\">\n"
                "</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
      assert(!Accepts(*grammar, empty));
      auto wrong_type = call;
      wrong_type.replace(wrong_type.find(edits), edits.size(), "false");
      assert(!Accepts(*grammar, wrong_type));
      for (const char* invalid :
           {R"([{}])", R"([{"oldText":"a"}])",
            R"([{"oldText":"a","newText":42}])",
            R"([{"oldText":"a","newText":"b","oldText":42}])",
            R"([{"oldText":"a","newText":"b","\u006fldText":42}])"}) {
        auto nested = call;
        nested.replace(nested.find(edits), edits.size(), invalid);
        assert(!Accepts(*grammar, nested));
      }
      // #304: preserve multiline edits through JSON escapes. Raw controls in
      // an array's string must be excluded during generation, before parsing.
      auto multiline = call;
      multiline.replace(
          multiline.find(edits), edits.size(),
          R"([{"oldText":"a,\n  b","newText":"a,\n\tJSX,\n  b"}])");
      assert(Accepts(*grammar, multiline));
      auto raw_controls = call;
      raw_controls.replace(
          raw_controls.find(edits), edits.size(),
          "[{\"oldText\":\"a,\n  b\",\"newText\":\"a,\n\tJSX,\n  b\"}]");
      assert(!Accepts(*grammar, raw_controls));
      auto extra = call;
      extra.replace(extra.find(edits), edits.size(),
                    R"([{"oldText":"a","newText":"b","note":{"line":3}}])");
      assert(Accepts(*grammar, extra));
      auto missing = call;
      const auto edits_begin = missing.find(
          format == Format::kQwen ? "<parameter=edits>"
                                  : "<｜DSML｜parameter name=\"edits\"");
      const std::string close =
          format == Format::kQwen ? "</parameter>\n" : "</｜DSML｜parameter>\n";
      missing.erase(edits_begin, missing.find(close, edits_begin) +
                                     close.size() - edits_begin);
      assert(!Accepts(*grammar, missing));
      auto unknown = call;
      unknown.replace(unknown.find("edit"), 4, "undeclared");
      assert(!Accepts(*grammar, unknown));
    }
    bool rejected = false;
    try {
      (void)JsonConstraint::ToolParameters(schema, true, format);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
    // A literal holding the native closing delimiter cannot be written in
    // native tags. As llama.cpp, the call stays native and the string raw,
    // while the neighbor keeps its nested requirements.
    auto ambiguous = schema;
    ambiguous["properties"]["path"]["const"] = format == Format::kQwen
                                                   ? "a\n</parameter>\nb"
                                                   : "a</｜DSML｜parameter>b";
    const auto delimited = JsonConstraint::WithTools(
        nullptr,
        {{"edit", JsonConstraint::ToolParameters(ambiguous, false, format)}},
        true, false, format);
    assert(Accepts(*delimited, call));
    auto delimited_nested = call;
    delimited_nested.replace(delimited_nested.find(edits), edits.size(),
                             R"([{"oldText":"a"}])");
    assert(!Accepts(*delimited, delimited_nested));
    // Unsupported root annotations do not relax supported nested requirements.
    auto annotated = schema;
    annotated["unevaluatedProperties"] = true;
    annotated["properties"]["edits"]["x-client-extension"] = true;
    annotated["properties"]["edits"]["items"]["unevaluatedProperties"] = true;
    const auto annotated_parameters =
        JsonConstraint::ToolParameters(annotated, false, format);
    // Root wildcards cannot preserve arbitrary JSON types in Qwen's tags. As
    // llama.cpp, Qwen then generates only the declared parameters.
    assert(annotated_parameters);
    const auto annotated_call = JsonConstraint::WithTools(
        nullptr, {{"edit", annotated_parameters}}, true, false, format);
    assert(Accepts(*annotated_call, call));
    if (format == Format::kQwen) {
      auto wildcard = call;
      wildcard.insert(wildcard.find("</function>"),
                      "<parameter=extra>\n1\n</parameter>\n");
      assert(!Accepts(*annotated_call, wildcard));
    }
    const auto annotated_json =
        JsonConstraint::ToolParameters(annotated, false, Format::kJson);
    assert(Accepts(*annotated_json,
                   R"({"path":"a","edits":[{"oldText":"a","newText":"b"}]})"));
    auto referenced = parse(R"({"$ref":"#/$defs/edit","$defs":{}})");
    referenced["$defs"]["edit"] = schema;
    const auto referenced_parameters =
        JsonConstraint::ToolParameters(referenced, false, format);
    assert(referenced_parameters);
    const auto referenced_call = JsonConstraint::WithTools(
        nullptr, {{"edit", referenced_parameters}}, true, false, format);
    assert(Accepts(*referenced_call, call));
    for (const char* invalid : {R"([{}])", R"([{"oldText":"a"}])",
                                R"([{"oldText":"a","newText":42}])"}) {
      auto nested = call;
      nested.replace(nested.find(edits), edits.size(), invalid);
      assert(!Accepts(*annotated_call, nested));
      assert(!Accepts(*annotated_json,
                      std::string(R"({"path":"a","edits":)") + invalid + "}"));
      assert(!Accepts(*referenced_call, nested));
    }
  }
  // Best-effort reference handling must remain bounded too.
  const auto cyclic = parse(R"({"$ref":"#","type":"object"})");
  assert(JsonConstraint::ToolParameters(cyclic, false, Format::kQwen));
  assert(JsonConstraint::ToolParameters(cyclic, false, Format::kJson));

  // Open-object support belongs to non-strict tools, never strict response
  // schemas. Both native and JSON fallback arguments retain nested schemas.
  const auto json =
      JsonConstraint::ToolParameters(schema, false, Format::kJson);
  assert(Accepts(*json,
                 R"({"path":"a","edits":[{"oldText":"a","newText":"b"}]})"));
  assert(!Accepts(*json, R"({"path":"a","edits":[{}]})"));
  bool rejected = false;
  try {
    (void)JsonConstraint::Compile(schema, false);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);

  // Additional-property schemas and escaped/metacharacter keys must not
  // weaken the declared property rules. The key matcher sees decoded Unicode.
  const auto extras = parse(R"({
    "type":"object","properties":{"value":{"type":"object",
      "properties":{"a.b":{"type":"integer"},"😀":{"type":"string"}},
      "required":["a.b","😀"],"additionalProperties":{"type":"boolean"}}},
    "required":["value"],"additionalProperties":false})");
  const auto extra_json =
      JsonConstraint::ToolParameters(extras, false, Format::kJson);
  assert(Accepts(*extra_json, R"({"value":{"a.b":3,"😀":"x","other":true}})"));
  assert(Accepts(*extra_json, R"({"value":{"a.b":3,"😀":"x","aXb":false}})"));
  for (const char* invalid :
       {R"({"value":{"a.b":3,"😀":"x","other":1}})",
        R"({"value":{"a.b":3,"😀":"x","a.b":false}})",
        R"({"value":{"a.b":3,"😀":"x","a\u002eb":false}})",
        R"({"value":{"a.b":3,"😀":"x","\ud83d\ude00":false}})"})
    assert(!Accepts(*extra_json, invalid));
  auto finite = extras;
  finite["properties"]["value"]["enum"] = parse(
      R"([{"a.b":3,"😀":"x","other":true},{"a.b":3,"😀":"x","other":42}])");
  const auto finite_json =
      JsonConstraint::ToolParameters(finite, false, Format::kJson);
  assert(Accepts(*finite_json, R"({"value":{"a.b":3,"😀":"x","other":true}})"));
  assert(!Accepts(*finite_json, R"({"value":{"a.b":3,"😀":"x","other":42}})"));
}

void TestUntypedNonStrictTools() {
  using Format = JsonConstraint::ToolFormat;
  for (const auto format : {Format::kQwen, Format::kDeepSeek}) {
    const bool qwen = format == Format::kQwen;
    const std::string begin =
        qwen ? "<tool_call>\n<function=record>\n"
             : "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"record\">\n";
    const std::string end = qwen ? "</function>\n</tool_call>"
                                 : "</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    const std::string value =
        qwen ? "<parameter=value>\nalpha\n</parameter>\n"
             : "<｜DSML｜parameter name=\"value\" string=\"true\">alpha"
               "</｜DSML｜parameter>\n";
    for (
        const auto text :
        {R"({"type":"object","additionalProperties":true})",
         R"({"type":"object","unevaluatedProperties":true})",
         R"({"type":"object","additionalProperties":true,"dependentRequired":{"a":["b"]}})",
         R"({"type":"object","properties":{},"additionalProperties":true})",
         R"({"type":"object","properties":{},"additionalProperties":false,"unevaluatedProperties":true})"}) {
      const auto parameters =
          JsonConstraint::ToolParameters(parse(text), false, format);
      assert(parameters);
      const auto grammar = JsonConstraint::WithTools(
          nullptr, {{"record", parameters}}, false, false, format);
      assert(Accepts(*grammar, begin + value + end));
      assert(Accepts(*grammar, begin + end));
      const auto final_answer = JsonConstraint::Compile(
          parse(
              R"({"type":"object","properties":{"done":{"type":"boolean","const":true}},
                    "required":["done"],"additionalProperties":false})"),
          false);
      const auto combined = JsonConstraint::WithTools(
          final_answer, {{"record", parameters}}, false, false, format);
      assert(Accepts(*combined, R"({"done":true})"));
      assert(!Accepts(*combined, R"({"done":false})"));
      assert(!Accepts(*combined, R"({"other":true})"));
      assert(Accepts(*combined, begin + value + end));
    }
    // A genuinely closed empty schema still describes a zero-argument tool.
    for (const auto text :
         {R"({"type":"object","properties":{},"additionalProperties":false})",
          R"({"type":"object","unevaluatedProperties":false})", R"({})"}) {
      const auto parameters =
          JsonConstraint::ToolParameters(parse(text), false, format);
      assert(parameters);
      const auto grammar = JsonConstraint::WithTools(
          nullptr, {{"record", parameters}}, true, false, format);
      assert(Accepts(*grammar, begin + end));
      assert(!Accepts(*grammar, begin + value + end));
      const auto json =
          JsonConstraint::ToolParameters(parse(text), false, Format::kJson);
      assert(Accepts(*json, "{}"));
      assert(!Accepts(*json, R"({"value":"alpha"})"));
    }
    // Open properties do not erase exact finite values or typed extras.
    for (
        const auto text :
        {R"({"type":"object","additionalProperties":true,"const":{"value":"alpha"}})",
         R"({"type":"object","const":{"value":"alpha"}})",
         R"({"type":"object","enum":[{"value":"alpha"}],"unevaluatedProperties":true})",
         R"({"type":"object","additionalProperties":true,"const":{"value":"alpha"},"unevaluatedProperties":true})",
         R"({"type":"object","additionalProperties":true,"enum":[{"value":"alpha"}]})",
         R"({"type":"object","additionalProperties":{"type":"string"}})"}) {
      const auto schema = parse(text);
      // No declared properties: as in llama.cpp the call stays native and
      // takes no arguments.
      const auto native = JsonConstraint::WithTools(
          nullptr,
          {{"record", JsonConstraint::ToolParameters(schema, false, format)}},
          true, false, format);
      assert(Accepts(*native, begin + end));
      assert(!Accepts(*native, begin + value + end));
      const auto json =
          JsonConstraint::ToolParameters(schema, false, Format::kJson);
      assert(Accepts(*json, R"({"value":"alpha"})"));
      assert(!Accepts(*json, R"({"value":42})"));
    }
  }
}

void TestMixedBestEffortToolRoutes() {
  using Format = JsonConstraint::ToolFormat;
  const auto call = [](Format format, std::string_view name,
                       std::string_view value, bool string = true) {
    return format == Format::kQwen
               ? "<tool_call>\n<function=record>\n<parameter=" +
                     std::string(name) + ">\n" + std::string(value) +
                     "\n</parameter>\n</function>\n</tool_call>"
               : "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"record\">\n"
                 "<｜DSML｜parameter name=\"" +
                     std::string(name) + "\" string=\"" +
                     (string ? "true" : "false") + "\">" + std::string(value) +
                     "</｜DSML｜parameter>\n</｜DSML｜invoke>\n"
                     "</｜DSML｜tool_calls>";
  };
  for (
      const auto text :
      {R"({"type":"object","unevaluatedProperties":true})",
       R"({"type":"object","oneOf":[{"properties":{"a":{"type":"string"}},"required":["a"]}]})",
       R"({"type":"object","allOf":[{"properties":{"a":{"type":"string"}},"required":["a"]}]})"}) {
    const auto json =
        JsonConstraint::ToolParameters(parse(text), false, Format::kJson);
    assert(json && Accepts(*json, R"({"a":"x"})"));
  }
  for (
      const auto text :
      {R"({"$schema":"http://json-schema.org/draft-07/schema#","$id":"urn:record","$comment":"annotation",
            "$ref":"#/definitions/A","definitions":{"A":{"type":"object",
            "properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}}})",
       R"({"type":"object","$ref":"#/$defs/A","$defs":{"A":{"type":"object",
            "properties":{"a":{"type":"string"}},"required":["a"]}}})",
       R"({"$ref":"#/$defs/A","default":{"a":"x"},"examples":[{"a":"x"}],
            "$defs":{"A":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}}})",
       R"({"type":"object","properties":{"a":{"type":"string","format":"uri","minLength":1}},
            "required":["a"],"additionalProperties":false})"}) {
    for (const auto format :
         {Format::kQwen, Format::kDeepSeek, Format::kJson}) {
      const auto parameters =
          JsonConstraint::ToolParameters(parse(text), false, format);
      assert(parameters);
      if (format == Format::kJson) {
        assert(Accepts(*parameters, R"({"a":"x"})"));
        assert(!Accepts(*parameters, "{}"));
      } else {
        const auto grammar = JsonConstraint::WithTools(
            nullptr, {{"record", parameters}}, true, false, format);
        assert(Accepts(*grammar, call(format, "a", "x")));
        assert(!Accepts(*grammar, call(format, "other", "x")));
      }
    }
  }
  const auto nested =
      parse(R"({"type":"object","properties":{"m":{"type":"object",
    "patternProperties":{"^k":{"type":"integer"}},"additionalProperties":false}},
    "required":["m"]})");
  for (const auto format : {Format::kQwen, Format::kDeepSeek, Format::kJson}) {
    const auto parameters =
        JsonConstraint::ToolParameters(nested, false, format);
    assert(parameters);
    if (format == Format::kJson) {
      assert(Accepts(*parameters, R"({"m":{"k1":1}})"));
      assert(!Accepts(*parameters, "{}"));
    } else {
      const auto grammar = JsonConstraint::WithTools(
          nullptr, {{"record", parameters}}, true, false, format);
      assert(Accepts(*grammar, call(format, "m", R"({"k1":1})", false)));
    }
  }
  auto finite = nested;
  finite["const"] = parse(R"({"m":{"k1":1}})");
  const auto finite_json =
      JsonConstraint::ToolParameters(finite, false, Format::kJson);
  assert(Accepts(*finite_json, R"({"m":{"k1":1}})"));
  assert(!Accepts(*finite_json, R"({"m":{}})"));
  for (
      const auto text :
      {R"({"type":"object","properties":{"a":{"type":"string"}},"required":["a"],
            "patternProperties":{"^x_":{"type":"string"}}})",
       R"({"type":"object","properties":{"a":{"type":"string"}},"required":["a"],
            "dependentSchemas":{"a":{"properties":{"x_b":{"type":"string"}}}}})",
       R"({"type":"object","properties":{"a":{"type":"string"}},"required":["a"],
            "additionalProperties":true})"}) {
    for (const auto format :
         {Format::kQwen, Format::kDeepSeek, Format::kJson}) {
      const auto parameters =
          JsonConstraint::ToolParameters(parse(text), false, format);
      assert(parameters);
      if (format == Format::kQwen) {
        // Qwen tags cannot type wildcard values. As llama.cpp, only declared
        // parameters are generated and the call stays native.
        const auto grammar = JsonConstraint::WithTools(
            nullptr, {{"record", parameters}}, true, false, format);
        auto with_extra = call(format, "a", "1");
        with_extra.insert(with_extra.find("</function>"),
                          "<parameter=x_b>\n2\n</parameter>\n");
        assert(Accepts(*grammar, call(format, "a", "1")));
        assert(!Accepts(*grammar, with_extra));
        assert(!Accepts(*grammar, call(format, "x_b", "2")));
        continue;
      }
      if (format == Format::kJson) {
        assert(Accepts(*parameters, R"({"a":"1","x_b":"2"})"));
        assert(!Accepts(*parameters, R"({"a":1,"x_b":"2"})"));
        assert(!Accepts(*parameters, R"({"x_b":"2"})"));
        assert(!Accepts(*parameters, R"({"a":"1","a":2})"));
      } else {
        const auto grammar = JsonConstraint::WithTools(
            nullptr, {{"record", parameters}}, true, false, format);
        auto both = call(format, "a", "1");
        const auto extra =
            format == Format::kQwen
                ? "<parameter=x_b>\n2\n</parameter>\n"
                : "<｜DSML｜parameter name=\"x_b\" string=\"true\">2"
                  "</｜DSML｜parameter>\n";
        const auto offset = both.find(
            format == Format::kQwen ? "</function>" : "</｜DSML｜invoke>");
        both.insert(offset, extra);
        assert(Accepts(*grammar, both));
        assert(!Accepts(*grammar, call(format, "x_b", "2")));
        auto duplicate = both;
        duplicate.replace(duplicate.find("x_b"), 3, "a");
        assert(!Accepts(*grammar, duplicate));
      }
    }
  }
  // Non-strict unions keep native framing, as in llama.cpp. A JSON envelope
  // would contradict the template and the native calls in history (#383).
  // Qwen admits raw text for a union with strings; DeepSeek's string flag
  // selects raw text or the typed JSON alternatives.
  for (const auto format : {Format::kQwen, Format::kDeepSeek}) {
    const auto nullable = parse(R"({"type":"object","properties":{"a":{
      "anyOf":[{"type":"string"},{"type":"null"}]}},"required":["a"]})");
    const auto parameters =
        JsonConstraint::ToolParameters(nullable, false, format);
    assert(parameters);
    const auto grammar = JsonConstraint::WithTools(
        nullptr, {{"record", parameters}}, true, false, format);
    assert(Accepts(*grammar, call(format, "a", "null", false)));
    assert(Accepts(*grammar, call(format, "a", "#include \"a.h\"\nx")));
    assert(!Accepts(*grammar, call(format, "b", "x")));
    if (format == Format::kDeepSeek)
      assert(!Accepts(*grammar, call(format, "a", "x", false)));
    // Strict unions keep native framing too; llama.cpp has no strict mode
    // and the parser types the value against the schema.
    auto closed = nullable;
    closed["additionalProperties"] = false;
    const auto strict_grammar = JsonConstraint::WithTools(
        nullptr,
        {{"record", JsonConstraint::ToolParameters(closed, true, format)}},
        true, false, format);
    assert(Accepts(*strict_grammar, call(format, "a", "null", false)));
    assert(Accepts(*strict_grammar, call(format, "a", "text")));
    assert(!Accepts(*strict_grammar, call(format, "b", "x")));
    const auto json =
        JsonConstraint::ToolParameters(nullable, false, Format::kJson);
    assert(Accepts(*json, R"({"a":null})"));
    assert(Accepts(*json, R"({"a":"null"})"));
    // Finite string alternatives and untyped values are text; a union
    // without strings keeps its JSON value constraint inside the tag.
    const auto agent = parse(R"({"type":"object","properties":{
      "query":{"type":"string"},
      "provider":{"anyOf":[{"type":"string","const":"brave"},
                           {"type":"string","const":"exa"}]},
      "args":{"anyOf":[{"type":"string"},
                       {"type":"object","additionalProperties":true}]},
      "extra":{"description":"anything"},
      "limit":{"anyOf":[{"type":"integer"},{"type":"null"}]}}})");
    const auto tools = JsonConstraint::ToolParameters(agent, false, format);
    assert(tools);
    const auto agent_grammar = JsonConstraint::WithTools(
        nullptr, {{"record", tools}}, true, false, format);
    assert(Accepts(*agent_grammar, call(format, "query", "a \"quoted\" b")));
    assert(Accepts(*agent_grammar, call(format, "provider", "exa")));
    assert(Accepts(*agent_grammar, call(format, "extra", "x")));
    assert(Accepts(*agent_grammar, call(format, "limit", "3", false)));
    assert(Accepts(*agent_grammar, call(format, "limit", "null", false)));
    assert(!Accepts(*agent_grammar, call(format, "limit", "x", false)));
  }
  // Ignoring a numeric format annotation must retain supported bounds.
  const auto integer = parse(R"({"type":"object","properties":{"a":{
    "type":"integer","format":"int32","minimum":5,"maximum":10}},
    "required":["a"]})");
  for (const auto format : {Format::kQwen, Format::kDeepSeek, Format::kJson}) {
    const auto parameters =
        JsonConstraint::ToolParameters(integer, false, format);
    assert(parameters);
    if (format == Format::kJson) {
      assert(Accepts(*parameters, R"({"a":5})"));
      assert(!Accepts(*parameters, R"({"a":4})"));
    } else {
      const auto grammar = JsonConstraint::WithTools(
          nullptr, {{"record", parameters}}, true, false, format);
      assert(Accepts(*grammar, call(format, "a", "5", false)));
      assert(!Accepts(*grammar, call(format, "a", "4", false)));
    }
  }
}

void TestToolSchemaSafety() {
  using Format = JsonConstraint::ToolFormat;
  const auto ordinary = parse(R"({"type":"object","properties":{
    "value":{"type":"string","const":"alpha"}},"required":["value"],
    "additionalProperties":false})");
  const auto uri = parse(R"({"type":"object","properties":{
    "url":{"type":"string","format":"uri"}},"required":["url"]})");
  for (const auto format : {Format::kQwen, Format::kDeepSeek}) {
    // As in llama.cpp, a required call uses the same native arguments as an
    // automatic one, whatever annotations the schema carries.
    assert(JsonConstraint::ToolParameters(ordinary, false, format));
    assert(JsonConstraint::ToolParameters(uri, false, format));
    const auto required = JsonConstraint::WithTools(
        nullptr,
        {{"record", JsonConstraint::ToolParameters(uri, false, format)}}, true,
        false, format);
    assert(Accepts(*required,
                   format == Format::kQwen
                       ? "<tool_call>\n<function=record>\n<parameter=url>\n"
                         "https://a.b\n</parameter>\n</function>\n</tool_call>"
                       : "<｜DSML｜tool_calls>\n<｜DSML｜invoke "
                         "name=\"record\">\n<｜DSML｜parameter name=\"url\" "
                         "string=\"true\">https://a.b</｜DSML｜parameter>\n"
                         "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"));
  }
  for (
      const auto text :
      {R"({"type":"object","properties":{"x":{"type":"integer","minimum":5,"maximum":2}},"required":["x"]})",
       R"({"type":"object","properties":{"x":{"$ref":"#"}},"required":["x"]})",
       R"({"$ref":"#"})"}) {
    const auto schema = parse(text);
    // Unsatisfiable or recursive guidance cannot be enforced; the value stays
    // a generic native argument instead of switching to a JSON envelope. A
    // cyclic root declares no properties, so as llama.cpp it takes none.
    const bool declared = schema.contains("properties");
    for (const auto format : {Format::kQwen, Format::kDeepSeek}) {
      const auto grammar = JsonConstraint::WithTools(
          nullptr,
          {{"record", JsonConstraint::ToolParameters(schema, false, format)}},
          true, false, format);
      assert(Accepts(*grammar, format == Format::kQwen
                                   ? "<tool_call>\n<function=record>\n</"
                                     "function>\n</tool_call>"
                                   : "<｜DSML｜tool_calls>\n<｜DSML｜invoke "
                                     "name=\"record\">\n</｜DSML｜invoke>\n</"
                                     "｜DSML｜tool_calls>") == !declared);
      assert(Accepts(*grammar,
                     format == Format::kQwen
                         ? "<tool_call>\n<function=record>\n<parameter=x>\n5\n"
                           "</parameter>\n</function>\n</tool_call>"
                         : "<｜DSML｜tool_calls>\n<｜DSML｜invoke "
                           "name=\"record\">\n<｜DSML｜parameter name=\"x\" "
                           "string=\"false\">5</｜DSML｜parameter>\n"
                           "</｜DSML｜invoke>\n</｜DSML｜tool_calls>") ==
             declared);
    }
    const auto json =
        JsonConstraint::ToolParameters(schema, false, Format::kJson);
    assert(json && Accepts(*json, R"({"x":5})"));
  }

  // Optional impossible branches are pruned rather than admitting dead
  // prefixes. The remaining native call must still have a finite completion.
  const auto optional = parse(R"({"type":"object","properties":{
    "x":{"type":"integer","minimum":5,"maximum":2},
    "y":{"type":"string","const":"ok"}},"required":["y"],
    "additionalProperties":false})");
  const auto parameters =
      JsonConstraint::ToolParameters(optional, false, Format::kQwen);
  assert(parameters);
  const auto native = JsonConstraint::WithTools(
      nullptr, {{"record", parameters}}, true, false, Format::kQwen);
  assert(Accepts(
      *native,
      "<tool_call>\n<function=record>\n<parameter=y>\nok\n</parameter>\n"
      "</function>\n</tool_call>"));
  auto state = native->Start();
  for (unsigned char byte :
       std::string("<tool_call>\n<function=record>\n<parameter=x>\n"))
    state = native->Advance(state, byte);
  assert(state.empty());

  for (
      const auto extension :
      {R"({"patternProperties":{"^payload$":{"type":"integer"}}})",
       R"({"if":{"properties":{"kind":{"const":"x"}}},"then":{"properties":{"payload":{"type":"integer"}},"required":["payload"]}})",
       R"({"if":{"properties":{"kind":{"const":"y"}}},"else":{"properties":{"payload":{"type":"integer"}},"required":["payload"]}})",
       R"({"dependencies":{"kind":{"properties":{"payload":{"type":"integer"}},"required":["payload"]}}})"}) {
    auto schema = parse(extension);
    schema["type"] = "object";
    schema["properties"] = parse(R"({"kind":{"type":"string"}})");
    schema["required"] = parse(R"(["kind"])");
    // Qwen stays native and, as llama.cpp, generates only the root object's
    // declared names; branch keywords declare none.
    const auto qwen = JsonConstraint::WithTools(
        nullptr,
        {{"record",
          JsonConstraint::ToolParameters(schema, false, Format::kQwen)}},
        true, false, Format::kQwen);
    assert(Accepts(*qwen,
                   "<tool_call>\n<function=record>\n<parameter=kind>\nx\n"
                   "</parameter>\n</function>\n</tool_call>"));
    assert(!Accepts(*qwen,
                    "<tool_call>\n<function=record>\n<parameter=kind>\nx\n"
                    "</parameter>\n<parameter=payload>\n1\n</parameter>\n"
                    "</function>\n</tool_call>"));
    const auto json =
        JsonConstraint::ToolParameters(schema, false, Format::kJson);
    for (const auto value : {"1", "true", "null", "[1]", R"({"x":1})"}) {
      // Unsupported branches remain guidance. Types that the model chooses
      // must reach the HTTP parser intact, rather than becoming raw strings.
      assert(Accepts(*json,
                     std::string(R"({"kind":"x","payload":)") + value + "}"));
    }
    assert(!Accepts(*json, R"({"kind":1,"payload":1})"));
    assert(!Accepts(*json, R"({"payload":1})"));
    const auto ds =
        JsonConstraint::ToolParameters(schema, false, Format::kDeepSeek);
    assert(ds);
  }
  const auto metadata = parse(R"({"type":"object","properties":{
    "edits":{"type":"array","items":{"type":"object","properties":{
      "oldText":{"type":"string"},"newText":{"type":"string"},"metadata":{}},
      "required":["oldText","newText"]}}},"required":["edits"]})");
  const auto json =
      JsonConstraint::ToolParameters(metadata, false, Format::kJson);
  assert(Accepts(
      *json,
      R"({"edits":[{"oldText":"a","newText":"b","metadata":{"x":1}}]})"));
  assert(!Accepts(*json, R"({"edits":[{}]})"));
  assert(!Accepts(*json, R"({"edits":[{"oldText":1,"newText":"b"}]})"));
  for (const auto format : {Format::kQwen, Format::kDeepSeek})
    assert(JsonConstraint::ToolParameters(metadata, false, format));
  for (const auto format : {Format::kQwen, Format::kDeepSeek}) {
    const auto required = JsonConstraint::WithTools(
        nullptr,
        {{"record", JsonConstraint::ToolParameters(metadata, false, format)}},
        true, false, format);
    const std::string edits =
        R"([{"oldText":"a","newText":"b","metadata":{"x":1}}])";
    const auto call =
        format == Format::kQwen
            ? "<tool_call>\n<function=record>\n<parameter=edits>\n" + edits +
                  "\n</parameter>\n</function>\n</tool_call>"
            : "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"record\">\n"
              "<｜DSML｜parameter name=\"edits\" string=\"false\">" +
                  edits +
                  "</｜DSML｜parameter>\n</｜DSML｜invoke>\n"
                  "</｜DSML｜tool_calls>";
    assert(Accepts(*required, call));
    auto missing = call;
    missing.replace(missing.find(edits), edits.size(), R"([{}])");
    assert(!Accepts(*required, missing));
  }
}

// llama.cpp common/parsers/qwen3-coder.cpp and deepseek.cpp keep native call
// syntax for every schema: strings are raw text and other values follow their
// schema as far as it is supported. No schema, tool choice or strict flag may
// move a native model to a JSON envelope (#383, #438).
void TestNativeToolSchemaMatrix() {
  using Format = JsonConstraint::ToolFormat;
  struct Property {
    const char* schema;
    const char* value;
    bool string;
    const char* rejected;  // A value the native grammar must still refuse.
  };
  const std::vector<Property> properties{
      {R"({"type":"string"})", "plain \"text\"\n<b>", true, nullptr},
      {R"({"type":"string","pattern":"^[0-9]{4}$"})", "not a year", true,
       nullptr},
      {R"({"type":"string","format":"uri"})", "https://example.com", true,
       nullptr},
      {R"({"type":"string","default":"x","examples":["y"],"deprecated":true,"nullable":true,"title":"T"})",
       "x", true, nullptr},
      {R"({"type":"string","enum":["red","blue"]})", "red", true, nullptr},
      {R"({"type":"string","minLength":2})", "ab", true, nullptr},
      {R"({"type":"integer","exclusiveMinimum":0,"maximum":9007199254740991})",
       "240000", false, "x"},
      {R"({"type":"integer","minimum":5,"maximum":2})", "5", false, nullptr},
      {R"({"type":"boolean"})", "true", false, "yes"},
      {R"({"oneOf":[{"type":"string"},{"type":"integer"}]})", "free text", true,
       nullptr},
      {R"({"anyOf":[{"type":"integer"},{"type":"null"}]})", "3", false, "x"},
      {R"({"allOf":[{"type":"string"},{"minLength":1}]})", "text", true,
       nullptr},
      {R"({"allOf":[{"type":"integer"},{"minimum":1}]})", "2", false, "x"},
      {R"({"not":{"type":"null"}})", "anything", true, nullptr},
      {R"({"type":["string","null"]})", "text", true, nullptr},
      {R"({"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]})",
       R"({"x": 1})", false, "x"},
      {R"({"type":"array","items":{"type":"object","properties":{"k":{"type":"string"}},"required":["k"]}})",
       R"([{"k": "v"}])", false, "[{}]"},
      {R"({"$ref":"#/$defs/S"})", "referenced", true, nullptr},
      {R"({})", "untyped", true, nullptr},
      {R"({"if":{"type":"string"},"then":{"minLength":1}})", "conditional",
       true, nullptr},
  };
  // Roots marked open admit extra names, which only DeepSeek's typed string
  // flag can carry; Qwen generates declared names only, as llama.cpp.
  const std::vector<std::pair<const char*, bool>> roots{
      {R"({"type":"object"})", false},
      {R"({"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","additionalProperties":false})",
       false},
      {R"({"type":"object","additionalProperties":true})", true},
      {R"({"type":"object","patternProperties":{"^x_":{"type":"integer"}}})",
       true},
      {R"({"type":"object","if":{"required":["v"]},"then":{"required":["v"]}})",
       true},
      {R"({"type":"object","unevaluatedProperties":true,"title":"Args","description":"d"})",
       true},
  };
  std::size_t checked = 0;
  for (const auto& [root_text, open] : roots) {
    for (const auto& property : properties) {
      auto schema = parse(root_text);
      schema["properties"] = gufo::json::Value::object();
      schema["properties"]["v"] = parse(property.schema);
      schema["properties"]["other"] = parse(R"({"type":"string"})");
      schema["required"] = parse(R"(["v"])");
      schema["$defs"] = parse(R"({"S":{"type":"string"}})");
      for (const auto format : {Format::kQwen, Format::kDeepSeek}) {
        const bool qwen = format == Format::kQwen;
        const auto argument = [&](std::string_view name, std::string_view value,
                                  bool string) {
          return qwen ? "<parameter=" + std::string(name) + ">\n" +
                            std::string(value) + "\n</parameter>\n"
                      : "<｜DSML｜parameter name=\"" + std::string(name) +
                            "\" string=\"" + (string ? "true" : "false") +
                            "\">" + std::string(value) +
                            "</｜DSML｜parameter>\n";
        };
        const auto call = [&](const std::string& arguments) {
          return (qwen
                      ? "<tool_call>\n<function=t>\n"
                      : "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"t\">\n") +
                 arguments +
                 (qwen ? "</function>\n</tool_call>"
                       : "</｜DSML｜invoke>\n</｜DSML｜tool_calls>");
        };
        for (const bool strict : {false, true}) {
          for (const bool required : {false, true}) {
            std::shared_ptr<const JsonConstraint> parameters;
            try {
              parameters =
                  JsonConstraint::ToolParameters(schema, strict, format);
            } catch (const std::invalid_argument&) {
              // Only an impossible or unsupported strict schema is rejected.
              assert(strict);
              continue;
            }
            assert(parameters && "a native model never uses a JSON envelope");
            const auto grammar = JsonConstraint::WithTools(
                nullptr, {{"t", parameters}}, required, false, format);
            const auto value = argument("v", property.value, property.string);
            if (!Accepts(*grammar, call(value)))
              std::cerr << "value refused: root=" << root_text
                        << " v=" << property.schema << " qwen=" << qwen
                        << " strict=" << strict << "\n";
            assert(Accepts(*grammar, call(value)));
            assert(
                Accepts(*grammar, call(argument("other", "o", true) + value)));
            if (Accepts(*grammar, call(argument("other", "o", true))))
              std::cerr << "missing required accepted: root=" << root_text
                        << " v=" << property.schema << " qwen=" << qwen
                        << " strict=" << strict << " required=" << required
                        << "\n";
            assert(!Accepts(*grammar, call(argument("other", "o", true))));
            // Undeclared names are never generated for Qwen, strict tools or
            // closed roots; DeepSeek's typed string flag may carry them for an
            // open root, as its native ToolParameters already admits.
            if (qwen || strict || !open) {
              if (Accepts(*grammar, call(value + argument("zz", "1", true))))
                std::cerr << "undeclared accepted: root=" << root_text
                          << " v=" << property.schema << " qwen=" << qwen
                          << " strict=" << strict << "\n";
              assert(
                  !Accepts(*grammar, call(value + argument("zz", "1", true))));
            }
            if (property.rejected &&
                Accepts(*grammar, call(argument("v", property.rejected,
                                                property.string))))
              std::cerr << "rejected value accepted: root=" << root_text
                        << " v=" << property.schema << " qwen=" << qwen
                        << " strict=" << strict << "\n";
            if (property.rejected)
              assert(!Accepts(*grammar, call(argument("v", property.rejected,
                                                      property.string))));
            ++checked;
          }
        }
      }
    }
  }
  assert(checked >= roots.size() * properties.size() * 2 * 2);
}

// Verdicts of llama.cpp's own Qwen3-Coder and DeepSeek tool grammars (GBNF
// engine, tool_choice required) for 86 schemas x 4 outputs. Every difference
// is recorded with its reason. tools/llama_parity regenerates the fixture.
void TestLlamaCppToolGrammarGoldens() {
  using Format = JsonConstraint::ToolFormat;
  std::ifstream in(GUFO_LLAMA_TOOL_GRAMMAR_GOLDENS);
  assert(in && "llama.cpp tool grammar goldens are present");
  const auto fixture =
      parse(std::string(std::istreambuf_iterator<char>(in), {}));
  std::size_t rows = 0, differences = 0;
  for (const auto& row : fixture.find("rows")->items()) {
    const auto format =
        row.member_str("format") == "qwen" ? Format::kQwen : Format::kDeepSeek;
    const auto grammar = JsonConstraint::WithTools(
        nullptr,
        {{"record", JsonConstraint::ToolParameters(*row.find("parameters"),
                                                   false, format)}},
        true, false, format);
    const bool accepted = Accepts(*grammar, row.member_str("output"));
    const bool llama = row.find("llama_cpp")->as_bool();
    const bool expected = row.find("gufo")->as_bool();
    if (accepted != expected)
      std::cerr << "llama.cpp golden mismatch: " << row.member_str("format")
                << " " << row.member_str("case") << " "
                << row.member_str("class") << " gufo=" << accepted
                << " recorded=" << expected << "\n";
    assert(accepted == expected);
    const bool explained = row.contains("difference");
    assert(explained == (accepted != llama));
    // gufo is never stricter on a valid output, except a const it enforces.
    if (llama && row.member_str("class") == "valid" && !accepted)
      assert(row.member_str("difference").find("const") != std::string::npos);
    ++rows;
    differences += explained;
  }
  assert(rows == 688 && differences < rows / 5);
}

int main(int argc, char** argv) {
  // Batch probes for the independent Python JSON Schema validator. This
  // exercises the production byte matcher without requiring model weights.
  if (argc == 2 && std::string_view(argv[1]) == "--probe") {
    std::string line;
    while (std::getline(std::cin, line)) {
      auto output = gufo::json::Value::object();
      try {
        const auto input = parse(line);
        const auto grammar =
            JsonConstraint::Compile(*input.find("schema"), true);
        output["accepted"] = gufo::json::Value::array();
        for (const auto& text : input.find("texts")->items())
          output["accepted"].push_back(Accepts(*grammar, text.str()));
        if (const auto* prefixes = input.find("prefixes")) {
          output["viable"] = gufo::json::Value::array();
          for (const auto& text : prefixes->items()) {
            auto state = grammar->Start();
            for (unsigned char byte : text.str())
              state = grammar->Advance(state, byte);
            output["viable"].push_back(!state.empty());
          }
        }
      } catch (const std::exception& error) {
        output["error"] = error.what();
      }
      std::cout << output.dump() << '\n';
    }
    return 0;
  }
  TestJsonLanguage();
  TestSchemaLanguage();
  TestSchemaIntersections();
  TestRejectedSchemas();
  TestIntegerBoundsAndRecursion();
  TestPrimitiveConstraints();
  TestTokensAndSampling();
  TestStringMaskCache();
  TestUnsupportedPatterns();
  TestNativeRequestMaskReuse();
  TestReasoningConstraint();
  TestAutomaticTools();
  TestNativeTools();
  TestOpenNativeTools();
  TestNonStrictAgentTools();
  TestUntypedNonStrictTools();
  TestMixedBestEffortToolRoutes();
  TestToolSchemaSafety();
  TestNativeToolSchemaMatrix();
  TestLlamaCppToolGrammarGoldens();
  std::cout << "JSON constraints: language, schema, Unicode and sampler checks "
               "passed\n";
}
