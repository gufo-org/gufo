#include "src/core/json_constraint.hpp"

#include <algorithm>
#include <array>
#include <barrier>
#include <cassert>
#include <cmath>
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
  TestReasoningConstraint();
  std::cout << "JSON constraints: language, schema, Unicode and sampler checks "
               "passed\n";
}
