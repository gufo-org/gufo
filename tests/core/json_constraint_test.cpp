#include "src/core/json_constraint.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <string_view>

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

void TestPrimitiveConstraints() {
  auto compile = [](std::string_view property) {
    return JsonConstraint::Compile(
        parse(R"({"type":"object","properties":{"x":)" + std::string(property) +
              R"(},"required":["x"],"additionalProperties":false})"),
        true);
  };
  const auto pattern = compile(
      R"({"type":"string","pattern":"^@[a-zA-Z0-9_]+$","maxLength":6})");
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
      "{\"x\":",   "true",      "false",    "}",    "INVALID",
      "",          "{\"x\":\"", "\xe2\x94", "\x8c", "\"}",
      "false}BAD", "false}",    "<think>"};
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
  std::fill(logits.begin(), logits.end(),
            -std::numeric_limits<float>::infinity());
  bool rejected = false;
  try {
    (void)sampler.Sample(logits);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
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
  const auto mixed = JsonConstraint::WithTools(plain, {{"score", tool}}, false);
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
  const auto required =
      JsonConstraint::WithTools(plain, {{"score", tool}}, true);
  assert(!Accepts(*required, "{\"final\":true}"));
  assert(Accepts(
      *JsonConstraint::WithReasoning(required),
      R"(Use the scoring tool.</think><tool_call>{"name":"score","arguments":{"value":3}}</tool_call>)"));
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
      } catch (const std::exception& error) {
        output["error"] = error.what();
      }
      std::cout << output.dump() << '\n';
    }
    return 0;
  }
  TestJsonLanguage();
  TestSchemaLanguage();
  TestRejectedSchemas();
  TestIntegerBoundsAndRecursion();
  TestPrimitiveConstraints();
  TestTokensAndSampling();
  TestReasoningConstraint();
  std::cout << "JSON constraints: language, schema, Unicode and sampler checks "
               "passed\n";
}
