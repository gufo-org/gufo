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
       R"({"type":"object","properties":{"x":{"type":"string","pattern":"a"}},"additionalProperties":false,"required":["x"]})",
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

int main() {
  TestJsonLanguage();
  TestSchemaLanguage();
  TestRejectedSchemas();
  TestTokensAndSampling();
  std::cout << "JSON constraints: language, schema, Unicode and sampler checks "
               "passed\n";
}
