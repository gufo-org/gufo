#include "src/models/gemma4/tool_syntax.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/json_constraint.hpp"
#include "src/models/gemma4/chat_template.hpp"
#include "tests/models/gemma4/check.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;
using gufo::sampling::JsonConstraint;

namespace {

std::string ParseToJson(const std::string& body, std::string* name = nullptr) {
  std::string error;
  const auto call = g4::ParseToolCall(body, &error);
  Require(call.has_value(), "rejected " + body + ": " + error);
  if (name != nullptr) {
    *name = call->name;
  }
  return call->arguments.dump();
}

void CheckValues() {
  std::string name;
  Require(ParseToJson("call:get_weather{location:<|\"|>Paris, FR<|\"|>}",
                      &name) == R"({"location":"Paris, FR"})",
          "string argument");
  Require(name == "get_weather", "function name");
  Require(ParseToJson("call:mcp/github.create_issue:v2{n:1}", &name) ==
                  R"({"n":1})" &&
              name == "mcp/github.create_issue:v2",
          "namespaced function name");
  Require(ParseToJson("call:f{}") == "{}", "no arguments");
  Require(ParseToJson("call:f{n:42,x:-1.5,e:1e-05,t:true,f:false,z:null}") ==
              R"({"n":42,"x":-1.5,"e":1e-05,"t":true,"f":false,"z":null})",
          "literals");
  Require(ParseToJson("call:f{a:[1,<|\"|>two<|\"|>,[]],o:{k:{deep:1}}}") ==
              R"({"a":[1,"two",[]],"o":{"k":{"deep":1}}})",
          "nested containers");
  Require(
      ParseToJson("call:f{<|\"|>quoted key<|\"|>:1}") == R"({"quoted key":1})",
      "quoted key");
  // Strings are raw between delimiters: quotes, braces, commas, newlines.
  Require(ParseToJson("call:f{s:<|\"|>a \"b\" {c}, d\n<|\"|>}") ==
              R"({"s":"a \"b\" {c}, d\n"})",
          "raw string payload");
  Require(ParseToJson("call:f{mode:fast}") == R"({"mode":"fast"})",
          "bare word kept as string");
  Require(ParseToJson("call:f{ a : 1 , b : <|\"|>x<|\"|> }") ==
              R"({"a":1,"b":"x"})",
          "whitespace between tokens");
}

void CheckRejections() {
  for (const char* body :
       {"", "call:", "call:{}", "f{a:1}", "call:f{a:1", "call:f{a:<|\"|>x}",
        "call:f{a:1}}", "call:f{a}", "call:f{a:[1,2}", "call:f{:1}"}) {
    std::string error;
    Require(!g4::ParseToolCall(body, &error).has_value() && !error.empty(),
            std::string("accepted malformed call: ") + body);
  }
  std::string deep = "call:f{a:";
  for (int i = 0; i < 100; ++i) {
    deep += "[";
  }
  Require(!g4::ParseToolCall(deep).has_value(), "unbounded nesting accepted");
}

bool Accepts(const JsonConstraint& grammar, std::string_view text) {
  auto state = grammar.Start();
  for (const unsigned char byte : text) {
    state = grammar.Advance(state, byte);
    if (state.empty()) {
      return false;
    }
  }
  return grammar.Complete(state);
}

/// The call exactly as the chat template renders it in conversation history.
std::string RenderedCall(const std::string& name,
                         const gufo::json::Value& arguments) {
  gufo::tokenization::ChatMessage user;
  user.role = gufo::tokenization::ChatRole::kUser;
  user.content = "go";
  gufo::tokenization::ChatMessage assistant;
  assistant.role = gufo::tokenization::ChatRole::kAssistant;
  gufo::tokenization::ChatMessage::ToolCall call{.id = "c", .name = name};
  for (const auto& [key, value] : arguments.members()) {
    call.arguments.push_back(
        {.name = key,
         .value = value.is_string() ? value.str() : value.dump(),
         .is_string = value.is_string()});
  }
  assistant.tool_calls.push_back(std::move(call));
  const std::vector messages{user, assistant};
  std::string error;
  const auto rendered = g4::ChatTemplate::Render(
      messages, {}, {.add_generation_prompt = false}, &error);
  Require(rendered.has_value(), "template rejected the call: " + error);
  const auto begin = rendered->text.find(g4::kToolCallStart);
  const auto end = rendered->text.find(g4::kToolCallEnd);
  Require(begin != std::string::npos && end != std::string::npos,
          "rendered history holds no call");
  return rendered->text.substr(begin, end + g4::kToolCallEnd.size() - begin);
}

void CheckGrammar() {
  using Format = JsonConstraint::ToolFormat;
  const auto schema = gufo::json::parse(R"({
    "type":"object",
    "properties":{
      "path":{"type":"string","minLength":1},
      "edits":{"type":"array","minItems":1,"items":{
        "type":"object",
        "properties":{"oldText":{"type":"string"},"newText":{"type":"string"}},
        "required":["oldText","newText"]}},
      "mode":{"type":"string","enum":["fast","safe"]},
      "count":{"type":"integer","minimum":1,"maximum":9},
      "dry":{"type":["boolean","null"]}},
    "required":["path","edits"]})");
  const auto parameters =
      JsonConstraint::ToolParameters(schema, false, Format::kGemma4);
  Require(parameters != nullptr, "Gemma 4 tool grammar compiles");
  const auto required = JsonConstraint::WithTools(
      nullptr, {{"edit", parameters}}, true, false, Format::kGemma4);
  const auto arguments = gufo::json::parse(R"({
    "path":"calc.py","mode":"safe","count":3,"dry":null,
    "edits":[{"oldText":"a, {b}: \"c\"\n","newText":""}]})");
  const auto call = RenderedCall("edit", arguments);
  // The grammar admits the template's own rendering (sorted, compact keys).
  Require(Accepts(*required, call), "template call accepted: " + call);
  const auto body = call.substr(
      g4::kToolCallStart.size(),
      call.size() - g4::kToolCallStart.size() - g4::kToolCallEnd.size());
  Require(ParseToJson(body) == g4::ParseToolCall(body)->arguments.dump() &&
              g4::ParseToolCall(body)->arguments.dump() ==
                  gufo::json::parse(R"({"count":3,"dry":null,
                    "edits":[{"newText":"","oldText":"a, {b}: \"c\"\n"}],
                    "mode":"safe","path":"calc.py"})")
                      .dump(),
          "accepted call parses to its arguments");
  for (const std::string invalid : {
           // Missing a required argument, wrong types, enum and bounds.
           "<|tool_call>call:edit{path:<|\"|>x<|\"|>}<tool_call|>",
           "<|tool_call>call:edit{edits:[],path:<|\"|>x<|\"|>}<tool_call|>",
           "<|tool_call>call:edit{count:0,edits:[{newText:<|\"|><|\"|>,"
           "oldText:<|\"|><|\"|>}],path:<|\"|>x<|\"|>}<tool_call|>",
           "<|tool_call>call:edit{edits:[{newText:<|\"|><|\"|>,oldText:"
           "<|\"|><|\"|>}],mode:<|\"|>slow<|\"|>,path:<|\"|>x<|\"|>}"
           "<tool_call|>",
           // Unsorted keys, JSON spelling, an undeclared tool and prose.
           "<|tool_call>call:edit{path:<|\"|>x<|\"|>,edits:[{newText:"
           "<|\"|><|\"|>,oldText:<|\"|><|\"|>}]}<tool_call|>",
           "<|tool_call>call:edit{\"edits\":[],\"path\":\"x\"}<tool_call|>",
           "<|tool_call>call:other{}<tool_call|>",
           "ordinary text",
       }) {
    Require(!Accepts(*required, invalid), "rejected: " + invalid);
  }
  Require(!Accepts(*required, call + call), "one call without parallel calls");
  const auto automatic = JsonConstraint::WithTools(
      nullptr, {{"edit", parameters}}, false, true, Format::kGemma4);
  Require(Accepts(*automatic, "Let me fix it. " + call + call) &&
              Accepts(*automatic, "No call needed."),
          "automatic calls keep prose and parallel calls");
  // With thinking on, the output may open the thought channel first.
  const auto thinking = JsonConstraint::WithReasoning(required, g4::kThoughtEnd,
                                                      g4::kThoughtStart);
  Require(Accepts(*thinking, call) &&
              Accepts(*thinking, std::string(g4::kThoughtStart) +
                                     "Plan <|tool_call>." +
                                     std::string(g4::kThoughtEnd) + call) &&
              !Accepts(*thinking, "Plan" + call),
          "optional thought channel before a required call");

  // Keys the parser cannot read bare are quoted; a key that must be quoted
  // but holds the quote has no spelling: strict fails, best effort stays open.
  for (const std::string key : {"a:b", " value ", "<x>"}) {
    auto unusual = gufo::json::Value::object();
    unusual["type"] = "object";
    unusual["properties"][key]["type"] = "string";
    unusual["required"] = gufo::json::Value::array();
    unusual["required"].push_back(key);
    unusual["additionalProperties"] = false;
    const auto grammar =
        JsonConstraint::ToolParameters(unusual, true, Format::kGemma4);
    const auto quoted = "{<|\"|>" + key + "<|\"|>:<|\"|>v<|\"|>}";
    Require(grammar != nullptr && Accepts(*grammar, quoted) &&
                ParseToJson("call:f" + quoted) ==
                    gufo::json::parse("{}").dump().replace(
                        1, 0, gufo::json::Value(key).dump() + ":\"v\""),
            "quoted key round trip: " + key);
  }
  const auto quote = gufo::json::parse(R"({"type":"object",
    "properties":{"<|\"|>b":{"type":"string"}},"required":["<|\"|>b"],
    "additionalProperties":false})");
  bool rejected = false;
  try {
    (void)JsonConstraint::ToolParameters(quote, true, Format::kGemma4);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "strict unrepresentable key rejected");
  const auto open =
      JsonConstraint::ToolParameters(quote, false, Format::kGemma4);
  Require(open != nullptr && Accepts(*open, "{x:1,y:<|\"|>z<|\"|>}"),
          "best-effort unrepresentable key keeps native open arguments");
  rejected = false;
  try {
    (void)JsonConstraint::WithTools(nullptr, {{"x{y", parameters}}, true, false,
                                    Format::kGemma4);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "a function name with a brace has no Gemma 4 call");
}

}  // namespace

int main() {
  return gemma4_test::Run([] {
    CheckValues();
    CheckRejections();
    CheckGrammar();
  });
}
