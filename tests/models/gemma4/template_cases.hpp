#ifndef GUFO_TESTS_MODELS_GEMMA4_TEMPLATE_CASES_HPP_
#define GUFO_TESTS_MODELS_GEMMA4_TEMPLATE_CASES_HPP_

#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/gemma4/chat_template.hpp"
#include "tests/models/gemma4/check.hpp"

namespace gemma4_test {

struct TemplateCase {
  std::string name;
  std::vector<gufo::tokenization::ChatMessage> messages;
  std::vector<gufo::tokenization::ChatTool> tools;
  gufo::models::gemma4::ChatOptions options;
};

inline gufo::json::Value ReadJson(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  Require(input.good(), "cannot open " + path);
  std::stringstream buffer;
  buffer << input.rdbuf();
  return gufo::json::parse(buffer.str());
}

inline gufo::tokenization::ChatRole ParseRole(const std::string& role) {
  using gufo::tokenization::ChatRole;
  if (role == "system")
    return ChatRole::kSystem;
  if (role == "developer")
    return ChatRole::kDeveloper;
  if (role == "assistant")
    return ChatRole::kAssistant;
  if (role == "tool")
    return ChatRole::kTool;
  return ChatRole::kUser;
}

/// Converts the shared fixture into the server's message types, the same
/// way the OpenAI adapter does (tool arguments as name/value pairs).
inline std::vector<TemplateCase> LoadTemplateCases(const std::string& path) {
  const auto spec = ReadJson(path);
  std::map<std::string, gufo::tokenization::ChatTool> tools;
  for (const auto& [name, tool] : spec.find("tools")->members()) {
    const auto* function = tool.find("function");
    gufo::tokenization::ChatTool t;
    t.name = function->member_str("name");
    t.description = function->member_str("description");
    const auto* params = function->find("parameters");
    t.parameters_json = params != nullptr ? params->dump() : "{}";
    t.definition_json = tool.dump();
    tools.emplace(name, t);
  }
  std::vector<TemplateCase> cases;
  for (const auto& [name, c] : spec.find("cases")->members()) {
    TemplateCase tc;
    tc.name = name;
    if (const auto* v = c.find("add_generation_prompt")) {
      tc.options.add_generation_prompt = v->as_bool();
    }
    if (const auto* v = c.find("enable_thinking")) {
      tc.options.enable_thinking = v->as_bool();
    }
    if (const auto* v = c.find("preserve_thinking")) {
      tc.options.preserve_thinking = v->as_bool();
    }
    if (const auto* list = c.find("tools")) {
      for (const auto& tool : list->items()) {
        tc.tools.push_back(tools.at(tool.str()));
      }
    }
    for (const auto& m : c.find("messages")->items()) {
      gufo::tokenization::ChatMessage message;
      message.role = ParseRole(m.member_str("role"));
      const auto* content = m.find("content");
      if (content != nullptr && content->is_array()) {
        // Content parts, flattened as the OpenAI adapter does: text appends,
        // an image records its byte offset.
        for (const auto& part : content->items()) {
          if (part.member_str("type") == "image_url") {
            message.images.push_back(
                {message.content.size(),
                 std::make_shared<const std::vector<std::uint8_t>>()});
          } else {
            message.content += part.member_str("text");
          }
        }
      } else {
        message.content = m.member_str("content");
      }
      message.name = m.member_str("name");
      message.thought = m.member_str("reasoning_content");
      message.tool_call_id = m.member_str("tool_call_id");
      if (const auto* calls = m.find("tool_calls")) {
        for (const auto& call : calls->items()) {
          gufo::tokenization::ChatMessage::ToolCall tc_call;
          tc_call.id = call.member_str("id");
          tc_call.name = call.member_str("name");
          for (const auto& [arg, value] : call.find("arguments")->members()) {
            tc_call.arguments.push_back(
                {arg, value.is_string() ? value.str() : value.dump(),
                 value.is_string()});
          }
          message.tool_calls.push_back(std::move(tc_call));
        }
      }
      tc.messages.push_back(std::move(message));
    }
    cases.push_back(std::move(tc));
  }
  return cases;
}

}  // namespace gemma4_test

#endif  // GUFO_TESTS_MODELS_GEMMA4_TEMPLATE_CASES_HPP_
