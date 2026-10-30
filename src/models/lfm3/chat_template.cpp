#include "src/models/lfm3/chat_template.hpp"

#include <algorithm>
#include <stdexcept>

#include "src/core/json.hpp"
#include "src/models/lfm3/tokenizer.hpp"
#include "src/models/qwen/control_tokens.hpp"
namespace gufo::models::lfm3 {
namespace {
using tokenization::ChatRole;
using tokenization::QwenToolSyntax;
struct Rendered {
  std::string text{"<|startoftext|>"};
  std::vector<std::pair<std::size_t, std::size_t>> literals;
  void Literal(std::string_view value) {
    if (!value.empty())
      literals.emplace_back(text.size(), value.size());
    text += value;
  }
};
std::string_view Trim(std::string_view s) {
  const auto first = s.find_first_not_of(" \t\r\n");
  if (first == s.npos)
    return {};
  return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}
std::string ToolDefinition(const tokenization::ChatTool& tool) {
  auto fn = json::Value::object();
  if (!tool.definition_json.empty()) {
    auto def = json::parse(tool.definition_json);
    fn = def.find("function") ? *def.find("function") : std::move(def);
  } else {
    fn["name"] = tool.name;
    fn["description"] = tool.description;
    fn["parameters"] =
        json::parse(tool.parameters_json.empty() ? "{}" : tool.parameters_json);
  }
  auto filtered = json::Value::object();
  for (const auto& [key, value] : fn.members())
    if (key != "defer_loading")
      filtered[key] = value;
  return filtered.tojson();
}
void Calls(Rendered& out,
           std::span<const tokenization::ChatMessage::ToolCall> calls) {
  for (std::size_t i = 0; i < calls.size(); ++i) {
    if (i)
      out.text += "\n\n";
    const auto& call = calls[i];
    out.text += QwenToolSyntax::kCallOpen;
    out.text += '\n';
    out.text += QwenToolSyntax::kFunctionOpen;
    out.text += call.name + ">\n";
    for (const auto& arg : call.arguments) {
      out.text += QwenToolSyntax::kParameterOpen;
      out.text += arg.name + ">\n";
      if (arg.is_string)
        out.Literal(arg.value);
      else
        out.Literal(json::parse(arg.value).tojson());
      out.text += '\n';
      out.text += QwenToolSyntax::kParameterClose;
      out.text += '\n';
    }
    out.text += QwenToolSyntax::kFunctionClose;
    out.text += '\n';
    out.text += QwenToolSyntax::kCallClose;
  }
}
Rendered Render(std::span<const tokenization::ChatMessage> messages,
                std::span<const tokenization::ChatTool> tools,
                const ReasoningOptions& reasoning, bool generation,
                std::span<const ImageExpansion> images) {
  Rendered out;
  std::size_t first = 0;
  std::optional<std::size_t> last_user;
  for (std::size_t i = 0; i < messages.size(); ++i) {
    if (!messages[i].images.empty() && images.empty())
      throw std::invalid_argument("image input requires an LFM3-VL projector");
    if (!messages[i].images.empty() && messages[i].role != ChatRole::kUser)
      throw std::invalid_argument("LFM3-VL images require a user message");
    if (messages[i].role == ChatRole::kUser)
      last_user = i;
  }
  std::size_t image_index = 0;
  std::string system;
  if (!messages.empty() && messages.front().role == ChatRole::kSystem) {
    system = messages.front().content + messages.front().framing_suffix;
    first = 1;
  }
  if (!system.empty() || !tools.empty()) {
    out.text += tokenization::kImStart;
    out.text += "system\n";
    out.Literal(system);
    if (!tools.empty()) {
      if (!system.empty())
        out.text += "\n\n";
      out.text += "<tools>\n";
      for (std::size_t i = 0; i < tools.size(); ++i) {
        if (i)
          out.text += '\n';
        out.Literal(ToolDefinition(tools[i]));
      }
      out.text += "\n</tools>";
    }
    out.text += tokenization::kImEnd;
    out.text += '\n';
  }
  for (std::size_t i = first; i < messages.size(); ++i) {
    const auto& m = messages[i];
    out.text += tokenization::kImStart;
    out.text += tokenization::ToString(m.role);
    out.text += '\n';
    if (m.role == ChatRole::kAssistant) {
      auto content = std::string_view(m.content);
      const bool keep = reasoning.preserve_thinking.value_or(false) ||
                        !last_user || i > *last_user;
      if (keep) {
        if (!m.thought.empty()) {
          out.text += "<think>";
          out.Literal(m.thought);
          out.text += "</think>";
        } else if (content.find("</think>") == content.npos)
          out.text += "<think></think>";
      } else {
        auto end = content.rfind("</think>");
        if (end != content.npos)
          content = Trim(content.substr(end + 8));
        out.text += "<think></think>";
      }
      out.Literal(content);
      out.Literal(m.framing_suffix);
      Calls(out, m.tool_calls);
    } else if (m.role == ChatRole::kTool) {
      // Adjacent parts of one call are one native tool response.
      std::string content = m.content;
      while (i + 1 < messages.size() &&
             messages[i + 1].role == ChatRole::kTool &&
             !m.tool_call_id.empty() &&
             messages[i + 1].tool_call_id == m.tool_call_id) {
        if (!content.empty() && !messages[i + 1].content.empty())
          content += '\n';
        content += messages[++i].content;
      }
      if (content.find("<tool_response>") != std::string::npos)
        out.Literal(content);
      else {
        out.text += "<tool_response>\n";
        out.Literal(content);
        out.text += "\n</tool_response>";
      }
    } else {
      std::size_t cursor = 0;
      for (const auto& image : m.images) {
        if (image.offset < cursor || image.offset > m.content.size() ||
            image_index >= images.size())
          throw std::invalid_argument("invalid LFM3-VL image placement");
        out.Literal(
            std::string_view(m.content).substr(cursor, image.offset - cursor));
        const auto& expansion = images[image_index++];
        if (!expansion.count || expansion.count > 16384 ||
            expansion.token.empty())
          throw std::invalid_argument("invalid LFM3-VL image expansion");
        out.text += tokenization::kVisionStart;
        for (std::size_t j = 0; j < expansion.count; ++j)
          out.text += expansion.token;
        out.text += tokenization::kVisionEnd;
        cursor = image.offset;
      }
      out.Literal(std::string_view(m.content).substr(cursor));
      out.Literal(m.framing_suffix);
    }
    out.text += tokenization::kImEnd;
    out.text += '\n';
  }
  if (image_index != images.size())
    throw std::invalid_argument("LFM3-VL image expansion count mismatch");
  if (generation) {
    out.text += tokenization::kImStart;
    out.text += "assistant\n";
    out.text += ThinkingEnabled(reasoning) ? "<think>" : "<think></think>";
  }
  return out;
}
}  // namespace
bool ThinkingEnabled(const ReasoningOptions& reasoning) {
  return reasoning.enabled.value_or(true);
}
std::string RenderChat(std::span<const tokenization::ChatMessage> messages,
                       std::span<const tokenization::ChatTool> tools,
                       const ReasoningOptions& reasoning, bool generation,
                       std::span<const ImageExpansion> images) {
  return Render(messages, tools, reasoning, generation, images).text;
}
std::vector<std::uint32_t> EncodeChat(
    const Tokenizer& tokenizer,
    std::span<const tokenization::ChatMessage> messages,
    std::span<const tokenization::ChatTool> tools,
    const ReasoningOptions& reasoning, bool generation,
    std::span<const ImageExpansion> images) {
  const auto rendered = Render(messages, tools, reasoning, generation, images);
  return tokenizer.EncodeFramed(rendered.text, rendered.literals);
}
}  // namespace gufo::models::lfm3
