#include <fstream>
#include <iterator>

#include "src/models/deepseek_v4_flash/engine.hpp"
#include "tests/models/chat_template_golden_helpers.hpp"
namespace gufo::testing::ds4 {
namespace ds4 = gufo::models::deepseek_v4_flash;
using namespace gufo::testing::chat_goldens;
void CheckDeepSeekCase(const json::Value& fixture, std::string_view case_name,
                       const ds4::Model& model,
                       std::span<const ds4::ChatMessage> messages,
                       std::span<const ds4::ChatTool> tools,
                       const ds4::ChatTemplateOptions& options) {
  const std::string rendered = ds4::RenderChat(messages, tools, options);
  const std::vector<int> tokens = model.EncodeChat(messages, tools, options);

  const json::Value& expected = GoldenCase(fixture, "deepseek", case_name);
  Expect(Sha256(rendered) == expected.member_str("rendered_sha256"),
         "DeepSeek rendered bytes match Hugging Face");
  Expect(tokens.size() == expected.member_size("token_count"),
         "DeepSeek token count matches Hugging Face");
  Expect(TokenSha256<int>(tokens) ==
             expected.member_str("token_ids_le_u32_sha256"),
         "DeepSeek token IDs match Hugging Face");
}

void CheckTokenGoldens(const ds4::Model& model) {
  std::ifstream input(GUFO_CHAT_TEMPLATE_HF_GOLDENS, std::ios::binary);
  Expect(input.good(), "Hugging Face golden fixture opens");
  const std::string text{std::istreambuf_iterator<char>(input), {}};
  const auto fixture = json::parse(text);
  const std::vector<ds4::ChatMessage> base = {
      {.role = "system",
       .content = "Be concise.",
       .reasoning_content = {},
       .tool_calls = {}},
      {.role = "user",
       .content = "Name one color.",
       .reasoning_content = {},
       .tool_calls = {}},
  };
  const std::vector<ds4::ChatMessage> history = {
      {.role = "user",
       .content = "Name one color.",
       .reasoning_content = {},
       .tool_calls = {}},
      {.role = "assistant",
       .content = "Red",
       .reasoning_content = "I should answer with one color.",
       .tool_calls = {}},
      {.role = "user",
       .content = "Name another.",
       .reasoning_content = {},
       .tool_calls = {}},
  };

  ds4::ChatTemplateOptions options;
  CheckDeepSeekCase(fixture, "chat", model, base, {}, options);
  options.enable_thinking = true;
  options.reasoning_effort = gufo::ReasoningEffort::kLow;
  CheckDeepSeekCase(fixture, "thinking_low", model, base, {}, options);
  options.reasoning_effort = gufo::ReasoningEffort::kHigh;
  CheckDeepSeekCase(fixture, "thinking_high", model, base, {}, options);
  options.reasoning_effort = gufo::ReasoningEffort::kMax;
  CheckDeepSeekCase(fixture, "thinking_max", model, base, {}, options);

  options.reasoning_effort = gufo::ReasoningEffort::kLow;
  options.preserve_thinking = false;
  CheckDeepSeekCase(fixture, "history_drop", model, history, {}, options);
  options.preserve_thinking = true;
  CheckDeepSeekCase(fixture, "history_preserve", model, history, {}, options);

  ds4::ChatMessage assistant{
      .role = "assistant",
      .content = "",
      .reasoning_content = "Use the tool.",
      .tool_calls = {},
  };
  assistant.tool_calls.push_back({
      .name = "get_weather",
      .arguments = {{.name = "city", .value = "Rome", .is_string = true}},
  });
  const std::vector<ds4::ChatMessage> tool_messages = {
      {.role = "system",
       .content = "Be concise.",
       .reasoning_content = {},
       .tool_calls = {}},
      {.role = "user",
       .content = "Weather in Rome?",
       .reasoning_content = {},
       .tool_calls = {}},
      assistant,
      {.role = "tool",
       .content = "{\"temperature\":21}",
       .reasoning_content = {},
       .tool_calls = {}},
      {.role = "user",
       .content = "Summarize.",
       .reasoning_content = {},
       .tool_calls = {}},
  };
  const std::vector<ds4::ChatTool> tools = {{
      .name = "get_weather",
      .description = "Get weather",
      .parameters_json = "{\"type\":\"object\",\"properties\":{\"city\":{"
                         "\"type\":\"string\"}},"
                         "\"required\":[\"city\"]}",
  }};
  options = {};
  options.enable_thinking = true;
  CheckDeepSeekCase(fixture, "tools", model, tool_messages, tools, options);
}

}  // namespace gufo::testing::ds4
