#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/json.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "tests/models/chat_template_golden_helpers.hpp"

namespace {

namespace json = gufo::server::json;
namespace qwen = gufo::tokenization;

using namespace gufo::testing::chat_goldens;

void CheckQwenCase(const json::Value& fixture, std::string_view case_name,
                   const qwen::QwenTokenizer& tokenizer,
                   std::span<const qwen::ChatMessage> messages,
                   std::span<const qwen::ChatTool> tools,
                   const qwen::ChatTemplateOptions& options) {
  std::string error;
  const auto rendered =
      qwen::QwenChatTemplate::Render(messages, tools, options, &error);
  Expect(rendered.has_value(), "Qwen golden prompt renders: " + error);
  const auto tokens = qwen::QwenChatTemplate::RenderAndTokenize(
      tokenizer, messages, tools, options, &error);
  Expect(tokens.has_value(), "Qwen golden prompt tokenizes: " + error);

  const json::Value& expected = GoldenCase(fixture, "qwen", case_name);
  Expect(Sha256(*rendered) == expected.member_str("rendered_sha256"),
         "Qwen rendered bytes match Hugging Face");
  if (tokens->size() != expected.member_size("token_count")) {
    std::cerr << "Qwen case " << case_name << ": expected "
              << expected.member_size("token_count") << " tokens, got "
              << tokens->size() << "; token SHA-256 "
              << TokenSha256<qwen::TokenId>(*tokens) << '\n';
  }
  Expect(tokens->size() == expected.member_size("token_count"),
         "Qwen token count matches Hugging Face");
  Expect(TokenSha256<qwen::TokenId>(*tokens) ==
             expected.member_str("token_ids_le_u32_sha256"),
         "Qwen token IDs match Hugging Face");
}

void TestQwenGoldens(const json::Value& fixture,
                     const std::string& model_path) {
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  Expect(reader != nullptr, "Qwen GGUF opens: " + error);
  Expect(reader->GetMetadataString("tokenizer.ggml.pre") ==
             std::optional<std::string_view>{"qwen35"},
         "Qwen GGUF declares the qwen35 pre-tokenizer");
  Expect(qwen::QwenChatTemplate::ValidateGgufTemplate(*reader, &error),
         "Qwen GGUF template is recognized: " + error);
  const auto tokenizer = qwen::QwenTokenizer::CreateFromGguf(*reader, &error);
  Expect(tokenizer != nullptr, "Qwen tokenizer loads: " + error);

  const std::vector<qwen::ChatMessage> base = {
      {qwen::ChatRole::kUser, "Name one color.", "", ""},
  };
  const std::vector<qwen::ChatMessage> history = {
      {qwen::ChatRole::kUser, "Name one color.", "", ""},
      {qwen::ChatRole::kAssistant, "Red", "",
       "I should answer with one color."},
      {qwen::ChatRole::kUser, "Name another.", "", ""},
  };
  const std::vector<qwen::ChatTool> tools = {{
      .name = "get_weather",
      .description = "Get weather",
      .parameters_json = "{\"type\":\"object\",\"properties\":{\"city\":{"
                         "\"type\":\"string\"}},"
                         "\"required\":[\"city\"]}",
  }};

  qwen::ChatTemplateOptions options;
  options.enable_thinking = false;
  CheckQwenCase(fixture, "chat", *tokenizer, base, {}, options);

  options.enable_thinking = true;
  options.reasoning_effort = qwen::QwenReasoningEffort::kLow;
  CheckQwenCase(fixture, "thinking_low", *tokenizer, base, {}, options);
  options.reasoning_effort = qwen::QwenReasoningEffort::kMedium;
  CheckQwenCase(fixture, "thinking_medium", *tokenizer, base, {}, options);
  options.reasoning_effort = qwen::QwenReasoningEffort::kXHigh;
  CheckQwenCase(fixture, "thinking_xhigh", *tokenizer, base, {}, options);

  options.reasoning_effort = qwen::QwenReasoningEffort::kMedium;
  options.preserve_thinking = false;
  CheckQwenCase(fixture, "history_drop", *tokenizer, history, {}, options);
  options.enable_thinking = false;
  options.preserve_thinking = true;
  CheckQwenCase(fixture, "history_preserve", *tokenizer, history, {}, options);

  options = {};
  CheckQwenCase(fixture, "tools", *tokenizer, base, tools, options);
}

}  // namespace

int main() {
  const char* qwen_model = std::getenv("GUFO_QWEN_GGUF");
  if (qwen_model == nullptr || qwen_model[0] == '\0') {
    std::cout << "SKIP: set GUFO_QWEN_GGUF\n";
    return 77;
  }

  std::ifstream input(GUFO_CHAT_TEMPLATE_HF_GOLDENS, std::ios::binary);
  Expect(input.good(), "Hugging Face golden fixture opens");
  const std::string fixture_text{std::istreambuf_iterator<char>(input),
                                 std::istreambuf_iterator<char>()};
  const json::Value fixture = json::parse(fixture_text);

  TestQwenGoldens(fixture, qwen_model);
  std::cout << "Hugging Face chat-template token goldens passed\n";
  return 0;
}
