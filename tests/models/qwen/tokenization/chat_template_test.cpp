#include "src/models/qwen/chat_template.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {

void Expect(bool condition, std::string_view msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

// Simple in-memory GGUF builder for template tests
class GgufTemplateBuilder {
public:
  GgufTemplateBuilder() {
    const char magic[4] = {'G', 'G', 'U', 'F'};
    AppendBytes(magic, 4);
    AppendPod(static_cast<std::uint32_t>(3));
    tensor_count_pos_ = buffer_.size();
    AppendPod(static_cast<std::uint64_t>(0));
    metadata_count_pos_ = buffer_.size();
    AppendPod(static_cast<std::uint64_t>(0));
  }

  void AddMetadataString(std::string_view key, std::string_view val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kString));
    AppendString(val);
    metadata_count_++;
  }

  std::vector<std::uint8_t> Build() {
    std::memcpy(buffer_.data() + metadata_count_pos_, &metadata_count_,
                sizeof(metadata_count_));
    std::size_t rem = buffer_.size() % 32;
    if (rem != 0) {
      buffer_.resize(buffer_.size() + (32 - rem), 0);
    }
    return buffer_;
  }

private:
  void AppendBytes(const void* data, std::size_t len) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    buffer_.insert(buffer_.end(), p, p + len);
  }

  template<typename T>
  void AppendPod(T val) {
    AppendBytes(&val, sizeof(T));
  }

  void AppendString(std::string_view s) {
    std::uint64_t len = s.size();
    AppendPod(len);
    AppendBytes(s.data(), len);
  }

  std::vector<std::uint8_t> buffer_;
  std::size_t tensor_count_pos_{0};
  std::size_t metadata_count_pos_{0};
  std::uint64_t metadata_count_{0};
};

void TestBasicChatRendering() {
  auto tpl = gufo::tokenization::QwenChatTemplate::CreateDefault();
  Expect(tpl != nullptr, "CreateDefault succeeds");

  std::vector<gufo::tokenization::ChatMessage> messages = {
      {gufo::tokenization::ChatRole::kSystem, "You are a concise assistant.",
       "", ""},
      {gufo::tokenization::ChatRole::kUser, "What is 2+2?", "", ""},
  };

  gufo::tokenization::ChatTemplateOptions opts;
  opts.add_generation_prompt = true;
  opts.enable_thinking = false;

  std::string err;
  auto rendered = tpl->Render(messages, opts, &err);
  Expect(rendered.has_value(), "Render succeeds: " + err);

  const std::string expected =
      "<|im_start|>system\nYou are a concise assistant.<|im_end|>\n"
      "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
      "<|im_start|>assistant\n";

  Expect(*rendered == expected, "Rendered output matches ChatML golden");
}

void TestThinkingFraming() {
  auto tpl = gufo::tokenization::QwenChatTemplate::CreateDefault();

  std::vector<gufo::tokenization::ChatMessage> messages = {
      {gufo::tokenization::ChatRole::kUser, "Solve this equation.", "", ""},
      {gufo::tokenization::ChatRole::kAssistant, "The roots are 2 and 3.", "",
       "First let's factor the polynomial."},
  };

  gufo::tokenization::ChatTemplateOptions opts;
  opts.add_generation_prompt = true;
  opts.enable_thinking = true;

  auto rendered = tpl->Render(messages, opts);
  Expect(rendered.has_value(), "Render with thinking succeeds");

  const std::string expected =
      "<|im_start|>user\nSolve this equation.<|im_end|>\n"
      "<|im_start|>assistant\n<think>\nFirst let's factor the "
      "polynomial.\n</think>\n"
      "The roots are 2 and 3.<|im_end|>\n"
      "<|im_start|>assistant\n<think>\n";

  Expect(*rendered == expected,
         "Rendered output includes thinking blocks and prompt");
}

void TestHistoricalThinkingDoesNotEnableNewThinking() {
  auto tpl = gufo::tokenization::QwenChatTemplate::CreateDefault();
  const std::vector<gufo::tokenization::ChatMessage> messages = {
      {gufo::tokenization::ChatRole::kUser, "Name one color.", "", ""},
      {gufo::tokenization::ChatRole::kAssistant, "\nRed", "",
       "I should answer with one color."},
      {gufo::tokenization::ChatRole::kUser, "Name another.", "", ""},
  };

  gufo::tokenization::ChatTemplateOptions opts;
  opts.add_generation_prompt = true;
  opts.enable_thinking = false;
  const auto rendered = tpl->Render(messages, opts);
  Expect(rendered.has_value(), "Historical thinking renders");
  Expect(*rendered ==
             "<|im_start|>user\nName one color.<|im_end|>\n"
             "<|im_start|>assistant\n<think>\nI should answer with one "
             "color.\n</think>\n\nRed<|im_end|>\n"
             "<|im_start|>user\nName another.<|im_end|>\n"
             "<|im_start|>assistant\n",
         "Historical reasoning round-trips without changing the next prompt");
}

void TestBoundedOutputLimit() {
  auto tpl = gufo::tokenization::QwenChatTemplate::CreateDefault();

  std::vector<gufo::tokenization::ChatMessage> messages = {
      {gufo::tokenization::ChatRole::kUser, std::string(2000, 'A'), "", ""},
  };

  gufo::tokenization::ChatTemplateOptions opts;
  opts.max_output_bytes = 100;  // Intentionally tiny limit

  std::string err;
  auto rendered = tpl->Render(messages, opts, &err);
  Expect(!rendered.has_value(),
         "Render fails closed when exceeding max_output_bytes");
  Expect(err.find("exceeds maximum bound") != std::string::npos,
         "Error reports bound exceeded");
}

void TestGgufTemplateExtraction() {
  GgufTemplateBuilder builder;
  builder.AddMetadataString(
      "tokenizer.chat_template",
      "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}");

  auto binary = builder.Build();

  std::string err;
  auto reader =
      gufo::core::GgufReader::OpenMemory(binary.data(), binary.size(), &err);
  Expect(reader != nullptr, "Reader open succeeds");

  auto tpl =
      gufo::tokenization::QwenChatTemplate::CreateFromGguf(*reader, &err);
  Expect(tpl != nullptr, "Template extracted from GGUF");
  Expect(tpl->GetTemplateString().find("{% for m in messages %}") !=
             std::string_view::npos,
         "Template string matches GGUF header");
}

void TestRenderAndTokenize() {
  auto tpl = gufo::tokenization::QwenChatTemplate::CreateDefault();

  std::vector<std::string> vocab;
  for (int i = 0; i < 256; ++i) {
    vocab.emplace_back(1, static_cast<char>(i));
  }
  vocab.emplace_back("<|im_start|>");
  vocab.emplace_back("<|im_end|>");
  vocab.emplace_back("<think>");
  vocab.emplace_back("</think>");

  std::unordered_map<std::string, gufo::tokenization::TokenId> specials = {
      {"<|im_start|>", 256},
      {"<|im_end|>", 257},
      {"<think>", 258},
      {"</think>", 259},
  };

  std::string err;
  auto tokenizer = gufo::tokenization::QwenTokenizer::CreateFromVocabulary(
      vocab, {}, specials, &err);
  Expect(tokenizer != nullptr, "Tokenizer initialized");

  std::vector<gufo::tokenization::ChatMessage> messages = {
      {gufo::tokenization::ChatRole::kUser, "Hello", "", ""},
  };

  gufo::tokenization::ChatTemplateOptions opts;
  opts.add_generation_prompt = true;

  auto token_ids = tpl->RenderAndTokenize(*tokenizer, messages, opts, &err);
  Expect(token_ids.has_value(), "RenderAndTokenize succeeds: " + err);
  Expect(!token_ids->empty(), "Token IDs not empty");

  auto decoded = tokenizer->Decode(*token_ids);
  const std::string expected =
      "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
  Expect(decoded == expected, "Decoded tokens match rendered prompt exactly");
}

void TestChatCorpusConformance() {
  auto tpl = gufo::tokenization::QwenChatTemplate::CreateDefault();

  // Test Case 1: single turn user
  {
    std::vector<gufo::tokenization::ChatMessage> msgs = {
        {gufo::tokenization::ChatRole::kUser, "Hello, Strix Halo!", "", ""}};
    gufo::tokenization::ChatTemplateOptions opts;
    opts.add_generation_prompt = true;
    auto res = tpl->Render(msgs, opts);
    Expect(res.has_value() && *res ==
                                  "<|im_start|>user\nHello, Strix "
                                  "Halo!<|im_end|>\n<|im_start|>assistant\n",
           "Case 1 single_turn_user matches golden");
  }

  // Test Case 2: tool message
  {
    std::vector<gufo::tokenization::ChatMessage> msgs = {
        {gufo::tokenization::ChatRole::kUser, "Fetch weather.", "", ""},
        {gufo::tokenization::ChatRole::kTool,
         "{\"temp\": 22, \"city\": \"Rome\"}", "", ""}};
    gufo::tokenization::ChatTemplateOptions opts;
    opts.add_generation_prompt = true;
    auto res = tpl->Render(msgs, opts);
    Expect(res.has_value() &&
               *res ==
                   "<|im_start|>user\nFetch "
                   "weather.<|im_end|>\n<|im_start|>user\n<tool_response>\n"
                   "{\"temp\": 22, \"city\": \"Rome\"}\n</tool_response>"
                   "<|im_end|>\n<|im_start|>assistant\n",
           "Case 2 tool_message matches golden");
  }

  // Test Case 3: Unicode CJK
  {
    std::vector<gufo::tokenization::ChatMessage> msgs = {
        {gufo::tokenization::ChatRole::kUser, "你好，世界！🚀", "", ""}};
    gufo::tokenization::ChatTemplateOptions opts;
    opts.add_generation_prompt = true;
    auto res = tpl->Render(msgs, opts);
    Expect(res.has_value() && *res ==
                                  "<|im_start|>user\n你好，世界！🚀<|im_end|>"
                                  "\n<|im_start|>assistant\n",
           "Case 3 unicode_cjk matches golden");
  }
}

void TestToolRendering() {
  auto tpl = gufo::tokenization::QwenChatTemplate::CreateDefault();

  gufo::tokenization::ChatMessage assistant{
      gufo::tokenization::ChatRole::kAssistant, "", "", ""};
  assistant.tool_calls.push_back({
      .id = "call_weather",
      .name = "get_weather",
      .arguments =
          {
              {
                  .name = "city",
                  .value = "Rome",
                  .is_string = true,
              },
          },
  });
  const std::vector<gufo::tokenization::ChatMessage> messages = {
      {gufo::tokenization::ChatRole::kSystem, "Be concise.", "", ""},
      {gufo::tokenization::ChatRole::kUser, "What is the weather?", "", ""},
      std::move(assistant),
  };
  const std::vector<gufo::tokenization::ChatTool> tools = {
      {
          .name = "get_weather",
          .description = "Return current weather",
          .parameters_json =
              R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})",
      },
  };

  gufo::tokenization::ChatTemplateOptions options;
  options.require_tool_call = true;
  const auto rendered = tpl->Render(messages, tools, options);
  Expect(rendered.has_value(), "Tool-aware render succeeds");
  Expect(rendered->find("# Tools") != std::string::npos,
         "Tool prompt is rendered");
  Expect(rendered->find(R"("name":"get_weather")") != std::string::npos,
         "Tool schema is rendered");
  Expect(rendered->find("<function=get_weather>") != std::string::npos,
         "Assistant tool call uses native Qwen syntax");
  Expect(rendered->find("<parameter=city>\nRome\n</parameter>") !=
             std::string::npos,
         "Tool arguments use native Qwen syntax");
  Expect(rendered->find("You must call at least one available function") !=
             std::string::npos,
         "Required tool choice is included in the model prompt");
}

void TestToolReplayPreservesGeneratedPrefix() {
  auto tpl = gufo::tokenization::QwenChatTemplate::CreateDefault();

  constexpr std::string_view kGeneratedText =
      "<think>\nI should read the requested file.\n</think>\n\n";
  gufo::tokenization::ChatMessage assistant{
      gufo::tokenization::ChatRole::kAssistant, std::string(kGeneratedText), "",
      ""};
  assistant.tool_calls.push_back({
      .id = "call_read",
      .name = "read",
      .arguments =
          {
              {
                  .name = "path",
                  .value = "/etc/hostname",
                  .is_string = true,
              },
          },
  });

  const std::vector<gufo::tokenization::ChatMessage> messages = {
      std::move(assistant),
  };
  gufo::tokenization::ChatTemplateOptions options;
  options.add_generation_prompt = false;

  const auto rendered = tpl->Render(messages, options);
  Expect(rendered.has_value(), "Assistant tool replay renders");

  const std::string generated =
      std::string(kGeneratedText) +
      "<tool_call>\n<function=read>\n<parameter=path>\n/etc/hostname\n"
      "</parameter>\n</function>\n</tool_call>";
  const std::string expected =
      "<|im_start|>assistant\n" + generated + "<|im_end|>\n";
  Expect(*rendered == expected,
         "Structured tool replay is byte-identical to generated syntax");
}

}  // namespace

int main() {
  std::cout << "Running QwenChatTemplate unit tests...\n";
  TestBasicChatRendering();
  TestThinkingFraming();
  TestHistoricalThinkingDoesNotEnableNewThinking();
  TestBoundedOutputLimit();
  TestGgufTemplateExtraction();
  TestRenderAndTokenize();
  TestChatCorpusConformance();
  TestToolRendering();
  TestToolReplayPreservesGeneratedPrefix();
  std::cout << "All QwenChatTemplate tests passed successfully!\n";
  return 0;
}
