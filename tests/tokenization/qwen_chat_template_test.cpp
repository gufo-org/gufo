#include "src/models/qwen/qwen_chat_template.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/qwen_tokenizer.hpp"

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
    AppendPod(static_cast<std::uint32_t>(strix::core::GgufValueType::kString));
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
  auto tpl = strix::tokenization::QwenChatTemplate::CreateDefault();
  Expect(tpl != nullptr, "CreateDefault succeeds");

  std::vector<strix::tokenization::ChatMessage> messages = {
      {strix::tokenization::ChatRole::kSystem, "You are a concise assistant.",
       "", ""},
      {strix::tokenization::ChatRole::kUser, "What is 2+2?", "", ""},
  };

  strix::tokenization::ChatTemplateOptions opts;
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
  auto tpl = strix::tokenization::QwenChatTemplate::CreateDefault();

  std::vector<strix::tokenization::ChatMessage> messages = {
      {strix::tokenization::ChatRole::kUser, "Solve this equation.", "", ""},
      {strix::tokenization::ChatRole::kAssistant, "The roots are 2 and 3.", "",
       "First let's factor the polynomial."},
  };

  strix::tokenization::ChatTemplateOptions opts;
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

void TestBoundedOutputLimit() {
  auto tpl = strix::tokenization::QwenChatTemplate::CreateDefault();

  std::vector<strix::tokenization::ChatMessage> messages = {
      {strix::tokenization::ChatRole::kUser, std::string(2000, 'A'), "", ""},
  };

  strix::tokenization::ChatTemplateOptions opts;
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
      strix::core::GgufReader::OpenMemory(binary.data(), binary.size(), &err);
  Expect(reader != nullptr, "Reader open succeeds");

  auto tpl =
      strix::tokenization::QwenChatTemplate::CreateFromGguf(*reader, &err);
  Expect(tpl != nullptr, "Template extracted from GGUF");
  Expect(tpl->GetTemplateString().find("{% for m in messages %}") !=
             std::string_view::npos,
         "Template string matches GGUF header");
}

void TestRenderAndTokenize() {
  auto tpl = strix::tokenization::QwenChatTemplate::CreateDefault();

  std::vector<std::string> vocab;
  for (int i = 0; i < 256; ++i) {
    vocab.emplace_back(1, static_cast<char>(i));
  }
  vocab.emplace_back("<|im_start|>");
  vocab.emplace_back("<|im_end|>");
  vocab.emplace_back("<think>");
  vocab.emplace_back("</think>");

  std::unordered_map<std::string, strix::tokenization::TokenId> specials = {
      {"<|im_start|>", 256},
      {"<|im_end|>", 257},
      {"<think>", 258},
      {"</think>", 259},
  };

  std::string err;
  auto tokenizer = strix::tokenization::QwenTokenizer::CreateFromVocabulary(
      vocab, {}, specials, &err);
  Expect(tokenizer != nullptr, "Tokenizer initialized");

  std::vector<strix::tokenization::ChatMessage> messages = {
      {strix::tokenization::ChatRole::kUser, "Hello", "", ""},
  };

  strix::tokenization::ChatTemplateOptions opts;
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
  auto tpl = strix::tokenization::QwenChatTemplate::CreateDefault();

  // Test Case 1: single turn user
  {
    std::vector<strix::tokenization::ChatMessage> msgs = {
        {strix::tokenization::ChatRole::kUser, "Hello, Strix Halo!", "", ""}};
    strix::tokenization::ChatTemplateOptions opts;
    opts.add_generation_prompt = true;
    auto res = tpl->Render(msgs, opts);
    Expect(res.has_value() && *res ==
                                  "<|im_start|>user\nHello, Strix "
                                  "Halo!<|im_end|>\n<|im_start|>assistant\n",
           "Case 1 single_turn_user matches golden");
  }

  // Test Case 2: tool message
  {
    std::vector<strix::tokenization::ChatMessage> msgs = {
        {strix::tokenization::ChatRole::kUser, "Fetch weather.", "", ""},
        {strix::tokenization::ChatRole::kTool,
         "{\"temp\": 22, \"city\": \"Rome\"}", "", ""}};
    strix::tokenization::ChatTemplateOptions opts;
    opts.add_generation_prompt = true;
    auto res = tpl->Render(msgs, opts);
    Expect(res.has_value() &&
               *res ==
                   "<|im_start|>user\nFetch "
                   "weather.<|im_end|>\n<|im_start|>tool\n{\"temp\": 22, "
                   "\"city\": \"Rome\"}<|im_end|>\n<|im_start|>assistant\n",
           "Case 2 tool_message matches golden");
  }

  // Test Case 3: Unicode CJK
  {
    std::vector<strix::tokenization::ChatMessage> msgs = {
        {strix::tokenization::ChatRole::kUser, "你好，世界！🚀", "", ""}};
    strix::tokenization::ChatTemplateOptions opts;
    opts.add_generation_prompt = true;
    auto res = tpl->Render(msgs, opts);
    Expect(res.has_value() && *res ==
                                  "<|im_start|>user\n你好，世界！🚀<|im_end|>"
                                  "\n<|im_start|>assistant\n",
           "Case 3 unicode_cjk matches golden");
  }
}

}  // namespace

int main() {
  std::cout << "Running QwenChatTemplate unit tests...\n";
  TestBasicChatRendering();
  TestThinkingFraming();
  TestBoundedOutputLimit();
  TestGgufTemplateExtraction();
  TestRenderAndTokenize();
  TestChatCorpusConformance();
  std::cout << "All QwenChatTemplate tests passed successfully!\n";
  return 0;
}
