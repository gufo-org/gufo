#include "src/models/qwen/chat_template.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "tests/models/chat_template_golden_helpers.hpp"

namespace {

void Expect(bool condition, std::string_view msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

std::string Sha256(std::string_view value) {
  const auto* data = reinterpret_cast<const std::uint8_t*>(value.data());
  return gufo::crypto::Sha256Hex({data, value.size()});
}

std::string ReadText(std::string_view path) {
  std::ifstream input(std::string(path), std::ios::binary);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
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
      "<|im_start|>assistant\n<think>\n\n</think>\n\n";

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
  opts.reasoning_effort = gufo::tokenization::QwenReasoningEffort::kMedium;

  auto rendered = tpl->Render(messages, opts);
  Expect(rendered.has_value(), "Render with thinking succeeds");

  const std::string expected =
      "<|im_start|>user\nSolve this equation.<|im_end|>\n"
      "<|im_start|>assistant\n<think>\nFirst let's factor the "
      "polynomial.\n</think>\n\n"
      "The roots are 2 and 3.<|im_end|>\n"
      "<|im_start|>assistant\n<think>\n";

  Expect(*rendered == expected,
         "Rendered output includes thinking blocks and prompt");
}

void TestHistoricalThinkingDoesNotEnableNewThinking() {
  auto tpl = gufo::tokenization::QwenChatTemplate::CreateDefault();
  const std::vector<gufo::tokenization::ChatMessage> messages = {
      {gufo::tokenization::ChatRole::kUser, "Name one color.", "", ""},
      {gufo::tokenization::ChatRole::kAssistant, "Red", "",
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
             "<|im_start|>assistant\n<think>\n\n</think>\n\n",
         "Historical reasoning round-trips without changing the next prompt");
}

void TestReasoningEffortAndPreservation() {
  const std::vector<gufo::tokenization::ChatMessage> messages = {
      {gufo::tokenization::ChatRole::kSystem, "System rules.", "", ""},
      {gufo::tokenization::ChatRole::kDeveloper, "Developer rules.", "", ""},
      {gufo::tokenization::ChatRole::kUser, "First question.", "", ""},
      {gufo::tokenization::ChatRole::kAssistant, "First answer.", "",
       "Old private reasoning."},
      {gufo::tokenization::ChatRole::kUser, "Second question.", "", ""},
  };

  gufo::tokenization::ChatTemplateOptions options;
  options.enable_thinking = true;
  options.reasoning_effort = gufo::tokenization::QwenReasoningEffort::kLow;
  options.preserve_thinking = false;
  const auto rendered =
      gufo::tokenization::QwenChatTemplate::Render(messages, options);
  Expect(rendered.has_value(), "Reasoning options render");
  Expect(rendered->find("Reasoning effort is set to low.") != std::string::npos,
         "Low effort instruction is compiled into the system message");
  Expect(rendered->find("System rules.\nDeveloper rules.") != std::string::npos,
         "Leading system and developer messages are merged");
  Expect(rendered->find("Old private reasoning.") == std::string::npos,
         "Historical reasoning is dropped when preservation is disabled");
  Expect(rendered->find("First answer.") != std::string::npos,
         "Visible historical assistant content is retained");
  Expect(rendered->ends_with("<|im_start|>assistant\n<think>\n"),
         "Thinking generation starts inside the reasoning block");
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

void TestRenderedPromptBoundFollowsContext() {
  using gufo::tokenization::ChatMessage;
  using gufo::tokenization::ChatRole;
  using gufo::tokenization::ChatTemplateOptions;
  using gufo::tokenization::QwenChatTemplate;
  using gufo::tokenization::RenderedPromptBoundBytes;
  constexpr std::size_t kMiB = 1024ULL * 1024ULL;
  Expect(RenderedPromptBoundBytes(0) == kMiB, "zero context keeps 1 MiB");
  Expect(RenderedPromptBoundBytes(4096) == kMiB,
         "small contexts keep the 1 MiB floor");
  Expect(RenderedPromptBoundBytes(262144) == 262144ULL * 128ULL,
         "native Flash-Next context scales the bound");
  Expect(RenderedPromptBoundBytes(409600) == 409600ULL * 128ULL,
         "extended context scales the bound");
  // gufo #285: ~1.05 MB rendered, far below a 262,144-token context.
  const std::vector<ChatMessage> messages = {
      {ChatRole::kUser, std::string(1100000, 'a'), "", ""}};
  ChatTemplateOptions options;
  Expect(!QwenChatTemplate::Render(messages, options).has_value(),
         "the default 1 MiB bound still refuses");
  options.max_output_bytes = RenderedPromptBoundBytes(262144);
  Expect(QwenChatTemplate::Render(messages, options).has_value(),
         "a context-sized bound admits a prompt the context can hold");
}

void TestGgufTemplateExtraction() {
  GgufTemplateBuilder builder;
  builder.AddMetadataString("general.name", "Qwen3.8-27B");
  std::string reference = ReadText(GUFO_QWEN38_CHAT_TEMPLATE_REFERENCE);
  Expect(!reference.empty(), "Pinned Qwen template reference is readable");
  if (reference.ends_with('\n')) {
    reference.pop_back();
  }
  builder.AddMetadataString("tokenizer.chat_template", reference);

  auto binary = builder.Build();

  std::string err;
  auto reader =
      gufo::core::GgufReader::OpenMemory(binary.data(), binary.size(), &err);
  Expect(reader != nullptr, "Reader open succeeds");

  auto tpl =
      gufo::tokenization::QwenChatTemplate::CreateFromGguf(*reader, &err);
  Expect(tpl != nullptr, "Template extracted from GGUF");
  Expect(tpl->GetTemplateString().find("preserve_thinking") !=
             std::string_view::npos,
         "Template string matches GGUF header");
  Expect(tpl->GetProfile() ==
             gufo::tokenization::QwenChatTemplate::Profile::kQwen38Reasoning,
         "Recognized Qwen3.8 template profile is classified");
  Expect(tpl->GetTemplateId() == "qwen38-reasoning-compiled-v3",
         "Compiled template version is stable");
  Expect(tpl->GetTemplateSha256().size() == 64,
         "Embedded template provenance is hashed");
  Expect(tpl->GetTemplateSha256() ==
             gufo::tokenization::QwenChatTemplate::OfficialTemplateSha256(),
         "Pinned upstream template hash is recognized exactly");

  GgufTemplateBuilder lookalike_builder;
  lookalike_builder.AddMetadataString("general.name", "Qwen3.8-27B");
  lookalike_builder.AddMetadataString(
      "tokenizer.chat_template",
      "{% set enable_thinking = true %}{% set preserve_thinking = true %}"
      "<|im_start|><think>");
  const auto lookalike_binary = lookalike_builder.Build();
  auto lookalike_reader = gufo::core::GgufReader::OpenMemory(
      lookalike_binary.data(), lookalike_binary.size(), &err);
  Expect(lookalike_reader != nullptr, "Look-alike GGUF opens");
  Expect(gufo::tokenization::QwenChatTemplate::CreateFromGguf(*lookalike_reader,
                                                              &err) == nullptr,
         "Qwen3.8 look-alike template is rejected by hash");
}

void TestHuggingFaceRenderedGoldens() {
  using gufo::tokenization::ChatMessage;
  using gufo::tokenization::ChatRole;
  using gufo::tokenization::ChatTemplateOptions;
  using gufo::tokenization::QwenChatTemplate;
  using gufo::tokenization::QwenReasoningEffort;
  using gufo::tokenization::ResolveQwenChatOptions;

  const std::vector<ChatMessage> base = {
      {ChatRole::kUser, "Name one color.", "", ""},
  };
  const std::vector<ChatMessage> history = {
      {ChatRole::kUser, "Name one color.", "", ""},
      {ChatRole::kAssistant, "Red", "", "I should answer with one color."},
      {ChatRole::kUser, "Name another.", "", ""},
  };
  std::ifstream input(GUFO_CHAT_TEMPLATE_HF_GOLDENS);
  Expect(input.good(), "Template fixture opens");
  const auto fixture = gufo::json::parse(std::string{
      std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()});
  const auto check =
      [&](const std::vector<ChatMessage>& messages,
          const ChatTemplateOptions& options, std::string_view name,
          std::span<const gufo::tokenization::ChatTool> tools = {}) {
        const auto rendered =
            QwenChatTemplate::Render(messages, tools, options);
        Expect(rendered.has_value(), "Qwen Hugging Face golden renders");
        const auto expected =
            gufo::testing::chat_goldens::GoldenCase(fixture, "qwen", name)
                .member_str("rendered_sha256");
        Expect(Sha256(*rendered) == expected,
               "Qwen rendered bytes match the pinned Hugging Face golden");
      };

  ChatTemplateOptions options;
  check(base, options, "thinking_xhigh");
  check(base, ResolveQwenChatOptions({}), "thinking_xhigh");
  options.enable_thinking = false;
  check(base, options, "chat");

  options.enable_thinking = true;
  options.reasoning_effort = QwenReasoningEffort::kLow;
  check(base, options, "thinking_low");
  options.reasoning_effort = QwenReasoningEffort::kMedium;
  check(base, options, "thinking_medium");
  options.reasoning_effort = QwenReasoningEffort::kXHigh;
  check(base, options, "thinking_xhigh");
  for (const auto effort :
       {gufo::ReasoningEffort::kLow, gufo::ReasoningEffort::kMedium,
        gufo::ReasoningEffort::kXHigh}) {
    const auto resolved = ResolveQwenChatOptions({.effort = effort});
    const auto expected = effort == gufo::ReasoningEffort::kLow
                              ? QwenReasoningEffort::kLow
                          : effort == gufo::ReasoningEffort::kMedium
                              ? QwenReasoningEffort::kMedium
                              : QwenReasoningEffort::kXHigh;
    options.reasoning_effort = expected;
    Expect(resolved.enable_thinking && resolved.preserve_thinking,
           "Effort selection retains official thinking/history defaults");
    Expect(QwenChatTemplate::Render(base, resolved) ==
               QwenChatTemplate::Render(base, options),
           "Shared CLI/server effort resolution matches the native template");
  }
  check(base,
        ResolveQwenChatOptions(
            {.enabled = false, .effort = gufo::ReasoningEffort::kLow}),
        "chat");

  options.reasoning_effort = QwenReasoningEffort::kMedium;
  options.preserve_thinking = false;
  check(history, options, "history_drop");
  options.enable_thinking = false;
  options.preserve_thinking = true;
  check(history, options, "history_preserve");
  const std::vector<gufo::tokenization::ChatTool> tools = {
      {.name = "get_weather",
       .description = "Get weather",
       .parameters_json =
           R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})"}};
  std::vector<ChatMessage> system_tools{{ChatRole::kSystem, "Be concise."},
                                        base[0]};
  check(system_tools, {}, "system_tools", tools);
  system_tools.insert(system_tools.begin() + 1,
                      {ChatRole::kDeveloper, "Use metric units."});
  check(system_tools, {}, "developer_tools", tools);

  auto image = std::make_shared<const std::vector<std::uint8_t>>(1, 0);
  std::vector<ChatMessage> vision{
      {ChatRole::kUser, "\u2003 before  after \u00a0"},
      {ChatRole::kAssistant, "Done."},
      {ChatRole::kUser, " \t \n"}};
  vision[0].images.push_back({std::string("\u2003 before ").size(), image});
  vision[2].images.push_back({2, image});
  options = {};
  options.enable_thinking = false;
  check(vision, options, "vision_whitespace");
  options.add_vision_id = true;
  check(vision, options, "vision_ids");
  std::vector<std::size_t> offsets;
  const auto rendered =
      QwenChatTemplate::Render(vision, {}, options, nullptr, &offsets);
  Expect(offsets.size() == 2, "Every image retains a placeholder offset");
  for (const auto offset : offsets)
    Expect(rendered->substr(offset, 13) == "<|image_pad|>",
           "Image offsets follow whitespace trimming and Picture prefixes");

  ChatMessage call{ChatRole::kAssistant, "", "", "Use the tool."};
  call.tool_calls.push_back({.name = "get_weather",
                             .arguments = {{.name = "city", .value = "Rome"}}});
  options = {};
  options.reasoning_effort = QwenReasoningEffort::kMedium;
  options.preserve_thinking = false;
  check(
      {base[0], call, {ChatRole::kUser, "<tool_response>21 C</tool_response>"}},
      options, "tool_loop_preserve");
  for (const auto role : {ChatRole::kSystem, ChatRole::kDeveloper}) {
    Expect(!QwenChatTemplate::Render(
               std::vector<ChatMessage>{base[0], {role, "Late instructions"}}),
           "Late system/developer messages are rejected");
  }
  for (const auto role : {ChatRole::kSystem, ChatRole::kDeveloper,
                          ChatRole::kAssistant, ChatRole::kTool}) {
    std::string error;
    Expect(!QwenChatTemplate::Render(
               std::vector<ChatMessage>{{role, "No user query"}}, {}, &error) &&
               error == "No user query found in messages.",
           "A system/assistant/tool-only transcript has no user query");
  }
  Expect(!QwenChatTemplate::Render(std::vector<ChatMessage>{
             {ChatRole::kUser, "<tool_response>result</tool_response>"}}),
         "A wrapped tool response is not a user query");
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

  opts.enable_thinking = false;
  auto token_ids = tpl->RenderAndTokenize(*tokenizer, messages, opts, &err);
  Expect(token_ids.has_value(), "RenderAndTokenize succeeds: " + err);
  Expect(!token_ids->empty(), "Token IDs not empty");

  auto decoded = tokenizer->Decode(*token_ids);
  const std::string expected =
      "<|im_start|>user\nHello<|im_end|>\n"
      "<|im_start|>assistant\n<think>\n\n</think>\n\n";
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
    opts.enable_thinking = false;
    auto res = tpl->Render(msgs, opts);
    Expect(res.has_value() && *res ==
                                  "<|im_start|>user\nHello, Strix "
                                  "Halo!<|im_end|>\n<|im_start|>assistant\n"
                                  "<think>\n\n</think>\n\n",
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
    opts.enable_thinking = false;
    auto res = tpl->Render(msgs, opts);
    Expect(res.has_value() &&
               *res ==
                   "<|im_start|>user\nFetch "
                   "weather.<|im_end|>\n<|im_start|>user\n<tool_response>\n"
                   "{\"temp\": 22, \"city\": \"Rome\"}\n</tool_response>"
                   "<|im_end|>\n<|im_start|>assistant\n"
                   "<think>\n\n</think>\n\n",
           "Case 2 tool_message matches golden");
  }

  // Test Case 3: Unicode CJK
  {
    std::vector<gufo::tokenization::ChatMessage> msgs = {
        {gufo::tokenization::ChatRole::kUser, "你好，世界！🚀", "", ""}};
    gufo::tokenization::ChatTemplateOptions opts;
    opts.add_generation_prompt = true;
    opts.enable_thinking = false;
    auto res = tpl->Render(msgs, opts);
    Expect(res.has_value() && *res ==
                                  "<|im_start|>user\n你好，世界！🚀<|im_end|>"
                                  "\n<|im_start|>assistant\n"
                                  "<think>\n\n</think>\n\n",
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
  Expect(rendered->find(R"("name": "get_weather")") != std::string::npos,
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
      gufo::tokenization::ChatRole::kAssistant, "", "",
      "I should read the requested file."};
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
      {gufo::tokenization::ChatRole::kUser, "Read the file."},
      std::move(assistant),
  };
  gufo::tokenization::ChatTemplateOptions options;
  options.add_generation_prompt = false;
  options.enable_thinking = false;

  const auto rendered = tpl->Render(messages, options);
  Expect(rendered.has_value(), "Assistant tool replay renders");

  const std::string generated =
      std::string(kGeneratedText) +
      "<tool_call>\n<function=read>\n<parameter=path>\n/etc/hostname\n"
      "</parameter>\n</function>\n</tool_call>";
  const std::string expected =
      "<|im_start|>user\nRead the file.<|im_end|>\n"
      "<|im_start|>assistant\n" +
      generated + "<|im_end|>\n";
  Expect(*rendered == expected,
         "Structured tool replay is byte-identical to generated syntax");
}

}  // namespace

int main() {
  std::cout << "Running QwenChatTemplate unit tests...\n";
  TestBasicChatRendering();
  TestThinkingFraming();
  TestHistoricalThinkingDoesNotEnableNewThinking();
  TestReasoningEffortAndPreservation();
  TestBoundedOutputLimit();
  TestRenderedPromptBoundFollowsContext();
  TestGgufTemplateExtraction();
  TestHuggingFaceRenderedGoldens();
  TestRenderAndTokenize();
  TestChatCorpusConformance();
  TestToolRendering();
  TestToolReplayPreservesGeneratedPrefix();
  std::cout << "All QwenChatTemplate tests passed successfully!\n";
  return 0;
}
