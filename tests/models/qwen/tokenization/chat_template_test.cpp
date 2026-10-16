#include "src/models/qwen/chat_template.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/control_tokens.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen/vision/prompt.hpp"
#include "tests/models/chat_template_golden_helpers.hpp"

using gufo::tokenization::kEndOfText;
using gufo::tokenization::kImagePad;
using gufo::tokenization::kImEnd;
using gufo::tokenization::kImStart;

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
    Expect(rendered->substr(offset, 13) == kImagePad,
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
    const std::vector<ChatMessage> late{{ChatRole::kSystem, "Be concise."},
                                        history[0],
                                        history[1],
                                        {role, "Late instructions"},
                                        history[2]};
    const std::vector<ChatMessage> leading{late[0], late[3], late[1], late[2],
                                           late[4]};
    const auto rendered = QwenChatTemplate::Render(late, tools, options);
    Expect(rendered &&
               rendered == QwenChatTemplate::Render(leading, tools, options) &&
               rendered->find("Be concise.\nLate instructions<|im_end|>") !=
                   std::string::npos,
           "Late system/developer messages join the leading system turn");
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
  vocab.emplace_back(kImStart);
  vocab.emplace_back(kImEnd);
  vocab.emplace_back("<think>");
  vocab.emplace_back("</think>");

  std::unordered_map<std::string, gufo::tokenization::TokenId> specials = {
      {std::string(kImStart), 256},
      {std::string(kImEnd), 257},
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

// Issue #383: message text is text. A client that sends the characters of a
// vocabulary token must not inject that control token into the prompt — an
// injected `<|im_end|>` is a turn end sitting in the middle of a conversation,
// which is what the model then imitates. The literal characters still reach
// the model, and the framing around them stays framing.
void TestContentSpellingATokenIsNotParsedAsOne() {
  auto tpl = gufo::tokenization::QwenChatTemplate::CreateDefault();

  std::vector<std::string> vocab;
  for (int i = 0; i < 256; ++i) {
    vocab.emplace_back(1, static_cast<char>(i));
  }
  vocab.emplace_back(kImStart);
  vocab.emplace_back(kImEnd);
  vocab.emplace_back("<think>");
  vocab.emplace_back("</think>");

  vocab.emplace_back(kEndOfText);

  std::unordered_map<std::string, gufo::tokenization::TokenId> specials = {
      {std::string(kImStart), 256},
      {std::string(kImEnd), 257},
      {"<think>", 258},
      {"</think>", 259},
      {std::string(kEndOfText), 260},
  };

  std::string err;
  auto tokenizer = gufo::tokenization::QwenTokenizer::CreateFromVocabulary(
      vocab, {}, specials, &err);
  Expect(tokenizer != nullptr, "Tokenizer initialized: " + err);

  const std::string text =
      "the vocabulary token `<|endoftext|>` is written here as prose, and the "
      "thinking phase ends with `</think>`";
  std::vector<gufo::tokenization::ChatMessage> messages = {
      {gufo::tokenization::ChatRole::kUser, text, "", ""},
  };

  gufo::tokenization::ChatTemplateOptions opts;
  opts.enable_thinking = false;

  const auto token_ids =
      tpl->RenderAndTokenize(*tokenizer, messages, opts, &err);
  Expect(token_ids.has_value(), "RenderAndTokenize succeeds: " + err);

  // The template never writes this one, so it can only appear if the message
  // text was read as a control token.
  const auto injected = tokenizer->FindSpecialToken(kEndOfText);
  Expect(injected.has_value(), "the message token is a vocabulary token");
  Expect(std::find(token_ids->begin(), token_ids->end(), *injected) ==
             token_ids->end(),
         "content spelling a control token does not inject it into the prompt");

  const auto framing = tokenizer->FindSpecialToken(kImStart);
  Expect(framing.has_value() && std::find(token_ids->begin(), token_ids->end(),
                                          *framing) != token_ids->end(),
         "template framing still tokenizes as control tokens");
  Expect(tokenizer->Decode(*token_ids).find(text) != std::string::npos,
         "the literal characters of the message reach the model");
}

void TestToolReplayArgumentsAreContent() {
  std::vector<std::string> vocab;
  for (int i = 0; i < 256; ++i) {
    vocab.emplace_back(1, static_cast<char>(i));
  }
  for (std::string_view token : std::initializer_list<std::string_view>{
           kImStart, kImEnd, "<tool_call>", "</tool_call>"}) {
    vocab.emplace_back(token);
  }
  std::unordered_map<std::string, gufo::tokenization::TokenId> specials = {
      {std::string(kImStart), 256},
      {std::string(kImEnd), 257},
      {"<tool_call>", 258},
      {"</tool_call>", 259},
  };
  std::string err;
  auto tokenizer = gufo::tokenization::QwenTokenizer::CreateFromVocabulary(
      vocab, {}, specials, &err);
  Expect(tokenizer != nullptr, "Tokenizer initialized: " + err);

  gufo::tokenization::ChatTemplateOptions opts;
  opts.add_generation_prompt = false;
  opts.enable_thinking = false;
  gufo::tokenization::TokenizerOptions tok_options;
  tok_options.add_bos = false;
  tok_options.add_eos = false;
  tok_options.parse_special_tokens = true;

  // Both string arguments and serialized JSON values are replayed data.
  for (const bool is_string : {true, false}) {
    const std::string value =
        is_string ? "EOS = \"<|im_end|>\""
                  : R"({"text":"<|im_end|><tool_call></tool_call>"})";
    gufo::tokenization::ChatMessage assistant{
        gufo::tokenization::ChatRole::kAssistant, "", "", ""};
    assistant.tool_calls.push_back({
        .id = "call_write",
        .name = "write",
        .arguments = {{.name = "content",
                       .value = value,
                       .is_string = is_string}},
    });
    const std::vector<gufo::tokenization::ChatMessage> messages = {
        {gufo::tokenization::ChatRole::kUser, "Write this file.", "", ""},
        std::move(assistant),
        {gufo::tokenization::ChatRole::kTool, value, "", ""},
    };
    std::vector<gufo::tokenization::ContentSpan> spans;
    const auto rendered = gufo::tokenization::QwenChatTemplate::Render(
        messages, {}, opts, &err, nullptr, nullptr, &spans);
    Expect(rendered.has_value(), "Tool replay renders: " + err);
    const auto argument_offset = rendered->find(value);
    const auto result_offset =
        rendered->find(value, argument_offset + value.size());
    Expect(argument_offset != std::string::npos &&
               result_offset != std::string::npos,
           "The argument and result retain their literal bytes");
    const auto argument = gufo::tokenization::QwenChatTemplate::EncodeRendered(
        *tokenizer, *rendered, argument_offset, argument_offset + value.size(),
        spans, tok_options);
    const auto result = gufo::tokenization::QwenChatTemplate::EncodeRendered(
        *tokenizer, *rendered, result_offset, result_offset + value.size(),
        spans, tok_options);
    auto text_options = tok_options;
    text_options.parse_special_tokens = false;
    Expect(argument == result &&
               argument == tokenizer->Encode(value, text_options),
           "Replayed argument token IDs match tool-result text, including "
           "literal EOS");

    const auto direct = gufo::tokenization::QwenChatTemplate::RenderAndTokenize(
        *tokenizer, messages, opts, &err);
    Expect(direct.has_value(), "Tool replay tokenizes: " + err);
    const auto prepared = gufo::models::qwen::vision::Prepare(
        *tokenizer, messages, {}, opts, {}, 1024);
    Expect(prepared.tokens == *direct,
           "Serving and direct tokenization agree on replayed arguments");
    Expect(tokenizer->Decode(*direct) == *rendered,
           "Replay preserves the rendered conversation exactly");
    Expect(std::count(direct->begin(), direct->end(), 257) == 3,
           "Only actual message boundaries become end-of-turn tokens");
    Expect(std::count(direct->begin(), direct->end(), 258) == 1 &&
               std::count(direct->begin(), direct->end(), 259) == 1,
           "Native tool-call tags stay framing while argument spellings stay "
           "text");
  }
}

void TestNewLiteralTokenDoesNotRetokenizeHistory() {
  using namespace gufo::tokenization;
  std::vector<std::string> vocab;
  for (int i = 0; i < 256; ++i)
    vocab.emplace_back(1, static_cast<char>(i));
  for (std::string_view token : std::initializer_list<std::string_view>{
           kImStart, kImEnd, "<tool_call>", "</tool_call>", ".\n"})
    vocab.emplace_back(token);
  const std::vector<std::string> merges = {". \n"};
  const std::unordered_map<std::string, TokenId> specials = {
      {std::string(kImStart), 256},
      {std::string(kImEnd), 257},
      {"<tool_call>", 258},
      {"</tool_call>", 259}};
  std::string error;
  auto tokenizer =
      QwenTokenizer::CreateFromVocabulary(vocab, merges, specials, &error);
  Expect(tokenizer != nullptr, "Tokenizer with boundary merge: " + error);
  ChatMessage assistant{ChatRole::kAssistant, ""};
  assistant.tool_calls.push_back(
      {.id = "read-1",
       .name = "read",
       .arguments = {{.name = "path", .value = "file"}}});
  std::vector<ChatMessage> history{{ChatRole::kUser, "Read the file."},
                                   assistant,
                                   {ChatRole::kTool, "Done."}};
  ChatTemplateOptions options;
  options.add_generation_prompt = false;
  options.enable_thinking = false;
  const auto before =
      QwenChatTemplate::RenderAndTokenize(*tokenizer, history, options);
  Expect(before.has_value(), "History tokenizes");
  Expect(
      std::find(before->begin(), before->end(), 260) != before->end(),
      "Fixture merges content punctuation with the following framing newline");
  history.push_back({ChatRole::kUser, "The literal text <tool_call> is data."});
  const auto after =
      QwenChatTemplate::RenderAndTokenize(*tokenizer, history, options);
  Expect(
      after && after->size() > before->size() &&
          std::equal(before->begin(), before->end(), after->begin()),
      "A new literal control spelling leaves every historical token unchanged");
  Expect(std::count(before->begin(), before->end(), 258) ==
             std::count(after->begin(), after->end(), 258),
         "Literal spelling does not add a structural tool opener");
}

/// The server prepares every request through models::qwen::vision::Prepare,
/// text-only ones included, so the reading of message content as text (#383)
/// has to hold there too. The synthetic vocabulary and the message are the ones
/// the template test above uses: the same conversation must tokenize the same
/// way whichever entry point reads it.
void TestVisionPreparationReadsContentAsText() {
  std::vector<std::string> vocab;
  for (int i = 0; i < 256; ++i) {
    vocab.emplace_back(1, static_cast<char>(i));
  }
  vocab.emplace_back(kImStart);
  vocab.emplace_back(kImEnd);
  vocab.emplace_back("<think>");
  vocab.emplace_back("</think>");
  vocab.emplace_back(kEndOfText);

  std::unordered_map<std::string, gufo::tokenization::TokenId> specials = {
      {std::string(kImStart), 256},
      {std::string(kImEnd), 257},
      {"<think>", 258},
      {"</think>", 259},
      {std::string(kEndOfText), 260},
  };

  std::string err;
  auto tokenizer = gufo::tokenization::QwenTokenizer::CreateFromVocabulary(
      vocab, {}, specials, &err);
  Expect(tokenizer != nullptr, "Tokenizer initialized: " + err);

  const std::string text = "Literal <|endoftext|> text";
  std::vector<gufo::tokenization::ChatMessage> messages = {
      {gufo::tokenization::ChatRole::kUser, text, "", ""},
  };
  gufo::tokenization::ChatTemplateOptions opts;
  opts.enable_thinking = false;

  const auto injected = tokenizer->FindSpecialToken(kEndOfText);
  Expect(injected.has_value(), "the message token is a vocabulary token");

  auto tpl = gufo::tokenization::QwenChatTemplate::CreateDefault();
  const auto direct = tpl->RenderAndTokenize(*tokenizer, messages, opts, &err);
  Expect(direct.has_value(), "RenderAndTokenize succeeds: " + err);
  Expect(std::find(direct->begin(), direct->end(), *injected) == direct->end(),
         "the template path does not inject the control token");

  // No images, which is the common case: the serving path still goes through
  // the vision preparation, and it must read the message as text as well.
  const auto prepared = gufo::models::qwen::vision::Prepare(
      *tokenizer, messages, {}, opts, {}, 512);
  Expect(std::find(prepared.tokens.begin(), prepared.tokens.end(), *injected) ==
             prepared.tokens.end(),
         "vision::Prepare does not inject a control token a message spelled");
  Expect(!prepared.tokens.empty() &&
             tokenizer->Decode(prepared.tokens).find(text) != std::string::npos,
         "the literal characters of the message reach the model");

  const auto framing = tokenizer->FindSpecialToken(kImStart);
  Expect(framing.has_value() &&
             std::find(prepared.tokens.begin(), prepared.tokens.end(),
                       *framing) != prepared.tokens.end(),
         "template framing still tokenizes as control tokens");
}

/// A picture in the message changes the framing the prompt is built from, and
/// the content around it still reads as text. The preparation walks an
/// image-bearing prompt in slices, so the helper is asked here with those same
/// boundaries, image placeholders included. Decoding the picture itself needs
/// the model's projector, which this test has no reason to carry: the
/// image-inputs functional suite covers that path end to end.
void TestEncodeRenderedReadsImageContentAsText() {
  std::vector<std::string> vocab;
  for (int i = 0; i < 256; ++i) {
    vocab.emplace_back(1, static_cast<char>(i));
  }
  for (std::string_view token : std::initializer_list<std::string_view>{
           kImStart, kImEnd, "<think>", "</think>", kEndOfText, kImagePad}) {
    vocab.emplace_back(token);
  }

  std::unordered_map<std::string, gufo::tokenization::TokenId> specials = {
      {std::string(kImStart), 256},
      {std::string(kImEnd), 257},
      {"<think>", 258},
      {"</think>", 259},
      {std::string(kEndOfText), 260},
      {std::string(kImagePad), 261},
  };

  std::string err;
  auto tokenizer = gufo::tokenization::QwenTokenizer::CreateFromVocabulary(
      vocab, {}, specials, &err);
  Expect(tokenizer != nullptr, "Tokenizer initialized: " + err);

  const std::string text = "Literal <|endoftext|> text";
  std::vector<gufo::tokenization::ChatMessage> messages = {
      {gufo::tokenization::ChatRole::kUser, text, "", ""},
  };
  // Rendering only, so the placeholder is enough: nothing decodes these bytes.
  messages.back().images.push_back(
      {0, std::make_shared<const std::vector<std::uint8_t>>(1, 0)});
  gufo::tokenization::ChatTemplateOptions opts;
  opts.enable_thinking = false;

  const auto injected = tokenizer->FindSpecialToken(kEndOfText);
  const auto framing = tokenizer->FindSpecialToken(kImStart);
  Expect(injected.has_value() && framing.has_value(),
         "the message and framing tokens are vocabulary tokens");

  std::vector<std::size_t> offsets;
  std::vector<gufo::tokenization::ContentSpan> spans;
  const auto rendered = gufo::tokenization::QwenChatTemplate::Render(
      messages, {}, opts, &err, &offsets, nullptr, &spans);
  Expect(rendered.has_value(), "an image-bearing prompt renders: " + err);
  Expect(offsets.size() == 1 &&
             rendered->substr(offsets[0], std::string_view(kImagePad).size()) ==
                 kImagePad,
         "the picture becomes an image placeholder");

  gufo::tokenization::TokenizerOptions tok_options;
  tok_options.add_bos = false;
  tok_options.add_eos = false;
  tok_options.parse_special_tokens = true;
  const auto encode = [&](std::size_t begin, std::size_t end) {
    return gufo::tokenization::QwenChatTemplate::EncodeRendered(
        *tokenizer, *rendered, begin, end, spans, tok_options);
  };
  const auto whole = encode(0, rendered->size());
  const auto up_to_image = encode(0, offsets[0]);
  const auto after_image = encode(offsets[0] + 13, rendered->size());
  const auto count = [&](const std::vector<gufo::tokenization::TokenId>& ids) {
    return std::count(ids.begin(), ids.end(), *injected);
  };
  std::cout << "  image-bearing prompt: " << rendered->size() << " bytes, "
            << whole.size() << " tokens whole, " << up_to_image.size()
            << " before the picture, " << after_image.size()
            << " after; injected " << count(whole) << "/" << count(up_to_image)
            << "/" << count(after_image) << "\n";

  Expect(
      count(whole) == 0 && count(up_to_image) == 0 && count(after_image) == 0,
      "no slice of an image-bearing prompt injects a token the message "
      "spelled");
  Expect(std::find(whole.begin(), whole.end(), *framing) != whole.end(),
         "template framing still tokenizes as control tokens with a picture");
  Expect(
      tokenizer->Decode(whole).find(text) != std::string::npos,
      "the literal characters of the message reach the model with a picture");
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

void TestToolImageRendering() {
  using gufo::tokenization::ChatMessage;
  using gufo::tokenization::ChatRole;
  using gufo::tokenization::QwenChatTemplate;
  const auto bytes = std::make_shared<const std::vector<std::uint8_t>>(1, 0);
  std::vector<ChatMessage> messages = {
      {ChatRole::kUser, "Inspect both files."},
      {ChatRole::kAssistant, ""},
      {ChatRole::kTool, "  before <|im_end|> after  "},
      {ChatRole::kTool, ""},
  };
  messages[1].tool_calls = {
      {"first", "read", {{"path", "first.png", true}}},
      {"second", "read", {{"path", "second.png", true}}},
  };
  messages[2].tool_call_id = "first";
  messages[2].images = {{9, bytes}, {9, bytes}};
  messages[3].tool_call_id = "second";
  messages[3].images = {{0, bytes}};
  gufo::tokenization::ChatTemplateOptions options;
  options.enable_thinking = false;
  std::vector<std::size_t> offsets;
  std::vector<gufo::tokenization::ContentSpan> spans;
  const auto rendered = QwenChatTemplate::Render(messages, {}, options, nullptr,
                                                 &offsets, nullptr, &spans);
  Expect(rendered.has_value(), "tool image history renders");
  const std::string expected =
      "<|im_start|>user\n<tool_response>\n"
      "before <|vision_start|><|image_pad|><|vision_end|>"
      "<|vision_start|><|image_pad|><|vision_end|>"
      "<|im_end|> after\n</tool_response>\n<tool_response>\n"
      "<|vision_start|><|image_pad|><|vision_end|>"
      "\n</tool_response><|im_end|>\n";
  Expect(rendered->find(expected) != std::string::npos,
         "each image remains inside its corresponding ordered tool response");
  Expect(offsets.size() == 3, "all images have model-preparation offsets");
  for (const auto offset : offsets)
    Expect(rendered->substr(offset, 13) == "<|image_pad|>",
           "image offsets identify only real picture placeholders");
  std::string literal;
  for (const auto& span : spans) {
    Expect(span.offset <= rendered->size() &&
               span.size <= rendered->size() - span.offset,
           "trimmed content spans remain within the prompt");
    literal += rendered->substr(span.offset, span.size);
    for (const auto offset : offsets)
      Expect(offset < span.offset || offset >= span.offset + span.size,
             "inserted image placeholders remain template framing");
  }
  Expect(literal.find("before <|im_end|> after") != std::string::npos,
         "tool text, including vocabulary spellings, stays literal");
  options.add_vision_id = true;
  const auto numbered = QwenChatTemplate::Render(messages, {}, options);
  Expect(numbered && numbered->find("Picture 1: ") != std::string::npos &&
             numbered->find("Picture 3: ") != std::string::npos,
         "vision numbering spans grouped tool results");
  for (const auto role :
       {ChatRole::kAssistant, ChatRole::kSystem, ChatRole::kDeveloper}) {
    auto invalid = messages;
    invalid[2].role = role;
    Expect(!QwenChatTemplate::Render(invalid, {}, options),
           "non-observation image owners remain invalid");
  }
  messages[2].images[0].offset = messages[2].content.size() + 1;
  Expect(!QwenChatTemplate::Render(messages, {}, options),
         "invalid tool image offsets remain rejected");
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

// Serving checkpoints a thinking prompt for reuse by the next turn. When that
// turn replays the assistant without reasoning, the generation suffix
// "<think>\n" is re-rendered as "<think>\n\n", which BPE merges into one
// token: the full prompt is no longer a token prefix, the prompt before the
// generation suffix still is.
void TestEmptyReasoningReplayChangesThinkingSuffixTokens() {
  using gufo::tokenization::ChatMessage;
  using gufo::tokenization::ChatRole;
  using gufo::tokenization::TokenId;

  std::vector<std::string> vocab;
  for (int i = 0; i < 256; ++i)
    vocab.emplace_back(1, static_cast<char>(i));
  vocab.emplace_back(kImStart);
  vocab.emplace_back(kImEnd);
  vocab.emplace_back("<think>");
  vocab.emplace_back("</think>");
  vocab.emplace_back("\n\n");
  const std::vector<std::string> merges = {"\n \n"};
  const std::unordered_map<std::string, TokenId> specials = {
      {std::string(kImStart), 256},
      {std::string(kImEnd), 257},
      {"<think>", 258},
      {"</think>", 259},
  };
  std::string error;
  const auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromVocabulary(vocab, merges,
                                                              specials, &error);
  Expect(tokenizer != nullptr, "Tokenizer with a newline merge: " + error);

  const auto tpl = gufo::tokenization::QwenChatTemplate::CreateDefault();
  gufo::tokenization::ChatTemplateOptions options;
  options.add_generation_prompt = true;
  options.enable_thinking = true;
  options.preserve_thinking = true;

  const std::vector<ChatMessage> first = {{ChatRole::kUser, "Hello"}};
  const auto prompt =
      tpl->RenderAndTokenize(*tokenizer, first, options, &error);
  Expect(prompt.has_value(), "First turn tokenizes: " + error);
  const auto suffix = tokenizer->Encode(
      gufo::tokenization::GenerationPrompt(options.enable_thinking),
      {.add_bos = false, .add_eos = false, .parse_special_tokens = true});
  Expect(prompt->size() > suffix.size() &&
             std::equal(
                 suffix.begin(), suffix.end(),
                 prompt->end() - static_cast<std::ptrdiff_t>(suffix.size())),
         "First turn ends with the thinking generation suffix");

  const auto is_prefix = [](std::span<const TokenId> prefix,
                            std::span<const TokenId> tokens) {
    return prefix.size() <= tokens.size() &&
           std::equal(prefix.begin(), prefix.end(), tokens.begin());
  };
  const auto next_turn = [&](std::string thought) {
    const std::vector<ChatMessage> messages = {
        {ChatRole::kUser, "Hello"},
        {ChatRole::kAssistant, "Hi", "", std::move(thought)},
        {ChatRole::kUser, "Again"},
    };
    const auto tokens =
        tpl->RenderAndTokenize(*tokenizer, messages, options, &error);
    Expect(tokens.has_value(), "Next turn tokenizes: " + error);
    return *tokens;
  };
  const auto stable =
      std::span<const TokenId>(*prompt).first(prompt->size() - suffix.size());

  const auto dropped = next_turn("");
  Expect(!is_prefix(*prompt, dropped),
         "Empty reasoning replay merges the suffix newline");
  Expect(is_prefix(stable, dropped),
         "Prompt before the generation suffix survives empty reasoning");

  const auto replayed = next_turn("Greeting.");
  Expect(is_prefix(*prompt, replayed),
         "Replayed reasoning keeps the full prompt as a prefix");
}

void TestStableBoundaryPrecedesReplacedFinalUserTurn() {
  using gufo::tokenization::ChatMessage;
  using gufo::tokenization::ChatRole;
  using gufo::tokenization::QwenChatTemplate;

  gufo::tokenization::ChatTemplateOptions options;
  options.add_generation_prompt = true;
  const auto render = [&](const std::vector<ChatMessage>& messages) {
    std::size_t stable = 0;
    auto text = QwenChatTemplate::Render(messages, {}, options, nullptr,
                                         nullptr, &stable);
    Expect(text.has_value(), "conversation renders");
    return std::pair{*text, stable};
  };
  const std::vector<ChatMessage> history = {
      {ChatRole::kSystem, "You are an agent."},
      {ChatRole::kUser, "Start the task."},
      {ChatRole::kAssistant, "Reading."},
      {ChatRole::kTool, "file contents"},
  };
  // Each request ends with per-turn context that the next request replaces.
  auto first = history;
  first.push_back({ChatRole::kUser, "Runtime context, turn 1."});
  auto second = history;
  second.push_back({ChatRole::kAssistant, "Done."});
  second.push_back({ChatRole::kUser, "Runtime context, turn 2."});
  const auto [first_text, first_stable] = render(first);
  const auto [second_text, second_stable] = render(second);
  Expect(first_stable < first_text.size() &&
             first_text.compare(first_stable, 12, kImStart) == 0,
         "boundary starts the final user turn");
  Expect(second_text.starts_with(first_text.substr(0, first_stable)),
         "boundary prefixes a request that replaces the final user turn");
  Expect(second_stable > first_stable, "boundary advances with the history");

  // A client that keeps the user turn still finds the boundary as a prefix.
  auto kept = first;
  kept.push_back({ChatRole::kAssistant, "Done."});
  kept.push_back({ChatRole::kUser, "Next."});
  const auto [kept_text, kept_stable] = render(kept);
  Expect(kept_text.starts_with(first_text.substr(0, first_stable)),
         "boundary prefixes an appended conversation");
  // A user turn after an assistant reply is ordinary chat: it is kept, so the
  // boundary stays before the generation prompt.
  Expect(kept_text.compare(kept_stable, std::string::npos,
                           gufo::tokenization::GenerationPrompt(
                               options.enable_thinking)) == 0,
         "ordinary chat keeps the boundary before the generation prompt");

  // Context sent as a second user message after the real query also moves it.
  auto query = history;
  query.push_back({ChatRole::kAssistant, "Done."});
  query.push_back({ChatRole::kUser, "Real question."});
  query.push_back({ChatRole::kUser, "Runtime context, turn 2."});
  const auto [query_text, query_stable] = render(query);
  Expect(query_text.compare(query_stable, 12, kImStart) == 0 &&
             query_text.substr(query_stable).find("Runtime context") !=
                 std::string::npos &&
             query_text.substr(query_stable).find("Real question") ==
                 std::string::npos,
         "boundary precedes trailing context after the real query");

  // The opening user turn has no earlier assistant: nothing to replace.
  const std::vector<ChatMessage> opening = {{ChatRole::kUser, "Hello"}};
  const auto [opening_text, opening_stable] = render(opening);
  Expect(opening_text.compare(opening_stable, std::string::npos,
                              gufo::tokenization::GenerationPrompt(
                                  options.enable_thinking)) == 0,
         "opening turn keeps the boundary before the generation prompt");
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
  TestContentSpellingATokenIsNotParsedAsOne();
  TestToolReplayArgumentsAreContent();
  TestNewLiteralTokenDoesNotRetokenizeHistory();
  TestVisionPreparationReadsContentAsText();
  TestEncodeRenderedReadsImageContentAsText();
  TestChatCorpusConformance();
  TestToolImageRendering();
  TestToolRendering();
  TestToolReplayPreservesGeneratedPrefix();
  TestEmptyReasoningReplayChangesThinkingSuffixTokens();
  TestStableBoundaryPrecedesReplacedFinalUserTurn();
  std::cout << "All QwenChatTemplate tests passed successfully!\n";
  return 0;
}
