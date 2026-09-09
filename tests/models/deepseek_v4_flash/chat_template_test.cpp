#include "src/models/deepseek_v4_flash/chat_template.hpp"

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

namespace {

using gufo::ReasoningEffort;
using gufo::models::deepseek_v4_flash::ChatMessage;
using gufo::models::deepseek_v4_flash::ChatTemplateOptions;
using gufo::models::deepseek_v4_flash::ChatTool;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

std::string Sha256(std::string_view value) {
  const auto* data = reinterpret_cast<const std::uint8_t*>(value.data());
  return gufo::crypto::Sha256Hex({data, value.size()});
}

class GgufTemplateBuilder {
public:
  GgufTemplateBuilder() {
    const char magic[4] = {'G', 'G', 'U', 'F'};
    AppendBytes(magic, 4);
    AppendPod(static_cast<std::uint32_t>(3));
    AppendPod(static_cast<std::uint64_t>(0));
    metadata_count_pos_ = buffer_.size();
    AppendPod(static_cast<std::uint64_t>(0));
  }

  void AddString(std::string_view key, std::string_view value) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kString));
    AppendString(value);
    ++metadata_count_;
  }

  std::vector<std::uint8_t> Build() {
    std::memcpy(buffer_.data() + metadata_count_pos_, &metadata_count_,
                sizeof(metadata_count_));
    if (const std::size_t remainder = buffer_.size() % 32; remainder != 0) {
      buffer_.resize(buffer_.size() + 32 - remainder, 0);
    }
    return buffer_;
  }

private:
  void AppendBytes(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    buffer_.insert(buffer_.end(), bytes, bytes + size);
  }

  template<typename T>
  void AppendPod(T value) {
    AppendBytes(&value, sizeof(value));
  }

  void AppendString(std::string_view value) {
    AppendPod(static_cast<std::uint64_t>(value.size()));
    AppendBytes(value.data(), value.size());
  }

  std::vector<std::uint8_t> buffer_;
  std::size_t metadata_count_pos_{0};
  std::uint64_t metadata_count_{0};
};

void TestChatAndThinkingPrefixes() {
  const std::vector<ChatMessage> messages = {
      {.role = "system", .content = "Be concise.", .reasoning_content = {}},
      {.role = "user", .content = "Answer.", .reasoning_content = {}},
  };

  const std::string chat =
      gufo::models::deepseek_v4_flash::RenderChat(messages);
  Expect(chat ==
             "<｜begin▁of▁sentence｜>Be concise.<｜User｜>Answer."
             "<｜Assistant｜></think>",
         "Chat mode uses the no-thinking assistant prefix");

  ChatTemplateOptions options;
  options.enable_thinking = true;
  options.reasoning_effort = ReasoningEffort::kHigh;
  const std::string thinking =
      gufo::models::deepseek_v4_flash::RenderChat(messages, options);
  Expect(
      thinking.find("Reasoning Effort: Absolute maximum") != std::string::npos,
      "High effort uses the pinned DeepSeek instruction");
  Expect(thinking.ends_with("<｜Assistant｜><think>"),
         "Thinking mode starts inside a reasoning block");
}

void TestHistoricalThinkingPolicy() {
  const std::vector<ChatMessage> messages = {
      {.role = "user", .content = "First.", .reasoning_content = {}},
      {.role = "assistant",
       .content = "Visible answer.",
       .reasoning_content = "Private reasoning."},
      {.role = "user", .content = "Second.", .reasoning_content = {}},
  };

  ChatTemplateOptions options;
  options.enable_thinking = true;
  options.reasoning_effort = ReasoningEffort::kLow;
  const std::string dropped =
      gufo::models::deepseek_v4_flash::RenderChat(messages, options);
  Expect(dropped.find("Private reasoning.") == std::string::npos,
         "DeepSeek drops old thinking by default");
  Expect(dropped.find("<｜Assistant｜></think>Visible answer.") !=
             std::string::npos,
         "Dropping thought preserves visible assistant content");

  options.preserve_thinking = true;
  const std::string preserved =
      gufo::models::deepseek_v4_flash::RenderChat(messages, options);
  Expect(
      preserved.find(
          "<｜Assistant｜><think>Private reasoning.</think>Visible answer.") !=
          std::string::npos,
      "Explicit preservation replays historical reasoning");

  options.enable_thinking = false;
  const std::string chat_mode =
      gufo::models::deepseek_v4_flash::RenderChat(messages, options);
  Expect(chat_mode.find("Private reasoning.") == std::string::npos,
         "Standard DeepSeek chat mode never replays reasoning history");
  Expect(chat_mode.find("<｜Assistant｜></think>Visible answer.") !=
             std::string::npos,
         "Chat mode reconstructs assistant history into visible content");

  options.enable_thinking = true;
  options.preserve_thinking = false;
  options.tools_present = true;
  const std::string tool_loop =
      gufo::models::deepseek_v4_flash::RenderChat(messages, options);
  Expect(tool_loop.find("Private reasoning.") != std::string::npos,
         "Tool conversations force reasoning preservation");
}

void TestEffortMappingAndToolResults() {
  using gufo::models::deepseek_v4_flash::DeepSeekReasoningEffortName;
  Expect(DeepSeekReasoningEffortName(ReasoningEffort::kMinimal) == "low",
         "minimal maps to DeepSeek low");
  Expect(DeepSeekReasoningEffortName(ReasoningEffort::kMedium) == "high",
         "medium maps to DeepSeek high");
  Expect(DeepSeekReasoningEffortName(ReasoningEffort::kXHigh) == "max",
         "xhigh maps to DeepSeek max");

  const std::vector<ChatMessage> messages = {
      {.role = "tool", .content = "one", .reasoning_content = {}},
      {.role = "tool", .content = "two", .reasoning_content = {}},
  };
  const std::string rendered =
      gufo::models::deepseek_v4_flash::RenderChat(messages);
  Expect(rendered.find("<｜User｜><tool_result>one</tool_result>\n\n"
                       "<tool_result>two</tool_result>") != std::string::npos,
         "Consecutive tool results share one user turn");
}

void TestOfficialDsmlToolLoop() {
  ChatMessage assistant{
      .role = "assistant",
      .content = "",
      .reasoning_content = "Use the tool.",
  };
  assistant.tool_calls.push_back({
      .name = "get_weather",
      .arguments = {{.name = "city", .value = "Rome", .is_string = true}},
  });
  const std::vector<ChatMessage> messages = {
      {.role = "system", .content = "Be concise.", .reasoning_content = {}},
      {.role = "user", .content = "Weather in Rome?", .reasoning_content = {}},
      assistant,
      {.role = "tool",
       .content = "{\"temperature\":21}",
       .reasoning_content = {}},
      {.role = "user", .content = "Summarize.", .reasoning_content = {}},
  };
  const std::vector<ChatTool> tools = {{
      .name = "get_weather",
      .description = "Get weather",
      .parameters_json = "{\"type\":\"object\",\"properties\":{\"city\":{"
                         "\"type\":\"string\"}},"
                         "\"required\":[\"city\"]}",
  }};
  ChatTemplateOptions options;
  options.enable_thinking = true;
  const std::string rendered =
      gufo::models::deepseek_v4_flash::RenderChat(messages, tools, options);
  Expect(rendered.find("<｜DSML｜tool_calls>") != std::string::npos,
         "DeepSeek uses the official DSML tool-calls token");
  Expect(rendered.find("<｜DSML｜invoke name=\"get_weather\">") !=
             std::string::npos,
         "DeepSeek uses the official DSML invoke token");
  Expect(rendered.find("<｜DSML｜parameter name=\"city\" string=\"true\">"
                       "Rome</｜DSML｜parameter>") != std::string::npos,
         "DeepSeek uses official typed DSML parameters");
  Expect(rendered.find("<｜DS｜") == std::string::npos,
         "Legacy nonstandard DS tool markers are absent");
  Expect(Sha256(rendered) ==
             "cd89794650716ce13bac7ac2392db6a06231ce9927ceb1d712f7c399a0a2fde6",
         "DeepSeek tool loop matches the pinned official encoder golden");
}

void TestHuggingFaceRenderedGoldens() {
  const std::vector<ChatMessage> base = {
      {.role = "system", .content = "Be concise.", .reasoning_content = {}},
      {.role = "user", .content = "Name one color.", .reasoning_content = {}},
  };
  const std::vector<ChatMessage> history = {
      {.role = "user", .content = "Name one color.", .reasoning_content = {}},
      {.role = "assistant",
       .content = "Red",
       .reasoning_content = "I should answer with one color."},
      {.role = "user", .content = "Name another.", .reasoning_content = {}},
  };
  const auto check = [](const std::vector<ChatMessage>& messages,
                        const ChatTemplateOptions& options,
                        std::string_view expected) {
    const std::string rendered =
        gufo::models::deepseek_v4_flash::RenderChat(messages, options);
    Expect(Sha256(rendered) == expected,
           "DeepSeek rendered bytes match the pinned Hugging Face golden");
  };
  ChatTemplateOptions options;
  check(base, options,
        "d3953e69808de7acaa73269d1fbd13a930a4ee743b65ec2883b91d206ceb4121");
  options.enable_thinking = true;
  options.reasoning_effort = ReasoningEffort::kLow;
  check(base, options,
        "302d80db67339dedc8489476bde608f250f13f2b7bb89d6c4467538e057e0dba");
  options.reasoning_effort = ReasoningEffort::kHigh;
  check(base, options,
        "4e0a1a55cd4c39b593c49f3ec8384e32cde119bff97416a611d7ab459624aaa9");
  options.reasoning_effort = ReasoningEffort::kMax;
  check(base, options,
        "b61a0552e96f0458cada39d46bd11a15adc206f0cd3a1dfea096687fe3fb4968");
  options.reasoning_effort = ReasoningEffort::kLow;
  options.preserve_thinking = false;
  check(history, options,
        "df3c2370812a2f993fca67aee691ccf730d57f24f9207cb8e5b1be38ba916e88");
  options.preserve_thinking = true;
  check(history, options,
        "2fc631d9339a08cd4fffd643eae2b58426fb0d709e69daaaffb978d55b6906d3");
}

void TestPinnedArtifactTemplateValidation() {
  std::ifstream input(GUFO_DEEPSEEK_CHAT_TEMPLATE_REFERENCE, std::ios::binary);
  const std::string reference{std::istreambuf_iterator<char>(input),
                              std::istreambuf_iterator<char>()};
  Expect(!reference.empty(), "Pinned DeepSeek artifact template is readable");
  Expect(Sha256(reference) ==
             gufo::models::deepseek_v4_flash::ArtifactTemplateSha256(),
         "Versioned DeepSeek Jinja has the pinned artifact hash");

  GgufTemplateBuilder builder;
  builder.AddString("general.architecture", "deepseek4");
  builder.AddString("general.name", "DeepSeek V4 Flash");
  builder.AddString("tokenizer.chat_template", reference);
  auto binary = builder.Build();
  std::string error;
  auto reader =
      gufo::core::GgufReader::OpenMemory(binary.data(), binary.size(), &error);
  Expect(reader != nullptr, "DeepSeek template GGUF opens");
  Expect(gufo::models::deepseek_v4_flash::ValidateGgufTemplate(*reader, &error),
         "Pinned DeepSeek artifact template is accepted");

  GgufTemplateBuilder lookalike;
  lookalike.AddString("general.architecture", "deepseek4");
  lookalike.AddString("general.name", "DeepSeek V4 Flash");
  lookalike.AddString("tokenizer.chat_template", reference + " ");
  binary = lookalike.Build();
  reader =
      gufo::core::GgufReader::OpenMemory(binary.data(), binary.size(), &error);
  Expect(reader != nullptr, "DeepSeek look-alike GGUF opens");
  Expect(
      !gufo::models::deepseek_v4_flash::ValidateGgufTemplate(*reader, &error),
      "DeepSeek look-alike template is rejected by hash");
}

}  // namespace

int main() {
  TestChatAndThinkingPrefixes();
  TestHistoricalThinkingPolicy();
  TestEffortMappingAndToolResults();
  TestOfficialDsmlToolLoop();
  TestHuggingFaceRenderedGoldens();
  TestPinnedArtifactTemplateValidation();
  std::cout << "DeepSeek V4 Flash chat template tests passed\n";
  return 0;
}
