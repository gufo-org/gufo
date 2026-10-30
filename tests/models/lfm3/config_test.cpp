#include "src/models/lfm3/config.hpp"

#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <variant>

#include "src/core/text_sampling_defaults.hpp"
#include "src/models/lfm3/chat_template.hpp"
#include "src/models/lfm3/tokenizer.hpp"
namespace l = gufo::models::lfm3;
namespace {
using Value =
    std::variant<std::uint32_t, float, bool, std::string,
                 std::vector<std::uint64_t>, std::vector<std::string>>;
using Fields = std::map<std::string, Value>;
void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}
std::optional<l::Config> Parse(const Fields& fields) {
  std::vector<std::uint8_t> bytes;
  auto pod = [&]<class T>(T value) {
    auto at = bytes.size();
    bytes.resize(at + sizeof(T));
    std::memcpy(bytes.data() + at, &value, sizeof(T));
  };
  auto text = [&](const std::string& value) {
    pod(std::uint64_t(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
  };
  pod(std::uint32_t(0x46554747));
  pod(std::uint32_t(3));
  pod(std::uint64_t(0));
  pod(std::uint64_t(fields.size()));
  for (const auto& [key, value] : fields) {
    text(key);
    std::visit(
        [&](const auto& v) {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, std::uint32_t>) {
            pod(std::uint32_t(4));
            pod(v);
          } else if constexpr (std::is_same_v<T, float>) {
            pod(std::uint32_t(6));
            pod(v);
          } else if constexpr (std::is_same_v<T, bool>) {
            pod(std::uint32_t(7));
            pod(std::uint8_t(v));
          } else if constexpr (std::is_same_v<T, std::string>) {
            pod(std::uint32_t(8));
            text(v);
          } else {
            pod(std::uint32_t(9));
            pod(std::uint32_t(
                std::is_same_v<T, std::vector<std::string>> ? 8 : 10));
            pod(std::uint64_t(v.size()));
            for (const auto& x : v) {
              if constexpr (std::is_same_v<T, std::vector<std::string>>)
                text(x);
              else
                pod(x);
            }
          }
        },
        value);
  }
  while (bytes.size() % 32)
    bytes.push_back(0);
  std::string error;
  auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error);
  Require(bool(reader), error.c_str());
  return l::Config::FromGguf(*reader, &error);
}
Fields Good() {
  return {{"general.architecture", std::string("lfm3")},
          {"lfm3.block_count", 3U},
          {"lfm3.context_length", 262144U},
          {"lfm3.embedding_length", 2048U},
          {"lfm3.feed_forward_length", 7680U},
          {"lfm3.attention.head_count", 16U},
          {"lfm3.attention.key_length", 128U},
          {"lfm3.attention.value_length", 128U},
          {"lfm3.rope.dimension_count", 128U},
          {"lfm3.attention.sliding_window", 256U},
          {"lfm3.shortconv.l_cache", 7U},
          {"lfm3.attention.layer_norm_rms_epsilon", 1e-5F},
          {"lfm3.rope.freq_base", 1e6F},
          {"lfm3.rope.freq_base_swa", 1e4F},
          {"lfm3.attention.head_count_kv", std::vector<std::uint64_t>{0, 4, 4}},
          {"lfm3.attention.sliding_window_pattern",
           std::vector<std::uint64_t>{0, 0, 1}},
          {"tokenizer.ggml.tokens", std::vector<std::string>{"a", "b"}}};
}
}  // namespace
int main() {
  try {
    auto good = Good();
    auto c = Parse(good);
    Require(bool(c), "valid LFM3 metadata rejected");
    Require(c->layers == std::vector{l::LayerType::kConvolution,
                                     l::LayerType::kFullAttention,
                                     l::LayerType::kSlidingAttention},
            "wrong layer layout");
    for (auto key : {"lfm3.block_count", "lfm3.attention.key_length",
                     "lfm3.rope.freq_base", "lfm3.attention.head_count_kv",
                     "lfm3.attention.sliding_window_pattern"}) {
      auto bad = good;
      bad.erase(key);
      Require(!Parse(bad), "missing metadata accepted");
    }
    for (auto heads :
         {std::vector<std::uint64_t>{0, 4}, {0, 3, 4}, {0, 17, 4}}) {
      auto bad = good;
      bad["lfm3.attention.head_count_kv"] = heads;
      Require(!Parse(bad), "bad heads accepted");
    }
    auto bad = good;
    bad["lfm3.attention.sliding_window_pattern"] =
        std::vector<std::uint64_t>{1, 0, 1};
    Require(!Parse(bad), "sliding convolution accepted");
    bad = good;
    bad["lfm3.attention.causal"] = false;
    Require(!Parse(bad), "unknown decision model accepted");
    bad["lfm3.decision.type"] = std::string("lfm3-d1-omni");
    auto decision = Parse(bad);
    Require(decision && decision->decision,
            "d1-Omni decision metadata rejected");
    bad["lfm3.attention.head_count"] = 12U;
    Require(!Parse(bad), "unsupported d1 attention grouping accepted");
    bad = good;
    bad["lfm3.rope.freq_base"] = 0.0F;
    Require(!Parse(bad), "zero theta accepted");
    const std::vector<std::pair<std::string, std::string>> chat{
        {"system", "Be concise."}, {"user", "Hello"}};
    Require(l::Tokenizer::Render(chat) ==
                "<|startoftext|><|im_start|>system\nBe "
                "concise.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_"
                "start|>assistant\n<think>",
            "thinking template differs from HF");
    Require(l::Tokenizer::Render(chat, false)
                .ends_with("assistant\n<think></think>"),
            "nonthinking prefix differs from HF");
    auto multi = chat;
    multi.emplace_back("assistant", "<think>old thought</think>  Hello! \n");
    multi.emplace_back("user", "Again");
    Require(
        l::Tokenizer::Render(multi).find(
            "assistant\n<think></think>Hello!<|im_end|>") != std::string::npos,
        "historical reasoning not removed");
    using gufo::tokenization::ChatMessage;
    using gufo::tokenization::ChatRole;
    std::vector<ChatMessage> native{{ChatRole::kSystem, "Be concise."},
                                    {ChatRole::kUser, "Hello"}};
    Require(l::RenderChat(native) == l::Tokenizer::Render(chat),
            "native plain template differs from HF golden");
    gufo::tokenization::ChatTool tool{
        "weather", "Weather",
        R"({"type":"object","properties":{"city":{"type":"string"}}})", ""};
    Require(l::RenderChat(native, std::span(&tool, 1))
                    .find(
                        R"(<tools>
{"name": "weather", "description": "Weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}
</tools>)") != std::string::npos,
            "native tool definition differs from Jinja golden");
    ChatMessage assistant{ChatRole::kAssistant, "", "", "Look it up."};
    assistant.tool_calls.push_back(
        {"call1",
         "weather",
         {{"city", "Paris", true}, {"days", "[1,2]", false}}});
    native.push_back(assistant);
    ChatMessage response{ChatRole::kTool, "sunny"};
    response.tool_call_id = "call1";
    native.push_back(response);
    auto rendered = l::RenderChat(native);
    Require(rendered.find("<think>Look it "
                          "up.</"
                          "think><tool_call>\n<function=weather>\n<parameter="
                          "city>\nParis\n</parameter>\n<parameter=days>\n[1, "
                          "2]\n</parameter>\n</function>\n</tool_call>") !=
                std::string::npos,
            "native call/thought framing differs from Jinja golden");
    Require(rendered.find("<|im_start|>tool\n<tool_response>\nsunny\n</"
                          "tool_response><|im_end|>\n") != std::string::npos,
            "native tool response differs from Jinja golden");
    native.emplace_back(ChatRole::kUser, "Next");
    Require(l::RenderChat(native).find("Look it up.") == std::string::npos,
            "stale native thinking not stripped");
    gufo::ReasoningOptions preserve;
    preserve.preserve_thinking = true;
    preserve.effort = gufo::ReasoningEffort::kLow;
    Require(l::RenderChat(native, {}, preserve)
                    .find("<think>Look it up.</think>") != std::string::npos,
            "preserved thinking discarded");
    preserve.enabled = false;
    Require(l::RenderChat(native, {}, preserve)
                .ends_with("assistant\n<think></think>"),
            "thinking-off override ignored");
    const auto sampling = gufo::sampling::ResolveTextSampling(
        gufo::sampling::TextModelPreset::kLfm3, true, {}, {});
    Require(sampling.temperature == 0.1F && sampling.top_k == 50 &&
                sampling.repeat_penalty == 1.1F,
            "wrong LFM3 defaults");
    std::cout << "PASS LFM3 metadata, chat framing and sampling defaults\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
