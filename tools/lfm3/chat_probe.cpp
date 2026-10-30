// Model-private reproduction tool: preserves the exact prompt and raw output
// for independent teacher comparisons, without adding a serving debug API.
#include <fstream>
#include <iostream>
#include <iterator>

#include "src/core/json.hpp"
#include "src/core/sampling.hpp"
#include "src/core/text_sampling_defaults.hpp"
#include "src/models/lfm3/chat_template.hpp"
#include "src/models/lfm3/engine.hpp"

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: lfm3_chat_probe MODEL REQUEST.json OUTPUT.json\n";
    return 2;
  }
  try {
    using namespace gufo;
    std::ifstream input(argv[2]);
    if (!input)
      throw std::runtime_error("cannot open request recipe");
    auto recipe =
        json::parse(std::string(std::istreambuf_iterator<char>(input), {}));
    const auto* history = recipe.find("messages");
    if (!history || !history->is_array())
      throw std::invalid_argument("recipe messages must be an array");
    std::vector<tokenization::ChatMessage> messages;
    for (const auto& m : history->items()) {
      auto name = m.member_str("role");
      if (name != "system" && name != "assistant" && name != "tool" &&
          name != "user")
        throw std::invalid_argument("unsupported recipe message role");
      auto role = name == "system"      ? tokenization::ChatRole::kSystem
                  : name == "assistant" ? tokenization::ChatRole::kAssistant
                  : name == "tool"      ? tokenization::ChatRole::kTool
                                        : tokenization::ChatRole::kUser;
      messages.emplace_back(role, m.member_str("content"), "",
                            m.member_str("reasoning_content"));
      messages.back().tool_call_id = m.member_str("tool_call_id");
      if (const auto* calls = m.find("tool_calls"))
        for (const auto& call : calls->items()) {
          if (!call.find("function"))
            throw std::invalid_argument("recipe tool call has no function");
          const auto& fn = *call.find("function");
          tokenization::ChatMessage::ToolCall parsed{
              call.member_str("id"), fn.member_str("name"), {}};
          const auto args = json::parse(fn.member_str("arguments", "{}"));
          for (const auto& [key, value] : args.members())
            parsed.arguments.push_back(
                {key, value.is_string() ? value.str() : value.tojson(),
                 value.is_string()});
          messages.back().tool_calls.push_back(std::move(parsed));
        }
    }
    std::vector<tokenization::ChatTool> tools;
    if (const auto* list = recipe.find("tools"))
      for (const auto& t : list->items()) {
        if (!t.find("function"))
          throw std::invalid_argument("recipe tool has no function");
        const auto& fn = *t.find("function");
        tools.push_back(
            {fn.member_str("name"), fn.member_str("description"),
             fn.find("parameters") ? fn.find("parameters")->dump() : "{}",
             t.dump()});
      }
    ReasoningOptions reasoning;
    reasoning.enabled =
        recipe.find("thinking") && recipe.find("thinking")->as_bool();
    std::shared_ptr<const sampling::JsonConstraint> grammar;
    if (const auto* schema = recipe.find("response_schema")) {
      grammar = sampling::JsonConstraint::Compile(*schema, true, true);
      if (messages.empty() ||
          messages.front().role != tokenization::ChatRole::kSystem)
        messages.insert(messages.begin(),
                        {tokenization::ChatRole::kSystem, ""});
      messages.front().framing_suffix =
          (messages.front().content.empty() ? "" : "\n\n") + grammar->prompt();
    }
    if (!tools.empty()) {
      std::vector<sampling::JsonConstraint::Tool> constraints;
      for (const auto& t : tools)
        constraints.emplace_back(
            t.name, sampling::JsonConstraint::ToolParameters(
                        json::parse(t.parameters_json), false,
                        sampling::JsonConstraint::ToolFormat::kQwen));
      grammar = sampling::JsonConstraint::WithTools(
          grammar, std::move(constraints),
          recipe.member_str("tool_choice") == "required", false,
          sampling::JsonConstraint::ToolFormat::kQwen);
    }
    if (grammar && *reasoning.enabled)
      grammar = sampling::JsonConstraint::WithReasoning(grammar);
    std::string error;
    auto reader = core::GgufReader::OpenFile(argv[1], &error);
    if (!reader)
      throw std::runtime_error(error);
    auto tokenizer = models::lfm3::Tokenizer::Load(*reader, &error);
    if (!tokenizer)
      throw std::runtime_error(error);
    auto tokens =
        models::lfm3::EncodeChat(*tokenizer, messages, tools, reasoning);
    const auto prompt_count = tokens.size();
    if (auto prefix = recipe.member_str("prefix"); !prefix.empty()) {
      auto ids = tokenizer->Encode(prefix, false);
      tokens.insert(tokens.end(), ids.begin(), ids.end());
    }
    auto report = json::Value::object();
    report["prompt"] = models::lfm3::RenderChat(messages, tools, reasoning);
    report["prompt_count"] = prompt_count;
    report["tokens"] = json::Value::array();
    for (auto t : tokens)
      report["tokens"].push_back(std::uint64_t(t));
    report["generated_tokens"] = json::Value::array();
    const auto limit = recipe.member_size("max_tokens", 128);
    if (limit) {
      auto model = models::lfm3::Model::Load(argv[1], &error);
      if (!model)
        throw std::runtime_error(error);
      auto session = model->CreateSession(tokens.size() + limit, &error);
      std::vector<float> logits(model->config().vocab);
      if (!session || !session->Forward(tokens, logits, &error))
        throw std::runtime_error(error);
      auto config = sampling::ResolveTextSampling(
          sampling::TextModelPreset::kLfm3, *reasoning.enabled, {}, {});
      config.temperature = 0;
      if (grammar) {
        auto bound = std::make_shared<sampling::TokenConstraint>();
        bound->grammar = grammar;
        bound->vocabulary = std::make_shared<sampling::ConstraintVocabulary>(
            model->config().vocab, [&](std::uint32_t id) {
              return sampling::ConstraintVocabulary::Piece{
                  std::string(tokenizer->Piece(id)), tokenizer->IsStop(id)};
            });
        config.constraint = std::move(bound);
      }
      sampling::SamplerState sampler(config, tokens);
      std::string text;
      for (std::size_t i = 0; i < limit; ++i) {
        auto token = sampler.Sample(logits);
        if (tokenizer->IsStop(token))
          break;
        text += tokenizer->Piece(token);
        report["generated_tokens"].push_back(std::uint64_t(token));
        sampler.Accept(token);
        if (!session->Forward(std::span(&token, 1), logits, &error))
          throw std::runtime_error(error);
      }
      report["text"] = text;
    }
    std::ofstream output(argv[3]);
    output << report.dump() << '\n';
    if (!output)
      throw std::runtime_error("cannot write probe report");
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
