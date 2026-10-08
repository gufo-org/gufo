#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/models/gemma4/chat_template.hpp"
#include "tests/models/gemma4/check.hpp"
#include "tests/models/gemma4/template_cases.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;

namespace {

std::string Sha256(const std::string& text) {
  return gufo::crypto::Sha256Hex(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(text.data()), text.size()));
}

/// The renderer ends a replayed thought at <channel|>, where the Jinja source
/// adds a newline Gemma does not generate. Restores it for Jinja parity.
std::string WithJinjaThoughtNewlines(
    std::string text,
    std::span<const gufo::tokenization::ChatMessage> messages) {
  for (const auto& message : messages) {
    if (message.thought.empty())
      continue;
    const auto replayed = std::string(g4::kThoughtStart) + message.thought +
                          std::string(g4::kThoughtEnd);
    if (const auto at = text.find(replayed); at != std::string::npos)
      text.insert(at + g4::kThoughtStart.size() + message.thought.size(), "\n");
  }
  return text;
}

/// Every case must render byte-identically to the Jinja source, apart from
/// the documented end of a replayed thought.
void CheckGoldens() {
  const auto goldens = gemma4_test::ReadJson(GUFO_CHAT_TEMPLATE_HF_GOLDENS);
  const auto* gemma = goldens.find("gemma4");
  Require(gemma != nullptr, "goldens lack a gemma4 section");
  Require(gemma->member_str("template_sha256") ==
              g4::ChatTemplate::UnslothTemplateSha256(),
          "goldens were rendered from another template");
  const auto* expected = gemma->find("cases");
  const auto cases = gemma4_test::LoadTemplateCases(
      std::string(GUFO_GEMMA4_FIXTURES) + "/template_cases.json");
  Require(cases.size() == expected->size(), "case/golden count mismatch");
  for (const auto& c : cases) {
    std::string error;
    std::vector<std::size_t> images;
    const auto rendered = g4::ChatTemplate::Render(c.messages, c.tools,
                                                   c.options, &error, &images);
    Require(rendered.has_value(), c.name + ": " + error);
    std::size_t expected_images = 0;
    for (const auto& message : c.messages) {
      expected_images += message.images.size();
    }
    Require(images.size() == expected_images, c.name + ": image count");
    for (const auto offset : images) {
      Require(rendered->text.compare(offset, g4::kImagePlaceholder.size(),
                                     g4::kImagePlaceholder) == 0,
              c.name + ": image offset is not a placeholder");
    }
    const auto* golden = expected->find(c.name);
    Require(golden != nullptr, c.name + ": no golden");
    if (Sha256(WithJinjaThoughtNewlines(rendered->text, c.messages)) !=
        golden->member_str("rendered_sha256")) {
      std::cerr << "---- " << c.name << " rendered ----\n"
                << rendered->text << "\n----\n";
      throw std::runtime_error(c.name + ": rendering differs from Jinja");
    }

    // The generation prompt is a pure suffix of the stable conversation.
    auto without = c.options;
    without.add_generation_prompt = false;
    const auto prefix = g4::ChatTemplate::Render(c.messages, c.tools, without,
                                                 nullptr, &images);
    Require(
        prefix && prefix->text == rendered->text.substr(
                                      0, rendered->generation_prompt_offset),
        c.name + ": generation prompt is not a suffix");
  }
}

void CheckOptions() {
  gufo::ReasoningOptions reasoning;
  Require(!g4::ResolveChatOptions(reasoning).enable_thinking,
          "thinking is off by default");
  reasoning.effort = gufo::ReasoningEffort::kHigh;
  Require(g4::ResolveChatOptions(reasoning).enable_thinking,
          "an effort level enables thinking");
  reasoning.effort = gufo::ReasoningEffort::kMinimal;
  Require(!g4::ResolveChatOptions(reasoning).enable_thinking,
          "minimal effort keeps thinking off");
  reasoning.enabled = true;
  Require(g4::ResolveChatOptions(reasoning).enable_thinking,
          "explicit enable wins");
}

void CheckRejections() {
  gufo::tokenization::ChatMessage image;
  image.images.push_back({});
  std::string error;
  Require(!g4::ChatTemplate::Render(std::span(&image, 1), {}, {}, &error) &&
              !error.empty(),
          "image input accepted without image offsets");
  gufo::tokenization::ChatTool bad;
  bad.definition_json = "[1]";
  Require(!g4::ChatTemplate::Render({}, std::span(&bad, 1), {}, &error),
          "non-object tool accepted");
  gufo::tokenization::ChatTool braced;
  braced.name = "a{b";
  Require(!g4::ChatTemplate::Render({}, std::span(&braced, 1), {}, &error),
          "braced tool name accepted");
  gufo::tokenization::ChatMessage call(gufo::tokenization::ChatRole::kAssistant,
                                       "");
  call.tool_calls.push_back({.id = "c", .name = "x{y", .arguments = {}});
  Require(!g4::ChatTemplate::Render(std::span(&call, 1), {}, {}, &error),
          "braced replayed call accepted");
  gufo::tokenization::ChatTool dotted;
  dotted.name = "github.create_issue";
  dotted.definition_json =
      R"({"type":"function","function":{"name":"github.create_issue"}})";
  const auto rendered =
      g4::ChatTemplate::Render({}, std::span(&dotted, 1), {}, &error);
  Require(rendered && rendered->text.find("declaration:github.create_issue{") !=
                          std::string::npos,
          "dotted tool name rendered: " + error);
}

void CheckShownConstants() {
  gufo::tokenization::ChatTool tool;
  tool.name = "record";
  tool.definition_json = R"({"type":"function","function":{"name":"record",
    "parameters":{"type":"object","properties":{
      "value":{"type":"string","const":"alpha"},
      "items":{"type":"array","items":{"type":"object","properties":{
        "kind":{"type":"string","const":"x"}}}},
      "n":{"type":"integer","const":3}},"required":["value"]}}})";
  std::string error;
  const auto rendered =
      g4::ChatTemplate::Render({}, std::span(&tool, 1), {}, &error);
  Require(rendered.has_value(), "const tool rendered: " + error);
  const auto& text = rendered->text;
  Require(text.find("value:{enum:[<|\"|>alpha<|\"|>]") != std::string::npos &&
              text.find("kind:{enum:[<|\"|>x<|\"|>]") != std::string::npos &&
              text.find("n:{type:") != std::string::npos,
          "a string const is shown as a one-value enum: " + text);
}

/// A replayed thought renders as Gemma generates it, so a continuation's
/// prompt reproduces the reasoning and call tokens of the previous turn.
void CheckReplayedThought() {
  using gufo::tokenization::ChatMessage;
  using gufo::tokenization::ChatRole;
  ChatMessage call(ChatRole::kAssistant, "", "", "Read the file first.");
  call.tool_calls.push_back(
      {.id = "c", .name = "read", .arguments = {{"path", "a.py", true}}});
  ChatMessage result(ChatRole::kTool, "x = 1");
  result.tool_call_id = "c";
  const std::vector<ChatMessage> messages{
      ChatMessage(ChatRole::kUser, "Read a.py."), call, result};
  g4::ChatOptions options;
  options.enable_thinking = true;
  std::string error;
  const auto rendered = g4::ChatTemplate::Render(messages, {}, options, &error);
  Require(rendered.has_value(), "replayed thought rendered: " + error);
  Require(rendered->text.find("<|channel>thought\nRead the file first."
                              "<channel|><|tool_call>call:read{") !=
              std::string::npos,
          "a replayed thought ends at <channel|>: " + rendered->text);
}

/// Server-authored framing such as a response-format instruction renders
/// after the client's system text, including in a system turn it creates.
void CheckFramingSuffix() {
  using gufo::tokenization::ChatMessage;
  using gufo::tokenization::ChatRole;
  ChatMessage system(ChatRole::kSystem, "Be brief.");
  system.framing_suffix = "\n\nAnswer in JSON.";
  const std::vector<ChatMessage> with_system{
      system, ChatMessage(ChatRole::kUser, "Hi")};
  std::string error;
  auto rendered = g4::ChatTemplate::Render(with_system, {}, {}, &error);
  Require(rendered.has_value(), "framing rendered: " + error);
  Require(rendered->text.find("<|turn>system\nBe brief.\n\nAnswer in JSON."
                              "<turn|>") != std::string::npos,
          "framing follows system text: " + rendered->text);
  ChatMessage framing(ChatRole::kSystem, "");
  framing.framing_suffix = "Answer in JSON.";
  const std::vector<ChatMessage> created{framing,
                                         ChatMessage(ChatRole::kUser, "Hi")};
  rendered = g4::ChatTemplate::Render(created, {}, {}, &error);
  Require(rendered.has_value(), "created framing rendered: " + error);
  Require(rendered->text.find("<|turn>system\nAnswer in JSON.<turn|>") !=
              std::string::npos,
          "framing renders in a created system turn: " + rendered->text);
}

/// Agent clients may end each request with user context that the next
/// request replaces. The stable boundary then precedes that turn, so it
/// prefixes the next request; ordinary chat keeps it at the generation prompt.
void CheckReplacedFinalUserTurn() {
  using gufo::tokenization::ChatMessage;
  using gufo::tokenization::ChatRole;
  const auto render = [](const std::vector<ChatMessage>& messages) {
    std::string error;
    auto rendered = g4::ChatTemplate::Render(messages, {}, {}, &error);
    Require(rendered.has_value(), "conversation rendered: " + error);
    Require(
        rendered->stable_prefix_offset <= rendered->generation_prompt_offset,
        "boundary precedes the generation prompt");
    return std::pair{rendered->text, rendered->stable_prefix_offset};
  };
  const auto prefixes = [](const std::string& text, const std::string& first,
                           std::size_t stable) {
    return text.starts_with(std::string_view(first).substr(0, stable));
  };
  ChatMessage call(ChatRole::kAssistant, "");
  call.tool_calls.push_back(
      {.id = "c", .name = "read", .arguments = {{"path", "a.py", true}}});
  ChatMessage result(ChatRole::kTool, "x = 1");
  result.tool_call_id = "c";
  const std::vector<ChatMessage> history{
      ChatMessage(ChatRole::kSystem, "You are an agent."),
      ChatMessage(ChatRole::kUser, "Start the task."), call, result};

  // After a tool result the model turn stays open on the next request.
  auto first = history;
  first.emplace_back(ChatRole::kUser, "Runtime context, turn 1.");
  auto second = history;
  second.emplace_back(ChatRole::kAssistant, "Done.");
  second.emplace_back(ChatRole::kUser, "Runtime context, turn 2.");
  const auto [first_text, first_stable] = render(first);
  const auto [second_text, second_stable] = render(second);
  Require(first_text.compare(first_stable, 7, "<turn|>") == 0 &&
              first_text.substr(0, first_stable).ends_with("<tool_response|>"),
          "boundary closes the tool result: " + first_text);
  Require(prefixes(second_text, first_text, first_stable),
          "boundary prefixes a request that replaces the final user turn");
  Require(second_stable > first_stable, "boundary advances with the history");

  // A client that keeps the user turn still finds the boundary as a prefix;
  // a user turn after an assistant reply is ordinary chat.
  auto kept = first;
  kept.emplace_back(ChatRole::kAssistant, "Done.");
  kept.emplace_back(ChatRole::kUser, "Next.");
  const auto [kept_text, kept_stable] = render(kept);
  Require(prefixes(kept_text, first_text, first_stable),
          "boundary prefixes an appended conversation");
  Require(kept_text.compare(kept_stable, std::string::npos,
                            "<|turn>model\n<|channel>thought\n<channel|>") == 0,
          "ordinary chat keeps the boundary at the generation prompt");

  // Context sent as a second user message after the real query.
  auto query = second;
  query.back().content = "Real question.";
  query.emplace_back(ChatRole::kUser, "Runtime context, turn 2.");
  const auto [query_text, query_stable] = render(query);
  Require(query_text.compare(query_stable, 12, "<|turn>user\n") == 0 &&
              query_text.substr(query_stable).find("Runtime context") !=
                  std::string::npos &&
              query_text.substr(query_stable).find("Real question") ==
                  std::string::npos,
          "boundary precedes trailing context after the real query");

  // The opening user turn has no earlier assistant: nothing to replace.
  const auto [opening_text, opening_stable] =
      render({ChatMessage(ChatRole::kUser, "Hello")});
  Require(
      opening_text.compare(opening_stable, std::string::npos,
                           "<|turn>model\n<|channel>thought\n<channel|>") == 0,
      "opening turn keeps the boundary at the generation prompt");
}

}  // namespace

int main() {
  return gemma4_test::Run([] {
    CheckGoldens();
    CheckOptions();
    CheckRejections();
    CheckShownConstants();
    CheckReplayedThought();
    CheckFramingSuffix();
    CheckReplacedFinalUserTurn();
  });
}
