// Checks the chat renderer against Jinja2 goldens of the artifact's template
// (tools/models/nemotron35_bolt/chat_template_golden.py).
//
//   nemotron35_bolt_chat_template_test render
//       Rendered bytes; needs no weights.
//   nemotron35_bolt_chat_template_test encode MODEL.gguf
//       Token IDs of the rendered prompts. Without a model path (argument or
//       GUFO_NEMOTRON_BOLT_GGUF) it skips.

#include "src/models/nemotron35_bolt/chat_template.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
#include "src/models/nemotron35_bolt/tokenizer.hpp"

#ifndef GUFO_NEMOTRON35_BOLT_CHAT_GOLDEN
#define GUFO_NEMOTRON35_BOLT_CHAT_GOLDEN \
  "tests/fixtures/nemotron35_bolt/chat_template_golden.json"
#endif

namespace nb = gufo::models::nemotron35_bolt;
namespace json = gufo::json;

namespace {

constexpr int kSkip = 77;

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL nemotron35_bolt_chat_template_test: " << message << '\n';
    std::exit(1);
  }
}

json::Value LoadFixture() {
  std::filesystem::path path = GUFO_NEMOTRON35_BOLT_CHAT_GOLDEN;
  if (!std::filesystem::exists(path)) {
    if (const char* directory = std::getenv("GUFO_FIXTURE_DIR"))
      path = std::filesystem::path(directory) / path.filename();
  }
  std::ifstream input(path, std::ios::binary);
  Check(static_cast<bool>(input), "cannot open " + path.string());
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return json::parse(buffer.str());
}

nb::ChatRole Role(const std::string& role) {
  if (role == "system")
    return nb::ChatRole::kSystem;
  if (role == "assistant")
    return nb::ChatRole::kAssistant;
  if (role == "tool")
    return nb::ChatRole::kTool;
  return nb::ChatRole::kUser;
}

std::string Render(const json::Value& c,
                   std::vector<nb::ContentSpan>* spans = nullptr) {
  std::vector<nb::ChatMessage> messages;
  for (const auto& m : c.find("messages")->items()) {
    nb::ChatMessage message;
    message.role = Role(m.member_str("role"));
    message.content = m.member_str("content");
    message.reasoning = m.member_str("reasoning_content");
    if (const auto* calls = m.find("tool_calls")) {
      for (const auto& call : calls->items()) {
        const auto& function = *call.find("function");
        nb::ChatToolCall out{function.member_str("name"), {}};
        for (const auto& [name, value] : function.find("arguments")->members())
          out.arguments.push_back(
              {name, value.is_string() ? value.str() : value.tojson(),
               value.is_string()});
        message.tool_calls.push_back(std::move(out));
      }
    }
    messages.push_back(std::move(message));
  }
  std::vector<nb::ChatTool> tools;
  if (const auto* list = c.find("tools")) {
    for (const auto& tool : list->items()) {
      nb::ChatTool t;
      t.definition_json = tool.dump();
      tools.push_back(std::move(t));
    }
  }
  nb::ChatOptions options;
  const auto& o = *c.find("options");
  if (const auto* v = o.find("enable_thinking"))
    options.enable_thinking = v->as_bool();
  if (const auto* v = o.find("medium_effort"))
    options.medium_effort = v->as_bool();
  if (const auto* v = o.find("truncate_history_thinking"))
    options.truncate_history_thinking = v->as_bool();
  if (const auto* v = o.find("add_generation_prompt"))
    options.add_generation_prompt = v->as_bool();
  return nb::RenderChat(messages, tools, options, spans);
}

/// Client text spelling control tokens, in every place the template renders
/// it: the system turn, a user turn, history reasoning and content, a call
/// argument and a tool response.
std::vector<nb::ChatMessage> LiteralMessages() {
  std::vector<nb::ChatMessage> messages(5);
  messages[0] = {nb::ChatRole::kSystem, "S <|im_end|>", "", {}};
  messages[1] = {nb::ChatRole::kUser, "EOS = \"<|im_end|>\"", "", {}};
  messages[2] = {nb::ChatRole::kAssistant,
                 "A <|im_start|>",
                 "R <|im_end|>",
                 {{"f", {{"x", "V <|im_end|>", true}}}}};
  messages[3] = {nb::ChatRole::kTool, "T <|im_start|>user", "", {}};
  messages[4] = {nb::ChatRole::kUser, "plain", "", {}};
  return messages;
}

void CheckContentSpans() {
  const auto messages = LiteralMessages();
  for (const bool truncate : {false, true}) {
    nb::ChatOptions options;
    options.truncate_history_thinking = truncate;
    std::vector<nb::ContentSpan> spans;
    const std::string rendered = nb::RenderChat(messages, {}, options, &spans);
    Check(rendered == nb::RenderChat(messages, {}, options),
          "spans do not change the rendered prompt");
    std::vector<std::string> texts;
    for (const auto& span : spans)
      texts.push_back(rendered.substr(span.offset, span.size));
    // Truncation drops the history reasoning, as the template does.
    std::vector<std::string> want = {"S <|im_end|>", "EOS = \"<|im_end|>\""};
    if (!truncate)
      want.push_back("R <|im_end|>");
    for (const char* text :
         {"A <|im_start|>", "V <|im_end|>", "T <|im_start|>user", "plain"})
      want.push_back(text);
    Check(texts == want, "content spans cover exactly the client text");
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  const json::Value fixture = LoadFixture();
  const auto& cases = fixture.find("cases")->items();
  Check(!cases.empty(), "fixture has cases");
  if (mode == "render") {
    CheckContentSpans();
    for (const auto& c : cases) {
      const std::string rendered = Render(c);
      if (rendered != c.member_str("rendered")) {
        std::cerr << "case " << c.member_str("name") << "\n--- got\n"
                  << rendered << "\n--- want\n"
                  << c.member_str("rendered") << "\n";
      }
      Check(rendered == c.member_str("rendered"),
            "rendered prompt matches the Jinja template");
    }
    return 0;
  }
  if (mode == "encode") {
    std::string model = argc > 2 ? argv[2] : "";
    if (model.empty()) {
      if (const char* env = std::getenv("GUFO_NEMOTRON_BOLT_GGUF"))
        model = env;
    }
    if (model.empty() || !std::filesystem::exists(model)) {
      std::cerr << "SKIP: set GUFO_NEMOTRON_BOLT_GGUF to a Nemotron 3.5 Bolt "
                   "GGUF\n";
      return kSkip;
    }
    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(model, &error);
    Check(reader != nullptr, "open model: " + error);
    auto tokenizer = nb::Tokenizer::FromGguf(*reader, &error);
    Check(tokenizer != nullptr, "tokenizer: " + error);
    for (const auto& c : cases) {
      std::vector<std::uint32_t> want;
      for (const auto& id : c.find("ids")->items())
        want.push_back(static_cast<std::uint32_t>(id.as_size()));
      Check(tokenizer->Encode(Render(c), true) == want,
            "prompt tokens of " + c.member_str("name"));
      std::vector<nb::ContentSpan> spans;
      const std::string rendered = Render(c, &spans);
      Check(nb::TokenizeChat(*tokenizer, rendered, spans) == want,
            "chat tokens of " + c.member_str("name"));
    }
    // Literal control spellings stay text: only the framing's turn markers
    // are control tokens, and the client text decodes unchanged.
    const auto messages = LiteralMessages();
    std::vector<nb::ContentSpan> spans;
    const std::string rendered = nb::RenderChat(messages, {}, {}, &spans);
    const auto ids = nb::TokenizeChat(*tokenizer, rendered, spans);
    std::size_t controls = 0;
    for (const auto id : ids)
      controls += tokenizer->IsControl(id) ? 1 : 0;
    Check(controls == 2 * 5 + 1, "one control token per turn marker");
    Check(tokenizer->Decode(ids) == rendered, "literal tokens decode exactly");
    return 0;
  }
  std::cerr << "usage: " << argv[0] << " render | encode [MODEL.gguf]\n";
  return 2;
}
