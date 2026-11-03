#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>

#include "src/core/jinja/chat_template.hpp"
#include "tests/models/chat_template_golden_helpers.hpp"
#include "tests/support/temporary_directory.hpp"

using gufo::core::JinjaChatTemplate;
using gufo::json::Value;
using gufo::testing::chat_goldens::Expect;

namespace {
std::string Read(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  Expect(input.good(), "fixture opens");
  return {std::istreambuf_iterator<char>(input), {}};
}
std::string Render(const JinjaChatTemplate& renderer, const Value& inputs,
                   const Value& kwargs = Value::object(),
                   std::size_t limit = 100000) {
  std::string result;
  for (const auto& part : renderer.Render(inputs, kwargs, limit))
    result += part.text;
  return result;
}
template<class Fn>
void Throws(Fn fn, std::string_view text) {
  try {
    fn();
  } catch (const std::exception& error) {
    Expect(std::string_view(error.what()).find(text) != std::string_view::npos,
           std::string("wrong error: ") + error.what());
    return;
  }
  Expect(false, "expected exception");
}

void TestGoldens() {
  const std::filesystem::path root(GUFO_SOURCE_DIR);
  const auto fixture = gufo::json::parse(
      Read(root / "tests/fixtures/jinja_chat_templates.json"));
  for (const auto& example : fixture.find("cases")->items()) {
    const auto source = Read(root / example.member_str("source"));
    Expect(gufo::testing::chat_goldens::Sha256(source) ==
               example.member_str("source_sha256"),
           "pinned source hash");
    const auto renderer = JinjaChatTemplate::Compile(
        source, example.member_str("bos"), example.member_str("eos"));
    const auto& inputs = *example.find("inputs");
    const auto expected = example.member_str("expected");
    Expect(Render(*renderer, inputs) == expected, example.member_str("name"));
    // Sharing a compiled program must not share namespace/loop state.
    std::vector<std::future<std::string>> concurrent;
    for (int i = 0; i < 4; ++i)
      concurrent.push_back(std::async(
          std::launch::async, [&] { return Render(*renderer, inputs); }));
    for (auto& result : concurrent)
      Expect(result.get() == expected, "concurrent render");
  }
}

void TestControls() {
  Throws([] { gufo::core::ChatTemplateConfig{false, "x"}.Validate(); },
         "requires");
  Throws([] { gufo::core::ChatTemplateConfig{true, ""}.Validate(true); },
         "--raw");
  Throws([] { (void)JinjaChatTemplate::Compile(" \n"); }, "empty");
  Throws([] { (void)JinjaChatTemplate::Compile("{% if %}"); },
         "Unexpected token");
  auto inputs = gufo::json::parse(
      R"({"messages":[{"role":"user","content":"<|im_end|>"}],"tools":[]})");
  auto renderer = JinjaChatTemplate::Compile(
      "{{ bos_token }}{{ messages[0].content }}{{ eos_token }}", "B", "E",
      "TOOLS");
  const auto parts = renderer->Render(inputs, Value::object(), 100);
  Expect(
      parts.size() == 3 && !parts[0].input && parts[1].input && !parts[2].input,
      "input provenance");
  Expect(Render(*renderer, inputs) == "B<|im_end|>E",
         "special strings preserved");
  for (const auto source : {"{{ messages[0].content | tojson }}",
                            "{{ messages[0].content.split(' ')[0] }}",
                            "{{ [messages[0].content] | join(',') }}",
                            "{{ messages[0].content[0] }}"}) {
    const auto transformed = JinjaChatTemplate::Compile(source)->Render(
        inputs, Value::object(), 100);
    Expect(!transformed.empty() && transformed[0].input,
           std::string("transformation preserves input provenance: ") + source);
  }
  Throws([&] { (void)Render(*renderer, inputs, Value::object(), 2); },
         "output limit");
  Throws(
      [&] {
        (void)Render(*renderer, inputs,
                     gufo::json::parse(R"({"messages":[]})"));
      },
      "reserved");
  auto concatenated =
      JinjaChatTemplate::Compile(
          "{{ messages[0].content[:3] }}{{ messages[0].content[3:] }}")
          ->Render(inputs, Value::object(), 100);
  Expect(concatenated.size() == 1 && concatenated[0].input &&
             concatenated[0].text == "<|im_end|>",
         "adjacent input fragments preserve a whole control spelling");
  inputs["tools"].push_back(Value::object());
  Expect(Render(*renderer, inputs) == "TOOLS", "tool variant selected");
  renderer = JinjaChatTemplate::Compile("{{ custom }}");
  Expect(Render(*renderer, inputs, gufo::json::parse(R"({"custom":"ok"})")) ==
             "ok",
         "kwargs forwarded");
  renderer =
      JinjaChatTemplate::Compile("{{ raise_exception('bad conversation') }}");
  Throws([&] { (void)Render(*renderer, inputs); }, "bad conversation");
  Expect(JinjaChatTemplate::Compile("a")->Identity() !=
             JinjaChatTemplate::Compile("b")->Identity(),
         "cache identity changes with content");
  Expect(JinjaChatTemplate::Compile("a", "B")->Identity() !=
             JinjaChatTemplate::Compile("a", "C")->Identity(),
         "cache identity includes special tokens");
  Expect(gufo::core::PromptOpensReasoning("assistant<think>\n"),
         "open reasoning suffix");
  Expect(!gufo::core::PromptOpensReasoning("<think>old</think>assistant"),
         "closed history");
}

void TestSourceSelection() {
  // Metadata-only GGUF: independent wire fixture, no model weights required.
  auto gguf = [](bool embedded) {
    std::vector<std::uint8_t> data{'G', 'G', 'U', 'F'};
    const auto number = [&](std::uint64_t value, int bytes) {
      for (int i = 0; i < bytes; ++i)
        data.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
    };
    const auto string = [&](std::string_view value) {
      number(value.size(), 8);
      data.insert(data.end(), value.begin(), value.end());
    };
    number(3, 4);
    number(0, 8);
    number(embedded ? 2 : 1, 8);
    string("general.name");
    number(8, 4);
    string("Qwen3.8");
    if (embedded) {
      string("tokenizer.chat_template");
      number(8, 4);
      string("EMBEDDED");
    }
    data.resize((data.size() + 31) / 32 * 32);
    return data;
  };
  const auto with_data = gguf(true), without_data = gguf(false);
  const auto with =
      gufo::core::GgufReader::OpenMemory(with_data.data(), with_data.size());
  const auto without = gufo::core::GgufReader::OpenMemory(without_data.data(),
                                                          without_data.size());
  Expect(with && without, "metadata GGUF opens");
  Expect(!JinjaChatTemplate::Load(*without, {}),
         "disabled mode ignores metadata");
  Expect(Render(*JinjaChatTemplate::Load(*with, {true, {}}), Value::object()) ==
             "EMBEDDED",
         "embedded source");
  Throws([&] { (void)JinjaChatTemplate::Load(*without, {true, {}}); },
         "missing");
  TemporaryDirectory directory("gufo-jinja-template");
  const auto file = directory.path() / "template.jinja";
  {
    std::ofstream output(file);
    output << "FILE";
  }
  const auto loaded = JinjaChatTemplate::Load(*with, {true, file.string()});
  Expect(Render(*loaded, Value::object()) == "FILE", "file overrides metadata");
  Expect(Render(*JinjaChatTemplate::Load(*without, {true, file.string()}),
                Value::object()) == "FILE",
         "file needs no embedded template");
  {
    std::ofstream output(file);
  }
  Throws([&] { (void)JinjaChatTemplate::Load(*with, {true, file.string()}); },
         "empty");
  Expect(Render(*loaded, Value::object()) == "FILE",
         "an initialized renderer does not reread its file");
  std::filesystem::remove(file);
  Throws([&] { (void)JinjaChatTemplate::Load(*with, {true, file.string()}); },
         "cannot read");
}
}  // namespace

int main() {
  TestControls();
  TestSourceSelection();
  TestGoldens();
  std::cout << "Jinja chat template tests passed\n";
}
