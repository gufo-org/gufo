// Checks the tokenizer against Hugging Face goldens
// (tools/models/nemotron35_bolt/tokenizer_golden.py).
//
//   nemotron35_bolt_tokenizer_test pretokenizer
//       Split byte ranges; needs no weights.
//   nemotron35_bolt_tokenizer_test encode MODEL.gguf
//       Token IDs with and without control tokens, and decode round trips.
//       Any GGUF with this vocabulary works, including the LiLiCorr drafter.
//       Without a model path (argument or GUFO_NEMOTRON_BOLT_GGUF) it skips.

#include "src/models/nemotron35_bolt/tokenizer.hpp"

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

#ifndef GUFO_NEMOTRON35_BOLT_TOKENIZER_GOLDEN
#define GUFO_NEMOTRON35_BOLT_TOKENIZER_GOLDEN \
  "tests/fixtures/nemotron35_bolt/tokenizer_golden.json"
#endif

namespace nb = gufo::models::nemotron35_bolt;
namespace json = gufo::json;

namespace {

constexpr int kSkip = 77;

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL nemotron35_bolt_tokenizer_test: " << message << '\n';
    std::exit(1);
  }
}

// The compiled-in fixture path exists only where the tree was built. A packaged
// test image points GUFO_FIXTURE_DIR at a directory holding the fixtures.
json::Value LoadFixture() {
  std::filesystem::path path = GUFO_NEMOTRON35_BOLT_TOKENIZER_GOLDEN;
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

std::vector<std::uint32_t> Ids(const json::Value& array) {
  std::vector<std::uint32_t> out;
  for (const auto& v : array.items())
    out.push_back(static_cast<std::uint32_t>(v.as_size()));
  return out;
}

void CheckPreTokenizer(const json::Value& fixture) {
  for (const auto& c : fixture.find("cases")->items()) {
    const std::string text = c.member_str("text");
    const auto pieces = nb::PreTokenize(text);
    const auto& expected = c.find("pieces")->items();
    bool same = pieces.size() == expected.size();
    for (std::size_t i = 0; same && i < pieces.size(); ++i) {
      const auto begin =
          static_cast<std::size_t>(pieces[i].data() - text.data());
      same = begin == expected[i].items()[0].as_size() &&
             begin + pieces[i].size() == expected[i].items()[1].as_size();
    }
    if (!same) {
      std::cerr << "pieces of \"" << text << "\":";
      for (const auto piece : pieces)
        std::cerr << " [" << piece << "]";
      std::cerr << '\n';
    }
    Check(same, "pre-tokenizer pieces match Hugging Face");
  }
}

int CheckEncode(const json::Value& fixture, const std::string& model) {
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model, &error);
  Check(reader != nullptr, "open " + model + ": " + error);
  auto tokenizer = nb::Tokenizer::FromGguf(*reader, &error);
  Check(tokenizer != nullptr, "tokenizer loads: " + error);
  Check(tokenizer->VocabSize() == 131072, "vocabulary size");
  Check(tokenizer->Eos() == 11 && tokenizer->IsStop(11) && tokenizer->IsStop(2),
        "<|im_end|> and </s> stop generation");
  Check(tokenizer->IsControl(10) && !tokenizer->IsControl(12),
        "<|im_start|> is a control token, <think> a plain added token");
  Check(tokenizer->AddedToken("<tool_call>") == 14U, "<tool_call> id");

  for (const auto& c : fixture.find("cases")->items()) {
    const std::string text = c.member_str("text");
    const auto ids = tokenizer->Encode(text, true);
    const auto plain = tokenizer->Encode(text, false);
    if (ids != Ids(*c.find("ids")) || plain != Ids(*c.find("ids_plain"))) {
      std::cerr << "text: " << text << "\ngot:";
      for (auto id : ids)
        std::cerr << ' ' << id;
      std::cerr << '\n';
    }
    Check(ids == Ids(*c.find("ids")), "token IDs match Hugging Face");
    Check(plain == Ids(*c.find("ids_plain")),
          "IDs with control tokens as text match Hugging Face");
    Check(tokenizer->Decode(ids) == text && tokenizer->Decode(plain) == text,
          "decoding restores the text");
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  const json::Value fixture = LoadFixture();
  Check(fixture.find("cases") != nullptr && fixture.find("cases")->size() > 0,
        "fixture has cases");
  if (mode == "pretokenizer") {
    CheckPreTokenizer(fixture);
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
    return CheckEncode(fixture, model);
  }
  std::cerr << "usage: " << argv[0] << " pretokenizer | encode [MODEL.gguf]\n";
  return 2;
}
