// Model-backed host checks on the real Gemma 4 artifacts: metadata, tensor
// binding, template pin and tokenizer parity with the recorded references.
// Set GUFO_GEMMA4_MODEL (and optionally GUFO_GEMMA4_MTP_MODEL); without it
// the test exits 77, which is a skip and not a pass.
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/gemma4/chat_template.hpp"
#include "src/models/gemma4/tokenizer.hpp"
#include "src/models/gemma4/weights.hpp"
#include "tests/models/gemma4/check.hpp"
#include "tests/models/gemma4/template_cases.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;

namespace {

std::string FromHex(const std::string& hex) {
  std::string out;
  for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
    out.push_back(static_cast<char>(std::stoi(hex.substr(i, 2), nullptr, 16)));
  }
  return out;
}

std::string IdsSha256(const std::vector<g4::TokenId>& ids) {
  std::vector<std::uint8_t> bytes;
  for (auto id : ids) {
    const auto v = static_cast<std::uint32_t>(id);
    for (int b = 0; b < 4; ++b) {
      bytes.push_back(static_cast<std::uint8_t>(v >> (8 * b)));
    }
  }
  return gufo::crypto::Sha256Hex(bytes);
}

void CheckTarget(const gufo::core::GgufReader& reader,
                 const g4::ModelWeights& w) {
  const auto& c = w.config;
  Require(c.num_layers == 60 && c.hidden_size == 5376 && c.ffn_size == 21504,
          "31B dimensions");
  Require(c.num_heads == 32 && c.GlobalLayerCount() == 10, "attention layout");
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const bool global = (l + 1) % 6 == 0;
    Require(c.IsSliding(l) != global,
            "sliding pattern at layer " + std::to_string(l));
    Require(c.kv_heads[l] == (global ? 4U : 16U), "KV heads");
    Require(w.layers[l].attn_v.empty() == global, "K=V on global layers");
  }
  Require(c.head_dim_global == 512 && c.head_dim_sliding == 256 &&
              c.sliding_window == 1024 && c.final_logit_softcap == 30.0F,
          "attention constants");
  Require(w.vocab_size == 262144 && w.TiedOutput(), "tied 262144 vocabulary");
  std::size_t rotating = 0;
  for (float f : w.rope_factors.values) {
    rotating += f == 1.0F ? 1 : 0;
  }
  Require(rotating == 64, "64 rotating global rope pairs");
  Require(w.layers[0].output_scale > 0.0F && w.layers[0].output_scale < 1.0F,
          "layer output scale read");
  std::string error;
  Require(g4::ChatTemplate::ValidateGgufTemplate(reader, &error), error);
}

void CheckTokenizer(const g4::Tokenizer& tokenizer) {
  Require(tokenizer.VocabSize() == 262144, "vocabulary size");
  Require(tokenizer.BosToken() == 2 && tokenizer.EosToken() == 106,
          "BOS/EOS ids");
  const std::vector<g4::TokenId> eog(tokenizer.EndOfGenerationTokens().begin(),
                                     tokenizer.EndOfGenerationTokens().end());
  Require(eog == std::vector<g4::TokenId>({1, 50, 106}),
          "end-of-generation set {<eos>, <|tool_response>, <turn|>}");

  const auto goldens = gemma4_test::ReadJson(std::string(GUFO_GEMMA4_FIXTURES) +
                                             "/tokenizer_goldens.json");
  std::size_t checked = 0;
  for (const auto& entry : goldens.find("entries")->items()) {
    const std::string name = entry.member_str("name");
    const std::string text = FromHex(entry.member_str("text_hex"));
    const auto ids = tokenizer.Encode(text, false, true);
    if (const auto* expected = entry.find("llama_cpp_ids")) {
      std::vector<g4::TokenId> want;
      for (const auto& id : expected->items()) {
        want.push_back(static_cast<g4::TokenId>(id.as_double()));
      }
      if (ids != want) {
        std::cerr << name << " got:";
        for (auto id : ids)
          std::cerr << ' ' << id;
        std::cerr << "\n" << name << " want:";
        for (auto id : want)
          std::cerr << ' ' << id;
        std::cerr << '\n';
      }
      Require(ids == want, "token ids differ from llama.cpp: " + name);
    } else {
      Require(ids.size() == entry.member_size("llama_cpp_count") &&
                  IdsSha256(ids) == entry.member_str("llama_cpp_ids_sha256"),
              "token ids differ from llama.cpp: " + name);
    }
    // Valid text round-trips through decoding with special tokens shown.
    if (entry.member_str("hf") == "agrees") {
      Require(tokenizer.Decode(ids, true) == text,
              "decode does not round-trip: " + name);
    }
    ++checked;
  }
  Require(checked >= 20, "tokenizer corpus incomplete");
}

/// Rendered golden prompts tokenize to stable ids; the prompt starts with a
/// single BOS emitted by the template.
void CheckRenderedPrompts(const g4::Tokenizer& tokenizer) {
  const auto cases = gemma4_test::LoadTemplateCases(
      std::string(GUFO_GEMMA4_FIXTURES) + "/template_cases.json");
  for (const auto& c : cases) {
    std::string error;
    std::vector<std::size_t> images;
    const auto rendered = g4::ChatTemplate::Render(c.messages, c.tools,
                                                   c.options, &error, &images);
    Require(rendered.has_value(), c.name + ": " + error);
    const auto ids = tokenizer.Encode(rendered->text, false, true);
    Require(!ids.empty() && ids[0] == tokenizer.BosToken() &&
                (ids.size() < 2 || ids[1] != tokenizer.BosToken()),
            c.name + ": exactly one leading BOS");
  }
}

void CheckDraft(const std::string& path, const g4::ModelWeights& target) {
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  Require(reader != nullptr, error);
  const auto d = g4::DraftWeights::Bind(*reader, target, &error);
  Require(d.has_value(), "drafter rejected: " + error);
  Require(d->config.num_layers == 4 && d->config.hidden_size == 1024 &&
              d->config.target_hidden_size == 5376,
          "drafter dimensions");
  Require(d->config.SharedKvSource(0, target.config) == 58 &&
              d->config.SharedKvSource(3, target.config) == 59,
          "drafter reads target layers 58 (sliding) and 59 (global)");
}

}  // namespace

int main() {
  const char* model = std::getenv("GUFO_GEMMA4_MODEL");
  if (model == nullptr || *model == '\0') {
    std::cout << "SKIP: GUFO_GEMMA4_MODEL is not set\n";
    return 77;
  }
  return gemma4_test::Run([&] {
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(model, &error);
    Require(reader != nullptr, error);
    const auto weights = g4::ModelWeights::Bind(*reader, &error);
    Require(weights.has_value(), "target rejected: " + error);
    CheckTarget(*reader, *weights);
    const auto tokenizer = g4::Tokenizer::CreateFromGguf(*reader, &error);
    Require(tokenizer != nullptr, "tokenizer rejected: " + error);
    CheckTokenizer(*tokenizer);
    CheckRenderedPrompts(*tokenizer);
    if (const char* draft = std::getenv("GUFO_GEMMA4_MTP_MODEL");
        draft != nullptr && *draft != '\0') {
      CheckDraft(draft, *weights);
    } else {
      std::cout << "note: GUFO_GEMMA4_MTP_MODEL not set; drafter unchecked\n";
    }
  });
}
