// While a layer holds at most DS4_N_INDEXER_TOP_K compressed rows, the indexer
// selects every row, so logits cannot catch a wrong indexer compressor row.
// The session snapshot's target sections can.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/deepseek_v4_flash/engine.hpp"
#include "src/models/deepseek_v4_flash/runtime/dspark_internal.h"
#include "src/models/deepseek_v4_flash/runtime/model_data_internal.h"

namespace {

using gufo::core::SessionMode;
using gufo::models::deepseek_v4_flash::Model;
using gufo::models::deepseek_v4_flash::ModelOptions;
using gufo::models::deepseek_v4_flash::Session;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

struct PayloadSection {
  std::string name;
  std::span<const std::uint8_t> bytes;
};

std::vector<PayloadSection> TargetSections(
    std::span<const std::uint8_t> payload) {
  const auto word = [&](std::size_t index) {
    std::uint32_t value = 0;
    std::memcpy(&value, payload.data() + index * sizeof(value), sizeof(value));
    return value;
  };
  constexpr std::size_t kHeaderWords = 19 + DS4_DSPARK_MAX_TARGET_LAYERS;
  constexpr std::size_t kRequestStateBytes = 12 * 8 + 8 * 4;
  const std::uint32_t tokens = word(7);
  const std::uint32_t raw_rows = word(12);
  std::size_t at = kHeaderWords * 4 + kRequestStateBytes;
  std::vector<PayloadSection> sections;
  const auto add = [&](std::string name, std::size_t bytes) {
    Expect(at + bytes <= payload.size(), "snapshot holds " + name);
    sections.push_back({std::move(name), payload.subspan(at, bytes)});
    at += bytes;
  };
  add("tokens", std::size_t{tokens} * 4);
  add("logits", std::size_t{DS4_N_VOCAB} * 4);
  const std::size_t comp_counts = at / 4;
  add("n_comp", DS4_N_LAYER * 4);
  const std::size_t index_counts = at / 4;
  add("n_index_comp", DS4_N_LAYER * 4);
  for (std::uint32_t il = 0; il < DS4_N_LAYER; ++il) {
    const std::string layer = "L" + std::to_string(il) + ".";
    add(layer + "raw", std::size_t{raw_rows} * DS4_N_HEAD_DIM * 4);
    const std::uint32_t ratio = ds4_layer_compress_ratio(il);
    if (ratio == 0)
      continue;
    const std::size_t coff = ratio == 4 ? 2 : 1;
    const std::size_t attn_state = coff * DS4_N_HEAD_DIM * coff * ratio * 4;
    add(layer + "attn_comp",
        std::size_t{word(comp_counts + il)} * DS4_N_HEAD_DIM * 4);
    add(layer + "attn_state_kv", attn_state);
    add(layer + "attn_state_score", attn_state);
    if (ratio != 4)
      continue;
    const std::size_t index_state =
        coff * DS4_N_INDEXER_HEAD_DIM * coff * ratio * 4;
    add(layer + "index_comp",
        std::size_t{word(index_counts + il)} * DS4_N_INDEXER_HEAD_DIM * 4);
    add(layer + "index_state_kv", index_state);
    add(layer + "index_state_score", index_state);
  }
  return sections;
}

std::vector<std::uint8_t> SnapshotBytes(const Session& session) {
  std::string error;
  const auto snapshot = session.SaveSnapshot(&error);
  Expect(snapshot != nullptr, error);
  return {snapshot->bytes().begin(), snapshot->bytes().end()};
}

std::string TargetStateDifference(const Session& ar, const Session& dspark) {
  const auto ar_bytes = SnapshotBytes(ar);
  const auto dspark_bytes = SnapshotBytes(dspark);
  const auto expected = TargetSections(ar_bytes);
  const auto actual = TargetSections(dspark_bytes);
  Expect(expected.back().bytes.data() + expected.back().bytes.size() ==
             ar_bytes.data() + ar_bytes.size(),
         "the AR snapshot ends after its target sections");
  std::string difference;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto& a = expected[i].bytes;
    const auto& b = actual[i].bytes;
    if (a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size()) == 0)
      continue;
    std::size_t words = 0;
    for (std::size_t at = 0; at < std::min(a.size(), b.size()); at += 4)
      words += std::memcmp(a.data() + at, b.data() + at, 4) != 0;
    difference += (difference.empty() ? "" : ",") + expected[i].name + ":" +
                  std::to_string(words) + "/" + std::to_string(a.size() / 4);
  }
  return difference;
}

constexpr std::size_t kGeneratedTokens = 32;

}  // namespace

int main() try {
  std::cout << std::unitbuf;
  const char* model_path = std::getenv("GUFO_DEEPSEEK_V4_FLASH_MODEL");
  const char* dspark_path = std::getenv("GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL");
  if (model_path == nullptr || model_path[0] == '\0' ||
      dspark_path == nullptr || dspark_path[0] == '\0') {
    std::cout << "SKIP: GUFO_DEEPSEEK_V4_FLASH_MODEL and "
                 "GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL must both be set\n";
    return 77;
  }
  std::string error;
  const auto model = Model::Load(
      model_path, ModelOptions{.dspark_model_path = dspark_path}, &error);
  Expect(model != nullptr && model->HasDspark(), error);

  const auto prompt = model->EncodeChat(
      "",
      "Continue this technical explanation with concrete examples and "
      "tradeoffs: Virtual memory lets an operating system");
  auto ar = model->CreateSession(SessionMode::kAutoregressive, 4096, &error);
  auto dspark = model->CreateSession(SessionMode::kSpeculative, 4096, &error);
  Expect(ar && dspark, error);
  Expect(ar->Sync(prompt, &error) && dspark->Sync(prompt, &error), error);
  const std::string prefill = TargetStateDifference(*ar, *dspark);
  Expect(prefill.empty(),
         "AR and DSpark prefill leave different target state: " + prefill);

  std::size_t generated = 0;
  while (generated < kGeneratedTokens) {
    std::vector<int> emitted;
    Expect(
        dspark->DsparkStep(kGeneratedTokens - generated, 7, &emitted, &error),
        error);
    Expect(!emitted.empty(), "DSpark step emits a token");
    for (const int token : emitted) {
      const int expected = ar->SelectNext(0.0F, nullptr);
      Expect(token == expected, "DSpark token " + std::to_string(generated) +
                                    " is " + std::to_string(token) +
                                    ", AR chose " + std::to_string(expected));
      Expect(ar->Evaluate(token, &error), error);
      ++generated;
    }
    const std::string state = TargetStateDifference(*ar, *dspark);
    Expect(state.empty(),
           "after " + std::to_string(generated) + " tokens at position " +
               std::to_string(dspark->Position()) +
               " the DSpark target state differs from AR: " + state);
  }
  const auto stats = dspark->DsparkStatistics();
  Expect(stats.support_drafted > 0, "DSpark verified drafted tokens");
  std::cout << "DSpark target state matches AR through " << generated
            << " tokens at position " << dspark->Position() << " ("
            << stats.steps << " steps, " << stats.support_accepted << "/"
            << stats.support_drafted << " drafts accepted)\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "FAIL: " << error.what() << '\n';
  return EXIT_FAILURE;
}
