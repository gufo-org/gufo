// CPU contracts of the YuE2 port: the torch-compatible noise, the sampler's
// masks and budgets, the song protocol and, when the checkpoint is present,
// the tokenizer against upstream encodings.
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "src/models/yue2/noise.hpp"
#include "src/models/yue2/protocol.hpp"
#include "src/models/yue2/sampler.hpp"
#include "src/models/yue2/tokenizer.hpp"

namespace yue2 = gufo::models::yue2;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void TestNoise() {
  // torch.randn(80, generator=torch.Generator().manual_seed(seed)).
  struct Case {
    std::uint64_t seed;
    float values[8];
  };
  const Case cases[] = {
      {831001,
       {0.524257F, -0.655141F, 0.833387F, -1.533694F, 0.861342F, 0.689913F,
        -0.285379F, 0.487814F}},
      {7,
       {-0.820135F, 0.395631F, -0.893655F, 0.926543F, 1.090165F, 0.754193F,
        0.13324F, 0.59203F}},
  };
  const int indices[] = {0, 1, 7, 8, 15, 63, 64, 79};
  for (const Case& c : cases) {
    const std::vector<float> noise = yue2::TorchRandn(80, c.seed);
    for (int i = 0; i < 8; ++i)
      Expect(std::abs(noise[indices[i]] - c.values[i]) < 2e-6F,
             "torch randn value " + std::to_string(indices[i]) + " of seed " +
                 std::to_string(c.seed));
  }
}

std::vector<float> Window(yue2::Phase phase, float fill) {
  return std::vector<float>(yue2::PhaseWindow(phase).size(), fill);
}

void TestSampler() {
  using yue2::Phase;
  // Greedy: the end token is banned below min_tokens, then wins.
  yue2::Sampling greedy{.temperature = 0.0F,
                        .top_p = 1.0F,
                        .top_k = 1,
                        .repetition_penalty = 1.0F,
                        .penalty_window = 1,
                        .min_tokens = 2,
                        .max_tokens = 10};
  yue2::TokenSampler sampler(Phase::kSemantic, greedy, 1, false);
  std::vector<float> logits = Window(Phase::kSemantic, 0.0F);
  logits[0] = 5.0F;  // </music>
  logits[10] = 1.0F;
  Expect(sampler.Next(logits, {}, 1.0F) == yue2::kMusicEnd + 10,
         "end token banned before min_tokens");
  Expect(sampler.Next(logits, {}, 1.0F) == yue2::kMusicEnd + 10,
         "end token still banned at step 1");
  Expect(sampler.Next(logits, {}, 1.0F) == yue2::kMusicEnd,
         "end token allowed after min_tokens");

  // Planning masks the control rows between the text vocabulary and </abc>.
  yue2::TokenSampler planner(Phase::kAbc, greedy, 1, false);
  std::vector<float> text = Window(Phase::kAbc, 0.0F);
  text[yue2::kEndOfDocument + 3] = 9.0F;
  text[42] = 1.0F;
  Expect(planner.Next(text, {}, 1.0F) == 42, "planning masks control rows");

  // Repetition penalty divides a positive repeated logit.
  yue2::Sampling penalised = greedy;
  penalised.repetition_penalty = 2.0F;
  penalised.penalty_window = 4;
  penalised.min_tokens = 0;
  yue2::TokenSampler repeat(Phase::kSemantic, penalised, 1, false);
  std::vector<float> two = Window(Phase::kSemantic, -10.0F);
  two[5] = 3.0F;
  two[6] = 2.0F;
  Expect(repeat.Next(two, {}, 1.0F) == yue2::kMusicEnd + 5, "first pick");
  Expect(repeat.Next(two, {}, 1.0F) == yue2::kMusicEnd + 6,
         "penalty demotes the repeated token");

  // Top-k 1 sampling always returns the maximum; the seed is irrelevant.
  yue2::Sampling top1 = yue2::DefaultSemanticSampling();
  top1.top_k = 1;
  top1.min_tokens = 0;
  for (std::uint64_t seed = 0; seed < 4; ++seed) {
    yue2::TokenSampler s(Phase::kSemantic, top1, seed, false);
    std::vector<float> row = Window(Phase::kSemantic, 0.0F);
    row[77] = 4.0F;
    Expect(s.Next(row, {}, 1.0F) == yue2::kMusicEnd + 77, "top-k 1");
  }
  // Same seed, same logits, same draws.
  yue2::Sampling defaults = yue2::DefaultSemanticSampling();
  defaults.min_tokens = 0;
  std::vector<float> spread = Window(Phase::kSemantic, 0.0F);
  for (std::size_t i = 0; i < spread.size(); ++i)
    spread[i] = static_cast<float>((i * 2654435761U) % 1000) / 200.0F;
  yue2::TokenSampler a(Phase::kSemantic, defaults, 9, false);
  yue2::TokenSampler b(Phase::kSemantic, defaults, 9, false);
  for (int i = 0; i < 16; ++i)
    Expect(a.Next(spread, {}, 1.0F) == b.Next(spread, {}, 1.0F),
           "deterministic draws");
}

void TestProtocol() {
  const auto one = yue2::ChunkRanges(300, 715);
  Expect(one.size() == 1 && one[0].first == 0 && one[0].second == 300,
         "one chunk for a short song");
  const auto two = yue2::ChunkRanges(9000, 8000);
  Expect(two.size() == 2 && two[0].second == (24576 - 8000 - 3) / 2,
         "historical chunk cut");
  Expect(yue2::DefaultGuidance(yue2::Planning::kOff) == 1.01F,
         "planning-off guidance");
  yue2::Sampling invalid = yue2::DefaultAbcSampling();
  invalid.min_tokens = invalid.max_tokens + 1;
  Expect(!yue2::ValidateSampling(invalid, nullptr), "min above max rejected");
}

void TestTokenizer() {
  const char* root = std::getenv("YUE2_MODEL_DIR");
  const std::filesystem::path ranks =
      std::filesystem::path(root != nullptr ? root : "models/YuE2-3B") /
      "qwen.tiktoken";
  if (!std::filesystem::exists(ranks)) {
    std::cout << "SKIP tokenizer: " << ranks << " not found\n";
    return;
  }
  std::string error;
  const auto tokenizer = yue2::TextTokenizer::Load(ranks, &error);
  Expect(tokenizer != nullptr, "load tokenizer: " + error);
  if (!tokenizer)
    return;
  // Upstream `encode_ordinary` ids.
  Expect(tokenizer->Encode("") == std::vector<std::uint32_t>{}, "empty text");
  const auto ids = tokenizer->Encode("Hello world");
  Expect(ids == std::vector<std::uint32_t>{9707, 1879}, "Hello world");
  Expect(tokenizer->Decode(ids) == "Hello world", "decode round trip");
  const auto special = tokenizer->Encode("<|endoftext|>");
  Expect(special.size() > 1, "special text is ordinary text");
}

}  // namespace

int main() {
  TestNoise();
  TestSampler();
  TestProtocol();
  TestTokenizer();
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "yue2_cpu_test: ok\n";
  return 0;
}
