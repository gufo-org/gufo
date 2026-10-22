// Compares GPU embeddings of text with sentence-transformers output for
// google/embeddinggemma-2. The float32 reference fixture comes from
// tools/models/embedding_gemma2/reference_embeddings.py. Token ids must match
// exactly, and embeddings must agree within the model's own BF16 noise.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/embedding_gemma2/hip/encoder_runtime.hpp"
#include "src/models/embedding_gemma2/tokenizer.hpp"

#ifndef GUFO_EMBEDDING_GEMMA2_EMBEDDING_GOLDEN
#define GUFO_EMBEDDING_GEMMA2_EMBEDDING_GOLDEN \
  "tests/fixtures/embedding_gemma2/embedding_golden.json"
#endif

namespace embedding_gemma2 = gufo::models::embedding_gemma2;

namespace {

// The compiled-in fixture path exists only where the tree was built. A packaged
// test image points GUFO_FIXTURE_DIR at a directory holding the fixtures.
std::string FixturePath(std::string_view compiled) {
  if (!std::filesystem::exists(std::string(compiled))) {
    if (const char* directory = std::getenv("GUFO_FIXTURE_DIR")) {
      return (std::filesystem::path(directory) /
              std::filesystem::path(compiled).filename())
          .string();
    }
  }
  return std::string(compiled);
}

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL embedding_gemma2_embedding_golden_hip_test: " << message
              << '\n';
    std::exit(1);
  }
}

double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += double(a[i]) * b[i];
    na += double(a[i]) * a[i];
    nb += double(b[i]) * b[i];
  }
  return dot / std::sqrt(na * nb);
}

std::vector<float> Truncated(const std::vector<float>& vector,
                             std::size_t dimensions) {
  std::vector<float> result(vector.begin(), vector.begin() + dimensions);
  double norm = 0.0;
  for (const float value : result) {
    norm += double(value) * value;
  }
  norm = std::sqrt(norm);
  for (float& value : result) {
    value = static_cast<float>(value / norm);
  }
  return result;
}

// The checkpoint's own BF16 execution is within 0.99987 of float32 on these
// probes. This implementation keeps activations in float32 between operations
// and measures 0.99997 or better, so this bound leaves room for GPU variation
// but catches a wrong operation.
constexpr double kMinimumCosine = 0.9999;

}  // namespace

int main(int argc, char** argv) {
  const char* configured = std::getenv("EMBEDDING_GEMMA2_MODEL_ROOT");
  const std::filesystem::path model_root =
      argc > 1
          ? argv[1]
          : (configured != nullptr ? configured : "models/embeddinggemma-2");
  if (!std::filesystem::exists(model_root / "model.safetensors")) {
    std::cerr << "SKIP embedding_gemma2_embedding_golden_hip_test: external "
                 "model unavailable\n";
    return 77;
  }
  std::string error;
  auto tokenizer =
      embedding_gemma2::Tokenizer::Load(model_root.string(), &error);
  Check(tokenizer.has_value(), error);
  auto runtime = embedding_gemma2::hip::EncoderRuntime::Create(
      model_root.string(), &error);
  if (runtime == nullptr) {
    if (error.find("no ROCm-capable device") != std::string::npos ||
        error.find("invalid device ordinal") != std::string::npos) {
      std::cerr << "SKIP embedding_gemma2_embedding_golden_hip_test: " << error
                << '\n';
      return 77;
    }
    Check(false, error);
  }

  std::ifstream input(FixturePath(GUFO_EMBEDDING_GEMMA2_EMBEDDING_GOLDEN));
  Check(input.good(), "embedding fixture must be readable");
  std::ostringstream contents;
  contents << input.rdbuf();
  const gufo::json::Value fixture = gufo::json::parse(contents.str());
  const gufo::json::Value* cases = fixture.find("cases");
  Check(cases != nullptr && cases->is_array() && cases->size() > 0,
        "fixture must hold cases");

  std::vector<std::vector<std::uint32_t>> sequences;
  std::vector<std::vector<float>> expected;
  std::vector<std::string> names;
  for (const auto& entry : cases->items()) {
    const std::string name = entry.member_str("name");
    const std::string text = entry.member_str("text");
    std::vector<std::uint32_t> ids;
    for (const auto& id : entry.find("token_ids")->items()) {
      ids.push_back(static_cast<std::uint32_t>(id.as_size()));
    }
    Check(tokenizer->Encode(text) == ids,
          "token ids must match Hugging Face for " + name);
    std::vector<float> vector;
    for (const auto& value : entry.find("embedding")->items()) {
      vector.push_back(static_cast<float>(value.as_double()));
    }
    Check(vector.size() == 768, "reference width");
    sequences.push_back(std::move(ids));
    expected.push_back(std::move(vector));
    names.push_back(name);
  }

  std::vector<std::vector<float>> actual;
  Check(runtime->Embed(sequences, &actual, &error), error);
  Check(actual.size() == expected.size(), "one embedding per case");
  // Libraries loaded by the runtime can change the stream's precision.
  std::cout << std::setprecision(8);
  double worst = 1.0;
  double worst_truncated = 1.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double cosine = Cosine(actual[i], expected[i]);
    worst = std::min(worst, cosine);
    std::cout << names[i] << " (" << sequences[i].size() << " tokens): cosine "
              << cosine << '\n';
    Check(cosine >= kMinimumCosine,
          "embedding differs from the reference for " + names[i]);
    for (const std::size_t dimensions : {128U, 256U, 512U}) {
      worst_truncated =
          std::min(worst_truncated, Cosine(Truncated(actual[i], dimensions),
                                           Truncated(expected[i], dimensions)));
    }
  }
  std::cout << "worst full-width cosine " << worst
            << ", worst truncated cosine " << worst_truncated << '\n';
  Check(worst_truncated >= kMinimumCosine,
        "truncated embeddings differ from the reference");
  std::cout << "PASS embedding_gemma2_embedding_golden_hip_test\n";
  return 0;
}
