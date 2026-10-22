// Compares the GPU encoder with the float32 CPU reference on fixed token
// sequences, and checks that packing sequences into one batch does not change
// any embedding beyond rounding.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/embedding_gemma2/hip/encoder_runtime.hpp"
#include "src/models/embedding_gemma2/loader.hpp"
#include "src/models/embedding_gemma2/reference/cpu_encoder.hpp"

namespace embedding_gemma2 = gufo::models::embedding_gemma2;

namespace {

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL embedding_gemma2_encoder_hip_test: " << message << '\n';
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

std::vector<std::uint32_t> RandomTokens(std::size_t length, unsigned seed) {
  std::mt19937 random(seed);
  std::vector<std::uint32_t> tokens{2};
  while (tokens.size() + 1 < length) {
    tokens.push_back(1000 + random() % 100000);
  }
  tokens.push_back(1);
  return tokens;
}

}  // namespace

int main(int argc, char** argv) {
  const char* configured = std::getenv("EMBEDDING_GEMMA2_MODEL_ROOT");
  const std::filesystem::path model_root =
      argc > 1
          ? argv[1]
          : (configured != nullptr ? configured : "models/embeddinggemma-2");
  if (!std::filesystem::exists(model_root / "model.safetensors")) {
    std::cerr << "SKIP embedding_gemma2_encoder_hip_test: external model "
                 "unavailable\n";
    return 77;
  }

  std::string error;
  auto runtime = embedding_gemma2::hip::EncoderRuntime::Create(
      model_root.string(), &error);
  if (runtime == nullptr) {
    // A missing device is a skip. Any other failure is real.
    if (error.find("no ROCm-capable device") != std::string::npos ||
        error.find("invalid device ordinal") != std::string::npos) {
      std::cerr << "SKIP embedding_gemma2_encoder_hip_test: " << error << '\n';
      return 77;
    }
    Check(false, error);
  }

  const std::vector<std::vector<std::uint32_t>> sequences = {
      RandomTokens(9, 1), RandomTokens(24, 2), RandomTokens(3, 3),
      RandomTokens(48, 4)};
  std::vector<std::vector<float>> batched;
  Check(runtime->Embed(sequences, &batched, &error), error);
  Check(batched.size() == sequences.size(), "one embedding per sequence");

  const embedding_gemma2::LoadResult loaded =
      embedding_gemma2::LoadModelDirectory(model_root.string());
  Check(loaded.ok, loaded.error);
  const embedding_gemma2::reference::CpuEncoder cpu(loaded.config.text,
                                                    *loaded.store);
  for (std::size_t i = 0; i < sequences.size(); ++i) {
    Check(batched[i].size() == 768, "embedding width");
    double norm = 0.0;
    for (const float value : batched[i]) {
      norm += double(value) * value;
    }
    Check(std::abs(norm - 1.0) < 1e-3, "embeddings are unit length");
    const std::vector<float> expected = cpu.Embed(sequences[i]);
    const double cosine = Cosine(batched[i], expected);
    std::cout << std::setprecision(8);
    std::cout << "sequence " << i << " cosine vs CPU reference " << cosine
              << '\n';
    Check(cosine > 0.9995, "GPU embedding disagrees with the CPU reference");

    std::vector<std::vector<float>> single;
    Check(runtime->Embed({sequences[i]}, &single, &error), error);
    const double alone = Cosine(single[0], batched[i]);
    std::cout << "sequence " << i << " cosine alone vs batched " << alone
              << '\n';
    // hipBLAS may pick a different kernel for a different batch size, which
    // changes float32 accumulation order but not the result by more than
    // rounding. A leak between sequences would be far larger.
    Check(alone > 0.9999, "batching must not change an embedding");
  }

  Check(!runtime->Embed({{}}, &batched, &error), "empty input is rejected");
  Check(!runtime->Embed({{2, 262144}}, &batched, &error),
        "out-of-range token is rejected");
  std::cout << "PASS embedding_gemma2_encoder_hip_test\n";
  return 0;
}
