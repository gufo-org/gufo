// Independent official-processor fixtures: every resized patch byte must match.
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"
#include "src/models/embedding_gemma2/image_processor.hpp"
#if defined(GUFO_IMAGE_TEST_GPU)
#include "src/models/embedding_gemma2/hip/encoder_runtime.hpp"
#include "src/models/embedding_gemma2/tokenizer.hpp"
#endif

namespace model = gufo::models::embedding_gemma2;
namespace {
void Check(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}
gufo::core::Image MakeImage(const gufo::json::Value& specification) {
  gufo::core::Image image;
  image.width = static_cast<std::uint32_t>(specification.member_size("width"));
  image.height =
      static_cast<std::uint32_t>(specification.member_size("height"));
  const auto seed = specification.member_size("seed");
  for (std::size_t y = 0; y < image.height; ++y) {
    for (std::size_t x = 0; x < image.width; ++x) {
      for (std::size_t c = 0; c < 3; ++c) {
        image.pixels.push_back(static_cast<std::uint8_t>(
            (x * 17 + y * 31 + c * 67 + (x * y + seed * 13) % 251) % 256));
      }
    }
  }
  return image;
}
}  // namespace
int main(int argc, char** argv) {
  try {
    std::ifstream input(GUFO_EMBEDDING_GEMMA2_IMAGE_GOLDEN);
    Check(input.good(), "image reference fixture must exist");
    std::ostringstream contents;
    contents << input.rdbuf();
    const auto fixture = gufo::json::parse(contents.str());
#if defined(GUFO_IMAGE_TEST_GPU)
    std::string error;
    const char* configured = std::getenv("EMBEDDING_GEMMA2_MODEL_ROOT");
    const std::string path = argc > 1     ? argv[1]
                             : configured ? configured
                                          : "models/embeddinggemma-2";
    if (!std::filesystem::exists(std::filesystem::path(path) /
                                 "model.safetensors")) {
      std::cerr << "SKIP image_golden_test: external model unavailable\n";
      return 77;
    }
    auto encoder = model::hip::EncoderRuntime::Create(path, &error);
    Check(encoder != nullptr, error);
    auto tokenizer = model::Tokenizer::Load(path, &error);
    Check(tokenizer.has_value(), error);
#else
    (void)argc;
    (void)argv;
#endif
    for (const auto& probe : fixture.find("cases")->items()) {
      const auto name = probe.member_str("name");
#if defined(GUFO_IMAGE_TEST_GPU)
      model::EncoderSequence sequence;
      for (const auto& id : probe.find("token_ids")->items())
        sequence.tokens.push_back(static_cast<std::uint32_t>(id.as_size()));
      std::size_t cursor = 0;
#endif
      for (const auto& specification : probe.find("images")->items()) {
        auto patches = model::ProcessImage(MakeImage(specification));
        Check(patches.rows == specification.member_size("rows") &&
                  patches.columns == specification.member_size("columns"),
              name + " patch grid");
        std::vector<std::uint8_t> bytes;
        for (float pixel : patches.pixels)
          bytes.push_back(static_cast<std::uint8_t>(std::lround(pixel * 255)));
        const auto hash = gufo::crypto::Sha256Hex(bytes);
        Check(hash == specification.member_str("patch_sha256"),
              name + " resized patch bytes differ: " + hash);
#if defined(GUFO_IMAGE_TEST_GPU)
        while (cursor < sequence.tokens.size() &&
               sequence.tokens[cursor] != 258880)
          ++cursor;
        const auto count = patches.soft_tokens();
        sequence.images.push_back({cursor, std::move(patches)});
        cursor += count;
#endif
      }
#if defined(GUFO_IMAGE_TEST_GPU)
      std::vector<std::vector<float>> vectors;
      Check(encoder->EmbedMultimodal({sequence}, &vectors, &error), error);
      Check(vectors.size() == 1 && vectors[0].size() == 768,
            name + " embedding shape");
      const auto& reference = probe.find("embedding")->items();
      double dot = 0, aa = 0, bb = 0;
      for (std::size_t i = 0; i < 768; ++i) {
        const double a = vectors[0][i], b = reference[i].as_double();
        Check(std::isfinite(a), name + " finite vector");
        dot += a * b;
        aa += a * a;
        bb += b * b;
      }
      const double cosine = dot / std::sqrt(aa * bb);
      std::cout << name << " cosine=" << cosine << '\n';
      Check(cosine >= 0.9999, name + " reference image embedding cosine");
#endif
      std::cout << "PASS " << name << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL image_golden_test: " << error.what() << '\n';
    return 1;
  }
}
