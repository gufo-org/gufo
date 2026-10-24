#include "src/models/embedding_gemma2/loader.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

namespace embedding_gemma2 = gufo::models::embedding_gemma2;

namespace {

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL embedding_gemma2_loader_test: " << message << '\n';
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const char* configured = std::getenv("EMBEDDING_GEMMA2_MODEL_ROOT");
  const std::filesystem::path model_root =
      argc > 1
          ? argv[1]
          : (configured != nullptr ? configured : "models/embeddinggemma-2");
  if (!std::filesystem::exists(model_root / "model.safetensors")) {
    std::cerr << "SKIP embedding_gemma2_loader_test: external model "
                 "unavailable\n";
    return 77;
  }
  Check(embedding_gemma2::LooksLikeEmbeddingGemma2(model_root.string()),
        "model configuration must be recognized");
  const embedding_gemma2::LoadResult loaded =
      embedding_gemma2::LoadModelDirectory(model_root.string());
  Check(loaded.ok, loaded.error);
  Check(loaded.store != nullptr, "the tensor store must exist");
  const auto expected =
      embedding_gemma2::ExpectedTextTensors(loaded.config.text);
  Check(loaded.store->size() == expected.size(),
        "the store must hold exactly the text-backbone tensors");
  const embedding_gemma2::Tensor* embedding =
      loaded.store->Find("language_model.embed_tokens.weight");
  Check(embedding != nullptr &&
            embedding->dtype == embedding_gemma2::DType::kBF16 &&
            embedding->byte_count == 262144ULL * 512ULL * 2ULL,
        "the token embedding must be BF16 [262144, 512]");
  Check(loaded.store->Find("vision_tower.patch_embedder.input_proj.weight") ==
            nullptr,
        "vision tensors must not be indexed");

  const auto multimodal =
      embedding_gemma2::LoadModelDirectory(model_root.string(), true);
  Check(multimodal.ok, multimodal.error);
  Check(multimodal.config.supported_vision,
        "released vision configuration is supported");
  Check(multimodal.store->size() ==
            expected.size() + embedding_gemma2::ExpectedVisionTensors().size(),
        "multimodal store holds text and vision only");
  Check(multimodal.store->Find(
            "vision_tower.patch_embedder.input_proj.weight") != nullptr,
        "vision weights are indexed on demand");
  std::cout << "PASS embedding_gemma2_loader_test\n";
  return 0;
}
