#include "src/models/qwen3_asr/loader.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

namespace qwen3_asr = gufo::models::qwen3_asr;

namespace {

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL qwen3_asr_loader_test: " << message << '\n';
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const char* configured = std::getenv("QWEN3_ASR_MODEL_ROOT");
  const std::filesystem::path model_root =
      argc > 1 ? argv[1]
               : (configured != nullptr
                      ? configured
                      : "/var/llms/huggingface/hub/"
                        "models--Qwen--Qwen3-ASR-1.7B/snapshots/"
                        "7278e1e70fe206f11671096ffdd38061171dd6e5");
  if (!std::filesystem::exists(model_root / "config.json")) {
    std::cerr << "SKIP qwen3_asr_loader_test: external model unavailable\n";
    return 77;
  }
  Check(qwen3_asr::LooksLikeQwen3Asr(model_root.string()),
        "model configuration must be recognized");
  qwen3_asr::LoadResult loaded =
      qwen3_asr::LoadModelDirectory(model_root.string());
  Check(loaded.ok, loaded.error);
  Check(loaded.mapped_regions.size() == 2,
        "the official checkpoint must map two shards");
  Check(loaded.store != nullptr && loaded.store->size() > 300,
        "the tensor inventory must be populated");

  const qwen3_asr::Tensor* conv =
      loaded.store->Find("thinker.audio_tower.conv2d1.weight");
  Check(conv != nullptr && conv->dtype == qwen3_asr::DType::kBF16 &&
            conv->shape.size() == 4 && conv->shape[0] == 480,
        "audio convolution tensor must be present");
  const qwen3_asr::Tensor* embedding =
      loaded.store->Find("thinker.model.embed_tokens.weight");
  Check(embedding != nullptr && embedding->shape.size() == 2 &&
            embedding->shape[0] == 151936 && embedding->shape[1] == 2048,
        "text embedding tensor must be present");
  const qwen3_asr::Tensor* output =
      loaded.store->Find("thinker.lm_head.weight");
  Check(output != nullptr && output->shape == embedding->shape,
        "language-model output tensor must be present");

  const auto text_regions =
      loaded.RegionsFor({"thinker.model.", "thinker.lm_head."});
  const auto contains = [](const auto& regions,
                           const qwen3_asr::Tensor& tensor) {
    const auto address = reinterpret_cast<std::uintptr_t>(tensor.data);
    for (const auto& region : regions) {
      const auto base = reinterpret_cast<std::uintptr_t>(region.data);
      if (address >= base && address - base < region.size &&
          tensor.byte_count <= region.size - (address - base))
        return true;
    }
    return false;
  };
  Check(
      contains(text_regions, *embedding) && contains(text_regions, *output) &&
          !contains(text_regions, *conv),
      "text uploads must retain both heads without duplicating audio weights");
  const auto audio_regions = loaded.RegionsFor({"thinker.audio_tower."});
  Check(contains(audio_regions, *conv) && !contains(audio_regions, *embedding),
        "audio prefetch must not read the text decoder");
  std::size_t all_bytes = 0, text_bytes = 0;
  for (const auto& region : loaded.mapped_regions)
    all_bytes += region.size;
  for (const auto& region : text_regions)
    text_bytes += region.size;
  Check(text_bytes < all_bytes,
        "text weight selection did not remove unused bytes");
  std::cout << "PASS qwen3_asr_loader_test tensors=" << loaded.store->size()
            << " avoided_text_upload_bytes=" << all_bytes - text_bytes << '\n';
  return 0;
}
