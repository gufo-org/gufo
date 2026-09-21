#ifndef GUFO_MODELS_QWEN3_TTS_LOADER_HPP_
#define GUFO_MODELS_QWEN3_TTS_LOADER_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen3_tts/config.hpp"
#include "src/models/qwen3_tts/tensor.hpp"

namespace gufo::models::qwen3_tts {

struct MappedRegion {
  const std::byte* data = nullptr;
  std::size_t size = 0;
  // Byte offset of the safetensors payload. Tensor offsets are relative to it,
  // so a device copy can restore vectorized-load alignment by shifting here.
  std::size_t payload_offset = 0;
};

/// Result of loading the model directory. `tensors` payload pointers stay
/// valid for the lifetime of this struct (mappings are owned by it).
struct LoadResult {
  bool ok = false;
  std::string error;
  ModelConfig config;
  std::unique_ptr<TensorStore> store;

  // Owned file mappings; payload pointers in `store` point into these.
  std::vector<std::shared_ptr<void>> mappings;
  std::vector<MappedRegion> mapped_regions;

  /// Page-aligned envelopes of the named component's tensors. Payload
  /// alignment is preserved when these ranges are copied to device memory.
  [[nodiscard]] std::vector<MappedRegion> RegionsFor(
      std::string_view prefix) const;
};

/// Loads a quality-qualified Qwen3-TTS 12.5 Hz 1.7B model directory.
///
/// Layout expected:
///   <dir>/config.json
///   <dir>/model.safetensors
///   <dir>/speech_tokenizer/model.safetensors
[[nodiscard]] LoadResult LoadModelDirectory(const std::string& model_dir);

/// True if `model_dir/config.json` is a supported 1.7B Qwen3-TTS variant.
[[nodiscard]] bool LooksLikeQwen3Tts(const std::string& model_dir);

}  // namespace gufo::models::qwen3_tts

#endif  // GUFO_MODELS_QWEN3_TTS_LOADER_HPP_
