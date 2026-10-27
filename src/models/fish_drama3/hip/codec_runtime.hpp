#ifndef GUFO_MODELS_FISH_DRAMA3_HIP_CODEC_RUNTIME_HPP_
#define GUFO_MODELS_FISH_DRAMA3_HIP_CODEC_RUNTIME_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/fish_drama3/config.hpp"
#include "src/models/fish_drama3/prompt.hpp"

namespace gufo::models::fish_drama3::hip {

class FishDacCodecHipRuntime {
public:
  ~FishDacCodecHipRuntime();

  FishDacCodecHipRuntime(const FishDacCodecHipRuntime&) = delete;
  FishDacCodecHipRuntime& operator=(const FishDacCodecHipRuntime&) = delete;

  [[nodiscard]] static std::unique_ptr<FishDacCodecHipRuntime> Create(
      const std::string& model_dir, const CodecConfig& config = {},
      std::string* error = nullptr);

  // Encode 44.1 kHz mono waveform into [10, frames] FishDacCodes.
  [[nodiscard]] bool Encode(std::span<const float> mono_samples,
                            FishDacCodes* output, std::string* error = nullptr);

  // Offline full-sequence decode of [10, frames] FishDacCodes into 44.1 kHz
  // mono PCM.
  [[nodiscard]] bool Decode(const FishDacCodes& codes,
                            std::vector<float>* output_samples,
                            std::string* error = nullptr);

  // Reset streaming decoder state (convolution histories & transformer KV
  // caches).
  void ResetStream();

  // Streaming decode of a chunk (e.g. 4 frames) preserving causal state across
  // calls.
  [[nodiscard]] bool DecodeStreamChunk(const FishDacCodes& chunk_codes,
                                       std::vector<float>* output_samples,
                                       std::string* error = nullptr);

  [[nodiscard]] std::uint32_t sample_rate() const noexcept;
  [[nodiscard]] std::uint32_t frame_length() const noexcept;

private:
  struct Impl;
  explicit FishDacCodecHipRuntime(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::fish_drama3::hip

#endif  // GUFO_MODELS_FISH_DRAMA3_HIP_CODEC_RUNTIME_HPP_
