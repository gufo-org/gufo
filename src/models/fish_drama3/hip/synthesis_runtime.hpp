#ifndef GUFO_MODELS_FISH_DRAMA3_HIP_SYNTHESIS_RUNTIME_HPP_
#define GUFO_MODELS_FISH_DRAMA3_HIP_SYNTHESIS_RUNTIME_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/fish_drama3/config.hpp"
#include "src/models/fish_drama3/prompt.hpp"

namespace gufo::models::fish_drama3 {

struct SamplingParams {
  float temperature = 0.8F;
  float top_p = 0.8F;
  int top_k = 30;
  std::uint32_t seed = 42;
  std::size_t max_new_frames = 1024;
  std::size_t stream_chunk_frames = 4;
};

struct SynthesisRequest {
  std::string text;
  std::vector<float> reference_audio_mono;
  std::uint32_t reference_sample_rate = 44100;
  std::string reference_text;
  std::vector<Drama3Take> references;
  std::vector<Drama3Take> history;
  SamplingParams sampling;
  std::function<bool(std::span<const float>)> on_audio{};
};

struct SynthesisResult {
  std::uint32_t sample_rate = 44100;
  std::uint32_t code_groups = 10;
  std::vector<float> samples;
  FishDacCodes codes;
  std::size_t sample_count = 0;
  bool reached_limit = false;
  double backbone_prefill_ms = 0.0;
  double tower_prefill_ms = 0.0;
  double backbone_decode_ms = 0.0;
  double tower_decode_ms = 0.0;
  double codec_decode_ms = 0.0;
  double total_ms = 0.0;
};

using CancellationCheck = std::function<bool()>;

}  // namespace gufo::models::fish_drama3

namespace gufo::models::fish_drama3::hip {

class SynthesisHipRuntime {
public:
  ~SynthesisHipRuntime();

  SynthesisHipRuntime(const SynthesisHipRuntime&) = delete;
  SynthesisHipRuntime& operator=(const SynthesisHipRuntime&) = delete;

  [[nodiscard]] static std::unique_ptr<SynthesisHipRuntime> Create(
      const std::string& model_dir, std::size_t max_context_frames = 4096,
      std::string* error = nullptr);

  [[nodiscard]] bool EncodeReference(std::span<const float> mono_samples,
                                     std::uint32_t sample_rate,
                                     FishDacCodes* out_codes,
                                     std::string* error = nullptr);

  [[nodiscard]] bool Generate(const SynthesisRequest& request,
                              const CancellationCheck& is_cancelled,
                              SynthesisResult* result,
                              std::string* error = nullptr);

  [[nodiscard]] const ModelConfig& config() const noexcept;

private:
  struct Impl;
  explicit SynthesisHipRuntime(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::fish_drama3::hip

#endif  // GUFO_MODELS_FISH_DRAMA3_HIP_SYNTHESIS_RUNTIME_HPP_
