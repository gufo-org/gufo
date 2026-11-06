#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_HIP_AUDIO_RUNTIME_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_HIP_AUDIO_RUNTIME_HPP_
#include <hip/hip_runtime.h>

#include <memory>
#include <string>

#include "audio_processor.hpp"

namespace gufo::models::embedding_gemma2::hip {
// Float32 audio tower with checkpoint BF16 weights widened once at load. Owned
// by the text runtime and invoked only while its scheduler owns the device.
class AudioRuntime {
public:
  AudioRuntime(const std::string& model_dir, hipStream_t stream);
  ~AudioRuntime();
  AudioRuntime(const AudioRuntime&) = delete;
  AudioRuntime& operator=(const AudioRuntime&) = delete;
  // Returns device [valid_tokens, 512] features. Remains alive until next Run.
  [[nodiscard]] const float* Run(
      const AudioFeatures& features,
      const std::function<bool()>& is_cancelled = {});
  [[nodiscard]] const float* encoded_states() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gufo::models::embedding_gemma2::hip
#endif
