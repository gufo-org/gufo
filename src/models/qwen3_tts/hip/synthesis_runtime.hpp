#ifndef GUFO_MODELS_QWEN3_TTS_HIP_SYNTHESIS_RUNTIME_HPP_
#define GUFO_MODELS_QWEN3_TTS_HIP_SYNTHESIS_RUNTIME_HPP_

#include <cstddef>
#include <memory>
#include <string>

#include "src/models/qwen3_tts/config.hpp"
#include "src/models/qwen3_tts/synthesis.hpp"

namespace gufo::models::qwen3_tts::hip {

/// End-to-end native HIP synthesis for one supported Qwen3-TTS 12Hz 1.7B
/// variant.
/// Where synthesis runs: the GPU, or the XDNA NPU with the CPU (`--npu`).
enum class SynthesisDevice { kGpu, kNpu };

class SynthesisHipRuntime {
public:
  ~SynthesisHipRuntime();

  SynthesisHipRuntime(const SynthesisHipRuntime&) = delete;
  SynthesisHipRuntime& operator=(const SynthesisHipRuntime&) = delete;
  SynthesisHipRuntime(SynthesisHipRuntime&&) = delete;
  SynthesisHipRuntime& operator=(SynthesisHipRuntime&&) = delete;

  [[nodiscard]] static std::unique_ptr<SynthesisHipRuntime> Create(
      const std::string& model_root, std::size_t maximum_context_tokens,
      ModelVariant variant, std::string* error = nullptr,
      SynthesisDevice device = SynthesisDevice::kGpu);

  [[nodiscard]] bool Generate(const SynthesisRequest& request,
                              const CancellationCheck& is_cancelled,
                              SynthesisResult* result,
                              std::string* error = nullptr);

private:
  struct Impl;
  explicit SynthesisHipRuntime(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen3_tts::hip

#endif  // GUFO_MODELS_QWEN3_TTS_HIP_SYNTHESIS_RUNTIME_HPP_
