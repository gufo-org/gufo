#ifndef GUFO_MODELS_QWEN3_ASR_HIP_TRANSCRIPTION_RUNTIME_HPP_
#define GUFO_MODELS_QWEN3_ASR_HIP_TRANSCRIPTION_RUNTIME_HPP_

#include <cstddef>
#include <memory>
#include <string>

#include "src/models/qwen3_asr/transcription.hpp"

namespace gufo::models::qwen3_asr::hip {

/// End-to-end native HIP transcription for Qwen3-ASR-1.7B.
class TranscriptionHipRuntime {
public:
  ~TranscriptionHipRuntime();

  TranscriptionHipRuntime(const TranscriptionHipRuntime&) = delete;
  TranscriptionHipRuntime& operator=(const TranscriptionHipRuntime&) = delete;
  TranscriptionHipRuntime(TranscriptionHipRuntime&&) = delete;
  TranscriptionHipRuntime& operator=(TranscriptionHipRuntime&&) = delete;

  [[nodiscard]] static std::unique_ptr<TranscriptionHipRuntime> Create(
      const std::string& model_root, std::size_t maximum_context_tokens = 1024,
      std::string* error = nullptr);

  [[nodiscard]] bool Transcribe(const TranscriptionRequest& request,
                                const CancellationCheck& is_cancelled,
                                TranscriptionResult* result,
                                std::string* error = nullptr);

private:
  struct Impl;
  explicit TranscriptionHipRuntime(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen3_asr::hip

#endif  // GUFO_MODELS_QWEN3_ASR_HIP_TRANSCRIPTION_RUNTIME_HPP_
