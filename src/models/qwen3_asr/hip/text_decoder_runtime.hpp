#ifndef GUFO_MODELS_QWEN3_ASR_HIP_TEXT_DECODER_RUNTIME_HPP_
#define GUFO_MODELS_QWEN3_ASR_HIP_TEXT_DECODER_RUNTIME_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/qwen3_asr/transcription.hpp"

namespace gufo::models::qwen3_asr::hip {

struct TextDecoderTrace {
  // Decoder output after layer 0, row-major [prompt_tokens, 2048].
  std::vector<float> layer0;
  // BF16-rounded full-vocabulary logits for the final prompt position.
  std::vector<float> logits;
};

class TextDecoderHipRuntime {
public:
  ~TextDecoderHipRuntime();

  TextDecoderHipRuntime(const TextDecoderHipRuntime&) = delete;
  TextDecoderHipRuntime& operator=(const TextDecoderHipRuntime&) = delete;
  TextDecoderHipRuntime(TextDecoderHipRuntime&&) = delete;
  TextDecoderHipRuntime& operator=(TextDecoderHipRuntime&&) = delete;

  [[nodiscard]] static std::unique_ptr<TextDecoderHipRuntime> Create(
      const std::string& model_root, std::size_t maximum_tokens = 1024,
      std::string* error = nullptr);

  /// Embeds the prompt, replaces every audio-pad row with the supplied audio
  /// embeddings, and fills the causal KV cache.
  [[nodiscard]] bool Prefill(std::span<const std::uint32_t> prompt_ids,
                             std::span<const float> audio_embeddings,
                             std::size_t audio_tokens, TextDecoderTrace* output,
                             std::string* error = nullptr);

  /// Greedy generation. The output includes a terminal EOS token when one is
  /// produced, matching Transformers' returned generated-id suffix.
  [[nodiscard]] bool Generate(std::span<const std::uint32_t> prompt_ids,
                              std::span<const float> audio_embeddings,
                              std::size_t audio_tokens,
                              std::size_t maximum_new_tokens,
                              std::vector<std::uint32_t>* generated_ids,
                              std::string* error = nullptr);

  /// Greedy generation from device-resident float32 audio embeddings.
  [[nodiscard]] bool GenerateDevice(
      std::span<const std::uint32_t> prompt_ids,
      const float* audio_embeddings_device, std::size_t audio_tokens,
      std::size_t maximum_new_tokens, std::vector<std::uint32_t>* generated_ids,
      std::string* error = nullptr,
      const std::function<bool(std::span<const std::uint32_t>)>& on_tokens = {},
      const CancellationCheck& is_cancelled = {});

private:
  struct Impl;
  explicit TextDecoderHipRuntime(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen3_asr::hip

#endif  // GUFO_MODELS_QWEN3_ASR_HIP_TEXT_DECODER_RUNTIME_HPP_
