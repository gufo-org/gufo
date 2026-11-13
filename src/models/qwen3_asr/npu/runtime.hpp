#ifndef GUFO_MODELS_QWEN3_ASR_NPU_RUNTIME_HPP_
#define GUFO_MODELS_QWEN3_ASR_NPU_RUNTIME_HPP_

// Qwen3-ASR on the XDNA NPU (`--npu`). Matrix products run on the NPU:
// gufo::npu::BfpGemm for the many-row audio encoder and the decoder prefill,
// gufo::npu::Q8Gemm (BF16 weights quantized to Q8_0 at load) for the decode
// steps and the LM head. Convolutions' patches, norms, rotary embeddings,
// attention and sampling run on the CPU. The GPU is not used.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/qwen3_asr/transcription.hpp"

namespace gufo::models::qwen3_asr::npu {

class Runtime {
public:
  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  /// Loads the checkpoint's weights onto the NPU. Throws on failure.
  [[nodiscard]] static std::unique_ptr<Runtime> Create(
      const std::string& model_root, std::size_t maximum_context_tokens);

  /// Audio encoder: log-mel [128][frames] to embeddings [tokens][2048].
  /// Returns the token count.
  std::size_t Encode(std::span<const float> log_mel, std::size_t frames,
                     std::vector<float>* embeddings,
                     const CancellationCheck& is_cancelled);

  /// Greedy generation from `prompt_ids` with the audio embeddings spliced
  /// over the audio pad tokens, as TextDecoderHipRuntime::GenerateDevice.
  void Generate(
      std::span<const std::uint32_t> prompt_ids,
      std::span<const float> audio_embeddings, std::size_t audio_tokens,
      std::size_t maximum_new_tokens, std::vector<std::uint32_t>* generated,
      const std::function<bool(std::span<const std::uint32_t>)>& on_tokens,
      const CancellationCheck& is_cancelled);

  struct Impl;

private:
  Runtime();
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen3_asr::npu

#endif  // GUFO_MODELS_QWEN3_ASR_NPU_RUNTIME_HPP_
