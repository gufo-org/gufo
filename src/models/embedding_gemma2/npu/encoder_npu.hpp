#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_NPU_ENCODER_NPU_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_NPU_ENCODER_NPU_HPP_

// EmbeddingGemma 2 text embeddings on the XDNA NPU (`serve embedding --npu`):
// every matrix product runs on gufo::npu::BfpGemm, norms, rotary embeddings,
// attention and pooling on the CPU. The GPU is not used.

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "src/models/embedding_gemma2/batch_scheduler.hpp"
#include "src/models/embedding_gemma2/config.hpp"

namespace gufo::models::embedding_gemma2::npu {

class Encoder {
public:
  ~Encoder();
  Encoder(const Encoder&) = delete;
  Encoder& operator=(const Encoder&) = delete;

  /// Loads the text weights from `model_dir` onto the NPU. Null with `error`
  /// set when the NPU, its designs or the weights are unavailable.
  [[nodiscard]] static std::unique_ptr<Encoder> Create(
      const std::string& model_dir, std::string* error = nullptr);

  /// The checkpoint's configuration, with image, video and audio inputs
  /// marked unsupported.
  [[nodiscard]] const ModelConfig& config() const;

  /// Pooled, L2-normalized embeddings of text-only sequences, as
  /// hip::EncoderRuntime::EmbedMultimodal. Images and audio are refused.
  [[nodiscard]] bool EmbedMultimodal(
      const std::vector<EncoderSequence>& sequences,
      std::vector<std::vector<float>>* embeddings, std::string* error = nullptr,
      const std::function<bool()>& is_cancelled = {});

  struct Impl;

private:
  Encoder();
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::embedding_gemma2::npu

#endif  // GUFO_MODELS_EMBEDDING_GEMMA2_NPU_ENCODER_NPU_HPP_
