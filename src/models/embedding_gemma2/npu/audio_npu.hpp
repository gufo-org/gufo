#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_NPU_AUDIO_NPU_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_NPU_AUDIO_NPU_HPP_

// The EmbeddingGemma 2 audio tower (Gemma4 conformer) on the XDNA NPU: the
// linear layers on gufo::npu::BfpGemm, the subsampling convolutions, norms,
// chunked local attention and the light convolution on the CPU. Arithmetic
// follows hip/audio_runtime.hip.

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "gufo/ext/experimental/npu.hpp"
#include "src/models/embedding_gemma2/audio_processor.hpp"

namespace gufo::models::embedding_gemma2::npu {

class AudioTower {
public:
  /// Uploads the tower's weights from `model_dir` into `gemm`'s arena.
  AudioTower(gufo::npu::BfpGemm& gemm, const std::string& model_dir);
  ~AudioTower();
  AudioTower(const AudioTower&) = delete;
  AudioTower& operator=(const AudioTower&) = delete;

  /// The clip's soft tokens, [soft_tokens][512] text-width rows rounded to
  /// bf16, as spliced into the text sequence. Throws on cancellation.
  [[nodiscard]] std::vector<float> Run(const AudioFeatures& features,
                                       const std::function<bool()>& cancelled);

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::embedding_gemma2::npu

#endif  // GUFO_MODELS_EMBEDDING_GEMMA2_NPU_AUDIO_NPU_HPP_
