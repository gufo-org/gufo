#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_NPU_VISION_NPU_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_NPU_VISION_NPU_HPP_

// The EmbeddingGemma 2 vision tower on the XDNA NPU (images and video
// frames): matrix products and attention scores on gufo::npu::BfpGemm, norms,
// rotary embeddings, softmax and pooling on the CPU. Arithmetic follows
// hip/vision_ops.hip and EncoderRuntime::RunVision.

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "gufo/ext/experimental/npu.hpp"
#include "src/models/embedding_gemma2/image_processor.hpp"

namespace gufo::models::embedding_gemma2::npu {

class VisionTower {
public:
  /// Uploads the tower's weights from `model_dir` into `gemm`'s arena and
  /// reserves its attention scratch. Throws on missing weights or arena.
  VisionTower(gufo::npu::BfpGemm& gemm, const std::string& model_dir);
  ~VisionTower();
  VisionTower(const VisionTower&) = delete;
  VisionTower& operator=(const VisionTower&) = delete;

  /// The image's soft tokens, [soft_tokens][512] text-width rows rounded to
  /// bf16, as spliced into the text sequence. Throws on cancellation.
  [[nodiscard]] std::vector<float> Run(const ImagePatches& image,
                                       const std::function<bool()>& cancelled);

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::embedding_gemma2::npu

#endif  // GUFO_MODELS_EMBEDDING_GEMMA2_NPU_VISION_NPU_HPP_
