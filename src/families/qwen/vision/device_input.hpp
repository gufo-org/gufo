#ifndef GUFO_MODELS_QWEN_VISION_DEVICE_INPUT_HPP_
#define GUFO_MODELS_QWEN_VISION_DEVICE_INPUT_HPP_

#include <hip/hip_runtime.h>

#include <memory>
#include <utility>
#include <vector>

#include "src/families/qwen/hip_common/continuation_hooks.hpp"
#include "src/families/qwen/hip_common/rope.hpp"
#include "src/families/qwen/vision/admission.hpp"
#include "src/families/qwen/vision/encoder.hpp"
#include "src/families/qwen/vision/prompt.hpp"

namespace gufo::models::qwen::vision {

/// Per-session image inputs. The stable device descriptor lets captured
/// decode graphs follow a new request without capturing its image pointers.
class DeviceInput {
public:
  DeviceInput() = default;
  ~DeviceInput();
  DeviceInput(const DeviceInput&) = delete;
  DeviceInput& operator=(const DeviceInput&) = delete;
  /// Set once before installing any request. The descriptor remains part of
  /// the base session claim; the shared budget covers dynamic image storage.
  void SetMemoryAdmission(std::shared_ptr<AdmissionBudget> budget);
  void Configure(std::shared_ptr<const Prompt> prompt,
                 std::shared_ptr<Encoder> encoder, hipStream_t stream);
  void SetCancellationCheck(Encoder::CancellationCheck check) {
    is_cancelled_ = std::move(check);
  }
  void RestoreLayout(const RopeLayout& layout, hipStream_t stream);
  /// Reserve the stable empty descriptor before capturing text graphs in a
  /// session that can later receive images. Existing graphs then keep the
  /// same pointer when the image layout is installed.
  void PrepareRope(hipStream_t stream);
  void Inject(float* hidden, std::uint32_t position, std::uint32_t count,
              std::uint32_t width, std::uint32_t hc, hipStream_t stream);
  [[nodiscard]] const DeviceRope* rope() const noexcept { return descriptor_; }
  [[nodiscard]] const RopeLayout& layout() const noexcept { return layout_; }
  [[nodiscard]] const std::shared_ptr<const Prompt>& PromptAttachment()
      const noexcept {
    return prompt_;
  }
  [[nodiscard]] std::size_t Bytes() const noexcept {
    std::size_t bytes = capacity_ * 3 * sizeof(std::int32_t) +
                        (descriptor_ == nullptr ? 0 : sizeof(DeviceRope));
    for (const auto& embedding : embeddings_) {
      if (embedding)
        bytes +=
            std::size_t{embedding->rows()} * embedding->width() * sizeof(float);
    }
    return bytes;
  }

private:
  void ReserveRequest(const Prompt* prompt, const Encoder* encoder,
                      std::size_t prefix_rows, bool replacement,
                      bool reuse_embeddings, bool layout_work);
  [[nodiscard]] std::size_t OnDemandBytes(const Prompt* prompt,
                                          const Encoder* encoder,
                                          std::size_t prefix_rows,
                                          bool reuse_embeddings,
                                          bool layout_work) const;
  void SettleRequest();
  void ApplyLayout(const RopeLayout& layout, hipStream_t stream);
  void ReleaseImageStorage(hipStream_t stream);
  static void LaunchRows(float* hidden, const float* embedding,
                         std::uint32_t rows, std::uint32_t width,
                         std::uint32_t hc, hipStream_t stream);
  friend class models::qwen::ContinuationAdapter;
  models::qwen::ContinuationHooks* continuation_hooks_{nullptr};
  std::shared_ptr<AdmissionBudget> admission_budget_;
  // Declared before prompt/embeddings: owned host and device buffers must die
  // before the final reservation. Positions are freed in the destructor.
  AdmissionLease admission_;
  RopeLayout layout_;
  bool layout_ready_{true};
  std::shared_ptr<const Prompt> prompt_;
  std::shared_ptr<Encoder> encoder_;
  std::vector<std::shared_ptr<const Encoder::Embedding>> embeddings_;
  DeviceRope* descriptor_{nullptr};
  std::int32_t* positions_{nullptr};
  std::size_t capacity_{0};
  Encoder::CancellationCheck is_cancelled_;
};

}  // namespace gufo::models::qwen::vision
#endif
