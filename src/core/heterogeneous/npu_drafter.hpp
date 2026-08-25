#ifndef GUFO_CORE_HETEROGENEOUS_NPU_DRAFTER_HPP_
#define GUFO_CORE_HETEROGENEOUS_NPU_DRAFTER_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/speculative/draft_backend.hpp"

#ifdef ENGINE_ENABLE_XRT
#include <xrt/experimental/xrt_system.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#endif

namespace gufo::heterogeneous {

enum class NpuDraftMode {
  kMTP,
  kDFlash2,
};

struct NpuDrafterConfig {
  NpuDraftMode mode{NpuDraftMode::kDFlash2};
  std::string mtp_model_path{
      "models/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"};
  std::string dflash_model_path{
      "models/Qwen3.8-27B-GGUF/DFlash/dflash-Qwen3.8-27B-Q4_0.gguf"};
  // A DFlash NPU route is valid only with a versioned XRT program. Keeping
  // this empty makes --speculative-backend=dflash-npu fail explicitly instead
  // of silently using the host n-gram fallback.
  std::string dflash_xclbin_path{};
  std::uint32_t device_index{0};
  std::uint32_t max_draft_tokens{5};
  std::uint32_t vocab_size{152064};
  std::vector<std::uint32_t> target_layer_ids{15, 31, 47, 63};
  bool enable_xrt{true};
  std::size_t shared_buffer_size{
      256 * 1024};  // 256KB unified DMA buffer for features
};

struct NpuDrafterMetrics {
  std::size_t proposal_invocations{0};
  std::size_t npu_submissions{0};
  std::size_t dma_bytes_transferred{0};
  double total_npu_time_us{0.0};
  double total_dma_time_us{0.0};
  std::size_t generated_proposals{0};
};

/// Heterogeneous XDNA2 NPU Draft Backend via XRT unified memory
/// Executes MTP or DFlash-2 Block Diffusion Drafting on the 50 TOPS XDNA2 NPU
class NpuDraftBackend : public speculative::IDraftBackend {
public:
  explicit NpuDraftBackend(NpuDrafterConfig config = {});
  ~NpuDraftBackend() override;

  [[nodiscard]] std::string_view Name() const noexcept override {
    return config_.mode == NpuDraftMode::kDFlash2
               ? "NpuXdna2DFlash2DraftBackend"
               : "NpuXdna2MtpDraftBackend";
  }

  [[nodiscard]] bool RequiresTargetHiddenStates() const noexcept override {
    return config_.mode == NpuDraftMode::kDFlash2;
  }

  [[nodiscard]] std::span<const std::uint32_t> TargetHiddenLayerIds()
      const noexcept override {
    return config_.target_layer_ids;
  }

  [[nodiscard]] bool PrimeTargetContext(
      const speculative::DraftTargetContext& context) override;

  [[nodiscard]] speculative::DraftProposal Propose(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens) override;

  void AcceptFeedback(std::span<const tokenization::TokenId> accepted,
                      tokenization::TokenId correction_token) override;

  void Reset() noexcept override;

  [[nodiscard]] bool IsNpuActive() const noexcept { return npu_available_; }
  [[nodiscard]] bool HasModel() const noexcept { return has_model_; }
  [[nodiscard]] NpuDraftMode Mode() const noexcept { return config_.mode; }

  [[nodiscard]] const std::string& GetStatusMessage() const noexcept {
    return status_message_;
  }

  [[nodiscard]] const NpuDrafterMetrics& GetMetrics() const noexcept {
    return metrics_;
  }

private:
  void InitializeXrt();
  void LoadModelGguf();

  NpuDrafterConfig config_;
  bool npu_available_{false};
  bool has_model_{false};
  std::string status_message_{"Uninitialized"};
  std::unique_ptr<core::GgufReader> model_reader_;

#ifdef ENGINE_ENABLE_XRT
  std::unique_ptr<xrt::device> device_;
  std::unique_ptr<xrt::bo> shared_bo_;
#endif
  std::vector<tokenization::TokenId> history_;
  std::vector<float> cached_target_features_;
  NpuDrafterMetrics metrics_{};
};

}  // namespace gufo::heterogeneous

#endif  // GUFO_CORE_HETEROGENEOUS_NPU_DRAFTER_HPP_
