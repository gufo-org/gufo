#ifndef STRIX_CORE_SPECULATIVE_SPECULATIVE_VERIFIER_HPP_
#define STRIX_CORE_SPECULATIVE_SPECULATIVE_VERIFIER_HPP_

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "src/core/speculative/draft_backend.hpp"
#include "src/models/qwen/qwen_generator.hpp"
#include "src/models/qwen/qwen_tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/qwen_gpu_executor.hpp"
#endif

namespace strix::speculative {

struct SpeculativeOptions {
  std::uint32_t max_draft_tokens{4};
  std::uint32_t min_draft_tokens{1};
  std::uint32_t initial_draft_tokens{3};
  std::size_t rolling_window{16};
  float target_acceptance_rate{0.70F};
  bool enable_adaptive_draft_length{true};
};

struct SpeculativeStats {
  std::size_t total_draft_tokens{0};
  std::size_t total_accepted_tokens{0};
  std::size_t total_verification_steps{0};
  std::size_t total_emitted_tokens{0};
  [[nodiscard]] float AcceptanceRate() const noexcept {
    if (total_draft_tokens == 0)
      return 0.0F;
    return static_cast<float>(total_accepted_tokens) /
           static_cast<float>(total_draft_tokens);
  }
};

#if defined(ENGINE_ENABLE_HIP)

class ISpeculativeTargetExecutor {
public:
  virtual ~ISpeculativeTargetExecutor() = default;

  virtual void Reset() noexcept = 0;
  [[nodiscard]] virtual tokenization::TokenId ForwardPromptBatch(
      std::span<const tokenization::TokenId> prompt_tokens) = 0;
  [[nodiscard]] virtual tokenization::TokenId ForwardToken(
      tokenization::TokenId token_id, std::uint32_t pos,
      bool compute_logits = true) = 0;
  virtual void SaveState(std::uint32_t valid_context) = 0;
  virtual void RestoreState() = 0;
  virtual void SetPromptHiddenCapture(bool enabled) { (void)enabled; }
  [[nodiscard]] virtual std::span<const float> GetPromptHiddenStates()
      const noexcept {
    return {};
  }
  [[nodiscard]] virtual std::span<const float> CopyLastHidden() { return {}; }
  [[nodiscard]] virtual tokenization::TokenId GetEosTokenId()
      const noexcept = 0;
  [[nodiscard]] virtual std::string_view DecodeToken(
      tokenization::TokenId token_id) const noexcept = 0;
};

/// High-throughput speculative decoding verifier with transactional state
/// management
class SpeculativeVerifier {
public:
  SpeculativeVerifier(hip::QwenGpuExecutor& target_executor,
                      std::unique_ptr<IDraftBackend> draft_backend,
                      SpeculativeOptions options = {});
  SpeculativeVerifier(ISpeculativeTargetExecutor& target_executor,
                      std::unique_ptr<IDraftBackend> draft_backend,
                      SpeculativeOptions options = {});

  /// Runs speculative auto-regressive generation
  std::vector<tokenization::TokenId> Generate(
      std::span<const tokenization::TokenId> prompt_tokens,
      const models::GenerationOptions& options,
      const std::function<bool(tokenization::TokenId, std::string_view)>&
          on_token = nullptr);

  /// Resets target and draft state, prefills the prompt, and returns the first
  /// target token. This setup is outside benchmarked decode regions.
  [[nodiscard]] tokenization::TokenId Prime(
      std::span<const tokenization::TokenId> prompt_tokens);

  /// Performs a single speculative verification step
  struct StepResult {
    std::vector<tokenization::TokenId> emitted_tokens;
    std::size_t accepted_count{0};
    std::size_t draft_count{0};
    tokenization::TokenId next_token{0};
    bool hit_eos{false};
  };

  StepResult VerifyStep(std::vector<tokenization::TokenId>& current_sequence,
                        std::uint32_t cur_pos,
                        tokenization::TokenId current_token,
                        tokenization::TokenId eos_id);

  [[nodiscard]] const SpeculativeStats& GetStats() const noexcept {
    return stats_;
  }

  [[nodiscard]] std::uint32_t GetCurrentDraftLength() const noexcept {
    return current_draft_length_;
  }

  void Reset() noexcept;

private:
  void UpdateDraftTargetHidden();
  void UpdateAdaptiveDraftLength(std::size_t accepted, std::size_t drafted);

  std::unique_ptr<ISpeculativeTargetExecutor> owned_target_executor_;
  ISpeculativeTargetExecutor* target_executor_;
  std::unique_ptr<IDraftBackend> draft_backend_;
  SpeculativeOptions options_;
  SpeculativeStats stats_;
  std::uint32_t current_draft_length_{3};
  std::deque<float> rolling_acceptance_;
};

#endif  // defined(ENGINE_ENABLE_HIP)

}  // namespace strix::speculative

#endif  // STRIX_CORE_SPECULATIVE_SPECULATIVE_VERIFIER_HPP_
