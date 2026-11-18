#ifndef GUFO_MODELS_GLM5_NEXT_ENGINE_HPP_
#define GUFO_MODELS_GLM5_NEXT_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"
#include "staging/models/glm5_next/config.hpp"
#include "staging/models/glm5_next/tokenizer.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::glm5_next {

struct ModelWeights;

namespace rocm {
class DeviceModel;
class Executor;
class Session;
}  // namespace rocm

struct ModelOptions {
  std::uint32_t max_context = 4096;
};

class Session;

/// Gufo-owned API over the ROCm runtime: one resident model with a single
/// session. The executor owns the activation scratch, and every cache (KDA
/// conv rings and recurrent states, latent MLA rows, indexer pool) lives in
/// the one session object, so the model supports exactly one session.
class Model final : public std::enable_shared_from_this<Model> {
 public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::shared_ptr<Session> CreateSession(
      std::uint32_t max_context, std::string* error_msg = nullptr);

  [[nodiscard]] std::vector<std::int32_t> Tokenize(
      std::string_view text) const;
  [[nodiscard]] std::string Decode(
      std::span<const std::int32_t> tokens) const;
  [[nodiscard]] std::string TokenText(std::int32_t token) const;
  [[nodiscard]] std::int32_t EosToken() const noexcept;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept;
  /// True when the checkpoint carries the nextn draft block, making MTP
  /// speculative decoding available.
  [[nodiscard]] bool HasMtp() const noexcept;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return options_.max_context;
  }
  [[nodiscard]] std::string ModelName() const;
  /// Device bytes held by the loaded model (weights resident at Load).
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  /// Cache bytes one session consumes at `context` (caches plus the
  /// capacity-sized selection scratch).
  [[nodiscard]] std::size_t SessionBytes(std::uint32_t context) const noexcept;
  /// The executor scratch is allocated at Load; nothing is deferred.
  [[nodiscard]] std::size_t DeferredScratchBytes() const noexcept {
    return 0;
  }
  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] const GlmTokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }

 private:
  friend class Session;
  Model() = default;

  ModelOptions options_;
  std::shared_ptr<core::GgufReader> reader_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<GlmTokenizer> tokenizer_;
  std::unique_ptr<rocm::DeviceModel> device_;
  std::unique_ptr<rocm::Executor> executor_;
  std::weak_ptr<Session> live_session_;
};

/// One conversation's context. Sync feeds a prompt (appending the tail the
/// session has not seen yet); the caches cannot be truncated mid-history, so
/// any prompt divergence resets and replays it. Evaluate appends one token.
/// The logits of the last token are kept in host order (the executor leaves
/// them on the device; Sync/Evaluate copy them once at the end of the pass).
class Session final {
 public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Makes the session state equal to `prompt`: appends when the current
  /// tokens are a prefix of the prompt, otherwise restarts and replays it.
  /// Returns false on a context overflow (the executor itself aborts on a
  /// HIP failure, so no other failure mode reaches the caller).
  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr);
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr);
  struct DecodeResult {
    std::vector<std::int32_t> tokens;
    bool stop{false};
    std::size_t drafted{0};
    std::size_t accepted{0};
  };
  /// Draws the next token, then (with the nextn block enabled) chains up to
  /// k drafts through the draft block, verifies all k + 1 rows in one
  /// batched trunk pass and keeps the longest prefix the trunk agrees with;
  /// a rejection appends the trunk's own correction token. Without MTP it
  /// samples and evaluates exactly one token. The committed sequence and the
  /// round statistics land in `result`.
  [[nodiscard]] bool DecodeStep(std::size_t max_tokens,
                                sampling::SamplerState& sampler,
                                DecodeResult* result,
                                std::string* error_msg = nullptr,
                                bool stop_at_eos = true);
  [[nodiscard]] std::span<const float> Logits() const noexcept {
    return valid_ ? std::span<const float>(logits_) : std::span<const float>{};
  }
  [[nodiscard]] std::uint32_t Position() const noexcept;
  [[nodiscard]] std::uint32_t ContextSize() const noexcept {
    return static_cast<std::uint32_t>(tokens_.size());
  }
  [[nodiscard]] std::span<const std::int32_t> Tokens() const noexcept {
    return valid_ ? std::span<const std::int32_t>(tokens_)
                  : std::span<const std::int32_t>{};
  }
  void Reset();
  [[nodiscard]] bool IsValid() const noexcept { return valid_; }
  /// Bounds on the per-round MTP draft count. The session starts at `min`
  /// drafts and grows by one after two consecutive fully accepted rounds,
  /// dropping by one on any rejection; `min` == `max` pins the count.
  void SetDraftLimits(std::uint32_t min_drafts, std::uint32_t max_drafts);
  /// Enables the nextn draft cache and speculative decoding for this
  /// session. Must be called before the first Sync: the draft cache is
  /// filled during prompt processing and cannot be rebuilt lazily. A plain
  /// (non-speculative) session leaves it off so prefill skips the draft
  /// block entirely.
  void SetMtpEnabled(bool enabled);

 private:
  friend class Model;
  Session(std::shared_ptr<Model> model, std::unique_ptr<rocm::Session> state,
          std::uint32_t max_context);

  void SyncLogits();

  std::shared_ptr<Model> model_;
  std::unique_ptr<rocm::Session> state_;
  std::uint32_t max_context_;
  std::vector<std::int32_t> tokens_;
  std::vector<float> logits_;
  // Readback scratch for speculative decoding: the draft block's logits and
  // the k + 1 verify rows, copied once per round.
  std::vector<float> draft_logits_;
  std::vector<float> verify_rows_;
  std::uint32_t draft_min_{1};
  std::uint32_t draft_max_{1};
  std::uint32_t draft_k_{1};
  std::uint32_t draft_streak_{0};
  bool mtp_enabled_{false};
  bool valid_{true};
};

}  // namespace gufo::models::glm5_next

#endif  // GUFO_MODELS_GLM5_NEXT_ENGINE_HPP_
