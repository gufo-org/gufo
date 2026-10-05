#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ENGINE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/core/session_mode.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen/vision/encoder.hpp"
#include "src/models/qwen/vision/prompt.hpp"
#include "src/models/qwen38_flash_next/config.hpp"
#include "src/models/qwen38_flash_next/mtp_policy.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::qwen38_flash_next {

struct ModelWeights;
struct MtpWeights;
class NgramTable;
struct MtpCandidateLogits;
namespace rocm {
class DeviceModel;
class Executor;
class Session;
class SnapshotState;
}  // namespace rocm

struct ModelOptions {
  std::uint32_t max_context = 4096;
  /// Optional MTP draft sidecar (`mtp-...-shared-*.gguf`). Empty leaves
  /// speculative decoding off.
  std::string mtp_model_path;
  /// Maximum proposals per cycle; acceptance history selects the length.
  std::uint32_t max_draft_tokens = kMaxMtpDraftTokens;
  std::string vision_model_path;
  /// Fixed serving capacity used by the calibrated MTP cost model. Keeping
  /// it independent of scheduler timing preserves seeded request replay.
  std::uint32_t decode_concurrency = 1;
};

class Session;
class SessionSnapshot;

/// Gufo-owned API over the ROCm runtime: one resident model, any number of
/// sessions with shared projection batches and independent context state.
class Model final : public std::enable_shared_from_this<Model> {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      core::SessionMode mode, std::uint32_t max_context,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::vector<std::int32_t> Tokenize(std::string_view text) const;
  [[nodiscard]] std::string Decode(std::span<const std::int32_t> tokens) const;
  [[nodiscard]] std::string TokenText(std::int32_t token) const;
  [[nodiscard]] std::int32_t EosToken() const noexcept;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept;
  [[nodiscard]] std::uint32_t PrefillCapacity() const noexcept;
  [[nodiscard]] std::uint32_t PrefillThroughCapacity() const noexcept;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return options_.max_context;
  }
  [[nodiscard]] bool HasMtp() const noexcept;
  [[nodiscard]] std::uint32_t DecodeConcurrency() const noexcept {
    return options_.decode_concurrency;
  }
  [[nodiscard]] std::string ModelName() const;
  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] const tokenization::QwenTokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  /// Worst-case private device state, including the configured rollback cap.
  [[nodiscard]] std::size_t SessionBytes(core::SessionMode mode,
                                         std::uint32_t context) const noexcept;
  [[nodiscard]] std::size_t DeferredScratchBytes() const;
  [[nodiscard]] const std::shared_ptr<qwen::vision::Encoder>& VisionEncoder()
      const noexcept {
    return vision_;
  }

private:
  Model() = default;

  ModelOptions options_;
  std::shared_ptr<core::GgufReader> reader_;
  std::shared_ptr<core::GgufReader> mtp_reader_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<MtpWeights> mtp_weights_;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer_;
  std::unique_ptr<NgramTable> ngram_;
  std::unique_ptr<rocm::DeviceModel> device_;
  std::unique_ptr<rocm::Executor> executor_;
  std::shared_ptr<qwen::vision::Encoder> vision_;
  MtpBatchController batch_policy_;

  friend class Session;
};

/// One conversation's context. Sync feeds a prompt (reusing whatever prefix
/// the session already holds), Evaluate appends one token, DecodeStep
/// runs a draft-verify cycle. The logits of the last token are kept.
class Session final {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Makes the session state equal to `prompt`: keeps the longest common
  /// prefix with the current tokens, reprocesses the rest.
  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr);
  /// Prefills through a stable text boundary in the same forward, returning
  /// a checkpoint of that boundary while the live session reaches prompt.
  [[nodiscard]] bool SyncThrough(std::span<const std::int32_t> prompt,
                                 std::uint32_t boundary,
                                 std::unique_ptr<SessionSnapshot>* checkpoint,
                                 std::string* error_msg = nullptr,
                                 double* capture_ms = nullptr);
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr);
  struct DecodeResult {
    std::vector<std::int32_t> tokens;
    bool stop{false};
  };
  /// Greedy decoding verifies deterministic drafts. Sampled decoding draws
  /// compact MTP proposals and uses target/draft rejection with a residual
  /// correction. History contains only committed tokens; stochastic RNG
  /// consumption includes proposal and verification draws. A
  /// one-token budget or a model without MTP uses ordinary decoding.
  /// Benchmarks may continue past EOS by setting stop_at_eos to false.
  [[nodiscard]] bool DecodeStep(std::size_t max_tokens,
                                sampling::SamplerState& sampler,
                                DecodeResult* result,
                                std::string* error_msg = nullptr,
                                bool stop_at_eos = true);
  struct BatchOutcome {
    bool completed{false};
    std::string error{};
  };
  struct DecodeRequest {
    Session* session;
    std::size_t max_tokens;
    sampling::SamplerState* sampler;
    DecodeResult* result;
    bool stop_at_eos{true};
    BatchOutcome* outcome{nullptr};
  };
  /// A false return may include completed peers. Inspect each outcome; only
  /// sessions whose persistent state was partly mutated become invalid.
  [[nodiscard]] static bool DecodeBatch(std::span<const DecodeRequest> requests,
                                        std::string* error_msg = nullptr);
  struct AdvanceRequest {
    Session* session;
    std::int32_t token;
    BatchOutcome* outcome{nullptr};
  };
  [[nodiscard]] static bool EvaluateBatch(
      std::span<const AdvanceRequest> requests,
      std::string* error_msg = nullptr);
  [[nodiscard]] std::span<const float> Logits() const noexcept {
    return valid_ ? std::span<const float>(logits_) : std::span<const float>{};
  }
  [[nodiscard]] std::uint32_t Position() const noexcept;
  [[nodiscard]] std::uint32_t ContextSize() const noexcept;
  [[nodiscard]] std::span<const std::int32_t> Tokens() const noexcept {
    return valid_ ? std::span<const std::int32_t>(tokens_)
                  : std::span<const std::int32_t>{};
  }
  void Reset();
  /// Cancellation is checked at model, layer and snapshot-copy boundaries.
  void SetCancellationCheck(std::function<bool()> check);
  [[nodiscard]] bool IsValid() const noexcept { return valid_; }
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;
  void ConfigureVision(std::shared_ptr<const qwen::vision::Prompt> prompt);
  /// A new request reusing cached context starts its own acceptance history.
  void ResetDraftPolicy() noexcept { draft_length_.Reset(); }

  struct SpeculativeStats {
    std::uint64_t cycles{0};
    std::uint64_t drafted{0};
    std::uint64_t accepted{0};
  };
  [[nodiscard]] const SpeculativeStats& Statistics() const noexcept {
    return stats_;
  }

  /// Compatibility version; bump on payload or inference arithmetic changes.
  static constexpr std::uint32_t kSnapshotPayloadVersion = 16;
  /// Bytes a snapshot of the current context occupies.
  [[nodiscard]] std::uint64_t SnapshotBytes() const;
  [[nodiscard]] std::uint64_t PrefillCheckpointBytes(
      std::uint32_t position) const;
  /// Captures the whole context and speculative length controller. Mutable
  /// state is copied immediately; append-only cache rows borrow this session
  /// until a rewind, reset, destruction, fork or byte reader needs them.
  enum class SnapshotMode { kBorrowed, kMaterialized };
  [[nodiscard]] std::unique_ptr<SessionSnapshot> SaveSnapshot(
      std::string* error_msg = nullptr,
      SnapshotMode mode = SnapshotMode::kBorrowed) const;
  [[nodiscard]] bool OwnsSnapshot(
      const SessionSnapshot& snapshot) const noexcept;
  /// Replaces this session's context with a snapshot of the same model.
  /// Image snapshots require matching consumed images via ConfigureVision;
  /// the attached prompt may append future images. Pixels are not serialized.
  /// Text snapshots clear images; reattach the request after restoring.
  [[nodiscard]] bool RestoreSnapshot(const SessionSnapshot& snapshot,
                                     std::string* error_msg = nullptr);
  [[nodiscard]] bool RestoreSnapshot(std::span<const std::uint8_t> payload,
                                     std::string* error_msg = nullptr);

private:
  friend class Model;
  Session(std::shared_ptr<Model> model, std::unique_ptr<rocm::Session> session);

  bool Feed(std::span<const std::int32_t> tokens, std::string* error_msg,
            bool prefill = false, std::uint32_t boundary = 0,
            std::unique_ptr<SessionSnapshot>* checkpoint = nullptr,
            double* capture_ms = nullptr);
  bool SyncImpl(std::span<const std::int32_t> prompt, std::string* error_msg,
                std::uint32_t boundary,
                std::unique_ptr<SessionSnapshot>* checkpoint,
                double* capture_ms = nullptr);
  std::unique_ptr<SessionSnapshot> SaveSnapshotImpl(
      const rocm::Session& state, std::span<const std::int32_t> tokens,
      std::span<const float> logits, std::uint32_t hidden_rows,
      std::string* error_msg, SnapshotMode mode) const;
  bool RestoreSnapshotImpl(std::span<const std::uint8_t> payload,
                           const rocm::SnapshotState* deferred,
                           std::string* error_msg);
  /// Trunk rows the draft block may still read: [hidden_base_, size).
  [[nodiscard]] std::uint32_t KeptHiddenRows() const noexcept;
  bool DraftCatchUp(std::int32_t next_token, bool propose,
                    std::string* error_msg,
                    MtpCandidateLogits* candidates = nullptr);
  bool DraftReplay(std::int32_t next_token, std::vector<std::int32_t>* replay,
                   std::int32_t* hidden_row, std::string* error_msg) const;
  static bool DraftCatchUpBatch(std::span<const AdvanceRequest> requests,
                                std::string* error_msg);
  template<class Request>
  static bool RunIsolatedBatch(std::span<const Request> requests,
                               std::string* error_msg);
  static bool DecodeBatchImpl(std::span<const DecodeRequest> requests,
                              std::string* error_msg);
  static bool EvaluateBatchImpl(std::span<const AdvanceRequest> requests,
                                std::string* error_msg);
  struct PendingDecode;
  bool PrepareDecode(const DecodeRequest& request, PendingDecode* pending,
                     std::string* error_msg, bool defer_head = false,
                     std::optional<std::uint32_t> batch_drafts = {});
  static void AppendDraft(PendingDecode& pending);
  bool FinishDecode(const DecodeRequest& request, const PendingDecode& pending,
                    std::string* error_msg);

  std::shared_ptr<Model> model_;
  std::unique_ptr<rocm::Session> session_;
  std::vector<std::int32_t> tokens_;
  std::vector<float> logits_;
  std::int32_t draft_token_{0};
  std::vector<float> verify_logits_;
  std::uint32_t hidden_base_{0};  ///< first position whose hidden row is kept
  MtpLengthController draft_length_;
  SpeculativeStats stats_;
  std::shared_ptr<const qwen::vision::Prompt> image_prompt_;
  [[nodiscard]] std::span<const std::uint8_t> ImageIdentity(
      std::size_t token_count) const;
  bool valid_{true};
  [[nodiscard]] bool MtpEnabled() const noexcept;
};

/// Immutable session checkpoint. Committed cache rows may still reside in
/// the source session. The complete byte form restores in memory and on disk.
class SessionSnapshot final {
public:
  ~SessionSnapshot();
  SessionSnapshot(const SessionSnapshot&) = delete;
  SessionSnapshot& operator=(const SessionSnapshot&) = delete;
  SessionSnapshot(SessionSnapshot&&) = delete;
  SessionSnapshot& operator=(SessionSnapshot&&) = delete;

  /// Complete payload size, including borrowed rows, for cache admission.
  [[nodiscard]] std::uint64_t SizeBytes() const noexcept { return size_; }
  /// Frozen device storage, counting each shared backing block once.
  /// Borrowed live rows and allocator reservations are excluded.
  [[nodiscard]] std::uint64_t DeviceBytes() const;
  /// Materializes borrowed rows; may throw on a failed device transfer.
  [[nodiscard]] std::span<const std::uint8_t> bytes() const;
  [[nodiscard]] bool CopyTo(std::span<std::uint8_t> destination) const noexcept;

private:
  explicit SessionSnapshot(std::uint64_t size);

  std::shared_ptr<std::uint8_t[]> data_;
  std::uint64_t size_{0};
  std::shared_ptr<rocm::SnapshotState> deferred_;

  friend class Session;
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_ENGINE_HPP_
