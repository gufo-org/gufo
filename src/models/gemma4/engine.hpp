#ifndef GUFO_MODELS_GEMMA4_ENGINE_HPP_
#define GUFO_MODELS_GEMMA4_ENGINE_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/gemma4/config.hpp"
#include "src/models/gemma4/draft_policy.hpp"
#include "src/models/gemma4/tokenizer.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"
#include "src/models/qwen38_flash_next/prompt_lookup.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::gemma4 {

struct ModelWeights;
struct DraftWeights;
namespace rocm {
class DeviceModel;
class Executor;
struct KvCache;
}  // namespace rocm

/// Draft tokens per cycle when the caller sets none: the longest chain, which
/// the calibrated policy shortens by itself (rows cost little up to 16).
inline constexpr std::uint32_t kDefaultDraftTokens = 15;

struct ModelOptions {
  /// Largest context any session may use.
  std::uint32_t max_context = 4096;
  /// Tokens per prefill forward; sets the sliding-window ring size.
  std::uint32_t prefill_chunk = 2048;
  /// Rows of one forward that may request logits (qualification only needs
  /// more than one).
  std::uint32_t max_logit_rows = 64;
  /// Optional `gemma4-assistant` MTP drafter; empty leaves speculation off.
  std::string mtp_model_path;
  /// Draft tokens per cycle (at most rocm::kMaxDraftTokens).
  std::uint32_t draft_tokens = kDefaultDraftTokens;
  /// How many of them a cycle verifies.
  DraftPolicy draft_policy = DraftPolicy::kCalibrated;
  /// Drafts every cycle verifies before a policy may stop the chain.
  std::uint32_t min_draft_tokens = 1;
  /// Where the calibrated policy learns.
  DraftCalibrationScope draft_calibration = DraftCalibrationScope::kShared;
};

class Session;
/// Device memory and host staging of snapshots, one per model (engine.cpp).
struct SnapshotMemory;

/// One image of a prompt: its soft-token rows, and an identity that tells it
/// apart from any other image behind the same tokens.
struct ImageSpan {
  std::uint32_t offset{0};  ///< first soft token
  std::uint32_t rows{0};
  std::array<std::uint8_t, 32> identity{};
  bool operator==(const ImageSpan&) const = default;
};
/// Device embeddings ([rows][hidden] FP32) of prompt image `index`, valid
/// until Sync returns. Sync asks only for images it must evaluate.
using ImageEmbeddings = std::function<const float*(std::size_t index)>;

/// One session's context: tokens, the KV rows later tokens can still attend
/// (every global row, the last window-1 sliding rows in logical order), the
/// frontier hidden state and the last logits. The KV rows and hidden state
/// stay in device memory: on this unified-memory APU a device copy runs near
/// memory bandwidth, while a fresh host buffer pays first-touch page faults
/// (1.34 GB: 30 ms against 220 ms). Rows of whole position blocks live in
/// immutable blocks that snapshots of one session's history share, so a
/// snapshot copies only the blocks its session wrote since the last one and
/// the rows around them. The payload reads back in the layout
/// Session::RestoreSnapshot(payload) accepts.
class SessionSnapshot final {
public:
  using Sink = std::function<void(std::span<const std::uint8_t>)>;

  ~SessionSnapshot() = default;
  SessionSnapshot(const SessionSnapshot&) = delete;
  SessionSnapshot& operator=(const SessionSnapshot&) = delete;

  [[nodiscard]] std::uint64_t SizeBytes() const noexcept {
    return head_.size() + body_bytes_ + tail_.size();
  }
  /// Writes the payload; false on a size mismatch or a failed device copy.
  [[nodiscard]] bool CopyTo(std::span<std::uint8_t> destination) const;
  /// Hands the payload to `sink` in order, in pieces of at most 64 MiB;
  /// false on a failed device copy.
  [[nodiscard]] bool Stream(const Sink& sink) const;

private:
  SessionSnapshot() = default;
  /// Device bytes of the body, in payload order.
  struct Piece {
    const std::uint8_t* at;
    std::uint64_t bytes;
  };

  std::vector<std::uint8_t> head_;  // header, tokens, image records
  /// Device rows of position blocks [0, global_.size()) of the global layers
  /// and [sliding_first_, + sliding_.size()) of the sliding ones.
  std::vector<std::shared_ptr<const std::uint8_t>> global_;
  std::vector<std::shared_ptr<const std::uint8_t>> sliding_;
  std::uint32_t sliding_first_{0};
  /// Device: the rows outside those blocks, then the hidden state.
  std::shared_ptr<const std::uint8_t> rest_;
  std::vector<Piece> body_;
  std::uint64_t body_bytes_{0};
  std::vector<std::uint8_t> tail_;  // logits
  std::shared_ptr<SnapshotMemory> memory_;
  friend class Session;
};

/// Gufo-owned API over the Gemma 4 ROCm runtime: one resident model and any
/// number of sessions with independent context state. Forwards of different
/// sessions are serialized on the model's executor.
class Model final : public std::enable_shared_from_this<Model> {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::uint32_t max_context, std::string* error_msg = nullptr);

  /// Tokenizes text with special tokens parsed and no BOS; rendered chat
  /// prompts carry their own `<bos>`.
  [[nodiscard]] std::vector<TokenId> Tokenize(std::string_view text) const;
  [[nodiscard]] std::string Decode(std::span<const TokenId> tokens) const;
  [[nodiscard]] std::string TokenText(TokenId token) const;
  [[nodiscard]] bool IsStopToken(TokenId token) const noexcept;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return options_.max_context;
  }
  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] const Tokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] const core::GgufReader& reader() const noexcept {
    return *reader_;
  }
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  [[nodiscard]] std::size_t SessionBytes(std::uint32_t context) const noexcept;
  /// Sliding-window ring slots per session (window + prefill chunk).
  [[nodiscard]] std::size_t SessionRingSlots() const noexcept;
  [[nodiscard]] std::string ModelName() const;
  [[nodiscard]] bool HasMtp() const noexcept {
    return draft_weights_ != nullptr;
  }
  /// After each layer of every forward, `rows` points at that layer's
  /// residual rows on the device ([count][hidden], complete when called):
  /// the target features a DFlash drafter reads. For tools; unset in serving.
  using LayerTapSink = std::function<void(
      std::uint32_t layer, const float* rows, std::uint32_t count)>;
  void SetTapSink(LayerTapSink sink);
  [[nodiscard]] std::uint32_t DraftTokens() const noexcept {
    return options_.draft_tokens;
  }
  [[nodiscard]] const ModelOptions& options() const noexcept {
    return options_;
  }

private:
  Model();

  ModelOptions options_;
  std::shared_ptr<core::GgufReader> reader_;
  std::unique_ptr<ModelWeights> weights_;
  std::shared_ptr<core::GgufReader> draft_reader_;
  std::unique_ptr<DraftWeights> draft_weights_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<rocm::DeviceModel> device_;
  std::unique_ptr<rocm::Executor> executor_;
  std::mutex mutex_;
  /// Calibrated-policy tables shared by every session ([greedy, sampled]
  /// drafts, then their siblings); guarded by mutex_.
  std::array<DraftCalibration, 4> calibration_;
  std::shared_ptr<SnapshotMemory> snapshot_memory_;

  friend class Session;
};

/// One conversation's context. Sync feeds a prompt reusing the longest
/// common prefix the sliding-window ring can still serve; Evaluate appends
/// one token. The logits of the last evaluated token are kept on the host.
class Session final {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  [[nodiscard]] bool Sync(std::span<const TokenId> prompt,
                          std::string* error_msg = nullptr);
  /// Sync with images: `images` (ordered, each followed by a text token)
  /// take their rows from `embed`. Reuse stops at the first image whose
  /// identity or placement changed, and an image is always evaluated whole.
  /// `lookahead`: text tokens expected to follow `prompt` in the next Sync
  /// (the chat template's generation prompt after a cache boundary). Up to
  /// kSplitRows of them are evaluated in the prompt's last forward instead
  /// of a separate one, while the session still ends at `prompt`, so a
  /// snapshot taken now holds exactly `prompt`; the next Sync adopts them if
  /// it asks for exactly `prompt` followed by `lookahead`.
  [[nodiscard]] bool Sync(std::span<const TokenId> prompt,
                          std::span<const ImageSpan> images,
                          const ImageEmbeddings& embed,
                          std::string* error_msg = nullptr,
                          std::span<const TokenId> lookahead = {});
  [[nodiscard]] bool Evaluate(TokenId token, std::string* error_msg = nullptr);
  /// Appends `tokens` and returns the logits of every row ([n][vocab]);
  /// used for teacher-forced qualification.
  [[nodiscard]] bool EvaluateAll(std::span<const TokenId> tokens,
                                 std::vector<float>* logits,
                                 std::string* error_msg = nullptr);

  struct DecodeResult {
    std::vector<TokenId> tokens;
    bool stop{false};
  };
  /// Emits the next tokens. With the MTP drafter one call drafts, verifies
  /// and emits the accepted prefix plus the target's own next sample, all
  /// drawn with `sampler`, so greedy decoding reproduces autoregressive
  /// output exactly and sampled decoding keeps the target distribution.
  /// Without it (or with a one-token budget) one token is emitted. The last
  /// emitted token stays pending until the next call or a Sync.
  [[nodiscard]] bool DecodeStep(std::size_t max_tokens,
                                sampling::SamplerState& sampler,
                                DecodeResult* result,
                                std::string* error_msg = nullptr,
                                bool stop_at_eos = true);
  /// One session's work in DecodeBatch.
  struct BatchDecode {
    Session* session;
    std::size_t max_tokens;
    sampling::SamplerState* sampler;
    DecodeResult* result;
    bool stop_at_eos{true};
    std::string error;  ///< set when this session failed
  };
  /// DecodeStep for several sessions of one model with a single verification
  /// forward of at most 16 rows, so every row keeps the arithmetic of its own
  /// session's decode. Each session drafts at most 16 / sessions - 1 tokens,
  /// and calibrated sessions share the rows that split leaves over. Returns
  /// false if
  /// any session failed (see BatchDecode::error); the others completed.
  static bool DecodeBatch(std::span<BatchDecode> decodes);
  /// Evaluate for several sessions of one model in a single forward.
  static bool EvaluateBatch(std::span<Session* const> sessions,
                            std::span<const TokenId> tokens,
                            std::string* error_msg = nullptr);

  struct SpeculativeStats {
    std::uint64_t cycles{0};
    std::uint64_t verified{0};  ///< cycles that verified at least one draft
    std::uint64_t drafted{0};   ///< MTP drafts and copied tokens
    std::uint64_t accepted{0};  ///< MTP drafts and copied tokens
    std::uint64_t copied{0};    ///< prompt-lookup tokens among `drafted`
    std::uint64_t copied_accepted{0};
    /// Siblings verified beside MTP drafts, and those whose token and
    /// following one were emitted.
    std::uint64_t siblings{0};
    std::uint64_t siblings_accepted{0};
  };
  [[nodiscard]] const SpeculativeStats& Statistics() const noexcept {
    return stats_;
  }

  [[nodiscard]] std::span<const float> Logits() const noexcept {
    return valid_ ? std::span<const float>(logits_) : std::span<const float>{};
  }
  [[nodiscard]] std::span<const TokenId> Tokens() const noexcept {
    return tokens_;
  }
  /// Images inside Tokens().
  [[nodiscard]] std::span<const ImageSpan> Images() const noexcept {
    return images_;
  }
  [[nodiscard]] std::uint32_t Position() const noexcept {
    return static_cast<std::uint32_t>(tokens_.size());
  }
  [[nodiscard]] std::uint32_t ContextSize() const noexcept;
  /// True after DecodeStep: the last emitted token is not yet evaluated, so
  /// Logits() still predicts that token rather than the next one.
  [[nodiscard]] bool HasPendingToken() const noexcept {
    return pending_.has_value();
  }
  [[nodiscard]] bool IsValid() const noexcept { return valid_; }
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;
  void Reset();

  /// Compatibility version; bump on payload or inference arithmetic changes.
  static constexpr std::uint32_t kSnapshotPayloadVersion = 3;
  [[nodiscard]] std::uint64_t SnapshotBytes() const;
  [[nodiscard]] std::unique_ptr<SessionSnapshot> SaveSnapshot(
      std::string* error_msg = nullptr) const;
  [[nodiscard]] bool RestoreSnapshot(const SessionSnapshot& snapshot,
                                     std::string* error_msg = nullptr);
  [[nodiscard]] bool RestoreSnapshot(std::span<const std::uint8_t> payload,
                                     std::string* error_msg = nullptr);

private:
  /// Reads a payload head (header, tokens, image records) of a `total`-byte
  /// payload into tokens_ and images_; returns its size, or 0 after Reset.
  std::size_t ReadHead(std::span<const std::uint8_t> head, std::uint64_t total,
                       std::string* error_msg);
  /// Drops held blocks whose rows the cache rewrote since the last call
  /// (KvCache::written_from/to).
  void SettleHeld() const;

  Session(std::shared_ptr<Model> model, std::unique_ptr<rocm::KvCache> cache);
  /// Evaluates tokens_[begin, end) in prefill chunks that never split an
  /// image; the last row's logits land in logits_. `lookahead` rows follow
  /// in the last forward: their last row's logits and hidden state are
  /// stashed (ahead_), the session itself ends at tokens_.
  bool Extend(std::size_t begin, std::string* error_msg,
              const ImageEmbeddings& embed = {},
              std::span<const TokenId> lookahead = {});
  /// Forgets rows a Sync computed past the frontier.
  void DropLookahead() noexcept;

  /// A cycle's drafting state (engine.cpp).
  struct CycleDraft;
  /// The sessions of a batch drafting together: how many, and their
  /// verification rows and expected tokens so far.
  struct DraftShare {
    std::uint32_t sessions{0};
    std::uint32_t rows{0};
    float expected{0.0F};
    /// Each session's even share of the forward (rows past its pending
    /// one), and the rows the even split leaves over (16 - sessions x (share
    /// + 1)): calibrated sessions draw drafts, copies and siblings past their
    /// share from these.
    std::uint32_t fair{0};
    std::uint32_t spare{0};
  };
  /// One decode cycle: Begin emits the pending token if needed and drafts
  /// (rows = pending plus drafts), the caller verifies `rows` at `position`,
  /// and Finish accepts from those rows' logits.
  struct Cycle {
    std::size_t max_tokens{0};
    sampling::SamplerState* sampler{nullptr};
    DecodeResult* result{nullptr};
    bool stop_at_eos{true};
    bool done{false};  ///< nothing to verify: the result is complete
    std::uint32_t position{0};
    std::vector<TokenId> rows;
    bool sampled{false};
    std::vector<qwen38_flash_next::MtpProposal> proposals;
    std::size_t copied{0};
    /// The drafter's next choice beside MTP drafts 1..siblings.size(),
    /// verified in rows after the chain (sampled: drawn from the draft's
    /// proposal without it), and each one's calibration signal.
    std::vector<TokenId> siblings;
    std::vector<float> sibling_signals;
    /// Calibrated policy: the signal of each verified MTP draft.
    std::vector<float> signals;
    /// Calibrated policy: tokens the cycle is expected to emit and its
    /// drafter time, for the sessions after it in a batch.
    float expected{1.0F};
    float draft_ms{0.0F};
    /// Set while a batch's drafter chains are still to run (DecodeBatch).
    std::unique_ptr<CycleDraft> draft;
  };
  /// `others` describes the rest of a batched forward. With `share` (a
  /// batch drafting together) the drafting is left in `cycle.draft` for the
  /// caller to run, then FinishDraft; alone it runs here.
  bool BeginCycle(Cycle& cycle, std::uint32_t draft_limit,
                  std::string* error_msg, const DraftBatch& others = {},
                  DraftShare* share = nullptr);
  /// Appends a finished chain's drafts and copies to the cycle's rows.
  void FinishDraft(Cycle& cycle);
  void FinishCycle(Cycle& cycle, std::span<const float> logits,
                   std::uint32_t first_hidden_row);

  std::shared_ptr<Model> model_;
  std::unique_ptr<rocm::KvCache> cache_;
  std::vector<TokenId> tokens_;
  std::vector<ImageSpan> images_;
  /// Prompt-lookup index over tokens_; cleared whenever tokens_ is rewritten
  /// rather than appended to.
  qwen38_flash_next::PromptLookup lookup_;
  std::vector<float> logits_;
  bool valid_{false};
  /// Emitted but not yet evaluated; its predecessor's hidden state is the
  /// cache's frontier hidden.
  std::optional<TokenId> pending_;
  /// Tokens evaluated past tokens_ by Sync's lookahead: their KV rows are in
  /// the cache, the last one's hidden state in cache_->ahead_hidden.
  std::vector<TokenId> ahead_;
  std::vector<float> ahead_logits_;
  /// Snapshot blocks whose rows the cache still holds, by position block
  /// (expired where it does not), so the next snapshot shares them and a
  /// restore skips them.
  mutable std::vector<std::weak_ptr<const std::uint8_t>> held_global_;
  mutable std::vector<std::weak_ptr<const std::uint8_t>> held_sliding_;
  SpeculativeStats stats_;
  /// Calibrated-policy tables of a request-scoped session ([greedy,
  /// sampled] drafts, then their siblings), reset by Sync.
  std::array<DraftCalibration, 4> calibration_;
  [[nodiscard]] DraftCalibration& Calibration(bool sampled,
                                              bool sibling = false);

  friend class Model;
};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_ENGINE_HPP_
