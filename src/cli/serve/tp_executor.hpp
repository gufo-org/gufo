#ifndef GUFO_SERVER_TP_EXECUTOR_HPP_
#define GUFO_SERVER_TP_EXECUTOR_HPP_

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "src/cli/serve/text_model_runner.hpp"
#include "src/cli/serve/tp_control.hpp"

namespace gufo::server {

/// Running digest of the model calls one TP2 request made and their results.
///
/// Rank 0 adds each call as it runs it and rank 1 as it executes the same
/// instruction, through the same functions below, so a difference means the
/// ranks ran different calls or got different results. FNV-1a detects
/// accidents; it is not meant to resist an adversary.
class TpExecutionDigest {
public:
  void Add(std::uint64_t value) noexcept;
  void Add(std::span<const TextRunnerToken> tokens) noexcept;
  [[nodiscard]] std::uint64_t value() const noexcept { return value_; }

private:
  std::uint64_t value_{0xcbf29ce484222325ULL};
};

/// The instruction's operation and arguments, before the call runs. The index
/// and the `kEnd` digest are transport fields and are not part of the digest.
void DigestTpCall(TpExecutionDigest& digest, const TpInstruction& instruction,
                  std::span<const TextRunnerToken> prefill_prompt = {});
/// What a call produced, including the state's checkpoint position after it.
void DigestTpPrefill(TpExecutionDigest& digest, const TextPrefillStep& step,
                     std::size_t checkpoint);
void DigestTpAdvance(TpExecutionDigest& digest, std::size_t checkpoint);
void DigestTpDecode(TpExecutionDigest& digest, const TextDecodeStep& step,
                    std::size_t checkpoint);
/// A call that threw. Both ranks record a deterministic failure identically.
void DigestTpFailure(TpExecutionDigest& digest);

bool TpCacheAcknowledged(TpInstructionOp op) noexcept;

/// Where rank 0's instructions go: the control channel in production, a
/// capture in tests.
class TpInstructionSink {
public:
  virtual ~TpInstructionSink() = default;
  virtual void Synchronize(std::uint64_t sequence, TpInstruction instruction,
                           const std::function<void()>& local) = 0;
  /// Sends one instruction for `sequence` (zero for one that belongs to no
  /// single request) and assigns its channel-wide index, stored in `*index`.
  [[nodiscard]] virtual bool Send(std::uint64_t sequence,
                                  TpInstruction instruction,
                                  std::uint64_t* index, std::string* error) = 0;
};

class TpControlInstructionSink final : public TpInstructionSink {
public:
  explicit TpControlInstructionSink(std::shared_ptr<TpControlChannel> channel,
                                    std::shared_ptr<TpResponseBroker> broker)
      : channel_(std::move(channel)), broker_(std::move(broker)) {}
  void Synchronize(std::uint64_t sequence, TpInstruction instruction,
                   const std::function<void()>& local) override;

  [[nodiscard]] bool Send(std::uint64_t sequence, TpInstruction instruction,
                          std::uint64_t* index, std::string* error) override;

private:
  std::shared_ptr<TpControlChannel> channel_;
  std::shared_ptr<TpResponseBroker> broker_;
  std::mutex mutex_;
  std::uint64_t next_index_{0};
};

/// The collective scope of one model call. Both ranks bind the channel-wide
/// index of the call's instruction around it, so an exchange of one call can
/// never pair with an exchange of another, whichever requests are open.
/// Serving binds the communicator's operation scope; tests without one pass
/// none.
class TpCallScope {
public:
  virtual ~TpCallScope() = default;
  [[nodiscard]] virtual bool Begin(std::uint64_t index, std::string* error) = 0;
  [[nodiscard]] virtual bool End(std::uint64_t index, std::string* error) = 0;
};

/// Names the TP2 request a scheduled request belongs to. Serving passes it as
/// the request's prompt context, so `TpMirroredRunner::SetPromptContext`, which
/// the pool calls when it leases a state to the request, binds the state's
/// model calls to the request.
struct TpRequestContext final : TextPromptContext {
  explicit TpRequestContext(
      std::uint64_t sequence,
      std::shared_ptr<const TextPromptContext> images = {})
      : sequence(sequence), images(std::move(images)) {
    if (this->images != nullptr) {
      cache_identity = this->images->cache_identity;
    }
  }
  std::uint64_t sequence;
  /// The request's own prompt context from `PreparePrompt` (its images), which
  /// the model sees; rank 1 receives it with the request.
  std::shared_ptr<const TextPromptContext> images;
};

/// Rank 0's runner in a TP2 pair.
///
/// Runs every call on the wrapped runner. Just before each call that changes
/// model state it sends the call to rank 1 as an instruction, so rank 1 makes
/// the same call on the same state while rank 0 makes it; both then meet in the
/// call's exchanges. Selection (`SelectNext`, `PreviewFirstToken`) reads only
/// logits and stays local. The scheduler above is unchanged, so its stop,
/// cancellation and length decisions reach rank 1 simply as the calls rank 0
/// no longer makes. Several requests may be open: each call belongs to the
/// request whose state it runs on, and one batched advance to all of them.
///
/// Cache operations are acknowledged before the next forward. Snapshot bytes
/// stay local; only their monotonic IDs cross the control channel. It never
/// forwards a request's cancellation check to the model session: a session
/// aborting between layers would strand rank 1 inside an exchange, so
/// cancellation acts only between calls, where the scheduler already checks.
///
/// A request may reuse what another computed before rank 1 judged it; it then
/// fails if rank 1 rejects that producer (`AwaitDependencies`), and nothing a
/// rejected request left is reused (`CanReuse`), so a disagreement never
/// spreads through the cache.
class TpMirroredRunner final
    : public TextModelRunner,
      public std::enable_shared_from_this<TpMirroredRunner> {
public:
  TpMirroredRunner(
      std::shared_ptr<TextModelRunner> inner,
      std::shared_ptr<TpInstructionSink> sink,
      std::size_t snapshot_budget = std::numeric_limits<std::size_t>::max(),
      std::shared_ptr<TpCallScope> scope = {});

  /// Opens `sequence`, the request rank 1 was just told about. Its model calls
  /// are those on the state the pool leases to it, which `SetPromptContext`
  /// binds to the request through its `TpRequestContext`.
  void BeginRequest(std::uint64_t sequence);
  /// Sends `kEnd` with the request's instruction count and digest and
  /// releases its state. Returns false when the send failed.
  [[nodiscard]] bool EndRequest(std::uint64_t sequence, std::string* error);
  /// Records rank 1's verdict on an ended request, which must include
  /// `AwaitDependencies`. A state it left is not reused once rank 1 disagreed.
  void Settle(std::uint64_t sequence, bool agreed);
  /// Waits for the verdicts on the requests whose live state `sequence`
  /// reused before their verdict arrived; false when rank 1 rejected one, or
  /// a verdict did not arrive in time. The request then fails too.
  [[nodiscard]] bool AwaitDependencies(std::uint64_t sequence) const;

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override;
  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override;
  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override;
  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override;
  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override;
  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override;
  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override;
  [[nodiscard]] std::vector<std::uint8_t> EncodePromptContext(
      const TextPromptContext& context) const override;
  [[nodiscard]] std::shared_ptr<const TextPromptContext> DecodePromptContext(
      std::span<const std::uint8_t> bytes) const override;
  [[nodiscard]] TextGenerationBackend::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override;
  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override;
  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override;
  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override;
  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override;
  void PrepareBatchExecution(TextRunnerState& state) const override;
  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override;
  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override;
  void Advance(TextRunnerState& state, TextRunnerToken token) const override;
  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override;
  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override;
  void AdvanceBatch(std::span<const TextRunnerAdvance> advances) const override;
  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override;
  void PrepareCancellation(TextRunnerState& state) const override;
  std::size_t SnapshotPayloadBytes(const TextRunnerState& state) const override;
  std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override;
  void RestoreOrFork(TextRunnerState& state,
                     const TextRunnerSnapshot& snapshot) const override;
  [[nodiscard]] bool CanReuse(const TextRunnerState& state) const override;
  [[nodiscard]] bool CanReuse(
      const TextRunnerSnapshot& snapshot) const override;

private:
  class State;
  class SnapshotHandle;
  class CallScope;
  /// An open request's share of the instruction stream.
  struct Request {
    std::uint64_t count{0};
    TpExecutionDigest digest;
  };
  /// Where an instruction went: its request and channel-wide index.
  struct Sent {
    std::uint64_t sequence{0};
    std::uint64_t index{0};
  };

  void Drop(std::uint64_t id) const noexcept;
  /// Runs a cache call on both ranks and returns the request it belongs to.
  std::uint64_t CacheCall(std::uint32_t state, const TpInstruction& instruction,
                          const std::function<void()>& local) const;

  [[nodiscard]] static State& Mirrored(TextRunnerState& state);
  [[nodiscard]] static const State& Mirrored(const TextRunnerState& state);
  /// Records and sends an instruction for the request `state` is leased to.
  /// Throws when the state serves no request or the channel has failed; the
  /// call must not run then, because rank 1 would never join its exchanges.
  Sent Send(std::uint32_t state, const TpInstruction& instruction,
            std::span<const TextRunnerToken> prefill_prompt = {}) const;
  /// Sends a batched advance, recording each member in its own request.
  Sent SendBatch(TpInstruction& instruction) const;
  /// Sends a reset, which may happen between requests too. Cannot throw: the
  /// continuation cache resets states from `noexcept` paths, so a failed send
  /// is kept and fails the next call instead.
  void SendInvalidate(std::uint32_t state) const noexcept;
  void Record(std::uint64_t sequence,
              const std::function<void(TpExecutionDigest&)>& add) const;

  std::shared_ptr<TextModelRunner> inner_;
  std::shared_ptr<TpInstructionSink> sink_;
  const std::size_t snapshot_budget_;
  std::shared_ptr<TpCallScope> scope_;
  mutable std::uint64_t next_snapshot_id_{1};
  mutable std::recursive_mutex call_mutex_;
  bool multi_token_decode_{false};
  mutable std::mutex mutex_;
  mutable std::uint32_t next_state_id_{0};
  mutable std::unordered_map<std::uint64_t, Request> requests_;
  /// State id to the request it is leased to.
  mutable std::unordered_map<std::uint32_t, std::uint64_t> leases_;
  /// State id to the last request that changed it, zero after a reset.
  mutable std::unordered_map<std::uint32_t, std::uint64_t> producers_;
  /// Requests rank 1 has not judged yet, and those it disagreed with.
  mutable std::unordered_set<std::uint64_t> unsettled_;
  mutable std::unordered_set<std::uint64_t> rejected_;
  /// Request to the unjudged requests whose live state it reused.
  mutable std::unordered_map<std::uint64_t, std::unordered_set<std::uint64_t>>
      dependencies_;
  mutable std::condition_variable settled_;
  mutable std::string failure_;
};

/// Rank 1's side of a TP2 pair: executes rank 0's instructions on its own
/// states, which it creates up front in the same number and order as rank 0's
/// pool, so a state id names corresponding states on both ranks.
class TpExecutor {
public:
  TpExecutor(
      std::shared_ptr<TextModelRunner> runner, std::size_t state_count,
      std::size_t snapshot_budget = std::numeric_limits<std::size_t>::max(),
      std::shared_ptr<TpCallScope> scope = {});
  ~TpExecutor();

  using Respond = std::function<bool(const TpControlResponse&, std::string*)>;
  [[nodiscard]] std::size_t snapshot_count() const { return snapshots_.size(); }
  [[nodiscard]] std::size_t open_requests() const { return requests_.size(); }

  /// Executes one command from rank 0: `kSingle` opens a request, and an
  /// instruction makes a model call for the request it names, for the members
  /// of a batch, or (sequence 0) a reset or drop between requests. `respond`
  /// sends cache acknowledgements, and a request's verdict when its `kEnd`
  /// arrives: empty when both ranks agree, else the first difference or
  /// failure. False when the stream broke the protocol or a send failed: the
  /// pair can no longer be trusted.
  [[nodiscard]] bool Execute(const TpControlCommand& command,
                             const Respond& respond, std::string* error);

private:
  struct OwnChoice {
    bool stop{false};
    TextRunnerToken token{0};
  };
  struct Request;

  [[nodiscard]] bool Open(const TpControlCommand& begin, std::string* error);
  [[nodiscard]] bool ExecuteIdle(const TpInstruction& instruction,
                                 std::string* error);
  [[nodiscard]] bool ExecuteCall(std::uint64_t sequence, Request& request,
                                 const TpInstruction& instruction,
                                 const Respond& respond, std::string* error);
  [[nodiscard]] bool ExecuteBatch(const TpInstruction& instruction,
                                  std::string* error);
  [[nodiscard]] bool TakeIndex(const TpControlCommand& command,
                               std::string* error);
  [[nodiscard]] TextRunnerState& StateFor(std::uint32_t id);
  /// Rank 1's own greedy choice from the state's current logits.
  [[nodiscard]] OwnChoice ChooseGreedy(TextRunnerState& state) const;
  /// Binds or releases a forward's collective scope; false when the pair's
  /// collectives can no longer be trusted.
  [[nodiscard]] bool BeginScope(std::uint64_t index, std::string* error);
  [[nodiscard]] bool EndScope(std::uint64_t index, std::string* error);

  std::shared_ptr<TextModelRunner> runner_;
  std::shared_ptr<TpCallScope> scope_;
  std::vector<std::unique_ptr<TextRunnerState>> states_;
  std::unordered_map<std::uint64_t, std::unique_ptr<Request>> requests_;
  std::uint64_t next_index_{0};
  std::uint64_t last_snapshot_id_{0};
  const std::size_t snapshot_budget_;
  std::size_t snapshot_bytes_{0};
  void DropSnapshot(std::uint64_t id);
  std::unordered_map<std::uint64_t, std::unique_ptr<TextRunnerSnapshot>>
      snapshots_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_TP_EXECUTOR_HPP_
