#ifndef GUFO_SERVER_TP_CONTROL_HPP_
#define GUFO_SERVER_TP_CONTROL_HPP_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"

namespace gufo::server {

struct TpControlConfig {
  std::uint64_t snapshot_budget_bytes{0};
  std::uint32_t rank{0};
  std::uint32_t world_size{1};
  std::uint32_t max_context{4096};
  std::uint32_t max_draft_tokens{7};
  bool use_mtp{false};
  std::string auth_token;
  std::uint32_t prefill_chunk_tokens{512};
  /// Model states per rank: both ranks create the same number, in the same
  /// order, so an instruction's state id names corresponding states.
  std::uint32_t sessions{1};
  /// Both ranks load the vision encoder, or neither: rank 1 encodes the images
  /// of every request itself.
  bool vision{false};
  /// Both ranks keep a disk cache (`--cache-disk`), or neither.
  bool disk_cache{false};
};

enum class TpControlCommandKind : std::uint8_t {
  /// A request: its prompt, budget, cache request and sampling configuration.
  kSingle = 1,
  /// One model call for rank 1 to execute: rank 0's runner sends it just
  /// before running the same call itself, so both ranks make identical model
  /// calls in identical order. See `TpInstruction`.
  kInstruction = 2,
};

/// The model call an instruction asks rank 1 to make.
enum class TpInstructionOp : std::uint8_t {
  kNone = 0,
  /// Reset the state's model session. The continuation cache decides this, so
  /// it may arrive between requests (sequence 0) as well as within one.
  kInvalidate = 1,
  /// `Prefill(state, prompt[0, prompt_size), offset, count)`, where the prompt
  /// is the request's `kSingle` prompt.
  kPrefill = 2,
  /// `Advance(state, token)`.
  kAdvance = 3,
  /// `DecodeStep(state, count)`: one MTP draft-verify cycle. It carries rank
  /// 0's sampler draw state (`rng`, `pending`) from just before the call, so
  /// rank 1's draws, and therefore its decisions, are rank 0's.
  kDecode = 4,
  /// The request is over. `count` is the number of instructions rank 0 sent
  /// for it and `digest` its execution digest; rank 1 compares both.
  kEnd = 5,
  kSnapshot = 6,
  kDrop = 7,
  kRestore = 8,
  kReuse = 9,
  kCancelPrepare = 10,
  /// `AdvanceBatch` over `batch`, in order: one forward advances every listed
  /// state by its token. It belongs to the members' requests rather than to
  /// one, so it is sent with sequence 0 and each member names its request.
  kAdvanceBatch = 11,
  /// `SetPromptContext(state, context)`: binds the state to the request, and
  /// the model state to the request's prompt context (its images, if any), as
  /// the pool does whenever it leases a state; with sequence 0 it clears it.
  kPromptContext = 12,
  /// Persist the snapshot `snapshot_id` to rank 1's disk cache under
  /// `file_key`, in the background. Rank 0 writes its own half of the same
  /// snapshot to its disk cache; a failed or skipped write on rank 1 only
  /// turns a later restore into a cache miss. Sent with sequence 0.
  kPersist = 13,
  /// Restore the state from rank 1's disk-cache file `file_key`, as rank 0
  /// restores its own half. Acknowledged; a failure makes both ranks treat the
  /// entry as a miss.
  kRestoreDisk = 14,
};

/// One member of a batched instruction: a request, the state it leases, and
/// the token that state advances by.
struct TpBatchMember {
  std::uint64_t sequence{0};
  std::uint32_t state{0};
  std::int32_t token{0};

  bool operator==(const TpBatchMember&) const = default;
};

struct TpInstruction {
  TpInstructionOp op{TpInstructionOp::kNone};
  /// Channel-wide and monotonic, so rank 1 detects a lost or repeated frame.
  std::uint64_t index{0};
  std::uint32_t state{0};
  std::int32_t token{0};
  std::uint32_t offset{0};
  std::uint32_t count{0};
  std::uint32_t prompt_size{0};
  std::uint64_t digest{0};
  /// kDecode only: the sampler's RNG state and pending deferred draw (-1 for
  /// none) before the call.
  std::uint64_t rng{0};
  std::int32_t pending{-1};
  std::uint64_t snapshot_id{0};
  /// kAdvanceBatch only: two to eight members on distinct states.
  std::vector<TpBatchMember> batch;
  /// kPersist and kRestoreDisk only: names the snapshot's file on both ranks.
  std::uint64_t file_key{0};

  bool operator==(const TpInstruction&) const = default;
};

enum class TpControlResponseKind : std::uint8_t {
  /// A request's result.
  kSingle = 1,
  /// Completion of a cache operation, before either rank can forward again.
  kInstruction = 2,
};

struct TpControlCommand {
  /// The request: the response correlation key, and the request an
  /// instruction belongs to.
  std::uint64_t sequence{0};
  // kSingle only: the request's fields.
  std::uint32_t max_tokens{0};
  bool cache_prompt{false};
  std::uint32_t cache_prefix_tokens{0};
  std::vector<std::int32_t> prompt_tokens;
  std::string client_id;
  /// The request's prompt context (its images), as the runner encodes it;
  /// empty for a text-only request.
  std::vector<std::uint8_t> prompt_context;
  TpControlCommandKind kind{TpControlCommandKind::kSingle};
  /// kSingle only: the request's sampling configuration. Rank 1 builds the
  /// same sampler for multi-token decoding, and checks rank 0's tokens against
  /// its own choice only when this is greedy.
  sampling::SamplingConfig sampling{};
  /// kInstruction only. An instruction belongs to the `kSingle` request named
  /// by `sequence`, or with sequence 0 to none (a reset or drop between
  /// requests) or to the requests its batch names, and carries no request
  /// fields of its own.
  TpInstruction instruction{};
};

struct TpControlResponse {
  /// kInstruction only: the acknowledged instruction.
  std::uint64_t instruction_index{0};
  /// Response broker correlation key.
  std::uint64_t sequence{0};
  // kSingle only: the request's result.
  std::vector<std::int32_t> tokens;
  std::uint64_t draft_tokens{0};
  std::uint64_t draft_accepted_tokens{0};
  std::uint32_t cached_prompt_tokens{0};
  std::uint64_t cache_snapshot_bytes{0};
  std::string error;
  TpControlResponseKind kind{TpControlResponseKind::kSingle};
};

[[nodiscard]] bool ValidateTpControlCommand(const TpControlCommand& command,
                                            std::string* error);
[[nodiscard]] bool ValidateTpControlResponse(const TpControlResponse& response,
                                             std::string* error);

/// Default bound on a blocking control-channel send, and on a receive before
/// the handshake completes. It is generous because a rank waits in the
/// handshake while its peer is still loading the model.
inline constexpr std::chrono::milliseconds kTpControlIoTimeout =
    std::chrono::hours(24);

/// Ordered, versioned TCP control channel between the TP=2 ranks. Tensor
/// payload uses the RDMA communicator; this channel carries only requests,
/// instructions, responses, and lifecycle handshakes.
///
/// `io_timeout` bounds every blocking send, and every receive until the
/// handshake succeeds. After the handshake receives have no timeout: a worker
/// waits for its next command, and rank 0's reader for its next response, for
/// as long as the server stays idle. TCP keepalive still reports a peer whose
/// host died.
class TpControlChannel final {
public:
  [[nodiscard]] static std::shared_ptr<TpControlChannel> Listen(
      std::uint16_t port, std::string* error,
      std::chrono::milliseconds io_timeout = kTpControlIoTimeout);
  [[nodiscard]] static std::shared_ptr<TpControlChannel> Connect(
      const std::string& host, std::uint16_t port, std::string* error,
      std::chrono::milliseconds io_timeout = kTpControlIoTimeout);

  ~TpControlChannel();

  TpControlChannel(const TpControlChannel&) = delete;
  TpControlChannel& operator=(const TpControlChannel&) = delete;

  [[nodiscard]] bool Handshake(const TpControlConfig& config,
                               std::string* error);
  [[nodiscard]] bool SendCommand(const TpControlCommand& command,
                                 std::string* error);
  [[nodiscard]] bool ReceiveCommand(TpControlCommand* command,
                                    std::string* error);
  [[nodiscard]] bool SendResponse(const TpControlResponse& response,
                                  std::string* error);
  [[nodiscard]] bool ReceiveResponse(TpControlResponse* response,
                                     std::string* error);
  /// Wake a blocked directional reader/writer during broker shutdown.
  void Interrupt() noexcept;

  [[nodiscard]] std::uint64_t snapshot_budget_bytes() const noexcept {
    return snapshot_budget_bytes_;
  }
  [[nodiscard]] std::uint16_t port() const noexcept;
  [[nodiscard]] std::uint32_t rank() const noexcept;
  [[nodiscard]] std::uint32_t world_size() const noexcept;

private:
  TpControlChannel(int fd, std::uint32_t rank, std::uint16_t port);

  [[nodiscard]] bool SendFrame(std::uint16_t type, std::uint64_t sequence,
                               const std::vector<std::uint8_t>& payload,
                               std::string* error);
  [[nodiscard]] bool ReceiveFrame(std::uint16_t type, std::uint64_t* sequence,
                                  std::vector<std::uint8_t>* payload,
                                  std::string* error);
  [[nodiscard]] bool SendAll(const void* data, std::size_t bytes,
                             std::string* error);
  [[nodiscard]] bool RecvAll(void* data, std::size_t bytes, std::string* error);

  int fd_{-1};
  std::uint16_t port_{0};
  std::uint32_t rank_{0};
  std::uint32_t world_size_{1};
  std::uint32_t max_context_{0};
  bool handshaken_{false};
  std::uint64_t snapshot_budget_bytes_{0};
  std::string auth_token_;
  std::mutex send_mutex_;
  std::mutex receive_mutex_;
  std::atomic<bool> interrupted_{false};
  std::vector<std::uint8_t> command_prompt_;
  std::vector<std::uint8_t> response_payload_;
};

/// Rank-zero response owner. The dedicated reader routes final responses by
/// control sequence to their requests, at most `max_pending_responses` at
/// once, and cache acknowledgements to the instruction awaiting one.
class TpResponseBroker final {
public:
  TpResponseBroker(std::shared_ptr<TpControlChannel> control,
                   std::size_t max_pending_responses);
  ~TpResponseBroker();

  TpResponseBroker(const TpResponseBroker&) = delete;
  TpResponseBroker& operator=(const TpResponseBroker&) = delete;
  TpResponseBroker(TpResponseBroker&&) = delete;
  TpResponseBroker& operator=(TpResponseBroker&&) = delete;

  [[nodiscard]] bool RegisterInstruction(std::uint64_t sequence,
                                         std::uint64_t index,
                                         std::string* error);
  [[nodiscard]] bool WaitForInstruction(std::string* error);

  [[nodiscard]] bool RegisterPendingResponse(std::uint64_t sequence,
                                             std::string* error);
  [[nodiscard]] bool CancelUnsentResponse(std::uint64_t sequence,
                                          std::string* error);
  [[nodiscard]] bool WaitForResponse(std::uint64_t sequence,
                                     TpControlResponse* response,
                                     std::string* error);
  void FailAll(std::string reason);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_TP_CONTROL_HPP_
