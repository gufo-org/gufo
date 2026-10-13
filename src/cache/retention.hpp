#ifndef GUFO_CACHE_RETENTION_HPP_
#define GUFO_CACHE_RETENTION_HPP_

#include <array>
#include <bitset>
#include <functional>
#include <optional>
#include <stop_token>

#include "src/cache/adapter.hpp"
#include "src/cache/prefix_index.hpp"

namespace gufo::cache {
// Retention roles differ from execution provenance (prompt/grid/generated).
enum class RetentionPurpose : std::uint8_t {
  kRetry,
  kHistory,
  kContinuation,
  kBranchPoint
};
struct CaptureCandidate {
  Rows boundary{};
  CheckpointPurpose purpose{CheckpointPurpose::kPrompt};
  bool required{false};
  bool splits_pass{false};
};
struct CapturePlanRequest {
  Rows prompt{}, reused{}, stable{}, common{};
  bool grid{true}, learn{true};
  std::span<const Rows> already_retained;
  // Keeps the shallowest eligible grid points, including retained ones.
  std::size_t grid_limit{4};
};
// One optional grid point per twelve private states the RAM budget holds,
// up to four. Speculative edit boundaries then cannot crowd out the
// conversations a small budget is meant to keep.
[[nodiscard]] std::size_t GridLimit(std::size_t ram_bytes,
                                    std::size_t private_bytes);
// Queries the adapter once. Required stable/prompt/reused boundaries remain
// exact. Optional grids reuse a preceding pass end within 128 tokens; learned
// points coalesce with a planned capture within the existing 64-token slack.
[[nodiscard]] std::vector<CaptureCandidate> PlanCaptures(
    const Adapter&, const Slot&, const CapturePlanRequest&);

enum class RetentionAction : std::uint8_t {
  kAdmitted,
  kRefused,
  kRemoved,
  kTouched
};
enum class RetentionReason : std::uint8_t {
  kCaptured,
  kByteCapacity,
  kRecordCapacity,
  kExactReplacement,
  kEditedTail,
  kCaptureFailure,
  kCancelled,
  kInvalid,
  kExplicit,
  kShutdown
};
struct RetentionEvent {
  std::uint64_t sequence{};
  RetentionAction action{};
  RetentionReason reason{};
  CheckpointId checkpoint;
  Rows boundary{};
  RetentionPurpose purpose{};
  std::uint32_t rank{};
  std::uint64_t last_used{};
  std::size_t unique_bytes_freed{};
  std::size_t retained_records{};
  // Actual ledger totals, including shared storage, reservations, free backing,
  // other owners and metadata. A removal with external pins can free zero.
  std::size_t ledger_bytes{};
  std::size_t retained_payload_bytes{};
};
class RetentionEventSink {
public:
  RetentionEventSink() = default;
  virtual ~RetentionEventSink() = default;
  RetentionEventSink(const RetentionEventSink&) = delete;
  RetentionEventSink& operator=(const RetentionEventSink&) = delete;
  RetentionEventSink(RetentionEventSink&&) = delete;
  RetentionEventSink& operator=(RetentionEventSink&&) = delete;
  virtual void Emit(const RetentionEvent&) noexcept = 0;
};
struct RetentionRequest {
  std::span<const Token> tokens;
  Identity compatibility, input;
  RetentionPurpose purpose{RetentionPurpose::kContinuation};
  Rows stable{};
  CheckpointId source{};
  // Protects incomplete byte admission. Full-record publication can advance
  // a redundant source (or replace the sole record), matching today's cache.
  bool preserve_source{true};
  std::stop_token stop{};
  // Preflight with ExecutionHistory::NewPayloadBytes, or a conservative lower
  // bound on newly owned payload capacity. An impossible payload refuses
  // without evicting. Shared complete chunks contribute zero.
  std::size_t new_payload_bytes{};
  // Keep edit boundaries captured by this turn available until it commits.
  // This protects them from the turn's later captures, without changing ranks.
  std::span<const CheckpointId> protected_checkpoints{};
  // The turn restored or continued a cached prefix. Its admitted checkpoint
  // marks the conversation as reused (see RetentionPolicy byte pressure).
  bool reused{false};
};
// Caller serializes policy, index and history operations. Ledger and index
// outlive this object; sink, if present, does too. Callbacks/sinks cannot
// reenter. Policy owns resident retention; durable entries stay index-owned
// after RAM removal. Returned lookup/reader pins
// remain valid after removal and their ledger charges cannot be spent again
// prematurely.
//
// Record-capacity removal keeps the ported rank-then-LRU order. Byte
// pressure removes, in order: retry copies; cheap checkpoints, whose private
// state exceeds eight times the rows they restore; conversations not reused
// within one RAM-budget turnover, least recently used first; then reused
// ones. A conversation's recency includes its retained descendants. An
// unreused conversation gives up covered near-duplicates, then its leaf,
// then its shallowest history boundary, which other prompts may share. A
// reused conversation gives up covered records, then leaves. In both, a
// fork that two continuations extended directly outlives those leaves.
class RetentionPolicy {
public:
  static constexpr std::size_t kRecordLimit = 128;
  using Capture = std::function<std::shared_ptr<const Checkpoint>()>;
  RetentionPolicy(ResourceLedger&, PrefixIndex&, RetentionEventSink* = nullptr,
                  std::size_t record_limit = kRecordLimit);
  ~RetentionPolicy();
  RetentionPolicy(const RetentionPolicy&) = delete;
  RetentionPolicy& operator=(const RetentionPolicy&) = delete;
  RetentionPolicy(RetentionPolicy&&) = delete;
  RetentionPolicy& operator=(RetentionPolicy&&) = delete;
  // Capture uses ledger reservations BEFORE allocating/copying and completes
  // every transfer before return. ResourceExhausted permits rank-eligible
  // eviction and a retry; factory must be safe to retry at this SAME execution
  // boundary. Other exceptions (including ordinary bad_alloc) refuse retention.
  // No overwrite/next prefill pass occurs during this synchronous call. There
  // is no per-checkpoint payload charge.
  [[nodiscard]] bool Admit(const RetentionRequest&, const Capture&);
  // Reclaim one eligible record for a caller that captures outside its index
  // lock. The caller retries the same frozen frontier before publication.
  [[nodiscard]] bool Reclaim(const RetentionRequest&);
  void Touch(CheckpointId);
  void Remove(CheckpointId);
  [[nodiscard]] std::uint32_t Rank(CheckpointId) const;
  [[nodiscard]] std::size_t Size() const { return size_; }
  [[nodiscard]] std::size_t RetainedPayloadBytes() const noexcept;

private:
  struct Record {
    std::shared_ptr<const Checkpoint> checkpoint;
    IndexEntryId entry;
    RetentionPurpose purpose{};
    Rows stable{};
    std::uint64_t last_used{};
    // admitted_bytes_ at the last reuse of this record's conversation.
    std::optional<std::uint64_t> reused_at;
    // Two continuations once extended this record directly.
    bool forked{false};
    // Retained records extending this one, by slot.
    std::bitset<kRecordLimit> below;
  };
  std::size_t Find(CheckpointId) const;
  bool Branch(std::size_t) const;
  std::uint32_t Priority(std::size_t, const RetentionRequest* = nullptr) const;
  bool Related(std::size_t, std::size_t) const;
  void Relate(std::size_t);
  std::array<std::uint64_t, 4> PressureOrder(std::size_t,
                                             const RetentionRequest&) const;
  std::size_t UniqueBytes(std::size_t) const;
  bool Persisting(std::size_t) const;
  std::size_t Victim(const RetentionRequest&, bool records) const;
  void Erase(std::size_t, RetentionReason, std::uint32_t rank);
  void Emit(RetentionAction, RetentionReason, CheckpointId, Rows,
            RetentionPurpose, std::uint32_t, std::uint64_t,
            std::size_t freed = 0) noexcept;
  ResourceLedger& ledger_;
  PrefixIndex& index_;
  RetentionEventSink* sink_;
  ResourceCharge metadata_;
  std::array<Record, kRecordLimit> records_;
  std::size_t limit_, size_{};
  std::uint64_t clock_{}, sequence_{};
  // Unique payload bytes admitted so far; measures budget turnover.
  std::uint64_t admitted_bytes_{};
  // Payload capacities are immutable. Only record membership changes this
  // total; access recency and borrowed-row preservation do not.
  mutable std::optional<std::size_t> retained_payload_bytes_;
};
}  // namespace gufo::cache
#endif
