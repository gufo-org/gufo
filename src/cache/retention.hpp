#ifndef GUFO_CACHE_RETENTION_HPP_
#define GUFO_CACHE_RETENTION_HPP_

#include <array>
#include <functional>
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
};
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
};
// Caller serializes policy, index and history operations. Ledger and index
// outlive this object; sink, if present, does too. Callbacks/sinks cannot
// reenter. Policy owns only its index entries. Returned lookup/reader pins
// remain valid after removal and their ledger charges cannot be spent again
// prematurely.
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
  };
  std::size_t Find(CheckpointId) const;
  bool Branch(std::size_t) const;
  std::uint32_t Priority(std::size_t, const RetentionRequest* = nullptr) const;
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
};
}  // namespace gufo::cache
#endif
