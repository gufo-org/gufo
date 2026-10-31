#ifndef GUFO_CACHE_CHECKPOINT_HPP_
#define GUFO_CACHE_CHECKPOINT_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "src/cache/identity.hpp"
#include "src/cache/ledger.hpp"
#include "src/cache/slot.hpp"

namespace gufo::cache {
struct LineageId {
  std::uint64_t value{};
  bool operator==(const LineageId&) const = default;
};
struct ChunkId {
  std::uint64_t value{};
  bool operator==(const ChunkId&) const = default;
};
struct CheckpointId {
  std::uint64_t value{};
  bool operator==(const CheckpointId&) const = default;
};
struct PayloadRequest {
  ComponentId component;
  ResourceCategory category;
  Rows first{}, end{};
  std::size_t bytes{};
};

// Move-only ownership of one retained payload. The opaque owner retains actual
// committed storage. Managed borrowed rows retain assigned backing and their
// source guard. Physical storage is destroyed before its accounting token.
class Payload {
public:
  ~Payload() = default;
  Payload(Payload&&) noexcept = default;
  Payload& operator=(Payload&&) noexcept;
  Payload(const Payload&) = delete;
  Payload& operator=(const Payload&) = delete;
  static Payload Committed(ResourceCharge, std::shared_ptr<const void> owner);
  static Payload Borrowed(std::shared_ptr<BorrowedRows>);
  // Managed borrowed rows require this pin for every source transfer. Keep it
  // until Completion settles. Owner() is available only after preservation.
  [[nodiscard]] RowPin PinRows(std::stop_token = {}) const;
  [[nodiscard]] std::shared_ptr<BorrowedRows> BorrowedStorage() const {
    return rows_;
  }
  [[nodiscard]] bool IsValid() const;
  // Allocation capacity, not logical range size. Retain the owning checkpoint
  // or chunk handle throughout each access to Owner(); a bare buffer is no pin.
  [[nodiscard]] std::size_t Bytes() const noexcept { return bytes_; }
  [[nodiscard]] ResourceCategory Category() const noexcept {
    return rows_ ? rows_->Category() : category_;
  }
  [[nodiscard]] std::optional<BorrowedLocation> BorrowedFrom() const {
    return rows_ ? rows_->Location() : std::nullopt;
  }
  [[nodiscard]] std::shared_ptr<const void> Owner() const {
    return rows_ ? rows_->Owner() : owner_;
  }
  // Removing a checkpoint frees none of this storage while a disk job holds
  // it.
  [[nodiscard]] bool PersistencePinned() const;
  // Assigned immutable storage token, never the original pool handle. Keep
  // the checkpoint and physical owner alive throughout a persistence job.
  [[nodiscard]] ResourceCharge CommittedCharge() const {
    if (rows_)
      throw std::logic_error("borrowed payload requires preservation");
    return charge_;
  }

private:
  friend class ChunkReference;
  friend class CheckpointChunk;
  friend class ExecutionHistory;
  [[nodiscard]] PersistencePin PinPersistence() const;
  Payload() = default;
  ResourceCharge charge_;
  std::size_t bytes_{};
  ResourceCategory category_{};
  std::shared_ptr<const void> owner_;
  std::shared_ptr<BorrowedRows> rows_;
};
struct ChunkReferences {
  std::size_t checkpoints{}, readers{}, persistence{};
  bool operator==(const ChunkReferences&) const = default;
};
namespace detail {
struct Chunk;
struct Lineage;
}  // namespace detail

// Copies preserve the handle's reference kind. Reader and persistence handles
// retain both the chunk and its actual storage even after checkpoint eviction.
// Distinct handles are thread safe; history mutation remains caller serialized.
class CheckpointChunk;
class ChunkReference {
public:
  ChunkReference(const CheckpointChunk&);
  ChunkReference(const ChunkReference&);
  ChunkReference& operator=(const ChunkReference&);
  ChunkReference(ChunkReference&&) noexcept;
  ChunkReference& operator=(ChunkReference&&) noexcept;
  ~ChunkReference();
  [[nodiscard]] ChunkId Id() const;
  // Live references identify the same immutable chunk by its owner address,
  // regardless of reader/checkpoint/persistence kind. No ownership changes.
  [[nodiscard]] bool SameChunk(const ChunkReference& other) const noexcept {
    return chunk_ == other.chunk_;
  }
  [[nodiscard]] bool SameChunk(const CheckpointChunk&) const noexcept;
  [[nodiscard]] LineageId Lineage() const;
  [[nodiscard]] ComponentId Component() const;
  [[nodiscard]] Rows First() const;
  [[nodiscard]] Rows End() const;
  [[nodiscard]] const Payload& Storage() const;
  [[nodiscard]] ChunkReferences References() const;
  [[nodiscard]] ChunkReference PinReader() const;
  [[nodiscard]] ChunkReference PinPersistence() const;

private:
  friend class ExecutionHistory;
  friend class CheckpointChunk;
  enum class Kind : std::uint8_t { kCheckpoint, kReader, kPersistence };
  ChunkReference(std::shared_ptr<detail::Chunk>, Kind, PersistencePin = {},
                 RowPin = {});
  void Release() noexcept;
  std::shared_ptr<detail::Chunk> chunk_;
  Kind kind_;
  PersistencePin pin_;
  RowPin rows_pin_;
};
// Checkpoints retain immutable chunk ownership, not active source readers.
// Keep their large vectors compact; acquire the full storage pin only for an
// actual reader or persistence job, without allocating another handle object.
class CheckpointChunk {
public:
  CheckpointChunk(const CheckpointChunk&);
  CheckpointChunk& operator=(const CheckpointChunk&);
  CheckpointChunk(CheckpointChunk&&) noexcept;
  CheckpointChunk& operator=(CheckpointChunk&&) noexcept;
  ~CheckpointChunk();
  [[nodiscard]] ChunkId Id() const;
  [[nodiscard]] bool SameChunk(const CheckpointChunk& other) const noexcept {
    return chunk_ == other.chunk_;
  }
  [[nodiscard]] bool SameChunk(const ChunkReference& other) const noexcept {
    return chunk_ == other.chunk_;
  }
  [[nodiscard]] LineageId Lineage() const;
  [[nodiscard]] ComponentId Component() const;
  [[nodiscard]] Rows First() const;
  [[nodiscard]] Rows End() const;
  [[nodiscard]] const Payload& Storage() const;
  [[nodiscard]] ChunkReferences References() const;
  [[nodiscard]] ChunkReference PinReader() const;
  [[nodiscard]] ChunkReference PinPersistence() const;

private:
  friend class ExecutionHistory;
  friend class ChunkReference;
  explicit CheckpointChunk(std::shared_ptr<detail::Chunk>);
  void Release() noexcept;
  std::shared_ptr<detail::Chunk> chunk_;
};
struct CheckpointComponent {
  [[nodiscard]] bool IsValid() const;
  ComponentDescriptor descriptor;
  ComponentPosition position;
  std::vector<CheckpointChunk> chunks;
  std::optional<Payload> tail;
  std::optional<Payload> private_state;
};
enum class CheckpointPurpose : std::uint8_t {
  kPrompt,
  kGenerated,
  kGrid,
  kLearned,
};
struct CheckpointRequest {
  std::span<const Token> tokens;
  InputIdentity input;
  std::vector<ComponentPosition> positions;
  CheckpointPurpose purpose;
  std::uint32_t rank;
};
class Checkpoint {
public:
  ~Checkpoint() = default;
  Checkpoint(const Checkpoint&) = delete;
  Checkpoint& operator=(const Checkpoint&) = delete;
  Checkpoint(Checkpoint&&) = delete;
  Checkpoint& operator=(Checkpoint&&) = delete;
  [[nodiscard]] bool IsValid() const;
  [[nodiscard]] CheckpointId Id() const { return id_; }
  [[nodiscard]] LineageId Lineage() const;
  [[nodiscard]] Rows Boundary() const { return tokens_.size(); }
  [[nodiscard]] const Identity& Compatibility() const { return compatibility_; }
  [[nodiscard]] const Identity& Input() const { return input_; }
  [[nodiscard]] std::span<const Token> Tokens() const { return tokens_; }
  [[nodiscard]] std::span<const CheckpointComponent> Components() const {
    return components_;
  }
  [[nodiscard]] CheckpointPurpose Purpose() const { return purpose_; }
  [[nodiscard]] std::uint32_t Rank() const { return rank_; }
  [[nodiscard]] std::size_t MetadataBytes() const { return metadata_bytes_; }

private:
  friend class ExecutionHistory;
  Checkpoint() = default;
  ResourceCharge metadata_;
  std::shared_ptr<const detail::Lineage> lineage_;
  CheckpointId id_;
  std::vector<Token> tokens_;
  Identity compatibility_, input_;
  std::vector<CheckpointComponent> components_;
  CheckpointPurpose purpose_{CheckpointPurpose::kPrompt};
  std::uint32_t rank_{};
  std::size_t metadata_bytes_{};
};

// One history per uninterrupted execution branch, never per token hash or just
// per lineage ID. Restored inherits only its checkpoint's complete chunks;
// sibling suffixes and partial tails can never become candidates for sharing.
// The ledger facade outlives each history. Capture callbacks cannot reenter it.
// Cold/Restored attest provenance, not successful execution/restore: callers
// create them only after cold initialization or a successful adapter Validate.
class ExecutionHistory {
public:
  ~ExecutionHistory() = default;
  using CapturePayload = std::function<Payload(const PayloadRequest&)>;
  static ExecutionHistory Cold(ResourceLedger&,
                               std::span<const ComponentDescriptor>, Identity);
  static ExecutionHistory Restored(ResourceLedger&, const Checkpoint&);
  // A validated durable restore with no resident chunk owners starts a fresh
  // branch. Future captures copy its rows rather than infer hash sharing.
  static ExecutionHistory RestoredUnshared(ResourceLedger&,
                                           std::span<const ComponentDescriptor>,
                                           Identity);
  // Startup discovery calls this before creating histories. IDs of retained
  // durable checkpoints cannot be reused after a process restart.
  static void ObserveDurableId(CheckpointId);
  // Disk-only capture uses the same process/restart-safe identifier space.
  [[nodiscard]] static CheckpointId AllocateCheckpointId();
  ExecutionHistory(ExecutionHistory&&) noexcept = default;
  ExecutionHistory& operator=(ExecutionHistory&&) noexcept = delete;
  ExecutionHistory(const ExecutionHistory&) = delete;
  ExecutionHistory& operator=(const ExecutionHistory&) = delete;
  // The callback returns only complete private copies or fully reserved
  // borrowed descriptions, and must settle every transfer before returning. No
  // checkpoint escapes before all callbacks/validation succeed. Failure changes
  // no history; partially built owners and references unwind. No callback runs
  // under a lock. Every private state/tail callback must return a fresh copy
  // or a distinct committed assignment; a callback cannot alias private bytes.
  // Optional prepare sees the exact new payload plan with shared chunks pinned.
  // It can reserve every backing assignment before the first capture callback.
  // A refused preparation performs no payload copies and changes no history.
  [[nodiscard]] std::shared_ptr<const Checkpoint> Capture(
      const CheckpointRequest&, const CapturePayload&,
      const std::function<void(std::span<const PayloadRequest>)>& prepare = {});
  [[nodiscard]] LineageId Lineage() const;
  // Remove expired chunk bookkeeping. Does not evict a checkpoint or payload.
  void Prune();
  // New unique payload capacity needed at these component positions. Excludes
  // separately charged metadata; inherited full chunks contribute zero.
  [[nodiscard]] std::size_t NewPayloadBytes(
      std::span<const ComponentPosition>) const;
  [[nodiscard]] static std::size_t ChunkMetadataBytes();
  // This branch captured the checkpoint or was restored from it (among its
  // latest kRecordedCheckpoints), so its slot computed or loaded exactly those
  // rows; whether they remain is the adapter's ResidentRows.
  [[nodiscard]] bool Contains(CheckpointId) const noexcept;
  static constexpr std::size_t kRecordedCheckpoints = 32;

private:
  void Record(CheckpointId) noexcept;
  struct Entry {
    // Weak handles retain a control block, so keep its metadata charge too.
    ResourceCharge metadata;
    ComponentId component;
    Rows first;
    std::weak_ptr<detail::Chunk> chunk;
    [[nodiscard]] auto Key() const { return std::pair{component.value, first}; }
  };
  ExecutionHistory(ResourceLedger&, std::span<const ComponentDescriptor>,
                   Identity, std::shared_ptr<const detail::Lineage>);
  struct ChunkCursor {
    std::vector<Entry>::const_iterator next, end;
    ComponentId component;
    // Capture and estimation visit rows in increasing order.
    [[nodiscard]] std::shared_ptr<detail::Chunk> Find(Rows first);
  };
  [[nodiscard]] ChunkCursor Chunks(ComponentId) const;
  bool capturing_{false};
  ResourceLedger* ledger_;
  ResourceCharge metadata_, entries_metadata_, tokens_metadata_;
  std::shared_ptr<const detail::Lineage> lineage_;
  std::vector<ComponentDescriptor> descriptors_;
  Identity compatibility_, input_;
  std::vector<ComponentPosition> positions_;
  std::vector<Token> tokens_;
  // Ordered by component and first row. Restored and newly captured entries
  // are sorted before publication; pruning preserves the order.
  std::vector<Entry> entries_;
  std::array<CheckpointId, kRecordedCheckpoints> recorded_{};
  std::size_t recorded_count_{};
};
}  // namespace gufo::cache
#endif
