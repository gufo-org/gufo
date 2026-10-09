#ifndef GUFO_CACHE_CHECKPOINT_HPP_
#define GUFO_CACHE_CHECKPOINT_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "src/cache/identity.hpp"
#include "src/cache/ledger.hpp"

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
struct BorrowedLocation {
  SlotId slot;
  std::uint64_t generation{};
  bool operator==(const BorrowedLocation&) const = default;
};
struct PayloadRequest {
  ComponentId component;
  ResourceCategory category;
  Rows first{}, end{};
  std::size_t bytes{};
};

// Move-only ownership of one retained payload. The opaque owner retains actual
// committed storage (including spill backing for borrowed rows). It is
// destroyed before the accounting token. Borrowed source protection is supplied
// by card 06; this record deliberately contains no slot pointer or mutation
// implementation.
class Payload {
public:
  ~Payload() = default;
  Payload(Payload&&) noexcept = default;
  Payload& operator=(Payload&&) noexcept;
  Payload(const Payload&) = delete;
  Payload& operator=(const Payload&) = delete;
  static Payload Committed(ResourceCharge, std::shared_ptr<const void> owner);
  static Payload Borrowed(ResourceReservation,
                          std::shared_ptr<const void> backing_owner,
                          BorrowedLocation);
  // Allocation capacity, not logical range size. Retain the owning checkpoint
  // or chunk handle throughout each access to Owner(); a bare buffer is no pin.
  [[nodiscard]] std::size_t Bytes() const noexcept { return bytes_; }
  [[nodiscard]] ResourceCategory Category() const noexcept { return category_; }
  [[nodiscard]] const std::optional<BorrowedLocation>& BorrowedFrom() const {
    return borrowed_;
  }
  [[nodiscard]] const std::shared_ptr<const void>& Owner() const {
    return owner_;
  }

private:
  friend class ChunkReference;
  [[nodiscard]] PersistencePin PinPersistence() const;
  Payload() = default;
  ResourceCharge charge_;
  ResourceReservation reservation_;
  std::size_t bytes_{};
  ResourceCategory category_{};
  std::optional<BorrowedLocation> borrowed_;
  std::shared_ptr<const void> owner_;
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
class ChunkReference {
public:
  ChunkReference(const ChunkReference&);
  ChunkReference& operator=(const ChunkReference&);
  ChunkReference(ChunkReference&&) noexcept;
  ChunkReference& operator=(ChunkReference&&) noexcept;
  ~ChunkReference();
  [[nodiscard]] ChunkId Id() const;
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
  enum class Kind : std::uint8_t { kCheckpoint, kReader, kPersistence };
  ChunkReference(std::shared_ptr<detail::Chunk>, Kind, PersistencePin = {});
  void Release() noexcept;
  std::shared_ptr<detail::Chunk> chunk_;
  Kind kind_;
  PersistencePin pin_;
};
struct CheckpointComponent {
  ComponentDescriptor descriptor;
  ComponentPosition position;
  std::vector<ChunkReference> chunks;
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
  [[nodiscard]] std::shared_ptr<const Checkpoint> Capture(
      const CheckpointRequest&, const CapturePayload&);
  [[nodiscard]] LineageId Lineage() const;
  // Remove expired chunk bookkeeping. Does not evict a checkpoint or payload.
  void Prune();
  [[nodiscard]] static std::size_t ChunkMetadataBytes();

private:
  struct Entry {
    // Weak handles retain a control block, so keep its metadata charge too.
    ResourceCharge metadata;
    ComponentId component;
    Rows first;
    std::weak_ptr<detail::Chunk> chunk;
  };
  ExecutionHistory(ResourceLedger&, std::span<const ComponentDescriptor>,
                   Identity, std::shared_ptr<const detail::Lineage>);
  bool capturing_{false};
  ResourceLedger* ledger_;
  ResourceCharge metadata_, entries_metadata_, tokens_metadata_;
  std::shared_ptr<const detail::Lineage> lineage_;
  std::vector<ComponentDescriptor> descriptors_;
  Identity compatibility_, input_;
  std::vector<ComponentPosition> positions_;
  std::vector<Token> tokens_;
  std::vector<Entry> entries_;
};
}  // namespace gufo::cache
#endif
