#ifndef GUFO_CACHE_PREFIX_INDEX_HPP_
#define GUFO_CACHE_PREFIX_INDEX_HPP_

#include <memory>
#include <span>
#include <vector>

#include "src/cache/checkpoint.hpp"
#include "src/cache/disk.hpp"

namespace gufo::cache {
struct IndexEntryId {
  std::uint64_t value{};
  bool operator==(const IndexEntryId&) const = default;
};
// Both may be true. Missing entries mean unavailable components.
struct ComponentAvailability {
  ComponentId id;
  bool resident{false}, durable{false};
};
struct CandidateAvailability {
  ResourceCharge metadata;
  std::vector<ComponentAvailability> components;
};
struct LiveFrontier {
  BorrowedLocation location;
  std::vector<Token> tokens;
  Identity compatibility, input;
  std::vector<ComponentPosition> positions;
  bool available{true};
};
enum class SelectionReason : std::uint8_t {
  kExactLiveContinuation,
  kDeepestCheckpoint,
  kDeepestDurableCheckpoint,
  kDeepestMixedCheckpoint,
  kDisabled,
  kUnknownCompatibility,
  kNoCompatibleBoundary,
  kInputIdentity,
  kMissingComponent,
  kStablePrefixBoundary,
};
struct PrefixCandidate {
  IndexEntryId entry;
  Rows boundary{};
  // Live is exclusive. Checkpoint and durable may coexist for mixed component
  // restoration. Availability selects each component's source; immutable
  // descriptions and a publication-specific disk pin survive index eviction.
  std::shared_ptr<const Checkpoint> checkpoint;
  std::optional<BorrowedLocation> live;
  std::size_t transfer_bytes{};
  std::shared_ptr<const DiskDescription> durable;
  DiskReadPin disk_pin;
  std::shared_ptr<const CandidateAvailability> availability;
  [[nodiscard]] bool UsesDisk(ComponentId id) const {
    if (!durable)
      return false;
    if (!checkpoint)
      return true;
    for (const auto& component : availability->components)
      if (component.id == id)
        return !component.resident;
    return false;
  }
};
struct PrefixQuery {
  Identity compatibility;
  std::span<const Token> tokens;
  InputIdentity input;
  Rows stable_prefix_tokens{};
  bool reuse{true};
};
struct PrefixLookup {
  ResourceCharge metadata;
  SelectionReason reason{SelectionReason::kNoCompatibleBoundary};
  std::optional<PrefixCandidate> selected;
  // Deepest first, then live, resident, mixed and durable; then checkpoint ID
  // or slot/generation ascending. Entry ID resolves otherwise identical ties.
  std::vector<PrefixCandidate> candidates;
  PrefixLookup() = default;
  PrefixLookup(PrefixLookup&&) noexcept = default;
  PrefixLookup& operator=(PrefixLookup&&) noexcept;
  PrefixLookup(const PrefixLookup&) = delete;
};

// Caller serializes all operations, including lookups. A compressed token tree
// per compatibility identity shares token edges; it never scans every retained
// token list during lookup. No device operations, model/serving dependencies or
// slot leases. Ledger facade must outlive the index. Results pin checkpoints.
class PrefixIndex {
public:
  explicit PrefixIndex(ResourceLedger&);
  ~PrefixIndex();
  PrefixIndex(PrefixIndex&&) = delete;
  PrefixIndex& operator=(PrefixIndex&&) = delete;
  PrefixIndex(const PrefixIndex&) = delete;
  PrefixIndex& operator=(const PrefixIndex&) = delete;
  // One immutable adapter component inventory per compatibility identity.
  void Register(Identity, std::span<const ComponentDescriptor>,
                std::optional<CompatibilityDigest> = {});
  // established_stable_prefix attests a fallback established before capture,
  // even if it is no longer retained. Live frontiers never attest a fallback.
  [[nodiscard]] IndexEntryId Insert(std::shared_ptr<const Checkpoint>,
                                    std::vector<ComponentAvailability>,
                                    Rows established_stable_prefix = 0);
  [[nodiscard]] IndexEntryId Insert(LiveFrontier);
  [[nodiscard]] IndexEntryId Insert(Identity,
                                    std::shared_ptr<const DiskDescription>,
                                    Rows established_stable_prefix = 0);
  void AttachDurable(IndexEntryId, std::shared_ptr<const DiskDescription>);
  // RAM retention drops only its payload ownership when a durable copy exists.
  // A checkpoint without a usable durable description is erased instead.
  void DropResident(IndexEntryId);
  // Applies to every index entry bound to this publication. Corruption erases
  // RAM candidates too; ordinary disk eviction retains coherent RAM copies.
  void RemoveDurable(const DiskDescription&, bool corrupt = false);
  // Availability replaces the complete list. Updating a live frontier requires
  // erasing/reinserting its description after the owner changes its generation.
  void SetAvailability(IndexEntryId, std::vector<ComponentAvailability>);
  void SetLiveAvailable(IndexEntryId, bool);
  void Erase(IndexEntryId);
  [[nodiscard]] PrefixLookup Lookup(const PrefixQuery&) const;
  // Validate an admission against the registered inventory/layout and exact
  // private-state boundary, without publishing or allocating index metadata.
  [[nodiscard]] bool IsResidentCoherent(const Checkpoint&) const;
  // Ignores stable-prefix rules and the query's reuse flag; leases nothing.
  [[nodiscard]] Rows CachedPrefixTokens(const PrefixQuery&) const;
  // Conservative input-identity check at the stored record's full boundary,
  // including records longer than the query and temporarily unavailable lives.
  [[nodiscard]] Rows CommonPrefixTokens(const PrefixQuery&) const;
  // Charged object sizes and retained capacities, excluding allocator/control
  // block/ledger bookkeeping and separately charged checkpoint/descriptions.
  [[nodiscard]] std::size_t MetadataBytes() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gufo::cache
#endif
