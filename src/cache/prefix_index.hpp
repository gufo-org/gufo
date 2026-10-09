#ifndef GUFO_CACHE_PREFIX_INDEX_HPP_
#define GUFO_CACHE_PREFIX_INDEX_HPP_

#include <memory>
#include <span>
#include <vector>

#include "src/cache/checkpoint.hpp"

namespace gufo::cache {
struct IndexEntryId {
  std::uint64_t value{};
  bool operator==(const IndexEntryId&) const = default;
};
// Both may be true. Durable-only components are described but never selected
// until tiered lookup (card 13). Missing entries mean unavailable components.
struct ComponentAvailability {
  ComponentId id;
  bool resident{false}, durable{false};
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
  // Exactly one is populated. Checkpoint handles keep payloads alive; live
  // descriptions require generation validation and leasing in card 06.
  std::shared_ptr<const Checkpoint> checkpoint;
  std::optional<BorrowedLocation> live;
  std::size_t transfer_bytes{};
};
struct PrefixQuery {
  Identity compatibility;
  std::span<const Token> tokens;
  InputIdentity input;
  Rows stable_prefix_tokens{};
  bool reuse{true};
};
struct PrefixLookup {
  SelectionReason reason{SelectionReason::kNoCompatibleBoundary};
  std::optional<PrefixCandidate> selected;
  // Deepest first, then live before checkpoint, then checkpoint ID or
  // slot/generation ascending. Entry ID resolves otherwise identical ties.
  std::vector<PrefixCandidate> candidates;
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
  void Register(Identity, std::span<const ComponentDescriptor>);
  // established_stable_prefix attests a fallback established before capture,
  // even if it is no longer retained. Live frontiers never attest a fallback.
  [[nodiscard]] IndexEntryId Insert(std::shared_ptr<const Checkpoint>,
                                    std::vector<ComponentAvailability>,
                                    Rows established_stable_prefix = 0);
  [[nodiscard]] IndexEntryId Insert(LiveFrontier);
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
  // block/ledger bookkeeping and separately charged checkpoint storage.
  [[nodiscard]] std::size_t MetadataBytes() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gufo::cache
#endif
