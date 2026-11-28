#ifndef GUFO_CACHE_DISK_CATALOG_HPP_
#define GUFO_CACHE_DISK_CATALOG_HPP_

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "src/cache/prefix_index.hpp"

namespace gufo::cache {
// The entry one EvictOne retired, and the eviction class that selected it.
struct DiskEviction {
  std::shared_ptr<const DiskDescription> description;
  // Superseded generated history, or reused within one disk turnover.
  bool superseded{}, reused{};
  explicit operator bool() const { return description != nullptr; }
};
using DiskEvictionVisitor = std::function<void(const DiskEviction&)>;
#ifdef GUFO_CACHE_TESTING
// One record's ranking inputs and results, for tests that check the ranking
// against an independent model.
struct DiskCatalogRankForTesting {
  std::uint64_t id{}, used{}, published{};
  std::optional<std::uint64_t> reused_at;
  bool invalid{};
  std::size_t branch{};
  std::array<std::uint64_t, 5> order{};
  std::uint64_t unique_bytes{};
};
#endif
// The allowance for the RAM that tracked disk checkpoints' metadata holds in
// a RAM cache budget of `ram_budget` bytes; 0 (RAM cache off, no aggregate
// limit) gives no allowance. The catalog holds tracked checkpoints' tokens
// once per compatibility digest, in a token trie, so they grow linearly with
// a conversation; what still grows with each checkpoint is its chunk
// references (manifest chunks and the dependency map), which without the
// allowance would fill the ledger until RAM captures are refused (#1329).
// Backing takes about 90% of the budget. The allowance takes 1/32 of it; the
// rest of the metadata room covers RAM-tier metadata, index nodes and the
// in-flight persistence manifests the allowance does not count: up to 16
// captures held for the persistence queue (twice its depth), each holding its
// tokens twice, in DurableCapture::manifest and the PersistenceJob's copy
// (32 MiB at 262144 tokens).
// A 16 MiB floor leaves small budgets to the ledger's own refusals. Typical
// automatic budgets give hundreds of MiB, which a full default disk of the
// measured agent workload (about 62 MB) does not reach.
[[nodiscard]] std::size_t DiskMetadataAllowance(std::size_t ram_budget);
// Tokens a kSeparate DiskStore recovers, placed straight into the catalog's
// tries as the store decodes each manifest, so startup never holds every
// checkpoint's token list. Pass Sink() to the store's constructor, then this
// object to the constructor of a DiskCatalog over that store, which takes its
// contents; tokens whose checkpoint, count or digest differ from the entry of
// the same epoch throw. Charged to the ledger like the catalog's own tries.
class DiskCatalogTokens {
public:
  explicit DiskCatalogTokens(ResourceLedger&);
  ~DiskCatalogTokens();
  DiskCatalogTokens(DiskCatalogTokens&&) noexcept;
  DiskCatalogTokens& operator=(DiskCatalogTokens&&) noexcept;
  DiskCatalogTokens(const DiskCatalogTokens&) = delete;
  DiskCatalogTokens& operator=(const DiskCatalogTokens&) = delete;
  // Valid while this object lives and has not been given to a catalog.
  [[nodiscard]] DiskRecoveredTokens Sink();
  // Initialized token storage the tries' edges hold, for startup accounting
  // like DiskStore::OwnedManifestBytes.
  [[nodiscard]] std::size_t TokenBytes() const;

private:
  friend class DiskCatalog;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
// One lineage-aware catalog per store, including identities not currently
// loaded. Construct at quiescence after startup discovery; Track every
// successful new publication before exposing it through lookup. Operations are
// caller serialized. Store, index and ledger outlive the catalog. Touch records
// a successful use; startup ties and untouched publications use checkpoint ID.
// The metadata allowance bounds MetadataBytes by retiring entries in
// EvictOne's order. Recovery tracks every entry; call FitMetadata after it.
//
// The catalog keeps its own token trie per compatibility digest, apart from
// the prefix index's: it tracks digests the index has not loaded, tracks
// publications before they are indexed, and keeps records the index drops.
// Tracking needs a checkpoint's tokens: construct over a store whose tokens
// are not yet released (or with the DiskCatalogTokens its recovery filled),
// and Track a publication before DiskStore::ReleaseTokens.
class DiskCatalog {
public:
  // Retirements one publication may spend on the metadata allowance. A pass
  // ranks every record pairwise under the serving lock, so it ranks once and
  // retires along that order down to an eighth of the allowance below it;
  // later publications then fit without ranking. More than one also keeps a
  // growing conversation converging: its new leaf outweighs the prefix
  // retired.
  static constexpr std::size_t kMetadataEvictions = 16;
  DiskCatalog(ResourceLedger&, DiskStore&, PrefixIndex&,
              std::uint64_t turnover_bytes = UINT64_MAX,
              std::size_t metadata_allowance_bytes = SIZE_MAX,
              DiskCatalogTokens* recovered = nullptr);
  ~DiskCatalog();
  DiskCatalog(const DiskCatalog&) = delete;
  DiskCatalog& operator=(const DiskCatalog&) = delete;
  // When tracking a new publication would exceed the metadata allowance,
  // first retire up to kMetadataEvictions other entries until it fits with
  // an eighth of the allowance to spare; `retired` sees each. The checkpoint
  // being tracked is never a candidate. A failed retirement ends the pass,
  // never the tracking; the next retries.
  void Track(CheckpointId, const DiskEvictionVisitor& retired = {});
  [[nodiscard]] std::shared_ptr<const DiskDescription> Find(CheckpointId) const;
  // True when the tracked publication `id` at `epoch` is a prefix of
  // `tokens`, equal included. False when untracked, re-published or longer.
  // Compares the trie path without allocating.
  [[nodiscard]] bool IsPrefixOf(CheckpointId, std::uint64_t epoch,
                                std::span<const Token> tokens) const;
  // Rebuilds a tracked checkpoint's tokens from its trie path into one
  // charged buffer, valid only during `visit`. False, without calling
  // `visit`, when the checkpoint is not tracked.
  bool VisitTokens(
      CheckpointId,
      const std::function<void(std::span<const Token>)>& visit) const;
  void Touch(CheckpointId);
  // Quiescent maintenance/shutdown only: persist dirty advisory access history.
  // May perform filesystem I/O. Track and Touch only update in-memory history.
  void FlushUses();
  // Evict an unpinned entry, superseded history before conversation frontiers.
  // Busy writers cause an immediate false result;
  // capacity is credited only after DiskStore's durable retirement succeeds.
  // Returns the retired entry, or an empty result when nothing was retired.
  DiskEviction EvictOne();
  // Reverse dependency quarantine is immediate, with no I/O-lock wait. Retire
  // quarantined manifests at a later idle boundary after reader pins settle.
  void Invalidate(bool private_file, DiskFileId);
  // Reconcile worker-discovered quarantine at a serialized caller boundary.
  // Lookup rejects quarantined descriptions immediately, even before this.
  void Reconcile();
  std::size_t ReclaimInvalid();
  // When MetadataBytes exceeds the allowance, retire entries in EvictOne's
  // order until it fits with an eighth of the allowance to spare or nothing
  // more retires. Returns the number retired.
  std::size_t FitMetadata(const DiskEvictionVisitor& retired = {});
  // RAM held for tracked checkpoints: the store's decoded manifests and pins,
  // catalog records, their descriptions and dependency maps, and the
  // catalog's token tries. A kManifest store's manifests also count their
  // tokens; tokens a kSeparate store holds until ReleaseTokens do not count.
  // It excludes the store's entry slots (a high-water vector), its
  // verified-file cache, index records and nodes, and transient persistence.
  // A session's description can hold a retired manifest's charge until the
  // session publishes again.
  [[nodiscard]] std::size_t MetadataBytes() const;
  [[nodiscard]] std::size_t MetadataAllowance() const;
  [[nodiscard]] std::size_t Size() const;
#ifdef GUFO_CACHE_TESTING
  // Ranks every record as EvictOne does and returns them in checkpoint order.
  // Like EvictOne it reconciles first: records the store quarantined become
  // invalid and leave the index. It also replaces the ranking a retirement
  // pass walks, which clears the marks of candidates that pass tried.
  [[nodiscard]] std::vector<DiskCatalogRankForTesting> RankForTesting();
  // The same records without ranking or reconciling: branch and order are
  // what the last ranking left, and invalid holds only what the catalog has
  // already seen.
  [[nodiscard]] std::vector<DiskCatalogRankForTesting> RecordsForTesting()
      const;
  [[nodiscard]] std::uint64_t AdmittedBytesForTesting() const;
#endif

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gufo::cache
#endif
