#include "src/cache/disk_catalog.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <stdexcept>

namespace gufo::cache {
namespace {
template<class Visit>
void Dependencies(const DiskManifest& manifest, Visit visit) {
  for (const auto& component : manifest.components) {
    for (const auto& payload : component.chunks)
      visit(false, payload.file);
    for (const auto& payload : {component.tail, component.private_state})
      if (payload)
        visit(true, payload->file);
  }
}
// The store's availability record is private; this bounds its charge.
constexpr std::size_t kAvailabilityBytes = 128;
// RAM the store holds per durable entry: the decoded manifest PlanMetadata
// charges, its pin and availability, and one description.
std::size_t StoreBytes(const DiskManifest& manifest) {
  std::size_t bytes = manifest.input.size() +
                      manifest.tokens.size() * sizeof(Token) +
                      manifest.components.size() * sizeof(DiskComponent) +
                      sizeof(ResourceCharge) + kAvailabilityBytes +
                      sizeof(DiskDescription) + 64;
  for (const auto& component : manifest.components)
    bytes += component.chunks.size() * sizeof(DiskPayload);
  return bytes;
}
}  // namespace
std::size_t DiskMetadataAllowance(std::size_t ram_budget) {
  if (!ram_budget)
    return SIZE_MAX;
  return std::max(ram_budget / 32, std::size_t{16} << 20);
}
struct DiskCatalog::Impl {
  using Key = std::pair<bool, DiskFileId>;
  struct Record {
    ResourceCharge metadata;
    std::shared_ptr<const DiskDescription> description;
    std::uint64_t used{};
    std::optional<std::uint64_t> reused_at;
    bool invalid{};
    bool use_dirty{};
    // Prepaid with the catalog record so eviction can reclaim under a full
    // ledger without admitting a temporary candidate table.
    std::array<std::uint64_t, 5> eviction_order{};
    // Valid peers on this record's branch: its prefixes, its descendants and
    // checkpoints of the same tokens. EvictOne counts them before ranking.
    std::size_t branch{};
    // Admission tick; startup discovery uses the persisted last use.
    std::uint64_t published{};
    // MetadataBytes share: store terms and this record's admission.
    std::size_t footprint{};
  };
  struct Bucket {
    ResourceCharge metadata;
    std::map<std::uint64_t, ResourceCharge> checkpoints;
  };
  ResourceLedger* ledger;
  DiskStore* store;
  PrefixIndex* index;
  ResourceCharge metadata;
  std::map<std::uint64_t, Record> records;
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t> lru;
  std::map<Key, Bucket> reverse;
  std::uint64_t clock{}, admitted_bytes{}, turnover_bytes;
  // Records count their fixed terms; buckets and references are counted
  // only where AddDependency and RemoveDependency charge and release them.
  std::size_t metadata_bytes{}, metadata_allowance;
  bool loading{true};
  static constexpr std::size_t kRecordBytes =
      sizeof(decltype(records)::value_type) +
      sizeof(decltype(lru)::value_type) + 128;
  static constexpr std::size_t kBucketBytes =
      sizeof(decltype(reverse)::value_type) + 64;
  static constexpr std::size_t kReferenceBytes =
      sizeof(decltype(Bucket::checkpoints)::value_type) + 64;
  Impl(ResourceLedger& resources, DiskStore& disk, PrefixIndex& prefixes,
       ResourceReservation reservation, std::uint64_t turnover,
       std::size_t allowance)
      : ledger(&resources),
        store(&disk),
        index(&prefixes),
        metadata(reservation.Convert()),
        turnover_bytes(turnover),
        metadata_allowance(allowance) {}
  ~Impl() {
    FlushUses();
    while (!records.empty())
      Erase(records.begin());
  }
  void FlushUses() noexcept {
    for (auto& [id, record] : records) {
      try {
        if (record.use_dirty &&
            store->TryRecordUse(
                {id},
                DiskUseHistory{record.used, admitted_bytes, record.reused_at}))
          record.use_dirty = false;
      } catch (...) {
        // Advisory use history must not fail serving or teardown when a
        // path allocation or host I/O setup fails before the store's write.
      }
    }
  }
  std::uint64_t Tick() {
    if (clock == UINT64_MAX)
      throw std::overflow_error("disk LRU clock exhausted");
    return ++clock;
  }
  void AddDependency(Key key, std::uint64_t id) {
    auto bucket = reverse.find(key);
    if (bucket == reverse.end()) {
      auto admission =
          ledger->Reserve(ResourceCategory::kMetadata, kBucketBytes);
      bucket = reverse.emplace(key, Bucket{}).first;
      try {
        bucket->second.metadata = admission.Convert();
      } catch (...) {
        reverse.erase(bucket);
        throw;
      }
      metadata_bytes += kBucketBytes;
    }
    auto& references = bucket->second.checkpoints;
    if (references.contains(id))
      return;
    // A bucket left empty by a failure here is released by the caller's
    // Erase, which RemoveDependency handles for a missing reference.
    auto admission =
        ledger->Reserve(ResourceCategory::kMetadata, kReferenceBytes);
    auto reference = references.emplace(id, ResourceCharge{}).first;
    try {
      reference->second = admission.Convert();
    } catch (...) {
      references.erase(reference);
      throw;
    }
    metadata_bytes += kReferenceBytes;
  }
  void RemoveDependency(Key key, std::uint64_t id) {
    auto bucket = reverse.find(key);
    if (bucket == reverse.end())
      return;
    auto& references = bucket->second.checkpoints;
    if (auto it = references.find(id); it != references.end()) {
      auto charge = std::move(it->second);
      auto node = references.extract(it);
      metadata_bytes -= kReferenceBytes;
    }
    if (references.empty()) {
      auto charge = std::move(bucket->second.metadata);
      auto node = reverse.extract(bucket);
      metadata_bytes -= kBucketBytes;
    }
  }
  static bool Below(const DiskManifest& a, const DiskManifest& b) {
    return a.compatibility == b.compatibility &&
           (a.input == b.input || a.lineage == b.lineage) &&
           a.tokens.size() < b.tokens.size() &&
           std::equal(a.tokens.begin(), a.tokens.end(), b.tokens.begin());
  }
  std::uint64_t UniqueBytes(const DiskManifest& manifest) const {
    std::uint64_t bytes = 0;
    const auto add = [&](bool priv, const DiskPayload& payload) {
      const auto found = reverse.find({priv, payload.file});
      if (found != reverse.end() && found->second.checkpoints.size() == 1)
        bytes += payload.bytes;
    };
    for (const auto& component : manifest.components) {
      for (const auto& chunk : component.chunks)
        add(false, chunk);
      for (const auto& payload : {component.tail, component.private_state})
        if (payload)
          add(true, *payload);
    }
    return bytes;
  }
  std::array<std::uint64_t, 5> Order(const Record& record) const {
    const auto& manifest = record.description->Manifest();
    auto recency = record.used;
    bool reused =
        record.reused_at && admitted_bytes - *record.reused_at < turnover_bytes;
    bool covered = false, rooted = false,
         fork = manifest.purpose == CheckpointPurpose::kLearned;
    std::optional<Token> next;
    std::size_t newer = 0;
    for (const auto& [id, peer] : records) {
      (void)id;
      if (peer.invalid)
        continue;
      const auto& peer_manifest = peer.description->Manifest();
      if (Below(peer_manifest, manifest)) {
        rooted = true;
        // Every checkpoint on this record's branch is also on its prefix's
        // branch. Equal counts mean no other branch shares the prefix, so it
        // belongs to this conversation, and the record takes its admission:
        // a prefix published after its leaf (spacing and write order decide
        // that) then cannot outlive the leaf, and the role order below
        // retires the covered prefix first. A restore lends nothing: a
        // client that resumed at the prefix did not match the leaf. Shared
        // prefixes and learned branch points lend nothing either, so one
        // conversation never keeps another's leaves.
        if (peer_manifest.purpose != CheckpointPurpose::kLearned &&
            peer.branch == record.branch)
          recency = std::max(recency, peer.published);
      }
      if (!Below(manifest, peer_manifest))
        continue;
      covered = true;
      recency = std::max(recency, peer.used);
      reused |=
          peer.reused_at && admitted_bytes - *peer.reused_at < turnover_bytes;
      const auto token = peer_manifest.tokens[manifest.tokens.size()];
      fork |= next && *next != token;
      next = token;
      newer += peer_manifest.lineage == manifest.lineage &&
               peer_manifest.purpose == CheckpointPurpose::kGenerated;
    }
    const bool superseded = manifest.purpose == CheckpointPurpose::kGenerated &&
                            !fork && newer >= 3;
    const bool shared = fork || (covered && !rooted);
    // Superseded intermediates leave before any conversation's last frontier.
    // Within each conversation keep its leaves, then learned/shared forks.
    // Prefer a retirement that actually releases more unique payload when
    // recency and role tie; shared chunks are never credited twice.
    return {superseded ? 0U
            : reused   ? 2U
                       : 1U,
            superseded ? 0 : recency,
            shared    ? 2U
            : covered ? 0U
                      : 1U,
            std::numeric_limits<std::uint64_t>::max() - UniqueBytes(manifest),
            manifest.checkpoint.value};
  }
  void Erase(decltype(records)::iterator it) {
    const auto id = it->first;
    Dependencies(it->second.description->Manifest(), [&](bool priv, auto file) {
      RemoveDependency({priv, file}, id);
    });
    lru.erase({it->second.used, id});
    metadata_bytes -= it->second.footprint;
    auto charge = std::move(it->second.metadata);
    auto node = records.extract(it);
  }
  // What tracking this manifest adds to metadata_bytes.
  std::size_t Incoming(const DiskManifest& manifest) const {
    auto bytes = StoreBytes(manifest) + kRecordBytes;
    Dependencies(manifest, [&](bool priv, auto file) {
      bytes += kReferenceBytes;
      if (!reverse.contains({priv, file}))
        bytes += kBucketBytes;
    });
    return bytes;
  }
  void Reconcile() {
    for (auto& [id, record] : records) {
      (void)id;
      if (!record.invalid && record.description->Quarantined()) {
        record.invalid = true;
        index->RemoveDurable(*record.description, true);
      }
    }
  }
  // Rank every record into its prepaid eviction_order.
  void Rank() {
    Reconcile();
    for (auto& [id, record] : records) {
      (void)id;
      record.branch = 0;
    }
    // Count each valid pair once: prefix and descendant, or the same tokens.
    for (auto it = records.begin(); it != records.end(); ++it) {
      if (it->second.invalid)
        continue;
      const auto& manifest = it->second.description->Manifest();
      for (auto peer = records.begin(); peer != it; ++peer) {
        if (peer->second.invalid)
          continue;
        const auto& peer_manifest = peer->second.description->Manifest();
        const auto size = manifest.tokens.size(),
                   peer_size = peer_manifest.tokens.size();
        if (size == peer_size
                ? peer_manifest.compatibility == manifest.compatibility &&
                      peer_manifest.tokens == manifest.tokens
            : size < peer_size ? Below(manifest, peer_manifest)
                               : Below(peer_manifest, manifest)) {
          ++it->second.branch;
          ++peer->second.branch;
        }
      }
    }
    for (auto& [id, record] : records) {
      (void)id;
      record.eviction_order = Order(record);
    }
  }
  // Retire the best ranked untried record. Later calls continue along the
  // same ranking, which retirements leave slightly stale.
  DiskEviction RetireRanked() {
    while (true) {
      auto record = records.end();
      for (auto it = records.begin(); it != records.end(); ++it) {
        if (it->second.eviction_order[0] == UINT64_MAX)
          continue;
        if (record == records.end() ||
            it->second.eviction_order < record->second.eviction_order)
          record = it;
      }
      if (record == records.end())
        return {};
      const auto tier = record->second.eviction_order[0];
      // Mark reader-pinned candidates as tried without allocating a side
      // table.
      record->second.eviction_order.fill(UINT64_MAX);
      auto description = record->second.description;
      auto result = store->TryRetire(description->Manifest().checkpoint,
                                     description->Epoch());
      if (result == DiskRetireResult::kBusy)
        return {};
      if (result == DiskRetireResult::kPinned)
        continue;
      index->RemoveDurable(*description);
      Erase(record);
      if (result == DiskRetireResult::kRetired)
        return {std::move(description), tier == 0, tier == 2};
    }
  }
  bool Fits(std::size_t incoming) const {
    return metadata_bytes <= metadata_allowance &&
           incoming <= metadata_allowance - metadata_bytes;
  }
  // A pass starts when `incoming` does not fit. It ranks once and retires
  // along that order until `incoming` fits an eighth of the allowance below
  // it, so one ranking serves several later publications.
  std::size_t Fit(std::size_t incoming, std::size_t limit,
                  const DiskEvictionVisitor& retired) {
    if (!limit || Fits(incoming))
      return 0;
    const auto slack = metadata_allowance / 8;
    const auto target =
        incoming > SIZE_MAX - slack ? SIZE_MAX : incoming + slack;
    Rank();
    std::size_t count{};
    while (count < limit && !Fits(target)) {
      auto eviction = RetireRanked();
      if (!eviction)
        break;
      ++count;
      if (retired)
        retired(eviction);
    }
    return count;
  }
};
DiskCatalog::DiskCatalog(ResourceLedger& ledger, DiskStore& store,
                         PrefixIndex& index, std::uint64_t turnover_bytes,
                         std::size_t metadata_allowance_bytes)
    : impl_(std::make_unique<Impl>(
          ledger, store, index,
          ledger.Reserve(ResourceCategory::kMetadata, sizeof(Impl)),
          turnover_bytes, metadata_allowance_bytes)) {
  for (const auto& entry : store.Entries())
    if (entry.Durable())
      Track(entry.Manifest().checkpoint);
  impl_->loading = false;
}
DiskCatalog::~DiskCatalog() = default;
void DiskCatalog::Track(CheckpointId id, const DiskEvictionVisitor& retired) {
  auto description = impl_->store->Describe(id);
  if (!description)
    throw std::invalid_argument("cannot track missing disk checkpoint");
  auto existing = impl_->records.find(id.value);
  if (existing != impl_->records.end()) {
    if (existing->second.description->Epoch() == description->Epoch())
      return;
    impl_->index->RemoveDurable(
        *existing->second.description,
        existing->second.invalid ||
            existing->second.description->Quarantined());
    impl_->Erase(existing);
  }
  // Make room before admitting: the new record is not yet a candidate, and
  // its own reservations below then meet a ledger with the room freed.
  if (!impl_->loading) {
    try {
      (void)impl_->Fit(impl_->Incoming(description->Manifest()),
                       kMetadataEvictions, retired);
    } catch (...) {
      // The allowance is advisory; tracking the publication is not.
    }
  }
  const auto history =
      impl_->loading ? impl_->store->UseHistory(id) : DiskUseHistory{};
  const auto used = impl_->loading ? history.last_used : impl_->Tick();
  impl_->admitted_bytes =
      std::max(impl_->admitted_bytes, history.admitted_bytes);
  impl_->clock = std::max(impl_->clock, used);
  auto admission =
      impl_->ledger->Reserve(ResourceCategory::kMetadata, Impl::kRecordBytes);
  auto record =
      impl_->records
          .emplace(
              id.value,
              Impl::Record{
                  {}, std::move(description), used, history.reused_at, false})
          .first;
  record->second.published = used;
  record->second.footprint =
      StoreBytes(record->second.description->Manifest()) + Impl::kRecordBytes;
  impl_->metadata_bytes += record->second.footprint;
  try {
    impl_->lru.emplace(std::make_pair(used, id.value), id.value);
    Dependencies(record->second.description->Manifest(),
                 [&](bool priv, auto file) {
                   impl_->AddDependency({priv, file}, id.value);
                 });
    record->second.metadata = admission.Convert();
    if (!impl_->loading) {
      const auto bytes =
          impl_->UniqueBytes(record->second.description->Manifest());
      if (bytes > UINT64_MAX - impl_->admitted_bytes)
        throw std::overflow_error("disk admission clock exhausted");
      impl_->admitted_bytes += bytes;
      // Every surviving record must carry the newest byte clock at flush;
      // otherwise retiring the newest admission could revive old reuse on
      // a clean restart.
      for (auto& [peer_id, peer] : impl_->records) {
        (void)peer_id;
        peer.use_dirty = true;
      }
    }
  } catch (...) {
    impl_->Erase(record);
    throw;
  }
}
std::shared_ptr<const DiskDescription> DiskCatalog::Find(
    CheckpointId id) const {
  auto it = impl_->records.find(id.value);
  if (it == impl_->records.end() || it->second.invalid ||
      !it->second.description->Pin())
    return {};
  return it->second.description;
}
void DiskCatalog::Touch(CheckpointId id) {
  auto it = impl_->records.find(id.value);
  if (it == impl_->records.end() || it->second.invalid)
    return;
  const auto used = impl_->Tick();
  auto node = impl_->lru.extract({it->second.used, id.value});
  node.key() = {used, id.value};
  it->second.used = used;
  it->second.reused_at = impl_->admitted_bytes;
  impl_->lru.insert(std::move(node));
  it->second.use_dirty = true;
}
void DiskCatalog::FlushUses() {
  impl_->FlushUses();
}
DiskEviction DiskCatalog::EvictOne() {
  impl_->Rank();
  return impl_->RetireRanked();
}
void DiskCatalog::Invalidate(bool private_file, DiskFileId file) {
  impl_->store->InvalidateDependency(private_file, file);
  auto it = impl_->reverse.find({private_file, file});
  if (it == impl_->reverse.end())
    return;
  for (const auto& [id, charge] : it->second.checkpoints) {
    (void)charge;
    auto& record = impl_->records.at(id);
    record.invalid = true;
    const auto& description = *record.description;
    impl_->store->Invalidate(description.Manifest().checkpoint,
                             description.Epoch());
    impl_->index->RemoveDurable(description, true);
  }
}
void DiskCatalog::Reconcile() {
  impl_->Reconcile();
}
std::size_t DiskCatalog::ReclaimInvalid() {
  Reconcile();
  std::size_t retired{};
  for (auto it = impl_->records.begin(); it != impl_->records.end();) {
    auto current = it++;
    if (!current->second.invalid)
      continue;
    const auto& description = *current->second.description;
    auto result = impl_->store->TryRetire(description.Manifest().checkpoint,
                                          description.Epoch());
    if (result == DiskRetireResult::kBusy)
      break;
    if (result == DiskRetireResult::kPinned)
      continue;
    retired += result == DiskRetireResult::kRetired;
    impl_->Erase(current);
  }
  return retired;
}
std::size_t DiskCatalog::FitMetadata(const DiskEvictionVisitor& retired) {
  std::size_t count{};
  // Each pass ranks once; a later pass reranks what pins and stale ranks
  // left until nothing more retires.
  while (const auto pass = impl_->Fit(0, SIZE_MAX, retired))
    count += pass;
  return count;
}
std::size_t DiskCatalog::MetadataBytes() const {
  return impl_->metadata_bytes;
}
std::size_t DiskCatalog::MetadataAllowance() const {
  return impl_->metadata_allowance;
}
std::size_t DiskCatalog::Size() const {
  return impl_->records.size();
}
}  // namespace gufo::cache
