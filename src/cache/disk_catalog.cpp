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
}  // namespace
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
  bool loading{true};
  Impl(ResourceLedger& resources, DiskStore& disk, PrefixIndex& prefixes,
       ResourceReservation reservation, std::uint64_t turnover)
      : ledger(&resources),
        store(&disk),
        index(&prefixes),
        metadata(reservation.Convert()),
        turnover_bytes(turnover) {}
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
          ledger->Reserve(ResourceCategory::kMetadata,
                          sizeof(decltype(reverse)::value_type) + 64);
      bucket = reverse.emplace(key, Bucket{}).first;
      try {
        bucket->second.metadata = admission.Convert();
      } catch (...) {
        reverse.erase(bucket);
        throw;
      }
    }
    auto& references = bucket->second.checkpoints;
    if (references.contains(id))
      return;
    auto admission =
        ledger->Reserve(ResourceCategory::kMetadata,
                        sizeof(decltype(Bucket::checkpoints)::value_type) + 64);
    auto reference = references.emplace(id, ResourceCharge{}).first;
    try {
      reference->second = admission.Convert();
    } catch (...) {
      references.erase(reference);
      throw;
    }
  }
  void RemoveDependency(Key key, std::uint64_t id) {
    auto bucket = reverse.find(key);
    if (bucket == reverse.end())
      return;
    auto& references = bucket->second.checkpoints;
    if (auto it = references.find(id); it != references.end()) {
      auto charge = std::move(it->second);
      auto node = references.extract(it);
    }
    if (references.empty()) {
      auto charge = std::move(bucket->second.metadata);
      auto node = reverse.extract(bucket);
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
      if (Below(peer_manifest, manifest))
        rooted = true;
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
    auto charge = std::move(it->second.metadata);
    auto node = records.extract(it);
  }
};
DiskCatalog::DiskCatalog(ResourceLedger& ledger, DiskStore& store,
                         PrefixIndex& index, std::uint64_t turnover_bytes)
    : impl_(std::make_unique<Impl>(
          ledger, store, index,
          ledger.Reserve(ResourceCategory::kMetadata, sizeof(Impl)),
          turnover_bytes)) {
  for (const auto& entry : store.Entries())
    if (entry.Durable())
      Track(entry.Manifest().checkpoint);
  impl_->loading = false;
}
DiskCatalog::~DiskCatalog() = default;
void DiskCatalog::Track(CheckpointId id) {
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
  const auto history =
      impl_->loading ? impl_->store->UseHistory(id) : DiskUseHistory{};
  const auto used = impl_->loading ? history.last_used : impl_->Tick();
  impl_->admitted_bytes =
      std::max(impl_->admitted_bytes, history.admitted_bytes);
  impl_->clock = std::max(impl_->clock, used);
  auto admission = impl_->ledger->Reserve(
      ResourceCategory::kMetadata,
      sizeof(decltype(impl_->records)::value_type) +
          sizeof(decltype(impl_->lru)::value_type) + 128);
  auto record =
      impl_->records
          .emplace(
              id.value,
              Impl::Record{
                  {}, std::move(description), used, history.reused_at, false})
          .first;
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
bool DiskCatalog::EvictOne() {
  Reconcile();
  for (auto& [id, record] : impl_->records) {
    (void)id;
    record.eviction_order = impl_->Order(record);
  }
  while (true) {
    auto record = impl_->records.end();
    for (auto it = impl_->records.begin(); it != impl_->records.end(); ++it) {
      if (it->second.eviction_order[0] == UINT64_MAX)
        continue;
      if (record == impl_->records.end() ||
          it->second.eviction_order < record->second.eviction_order)
        record = it;
    }
    if (record == impl_->records.end())
      return false;
    // Mark reader-pinned candidates as tried without allocating a side table.
    record->second.eviction_order.fill(UINT64_MAX);
    const auto& description = *record->second.description;
    auto result = impl_->store->TryRetire(description.Manifest().checkpoint,
                                          description.Epoch());
    if (result == DiskRetireResult::kBusy)
      return false;
    if (result == DiskRetireResult::kPinned)
      continue;
    impl_->index->RemoveDurable(description);
    impl_->Erase(record);
    if (result == DiskRetireResult::kRetired)
      return true;
  }
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
  for (auto& [id, record] : impl_->records) {
    (void)id;
    if (!record.invalid && record.description->Quarantined()) {
      record.invalid = true;
      impl_->index->RemoveDurable(*record.description, true);
    }
  }
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
std::size_t DiskCatalog::Size() const {
  return impl_->records.size();
}
}  // namespace gufo::cache
