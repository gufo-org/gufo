#ifndef GUFO_CACHE_DISK_CATALOG_HPP_
#define GUFO_CACHE_DISK_CATALOG_HPP_

#include "src/cache/prefix_index.hpp"

namespace gufo::cache {
// One global LRU catalog per store, including identities not currently loaded.
// Construct at quiescence after startup discovery; Track every successful new
// publication before exposing it through lookup. Operations are caller
// serialized. Store, index and ledger outlive the catalog. Touch records a
// successful use; startup ties and untouched publications use checkpoint ID.
class DiskCatalog {
public:
  DiskCatalog(ResourceLedger&, DiskStore&, PrefixIndex&);
  ~DiskCatalog();
  DiskCatalog(const DiskCatalog&) = delete;
  DiskCatalog& operator=(const DiskCatalog&) = delete;
  void Track(CheckpointId);
  [[nodiscard]] std::shared_ptr<const DiskDescription> Find(CheckpointId) const;
  void Touch(CheckpointId);
  // Evict oldest unpinned entry. Busy writers cause an immediate false result;
  // capacity is credited only after DiskStore's durable retirement succeeds.
  bool EvictOne();
  // Reverse dependency quarantine is immediate, with no I/O-lock wait. Retire
  // quarantined manifests at a later idle boundary after reader pins settle.
  void Invalidate(bool private_file, DiskFileId);
  // Reconcile worker-discovered quarantine at a serialized caller boundary.
  // Lookup rejects quarantined descriptions immediately, even before this.
  void Reconcile();
  std::size_t ReclaimInvalid();
  [[nodiscard]] std::size_t Size() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gufo::cache
#endif
