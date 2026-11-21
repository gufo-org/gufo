#ifndef GUFO_MEM_SHARED_ROWS_HPP_
#define GUFO_MEM_SHARED_ROWS_HPP_

#include <cstddef>
#include <memory>
#include <span>

namespace gufo::mem {

/// What the RAM cache needs to keep rows in the session pages that hold them
/// instead of a copy (shared pages: src/core/hip/state_shared_page.hpp).
///
/// HIP-free so the cache backing builds without the state backend; the
/// backend installs the factory where pages can be shared.
class SharedRows {
public:
  virtual ~SharedRows() = default;
  /// Keeps the rows at `view`, which a session wrote in a demand-mapped
  /// region, in their page. The returned owner holds the page; its get() is
  /// where the cache reads the rows from now on, whatever the session does.
  /// That address is readable by the device only. Null when these rows cannot
  /// be shared (they cross a page, or the region does not share pages): the
  /// caller copies them.
  [[nodiscard]] virtual std::shared_ptr<const void> Share(
      std::span<const std::byte> view) = 0;
  /// Whether `address` is one Share() returned.
  [[nodiscard]] virtual bool Contains(const void* address) const = 0;
  /// Releases the addresses of owners that are gone. It waits for the
  /// device: call it on the thread that runs the model, between its
  /// launches, or while idle.
  virtual void Sweep() = 0;
  /// Free pages in the state pool, where the pages of dropped owners return.
  [[nodiscard]] virtual std::size_t FreePages() const = 0;
  /// Gives up to `bytes` of the state pool's free pages to the host and
  /// returns the bytes released. Same thread rule as Sweep().
  virtual std::size_t ReleaseFree(std::size_t bytes) = 0;
  /// Lets the device reach memory mapped where other memory was unmapped
  /// (PagePool::SettleRemaps). Two atomic reads when nothing waits.
  virtual bool SettleRemaps() = 0;
};

using SharedRowsFactory = std::shared_ptr<SharedRows> (*)();
void SetSharedRowsFactory(SharedRowsFactory factory) noexcept;
/// The current device's shared rows; null where the cache copies or borrows.
[[nodiscard]] std::shared_ptr<SharedRows> CreateSharedRows();

}  // namespace gufo::mem

#endif  // GUFO_MEM_SHARED_ROWS_HPP_
