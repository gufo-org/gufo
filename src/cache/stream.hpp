#ifndef GUFO_CACHE_STREAM_HPP_
#define GUFO_CACHE_STREAM_HPP_

#include "src/cache/completion.hpp"

namespace gufo::cache {

// The cache owns streams. Adapters may reject an incompatible stream type.
// Transfers on one stream are FIFO; different streams have no implicit order.
// Synchronize settles all submitted work and reports any failure it drains.
// Device-backed implementations live outside this package (card 08).
class Stream {
public:
  Stream() = default;
  virtual ~Stream() = default;
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  Stream(Stream&&) = delete;
  Stream& operator=(Stream&&) = delete;
  [[nodiscard]] virtual TransferResult Synchronize() noexcept = 0;
};

}  // namespace gufo::cache

#endif
