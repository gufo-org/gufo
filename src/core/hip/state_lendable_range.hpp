#ifndef GUFO_CORE_HIP_STATE_LENDABLE_RANGE_HPP_
#define GUFO_CORE_HIP_STATE_LENDABLE_RANGE_HPP_

#include <cstddef>
#include <memory>

#include "src/mem/lendable.hpp"

namespace gufo::mem {

/// The VMM backend of LendableRange (see src/mem/lendable.hpp): a reserved
/// device address range whose pages come from the state pool. Each page is
/// mapped for the GPU, exported as a dma-buf and mapped for the CPU at the
/// same address, so one pointer serves memcpy, kernels and copies. Measured
/// on gfx1151: CPU reads 28 GB/s and writes 26 GB/s (as malloc memory), GPU
/// copies 107 GB/s (registered host pages: about 51). Lending a page maps it
/// into a session as well; the cache keeps its own mapping, unused, until
/// the page comes back. Null when the pool's pages cannot be exported or the
/// runtime did not reserve the CPU range.
[[nodiscard]] std::shared_ptr<LendableRange> CreateVmmLendableRange(
    std::size_t bytes);

}  // namespace gufo::mem

#endif  // GUFO_CORE_HIP_STATE_LENDABLE_RANGE_HPP_
