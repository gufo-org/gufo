// Snapshot buffer pool: released buffers must return to the pool and serve
// the next capture without a fresh mapping, indefinitely — the pooled-byte
// budget tracks retained buffers, so repeated reuse of one buffer must not
// exhaust the cap cumulatively (the accounting regression found in review:
// Acquire removed a buffer without releasing its allowance, so ~4 GiB of
// cumulative reuse silently disabled recycling).
//
// Units are 1 KiB (review suggestion): the pool's limits are injectable and
// only the ratios matter, so the suite runs without GiB-scale virtual
// allocations while pinning the same six reuse cycles, smallest-fit and
// byte-bound behaviors as production sizes.
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "src/models/qwen38_flash_next/snapshot_buffer_pool.hpp"

namespace qfn = gufo::models::qwen38_flash_next;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

constexpr std::uint64_t kUnit = std::uint64_t{1} << 10;

void Run() {
  // A tight byte limit so six reuse cycles of one unit exceed it
  // cumulatively: the regression disabled reuse after four cycles here.
  qfn::SnapshotBufferPool pool({.max_buffers = 8, .max_bytes = 4 * kUnit});

  std::uint8_t* address = nullptr;
  for (int cycle = 0; cycle < 6; ++cycle) {
    auto acquired = pool.Acquire(kUnit);
    Require(acquired.data != nullptr, "acquire returned no buffer");
    Require(acquired.capacity == kUnit, "acquire reported the wrong capacity");
    if (cycle == 0) {
      Require(!acquired.populated, "first acquire cannot be recycled");
    } else {
      Require(acquired.populated,
              "reuse stopped at cycle " + std::to_string(cycle) +
                  "; pooled budget exhausted cumulatively");
      Require(acquired.data.get() == address,
              "reuse served a different buffer");
    }
    address = acquired.data.get();
    pool.Release(std::move(acquired.data), acquired.capacity);
    Require(pool.pooled_buffers() == 1, "released buffer was not retained");
    Require(pool.pooled_bytes() == kUnit,
            "pooled bytes do not track retained buffers");
  }

  // Smallest-fit: the unit buffer serves the small ask; the 2-unit pooled
  // buffer remains available for the larger one with its real capacity.
  auto large = pool.Acquire(2 * kUnit);
  Require(!large.populated, "no 2-unit buffer is pooled yet");
  pool.Release(std::move(large.data), large.capacity);
  auto small = pool.Acquire(1 * kUnit);
  Require(small.populated, "unit pooled buffer did not serve the small ask");
  Require(small.capacity == kUnit, "smallest-fit picked the wrong buffer");
  auto big = pool.Acquire(2 * kUnit);
  Require(big.populated, "2-unit pooled buffer did not serve the larger ask");
  Require(big.capacity == 2 * kUnit, "smallest-fit lost the real capacity");

  // Byte-budget bound: two distinct buffers whose pooled bytes exceed the
  // limit — the second release is dropped instead of pooled (the caller's
  // unique_ptr frees the buffer).
  qfn::SnapshotBufferPool bounded({.max_buffers = 8, .max_bytes = 2 * kUnit});
  auto first = bounded.Acquire(kUnit);
  Require(!first.populated, "fresh bounded pool cannot recycle");
  bounded.Release(std::move(first.data), first.capacity);
  auto second = bounded.Acquire(2 * kUnit);
  Require(!second.populated, "unit pooled buffer cannot serve a 2-unit ask");
  bounded.Release(std::move(second.data), second.capacity);
  Require(bounded.pooled_buffers() == 1,
          "byte-bound pool retained more than its budget");
  Require(bounded.pooled_bytes() == kUnit,
          "byte-bound pool miscounted its allowance");
}

}  // namespace

int main() {
  try {
    Run();
  } catch (const std::exception& error) {
    std::cerr << "snapshot_buffer_pool_test: " << error.what() << "\n";
    return 1;
  }
  std::cout << "snapshot_buffer_pool_test: passed\n";
  return 0;
}
