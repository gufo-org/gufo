// GPU checks of transient pass scratch (src/core/hip/scratch_arena.*): the
// resident rows of every buffer stay backed and keep their contents while the
// wide rows are mapped and released again, wide rows are usable while mapped,
// a kernel writing all rows works only then, and the reservation bounds Take.
#include "src/core/hip/scratch_arena.hpp"

#include <hip/hip_runtime.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/core/hip/state_region.hpp"

namespace {

using gufo::mem::ScratchArena;

constexpr std::size_t kPage = ScratchArena::kPage;

void Check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
    std::exit(1);
  }
}

__global__ void Fill(std::uint32_t* data, std::size_t words,
                     std::uint32_t salt) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
       i += gridDim.x * blockDim.x)
    data[i] = salt ^ static_cast<std::uint32_t>(i * 2654435761u);
}

void FillWords(void* data, std::size_t bytes, std::uint32_t salt) {
  Fill<<<256, 256>>>(static_cast<std::uint32_t*>(data), bytes / 4, salt);
  Check(hipGetLastError(), "fill launch");
  Check(hipDeviceSynchronize(), "fill");
}

bool Matches(const void* data, std::size_t bytes, std::uint32_t salt) {
  std::vector<std::uint32_t> host(bytes / 4);
  Check(hipMemcpy(host.data(), data, bytes, hipMemcpyDeviceToHost), "read");
  for (std::size_t i = 0; i < host.size(); ++i)
    if (host[i] != (salt ^ static_cast<std::uint32_t>(i * 2654435761u)))
      return false;
  return true;
}

}  // namespace

int main() {
  Check(hipSetDevice(0), "device");
  std::string error;
  auto arena = ScratchArena::Reserve(64 * kPage, &error);
  if (!gufo::mem::DemandMapped()) {
    // Without demand mapping callers allocate whole narrow scratch.
    assert(arena == nullptr);
    std::puts("scratch_arena_test: no demand mapping, arena unavailable");
    return 0;
  }
  assert(arena != nullptr);

  // Two buffers: 3 pages wide with one resident, and 10 pages carved into
  // two sub-buffers, each with its first page resident.
  const std::size_t a_bytes = 3 * kPage;
  auto* a = static_cast<std::uint8_t*>(arena->Take(a_bytes, kPage, &error));
  assert(a != nullptr && error.empty());
  auto* b = static_cast<std::uint8_t*>(arena->Take(10 * kPage, 0, &error));
  assert(b != nullptr);
  assert(reinterpret_cast<std::uintptr_t>(b) % kPage == 0);
  assert(arena->MapResident(b, kPage / 2, &error));
  assert(arena->MapResident(b + 5 * kPage, kPage + 1, &error));
  assert(arena->resident_bytes() == 4 * kPage);
  assert(arena->transient_bytes() == 9 * kPage);
  assert(!arena->transient_mapped());

  FillWords(a, kPage, 1);
  FillWords(b, kPage, 2);
  FillWords(b + 5 * kPage, 2 * kPage, 3);

  for (std::uint32_t round = 0; round < 3; ++round) {
    assert(arena->MapTransient(&error));
    assert(arena->transient_mapped());
    assert(arena->MapTransient(&error));  // idempotent
    // Every carved byte is writable now.
    FillWords(a + kPage, a_bytes - kPage, 10 + round);
    FillWords(b + kPage, 4 * kPage, 20 + round);
    FillWords(b + 7 * kPage, 3 * kPage, 30 + round);
    assert(Matches(a + kPage, a_bytes - kPage, 10 + round));
    assert(Matches(b + 7 * kPage, 3 * kPage, 30 + round));
    arena->ReleaseTransient();
    assert(!arena->transient_mapped());
    // Resident rows kept their contents through the cycle.
    assert(Matches(a, kPage, 1));
    assert(Matches(b, kPage, 2));
    assert(Matches(b + 5 * kPage, 2 * kPage, 3));
  }
  arena->ReleaseTransient();  // no-op when released

  // Carving stops at the reservation, and while transient pages are mapped.
  assert(arena->Take(52 * kPage, 0, &error) == nullptr);
  error.clear();
  assert(arena->MapTransient(&error));
  assert(arena->Take(kPage, 0, &error) == nullptr);
  arena->ReleaseTransient();
  assert(arena->Take(kPage, kPage, &error) != nullptr);
  // A resident range outside the carved buffers is refused.
  assert(!arena->MapResident(a + 40 * kPage, kPage, &error));

  arena.reset();
  std::puts("scratch_arena_test: PASS");
  return 0;
}
