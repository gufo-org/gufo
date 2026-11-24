// GPU checks of a session's declared state (src/core/hip/session_state.*):
// every kind lands where its declaration says, fixed buffers start zeroed,
// whole buffers are backed, growing buffers map pages only as rows are
// claimed and written, and the state gives every byte back when it ends.
#include "src/core/hip/session_state.hpp"

#include <hip/hip_runtime.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/core/hip/state_region.hpp"
#include "src/mem/ledger.hpp"
#include "src/mem/state_declaration.hpp"

namespace {

using gufo::mem::kPageBytes;
using gufo::mem::SessionState;
using gufo::mem::StateBytes;
using gufo::mem::StateDeclaration;
using gufo::mem::StateLedger;

void Check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
    std::exit(1);
  }
}

__global__ void CountNonZero(const std::uint8_t* data, std::size_t bytes,
                             std::uint32_t* bad) {
  std::uint32_t local = 0;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes;
       i += gridDim.x * blockDim.x)
    local += data[i] != 0;
  if (local)
    atomicAdd(bad, local);
}

__global__ void Fill(std::uint8_t* data, std::size_t bytes) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes;
       i += gridDim.x * blockDim.x)
    data[i] = 0x5a;
}

// Kernels read through the device's own translation of the address.
std::uint32_t NonZero(const void* data, std::size_t bytes) {
  std::uint32_t* bad = nullptr;
  Check(hipMalloc(&bad, sizeof(*bad)), "hipMalloc");
  Check(hipMemset(bad, 0, sizeof(*bad)), "hipMemset");
  CountNonZero<<<256, 256>>>(static_cast<const std::uint8_t*>(data), bytes,
                             bad);
  std::uint32_t host = 0;
  Check(hipMemcpy(&host, bad, sizeof(host), hipMemcpyDeviceToHost),
        "hipMemcpy");
  Check(hipFree(bad), "hipFree");
  return host;
}

void TestKinds() {
  const bool demand = gufo::mem::DemandMapped();
  const auto before = StateLedger::Process().Totals();
  StateDeclaration declaration(32);
  // K/V-like rows: two planes, 3 MiB each.
  const auto kv =
      declaration.Context({.row_bytes = 1536, .rows = 2048, .segments = 2}, 1);
  // Compressed rows, four tokens each.
  const auto compressed =
      declaration.Context({.row_bytes = 2048, .rows = 1026}, 4);
  const auto whole = declaration.Whole({.row_bytes = 256, .rows = 4097});
  const auto fixed = declaration.Fixed(12345);
  std::string error;
  {
    auto state = SessionState::Create(declaration, &error);
    assert(state && error.empty());
    assert(state->allocated_bytes() == declaration.Bytes(StateBytes::kSize));
    for (const auto id : {kv, compressed, whole, fixed})
      assert(state->data(id) != nullptr);
    assert(state->region(fixed) == nullptr);
    assert(state->region(kv)->demand_mapped() == demand);
    assert(state->region(kv)->shape().segments == 2);
    assert(state->data(kv) == state->region(kv)->data());
    // Every region before its first write reads zero, and the fixed buffer
    // too, from any stream once Create() returned.
    assert(NonZero(state->data(fixed), 12345) == 0);
    assert(NonZero(state->data(whole), 256 * 4097) == 0);
    // A whole buffer is backed at once; growing buffers only as written.
    assert(state->region(whole)->backed_rows() == 4097);
    const auto& account = state->account();
    assert(account != nullptr);
    assert(account->full_bytes() ==
           declaration.ContextBytes(StateBytes::kPages));
    if (demand) {
      assert(state->region(kv)->backed_rows() == 0);
      assert(state->region(compressed)->backed_rows() == 0);
      assert(account->mapped_bytes() == 0);
      // The ledger holds every region's pages as reserved.
      const auto during = StateLedger::Process().Totals();
      assert(during.reserved_bytes - before.reserved_bytes ==
             declaration.Bytes(StateBytes::kPages) -
                 declaration.Bytes(fixed, StateBytes::kPages));
      // A claim of 100 tokens backs 100 + 32 rows of K/V and 33 compressed
      // rows; the account charges their pages.
      assert(account->Claim(100));
      assert(state->region(kv)->Ensure(132, &error));
      assert(state->region(compressed)->Ensure(33, &error));
      assert(account->mapped_bytes() == 3 * kPageBytes);
      Fill<<<64, 256>>>(state->data<std::uint8_t>(kv), 132 * 1536);
      Check(hipDeviceSynchronize(), "fill");
    }
  }
  // Destruction gives back every reservation and mapping.
  Check(hipDeviceSynchronize(), "sync");
  const auto after = StateLedger::Process().Totals();
  assert(after.reserved_bytes == before.reserved_bytes);
  assert(after.mapped_bytes == before.mapped_bytes);
}

void TestNoContext() {
  // Without kContext buffers there is nothing to charge: no account.
  StateDeclaration declaration;
  const auto a = declaration.Fixed(64);
  const auto b = declaration.Whole({.row_bytes = 64, .rows = 64});
  std::string error;
  auto state = SessionState::Create(declaration, &error);
  assert(state && state->account() == nullptr);
  assert(state->data(a) != state->data(b));
}

void TestRepeatedSessions() {
  // Sessions come and go between requests: the pages return to the pool
  // and the next session's rows read zero again.
  StateDeclaration declaration(0);
  const auto rows = declaration.Context({.row_bytes = 4096, .rows = 1024});
  std::string error;
  for (int round = 0; round < 4; ++round) {
    auto state = SessionState::Create(declaration, &error);
    assert(state);
    assert(state->account()->Claim(1024));
    assert(state->region(rows)->Ensure(1024, &error));
    assert(NonZero(state->data(rows), 1024 * 4096) == 0);
    Fill<<<64, 256>>>(state->data<std::uint8_t>(rows), 1024 * 4096);
    Check(hipDeviceSynchronize(), "fill");
  }
}

}  // namespace

int main() {
  Check(hipSetDevice(0), "hipSetDevice");
  TestKinds();
  TestNoContext();
  TestRepeatedSessions();
  std::printf("mem_session_state_test: ok (demand_mapped=%d)\n",
              gufo::mem::DemandMapped() ? 1 : 0);
  return 0;
}
