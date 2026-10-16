#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "src/core/hip/committed_backing.hpp"
#include "src/core/hip/transfer_pool.hpp"

using Clock = std::chrono::steady_clock;
using namespace gufo;
namespace {
void Check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}
void Require(bool ok) {
  if (!ok)
    throw std::runtime_error("copy verification failed");
}
double Milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}
constexpr std::size_t block = 128ULL << 20;
constexpr std::size_t payload = block - 13;
void Run(hip::CommittedBackingPool& backing, hip::TransferPool& streams,
         std::size_t piece, hip::CopyDirection direction, bool independent) {
  auto a = backing.TryAcquire(cache::ResourceCategory::kPrivateTail);
  auto b = backing.TryAcquire(cache::ResourceCategory::kPrivateTail);
  Require(a && b);
  void* device{};
  Check(hipMalloc(&device, 2 * block));
  std::memset(a->Bytes().data(), 0x51, block);
  std::memset(b->Bytes().data(), 0xa7, block);
  Check(hipMemcpy(device, a->Bytes().data(), block, hipMemcpyHostToDevice));
  Check(hipMemcpy(static_cast<std::byte*>(device) + block, b->Bytes().data(),
                  block, hipMemcpyHostToDevice));
  const bool out = direction == hip::CopyDirection::kDeviceToHost;
  const auto kind = out ? hipMemcpyDeviceToHost : hipMemcpyHostToDevice;
  auto* da = static_cast<std::byte*>(device) + 3;
  auto* db = static_cast<std::byte*>(device) + block + 3;
  auto* ha = a->Bytes().data() + 3;
  auto* hb = b->Bytes().data() + 3;
  std::vector<double> elapsed;
  std::vector<std::byte> readback(payload);
  for (int iteration = -1; iteration < 7; ++iteration) {
    if (out) {
      std::memset(ha, 0, payload);
      std::memset(hb, 0, payload);
    } else {
      Check(hipMemset(da, 0, payload));
      Check(hipMemset(db, 0, payload));
    }
    auto first = streams.TryAcquire();
    auto second = independent ? streams.TryAcquire() : nullptr;
    Require(first && (!independent || second));
    const auto start = Clock::now();
    if (independent) {
      auto ca =
          first->Copy(out ? ha : da, out ? da : ha, payload, direction, piece);
      auto cb =
          second->Copy(out ? hb : db, out ? db : hb, payload, direction, piece);
      Require(ca.Wait() == cache::TransferResult::kSucceeded);
      Require(cb.Wait() == cache::TransferResult::kSucceeded);
    } else {
      for (int region = 0; region < 2; ++region) {
        auto* dst = out ? (region ? hb : ha) : (region ? db : da);
        auto* src = out ? (region ? db : da) : (region ? hb : ha);
        for (std::size_t offset = 0; offset < payload;) {
          const auto bytes = std::min(piece, payload - offset);
          Check(hipMemcpyAsync(dst + offset, src + offset, bytes, kind,
                               first->Native()));
          offset += bytes;
        }
      }
      auto done = first->Complete();
      Require(done.Wait() == cache::TransferResult::kSucceeded);
    }
    const auto ms = Milliseconds(start);
    if (iteration >= 0) {
      elapsed.push_back(ms);
      std::printf("sample,%s,%zu,%s,%d,%.6f\n", out ? "d2h" : "h2d", piece,
                  independent ? "independent" : "shared", iteration, ms);
    }
    // Verification outside the measured scope, every iteration, both buffers.
    for (int region = 0; region < 2; ++region) {
      const std::byte* bytes = region ? hb : ha;
      if (!out) {
        Check(hipMemcpy(readback.data(), region ? db : da, payload,
                        hipMemcpyDeviceToHost));
        bytes = readback.data();
      }
      const auto expected = region ? std::byte{0xa7} : std::byte{0x51};
      Require(std::all_of(bytes, bytes + payload,
                          [expected](std::byte b) { return b == expected; }));
    }
  }
  std::sort(elapsed.begin(), elapsed.end());
  const auto median = elapsed[elapsed.size() / 2];
  std::printf("median,%s,%zu,%s,%.6f,%.3f\n", out ? "d2h" : "h2d", piece,
              independent ? "independent" : "shared", median,
              2.0 * payload / (median * 1e6));
  Check(hipFree(device));
}
}  // namespace
int main() {
  Check(hipInit(0));
  Check(hipFree(nullptr));  // Initialize runtime outside the pool commit timer.
  hipDeviceProp_t properties{};
  Check(hipGetDeviceProperties(&properties, 0));
  int runtime{};
  Check(hipRuntimeGetVersion(&runtime));
  std::printf("device,%s,%s,hip,%d,compiler,%s\n", properties.name,
              properties.gcnArchName, runtime, __VERSION__);
  constexpr std::size_t capacity = 1ULL << 30;
  constexpr std::size_t metadata = 1ULL << 20;
  cache::ResourceLedger ledger(
      {capacity + metadata, capacity + metadata, 0, capacity});
  const auto start = Clock::now();
  hip::CommittedBackingPool backing(ledger,
                                    {capacity + metadata, block, metadata});
  const auto commit = Milliseconds(start);
  std::printf("startup,bytes,%zu,ms,%.6f,ms_per_GiB,%.6f,allocations,%zu\n",
              backing.CommittedBytes(), commit, commit,
              backing.PageCommitAllocations());
  hip::TransferPool streams(2);
  for (auto piece : {64ULL << 10, 1ULL << 20, 4ULL << 20, 16ULL << 20}) {
    for (auto direction : {hip::CopyDirection::kDeviceToHost,
                           hip::CopyDirection::kHostToDevice}) {
      Run(backing, streams, piece, direction, false);
      Run(backing, streams, piece, direction, true);
    }
  }
}
