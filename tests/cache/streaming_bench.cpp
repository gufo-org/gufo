#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include "src/cache/streaming.hpp"
#include "src/core/hip/staging.hpp"
#include "src/core/hip/transfer_pool.hpp"

using namespace gufo;
namespace {
using Clock = std::chrono::steady_clock;
void Check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}
void Require(bool ok) {
  if (!ok)
    throw std::runtime_error("streamed round-trip verification failed");
}
struct Device {
  cache::ResourceCharge charge;
  void* data{};
  ~Device() {
    if (data)
      (void)hipFree(data);
  }
};
cache::DiskFileId Id(unsigned n) {
  cache::DiskFileId id{};
  id[0] = n;
  id[1] = n >> 8;
  return id;
}
double Milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}
// Independent deterministic noncompressible bytes. No whole host image.
void Pattern(std::span<std::uint8_t> piece, std::uint64_t offset) {
  for (std::size_t i = 0; i < piece.size(); i += 8) {
    auto x = (offset + i) / 8 + 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= x >> 31;
    for (std::size_t j = 0; j < std::min(std::size_t{8}, piece.size() - i); ++j)
      piece[i + j] = x >> (j * 8);
  }
}
}  // namespace
int main(int argc, char** argv) {
  const std::size_t bytes = argc > 1 ? std::stoull(argv[1]) : 256ULL << 20;
  const std::size_t piece =
      argc > 2 ? std::stoull(argv[2]) : cache::kDefaultDiskStagingBytes;
  if (!bytes || !piece || piece % 8)
    throw std::invalid_argument(
        "positive bytes and eight-byte-aligned piece required");
  const auto parent = argc > 3 ? std::filesystem::path(argv[3]) : "/tmp";
  auto pattern = (parent / "gufo-streaming-bench-XXXXXX").string();
  auto* name = mkdtemp(pattern.data());
  if (!name)
    throw std::runtime_error("mkdtemp failed");
  const std::filesystem::path directory(name);
  cache::ResourceLedger ledger{{2 * bytes + 128ULL * 1024 * 1024,
                                2 * bytes + 64ULL * 1024 * 1024, piece, bytes}};
  hip::TransferPool transfers(1);
  auto make_device = [&] {
    auto admission =
        ledger.Reserve(cache::ResourceCategory::kPrivateState, bytes);
    auto device = std::make_shared<Device>();
    Check(hipMalloc(&device->data, bytes));
    device->charge = admission.Convert();
    return device;
  };
  auto source = make_device(), destination = make_device();
  auto allocator = hip::PinnedStagingAllocator(ledger);
  // Initialization and readback use the same bounded allocation, outside the
  // measured publication/restore scopes. Fully initialize device pages first.
  {
    auto admission =
        ledger.Reserve(cache::ResourceCategory::kTransferStaging, piece);
    cache::ResourceCharge charge;
    auto staging = allocator(piece);
    charge = admission.Convert();
    for (std::size_t offset = 0; offset < bytes; offset += piece) {
      auto buffer = staging.bytes.first(std::min(piece, bytes - offset));
      Pattern(buffer, offset);
      Check(hipMemcpy(static_cast<std::byte*>(source->data) + offset,
                      buffer.data(), buffer.size(), hipMemcpyHostToDevice));
    }
    Check(hipMemset(destination->data, 0, bytes));
  }
  cache::DiskManifest manifest;
  manifest.checkpoint = {1};
  manifest.lineage = {1};
  manifest.tokens.resize(100000, 42);
  manifest.components.push_back(
      {{{1}, 1, cache::ComponentKind::kPrivateState, 0, 0, bytes},
       {{1}, 100000},
       {},
       {},
       cache::DiskPayload{Id(1), bytes, 0}});
  cache::DiskStore disk(ledger, directory, bytes + (1 << 20));
  std::printf(
      "bytes,piece,phase,elapsed_ms,GB_s,allocator_ms,stream_acquire_ms,stream_"
      "wait_ms,metadata_lock_ms,filesystem_ms,fsync_ms,checksum_ms,peak_"
      "staging\n");
  cache::TransferTiming write_cost;
  double write_ms{};
  {
    cache::StreamedStore writer(ledger, disk, piece, allocator,
                                [&]() -> std::unique_ptr<cache::Stream> {
                                  return transfers.TryAcquire();
                                });
    std::vector<cache::PersistenceSource> sources;
    sources.emplace_back(
        true, Id(1), bytes, 1, source->charge, source,
        [source_ptr = source->data](auto offset, auto buffer,
                                    cache::Stream& stream) {
          return dynamic_cast<hip::TransferStream&>(stream).Copy(
              buffer.data(), static_cast<std::byte*>(source_ptr) + offset,
              buffer.size(), hip::CopyDirection::kDeviceToHost, buffer.size());
        });
    const auto start = Clock::now();
    writer.Write(Id(2), manifest, sources, write_cost);
    write_ms = Milliseconds(start);
    std::printf(
        "%zu,%zu,write,%.3f,%.6f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%zu\n",
        bytes, piece, write_ms, bytes / (write_ms * 1e6),
        write_cost.allocator_ns / 1e6, write_cost.stream_acquire_ns / 1e6,
        write_cost.stream_wait_ns / 1e6, write_cost.metadata_lock_ns / 1e6,
        write_cost.filesystem_ns / 1e6, write_cost.fsync_ns / 1e6,
        write_cost.checksum_ns / 1e6,
        ledger.Snapshot().peak_bytes[static_cast<std::size_t>(
            cache::ResourceCategory::kTransferStaging)]);
  }
  auto snapshot = disk.Open({1});
  const auto payload = *snapshot->Manifest().components[0].private_state;
  // Hint eviction only for this owned, synced payload. Report this as an
  // advised read, not proof that physical storage served every byte.
  auto file = directory / "v2/private" / cache::DiskFileName(Id(1));
  int fd = open(file.c_str(), O_RDONLY | O_CLOEXEC);
  Require(fd >= 0 && posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) == 0);
  close(fd);
  {
    auto admission =
        ledger.Reserve(cache::ResourceCategory::kTransferStaging, piece);
    cache::ResourceCharge charge;
    auto staging = allocator(piece);
    charge = admission.Convert();
    cache::TransferTiming costs;
    const auto start = Clock::now();
    const auto read_cost = disk.ReadPayload(
        *snapshot, true, payload, staging.bytes, [&](auto offset, auto buffer) {
          auto acquire = Clock::now();
          auto stream = transfers.TryAcquire();
          costs.stream_acquire_ns +=
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  Clock::now() - acquire)
                  .count();
          Require(bool(stream));
          auto copy =
              stream->Copy(static_cast<std::byte*>(destination->data) + offset,
                           buffer.data(), buffer.size(),
                           hip::CopyDirection::kHostToDevice, buffer.size());
          auto wait = Clock::now();
          Require(copy.Wait() == cache::TransferResult::kSucceeded);
          costs.stream_wait_ns +=
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  Clock::now() - wait)
                  .count();
        });
    const auto restore_ms = Milliseconds(start);
    Require(!read_cost.verification_cached && read_cost.bytes == bytes);
    std::printf(
        "%zu,%zu,restore-advised,%.3f,%.6f,0,%.3f,%.3f,0,%.3f,0,%.3f,%zu\n",
        bytes, piece, restore_ms, bytes / (restore_ms * 1e6),
        costs.stream_acquire_ns / 1e6, costs.stream_wait_ns / 1e6,
        read_cost.filesystem_ns / 1e6, read_cost.checksum_ns / 1e6,
        ledger.Snapshot().peak_bytes[static_cast<std::size_t>(
            cache::ResourceCategory::kTransferStaging)]);
    // Independent verification of every restored device byte, after timing.
    auto oracle_charge =
        ledger.Reserve(cache::ResourceCategory::kMetadata, piece);
    std::vector<std::uint8_t> expected(piece);
    for (std::size_t offset = 0; offset < bytes; offset += piece) {
      auto buffer = staging.bytes.first(std::min(piece, bytes - offset));
      Check(hipMemcpy(buffer.data(),
                      static_cast<std::byte*>(destination->data) + offset,
                      buffer.size(), hipMemcpyDeviceToHost));
      Pattern(std::span(expected).first(buffer.size()), offset);
      Require(std::equal(buffer.begin(), buffer.end(), expected.begin()));
    }
  }
  snapshot.reset();
  Require(disk.Retire({1}));
  std::filesystem::remove_all(directory);
}
