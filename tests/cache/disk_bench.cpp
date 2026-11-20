#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string_view>

#include "src/cache/disk.hpp"

using namespace gufo::cache;
namespace fs = std::filesystem;
namespace {
DiskFileId Id(std::uint64_t n) {
  DiskFileId id{};
  for (unsigned i = 0; i < 8; ++i)
    id[i] = (n >> (8 * i)) & 255;
  return id;
}
void Write(const fs::path& path, std::span<const std::uint8_t> bytes) {
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (!file.good())
    throw std::runtime_error("fixture write failed");
}
DiskManifest Manifest(std::size_t tokens, std::uint64_t checkpoint) {
  DiskManifest m;
  m.checkpoint = {checkpoint};
  m.lineage = {1};
  m.compatibility.fill(0x42);
  m.tokens.assign(tokens, 17);
  DiskComponent c{{{1}, 1, ComponentKind::kAppendRows, 4, 2048, 0},
                  {{1}, tokens},
                  {},
                  {},
                  {}};
  for (std::size_t i = 0; i < tokens / 2048; ++i)
    c.chunks.push_back({Id(i + 1), 8192, 1});
  if (tokens % 2048)
    c.tail = DiskPayload{Id(10000 + checkpoint), 4 * (tokens % 2048), 2};
  m.components.push_back(std::move(c));
  m.components.push_back({{{2}, 1, ComponentKind::kPrivateState, 0, 0, 16},
                          {{2}, tokens},
                          {},
                          {},
                          DiskPayload{Id(20000 + checkpoint), 16, 3}});
  return m;
}
// 256 row components of 256 chunks each: the reference limit.
DiskManifest WideManifest() {
  DiskManifest m;
  m.checkpoint = {1};
  m.lineage = {1};
  for (std::uint32_t i = 0; i < 256; ++i) {
    DiskComponent c{{{i + 1}, 1, ComponentKind::kAppendRows, 4, 16, 0},
                    {{i + 1}, 256 * 16},
                    {},
                    {},
                    {}};
    for (unsigned j = 0; j < 256; ++j)
      c.chunks.push_back({Id(i * 256 + j + 1), 64, j});
    m.components.push_back(std::move(c));
  }
  return m;
}
// The fastest of `runs` timed calls, in microseconds.
template<class F>
double Best(unsigned runs, F&& call) {
  double best = 1e300;
  for (unsigned i = 0; i < runs; ++i) {
    const auto begin = std::chrono::steady_clock::now();
    call();
    best = std::min(best, std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - begin)
                              .count());
  }
  return best;
}
void Codec() {
  const std::pair<const char*, DiskManifest> manifests[] = {
      {"100k_tokens", Manifest(100000, 1)},
      {"2m_tokens", Manifest(2 * 1024 * 1024, 1)},
      {"65536_references", WideManifest()},
  };
  for (const auto& [name, m] : manifests) {
    const auto bytes = EncodeManifest(m);
    std::size_t sink{};
    const auto encode = Best(200, [&] { sink += EncodeManifest(m).size(); });
    const auto decode =
        Best(200, [&] { sink += DecodeManifest(bytes).tokens.size(); });
    std::cout << "manifest=" << name << " bytes=" << bytes.size()
              << " encode_us=" << encode << " decode_us=" << decode
              << " sink=" << sink % 2 << '\n';
  }
}
void Checksum() {
  std::vector<std::uint8_t> data((std::size_t{64} << 20) + 64);
  std::mt19937_64 random(1);
  for (auto& b : data)
    b = random();
  for (std::size_t bytes :
       {std::size_t{40}, std::size_t{64}, std::size_t{128}, std::size_t{255},
        std::size_t{256}, std::size_t{4096}, std::size_t{1} << 16,
        std::size_t{1} << 20, std::size_t{64} << 20}) {
    // Each run covers at least 64 MiB, in pieces of `bytes` at unaligned
    // offsets, as staging pieces can be.
    const std::size_t pieces = std::max<std::size_t>(1, (64 << 20) / bytes);
    std::uint64_t crc{};
    const auto us = Best(10, [&] {
      for (std::size_t i = 0; i < pieces; ++i)
        crc = DiskChecksum(
            std::span(data).subspan(1 + (i * bytes) % (data.size() - bytes - 1),
                                    bytes),
            crc);
    });
    std::cout << "checksum_bytes=" << bytes
              << " gib_per_s=" << double(bytes) * pieces / us * 1e6 / (1 << 30)
              << " ns_per_call=" << us * 1e3 / pieces << " crc=" << crc % 2
              << '\n';
  }
}
}  // namespace
int main(int argc, char** argv) {
  ResourceLedger ledger{{1ULL << 32, 1ULL << 32, 0, 0}};
  std::cout << "100k_token_manifest_bytes="
            << EncodeManifest(Manifest(100000, 1)).size() << '\n';
  Codec();
  Checksum();
  if (argc > 1 && std::string_view(argv[1]) == "--codec")
    return 0;
  for (unsigned count : {100, 1000}) {
    char name[] = "/tmp/gufo-disk-bench-XXXXXX";
    fs::path root = mkdtemp(name);
    try {
      {
        DiskStore init(ledger, root, UINT64_MAX);
      }
      std::size_t expected{};
      for (unsigned i = 1; i <= count; ++i) {
        auto m = Manifest(100000, i);
        auto bytes = EncodeManifest(m);
        expected += bytes.size();
        Write(root / "v2/manifests" / DiskFileName(Id(i)), bytes);
        for (const auto& c : m.components) {
          for (const auto& p : c.chunks) {
            auto path = root / "v2/chunks" / DiskFileName(p.file);
            if (!fs::exists(path)) {
              Write(path, {});
              fs::resize_file(path, p.bytes);
            }
          }
          for (auto p : {c.tail, c.private_state}) {
            if (p) {
              auto path = root / "v2/private" / DiskFileName(p->file);
              Write(path, {});
              fs::resize_file(path, p->bytes);
            }
          }
        }
      }
      auto begin = std::chrono::steady_clock::now();
      DiskStore store(ledger, root, UINT64_MAX);
      auto elapsed = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - begin)
                         .count();
      assert(store.Entries().size() == count);
      assert(store.Stats().manifest_bytes_read == expected);
      std::cout << "manifests=" << count << " startup_ms=" << elapsed
                << " manifest_bytes_read=" << store.Stats().manifest_bytes_read
                << " dependency_stats=" << store.Stats().dependency_stats
                << '\n';
    } catch (...) {
      fs::remove_all(root);
      throw;
    }
    fs::remove_all(root);
  }
}
