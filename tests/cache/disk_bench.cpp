#include <unistd.h>

#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

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
}  // namespace
int main() {
  ResourceLedger ledger{{1ULL << 32, 1ULL << 32, 0, 0}};
  std::cout << "100k_token_manifest_bytes="
            << EncodeManifest(Manifest(100000, 1)).size() << '\n';
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
      assert(store.Stats().legacy_probe_bytes_read == 0);
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
