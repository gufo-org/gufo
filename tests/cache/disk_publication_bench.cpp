#include <unistd.h>

#include <cassert>
#include <chrono>
#include <filesystem>
#include <iostream>

#include "src/cache/disk.hpp"
using namespace gufo::cache;
DiskFileId Id(unsigned n) {
  DiskFileId id{};
  id[0] = n;
  id[1] = n >> 8;
  return id;
}
int main(int argc, char** argv) {
  std::filesystem::path parent = argc > 1 ? argv[1] : "/tmp";
  for (std::size_t tokens : {128, 100000}) {
    auto pattern = (parent / "gufo-publication-bench-XXXXXX").string();
    auto root = std::filesystem::path(mkdtemp(pattern.data()));
    {
      ResourceLedger ledger{{1ULL << 32, 1ULL << 32, 0, 0}};
      DiskStore store(ledger, root, 1ULL << 32);
      DiskManifest m;
      m.checkpoint = {1};
      m.lineage = {1};
      m.tokens.assign(tokens, 17);
      DiskComponent c{{{1}, 1, ComponentKind::kAppendRows, 4, 2048, 0},
                      {{1}, tokens},
                      {},
                      {},
                      {}};
      std::vector<std::vector<std::uint8_t>> data(tokens / 2048 + 2);
      std::vector<DiskWriteBuffer> buffers;
      for (std::size_t i = 0; i < tokens / 2048; ++i) {
        data[i].assign(8192, 42);
        c.chunks.push_back({Id(i + 1), 8192, DiskChecksum(data[i])});
        buffers.push_back({false, Id(i + 1), data[i]});
      }
      auto i = tokens / 2048;
      data[i].assign(4 * (tokens % 2048), 42);
      c.tail = DiskPayload{Id(1000), data[i].size(), DiskChecksum(data[i])};
      buffers.push_back({true, Id(1000), data[i]});
      m.components.push_back(std::move(c));
      data[i + 1].assign(16, 9);
      m.components.push_back(
          {{{2}, 1, ComponentKind::kPrivateState, 0, 0, 16},
           {{2}, tokens},
           {},
           {},
           DiskPayload{Id(1001), 16, DiskChecksum(data[i + 1])}});
      buffers.push_back({true, Id(1001), data[i + 1]});
      auto begin = std::chrono::steady_clock::now();
      store.Publish(Id(2000), m, buffers);
      auto ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - begin)
                    .count();
      assert(store.Entries().size() == 1);
      auto stats = store.PublicationStats();
      std::cout << "tokens=" << tokens
                << " bytes=" << store.Stats().managed_bytes
                << " publication_ms=" << ms
                << " fsync_calls=" << stats.fsync_calls
                << " fsync_ms=" << stats.fsync_ns / 1e6 << '\n';
    }
    std::filesystem::remove_all(root);
  }
}
