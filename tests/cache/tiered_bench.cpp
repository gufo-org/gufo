#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "src/cache/disk_catalog.hpp"

using namespace gufo::cache;
namespace fs = std::filesystem;
namespace {
using Clock = std::chrono::steady_clock;
double Milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}
DiskFileId Id(unsigned tag, std::uint64_t n) {
  DiskFileId file{};
  file[0] = tag;
  for (unsigned i = 0; i < 8; ++i)
    file[1 + i] = n >> (i * 8);
  return file;
}
struct Directory {
  fs::path path;
  explicit Directory(const fs::path& parent) {
    auto pattern = (parent / "gufo-tiered-bench-XXXXXX").string();
    auto* result = mkdtemp(pattern.data());
    if (!result)
      throw std::runtime_error("cannot create benchmark directory");
    path = result;
  }
  ~Directory() { fs::remove_all(path); }
};
}  // namespace
int main(int argc, char** argv) {
  try {
    Directory directory(argc > 1 ? argv[1] : "/tmp");
    ResourceLedger ledger{{1 << 28, 1 << 27, 1 << 20, 1 << 24}};
    auto source_charge =
        ledger.Reserve(ResourceCategory::kPrivateState, 32).Convert();
    const std::array<std::uint8_t, 32> source{};
    DiskStore disk(ledger, directory.path, 1 << 24);
    PrefixIndex index(ledger);
    CompatibilityDigest digest{};
    digest[0] = 42;
    const std::array<ComponentDescriptor, 1> inventory{
        {{{1}, 1, ComponentKind::kPrivateState, 0, 0, 32}}};
    index.Register({1}, inventory, digest);
    auto prompt_admission =
        ledger.Reserve(ResourceCategory::kMetadata, 1000 * sizeof(Token));
    ResourceCharge prompt_charge;
    std::vector<Token> prompt(1000, 7);
    prompt_charge = prompt_admission.Convert();
    for (unsigned n = 1; n <= 1000; ++n) {
      auto admission = ledger.Reserve(
          ResourceCategory::kMetadata,
          sizeof(DiskManifest) + sizeof(DiskComponent) + n * sizeof(Token));
      ResourceCharge charge;
      DiskManifest manifest;
      manifest.checkpoint = {n};
      manifest.lineage = {1};
      manifest.compatibility = digest;
      manifest.tokens.assign(prompt.begin(), prompt.begin() + n);
      manifest.components.push_back(
          {inventory[0],
           {{1}, n},
           {},
           {},
           DiskPayload{Id(1, n), source.size(), DiskChecksum(source)}});
      charge = admission.Convert();
      const std::array<DiskWriteBuffer, 1> buffers{{{true, Id(1, n), source}}};
      disk.Publish(Id(2, n), manifest, buffers);
    }
    DiskCatalog catalog(ledger, disk, index);
    for (unsigned n = 1; n <= 1000; ++n)
      (void)index.Insert(Identity{1}, catalog.Find({n}));
    PrefixQuery query{{1}, prompt, InputIdentity(prompt.size()), 0};
    auto lookup = [&] {
      const auto result = index.Lookup(query);
      if (!result.selected || result.selected->boundary != 1000 ||
          result.candidates.size() != 1000 ||
          result.reason != SelectionReason::kDeepestDurableCheckpoint)
        throw std::runtime_error("invalid lookup benchmark result");
    };
    for (unsigned n = 0; n < 10; ++n)
      lookup();
    std::cout << "phase,sample,checkpoints,ms,fsync_calls\n";
    for (unsigned n = 0; n < 100; ++n) {
      const auto start = Clock::now();
      lookup();
      std::cout << "lookup," << n << ",1000," << Milliseconds(start) << ",0\n";
    }
    const ComponentDescriptor rows{{2}, 1, ComponentKind::kAppendRows, 8, 1, 0};
    auto publish = [&](unsigned n, DiskFileId chunk, bool new_file) {
      auto admission =
          ledger.Reserve(ResourceCategory::kMetadata,
                         sizeof(DiskManifest) + sizeof(DiskComponent) +
                             sizeof(Token) + sizeof(DiskPayload));
      ResourceCharge charge;
      DiskManifest manifest;
      manifest.checkpoint = {n};
      manifest.lineage = {2};
      manifest.compatibility[0] = 43;
      manifest.tokens = {7};
      manifest.components = {
          {rows,
           {{2}, 1},
           {{chunk, 8, DiskChecksum(std::span(source).first(8))}},
           {},
           {}}};
      charge = admission.Convert();
      const std::array<DiskWriteBuffer, 1> buffers{
          {{false, chunk, std::span(source).first(8)}}};
      disk.Publish(Id(2, n), manifest,
                   new_file ? std::span<const DiskWriteBuffer>(buffers)
                            : std::span<const DiskWriteBuffer>{});
      catalog.Track({n});
    };
    const auto shared = Id(3, 2001), unshared = Id(3, 2003);
    publish(2001, shared, true);
    publish(2002, shared, false);
    publish(2003, unshared, true);
    for (unsigned n = 1; n <= 1000; ++n)
      catalog.Touch({n});
    auto evict = [&](const char* phase) {
      const auto before = disk.PublicationStats();
      const auto count = catalog.Size();
      const auto start = Clock::now();
      if (!catalog.EvictOne())
        throw std::runtime_error("benchmark eviction refused");
      const auto elapsed = Milliseconds(start);
      const auto after = disk.PublicationStats();
      std::cout << phase << ",0," << count << ',' << elapsed << ','
                << after.fsync_calls - before.fsync_calls << '\n';
    };
    evict("evict_shared");
    if (disk.Open({2001}) || !disk.Open({2002}) ||
        !fs::exists(directory.path / "v2/chunks" / DiskFileName(shared)))
      throw std::runtime_error("shared dependency eviction failed");
    catalog.Touch({2002});
    evict("evict_unshared");
    if (disk.Open({2003}) ||
        fs::exists(directory.path / "v2/chunks" / DiskFileName(unshared)))
      throw std::runtime_error("unshared dependency eviction failed");
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
