#include "src/cache/disk.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace gufo::cache;
namespace fs = std::filesystem;
namespace {
std::atomic<std::uint64_t> actual_manifest_bytes{}, actual_payload_bytes{};
std::atomic<bool> interrupt_next_read{}, shorten_reads{};
void CountRead(int fd, ssize_t bytes) {
  if (bytes <= 0)
    return;
  char path[4096];
  auto n = readlink(("/proc/self/fd/" + std::to_string(fd)).c_str(), path,
                    sizeof(path));
  if (n <= 0)
    return;
  std::string_view name(path, n);
  if (name.find("/v2/manifests/") != std::string_view::npos)
    actual_manifest_bytes += bytes;
  if (name.find("/v2/chunks/") != std::string_view::npos ||
      name.find("/v2/private/") != std::string_view::npos)
    actual_payload_bytes += bytes;
}
}  // namespace
extern "C" ssize_t __real_read(int, void*, size_t);
extern "C" ssize_t __wrap_read(int fd, void* buffer, size_t size) {
  if (interrupt_next_read.exchange(false)) {
    errno = EINTR;
    return -1;
  }
  if (shorten_reads && size > 1)
    size = 1;
  auto n = __real_read(fd, buffer, size);
  CountRead(fd, n);
  return n;
}
extern "C" ssize_t __real_pread(int, void*, size_t, off_t);
extern "C" ssize_t __wrap_pread(int fd, void* buffer, size_t size,
                                off_t offset) {
  auto n = __real_pread(fd, buffer, size, offset);
  CountRead(fd, n);
  return n;
}
namespace {
ResourceLedger ledger{{1ULL << 32, 1ULL << 32, 0, 0}};
struct Directory {
  fs::path path;
  Directory() {
    char name[] = "/tmp/gufo-disk-test-XXXXXX";
    path = mkdtemp(name);
  }
  ~Directory() { fs::remove_all(path); }
};
DiskFileId Id(unsigned n) {
  DiskFileId id{};
  id[0] = n & 255;
  id[1] = (n >> 8) & 255;
  return id;
}
DiskManifest Manifest() {
  DiskManifest m;
  m.checkpoint = {7};
  m.lineage = {3};
  m.compatibility.fill(0x42);
  m.input = {1, 2, 3};
  m.tokens = {17, 18, 19};
  m.components.push_back({{{1}, 4, ComponentKind::kAppendRows, 4, 2, 0},
                          {{1}, 3},
                          {{Id(1), 8, 99}},
                          DiskPayload{Id(2), 4, 100},
                          {}});
  m.components.push_back({{{2}, 5, ComponentKind::kPrivateState, 0, 0, 16},
                          {{2}, 3},
                          {},
                          {},
                          DiskPayload{Id(3), 16, 101}});
  return m;
}
void Write(const fs::path& p, std::span<const std::uint8_t> bytes) {
  std::ofstream f(p, std::ios::binary);
  f.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  assert(f.good());
}
void Payloads(const fs::path& root, const DiskManifest& m) {
  for (const auto& c : m.components) {
    for (auto p : c.chunks) {
      auto path = root / "v2/chunks" / DiskFileName(p.file);
      Write(path, {});
      fs::resize_file(path, p.bytes);
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
void Put(const fs::path& root, const DiskManifest& m, unsigned id = 7) {
  auto path = root / "v2/manifests" / DiskFileName(Id(id));
  Write(path, EncodeManifest(m));
}
template<typename F>
void Reject(F f) {
  bool failed = false;
  try {
    f();
  } catch (const std::exception&) {
    failed = true;
  }
  assert(failed);
}
void Format() {
  const std::string check = "123456789";
  auto bytes = std::span(reinterpret_cast<const std::uint8_t*>(check.data()),
                         check.size());
  assert(DiskChecksum(bytes) == 0x6c40df5f0b497347ULL);
  assert(DiskChecksum(bytes.subspan(4), DiskChecksum(bytes.first(4))) ==
         DiskChecksum(bytes));
  auto m = Manifest();
  auto encoded = EncodeManifest(m);
  auto decoded = DecodeManifest(encoded);
  assert(decoded.tokens == m.tokens && decoded.input == m.input);
  assert(decoded.compatibility == m.compatibility &&
         decoded.lineage == m.lineage);
  assert(decoded.components[0].chunks == m.components[0].chunks);
  assert(decoded.components[1].private_state == m.components[1].private_state);
  assert(EncodeManifest(decoded) == encoded);
  for (std::size_t i = 0; i < encoded.size(); ++i)
    Reject([&] { (void)DecodeManifest(std::span(encoded).first(i)); });
  for (std::size_t i = 0; i < encoded.size(); ++i) {
    auto corrupt = encoded;
    corrupt[i] ^= 1;
    Reject([&] { (void)DecodeManifest(corrupt); });
  }
  auto malformed = [&](std::size_t offset, std::uint8_t value) {
    auto changed = encoded;
    changed[offset] = value;
    auto checksum = DiskChecksum(std::span(changed).first(changed.size() - 8));
    for (unsigned i = 0; i < 8; ++i)
      changed[changed.size() - 8 + i] = (checksum >> (8 * i)) & 255;
    Reject([&] { (void)DecodeManifest(changed); });
  };
  malformed(8, 3);     // Unknown format with a valid checksum.
  malformed(60, 255);  // Invalid purpose.
  malformed(68, 255);  // Input length beyond the bounded remaining bytes.
  malformed(75, 255);  // Token count beyond the remaining bytes.
  m.components[0].position.valid_rows = 4;
  Reject([&] { (void)EncodeManifest(m); });
  m = Manifest();
  m.components[1].position.valid_rows = 2;
  Reject([&] { (void)EncodeManifest(m); });
  m = Manifest();
  m.components[0].descriptor.row_bytes = UINT64_MAX;
  Reject([&] { (void)EncodeManifest(m); });
  m = Manifest();
  auto& rows = m.components[0];
  rows.descriptor.row_bytes = 2;
  rows.descriptor.rows_per_chunk = 1ULL << 62;
  rows.position.valid_rows = 1ULL << 63;
  rows.chunks = {{Id(1), 1ULL << 63, 99}, {Id(2), 1ULL << 63, 100}};
  rows.tail.reset();
  // Each chunk size fits u64, but the complete component does not.
  Reject([&] { (void)EncodeManifest(m); });
}
void ChecksumReference() {
  std::vector<std::uint8_t> storage(65536 + 16);
  std::uint64_t random = 0x3fd52bb7c4186a95ULL;
  for (auto& byte : storage) {
    random ^= random << 13;
    random ^= random >> 7;
    random ^= random << 17;
    byte = random;
  }
  const auto reference = [](auto bytes, std::uint64_t crc) {
    for (const auto byte : bytes) {
      crc ^= std::uint64_t(byte) << 56;
      for (unsigned bit = 0; bit < 8; ++bit)
        crc = (crc << 1) ^ ((crc >> 63) ? 0x42f0e1eba9ea3693ULL : 0);
    }
    return crc;
  };
  for (std::size_t offset = 0; offset < 16; ++offset)
    for (const std::size_t length :
         {0,   1,   7,   8,   9,   15,   16,   17,   31,
          32,  33,  63,  64,  65,  127,  128,  129,  255,
          256, 257, 511, 512, 513, 4095, 4096, 4097, 65536}) {
      const auto bytes = std::span(storage).subspan(offset, length);
      for (const auto seed :
           std::array<std::uint64_t, 3>{0, UINT64_MAX, 0x1a3875eb2cf049d6ULL}) {
        const auto expected = reference(bytes, seed);
        assert(DiskChecksum(bytes, seed) == expected);
        for (const std::size_t split :
             {0, 1, 7, 8, 9, 63, 64, 65, 255, 256, 257}) {
          const auto first = std::min(length, split);
          assert(DiskChecksum(bytes.subspan(first),
                              DiskChecksum(bytes.first(first), seed)) ==
                 expected);
        }
      }
    }
}
void LargePieceReadAndCorruption() {
  Directory directory;
  DiskStore disk(ledger, directory.path, 100000);
  auto manifest = Manifest();
  std::array<std::uint8_t, 8> chunk{1, 2, 3};
  std::array<std::uint8_t, 4> tail{4, 5, 6};
  std::vector<std::uint8_t> state(1025);
  for (std::size_t i = 0; i < state.size(); ++i)
    state[i] = i * 131;
  manifest.components[0].chunks[0].checksum = DiskChecksum(chunk);
  manifest.components[0].tail->checksum = DiskChecksum(tail);
  auto& component = manifest.components[1];
  component.descriptor.state_bytes = state.size();
  component.private_state =
      DiskPayload{Id(3), state.size(), DiskChecksum(state)};
  const std::array<DiskWriteBuffer, 3> writes{
      {{false, Id(1), chunk}, {true, Id(2), tail}, {true, Id(3), state}}};
  disk.Publish(Id(7), manifest, writes);
  auto snapshot = disk.Open({7});
  assert(snapshot);
  const auto payload = *snapshot->Manifest().components[1].private_state;
  std::array<std::uint8_t, 513> staging{};
  std::size_t copied{};
  interrupt_next_read = true;
  shorten_reads = true;
  const auto read = disk.ReadPayload(
      *snapshot, true, payload, staging, [&](auto offset, auto bytes) {
        assert(offset == copied &&
               std::ranges::equal(
                   bytes, std::span(state).subspan(offset, bytes.size())));
        copied += bytes.size();
      });
  shorten_reads = false;
  assert(copied == state.size() && read.bytes == state.size() &&
         !read.verification_cached);
  disk.PrefetchPayload(*snapshot, true, payload, payload.bytes);
  {
    std::fstream corrupt(directory.path / "v2/private" / DiskFileName(Id(3)),
                         std::ios::in | std::ios::out | std::ios::binary);
    corrupt.seekp(513);
    corrupt.put('\xff');
  }
  bool invalid = false;
  try {
    (void)disk.ReadPayload(*snapshot, true, payload, staging,
                           [](auto, auto) {});
  } catch (const DiskDependencyError& error) {
    invalid = error.private_file && error.file == payload.file;
  }
  assert(invalid);
}
void Ownership() {
  Directory d;
  {
    DiskStore first(ledger, d.path, 100000);
    Reject([&] { DiskStore second(ledger, d.path, 100000); });
    pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
      try {
        DiskStore second(ledger, d.path, 100000);
      } catch (const std::exception& e) {
        _exit(std::string(e.what()).find(d.path.string()) != std::string::npos
                  ? 0
                  : 2);
      }
      _exit(1);
    }
    int status{};
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  }
  DiskStore reopened(ledger, d.path, 100000);
}
void Index() {
  Directory d;
  {
    DiskStore init(ledger, d.path, UINT64_MAX);
  }
  auto m = Manifest();
  Payloads(d.path, m);
  Put(d.path, m);
  auto encoded_size = EncodeManifest(m).size();
  actual_manifest_bytes = 0;
  actual_payload_bytes = 0;
  {
    DiskStore store(ledger, d.path, UINT64_MAX);
    assert(store.Entries().size() == 1);
    assert(store.Entries()[0].Durable() &&
           !store.Entries()[0].PayloadVerified());
    assert(store.Stats().manifest_bytes_read == encoded_size);
    assert(store.Stats().dependency_stats == 3);
    assert(actual_manifest_bytes == encoded_size && actual_payload_bytes == 0);
    assert(store.Stats().managed_bytes == encoded_size + 28);
  }
  // Sparse files scale declared payload by 1 GiB, without increasing metadata
  // or startup reads. Their contents deliberately fail the declared CRCs.
  m.components[1].descriptor.state_bytes = 1ULL << 30;
  m.components[1].private_state->bytes = 1ULL << 30;
  Payloads(d.path, m);
  Put(d.path, m);
  {
    DiskStore store(ledger, d.path, UINT64_MAX);
    assert(store.Entries().size() == 1);
    assert(store.Stats().manifest_bytes_read == encoded_size);
  }
  assert(actual_manifest_bytes == encoded_size * 2 &&
         actual_payload_bytes == 0);
  fs::resize_file(d.path / "v2/chunks" / DiskFileName(Id(1)), 7);
  {
    DiskStore store(ledger, d.path, UINT64_MAX);
    assert(store.Entries().empty() && store.Stats().rejected_manifests == 1);
  }
  Payloads(d.path, m);
  auto dependency = d.path / "v2/private" / DiskFileName(Id(3));
  fs::remove(dependency);
  fs::create_symlink("/dev/zero", dependency);
  {
    DiskStore store(ledger, d.path, UINT64_MAX);
    assert(store.Entries().empty());
  }
  fs::remove(dependency);
  Payloads(d.path, m);
  auto corrupt = EncodeManifest(m);
  corrupt[8] = 255;
  Write(d.path / "v2/manifests" / DiskFileName(Id(8)), corrupt);
  Write(d.path / "v2/manifests" / DiskFileName(Id(9)), {});
  fs::resize_file(d.path / "v2/manifests" / DiskFileName(Id(9)),
                  kMaxManifestBytes + 1);
  {
    DiskStore store(ledger, d.path, UINT64_MAX);
    assert(store.Entries().size() == 1 &&
           store.Stats().rejected_manifests == 2);
    assert(store.Stats().manifest_bytes_read == encoded_size * 2);
  }
}
void Conflicts() {
  Directory d;
  {
    DiskStore init(ledger, d.path, UINT64_MAX);
  }
  auto a = Manifest(), b = a;
  b.checkpoint = {8};
  b.components[0].tail->file = Id(12);
  b.components[1].private_state->file = Id(13);
  Payloads(d.path, a);
  Payloads(d.path, b);
  Put(d.path, a, 7);
  Put(d.path, b, 8);
  {
    DiskStore store(ledger, d.path, UINT64_MAX);
    assert(store.Entries().size() == 2);
  }
  b.components[0].chunks[0].checksum++;
  Put(d.path, b, 8);
  {
    DiskStore store(ledger, d.path, UINT64_MAX);
    assert(store.Entries().empty() && store.Stats().rejected_manifests == 2);
  }
  b = a;
  b.checkpoint = {8};
  Put(d.path, b, 8);
  {
    DiskStore store(ledger, d.path, UINT64_MAX);
    assert(store.Entries().empty());
  }
  fs::remove(d.path / "v2/manifests" / DiskFileName(Id(8)));
  Put(d.path, a, 9);
  {
    DiskStore store(ledger, d.path, UINT64_MAX);
    assert(store.Entries().empty());
  }
}
void Cleanup() {
  Directory d, unrelated;
  const std::array<std::uint8_t, 8> legacy{'G', 'U', 'F', 'O',
                                           'K', 'V', 'C', '1'};
  Write(d.path / "old.kvc", legacy);
  fs::resize_file(d.path / "old.kvc", 1000000);
  Write(d.path / ".tmp-owned", legacy);
  Write(d.path / "unknown.kvc", std::array<std::uint8_t, 4>{1, 2, 3, 4});
  Write(d.path / "unknown", legacy);
  fs::create_directory(d.path / ".tmp-directory");
  Write(d.path / ".tmp-directory/keep", legacy);
  Write(unrelated.path / "old.kvc", legacy);
  fs::create_directory_symlink(unrelated.path, d.path / "other");
  fs::create_symlink(unrelated.path / "old.kvc", d.path / "link.kvc");
  DiskStore store(ledger, d.path, 0);
  assert(!fs::exists(d.path / "old.kvc") && !fs::exists(d.path / ".tmp-owned"));
  assert(store.Stats().removed_legacy_bytes == 1000008);
  assert(store.Stats().managed_bytes == 0);
  for (const auto& p : {"unknown.kvc", "unknown", ".tmp-directory/keep",
                        "other/old.kvc", "link.kvc"})
    assert(fs::exists(d.path / p));
}
void InterruptedCleanup() {
  Directory d;
  const std::array<std::uint8_t, 4> legacy{'G', 'U', 'F', 'O'};
  Write(d.path / "old.kvc", legacy);
  fs::resize_file(d.path / "old.kvc", 1000000);
  interrupt_next_read = true;
  shorten_reads = true;
  {
    DiskStore store(ledger, d.path, 0);
    assert(!fs::exists(d.path / "old.kvc"));
    assert(store.Stats().removed_legacy_bytes == 1000000);
    assert(store.Stats().legacy_probe_bytes_read == 4);
  }
  shorten_reads = false;
}
void RootSymlinks() {
  Directory d, target;
  Write(target.path / "old.kvc",
        std::array<std::uint8_t, 4>{'G', 'U', 'F', 'O'});
  auto link = d.path / "alias";
  fs::create_directory_symlink(target.path, link);
  for (const auto& suffix : {"", "/", "/.", "/./"}) {
    Reject([&] { DiskStore store(ledger, link.string() + suffix, 0); });
    assert(fs::exists(target.path / "old.kvc"));
    assert(!fs::exists(target.path / "LOCK"));
  }
}
void ParentPathResolution() {
  Directory base, target;
  fs::create_directory(target.path / "sub");
  fs::create_directory_symlink(target.path / "sub", base.path / "alias");
  const std::array<std::uint8_t, 4> legacy{'G', 'U', 'F', 'O'};
  Write(base.path / "old.kvc", legacy);
  Write(target.path / "old.kvc", legacy);
  {
    DiskStore store(ledger, base.path / "alias/..", 0);
    assert(!fs::exists(target.path / "old.kvc"));
    assert(fs::exists(target.path / "LOCK"));
    assert(fs::exists(base.path / "old.kvc"));
    assert(!fs::exists(base.path / "LOCK"));
  }
}
void MetadataAdmission() {
  Directory d;
  {
    DiskStore init(ledger, d.path, UINT64_MAX);
  }
  auto m = Manifest();
  Payloads(d.path, m);
  Put(d.path, m);
  std::size_t peak{}, retained{};
  {
    DiskStore store(ledger, d.path, UINT64_MAX);
    auto snap = ledger.Snapshot();
    peak = snap.peak_ram_bytes;
    retained = snap.ram_bytes;
    assert(retained >= m.tokens.size() * sizeof(Token));
  }
  assert(ledger.Snapshot().total_bytes == 0);
  ResourceLedger exact{{peak, peak, 0, 0}};
  {
    DiskStore store(exact, d.path, UINT64_MAX);
    assert(store.Entries().size() == 1);
  }
  assert(exact.Snapshot().total_bytes == 0);
  // Discover the exact peak for this fixture rather than an earlier test.
  peak = exact.Snapshot().peak_ram_bytes;
  ResourceLedger short_budget{{peak - 1, peak - 1, 0, 0}};
  Reject([&] { DiskStore store(short_budget, d.path, UINT64_MAX); });
  assert(short_budget.Snapshot().total_bytes == 0);
  for (auto step : {LedgerStep::kReserve, LedgerStep::kConvert}) {
    bool succeeded = false;
    for (std::size_t n = 0; n < 32; ++n) {
      ResourceLedger faults{{1 << 20, 1 << 20, 0, 0}};
      faults.FailAfter(step, n);
      try {
        DiskStore store(faults, d.path, UINT64_MAX);
        assert(store.Entries().size() == 1);
        succeeded = true;
      } catch (const std::bad_alloc&) {
      }
      assert(faults.Snapshot().total_bytes == 0);
      // Failed constructors also relinquish ownership.
      {
        DiskStore reopened(ledger, d.path, UINT64_MAX);
      }
      if (succeeded)
        break;
    }
    assert(succeeded);
  }
  // Structurally rejected entries retain no decoded-vector charge.
  fs::resize_file(d.path / "v2/chunks" / DiskFileName(Id(1)), 7);
  {
    DiskStore rejected(ledger, d.path, UINT64_MAX);
    assert(rejected.Entries().empty());
    assert(ledger.Snapshot().ram_bytes < retained);
  }
  assert(ledger.Snapshot().total_bytes == 0);
}
void BudgetAndDirectories() {
  Directory d, unrelated;
  {
    DiskStore init(ledger, d.path, 1000);
  }
  auto m = Manifest();
  Payloads(d.path, m);
  Put(d.path, m);
  auto size = EncodeManifest(m).size() + 28;
  {
    DiskStore store(ledger, d.path, size);
    assert(store.Entries().size() == 1);
  }
  {
    // A lowered budget still opens the directory. Nothing publishes until
    // retirement brings it back within the budget.
    DiskStore store(ledger, d.path, size - 1);
    assert(store.Entries().size() == 1);
    assert(store.Stats().managed_bytes == size);
    auto next = m;
    next.checkpoint = {8};
    next.tokens = {17, 18};
    next.components[0].position.valid_rows = 2;
    next.components[1].position.valid_rows = 2;
    std::array<std::uint8_t, 8> chunk{1};
    std::array<std::uint8_t, 16> state{2};
    next.components[0].chunks[0] = {Id(11), chunk.size(), DiskChecksum(chunk)};
    next.components[0].tail.reset();
    next.components[1].private_state =
        DiskPayload{Id(13), state.size(), DiskChecksum(state)};
    const std::array<DiskWriteBuffer, 2> writes{
        {{false, Id(11), chunk}, {true, Id(13), state}}};
    bool exhausted = false;
    try {
      store.Publish(Id(18), next, writes);
    } catch (const DiskCapacityExhausted&) {
      exhausted = true;
    }
    assert(exhausted);
    assert(store.Retire({7}));
    assert(store.Stats().managed_bytes == 0);
    store.Publish(Id(18), next, writes);
    assert(store.Entries().size() == 1);
    assert(store.Stats().managed_bytes <= size - 1);
  }
  fs::remove_all(d.path / "v2/private");
  fs::create_directory_symlink(unrelated.path, d.path / "v2/private");
  Reject([&] { DiskStore store(ledger, d.path, UINT64_MAX); });
  assert(fs::is_empty(unrelated.path));
}
}  // namespace
int main() {
  Format();
  ChecksumReference();
  LargePieceReadAndCorruption();
  Ownership();
  Index();
  Conflicts();
  Cleanup();
  BudgetAndDirectories();
  InterruptedCleanup();
  RootSymlinks();
  ParentPathResolution();
  MetadataAdmission();
  assert(ledger.Snapshot().total_bytes == 0);
  std::cout << "disk format, ownership and metadata index passed\n";
}
