#include "src/cache/disk.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>

using namespace gufo::cache;
namespace fs = std::filesystem;
namespace {
std::atomic<std::uint64_t> actual_manifest_bytes{}, actual_payload_bytes{};
std::atomic<bool> interrupt_next_read{}, shorten_reads{};
thread_local std::int64_t fail_allocation_after{-1};
thread_local std::int64_t validation_failure_after{-1};
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
[[gnu::noinline]] void* operator new(std::size_t bytes) {
  if (fail_allocation_after >= 0 && fail_allocation_after-- == 0)
    throw std::bad_alloc();
  if (auto* memory = std::malloc(bytes ? bytes : 1))
    return memory;
  throw std::bad_alloc();
}
[[gnu::noinline]] void operator delete(void* memory) noexcept {
  std::free(memory);
}
[[gnu::noinline]] void operator delete(void* memory, std::size_t) noexcept {
  std::free(memory);
}
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
std::size_t ManifestStorage(const DiskManifest& m) {
  auto bytes = m.input.size() + m.tokens.size() * sizeof(Token) +
               m.components.size() * sizeof(DiskComponent);
  for (const auto& c : m.components)
    bytes += c.chunks.size() * sizeof(DiskPayload);
  return bytes;
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
    assert(init.OwnedManifestBytes() == 0);
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
    assert(store.Stats().dependency_references == 3);
    assert(store.Stats().distinct_dependency_files == 3);
    assert(store.Stats().dependency_stats == 3);
    assert(actual_manifest_bytes == encoded_size && actual_payload_bytes == 0);
    assert(store.Stats().managed_bytes == encoded_size + 28);
    assert(store.OwnedManifestBytes() == ManifestStorage(m));
    // Logical estimates and pending allocations do not establish owned bytes.
    const auto pending = ledger.Reserve(ResourceCategory::kMetadata, 8192);
    const auto estimated =
        ledger.Reserve(ResourceCategory::kMetadata, 16384).Convert();
    assert(store.OwnedManifestBytes() == ManifestStorage(m));
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
    // The retained entry-table capacity and its charge are not manifest data.
    assert(ledger.Snapshot().ram_bytes && store.OwnedManifestBytes() == 0);
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
void AllConflictOwners() {
  // Three owners share a claim, including two agreeing owners and an outlier.
  // Every side must be rejected, even if the outlier is visited first/last.
  for (unsigned conflict = 0; conflict < 4; ++conflict) {
    Directory d;
    {
      DiskStore init(ledger, d.path, UINT64_MAX);
    }
    std::array<DiskManifest, 4> manifests;
    for (unsigned i = 0; i < manifests.size(); ++i) {
      auto& m = manifests[i];
      m = Manifest();
      m.checkpoint = {7 + i};
      m.components[0].tail->file = Id(20 + i);
      m.components[1].private_state->file = Id(30 + i);
      if (i == 3 || conflict >= 2)
        m.components[0].chunks[0].file = Id(40 + i);
    }
    if (conflict == 0)
      ++manifests[2].components[0].chunks[0].checksum;
    if (conflict == 1)
      ++manifests[2].compatibility[0];
    if (conflict == 2)
      for (unsigned i = 0; i < 3; ++i)
        manifests[i].components[1].private_state->file = Id(30);
    if (conflict == 3)
      for (unsigned i = 0; i < 3; ++i)
        manifests[i].checkpoint = {7};
    for (const auto& m : manifests)
      Payloads(d.path, m);
    std::array<unsigned, 4> order{0, 1, 2, 3};
    do {
      // Reassign contents to filenames, exercising every enumeration order
      // without relying on a particular filesystem's directory ordering.
      for (unsigned i = 0; i < order.size(); ++i)
        Put(d.path, manifests[order[i]], 100 + i);
      actual_manifest_bytes = actual_payload_bytes = 0;
      {
        DiskStore store(ledger, d.path, UINT64_MAX);
        assert(store.Entries().size() == 1);
        assert(store.Entries().front().Manifest().checkpoint.value == 10);
        assert(store.Stats().rejected_manifests == 3);
        assert(actual_manifest_bytes && actual_payload_bytes == 0);
      }
      assert(ledger.Snapshot().total_bytes == 0);
    } while (std::next_permutation(order.begin(), order.end()));
  }
}
void MissingDependenciesAndRestart() {
  Directory d;
  auto m = Manifest();
  {
    DiskStore init(ledger, d.path, UINT64_MAX);
  }
  Put(d.path, m);
  for (unsigned failure = 0; failure < 3; ++failure) {
    Payloads(d.path, m);
    if (failure == 0)
      fs::remove(d.path / "v2/chunks" / DiskFileName(Id(1)));
    if (failure == 1)
      fs::remove(d.path / "v2/private" / DiskFileName(Id(3)));
    if (failure == 2)
      fs::resize_file(d.path / "v2/private" / DiskFileName(Id(3)), 15);
    {
      DiskStore rejected(ledger, d.path, UINT64_MAX);
      assert(rejected.Entries().empty());
      assert(rejected.Stats().rejected_manifests == 1);
    }
    Payloads(d.path, m);
    DiskStore restored(ledger, d.path, UINT64_MAX);
    assert(restored.Entries().size() == 1);
  }
}
void SharedDependencyRecovery() {
  Directory d;
  {
    DiskStore init(ledger, d.path, UINT64_MAX);
  }
  const std::array<std::uint8_t, 8> bytes{1, 2, 3, 4, 5, 6, 7, 8};
  std::array<DiskManifest, 3> owners;
  for (unsigned i = 0; i < owners.size(); ++i) {
    auto& m = owners[i];
    m = Manifest();
    m.checkpoint = {7 + i};
    m.components[0].chunks[0].checksum = DiskChecksum(bytes);
    m.components[0].tail->file = Id(10 + i);
    // Equal IDs in the two namespaces name different dependencies, whose
    // sizes also differ. A filename-only observation cache is incorrect.
    m.components[1].private_state->file = Id(i ? 20 + i : 1);
    Payloads(d.path, m);
    Put(d.path, m, 100 + i);
  }
  const auto shared = d.path / "v2/chunks" / DiskFileName(Id(1));
  const auto alias = d.path / "alias";
  Write(shared, bytes);
  const auto counts = [](const DiskStore& store) {
    const auto stats = store.Stats();
    assert(stats.dependency_references == 9);
    assert(stats.distinct_dependency_files == 7);
    assert(stats.dependency_stats == 7);
  };
  for (unsigned failure = 0; failure < 5; ++failure) {
    // Each recovery gets fresh metadata; observations must not persist across
    // stores. Every owner of the unavailable shared dependency is rejected.
    if (failure == 0)
      fs::remove(shared);
    if (failure == 1)
      fs::resize_file(shared, bytes.size() - 1);
    if (failure == 2) {
      fs::remove(shared);
      Write(alias, bytes);
      fs::create_symlink(alias, shared);
    }
    if (failure == 3)
      fs::create_hard_link(shared, alias);
    if (failure == 4) {
      fs::remove(shared);
      fs::create_directory(shared);
    }
    actual_manifest_bytes = actual_payload_bytes = 0;
    {
      DiskStore rejected(ledger, d.path, UINT64_MAX);
      counts(rejected);
      assert(rejected.Entries().empty());
      assert(rejected.Stats().rejected_manifests == owners.size());
      assert(actual_manifest_bytes && actual_payload_bytes == 0);
    }
    fs::remove(shared);
    fs::remove(alias);
    Write(shared, bytes);
    actual_manifest_bytes = actual_payload_bytes = 0;
    {
      DiskStore restored(ledger, d.path, UINT64_MAX);
      counts(restored);
      assert(restored.Entries().size() == owners.size());
      assert(restored.Stats().rejected_manifests == 0);
      assert(actual_manifest_bytes && actual_payload_bytes == 0);
      const auto snapshot = restored.Open({7});
      assert(snapshot);
      std::array<std::uint8_t, 3> staging{};
      std::size_t copied{};
      const auto read = restored.ReadPayload(
          *snapshot, false, owners[0].components[0].chunks[0], staging,
          [&](auto offset, auto piece) {
            assert(offset == copied &&
                   std::ranges::equal(
                       piece, std::span(bytes).subspan(offset, piece.size())));
            copied += piece.size();
          });
      assert(copied == bytes.size() && read.bytes == bytes.size() &&
             !read.verification_cached);
    }
    assert(ledger.Snapshot().total_bytes == 0);
  }
  // Startup metadata reuse never attests payload contents. A same-size
  // mutation after recovery must still fail the first restore checksum.
  {
    DiskStore store(ledger, d.path, UINT64_MAX);
    const auto snapshot = store.Open({7});
    assert(snapshot);
    auto corrupt = bytes;
    corrupt[0] ^= 1;
    Write(shared, corrupt);
    std::array<std::uint8_t, 3> staging{};
    bool invalid = false;
    try {
      (void)store.ReadPayload(*snapshot, false,
                              owners[0].components[0].chunks[0], staging,
                              [](auto, auto) {});
    } catch (const DiskDependencyError& error) {
      invalid = !error.private_file && error.file == Id(1);
    }
    assert(invalid);
  }
  assert(ledger.Snapshot().total_bytes == 0);
}
void UnavailableOwnersDoNotConflict() {
  // Availability filtering precedes both canonical-reference validation and
  // duplicate checkpoint IDs. An unavailable owner's claims cannot poison a
  // compatible available owner, regardless of filesystem enumeration order.
  for (unsigned failure = 0; failure < 2; ++failure) {
    Directory d;
    {
      DiskStore init(ledger, d.path, UINT64_MAX);
    }
    auto good = Manifest(), bad = good;
    bad.components[0].tail->file = Id(12);
    bad.components[1].private_state->file = Id(13);
    ++bad.components[0].chunks[0].checksum;
    if (failure == 0) {
      // Both references are individually well formed, but only the good
      // owner's size matches the observed file. Cache actual size, not the
      // outcome of checking the first reference encountered.
      bad.components[0].descriptor.row_bytes *= 2;
      bad.components[0].chunks[0].bytes *= 2;
      bad.components[0].tail->bytes *= 2;
    }
    Payloads(d.path, bad);
    Payloads(d.path, good);
    if (failure == 1)
      fs::remove(d.path / "v2/private" / DiskFileName(Id(13)));
    for (bool reverse : {false, true}) {
      Put(d.path, reverse ? bad : good, 100);
      Put(d.path, reverse ? good : bad, 101);
      actual_manifest_bytes = actual_payload_bytes = 0;
      {
        DiskStore store(ledger, d.path, UINT64_MAX);
        const auto stats = store.Stats();
        assert(stats.dependency_references == 6);
        assert(stats.distinct_dependency_files == 5);
        assert(stats.dependency_stats == 5);
        assert(stats.rejected_manifests == 1);
        assert(store.Entries().size() == 1);
        assert(EncodeManifest(store.Entries()[0].Manifest()) ==
               EncodeManifest(good));
        assert(actual_manifest_bytes && actual_payload_bytes == 0);
      }
      assert(ledger.Snapshot().total_bytes == 0);
    }
  }
}
void RootSymlinks() {
  Directory d, target;
  Write(target.path / "file", std::array<std::uint8_t, 4>{1, 2, 3, 4});
  auto link = d.path / "alias";
  fs::create_directory_symlink(target.path, link);
  for (const auto& suffix : {"", "/", "/.", "/./"}) {
    Reject([&] { DiskStore store(ledger, link.string() + suffix, 0); });
    assert(fs::exists(target.path / "file"));
    assert(!fs::exists(target.path / "LOCK"));
  }
}
void ParentPathResolution() {
  Directory base, target;
  fs::create_directory(target.path / "sub");
  fs::create_directory_symlink(target.path / "sub", base.path / "alias");
  {
    DiskStore store(ledger, base.path / "alias/..", 0);
    assert(fs::exists(target.path / "LOCK"));
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
  // Walk actual recovery-validation allocations, including ledger tokens,
  // nodes, arrays and dependency StatAt pathname storage. StatAt reports OOM
  // through its no-throw boundary; recovery must abort without rejecting valid
  // owners. Other startup filesystem wrappers remain outside this scope.
  // Whole-construction ledger faults remain covered above.
  DiskStore::SetReferenceValidationHook([](bool entering) noexcept {
    fail_allocation_after = entering ? validation_failure_after : -1;
  });
  bool allocated = false;
  for (std::int64_t n = 0; n < 512; ++n) {
    ResourceLedger faults{{1 << 20, 1 << 20, 0, 0}};
    validation_failure_after = n;
    try {
      DiskStore store(faults, d.path, UINT64_MAX);
      assert(store.Entries().size() == 1);
      allocated = true;
    } catch (const std::bad_alloc&) {
    }
    assert(fail_allocation_after == -1);
    validation_failure_after = -1;
    assert(faults.Snapshot().total_bytes == 0);
    {
      DiskStore reopened(ledger, d.path, UINT64_MAX);
      assert(reopened.Entries().size() == 1);
    }
    if (allocated)
      break;
  }
  DiskStore::SetReferenceValidationHook(nullptr);
  assert(allocated);
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
void ManagedUsageRequiresPayloadMetadata() {
  const std::array<std::uint8_t, 8> chunk{1, 2, 3, 4};
  const std::array<std::uint8_t, 4> tail{5, 6};
  const std::array<std::uint8_t, 16> state{7, 8, 9};
  auto first = Manifest();
  first.components[0].chunks[0].checksum = DiskChecksum(chunk);
  first.components[0].tail->checksum = DiskChecksum(tail);
  first.components[1].private_state->checksum = DiskChecksum(state);
  const std::array<DiskWriteBuffer, 3> initial{
      {{false, Id(1), chunk}, {true, Id(2), tail}, {true, Id(3), state}}};
  const auto first_bytes =
      EncodeManifest(first).size() + chunk.size() + tail.size() + state.size();
  {
    Directory directory;
    auto next = first;
    next.checkpoint = {8};
    next.components[0].tail->file = Id(12);
    next.components[1].private_state->file = Id(13);
    const std::array<DiskWriteBuffer, 2> additional{
        {{true, Id(12), tail}, {true, Id(13), state}}};
    // Exactly enough for both manifests and their distinct referenced files.
    // A manifest-only sum cannot account for unrelated orphan/temp storage.
    const auto budget =
        first_bytes + EncodeManifest(next).size() + tail.size() + state.size();
    {
      DiskStore store(ledger, directory.path, budget);
      store.Publish(Id(7), first, initial);
      assert(store.Stats().managed_bytes == first_bytes);
    }
    constexpr std::size_t kOrphanBytes = 64 << 10, kTemporaryBytes = 17;
    const auto orphan = directory.path / "v2/chunks" / DiskFileName(Id(90));
    const auto temporary = directory.path / "v2/tmp" / DiskFileName(Id(91));
    Write(orphan, {});
    fs::resize_file(orphan, kOrphanBytes);
    Write(temporary, {});
    fs::resize_file(temporary, kTemporaryBytes);
    actual_manifest_bytes = actual_payload_bytes = 0;
    {
      DiskStore store(ledger, directory.path, budget);
      assert(store.Entries().size() == 1);
      assert(store.Stats().managed_bytes ==
             first_bytes + kOrphanBytes + kTemporaryBytes);
      assert(store.Stats().dependency_stats == 3);
      assert(actual_manifest_bytes == EncodeManifest(first).size());
      assert(actual_payload_bytes == 0);
      bool exhausted = false;
      try {
        store.Publish(Id(8), next, additional);
      } catch (const DiskCapacityExhausted&) {
        exhausted = true;
      }
      assert(exhausted && actual_payload_bytes == 0);
      assert(store.Entries().size() == 1 && !store.Open({8}));
      store.ReclaimOrphans();
      store.WaitForReclamation();
      assert(!fs::exists(orphan) && !fs::exists(temporary));
      assert(store.Stats().managed_bytes == first_bytes);
      store.Publish(Id(8), next, additional);
      assert(store.Stats().managed_bytes == budget);
      assert(store.Entries().size() == 2);
    }
    {
      DiskStore reopened(ledger, directory.path, budget);
      assert(reopened.Stats().managed_bytes == budget);
      assert(reopened.Entries().size() == 2);
      const auto snapshot = reopened.Open(next.checkpoint);
      assert(snapshot);
      const auto read = [&](bool private_file, const DiskPayload& payload,
                            std::span<const std::uint8_t> expected) {
        std::array<std::uint8_t, 5> staging{};
        std::size_t copied{};
        const auto result = reopened.ReadPayload(
            *snapshot, private_file, payload, staging,
            [&](auto offset, auto bytes) {
              assert(offset == copied &&
                     std::ranges::equal(
                         bytes, expected.subspan(offset, bytes.size())));
              copied += bytes.size();
            });
        assert(copied == expected.size() && result.bytes == expected.size());
      };
      read(false, next.components[0].chunks[0], chunk);
      read(true, *next.components[0].tail, tail);
      read(true, *next.components[1].private_state, state);
    }
  }
  assert(ledger.Snapshot().total_bytes == 0);
  {
    Directory directory;
    {
      DiskStore store(ledger, directory.path, UINT64_MAX);
      store.Publish(Id(7), first, initial);
    }
    const auto chunks = directory.path / "v2/chunks";
    const auto directory_mtime = fs::last_write_time(chunks);
    // No namespace change: a persisted directory timestamp or manifest size
    // sum cannot attest the current managed bytes of an existing payload.
    constexpr std::size_t kGrownBytes = 64 << 10;
    fs::resize_file(chunks / DiskFileName(Id(1)), kGrownBytes);
    assert(fs::last_write_time(chunks) == directory_mtime);
    actual_manifest_bytes = actual_payload_bytes = 0;
    DiskStore recovered(ledger, directory.path, UINT64_MAX);
    assert(recovered.Stats().managed_bytes ==
           first_bytes - chunk.size() + kGrownBytes);
    assert(recovered.Stats().dependency_stats == 3);
    assert(recovered.Stats().rejected_manifests == 1);
    assert(recovered.Entries().empty() && !recovered.Open(first.checkpoint));
    assert(actual_manifest_bytes == EncodeManifest(first).size());
    assert(actual_payload_bytes == 0);
  }
  assert(ledger.Snapshot().total_bytes == 0);
}
void CapacityBeforePayloadReads() {
  Directory directory;
  auto first = Manifest();
  first.tokens.resize(2);
  auto& rows = first.components[0];
  rows.descriptor.row_bytes = 128 << 10;
  rows.position.valid_rows = 2;
  rows.tail.reset();
  std::vector<std::uint8_t> chunk(256 << 10, 42);
  std::array<std::uint8_t, 16> state{9};
  rows.chunks[0] = {Id(1), chunk.size(), DiskChecksum(chunk)};
  first.components[1].position.valid_rows = 2;
  first.components[1].private_state =
      DiskPayload{Id(3), state.size(), DiskChecksum(state)};
  const std::array<DiskWriteBuffer, 2> buffers{
      {{false, Id(1), chunk}, {true, Id(3), state}}};
  const auto capacity =
      EncodeManifest(first).size() + chunk.size() + state.size();
  auto next = first;
  next.checkpoint = {8};
  // Streamed jobs resolve shared checksums from the store's attestation.
  next.components[0].chunks[0].checksum = 0;
  next.components[1].private_state->file = Id(13);
  next.components[1].private_state->checksum = 0;
  std::size_t copied = 0;
  const std::array<DiskWriteSource, 2> sources{{
      {false, Id(1), chunk.size(),
       [&](auto offset, auto destination) {
         std::copy_n(chunk.begin() + offset, destination.size(),
                     destination.begin());
         copied += destination.size();
       }},
      {true, Id(13), state.size(),
       [&](auto offset, auto destination) {
         std::copy_n(state.begin() + offset, destination.size(),
                     destination.begin());
         copied += destination.size();
       }},
  }};
  std::array<std::uint8_t, 4096> staging{};
  for (const auto limit : {capacity, capacity + EncodeManifest(next).size()}) {
    DiskStore store(ledger, directory.path, limit);
    if (store.Entries().empty())
      store.Publish(Id(7), first, buffers);
    actual_payload_bytes = 0;
    // Eviction can retry the same checkpoint several times. None of those
    // refusals should read shared files or copy new payloads.
    for (int retry = 0; retry < 3; ++retry) {
      bool exhausted = false;
      try {
        (void)store.PublishStream(Id(8), next, sources, staging);
      } catch (const DiskCapacityExhausted&) {
        exhausted = true;
      }
      assert(exhausted && store.Stats().managed_bytes == capacity);
    }
    std::cout << "capacity refusals read " << actual_payload_bytes.load()
              << " payload bytes\n"
              << std::flush;
    assert(actual_payload_bytes == 0 && copied == 0);
  }
  {
    // The entire payload would exceed this budget; its shared chunk already
    // exists, so only the new manifest and private state need admission.
    DiskStore store(ledger, directory.path,
                    capacity + EncodeManifest(next).size() + state.size());
    actual_payload_bytes = 0;
    (void)store.PublishStream(Id(8), next, sources, staging);
    assert(store.Open({7}) && store.Open({8}));
    assert(actual_payload_bytes == chunk.size() && copied == state.size());
  }
  {
    DiskStore store(ledger, directory.path, 2 * capacity);
    // Admitting the write must still verify every reused byte, including
    // corruption in a file already referenced by multiple checkpoints.
    chunk[0] ^= 1;
    Write(directory.path / "v2/chunks" / DiskFileName(Id(1)), chunk);
    auto corrupted = next;
    corrupted.checkpoint = {9};
    bool rejected = false;
    try {
      (void)store.PublishStream(Id(9), corrupted, sources, staging);
    } catch (const DiskDependencyError&) {
      rejected = true;
    }
    assert(rejected && !store.Open({7}) && !store.Open({8}));
  }
}
void VerifiedDependencies() {
  Directory directory;
  DiskStore store(ledger, directory.path, 1ULL << 30);
  std::vector<std::uint8_t> chunk(8, 42);
  std::array<std::uint8_t, 16> state{};
  unsigned shared = 1;
  auto publish = [&](unsigned checkpoint) {
    auto m = Manifest();
    m.checkpoint = {checkpoint};
    m.tokens.resize(2);
    m.components[0].chunks[0] = {Id(shared), chunk.size(), DiskChecksum(chunk)};
    m.components[0].tail.reset();
    m.components[0].position.valid_rows = 2;
    m.components[1].position.valid_rows = 2;
    state[0] = checkpoint;
    m.components[1].private_state =
        DiskPayload{Id(checkpoint + 100), state.size(), DiskChecksum(state)};
    const std::array<DiskWriteBuffer, 2> buffers{
        {{false, Id(shared), chunk}, {true, Id(checkpoint + 100), state}}};
    actual_payload_bytes = 0;
    store.Publish(Id(checkpoint), m,
                  std::span(buffers).subspan(store.Entries().empty() ? 0 : 1));
    return actual_payload_bytes.load();
  };
  auto retire = [&] {
    for (unsigned id : {7, 8, 9})
      (void)store.Retire({id});
    store.ReclaimOrphans();
    store.WaitForReclamation();
    assert(store.Entries().empty());
    return ledger.Snapshot().total_bytes;
  };
  // Inherited chunks are read once after they were written, then their
  // unchanged identity is trusted by later publications.
  assert(publish(7) == 0);
  assert(publish(8) == chunk.size());
  assert(publish(9) == 0);
  const auto settled = retire();
  // Removing a file releases its verification record.
  shared = 2;
  assert(publish(7) == 0);
  assert(publish(8) == chunk.size());
  // A rewritten file changes identity and is verified again.
  chunk[0] ^= 1;
  Write(directory.path / "v2/chunks" / DiskFileName(Id(shared)), chunk);
  chunk[0] ^= 1;
  bool rejected = false;
  try {
    (void)publish(9);
  } catch (const DiskDependencyError&) {
    rejected = true;
  }
  assert(rejected);
  assert(retire() == settled);
}
}  // namespace
// A window manifest lists the chunks from its window's first row, uses the
// window format version and size, and shares chunk files with other
// manifests at different list positions. Window 3 rows, chunks of 2.
DiskManifest Windowed(Rows rows, std::vector<unsigned> chunks, unsigned tail,
                      unsigned checkpoint) {
  DiskManifest m;
  m.checkpoint = {checkpoint};
  m.lineage = {3};
  m.compatibility.fill(0x42);
  m.tokens.assign(rows, 17);
  DiskComponent window{{{1}, 4, ComponentKind::kWindowRows, 4, 2, 0, 3},
                       {{1}, rows},
                       {},
                       {},
                       {}};
  for (auto id : chunks)
    window.chunks.push_back({Id(id), 8, 90 + id});
  if (tail)
    window.tail = DiskPayload{Id(tail), 4, 90 + tail};
  m.components.push_back(window);
  m.components.push_back({{{2}, 5, ComponentKind::kPrivateState, 0, 0, 16},
                          {{2}, rows},
                          {},
                          {},
                          DiskPayload{Id(100 + checkpoint), 16, 101}});
  return m;
}
void WindowFormat() {
  const auto eight = Windowed(8, {1, 2}, 0, 7);  // Chunks [4, 6) and [6, 8).
  auto encoded = EncodeManifest(eight);
  assert(encoded[8] == kDiskWindowFormatVersion);
  auto decoded = DecodeManifest(encoded);
  assert(decoded.components[0].descriptor.window_rows == 3);
  assert(decoded.components[0].chunks == eight.components[0].chunks);
  assert(EncodeManifest(decoded) == encoded);
  for (std::size_t i = 0; i < encoded.size(); ++i)
    Reject([&] { (void)DecodeManifest(std::span(encoded).first(i)); });
  // Window components need the window format version.
  auto old = encoded;
  old[8] = kDiskFormatVersion;
  const auto checksum = DiskChecksum(std::span(old).first(old.size() - 8));
  for (unsigned i = 0; i < 8; ++i)
    old[old.size() - 8 + i] = (checksum >> (8 * i)) & 255;
  Reject([&] { (void)DecodeManifest(old); });
  // Chunks start at the window's first chunk, never at row 0.
  Reject([&] { (void)EncodeManifest(Windowed(8, {1}, 0, 7)); });
  Reject([&] { (void)EncodeManifest(Windowed(8, {9, 1, 2}, 0, 7)); });
  auto unsized = eight;
  unsized.components[0].descriptor.window_rows = 0;
  Reject([&] { (void)EncodeManifest(unsized); });
  // Chunk [6, 8) is second in one manifest and first in the other.
  Directory d;
  {
    DiskStore init(ledger, d.path, UINT64_MAX);
  }
  const auto nine = Windowed(9, {2}, 3, 8);  // Chunk [6, 8) and row 8.
  for (const auto* m : {&eight, &nine})
    Payloads(d.path, *m);
  Put(d.path, eight, 7);
  Put(d.path, nine, 8);
  {
    DiskStore store(ledger, d.path, UINT64_MAX);
    assert(store.Entries().size() == 2);
    assert(store.Stats().rejected_manifests == 0);
  }
}
// Publication validates its candidate against stored checkpoints. That check
// must not need RAM-ledger headroom proportional to every reference on disk:
// the RAM cache fills the same ledger, and a refusal there stopped all disk
// publication permanently once the tier had grown (#1030).
void ValidationIndependentOfStoredReferences() {
  Directory d;
  {
    DiskStore init(ledger, d.path, UINT64_MAX);
  }
  constexpr unsigned kStored = 2, kRows = 16384;
  const std::array<std::uint8_t, 1> byte{0};
  const auto checksum = DiskChecksum(byte);
  const auto file = [](unsigned manifest, unsigned row) {
    auto id = Id(row);
    id[2] = manifest + 1;
    return id;
  };
  const auto rows = [&](unsigned manifest, unsigned count) {
    DiskManifest m;
    m.checkpoint = {100 + manifest};
    m.lineage = {3};
    m.compatibility.fill(0x42);
    m.tokens = {17, 18, 19};
    DiskComponent c{{{1}, 4, ComponentKind::kAppendRows, 1, 1, 0},
                    {{1}, count},
                    {},
                    {},
                    {}};
    for (unsigned row = 0; row < count; ++row)
      c.chunks.push_back({file(manifest, row), 1, checksum});
    m.components.push_back(std::move(c));
    return m;
  };
  for (unsigned i = 0; i < kStored; ++i) {
    auto m = rows(i, kRows);
    Payloads(d.path, m);
    Put(d.path, m, 1000 + i);
  }
  // Ample for publishing one small checkpoint, far below a catalog of all
  // kStored * kRows stored references.
  constexpr std::size_t kHeadroom = 1 << 20;
  ResourceLedger full{{1ULL << 32, 1ULL << 32, 0, 0}};
  {
    DiskStore store(full, d.path, UINT64_MAX);
    assert(store.Entries().size() == kStored);
    const auto used = full.Snapshot().ram_bytes;
    // Stand-in for a RAM cache that already occupies the rest of the budget.
    auto cache = full.Reserve(ResourceCategory::kMetadata,
                              (1ULL << 32) - used - kHeadroom);
    auto candidate = rows(kStored, 1);
    const std::array<DiskWriteBuffer, 1> buffers{
        {{false, file(kStored, 0), byte}}};
    store.Publish(Id(2000), candidate, buffers);
    assert(store.Open(candidate.checkpoint));
    // The candidate is still checked against every stored checkpoint.
    auto conflict = rows(kStored + 1, 1);
    // Same bytes, so verification passes; a different lineage cannot share.
    conflict.lineage = {4};
    conflict.components[0].chunks[0].file = file(0, 0);
    bool invalid = false;
    try {
      store.Publish(Id(2001), conflict, {});
    } catch (const std::invalid_argument&) {
      invalid = true;
    }
    assert(invalid && !store.Open(conflict.checkpoint));
    auto duplicate = rows(kStored + 2, 1);
    duplicate.checkpoint = {100};
    duplicate.components[0].chunks[0].file = file(0, 0);
    invalid = false;
    try {
      store.Publish(Id(2002), duplicate, {});
    } catch (const std::invalid_argument&) {
      invalid = true;
    }
    assert(invalid);
  }
  assert(full.Snapshot().total_bytes == 0);
}
int main() {
  Format();
  ChecksumReference();
  LargePieceReadAndCorruption();
  Ownership();
  Index();
  Conflicts();
  AllConflictOwners();
  MissingDependenciesAndRestart();
  SharedDependencyRecovery();
  UnavailableOwnersDoNotConflict();
  BudgetAndDirectories();
  ManagedUsageRequiresPayloadMetadata();
  CapacityBeforePayloadReads();
  VerifiedDependencies();
  RootSymlinks();
  ParentPathResolution();
  MetadataAdmission();
  ValidationIndependentOfStoredReferences();
  WindowFormat();
  assert(ledger.Snapshot().total_bytes == 0);
  std::cout << "disk format, ownership and metadata index passed\n";
}
