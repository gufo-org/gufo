#include "src/core/gguf_identity.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_identity_internal.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/platform/env.hpp"
#include "src/core/platform/fs.hpp"

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

constexpr std::size_t kWindow = 4096;

/// Minimal GGUF v3 image: one string metadata entry and two tensors whose
/// payloads span `first_bytes` and `second_bytes`.
std::vector<std::uint8_t> BuildImage(std::string_view name_value,
                                     std::string_view first_tensor_name,
                                     std::size_t first_bytes,
                                     std::size_t second_bytes) {
  std::vector<std::uint8_t> buffer;
  auto append = [&buffer](const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    buffer.insert(buffer.end(), bytes, bytes + size);
  };
  auto pod = [&append](auto value) { append(&value, sizeof(value)); };
  auto string = [&](std::string_view value) {
    pod(static_cast<std::uint64_t>(value.size()));
    append(value.data(), value.size());
  };

  append("GGUF", 4);
  pod(static_cast<std::uint32_t>(3));
  pod(static_cast<std::uint64_t>(2));  // tensor count
  pod(static_cast<std::uint64_t>(1));  // metadata count
  string("general.name");
  pod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kString));
  string(name_value);

  string(first_tensor_name);
  pod(static_cast<std::uint32_t>(1));
  pod(static_cast<std::uint64_t>(first_bytes / 2));
  pod(static_cast<std::uint32_t>(gufo::core::GgmlType::kF16));
  pod(static_cast<std::uint64_t>(0));
  string("second");
  pod(static_cast<std::uint32_t>(1));
  pod(static_cast<std::uint64_t>(second_bytes / 2));
  pod(static_cast<std::uint32_t>(gufo::core::GgmlType::kF16));
  pod(static_cast<std::uint64_t>(first_bytes));

  while (buffer.size() % 32 != 0) {
    buffer.push_back(0);
  }
  const std::size_t payload_base = buffer.size();
  buffer.resize(payload_base + first_bytes + second_bytes);
  for (std::size_t index = 0; index < first_bytes + second_bytes; ++index) {
    buffer[payload_base + index] = static_cast<std::uint8_t>(index * 7U + 3U);
  }
  return buffer;
}

std::size_t PayloadBase(const std::vector<std::uint8_t>& image) {
  std::string error;
  const auto reader =
      gufo::core::GgufReader::OpenMemory(image.data(), image.size(), &error);
  Expect(reader != nullptr, "open image: " + error);
  return static_cast<std::size_t>(
      static_cast<const std::uint8_t*>(reader->GetTensors()[0].data) -
      image.data());
}

std::string Identity(const std::vector<std::uint8_t>& image) {
  std::string error;
  const auto reader =
      gufo::core::GgufReader::OpenMemory(image.data(), image.size(), &error);
  Expect(reader != nullptr, "open image: " + error);
  return gufo::core::GgufIdentityHex(*reader);
}

void TestDeterministic() {
  const auto image = BuildImage("model", "first", 64 * kWindow, 2 * kWindow);
  const std::string first = Identity(image);
  Expect(first.size() == 64, "identity is 64 hex characters");
  Expect(first == Identity(image), "identity is deterministic");
  Expect(first ==
             Identity(BuildImage("model", "first", 64 * kWindow, 2 * kWindow)),
         "identity depends on bytes, not on the buffer instance");
}

void TestHeaderChangesIdentity() {
  const auto base = BuildImage("model", "first", 64 * kWindow, 2 * kWindow);
  Expect(Identity(base) != Identity(BuildImage("model-v2", "first",
                                               64 * kWindow, 2 * kWindow)),
         "metadata change alters identity");
  Expect(Identity(base) != Identity(BuildImage("model", "renamed", 64 * kWindow,
                                               2 * kWindow)),
         "tensor name change alters identity");
  Expect(Identity(base) !=
             Identity(BuildImage("model", "first", 64 * kWindow, 4 * kWindow)),
         "tensor extent change alters identity");
}

void TestEveryPayloadByte() {
  const std::size_t first_bytes = 64 * kWindow;
  auto image = BuildImage("model", "first", first_bytes, 2 * kWindow);
  const std::string base = Identity(image);
  const std::size_t payload = PayloadBase(image);

  auto flipped = [&](std::size_t offset) {
    auto copy = image;
    copy[payload + offset] ^= 0xFFU;
    return Identity(copy);
  };

  Expect(flipped(0) != base, "start bytes are hashed");
  Expect(flipped(first_bytes - 1) != base, "end bytes are hashed");
  const std::size_t middle = ((first_bytes - kWindow) / 2) & ~(kWindow - 1);
  Expect(flipped(middle + 17) != base, "middle bytes are hashed");
  Expect(flipped(kWindow + 100) != base,
         "bytes outside the old sample windows alter identity");

  // The short second tensor is hashed whole.
  Expect(flipped(first_bytes + kWindow + 5) != base,
         "small tensors are hashed in full");
}

void TestFileDigestCache() {
  char temp[] = "/tmp/gufo-identity-XXXXXX";
  Expect(mkdtemp(temp) != nullptr, "create cache test directory");
  const std::filesystem::path root(temp);
  const char* old_cache = std::getenv("XDG_CACHE_HOME");
  const std::string saved_cache = old_cache ? old_cache : "";
  const bool had_cache = old_cache != nullptr;
  gufo::platform::SetEnv("XDG_CACHE_HOME", temp);
  auto bytes = BuildImage("model", "first", 64 * kWindow, 2 * kWindow);
  const auto path = root / "model.gguf";
  auto write = [&] {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    Expect(out.good(), "write GGUF fixture");
  };
  write();
  auto read = [&] {
    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(path.string(), &error);
    Expect(reader != nullptr, "open file: " + error);
    return gufo::core::GgufIdentityHex(*reader);
  };
  const auto first = read();
  Expect(first == Identity(bytes), "file and memory full digests agree");
  Expect(read() == first, "cached digest agrees with initial scan");
  const auto cache = root / "gufo" / "gguf-sha256-v1";
  // A just-written file may still be inside the timestamp ambiguity window.
  // Publication and actual cache hits are checked with controlled clocks below;
  // this real-filesystem regression must keep checking the edited identity.
  Expect(std::distance(std::filesystem::directory_iterator(cache),
                       std::filesystem::directory_iterator{}) <= 1,
         "at most one eligible full digest is published");
  struct stat before{};
  Expect(stat(path.c_str(), &before) == 0, "stat original file");
  bytes[PayloadBase(bytes) + kWindow + 100] ^= 0xff;
  write();
  const timespec times[] = {before.st_atim, before.st_mtim};
  Expect(utimensat(AT_FDCWD, path.c_str(), times, 0) == 0,
         "restore original mtime");
  Expect(read() != first,
         "same-size edit with restored mtime invalidates cache");
  Expect(read() == Identity(bytes),
         "edited file still matches full memory digest");
  for (const auto& entry : std::filesystem::directory_iterator(cache)) {
    std::ofstream(entry.path()) << "broken";
  }
  Expect(read() == Identity(bytes), "malformed cached digest is recomputed");
  if (had_cache)
    gufo::platform::SetEnv("XDG_CACHE_HOME", saved_cache);
  else
    gufo::platform::UnsetEnv("XDG_CACHE_HOME");
  std::filesystem::remove_all(root);
}

constexpr std::int64_t kSecond = 1'000'000'000;

// Only metadata and the clock are simulated. Opening, hashing every byte,
// record validation, permission checks and atomic publication are production
// operations on small real files. Every Read opens a new reader and creates
// new hooks, so persistence does not depend on any process-local cache.
class ControlledFile {
public:
  ControlledFile()
      : bytes(BuildImage("controlled", "first", 4 * kWindow, 2 * kWindow)) {
    char temp[] = "/tmp/gufo-identity-controlled-XXXXXX";
    Expect(mkdtemp(temp) != nullptr, "create controlled cache directory");
    root = temp;
    path = root / "model.gguf";
    cache = root / "gufo" / "gguf-sha256-v1";
    const char* old = std::getenv("XDG_CACHE_HOME");
    had_cache_ = old != nullptr;
    saved_cache_ = old ? old : "";
    gufo::platform::SetEnv("XDG_CACHE_HOME", root.string());
    Write();
    auto reader = Open();
    std::error_code error;
    Expect(gufo::platform::Stat(reader->GetMappedRegions()[0].file_descriptor,
                                &info, error),
           "inspect controlled file descriptor");
    info.modified_ns = info.changed_ns = 10 * kSecond;
  }

  ~ControlledFile() {
    if (had_cache_)
      gufo::platform::SetEnv("XDG_CACHE_HOME", saved_cache_);
    else
      gufo::platform::UnsetEnv("XDG_CACHE_HOME");
    std::filesystem::remove_all(root);
  }

  void Write() const {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    Expect(out.good(), "write controlled GGUF fixture");
  }

  std::unique_ptr<gufo::core::GgufReader> Open() const {
    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(path.string(), &error);
    Expect(reader != nullptr, "open controlled GGUF: " + error);
    return reader;
  }

  std::string Read() {
    hashed_bytes = stat_calls = now_calls = 0;
    gufo::core::detail::GgufIdentityHooks hooks;
    hooks.file_info = [&](gufo::platform::PlatformFile file) {
      gufo::platform::FileInfo actual{};
      std::error_code error;
      Expect(gufo::platform::Stat(file, &actual, error) && actual.regular,
             "identity inspects a live regular file descriptor");
      ++stat_calls;
      if (on_stat)
        on_stat();
      return info;
    };
    hooks.now_ns = [&] {
      ++now_calls;
      if (on_now)
        on_now();
      return now;
    };
    hooks.on_hash_read = [&](std::size_t count) {
      hashed_bytes += count;
      if (on_hash)
        on_hash();
    };
    const auto reader = Open();
    return gufo::core::detail::GgufIdentityHex(*reader, hooks);
  }

  void Edit() {
    bytes[PayloadBase(bytes) + kWindow + 100] ^= 0xff;
    Write();
  }

  std::string Key() const {
    const auto stamp =
        std::to_string(info.device) + ':' + std::to_string(info.inode) + ':' +
        std::to_string(info.size) + ':' + std::to_string(info.modified_ns) +
        ':' + std::to_string(info.changed_ns);
    return gufo::crypto::Sha256Hex(
        {reinterpret_cast<const std::uint8_t*>(stamp.data()), stamp.size()});
  }

  std::string Record(std::int64_t scanned, std::string_view digest) const {
    return "gguf-sha256-cache-v2\n" + std::to_string(scanned) + '\n' +
           std::string(digest) + '\n';
  }

  void Put(std::string_view contents) const {
    std::filesystem::create_directories(cache);
    Expect(chmod(cache.c_str(), 0700) == 0, "restrict fixture cache");
    std::ofstream out(cache / Key(), std::ios::binary);
    out << contents;
    Expect(out.good(), "write fixture cache record");
  }

  std::size_t Records() const {
    if (!std::filesystem::exists(cache))
      return 0;
    return std::distance(std::filesystem::directory_iterator(cache),
                         std::filesystem::directory_iterator{});
  }

  void RequireScan(std::string_view message) {
    // Fault hooks may edit the fixture during lookup; compute the expected
    // identity afterward, independent of operand evaluation order.
    const auto actual = Read();
    Expect(actual == Identity(bytes), message);
    Expect(hashed_bytes == bytes.size(), "miss hashes the complete file");
  }

  std::filesystem::path root, path, cache;
  std::vector<std::uint8_t> bytes;
  gufo::platform::FileInfo info{};
  std::int64_t now{20 * kSecond};
  std::size_t hashed_bytes{0}, stat_calls{0}, now_calls{0};
  std::function<void()> on_hash, on_stat, on_now;

private:
  bool had_cache_{false};
  std::string saved_cache_;
};

void TestRacyMetadataAndReopen() {
  ControlledFile file;
  file.now = file.info.changed_ns;
  const auto first = file.Read();
  Expect(file.hashed_bytes == file.bytes.size() && file.Records() == 0,
         "recent full digest is returned without publishing a cache entry");
  const auto unchanged_metadata = file.info;
  file.Edit();
  Expect(file.info == unchanged_metadata,
         "rewrite collides in every metadata field used by the key");
  file.RequireScan(
      "reopened colliding file gets its new full-content identity");
  Expect(Identity(file.bytes) != first,
         "edited fixture identity really differs");
  Expect(file.Records() == 0, "recent collision never becomes a trusted entry");
  file.now += 100 * kSecond;
  file.RequireScan("aging requires a fresh full scan before first publication");
  Expect(file.Records() == 1,
         "stable file publishes exactly one complete record");
  Expect(file.Read() == Identity(file.bytes) && file.hashed_bytes == 0,
         "reopened stable file uses the persistent fast path without hashing");
}

void TestPersistedSuspectAndLegacyRecords() {
  ControlledFile file;
  const auto stale = gufo::crypto::Sha256Hex(file.bytes);
  // The record carries its original racy scan time. Wall-clock aging must not
  // silently promote it to trusted, even in a new reader/process lifetime.
  file.Put(file.Record(file.info.changed_ns, stale));
  file.Edit();
  file.now = 200 * kSecond;
  file.RequireScan("aged persisted racy record is rehashed, not promoted");
  Expect(file.Read() == Identity(file.bytes) && file.hashed_bytes == 0,
         "new scan replaces the suspect record with a reusable digest");

  // Old versions wrote only the digest and newline. They provide no evidence
  // about whether the scan was temporally ambiguous, however old they are.
  file.Put(stale + '\n');
  file.now = 300 * kSecond;
  file.RequireScan(
      "legacy record is ignored after aging and simulated restart");
  Expect(file.Records() == 1, "legacy cache key is atomically replaced");
  Expect(file.Read() == Identity(file.bytes) && file.hashed_bytes == 0,
         "legacy migration pays one full eligible scan, then reuses it");
}

void TestScanStartAndClockBoundaries() {
  {
    ControlledFile file;
    file.now = file.info.changed_ns + kSecond;
    file.on_hash = [&] { file.now += 100 * kSecond; };
    file.RequireScan(
        "recent-at-start digest remains correct after a long hash");
    Expect(file.Records() == 0,
           "aging during the scan cannot authorize publication");
    file.on_hash = {};
    file.RequireScan("later lookup must rehash the formerly recent file");
    Expect(file.Records() == 1, "new stable-at-start scan is publishable");
  }
  for (const auto delta : {-kSecond, std::int64_t{0}, 2 * kSecond}) {
    ControlledFile file;
    file.now = file.info.changed_ns + delta;
    file.RequireScan(
        "future and exact-boundary metadata still gets a full hash");
    Expect(file.Records() == 0,
           "guard interval is strict, including its boundary");
    file.now = file.info.changed_ns + 2 * kSecond + 1;
    file.RequireScan("first eligible nanosecond requires a fresh scan");
    Expect(file.Records() == 1, "strictly older metadata permits publication");
  }
  {
    ControlledFile file;
    file.on_hash = [&] { --file.now; };
    file.RequireScan("backward clock during hashing does not corrupt identity");
    Expect(file.Records() == 0,
           "backward clock during scan prevents publication");
  }
  {
    ControlledFile file;
    file.RequireScan("establish a stable record before clock reversal");
    file.now = file.info.changed_ns;
    file.Edit();
    file.RequireScan(
        "clock reversal invalidates a matching formerly trusted record");
    Expect(file.Records() == 0,
           "suspect matching record is removed, not retained");
    file.now = 200 * kSecond;
    file.RequireScan("clock recovery cannot revive the stale removed digest");
  }
  for (const auto clock :
       {std::numeric_limits<std::int64_t>::min(), std::int64_t{0}}) {
    ControlledFile file;
    file.now = clock;
    file.RequireScan("invalid clock fails closed without signed overflow");
    Expect(file.Records() == 0, "invalid clock cannot publish");
  }
}

void TestClockReversalDuringCacheHit() {
  {
    ControlledFile file;
    const auto first = file.Read();
    Expect(file.Records() == 1,
           "prepare stable record before cached clock reversal");
    file.on_now = [&] {
      if (file.now_calls == 2) {
        --file.now;
        file.Edit();
      }
    };
    file.RequireScan(
        "clock reversal after a cache hit forces a full content reread");
    Expect(Identity(file.bytes) != first && file.Records() == 0,
           "cached clock reversal retires the stale record persistently");
    file.on_now = {};
    file.now = 200 * kSecond;
    file.RequireScan(
        "later clock recovery cannot resurrect a retired cache hit");
  }
  {
    ControlledFile file;
    file.RequireScan("prepare cache hit for failed retirement");
    const auto record = file.cache / file.Key();
    file.on_now = [&] {
      if (file.now_calls == 2) {
        --file.now;
        // Force unlink failure independently of user privileges or filesystem
        // timing: a directory cannot be removed with the record's unlink API.
        Expect(std::filesystem::remove(record), "remove cached record fixture");
        Expect(std::filesystem::create_directory(record),
               "inject unlink refusal");
      }
    };
    bool refused = false;
    try {
      (void)file.Read();
    } catch (const std::runtime_error& error) {
      refused = std::string_view(error.what()).find("cannot retire unsafe") !=
                std::string_view::npos;
    }
    Expect(refused && file.hashed_bytes == 0,
           "failed mandatory retirement refuses lookup instead of retaining "
           "trust");
    file.on_now = {};
    Expect(std::filesystem::remove(record),
           "remove injected unlink obstruction");
    file.now = 200 * kSecond;
    file.RequireScan("retirement failure does not prevent later fresh hashing");
  }
}

void TestChangedMetadataAndMalformedRecords() {
  for (const bool change_mtime : {false, true}) {
    ControlledFile file;
    file.RequireScan("stable original fixture hashes once");
    file.Edit();
    if (change_mtime)
      ++file.info.modified_ns;
    else
      ++file.info.changed_ns;
    file.RequireScan("changed metadata invalidates the cached full digest");
    Expect(file.Read() == Identity(file.bytes) && file.hashed_bytes == 0,
           "changed stable metadata gets its own reusable record");
  }
  ControlledFile file;
  const std::string wrong(64, '0');
  for (const auto& malformed :
       {std::string{}, std::string("broken"), wrong + '\n',
        file.Record(-1, wrong), file.Record(0, wrong),
        file.Record(file.now + 1, wrong),
        std::string("gguf-sha256-cache-v2\n9999999999999999999999999\n") +
            wrong + '\n',
        std::string("gguf-sha256-cache-v2\n20oops\n") + wrong + '\n',
        file.Record(file.now, std::string(64, 'G')),
        file.Record(file.now, std::string(63, '0')),
        file.Record(file.now, wrong) + "trailing"}) {
    file.Put(malformed);
    file.RequireScan(
        "malformed or temporally invalid persisted record is recomputed");
  }
}

void TestChangesDuringIdentityLookup() {
  {
    ControlledFile file;
    file.on_hash = [&] { ++file.info.changed_ns; };
    bool refused = false;
    try {
      (void)file.Read();
    } catch (const std::runtime_error&) {
      refused = true;
    }
    Expect(refused && file.hashed_bytes == file.bytes.size(),
           "metadata change during full hashing still rejects the lookup");
    Expect(file.Records() == 0, "changing file cannot publish a digest");
  }
  {
    ControlledFile file;
    file.RequireScan(
        "prepare fast-path record for concurrent-change regression");
    file.on_stat = [&] {
      if (file.stat_calls == 3)
        ++file.info.changed_ns;
    };
    bool refused = false;
    try {
      (void)file.Read();
    } catch (const std::runtime_error&) {
      refused = true;
    }
    Expect(refused && file.hashed_bytes == 0,
           "metadata change during cached lookup still rejects the identity");
  }
}

void TestCachePermissionsAndSymlinks() {
  ControlledFile file;
  file.RequireScan("prepare record for permission checks");
  const auto record = file.cache / file.Key();
  Expect(chmod(record.c_str(), 0666) == 0,
         "make cache record writable by others");
  file.RequireScan("writable cache record remains untrusted");
  Expect(chmod(file.cache.c_str(), 0755) == 0,
         "make cache directory nonprivate");
  file.RequireScan("nonprivate cache directory disables reuse");
  Expect(chmod(file.cache.c_str(), 0700) == 0,
         "restore private cache directory");
  const auto outside = file.root / "untouched";
  const std::string sentinel = "must not overwrite symlink target";
  std::ofstream(outside) << sentinel;
  std::filesystem::remove(record);
  std::filesystem::create_symlink(outside, record);
  file.RequireScan("symlink cache record cannot be followed");
  std::string retained;
  std::getline(std::ifstream(outside), retained);
  Expect(retained == sentinel,
         "cache publication leaves symlink target untouched");
}

void TestPipelinedFileHash() {
  const auto bytes = BuildImage("pipeline", "first", 40 * 1024 * 1024, 8194);
  const auto path = std::filesystem::temp_directory_path() /
                    ("gufo-identity-pipeline-" + std::to_string(getpid()));
  {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(path.string(), &error);
  Expect(reader != nullptr, "pipelined fixture opens");
  Expect(gufo::core::GgufIdentityHex(*reader) == Identity(bytes),
         "parallel disk reads hash all chunks and the unaligned tail in order");
  std::filesystem::remove(path);
}

void TestSha256() {
  const std::string text = "abc";
  gufo::crypto::Sha256Hasher hash;
  for (unsigned char byte : text)
    hash.Update({&byte, 1});
  Expect(hash.FinishHex() ==
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
         "incremental SHA-256 matches independent standard vector");
  Expect(gufo::crypto::Sha256Hex({}) ==
             "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
         "empty SHA-256 matches standard vector");
}

}  // namespace

int main() {
  TestSha256();
  TestFileDigestCache();
  TestRacyMetadataAndReopen();
  TestPersistedSuspectAndLegacyRecords();
  TestScanStartAndClockBoundaries();
  TestClockReversalDuringCacheHit();
  TestChangedMetadataAndMalformedRecords();
  TestChangesDuringIdentityLookup();
  TestCachePermissionsAndSymlinks();
  TestPipelinedFileHash();
  TestDeterministic();
  TestHeaderChangesIdentity();
  TestEveryPayloadByte();
  std::cout << "gguf_identity_test passed\n";
  return 0;
}
