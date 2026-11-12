#include "src/core/gguf_identity.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <future>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_identity_internal.hpp"
#include "src/core/platform/fs.hpp"
#include "src/core/platform/os_info.hpp"

namespace gufo::core {
namespace {

class FileDescriptor {
public:
  explicit FileDescriptor(platform::PlatformFile file) : file_(file) {}
  ~FileDescriptor() { platform::CloseFile(file_); }
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  [[nodiscard]] platform::PlatformFile get() const { return file_; }

private:
  platform::PlatformFile file_;
};

void HashString(crypto::Sha256Hasher& hash, std::string_view value) {
  hash.Update(
      {reinterpret_cast<const std::uint8_t*>(value.data()), value.size()});
}

// Larger than the timestamp tick on supported local filesystems. This is a
// bound on timestamp granularity/update lag, not a content-change detector.
// See docs/internal/cache.md#gguf-identity-cache for clock and filesystem
// assumptions.
constexpr std::int64_t kAmbiguityNs = 2'000'000'000;
constexpr std::string_view kDigestRecord = "gguf-sha256-cache-v2\n";

std::int64_t Now(const detail::GgufIdentityHooks& hooks) {
  return hooks.now_ns ? hooks.now_ns()
                      : std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
}

struct Stamp {
  std::string key;
  std::int64_t modified_ns;
  std::int64_t changed_ns;

  bool StableAt(std::int64_t time) const {
    // Subtract only from a positive, bounded clock value, never from file
    // timestamps (which may be negative, future-dated or malformed).
    return time > kAmbiguityNs && modified_ns > 0 && changed_ns > 0 &&
           modified_ns < time - kAmbiguityNs &&
           changed_ns < time - kAmbiguityNs;
  }
};

Stamp FileStamp(platform::PlatformFile file, std::size_t size,
                const detail::GgufIdentityHooks& hooks) {
  platform::FileInfo info{};
  std::error_code ec;
  if (hooks.file_info)
    info = hooks.file_info(file);
  else if (!platform::Stat(file, &info, ec))
    throw std::runtime_error("GGUF file cannot be inspected");
  if (!info.regular || info.size != size) {
    throw std::runtime_error("GGUF file changed or cannot be inspected");
  }
  return {std::to_string(info.device) + ':' + std::to_string(info.inode) + ':' +
              std::to_string(info.size) + ':' +
              std::to_string(info.modified_ns) + ':' +
              std::to_string(info.changed_ns),
          info.modified_ns, info.changed_ns};
}

platform::Directory OpenCacheDirectory() {
  auto base = platform::UserCacheDirectory();
  if (base.empty())
    return {};
  std::error_code error;
  base /= "gufo";
  std::filesystem::create_directories(base, error);
  if (error)
    return {};
  base /= "gguf-sha256-v1";
  (void)platform::MakeDirectory(base, error);
  platform::Directory directory;
  if (!directory.Open(base, error))
    return {};
  platform::FileInfo info{};
  if (!platform::StatAt(directory, ".", false, &info, error) ||
      !platform::OwnerOnly(info, 0077))
    return {};
  return directory;
}

struct CachedDigest {
  std::string value;
  bool must_retire{false};
};

CachedDigest ReadDigest(const platform::Directory& directory,
                        const std::string& key, const Stamp& stamp,
                        std::int64_t now) {
  if (!directory.valid())
    return {};
  std::error_code ec;
  FileDescriptor file(
      platform::OpenAt(directory, key, platform::FileMode::kRead, ec));
  platform::FileInfo info{};
  if (!platform::FileValid(file.get()) ||
      !platform::Stat(file.get(), &info, ec) || !info.regular ||
      !platform::OwnerOnly(info, 0022) || info.size > 128)
    return {};
  std::array<char, 128> bytes{};
  if (platform::Read(file.get(), bytes.data(), info.size, ec) != info.size)
    return {};
  platform::FileInfo after{};
  if (!platform::Stat(file.get(), &after, ec) || after != info)
    return {};
  std::string_view record(bytes.data(), info.size);
  if (!record.starts_with(kDigestRecord))
    return {};  // Legacy records have no proof of a non-racy full scan.
  record.remove_prefix(kDigestRecord.size());
  const auto newline = record.find('\n');
  if (newline == std::string_view::npos || newline == 0)
    return {};
  std::int64_t scanned_at{};
  const auto [end, error] =
      std::from_chars(record.data(), record.data() + newline, scanned_at);
  if (error != std::errc{} || end != record.data() + newline ||
      !stamp.StableAt(scanned_at))
    return {};
  record.remove_prefix(newline + 1);
  if (record.size() != 65 || record.back() != '\n' ||
      !std::all_of(record.begin(), record.end() - 1, [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      }))
    return {};
  if (now < scanned_at || !stamp.StableAt(now))
    return {{}, true};
  return {std::string(record.substr(0, 64))};
}

void DiscardDigest(const platform::Directory& directory, const std::string& key,
                   bool required = false) {
  if (!directory.valid())
    return;
  std::error_code ec;
  if (!platform::Unlink(directory, key, ec) && required &&
      ec != std::errc::no_such_file_or_directory)
    throw std::runtime_error("cannot retire unsafe GGUF digest cache record: " +
                             ec.message());
}

void StoreDigest(const platform::Directory& directory, const std::string& key,
                 const std::string& digest, std::int64_t scanned_at) {
  if (!directory.valid())
    return;
  static std::atomic<unsigned long> serial{0};
  const std::string temp = "." + key + "." +
                           std::to_string(platform::ProcessId()) + "." +
                           std::to_string(serial.fetch_add(1));
  std::error_code ec;
  FileDescriptor file(
      platform::OpenAt(directory, temp, platform::FileMode::kCreateNew, ec));
  if (!platform::FileValid(file.get()))
    return;
  const std::string contents = std::string(kDigestRecord) +
                               std::to_string(scanned_at) + '\n' + digest +
                               '\n';
  if (platform::Write(file.get(), contents.data(), contents.size(), ec) ==
      contents.size()) {
    (void)platform::Rename(directory, temp, directory, key, ec);
  }
  (void)platform::Unlink(directory, temp, ec);
}

void HashFile(crypto::Sha256Hasher& hash, platform::PlatformFile file,
              std::size_t size, const detail::GgufIdentityHooks& hooks) {
  constexpr std::size_t kChunk = 8 * 1024 * 1024;
  constexpr std::size_t kAlignment = 4096;
  constexpr std::size_t kReaders = 4;
  // Preserve the open inode and avoid filling the remaining UMA with a second
  // copy of weights after GPU loading. Hashing stays in exact file order.
  FileDescriptor direct(size >= kChunk ? platform::ReopenUnbuffered(file)
                                       : platform::kInvalidFile);
  struct AlignedDeleter {
    std::size_t alignment;
    void operator()(std::uint8_t* pointer) const noexcept {
      ::operator delete(pointer, std::align_val_t(alignment));
    }
  };
  struct Slot {
    std::unique_ptr<std::uint8_t, AlignedDeleter> buffer{
        nullptr, AlignedDeleter{kAlignment}};
    std::future<void> read;
    std::size_t size{0};
  };
  std::array<Slot, kReaders> slots;
  std::size_t next = 0;
  const auto queue = [&](Slot& slot) {
    if (next == size)
      return;
    if (!slot.buffer) {
      slot.buffer.reset(static_cast<std::uint8_t*>(
          ::operator new(kChunk, std::align_val_t(kAlignment))));
    }
    const auto offset = next;
    slot.size = std::min(kChunk, size - next);
    next += slot.size;
    slot.read = std::async(std::launch::async, [&, offset] {
      std::size_t done = 0;
      bool buffered = !platform::FileValid(direct.get());
      while (done < slot.size) {
        const auto remaining = slot.size - done;
        const auto requested =
            buffered ? remaining
                     : (remaining + kAlignment - 1) / kAlignment * kAlignment;
        std::error_code ec;
        const auto count =
            platform::ReadAt(buffered ? file : direct.get(), offset + done,
                             slot.buffer.get() + done, requested, ec);
        if (count == 0 && ec == std::errc::interrupted)
          continue;
        if (count == 0 && !buffered &&
            (ec == std::errc::invalid_argument ||
             ec == std::errc::operation_not_supported)) {
          buffered = true;
          continue;
        }
        if (count == 0)
          throw std::runtime_error("cannot read complete GGUF for identity");
        done += std::min(count, remaining);
        if (done % kAlignment != 0)
          buffered = true;
      }
    });
  };
  for (auto& slot : slots)
    queue(slot);
  for (std::size_t index = 0; slots[index].read.valid();
       index = (index + 1) % slots.size()) {
    auto& slot = slots[index];
    slot.read.get();
    hash.Update({slot.buffer.get(), slot.size});
    if (hooks.on_hash_read)
      hooks.on_hash_read(slot.size);
    queue(slot);
  }
}

std::string RegionDigest(const GgufMappedRegion& region,
                         const detail::GgufIdentityHooks& hooks) {
  // Eligibility is captured BEFORE stat/hash, never at publication: a long
  // scan must not turn a recent, potentially stale digest into a trusted one.
  const auto started = Now(hooks);
  Stamp stamp{};
  if (platform::FileValid(region.file_descriptor))
    stamp = FileStamp(region.file_descriptor, region.size, hooks);
  platform::Directory cache =
      stamp.key.empty() ? platform::Directory{} : OpenCacheDirectory();
  crypto::Sha256Hasher key_hash;
  HashString(key_hash, stamp.key);
  const auto key = key_hash.FinishHex();
  auto record = ReadDigest(cache, key, stamp, started);
  std::string digest = std::move(record.value);
  const bool cached = !digest.empty();
  const auto hash_content = [&] {
    crypto::Sha256Hasher hash;
    constexpr std::size_t chunk = 8 * 1024 * 1024;
    if (platform::FileValid(region.file_descriptor)) {
      HashFile(hash, region.file_descriptor, region.size, hooks);
    } else {
      const auto* bytes = static_cast<const std::uint8_t*>(region.data);
      for (std::size_t offset = 0; offset < region.size;) {
        const auto length = std::min(chunk, region.size - offset);
        hash.Update({bytes + offset, length});
        if (hooks.on_hash_read)
          hooks.on_hash_read(length);
        offset += length;
      }
    }
    return hash.FinishHex();
  };
  const auto check_stamp = [&] {
    if (!stamp.key.empty() &&
        FileStamp(region.file_descriptor, region.size, hooks).key !=
            stamp.key) {
      DiscardDigest(cache, key);
      throw std::runtime_error("GGUF file changed during identity lookup");
    }
  };
  if (!cached) {
    // Persistently retire suspicious records. In particular a clock rollback
    // must not leave one to be reused when the wall clock advances again.
    DiscardDigest(cache, key, record.must_retire);
    digest = hash_content();
  }
  check_stamp();
  const bool clock_stable = Now(hooks) >= started;
  if (!clock_stable) {
    DiscardDigest(cache, key, cached);
    if (cached) {
      digest = hash_content();
      check_stamp();
    }
  } else if (!cached && stamp.StableAt(started)) {
    StoreDigest(cache, key, digest, started);
  }
  return digest;
}

}  // namespace

std::string GgufIdentityHex(const GgufReader& reader) {
  return detail::GgufIdentityHex(reader, {});
}

std::string detail::GgufIdentityHex(const GgufReader& reader,
                                    const GgufIdentityHooks& hooks) {
  std::vector<std::string> stamps;
  for (const auto& region : reader.GetMappedRegions())
    stamps.push_back(
        platform::FileValid(region.file_descriptor)
            ? FileStamp(region.file_descriptor, region.size, hooks).key
            : std::string{});
  crypto::Sha256Hasher hash;
  HashString(hash, kGgufIdentityScheme);
  for (const auto& region : reader.GetMappedRegions()) {
    HashString(hash, ":" + std::to_string(region.size) + ":");
    HashString(hash, RegionDigest(region, hooks));
  }
  for (std::size_t index = 0; index < stamps.size(); ++index) {
    const auto& region = reader.GetMappedRegions()[index];
    if (!stamps[index].empty() &&
        stamps[index] !=
            FileStamp(region.file_descriptor, region.size, hooks).key)
      throw std::runtime_error("GGUF artifact changed during identity lookup");
  }
  return hash.FinishHex();
}

}  // namespace gufo::core
