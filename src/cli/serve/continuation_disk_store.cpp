#include "src/cli/serve/continuation_disk_store.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <list>
#include <random>
#include <semaphore>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/core/crypto/sha256.hpp"

namespace gufo::server {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic = {'G', 'U', 'F', 'O',
                                                'K', 'V', 'C', '1'};
constexpr std::uint32_t kFileVersion = 1;
constexpr std::size_t kChecksumBytes = 64;
constexpr std::size_t kMagicOffset = 0;
constexpr std::size_t kFileVersionOffset = 8;
constexpr std::size_t kPayloadVersionOffset = 12;
constexpr std::size_t kIdentityBytesOffset = 16;
constexpr std::size_t kTokenCountOffset = 20;
constexpr std::size_t kPayloadBytesOffset = 24;
constexpr std::size_t kChecksumOffset = 32;
constexpr std::size_t kHeaderBytes = kChecksumOffset + kChecksumBytes;
constexpr std::size_t kMaxIdentityBytes = std::size_t{64} * 1024U;
constexpr std::size_t kMaxTokenCount = std::size_t{16} * 1024U * 1024U;
constexpr std::string_view kFileSuffix = ".kvc";
constexpr std::string_view kTemporaryPrefix = ".tmp-";

class ScopedFileDescriptor {
public:
  ScopedFileDescriptor() = default;
  explicit ScopedFileDescriptor(int descriptor) noexcept
      : descriptor_(descriptor) {}
  ~ScopedFileDescriptor() {
    if (descriptor_ >= 0) {
      ::close(descriptor_);
    }
  }

  ScopedFileDescriptor(const ScopedFileDescriptor&) = delete;
  ScopedFileDescriptor& operator=(const ScopedFileDescriptor&) = delete;

  ScopedFileDescriptor(ScopedFileDescriptor&& other) noexcept
      : descriptor_(std::exchange(other.descriptor_, -1)) {}
  ScopedFileDescriptor& operator=(ScopedFileDescriptor&& other) noexcept {
    if (this != &other) {
      if (descriptor_ >= 0) {
        ::close(descriptor_);
      }
      descriptor_ = std::exchange(other.descriptor_, -1);
    }
    return *this;
  }

  [[nodiscard]] int get() const noexcept { return descriptor_; }
  [[nodiscard]] int release() noexcept {
    return std::exchange(descriptor_, -1);
  }
  [[nodiscard]] explicit operator bool() const noexcept {
    return descriptor_ >= 0;
  }

private:
  int descriptor_{-1};
};

class ScopedOperationPermit {
public:
  explicit ScopedOperationPermit(std::binary_semaphore& gate) : gate_(gate) {
    gate_.acquire();
  }
  ~ScopedOperationPermit() { gate_.release(); }

  ScopedOperationPermit(const ScopedOperationPermit&) = delete;
  ScopedOperationPermit& operator=(const ScopedOperationPermit&) = delete;
  ScopedOperationPermit(ScopedOperationPermit&&) = delete;
  ScopedOperationPermit& operator=(ScopedOperationPermit&&) = delete;

private:
  std::binary_semaphore& gate_;
};

template<typename Integer>
void PutLittleEndian(std::span<std::uint8_t> destination, std::size_t offset,
                     Integer value) {
  static_assert(std::is_unsigned_v<Integer>);
  if (offset > destination.size() ||
      sizeof(Integer) > destination.size() - offset) {
    throw std::out_of_range("continuation disk header write overflow");
  }
  for (std::size_t byte = 0; byte < sizeof(Integer); ++byte) {
    destination[offset + byte] =
        static_cast<std::uint8_t>(value >> (byte * 8U));
  }
}

template<typename Integer>
bool GetLittleEndian(std::span<const std::uint8_t> source, std::size_t offset,
                     Integer* value) noexcept {
  static_assert(std::is_unsigned_v<Integer>);
  if (value == nullptr || offset > source.size() ||
      sizeof(Integer) > source.size() - offset) {
    return false;
  }
  Integer result = 0;
  for (std::size_t byte = 0; byte < sizeof(Integer); ++byte) {
    result |= static_cast<Integer>(source[offset + byte]) << (byte * 8U);
  }
  *value = result;
  return true;
}

bool IsLowerHexDigest(std::string_view digest) noexcept {
  if (digest.size() != kChecksumBytes) {
    return false;
  }
  return std::ranges::all_of(digest, [](char value) {
    return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
  });
}

std::string DefaultKeyHash(std::span<const std::uint8_t> bytes) {
  return crypto::Sha256Hex(bytes);
}

std::vector<std::uint8_t> MakeKeyBytes(
    const TextRunnerPersistenceDescriptor& persistence,
    std::span<const TextRunnerToken> tokens) {
  if (persistence.compatibility_identity.size() >
          std::numeric_limits<std::uint32_t>::max() ||
      tokens.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("continuation disk key is too large");
  }
  const std::size_t identity_bytes = persistence.compatibility_identity.size();
  if (tokens.size() >
      (std::numeric_limits<std::size_t>::max() - 12U - identity_bytes) /
          sizeof(std::uint32_t)) {
    throw std::overflow_error("continuation disk key size overflows");
  }
  std::vector<std::uint8_t> key(12U + identity_bytes +
                                tokens.size() * sizeof(std::uint32_t));
  PutLittleEndian<std::uint32_t>(key, 0,
                                 static_cast<std::uint32_t>(identity_bytes));
  PutLittleEndian<std::uint32_t>(key, 4, persistence.payload_version);
  PutLittleEndian<std::uint32_t>(key, 8,
                                 static_cast<std::uint32_t>(tokens.size()));
  std::copy(persistence.compatibility_identity.begin(),
            persistence.compatibility_identity.end(), key.begin() + 12);
  std::size_t offset = 12U + identity_bytes;
  for (const TextRunnerToken token : tokens) {
    PutLittleEndian<std::uint32_t>(key, offset, token);
    offset += sizeof(std::uint32_t);
  }
  return key;
}

std::size_t CheckedFileBytes(std::size_t identity_bytes,
                             std::size_t token_count,
                             std::size_t payload_bytes) {
  if (identity_bytes > kMaxIdentityBytes || token_count > kMaxTokenCount) {
    throw std::invalid_argument(
        "continuation disk metadata exceeds format limits");
  }
  if (token_count > (std::numeric_limits<std::size_t>::max() - kHeaderBytes -
                     identity_bytes) /
                        sizeof(std::uint32_t)) {
    throw std::overflow_error("continuation disk metadata size overflows");
  }
  const std::size_t metadata_bytes =
      kHeaderBytes + identity_bytes + token_count * sizeof(std::uint32_t);
  if (payload_bytes >
      std::numeric_limits<std::size_t>::max() - metadata_bytes) {
    throw std::overflow_error("continuation disk file size overflows");
  }
  return metadata_bytes + payload_bytes;
}

std::string ErrnoMessage(std::string_view operation) {
  return std::string(operation) + ": " + std::strerror(errno);
}

bool WriteAll(int descriptor, std::span<const std::uint8_t> bytes) noexcept {
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t written =
        ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (written == 0) {
      return false;
    }
    offset += static_cast<std::size_t>(written);
  }
  return true;
}

bool ReadAll(int descriptor, std::span<std::uint8_t> bytes) noexcept {
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t count =
        ::read(descriptor, bytes.data() + offset, bytes.size() - offset);
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (count == 0) {
      return false;
    }
    offset += static_cast<std::size_t>(count);
  }
  return true;
}

bool HasSuffix(std::string_view value, std::string_view suffix) noexcept {
  return value.size() >= suffix.size() &&
         value.substr(value.size() - suffix.size()) == suffix;
}

enum class ParseFailure : std::uint8_t {
  kNone,
  kCorrupt,
  kChecksum,
};

struct ParsedImage {
  TextRunnerPersistenceDescriptor persistence;
  std::vector<TextRunnerToken> tokens;
  std::size_t payload_offset{0};
  std::size_t payload_bytes{0};
};

ParseFailure ParseAndVerifyImage(std::vector<std::uint8_t>* image,
                                 ParsedImage* parsed) {
  if (image == nullptr || parsed == nullptr || image->size() < kHeaderBytes ||
      !std::equal(kMagic.begin(), kMagic.end(),
                  image->begin() + kMagicOffset)) {
    return ParseFailure::kCorrupt;
  }

  std::uint32_t file_version = 0;
  std::uint32_t payload_version = 0;
  std::uint32_t identity_bytes_u32 = 0;
  std::uint32_t token_count_u32 = 0;
  std::uint64_t payload_bytes_u64 = 0;
  const std::span<const std::uint8_t> readonly(*image);
  if (!GetLittleEndian(readonly, kFileVersionOffset, &file_version) ||
      !GetLittleEndian(readonly, kPayloadVersionOffset, &payload_version) ||
      !GetLittleEndian(readonly, kIdentityBytesOffset, &identity_bytes_u32) ||
      !GetLittleEndian(readonly, kTokenCountOffset, &token_count_u32) ||
      !GetLittleEndian(readonly, kPayloadBytesOffset, &payload_bytes_u64) ||
      file_version != kFileVersion || payload_version == 0 ||
      identity_bytes_u32 == 0 || identity_bytes_u32 > kMaxIdentityBytes ||
      token_count_u32 == 0 || token_count_u32 > kMaxTokenCount ||
      payload_bytes_u64 == 0 ||
      payload_bytes_u64 >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return ParseFailure::kCorrupt;
  }

  const std::size_t identity_bytes = identity_bytes_u32;
  const std::size_t token_count = token_count_u32;
  const std::size_t payload_bytes = static_cast<std::size_t>(payload_bytes_u64);
  std::size_t expected_bytes = 0;
  try {
    expected_bytes =
        CheckedFileBytes(identity_bytes, token_count, payload_bytes);
  } catch (...) {
    return ParseFailure::kCorrupt;
  }
  if (expected_bytes != image->size()) {
    return ParseFailure::kCorrupt;
  }

  const std::string stored_checksum(
      reinterpret_cast<const char*>(image->data() + kChecksumOffset),
      kChecksumBytes);
  if (!IsLowerHexDigest(stored_checksum)) {
    return ParseFailure::kCorrupt;
  }
  std::fill_n(image->begin() + kChecksumOffset, kChecksumBytes, 0);
  const std::string computed_checksum = crypto::Sha256Hex(*image);
  std::copy(stored_checksum.begin(), stored_checksum.end(),
            image->begin() + kChecksumOffset);
  if (computed_checksum != stored_checksum) {
    return ParseFailure::kChecksum;
  }

  parsed->persistence.payload_version = payload_version;
  parsed->persistence.compatibility_identity.resize(identity_bytes);
  std::memcpy(parsed->persistence.compatibility_identity.data(),
              image->data() + kHeaderBytes, identity_bytes);
  parsed->tokens.resize(token_count);
  std::size_t token_offset = kHeaderBytes + identity_bytes;
  for (TextRunnerToken& token : parsed->tokens) {
    if (!GetLittleEndian(readonly, token_offset, &token)) {
      return ParseFailure::kCorrupt;
    }
    token_offset += sizeof(std::uint32_t);
  }
  parsed->payload_offset = token_offset;
  parsed->payload_bytes = payload_bytes;
  return ParseFailure::kNone;
}

std::vector<std::uint8_t> BuildImage(
    const TextRunnerPersistenceDescriptor& persistence,
    std::span<const TextRunnerToken> tokens, std::size_t payload_bytes) {
  const std::size_t file_bytes = CheckedFileBytes(
      persistence.compatibility_identity.size(), tokens.size(), payload_bytes);
  std::vector<std::uint8_t> image(file_bytes);
  std::copy(kMagic.begin(), kMagic.end(), image.begin() + kMagicOffset);
  PutLittleEndian<std::uint32_t>(image, kFileVersionOffset, kFileVersion);
  PutLittleEndian<std::uint32_t>(image, kPayloadVersionOffset,
                                 persistence.payload_version);
  PutLittleEndian<std::uint32_t>(
      image, kIdentityBytesOffset,
      static_cast<std::uint32_t>(persistence.compatibility_identity.size()));
  PutLittleEndian<std::uint32_t>(image, kTokenCountOffset,
                                 static_cast<std::uint32_t>(tokens.size()));
  PutLittleEndian<std::uint64_t>(image, kPayloadBytesOffset,
                                 static_cast<std::uint64_t>(payload_bytes));
  std::copy(persistence.compatibility_identity.begin(),
            persistence.compatibility_identity.end(),
            image.begin() + kHeaderBytes);
  std::size_t token_offset =
      kHeaderBytes + persistence.compatibility_identity.size();
  for (const TextRunnerToken token : tokens) {
    PutLittleEndian<std::uint32_t>(image, token_offset, token);
    token_offset += sizeof(std::uint32_t);
  }
  return image;
}

}  // namespace

struct ContinuationDiskStore::Impl {
  struct Entry {
    std::string filename;
    std::string key_hash;
    TextRunnerPersistenceDescriptor persistence;
    std::vector<TextRunnerToken> tokens;
    std::size_t file_bytes{0};
    std::size_t payload_bytes{0};
    std::filesystem::file_time_type last_access;
  };

  using EntryIterator = std::list<Entry>::iterator;

  Impl(ContinuationDiskStoreOptions store_options, EventSink sink,
       KeyHashFunction hasher)
      : options(std::move(store_options)),
        event_sink(std::move(sink)),
        key_hash(std::move(hasher)) {
    ValidateOptions();
    InitializeDirectory();
    IndexExistingFiles();
    EvictToCapacity();
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  ~Impl() {
    if (directory_fd >= 0) {
      ::close(directory_fd);
    }
  }

  void ValidateOptions() {
    if (options.directory.empty()) {
      throw std::invalid_argument(
          "continuation disk directory must not be empty");
    }
    if (options.capacity_bytes == 0) {
      throw std::invalid_argument("continuation disk capacity must be nonzero");
    }
    if (options.staging_capacity_bytes < kHeaderBytes) {
      throw std::invalid_argument(
          "continuation disk staging capacity is too small");
    }
    if (!key_hash) {
      key_hash = DefaultKeyHash;
    }
  }

  void InitializeDirectory() {
    std::error_code error;
    const bool existed = std::filesystem::exists(options.directory, error);
    if (error) {
      throw std::runtime_error("failed to inspect continuation disk directory");
    }
    if (!existed) {
      std::filesystem::create_directories(options.directory, error);
      if (error) {
        throw std::runtime_error(
            "failed to create continuation disk directory");
      }
    }

    struct stat status{};
    if (::lstat(options.directory.c_str(), &status) != 0) {
      throw std::runtime_error(
          ErrnoMessage("failed to inspect continuation disk directory"));
    }
    if (!S_ISDIR(status.st_mode) || S_ISLNK(status.st_mode) ||
        status.st_uid != ::geteuid()) {
      throw std::runtime_error(
          "continuation disk directory is not a private owned directory");
    }
    if (::chmod(options.directory.c_str(), S_IRWXU) != 0) {
      throw std::runtime_error(
          ErrnoMessage("failed to secure continuation disk directory"));
    }

    directory_fd = ::open(options.directory.c_str(),
                          O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (directory_fd < 0) {
      throw std::runtime_error(
          ErrnoMessage("failed to open continuation disk directory"));
    }
  }

  void Emit(ContinuationDiskEventAction action,
            ContinuationDiskEventReason reason, std::size_t file_bytes,
            std::size_t payload_bytes, std::size_t token_count) const noexcept {
    if (!event_sink) {
      return;
    }
    try {
      event_sink({
          .action = action,
          .reason = reason,
          .file_bytes = file_bytes,
          .payload_bytes = payload_bytes,
          .token_count = token_count,
          .retained_bytes = retained,
          .capacity_bytes = options.capacity_bytes,
      });
    } catch (...) {
      return;
    }
  }

  [[nodiscard]] std::string Hash(std::span<const std::uint8_t> bytes) const {
    const std::string digest = key_hash(bytes);
    if (!IsLowerHexDigest(digest)) {
      throw std::runtime_error(
          "continuation disk key hash must be 64 lowercase hex characters");
    }
    return digest;
  }

  [[nodiscard]] std::string HashKey(
      const TextRunnerPersistenceDescriptor& persistence,
      std::span<const TextRunnerToken> tokens) const {
    return Hash(MakeKeyBytes(persistence, tokens));
  }

  [[nodiscard]] bool SafeRegularFile(std::string_view filename,
                                     struct stat* status) const noexcept {
    if (filename.empty() || filename.find('/') != std::string_view::npos ||
        filename.find("..") != std::string_view::npos || status == nullptr) {
      return false;
    }
    if (::fstatat(directory_fd, std::string(filename).c_str(), status,
                  AT_SYMLINK_NOFOLLOW) != 0) {
      return false;
    }
    return S_ISREG(status->st_mode) && !S_ISLNK(status->st_mode) &&
           status->st_uid == ::geteuid() &&
           (status->st_mode & (S_IRWXG | S_IRWXO)) == 0 &&
           (status->st_mode & S_IRUSR) != 0;
  }

  [[nodiscard]] bool ReadImage(
      std::string_view filename, std::vector<std::uint8_t>* image,
      ContinuationDiskEventReason* failure_reason) const {
    if (image == nullptr || failure_reason == nullptr) {
      return false;
    }
    struct stat status{};
    if (!SafeRegularFile(filename, &status)) {
      *failure_reason = ContinuationDiskEventReason::kUnsafeFile;
      return false;
    }
    if (status.st_size <= 0 ||
        static_cast<std::uint64_t>(status.st_size) >
            static_cast<std::uint64_t>(options.staging_capacity_bytes)) {
      *failure_reason = status.st_size > 0
                            ? ContinuationDiskEventReason::kStagingCapacity
                            : ContinuationDiskEventReason::kCorrupt;
      return false;
    }
    const ScopedFileDescriptor file(
        ::openat(directory_fd, std::string(filename).c_str(),
                 O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (!file) {
      *failure_reason = ContinuationDiskEventReason::kIoFailure;
      return false;
    }
    image->resize(static_cast<std::size_t>(status.st_size));
    if (!ReadAll(file.get(), *image)) {
      *failure_reason = ContinuationDiskEventReason::kIoFailure;
      return false;
    }
    std::uint8_t trailing = 0;
    while (true) {
      const ssize_t count = ::read(file.get(), &trailing, 1);
      if (count < 0 && errno == EINTR) {
        continue;
      }
      if (count != 0) {
        *failure_reason = count < 0 ? ContinuationDiskEventReason::kIoFailure
                                    : ContinuationDiskEventReason::kCorrupt;
        return false;
      }
      break;
    }
    return true;
  }

  void RemoveFileOnly(std::string_view filename) const noexcept {
    if (!filename.empty()) {
      (void)::unlinkat(directory_fd, std::string(filename).c_str(), 0);
    }
  }

  void IndexExistingFiles() {
    std::error_code error;
    const std::filesystem::directory_iterator iterator(
        options.directory,
        std::filesystem::directory_options::skip_permission_denied, error);
    if (error) {
      throw std::runtime_error(
          "failed to enumerate continuation disk directory");
    }
    for (const auto& directory_entry : iterator) {
      const std::string filename = directory_entry.path().filename().string();
      if (filename.starts_with(kTemporaryPrefix)) {
        RemoveFileOnly(filename);
        continue;
      }
      if (!HasSuffix(filename, kFileSuffix)) {
        continue;
      }

      std::vector<std::uint8_t> image;
      ContinuationDiskEventReason failure_reason =
          ContinuationDiskEventReason::kCorrupt;
      ParsedImage parsed;
      if (!ReadImage(filename, &image, &failure_reason)) {
        RemoveFileOnly(filename);
        Emit(ContinuationDiskEventAction::kRemoved, failure_reason, 0, 0, 0);
        continue;
      }
      const ParseFailure parse_failure = ParseAndVerifyImage(&image, &parsed);
      if (parse_failure != ParseFailure::kNone) {
        RemoveFileOnly(filename);
        Emit(ContinuationDiskEventAction::kRemoved,
             parse_failure == ParseFailure::kChecksum
                 ? ContinuationDiskEventReason::kChecksumMismatch
                 : ContinuationDiskEventReason::kCorrupt,
             image.size(), 0, 0);
        continue;
      }

      std::filesystem::file_time_type last_access =
          directory_entry.last_write_time(error);
      if (error) {
        error.clear();
        last_access = std::filesystem::file_time_type::min();
      }
      const std::string digest = HashKey(parsed.persistence, parsed.tokens);
      const EntryIterator duplicate =
          FindExact(digest, parsed.persistence, parsed.tokens);
      if (duplicate != entries.end()) {
        if (duplicate->last_access >= last_access) {
          RemoveFileOnly(filename);
          Emit(ContinuationDiskEventAction::kRemoved,
               ContinuationDiskEventReason::kExactReplacement, image.size(),
               parsed.payload_bytes, parsed.tokens.size());
          continue;
        }
        RemoveEntry(duplicate, ContinuationDiskEventReason::kExactReplacement);
      }
      entries.push_back({
          .filename = filename,
          .key_hash = digest,
          .persistence = std::move(parsed.persistence),
          .tokens = std::move(parsed.tokens),
          .file_bytes = image.size(),
          .payload_bytes = parsed.payload_bytes,
          .last_access = last_access,
      });
      const EntryIterator added = std::prev(entries.end());
      index.emplace(added->key_hash, added);
      retained += added->file_bytes;
    }
  }

  [[nodiscard]] EntryIterator FindExact(
      const std::string& digest,
      const TextRunnerPersistenceDescriptor& persistence,
      std::span<const TextRunnerToken> tokens) {
    const auto [begin, end] = index.equal_range(digest);
    for (auto current = begin; current != end; ++current) {
      const EntryIterator entry = current->second;
      if (entry->persistence == persistence &&
          std::ranges::equal(entry->tokens, tokens)) {
        return entry;
      }
    }
    return entries.end();
  }

  void EraseIndex(EntryIterator entry) {
    const auto [begin, end] = index.equal_range(entry->key_hash);
    for (auto current = begin; current != end; ++current) {
      if (current->second == entry) {
        index.erase(current);
        return;
      }
    }
  }

  bool RemoveEntry(EntryIterator entry,
                   ContinuationDiskEventReason reason) noexcept {
    if (entry == entries.end()) {
      return true;
    }
    if (::unlinkat(directory_fd, entry->filename.c_str(), 0) != 0 &&
        errno != ENOENT) {
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kIoFailure, entry->file_bytes,
           entry->payload_bytes, entry->tokens.size());
      return false;
    }
    const std::size_t file_bytes = entry->file_bytes;
    const std::size_t payload_bytes = entry->payload_bytes;
    const std::size_t token_count = entry->tokens.size();
    EraseIndex(entry);
    retained -= file_bytes;
    entries.erase(entry);
    Emit(ContinuationDiskEventAction::kRemoved, reason, file_bytes,
         payload_bytes, token_count);
    return true;
  }

  [[nodiscard]] EntryIterator LeastRecentlyUsed(EntryIterator excluded) {
    EntryIterator selected = entries.end();
    for (auto current = entries.begin(); current != entries.end(); ++current) {
      if (excluded != entries.end() && current == excluded) {
        continue;
      }
      if (selected == entries.end() ||
          current->last_access < selected->last_access ||
          (current->last_access == selected->last_access &&
           current->filename < selected->filename)) {
        selected = current;
      }
    }
    return selected;
  }

  void EvictToCapacity() {
    while (retained > options.capacity_bytes) {
      const EntryIterator victim = LeastRecentlyUsed(entries.end());
      if (victim == entries.end() ||
          !RemoveEntry(victim, ContinuationDiskEventReason::kLru)) {
        break;
      }
    }
  }

  bool MakeCapacity(std::size_t file_bytes, EntryIterator replacement) {
    const std::size_t replacement_bytes =
        replacement != entries.end() ? replacement->file_bytes : 0;
    if (file_bytes > options.capacity_bytes) {
      return false;
    }
    while (retained - replacement_bytes > options.capacity_bytes - file_bytes) {
      const EntryIterator victim = LeastRecentlyUsed(replacement);
      if (victim == entries.end() ||
          !RemoveEntry(victim, ContinuationDiskEventReason::kLru)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] std::string UniqueSuffix() {
    const std::uint64_t random_value =
        (static_cast<std::uint64_t>(random_device()) << 32U) ^
        static_cast<std::uint64_t>(random_device()) ^
        static_cast<std::uint64_t>(++unique_counter) ^
        static_cast<std::uint64_t>(::getpid());
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << random_value;
    return output.str();
  }

  [[nodiscard]] std::string NewFinalFilename(const std::string& digest) {
    for (int attempt = 0; attempt < 64; ++attempt) {
      const std::string filename =
          digest + "-" + UniqueSuffix() + std::string(kFileSuffix);
      struct stat status{};
      if (::fstatat(directory_fd, filename.c_str(), &status,
                    AT_SYMLINK_NOFOLLOW) != 0 &&
          errno == ENOENT) {
        return filename;
      }
    }
    throw std::runtime_error("failed to allocate a continuation disk filename");
  }

  [[nodiscard]] bool PublishImage(std::string_view final_filename,
                                  std::span<const std::uint8_t> image) {
    const std::string temporary_filename =
        std::string(kTemporaryPrefix) + UniqueSuffix();
    ScopedFileDescriptor temporary(
        ::openat(directory_fd, temporary_filename.c_str(),
                 O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                 S_IRUSR | S_IWUSR));
    if (!temporary) {
      return false;
    }
    bool valid = WriteAll(temporary.get(), image);
    valid = valid && ::fsync(temporary.get()) == 0;
    const int raw_descriptor = temporary.release();
    if (::close(raw_descriptor) != 0) {
      valid = false;
    }
    if (!valid) {
      RemoveFileOnly(temporary_filename);
      return false;
    }
    if (::renameat(directory_fd, temporary_filename.c_str(), directory_fd,
                   std::string(final_filename).c_str()) != 0) {
      RemoveFileOnly(temporary_filename);
      return false;
    }
    if (::fsync(directory_fd) != 0) {
      RemoveFileOnly(final_filename);
      (void)::fsync(directory_fd);
      return false;
    }
    return true;
  }

  [[nodiscard]] SaveResult Save(
      const TextModelRunner& runner,
      std::span<const TextRunnerToken> checkpoint_tokens,
      const TextRunnerSnapshot& snapshot) {
    const auto descriptor = runner.Descriptor();
    if (!descriptor.persistence.has_value()) {
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kUnsupported, 0, 0,
           checkpoint_tokens.size());
      return {};
    }
    if (checkpoint_tokens.empty()) {
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kSerializationFailure, 0, 0, 0);
      return {};
    }

    std::size_t payload_bytes = 0;
    std::vector<std::uint8_t> image;
    try {
      payload_bytes = runner.PersistentSnapshotPayloadBytes(snapshot);
      if (payload_bytes == 0) {
        throw std::runtime_error("persistent snapshot payload is empty");
      }
      const std::size_t file_bytes = CheckedFileBytes(
          descriptor.persistence->compatibility_identity.size(),
          checkpoint_tokens.size(), payload_bytes);
      if (file_bytes > options.staging_capacity_bytes) {
        Emit(ContinuationDiskEventAction::kSkipped,
             ContinuationDiskEventReason::kStagingCapacity, file_bytes,
             payload_bytes, checkpoint_tokens.size());
        return {};
      }
      if (file_bytes > options.capacity_bytes) {
        Emit(ContinuationDiskEventAction::kSkipped,
             ContinuationDiskEventReason::kByteCapacity, file_bytes,
             payload_bytes, checkpoint_tokens.size());
        return {};
      }
      image =
          BuildImage(*descriptor.persistence, checkpoint_tokens, payload_bytes);
      const std::size_t payload_offset = image.size() - payload_bytes;
      const std::size_t written = runner.SerializePersistentSnapshot(
          snapshot, std::span<std::uint8_t>(image).subspan(payload_offset,
                                                           payload_bytes));
      if (written != payload_bytes) {
        throw std::runtime_error(
            "persistent snapshot serializer returned the wrong byte count");
      }
    } catch (...) {
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kSerializationFailure, image.size(),
           payload_bytes, checkpoint_tokens.size());
      return {};
    }

    std::fill_n(image.begin() + kChecksumOffset, kChecksumBytes, 0);
    const std::string checksum = crypto::Sha256Hex(image);
    std::copy(checksum.begin(), checksum.end(),
              image.begin() + kChecksumOffset);

    const std::string digest =
        HashKey(*descriptor.persistence, checkpoint_tokens);
    const EntryIterator replacement =
        FindExact(digest, *descriptor.persistence, checkpoint_tokens);
    if (!MakeCapacity(image.size(), replacement)) {
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kByteCapacity, image.size(),
           payload_bytes, checkpoint_tokens.size());
      return {};
    }

    const std::string filename = NewFinalFilename(digest);
    if (!PublishImage(filename, image)) {
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kIoFailure, image.size(), payload_bytes,
           checkpoint_tokens.size());
      return {};
    }

    if (replacement != entries.end()) {
      if (::unlinkat(directory_fd, replacement->filename.c_str(), 0) != 0 &&
          errno != ENOENT) {
        RemoveFileOnly(filename);
        (void)::fsync(directory_fd);
        Emit(ContinuationDiskEventAction::kSkipped,
             ContinuationDiskEventReason::kIoFailure, image.size(),
             payload_bytes, checkpoint_tokens.size());
        return {};
      }
      const std::size_t old_file_bytes = replacement->file_bytes;
      const std::size_t old_payload_bytes = replacement->payload_bytes;
      retained -= old_file_bytes;
      replacement->filename = filename;
      replacement->persistence = *descriptor.persistence;
      replacement->tokens.assign(checkpoint_tokens.begin(),
                                 checkpoint_tokens.end());
      replacement->file_bytes = image.size();
      replacement->payload_bytes = payload_bytes;
      replacement->last_access = std::filesystem::file_time_type::clock::now();
      retained += image.size();
      (void)::fsync(directory_fd);
      Emit(ContinuationDiskEventAction::kRemoved,
           ContinuationDiskEventReason::kExactReplacement, old_file_bytes,
           old_payload_bytes, checkpoint_tokens.size());
    } else {
      entries.push_back({
          .filename = filename,
          .key_hash = digest,
          .persistence = *descriptor.persistence,
          .tokens = std::vector<TextRunnerToken>(checkpoint_tokens.begin(),
                                                 checkpoint_tokens.end()),
          .file_bytes = image.size(),
          .payload_bytes = payload_bytes,
          .last_access = std::filesystem::file_time_type::clock::now(),
      });
      const EntryIterator added = std::prev(entries.end());
      index.emplace(digest, added);
      retained += image.size();
    }
    Emit(ContinuationDiskEventAction::kStored,
         ContinuationDiskEventReason::kSaved, image.size(), payload_bytes,
         checkpoint_tokens.size());
    return {
        .stored = true,
        .file_bytes = image.size(),
        .payload_bytes = payload_bytes,
    };
  }

  [[nodiscard]] EntryIterator FindLongestCandidate(
      const TextRunnerPersistenceDescriptor& persistence,
      std::span<const TextRunnerToken> prompt) {
    std::set<std::size_t, std::greater<>> token_counts;
    for (const Entry& entry : entries) {
      if (entry.tokens.size() <= prompt.size()) {
        token_counts.insert(entry.tokens.size());
      }
    }
    for (const std::size_t token_count : token_counts) {
      const auto prefix = prompt.first(token_count);
      const std::string digest = HashKey(persistence, prefix);
      const auto [begin, end] = index.equal_range(digest);
      EntryIterator selected = entries.end();
      for (auto current = begin; current != end; ++current) {
        const EntryIterator candidate = current->second;
        if (candidate->persistence == persistence &&
            std::ranges::equal(candidate->tokens, prefix) &&
            (selected == entries.end() ||
             candidate->last_access > selected->last_access)) {
          selected = candidate;
        }
      }
      if (selected != entries.end()) {
        return selected;
      }
    }
    return entries.end();
  }

  [[nodiscard]] RestoreResult RestoreLongestPrefix(
      const TextModelRunner& runner, TextRunnerState& state,
      std::span<const TextRunnerToken> prompt) {
    const auto descriptor = runner.Descriptor();
    if (!descriptor.persistence.has_value()) {
      Emit(ContinuationDiskEventAction::kMiss,
           ContinuationDiskEventReason::kUnsupported, 0, 0, 0);
      return {};
    }

    while (true) {
      const EntryIterator candidate =
          FindLongestCandidate(*descriptor.persistence, prompt);
      if (candidate == entries.end()) {
        Emit(ContinuationDiskEventAction::kMiss,
             ContinuationDiskEventReason::kNotFound, 0, 0, 0);
        return {};
      }

      std::vector<std::uint8_t> image;
      ContinuationDiskEventReason failure_reason =
          ContinuationDiskEventReason::kCorrupt;
      if (!ReadImage(candidate->filename, &image, &failure_reason)) {
        const std::size_t file_bytes = candidate->file_bytes;
        const std::size_t payload_bytes = candidate->payload_bytes;
        const std::size_t token_count = candidate->tokens.size();
        (void)RemoveEntry(candidate, failure_reason);
        Emit(ContinuationDiskEventAction::kMiss, failure_reason, file_bytes,
             payload_bytes, token_count);
        continue;
      }

      ParsedImage parsed;
      const ParseFailure parse_failure = ParseAndVerifyImage(&image, &parsed);
      if (parse_failure != ParseFailure::kNone ||
          parsed.persistence != candidate->persistence ||
          parsed.tokens != candidate->tokens ||
          parsed.payload_bytes != candidate->payload_bytes) {
        const ContinuationDiskEventReason reason =
            parse_failure == ParseFailure::kChecksum
                ? ContinuationDiskEventReason::kChecksumMismatch
                : ContinuationDiskEventReason::kCorrupt;
        const std::size_t file_bytes = candidate->file_bytes;
        const std::size_t payload_bytes = candidate->payload_bytes;
        const std::size_t token_count = candidate->tokens.size();
        (void)RemoveEntry(candidate, reason);
        Emit(ContinuationDiskEventAction::kMiss, reason, file_bytes,
             payload_bytes, token_count);
        continue;
      }

      try {
        runner.RestorePersistentSnapshot(
            state, std::span<const std::uint8_t>(image).subspan(
                       parsed.payload_offset, parsed.payload_bytes));
      } catch (...) {
        state.Invalidate();
        const std::size_t file_bytes = candidate->file_bytes;
        const std::size_t payload_bytes = candidate->payload_bytes;
        const std::size_t token_count = candidate->tokens.size();
        (void)RemoveEntry(candidate,
                          ContinuationDiskEventReason::kRestoreFailure);
        Emit(ContinuationDiskEventAction::kMiss,
             ContinuationDiskEventReason::kRestoreFailure, file_bytes,
             payload_bytes, token_count);
        return {};
      }

      candidate->last_access = std::filesystem::file_time_type::clock::now();
      (void)::utimensat(directory_fd, candidate->filename.c_str(), nullptr,
                        AT_SYMLINK_NOFOLLOW);
      Emit(ContinuationDiskEventAction::kRestored,
           ContinuationDiskEventReason::kHit, candidate->file_bytes,
           candidate->payload_bytes, candidate->tokens.size());
      return {
          .restored = true,
          .token_count = candidate->tokens.size(),
          .file_bytes = candidate->file_bytes,
          .payload_bytes = candidate->payload_bytes,
      };
    }
  }

  ContinuationDiskStoreOptions options;
  EventSink event_sink;
  KeyHashFunction key_hash;
  int directory_fd{-1};
  mutable std::binary_semaphore operation_gate{1};
  std::list<Entry> entries;
  std::unordered_multimap<std::string, EntryIterator> index;
  std::size_t retained{0};
  std::random_device random_device;
  std::uint64_t unique_counter{0};
};

ContinuationDiskStore::ContinuationDiskStore(
    ContinuationDiskStoreOptions options, EventSink event_sink,
    KeyHashFunction key_hash)
    : impl_(std::make_unique<Impl>(std::move(options), std::move(event_sink),
                                   std::move(key_hash))) {}

ContinuationDiskStore::~ContinuationDiskStore() = default;

ContinuationDiskStore::SaveResult ContinuationDiskStore::Save(
    const TextModelRunner& runner,
    std::span<const TextRunnerToken> checkpoint_tokens,
    const TextRunnerSnapshot& snapshot) {
  const ScopedOperationPermit permit(impl_->operation_gate);
  return impl_->Save(runner, checkpoint_tokens, snapshot);
}

ContinuationDiskStore::RestoreResult
ContinuationDiskStore::RestoreLongestPrefix(
    const TextModelRunner& runner, TextRunnerState& state,
    std::span<const TextRunnerToken> prompt) {
  const ScopedOperationPermit permit(impl_->operation_gate);
  return impl_->RestoreLongestPrefix(runner, state, prompt);
}

std::size_t ContinuationDiskStore::entry_count() const noexcept {
  const ScopedOperationPermit permit(impl_->operation_gate);
  return impl_->entries.size();
}

std::size_t ContinuationDiskStore::retained_bytes() const noexcept {
  const ScopedOperationPermit permit(impl_->operation_gate);
  return impl_->retained;
}

std::size_t ContinuationDiskStore::capacity_bytes() const noexcept {
  return impl_->options.capacity_bytes;
}

std::size_t ContinuationDiskStore::staging_capacity_bytes() const noexcept {
  return impl_->options.staging_capacity_bytes;
}

}  // namespace gufo::server
