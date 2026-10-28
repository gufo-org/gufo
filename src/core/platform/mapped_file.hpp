#ifndef GUFO_CORE_PLATFORM_MAPPED_FILE_HPP_
#define GUFO_CORE_PLATFORM_MAPPED_FILE_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace gufo::platform {

/// Positional-read file handle: int fd on POSIX, HANDLE on Windows.
#if defined(_WIN32)
using PlatformFile = void*;
#else
using PlatformFile = int;
#endif

/// The invalid PlatformFile value.
#if defined(_WIN32)
inline constexpr PlatformFile kInvalidFile = nullptr;
#else
inline constexpr PlatformFile kInvalidFile = -1;
#endif

/// Access hint for a mapping. kSequential tells the OS the mapping is read
/// front to back once; kNone leaves the access pattern to the OS.
enum class MapHint { kNone, kSequential };

/// Read-only demand-paged view of a whole file.
///
/// Linux maps with PROT_READ/MAP_PRIVATE and a sequential-access hint;
/// Windows uses a read-only section mapping of a FILE_FLAG_SEQUENTIAL_SCAN
/// handle. Either way `data()` is backed by the OS page cache and stays
/// valid for the lifetime of this object. Empty files fail to open.
class ReadOnlyMappedFile {
public:
  ReadOnlyMappedFile() = default;
  ReadOnlyMappedFile(const ReadOnlyMappedFile&) = delete;
  ReadOnlyMappedFile& operator=(const ReadOnlyMappedFile&) = delete;
  ReadOnlyMappedFile(ReadOnlyMappedFile&& other) noexcept;
  ReadOnlyMappedFile& operator=(ReadOnlyMappedFile&& other) noexcept;
  ~ReadOnlyMappedFile();

  /// Map `path` read-only. Returns an empty view and sets `error` on failure.
  static ReadOnlyMappedFile Open(const std::filesystem::path& path,
                                 std::string* error = nullptr);

  const std::uint8_t* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }
  bool valid() const noexcept { return data_ != nullptr; }

  /// Borrowed file handle backing the mapping; invalid for an empty view.
  /// Valid for the lifetime of this object.
  PlatformFile PlatformHandle() const noexcept;

  void Close() noexcept;

private:
  void Reset() noexcept;

  const std::uint8_t* data_ = nullptr;
  std::size_t size_ = 0;
#if defined(_WIN32)
  void* handle_ = nullptr;   // HANDLE of the file
  void* mapping_ = nullptr;  // HANDLE of the section
#else
  int fd_ = -1;
#endif
};

/// C-style mapping helper for the safetensors loaders: opens, size-checks,
/// and maps a file read-only with a sequential hint, exposing the raw
/// pointer. Prefer ReadOnlyMappedFile for new code.
struct RawMappedFile {
  void* data{nullptr};
  std::size_t size{0};
  /// Start and length of the OS mapping; differs from `data`/`size` when
  /// OpenRange rounded the offset down to the mapping granularity.
  void* view_base{nullptr};
  std::size_t view_size{0};
#if defined(_WIN32)
  void* file{nullptr};
  void* section{nullptr};
#endif

  /// Maps `path`; on failure leaves the object untouched. Files of size zero
  /// fail, matching the loaders' fstat contract. The hint describes the
  /// expected access pattern.
  bool Open(const std::filesystem::path& path, MapHint hint = MapHint::kNone,
            std::string* error = nullptr) noexcept;
  /// Maps the byte range [offset, offset + length) of `path`. `data` points
  /// at the requested offset, which need not be page-aligned. Zero `length`
  /// fails; on failure the object is left untouched.
  bool OpenRange(const std::filesystem::path& path, std::uint64_t offset,
                 std::size_t length, MapHint hint = MapHint::kNone,
                 std::string* error = nullptr) noexcept;
  void Close() noexcept;
};

/// OS page size, never zero.
std::size_t SystemPageSize() noexcept;

/// Opens `path` read-only; check FileValid() before use. On failure,
/// `error` (when given) receives the reason.
PlatformFile OpenReadFile(const std::filesystem::path& path,
                          std::string* error = nullptr) noexcept;
bool FileValid(PlatformFile file) noexcept;
void CloseFile(PlatformFile file) noexcept;
/// Reads up to `bytes` at `offset`; returns bytes read, 0 on failure/EOF.
std::size_t ReadFileAt(PlatformFile file, void* output, std::size_t bytes,
                       std::uint64_t offset) noexcept;
/// Total file size, or nullopt when it cannot be determined.
std::optional<std::uint64_t> FileSizeBytes(PlatformFile file) noexcept;

/// Duplicates the handle of `file` without re-resolving any path. POSIX
/// shares the open file description; Windows shares the file object. Returns
/// kInvalidFile on failure.
PlatformFile DuplicateFile(PlatformFile file) noexcept;

/// Reopens `file` with unbuffered (direct) I/O, creating an independent
/// description from the retained handle without resolving the original,
/// replaceable pathname. Returns kInvalidFile when direct I/O is unavailable;
/// callers then fall back to buffered reads.
PlatformFile ReopenUnbuffered(PlatformFile file) noexcept;

/// Reopens a retained file for concurrent random reads. Windows uses cached
/// overlapped I/O; Linux attempts direct I/O. Sets `direct` when the returned
/// handle requires aligned reads. Returns kInvalidFile on failure.
PlatformFile ReopenRandomReadFile(PlatformFile file, bool& direct) noexcept;

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_MAPPED_FILE_HPP_
