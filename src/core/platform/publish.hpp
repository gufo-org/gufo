#ifndef GUFO_CORE_PLATFORM_PUBLISH_HPP_
#define GUFO_CORE_PLATFORM_PUBLISH_HPP_

#include <cstddef>
#include <filesystem>
#include <string>

namespace gufo::platform {
namespace detail {
#if defined(_WIN32)
using PublicationHandle = void*;
inline constexpr PublicationHandle kInvalidPublicationHandle = nullptr;
#else
using PublicationHandle = int;
inline constexpr PublicationHandle kInvalidPublicationHandle = -1;
#endif
}  // namespace detail

/// Advisory inter-process exclusion on a lock file (plan database, model
/// single-instance locks). POSIX blocks in flock(LOCK_EX); Windows takes an
/// exclusive byte-range lock. Held until destruction; released by the OS if
/// the process dies. Throws std::system_error when the lock cannot be taken.
class FileLock {
public:
  FileLock() = default;
  FileLock(const FileLock&) = delete;
  FileLock& operator=(const FileLock&) = delete;
  FileLock(FileLock&& other) noexcept;
  FileLock& operator=(FileLock&& other) noexcept;
  ~FileLock();

  static FileLock Acquire(const std::string& path);

  /// Attempts a non-blocking single-instance lock on `path`, recording the
  /// current PID on success. If another process holds the lock, sets
  /// `*contended = true` and populates `*holder_pid` (when available). If the
  /// lock file cannot be opened at all, sets `*contended = false` and returns
  /// an invalid FileLock.
  static FileLock TryAcquireProcessLock(
      const std::filesystem::path& path, bool* contended,
      std::string* holder_pid = nullptr) noexcept;

  bool valid() const noexcept {
    return handle_ != detail::kInvalidPublicationHandle;
  }

  void Release() noexcept;

private:
  explicit FileLock(detail::PublicationHandle handle) noexcept
      : handle_(handle) {}

  detail::PublicationHandle handle_ = detail::kInvalidPublicationHandle;
};

/// Returns `/tmp/<filename>` on POSIX and `<temp_directory_path>/<filename>`
/// on Windows.
std::filesystem::path DefaultProcessLockPath(std::string_view filename);

/// A temporary file created next to its final destination, published by
/// Commit() with flush-then-rename semantics: fsync, rename over the
/// destination, then fsync the containing directory (POSIX). On Windows the
/// temp file is flushed and moved with MOVEFILE_REPLACE_EXISTING.
class AtomicPublishedFile {
public:
  AtomicPublishedFile() = default;
  AtomicPublishedFile(const AtomicPublishedFile&) = delete;
  AtomicPublishedFile& operator=(const AtomicPublishedFile&) = delete;
  AtomicPublishedFile(AtomicPublishedFile&& other) noexcept;
  AtomicPublishedFile& operator=(AtomicPublishedFile&& other) noexcept;
  ~AtomicPublishedFile();

  /// Creates `<final_path>.tmp.XXXXXX` in the destination directory.
  /// Returns an invalid file and sets `error` on failure.
  static AtomicPublishedFile Create(const std::filesystem::path& final_path,
                                    std::string* error);

  bool valid() const noexcept {
    return handle_ != detail::kInvalidPublicationHandle;
  }

  bool Write(const void* data, std::size_t bytes, std::string* error);

  /// Flush, replace the destination, and flush the directory. Returns false
  /// and sets `error` on failure; the temporary file is removed either way.
  /// A directory-flush failure is reported after the destination was replaced.
  bool Commit(const std::filesystem::path& final_path, std::string* error);

private:
  AtomicPublishedFile(detail::PublicationHandle handle,
                      const std::filesystem::path& temporary)
      : handle_(handle), temporary_(temporary) {}

  void Discard() noexcept;

  detail::PublicationHandle handle_ = detail::kInvalidPublicationHandle;
  std::filesystem::path temporary_;
};

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_PUBLISH_HPP_
