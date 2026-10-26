#ifndef GUFO_CORE_PLATFORM_FS_HPP_
#define GUFO_CORE_PLATFORM_FS_HPP_

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <system_error>

#include "src/core/platform/mapped_file.hpp"

namespace gufo::platform {

/// File identity and metadata. Fields are explicit-width so identity
/// comparison is exact on every OS.
struct FileInfo {
  /// POSIX st_dev; Windows volume serial number.
  std::uint64_t device;
  /// POSIX st_ino; Windows 64-bit file index.
  std::uint64_t inode;
  std::uint64_t size;
  /// POSIX st_mtim; Windows LastWriteTime, in nanoseconds.
  std::int64_t modified_ns;
  /// POSIX st_ctim; Windows ChangeTime, in nanoseconds. Volumes without a
  /// change time (FAT, exFAT) report the last-write time instead.
  std::int64_t changed_ns;
  /// POSIX S_ISREG; Windows file rather than directory.
  bool regular;
  /// POSIX st_nlink; Windows hard-link count.
  std::uint32_t links;
  /// POSIX st_uid; Windows always 0 (no owner model).
  std::uint32_t user_id;
  /// POSIX st_mode & 07777; Windows always 0.
  std::uint32_t permissions;

  bool operator==(const FileInfo&) const = default;
};

/// Outcome of RenameNoReplace.
enum class RenameResult { kRenamed, kTargetExists, kFailed };

/// Open modes for OpenAt(). All modes open relative to a directory handle,
/// refuse symlinks, and set close-on-exec. Create modes write only and use
/// 0600 permissions; kOpenOrCreate reads and writes.
enum class FileMode {
  kRead,          ///< O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC
  kCreateNew,     ///< O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC
  kOpenOrCreate,  ///< O_RDWR | O_CREAT | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC
};

/// Move-only handle to an open directory. POSIX holds a directory file
/// descriptor. Windows holds the path and resolves names against it at open
/// time, because CreateFileW has no relative-open mode; Path() supplies the
/// prefix. Every operation that accepts a Directory treats `name` as a single
/// component and never resolves ".." out of the handle.
class Directory {
public:
  Directory() = default;
  Directory(const Directory&) = delete;
  Directory& operator=(const Directory&) = delete;
  Directory(Directory&& other) noexcept;
  Directory& operator=(Directory&& other) noexcept;
  ~Directory();

  /// Opens `path` as a directory, refusing a symlink final component.
  /// Returns false and sets `ec` on failure.
  bool Open(const std::filesystem::path& path, std::error_code& ec) noexcept;

  bool valid() const noexcept { return valid_; }
  const std::filesystem::path& Path() const noexcept { return path_; }

private:
  friend PlatformFile OpenAt(const Directory&, std::string_view, FileMode,
                             std::error_code&) noexcept;
  friend Directory OpenDirectoryAt(const Directory&, std::string_view,
                                   std::error_code&) noexcept;
  friend bool StatAt(const Directory&, std::string_view, bool, FileInfo*,
                     std::error_code&) noexcept;
  friend RenameResult RenameNoReplace(const Directory&, std::string_view,
                                      const Directory&, std::string_view,
                                      std::error_code&) noexcept;
  friend bool Rename(const Directory&, std::string_view, const Directory&,
                     std::string_view, std::error_code&) noexcept;
  friend bool Unlink(const Directory&, std::string_view,
                     std::error_code&) noexcept;
  friend bool MakeDirectoryAt(const Directory&, std::string_view,
                              std::error_code&) noexcept;
  friend bool SyncDirectory(const Directory&, std::error_code&) noexcept;
  friend bool SyncAncestors(const Directory&, std::error_code&) noexcept;
  friend bool ListDirectory(
      const Directory&,
      const std::function<void(std::string_view, const FileInfo&)>&,
      std::error_code&);
  friend bool TouchAt(const Directory&, std::string_view) noexcept;
  friend bool LockDirectoryShared(const Directory&) noexcept;
  friend bool TryLockDirectoryExclusive(const Directory&) noexcept;
  friend void UnlockDirectory(const Directory&) noexcept;

  void Close() noexcept;

  bool valid_ = false;
  std::filesystem::path path_;
#if !defined(_WIN32)
  int fd_ = -1;  // POSIX directory descriptor; Windows resolves by path.
#endif
};

/// Opens `name` inside `dir` with `mode`. Returns kInvalidFile on failure.
PlatformFile OpenAt(const Directory& dir, std::string_view name, FileMode mode,
                    std::error_code& ec) noexcept;

/// Opens `path` for appending, creating it with owner-only permissions.
/// Unlike OpenAt it follows symlinks and accepts devices and pipes, so
/// user-chosen sinks such as /dev/stdout work. The handle may also be
/// truncated (Truncate) to cut back a torn record. Returns kInvalidFile and
/// sets `ec` on failure.
PlatformFile OpenForAppend(const std::filesystem::path& path,
                           std::error_code& ec) noexcept;

/// Opens the subdirectory `name` of `dir`; `name` may be "." or "..".
/// Returns an invalid Directory on failure.
Directory OpenDirectoryAt(const Directory& dir, std::string_view name,
                          std::error_code& ec) noexcept;

/// Stats an open file. Returns false and sets `ec` on failure.
bool Stat(PlatformFile file, FileInfo* out, std::error_code& ec) noexcept;

/// Stats `name` inside `dir` without following symlinks unless
/// `follow_symlinks` is set. Returns false and sets `ec` on failure.
bool StatAt(const Directory& dir, std::string_view name, bool follow_symlinks,
            FileInfo* out, std::error_code& ec) noexcept;

/// Moves `from_name` in `from_dir` to `to_name` in `to_dir`, refusing to
/// replace an existing target. kTargetExists reports EEXIST.
RenameResult RenameNoReplace(const Directory& from_dir,
                             std::string_view from_name,
                             const Directory& to_dir, std::string_view to_name,
                             std::error_code& ec) noexcept;

/// Moves `from_name` in `from_dir` to `to_name` in `to_dir`, replacing an
/// existing target, like rename(2).
bool Rename(const Directory& from_dir, std::string_view from_name,
            const Directory& to_dir, std::string_view to_name,
            std::error_code& ec) noexcept;

/// Removes `name` from `dir`. Returns false and sets `ec` on failure.
bool Unlink(const Directory& dir, std::string_view name,
            std::error_code& ec) noexcept;

/// Creates directory `path` with owner-only permissions (0700). An
/// existing directory reports success. Returns false and sets `ec` on any
/// other failure.
bool MakeDirectory(const std::filesystem::path& path,
                   std::error_code& ec) noexcept;

/// Creates subdirectory `name` inside `dir` with owner-only permissions
/// (0700). An existing directory reports success.
bool MakeDirectoryAt(const Directory& dir, std::string_view name,
                     std::error_code& ec) noexcept;

/// Takes an exclusive non-blocking advisory lock on the whole file. Returns
/// false when another process holds the lock or the call fails, with `ec`
/// set either way.
bool TryLockExclusive(PlatformFile file, std::error_code& ec) noexcept;

/// Flushes file data to storage, retrying on EINTR. Windows requires the
/// handle to carry write access. Returns false and sets `ec` on failure.
bool SyncFile(PlatformFile file, std::error_code& ec) noexcept;

/// Flushes a directory's entries to storage. A no-op that returns true on
/// platforms without a directory flush. Returns false and sets `ec` on
/// failure.
bool SyncDirectory(const Directory& dir, std::error_code& ec) noexcept;

/// Flushes `dir` and every ancestor directory up to the filesystem root, so
/// entries created beneath `dir` survive a crash. Returns false and sets
/// `ec` on failure.
bool SyncAncestors(const Directory& dir, std::error_code& ec) noexcept;

/// Page-cache hints for an open file. Both are best effort: platforms
/// without the call, or ranges the OS refuses, are silently ignored.
enum class AdviseHint { kWillNeed, kDontNeed };
void Advise(PlatformFile file, AdviseHint hint, std::uint64_t offset,
            std::uint64_t length) noexcept;

/// Calls `visit(name, info)` for every entry of `dir` except "." and "..".
/// `info` describes the entry without following symlinks. Returns false and
/// sets `ec` when the directory cannot be read or an entry cannot be stat'd.
/// Not noexcept: the visitor runs subsystem code that may throw.
bool ListDirectory(
    const Directory& dir,
    const std::function<void(std::string_view, const FileInfo&)>& visit,
    std::error_code& ec);

/// Base directory for per-user caches: XDG_CACHE_HOME, then $HOME/.cache,
/// then Windows %LOCALAPPDATA%. Returns an empty path when the platform
/// cannot determine one; callers without a cache then skip caching.
std::filesystem::path UserCacheDirectory() noexcept;

/// Writes to the current file position. Clamps to INT_MAX bytes per call so
/// multi-gigabyte spans cannot truncate on 32-bit length limits. Returns the
/// bytes written; 0 with `ec` set on failure, including EINTR.
std::size_t Write(PlatformFile file, const void* data, std::size_t bytes,
                  std::error_code& ec) noexcept;

/// Reads from the current file position. Clamps to INT_MAX bytes per call.
/// Returns the bytes read; 0 with a clear `ec` reports end of file, 0 with
/// `ec` set reports failure, including EINTR.
std::size_t Read(PlatformFile file, void* output, std::size_t bytes,
                 std::error_code& ec) noexcept;

/// Reads up to `bytes` at `offset` in one call. Returns the bytes read; 0
/// with `ec` set on failure, including EINTR. On Windows the call moves the
/// file position; nothing may mix Read and ReadAt on one handle.
std::size_t ReadAt(PlatformFile file, std::uint64_t offset, void* output,
                   std::size_t bytes, std::error_code& ec) noexcept;

/// Moves the read/write position to `offset` bytes from the beginning of the
/// file. Returns false and sets `ec` on failure.
bool SeekTo(PlatformFile file, std::uint64_t offset,
            std::error_code& ec) noexcept;

/// Moves the read/write position to the end of the file. Returns the file
/// size in bytes, or 0 with `ec` set on failure.
std::uint64_t SeekToEnd(PlatformFile file, std::error_code& ec) noexcept;

/// Updates the access and modification times of `name` in `dir` to the current
/// time without following symlinks.
bool TouchAt(const Directory& dir, std::string_view name) noexcept;

/// Advisory directory locks used to coordinate staging vs. orphan sweeps.
bool LockDirectoryShared(const Directory& dir) noexcept;
bool TryLockDirectoryExclusive(const Directory& dir) noexcept;
void UnlockDirectory(const Directory& dir) noexcept;

/// Opens a writable stdio stream backed by `[buffer, buffer + capacity)`
/// (`fmemopen` on POSIX, temporary file on Windows). Finalize with
/// FinishMemoryWriteStream().
std::FILE* OpenMemoryWriteStream(void* buffer, std::size_t capacity) noexcept;
bool FinishMemoryWriteStream(std::FILE* stream, void* buffer,
                             std::size_t expected_bytes) noexcept;

/// Opens a readable stdio stream over `[buffer, buffer + bytes)` (`fmemopen`
/// on POSIX, temporary file on Windows). Close with `std::fclose`.
std::FILE* OpenMemoryReadStream(const void* buffer, std::size_t bytes) noexcept;

/// Sets the file size to `bytes`, shrinking or growing like ftruncate(2).
/// Returns false and sets `ec` on failure.
bool Truncate(PlatformFile file, std::uint64_t bytes,
              std::error_code& ec) noexcept;

/// Owner-consistency policy for files written by this user. POSIX requires
/// the file to belong to the current user and to carry none of the
/// `forbidden` permission bits (for example 0022 rejects group/other
/// writes). Always true on Windows until ACL checks exist.
bool OwnerOnly(const FileInfo& info, std::uint32_t forbidden,
               std::uint32_t required = 0) noexcept;

/// Makes an existing directory owner-only: chmod(path, 0700) on POSIX, so
/// its entries in a shared temporary root cannot be listed or entered by
/// other users. A no-op that reports success on Windows, where access is
/// left to the directory's inherited ACLs. Returns false on failure.
bool RestrictToOwner(const std::filesystem::path& path) noexcept;

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_FS_HPP_
