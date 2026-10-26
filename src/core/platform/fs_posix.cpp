#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <memory>
#include <utility>

#include "src/core/platform/fs.hpp"

namespace gufo::platform {
namespace {

constexpr std::size_t kMaxTransfer = 0x7ffff000;  // Linux write() cap.

std::error_code Errno() noexcept {
  return {errno, std::generic_category()};
}

std::int64_t ToNs(const timespec& ts) noexcept {
  return static_cast<std::int64_t>(ts.tv_sec) * 1'000'000'000 +
         static_cast<std::int64_t>(ts.tv_nsec);
}

FileInfo FromStat(const struct stat& st) noexcept {
  return {static_cast<std::uint64_t>(st.st_dev),
          static_cast<std::uint64_t>(st.st_ino),
          static_cast<std::uint64_t>(st.st_size),
          ToNs(st.st_mtim),
          ToNs(st.st_ctim),
          S_ISREG(st.st_mode) != 0,
          static_cast<std::uint32_t>(st.st_nlink),
          static_cast<std::uint32_t>(st.st_uid),
          static_cast<std::uint32_t>(st.st_mode & 07777)};
}

int OpenFlags(FileMode mode) noexcept {
  switch (mode) {
    case FileMode::kRead:
      // O_NONBLOCK keeps a FIFO carrying a cache file's name from blocking
      // the open; regular files ignore it.
      return O_RDONLY | O_NONBLOCK | O_CLOEXEC | O_NOFOLLOW;
    case FileMode::kCreateNew:
      return O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW;
    case FileMode::kOpenOrCreate:
      return O_RDWR | O_CREAT | O_NONBLOCK | O_CLOEXEC | O_NOFOLLOW;
  }
  return 0;
}

int DirectoryFlags() noexcept {
  return O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW;
}

int SyncLoop(int fd) noexcept {
  int result;
  do {
    result = fsync(fd);
  } while (result < 0 && errno == EINTR);
  return result;
}

}  // namespace

Directory::Directory(Directory&& other) noexcept
    : valid_(other.valid_), path_(std::move(other.path_)), fd_(other.fd_) {
  other.valid_ = false;
  other.fd_ = -1;
}

Directory& Directory::operator=(Directory&& other) noexcept {
  if (this != &other) {
    Close();
    valid_ = std::exchange(other.valid_, false);
    path_ = std::move(other.path_);
    fd_ = std::exchange(other.fd_, -1);
  }
  return *this;
}

Directory::~Directory() {
  Close();
}

void Directory::Close() noexcept {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
  valid_ = false;
}

bool Directory::Open(const std::filesystem::path& path,
                     std::error_code& ec) noexcept {
  ec.clear();
  Close();
  const int fd = ::open(path.c_str(), DirectoryFlags());
  if (fd < 0) {
    ec = Errno();
    return false;
  }
  valid_ = true;
  path_ = path;
  fd_ = fd;
  return true;
}

PlatformFile OpenAt(const Directory& dir, std::string_view name, FileMode mode,
                    std::error_code& ec) noexcept {
  ec.clear();
  const int fd = ::openat(dir.fd_, std::string(name).c_str(), OpenFlags(mode),
                          mode == FileMode::kRead ? 0 : 0600);
  if (fd < 0) {
    ec = Errno();
    return kInvalidFile;
  }
  return fd;
}

Directory OpenDirectoryAt(const Directory& dir, std::string_view name,
                          std::error_code& ec) noexcept {
  ec.clear();
  Directory result;
  const int fd = ::openat(dir.fd_, std::string(name).c_str(), DirectoryFlags());
  if (fd < 0) {
    ec = Errno();
    return result;
  }
  result.valid_ = true;
  result.path_ = dir.path_ / std::string(name);
  result.fd_ = fd;
  return result;
}

bool Stat(PlatformFile file, FileInfo* out, std::error_code& ec) noexcept {
  ec.clear();
  struct stat st{};
  if (::fstat(file, &st) != 0) {
    ec = Errno();
    return false;
  }
  *out = FromStat(st);
  return true;
}

bool StatAt(const Directory& dir, std::string_view name, bool follow_symlinks,
            FileInfo* out, std::error_code& ec) noexcept {
  ec.clear();
  struct stat st{};
  const int flags = follow_symlinks ? 0 : AT_SYMLINK_NOFOLLOW;
  if (::fstatat(dir.fd_, std::string(name).c_str(), &st, flags) != 0) {
    ec = Errno();
    return false;
  }
  *out = FromStat(st);
  return true;
}

RenameResult RenameNoReplace(const Directory& from_dir,
                             std::string_view from_name,
                             const Directory& to_dir, std::string_view to_name,
                             std::error_code& ec) noexcept {
  ec.clear();
  const int result = static_cast<int>(
      ::syscall(SYS_renameat2, from_dir.fd_, std::string(from_name).c_str(),
                to_dir.fd_, std::string(to_name).c_str(), RENAME_NOREPLACE));
  if (result == 0)
    return RenameResult::kRenamed;
  ec = Errno();
  return ec == std::errc::file_exists ? RenameResult::kTargetExists
                                      : RenameResult::kFailed;
}

bool Rename(const Directory& from_dir, std::string_view from_name,
            const Directory& to_dir, std::string_view to_name,
            std::error_code& ec) noexcept {
  ec.clear();
  if (::renameat(from_dir.fd_, std::string(from_name).c_str(), to_dir.fd_,
                 std::string(to_name).c_str()) != 0) {
    ec = Errno();
    return false;
  }
  return true;
}

bool Unlink(const Directory& dir, std::string_view name,
            std::error_code& ec) noexcept {
  ec.clear();
  if (::unlinkat(dir.fd_, std::string(name).c_str(), 0) != 0) {
    ec = Errno();
    return false;
  }
  return true;
}

bool MakeDirectory(const std::filesystem::path& path,
                   std::error_code& ec) noexcept {
  ec.clear();
  if (::mkdir(path.c_str(), 0700) != 0 && errno != EEXIST) {
    ec = Errno();
    return false;
  }
  return true;
}

bool MakeDirectoryAt(const Directory& dir, std::string_view name,
                     std::error_code& ec) noexcept {
  ec.clear();
  if (::mkdirat(dir.fd_, std::string(name).c_str(), 0700) != 0 &&
      errno != EEXIST) {
    ec = Errno();
    return false;
  }
  return true;
}

bool TryLockExclusive(PlatformFile file, std::error_code& ec) noexcept {
  ec.clear();
  if (::flock(file, LOCK_EX | LOCK_NB) != 0) {
    ec = Errno();
    return false;
  }
  return true;
}

bool SyncFile(PlatformFile file, std::error_code& ec) noexcept {
  ec.clear();
  if (SyncLoop(file) < 0) {
    ec = Errno();
    return false;
  }
  return true;
}

bool SyncDirectory(const Directory& dir, std::error_code& ec) noexcept {
  ec.clear();
  if (SyncLoop(dir.fd_) < 0) {
    ec = Errno();
    return false;
  }
  return true;
}

bool SyncAncestors(const Directory& dir, std::error_code& ec) noexcept {
  ec.clear();
  int ancestor = ::openat(dir.fd_, "..", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  for (;;) {
    if (ancestor < 0) {
      ec = Errno();
      return false;
    }
    const int synced = SyncLoop(ancestor);
    const int parent =
        ::openat(ancestor, "..", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    struct stat a{}, b{};
    const bool stat_ok =
        parent >= 0 && ::fstat(ancestor, &a) == 0 && ::fstat(parent, &b) == 0;
    const int parent_errno = errno;
    ::close(ancestor);
    if (synced < 0 || !stat_ok) {
      if (parent >= 0)
        ::close(parent);
      errno = parent_errno;
      ec = Errno();
      return false;
    }
    if (a.st_dev == b.st_dev && a.st_ino == b.st_ino) {
      ::close(parent);
      return true;
    }
    ancestor = parent;
  }
}

void Advise(PlatformFile file, AdviseHint hint, std::uint64_t offset,
            std::uint64_t length) noexcept {
  const auto max_offset =
      static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
  if (offset > max_offset)
    return;
  const auto clamped = std::min<std::uint64_t>(length, max_offset - offset);
  const int advice =
      hint == AdviseHint::kDontNeed ? POSIX_FADV_DONTNEED : POSIX_FADV_WILLNEED;
  (void)::posix_fadvise(file, static_cast<off_t>(offset),
                        static_cast<off_t>(clamped), advice);
}

bool ListDirectory(
    const Directory& dir,
    const std::function<void(std::string_view, const FileInfo&)>& visit,
    std::error_code& ec) {
  ec.clear();
  const int copy = ::openat(dir.fd_, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (copy < 0) {
    ec = Errno();
    return false;
  }
  auto close_directory = [](DIR* d) { closedir(d); };
  const std::unique_ptr<DIR, decltype(close_directory)> stream(fdopendir(copy),
                                                               close_directory);
  if (!stream) {
    ::close(copy);
    ec = Errno();
    return false;
  }
  for (;;) {
    errno = 0;
    const auto* entry = readdir(stream.get());
    if (entry == nullptr) {
      if (errno != 0) {
        ec = Errno();
        return false;
      }
      return true;
    }
    const std::string_view name(entry->d_name);
    if (name == "." || name == "..")
      continue;
    struct stat st{};
    if (::fstatat(dir.fd_, entry->d_name, &st, AT_SYMLINK_NOFOLLOW) != 0) {
      ec = Errno();
      return false;
    }
    visit(name, FromStat(st));
  }
}

std::filesystem::path UserCacheDirectory() noexcept {
  const char* base = std::getenv("XDG_CACHE_HOME");
  if (base != nullptr && *base != '\0')
    return base;
  const char* home = std::getenv("HOME");
  if (home == nullptr || *home == '\0')
    return {};
  return std::filesystem::path(home) / ".cache";
}

std::size_t Write(PlatformFile file, const void* data, std::size_t bytes,
                  std::error_code& ec) noexcept {
  ec.clear();
  const auto n = ::write(file, data, std::min(bytes, kMaxTransfer));
  if (n < 0) {
    ec = Errno();
    return 0;
  }
  return static_cast<std::size_t>(n);
}

std::size_t Read(PlatformFile file, void* output, std::size_t bytes,
                 std::error_code& ec) noexcept {
  ec.clear();
  const auto n = ::read(file, output, std::min(bytes, kMaxTransfer));
  if (n < 0) {
    ec = Errno();
    return 0;
  }
  return static_cast<std::size_t>(n);
}

std::size_t ReadAt(PlatformFile file, std::uint64_t offset, void* output,
                   std::size_t bytes, std::error_code& ec) noexcept {
  ec.clear();
  const auto n = ::pread(file, output, std::min(bytes, kMaxTransfer),
                         static_cast<off_t>(offset));
  if (n < 0) {
    ec = Errno();
    return 0;
  }
  return static_cast<std::size_t>(n);
}

bool OwnerOnly(const FileInfo& info, std::uint32_t forbidden,
               std::uint32_t required) noexcept {
  return info.user_id == ::geteuid() && (info.permissions & forbidden) == 0 &&
         (info.permissions & required) == required;
}

PlatformFile OpenForAppend(const std::filesystem::path& path,
                           std::error_code& ec) noexcept {
  ec.clear();
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC,
                        S_IRUSR | S_IWUSR);
  if (fd < 0)
    ec = Errno();
  return fd;
}

std::uint64_t SeekToEnd(PlatformFile file, std::error_code& ec) noexcept {
  ec.clear();
  const off_t end = ::lseek(file, 0, SEEK_END);
  if (end < 0) {
    ec = Errno();
    return 0;
  }
  return static_cast<std::uint64_t>(end);
}

bool SeekTo(PlatformFile file, std::uint64_t offset,
            std::error_code& ec) noexcept {
  ec.clear();
  if (::lseek(file, static_cast<off_t>(offset), SEEK_SET) !=
      static_cast<off_t>(offset)) {
    ec = Errno();
    return false;
  }
  return true;
}

bool TouchAt(const Directory& dir, std::string_view name) noexcept {
  return ::utimensat(dir.fd_, std::string(name).c_str(), nullptr,
                     AT_SYMLINK_NOFOLLOW) == 0;
}

bool LockDirectoryShared(const Directory& dir) noexcept {
  while (::flock(dir.fd_, LOCK_SH) != 0) {
    if (errno != EINTR)
      return false;
  }
  return true;
}

bool TryLockDirectoryExclusive(const Directory& dir) noexcept {
  while (::flock(dir.fd_, LOCK_EX | LOCK_NB) != 0) {
    if (errno != EINTR)
      return false;
  }
  return true;
}

void UnlockDirectory(const Directory& dir) noexcept {
  while (::flock(dir.fd_, LOCK_UN) != 0 && errno == EINTR) {
  }
}

std::FILE* OpenMemoryWriteStream(void* buffer, std::size_t capacity) noexcept {
  return ::fmemopen(buffer, capacity, "wb");
}

bool FinishMemoryWriteStream(std::FILE* stream, void*, std::size_t) noexcept {
  return stream != nullptr && std::fclose(stream) == 0;
}

std::FILE* OpenMemoryReadStream(const void* buffer,
                                std::size_t bytes) noexcept {
  return ::fmemopen(const_cast<void*>(buffer), bytes, "rb");
}

bool Truncate(PlatformFile file, std::uint64_t bytes,
              std::error_code& ec) noexcept {
  ec.clear();
  if (::ftruncate(file, static_cast<off_t>(bytes)) != 0) {
    ec = Errno();
    return false;
  }
  return true;
}

bool RestrictToOwner(const std::filesystem::path& path) noexcept {
  return ::chmod(path.c_str(), 0700) == 0;
}

}  // namespace gufo::platform
