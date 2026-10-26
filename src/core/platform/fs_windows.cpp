#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <cstdlib>
#include <cwchar>
#include <limits>
#include <string>
#include <utility>

#include "src/core/platform/fs.hpp"

#ifndef FILE_FLAG_OPEN_REPARSE_POINT
#define FILE_FLAG_OPEN_REPARSE_POINT 0x00200000
#endif

namespace gufo::platform {
namespace {

constexpr std::size_t kMaxTransfer = 0x7fffffff;  // INT_MAX per call.

constexpr DWORD kDirectoryShare =
    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;

// Maps the Windows errors callers branch on onto the generic category so
// they can compare against std::errc; everything else keeps the system
// category.
std::error_code Win32Error(DWORD code) noexcept {
  switch (code) {
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
    case ERROR_INVALID_NAME:
    case ERROR_DELETE_PENDING:
      return std::make_error_code(std::errc::no_such_file_or_directory);
    case ERROR_ALREADY_EXISTS:
    case ERROR_FILE_EXISTS:
      return std::make_error_code(std::errc::file_exists);
    case ERROR_ACCESS_DENIED:
      return std::make_error_code(std::errc::permission_denied);
    case ERROR_SHARING_VIOLATION:
      return std::make_error_code(std::errc::device_or_resource_busy);
    case ERROR_INVALID_PARAMETER:
    case ERROR_INVALID_FUNCTION:
      return std::make_error_code(std::errc::invalid_argument);
    case ERROR_NOT_SUPPORTED:
    case ERROR_CALL_NOT_IMPLEMENTED:
      return std::make_error_code(std::errc::operation_not_supported);
    case ERROR_LOCK_VIOLATION:
      return std::make_error_code(std::errc::resource_unavailable_try_again);
    default:
      return {static_cast<int>(code), std::system_category()};
  }
}

std::error_code LastError() noexcept {
  return Win32Error(GetLastError());
}

bool IsReparsePoint(HANDLE handle) noexcept {
  BY_HANDLE_FILE_INFORMATION info{};
  if (!GetFileInformationByHandle(handle, &info))
    return false;
  return (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
}

bool FromHandle(HANDLE handle, FileInfo* out) noexcept {
  BY_HANDLE_FILE_INFORMATION info{};
  if (handle == nullptr || handle == INVALID_HANDLE_VALUE ||
      !GetFileInformationByHandle(handle, &info))
    return false;
  FileInfo& result = *out;
  result.device = info.dwVolumeSerialNumber;
  result.inode = (static_cast<std::uint64_t>(info.nFileIndexHigh) << 32) |
                 info.nFileIndexLow;
  result.size = (static_cast<std::uint64_t>(info.nFileSizeHigh) << 32) |
                info.nFileSizeLow;
  result.regular = (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
  result.links = info.nNumberOfLinks;
  FILE_BASIC_INFO basic{};
  if (GetFileInformationByHandleEx(handle, FileBasicInfo, &basic,
                                   sizeof(basic))) {
    // NTFS carries a change time; volumes without one fall back to the
    // last-write time so the identity still moves on every write. Both
    // convert to Unix-epoch nanoseconds like the POSIX fields.
    constexpr std::int64_t kEpochOffset100ns = 116444736000000000;
    const std::int64_t changed = basic.ChangeTime.QuadPart != 0
                                     ? basic.ChangeTime.QuadPart
                                     : basic.LastWriteTime.QuadPart;
    result.modified_ns =
        (basic.LastWriteTime.QuadPart - kEpochOffset100ns) * 100;
    result.changed_ns = (changed - kEpochOffset100ns) * 100;
  }
  return true;
}

// Opens the final component without following it, so symlink refusal is
// exact. A reparse point opens as itself and is rejected by the caller.
HANDLE OpenNoFollow(const std::filesystem::path& path, DWORD access,
                    DWORD flags) noexcept {
  return CreateFileW(
      path.c_str(), access, kDirectoryShare, nullptr, OPEN_EXISTING,
      flags | FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
      nullptr);
}

// Joins a directory-anchored name; `name` stays a single component.
std::filesystem::path Join(const Directory& dir, std::string_view name) {
  return dir.Path() / std::string(name);
}

}  // namespace

Directory::Directory(Directory&& other) noexcept
    : valid_(other.valid_), path_(std::move(other.path_)) {
  other.valid_ = false;
}

Directory& Directory::operator=(Directory&& other) noexcept {
  if (this != &other) {
    Close();
    valid_ = std::exchange(other.valid_, false);
    path_ = std::move(other.path_);
  }
  return *this;
}

Directory::~Directory() {
  Close();
}

void Directory::Close() noexcept {
  valid_ = false;
}

bool Directory::Open(const std::filesystem::path& path,
                     std::error_code& ec) noexcept {
  ec.clear();
  Close();
  // Windows holds no directory handle: relative opens resolve against the
  // stored path. The probe validates the directory and refuses symlinks.
  HANDLE probe = OpenNoFollow(path, FILE_READ_ATTRIBUTES, 0);
  if (probe == INVALID_HANDLE_VALUE) {
    ec = LastError();
    return false;
  }
  BY_HANDLE_FILE_INFORMATION info{};
  if (!GetFileInformationByHandle(probe, &info)) {
    ec = LastError();
    CloseHandle(probe);
    return false;
  }
  CloseHandle(probe);
  if (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) {
    ec = std::make_error_code(std::errc::too_many_symbolic_link_levels);
    return false;
  }
  if (!(info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
    ec = std::make_error_code(std::errc::not_a_directory);
    return false;
  }
  valid_ = true;
  path_ = path;
  return true;
}

PlatformFile OpenAt(const Directory& dir, std::string_view name, FileMode mode,
                    std::error_code& ec) noexcept {
  ec.clear();
  const auto path = Join(dir, name);
  DWORD access = 0, disposition = 0, flags = FILE_FLAG_OPEN_REPARSE_POINT;
  switch (mode) {
    case FileMode::kRead:
      access = GENERIC_READ;
      disposition = OPEN_EXISTING;
      break;
    case FileMode::kCreateNew:
      access = GENERIC_WRITE;
      disposition = CREATE_NEW;
      break;
    case FileMode::kOpenOrCreate:
      access = GENERIC_READ | GENERIC_WRITE;
      disposition = OPEN_ALWAYS;
      break;
  }
  HANDLE file = CreateFileW(path.c_str(), access, kDirectoryShare, nullptr,
                            disposition, flags, nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    ec = LastError();
    return kInvalidFile;
  }
  if (IsReparsePoint(file)) {
    // O_NOFOLLOW: a symlink final component never opens.
    CloseHandle(file);
    ec = std::make_error_code(std::errc::too_many_symbolic_link_levels);
    return kInvalidFile;
  }
  return file;
}

Directory OpenDirectoryAt(const Directory& dir, std::string_view name,
                          std::error_code& ec) noexcept {
  ec.clear();
  Directory result;
  if (!result.Open(Join(dir, name), ec))
    return Directory{};
  return result;
}

bool Stat(PlatformFile file, FileInfo* out, std::error_code& ec) noexcept {
  ec.clear();
  if (!FileValid(file) || !FromHandle(static_cast<HANDLE>(file), out)) {
    ec = std::make_error_code(std::errc::bad_file_descriptor);
    return false;
  }
  return true;
}

bool StatAt(const Directory& dir, std::string_view name, bool follow_symlinks,
            FileInfo* out, std::error_code& ec) noexcept {
  ec.clear();
  const auto path = Join(dir, name);
  HANDLE probe = follow_symlinks
                     ? CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                                   kDirectoryShare, nullptr, OPEN_EXISTING,
                                   FILE_FLAG_BACKUP_SEMANTICS, nullptr)
                     : OpenNoFollow(path, FILE_READ_ATTRIBUTES, 0);
  if (probe == INVALID_HANDLE_VALUE) {
    ec = LastError();
    return false;
  }
  if (!follow_symlinks && IsReparsePoint(probe)) {
    CloseHandle(probe);
    ec = std::make_error_code(std::errc::too_many_symbolic_link_levels);
    return false;
  }
  if (!FromHandle(probe, out)) {
    ec = LastError();
    CloseHandle(probe);
    return false;
  }
  CloseHandle(probe);
  return true;
}

RenameResult RenameNoReplace(const Directory& from_dir,
                             std::string_view from_name,
                             const Directory& to_dir, std::string_view to_name,
                             std::error_code& ec) noexcept {
  ec.clear();
  // MOVEFILE_WRITE_THROUGH without MOVEFILE_REPLACE_EXISTING: an existing
  // target must never be overwritten.
  if (MoveFileExW(Join(from_dir, from_name).c_str(),
                  Join(to_dir, to_name).c_str(), MOVEFILE_WRITE_THROUGH) == 0) {
    ec = LastError();
    return ec == std::errc::file_exists ? RenameResult::kTargetExists
                                        : RenameResult::kFailed;
  }
  return RenameResult::kRenamed;
}

bool Rename(const Directory& from_dir, std::string_view from_name,
            const Directory& to_dir, std::string_view to_name,
            std::error_code& ec) noexcept {
  ec.clear();
  if (MoveFileExW(Join(from_dir, from_name).c_str(),
                  Join(to_dir, to_name).c_str(),
                  MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) == 0) {
    ec = LastError();
    return false;
  }
  return true;
}

bool Unlink(const Directory& dir, std::string_view name,
            std::error_code& ec) noexcept {
  ec.clear();
  if (!DeleteFileW(Join(dir, name).c_str())) {
    ec = LastError();
    return false;
  }
  return true;
}

bool MakeDirectory(const std::filesystem::path& path,
                   std::error_code& ec) noexcept {
  ec.clear();
  if (!CreateDirectoryW(path.c_str(), nullptr) &&
      GetLastError() != ERROR_ALREADY_EXISTS) {
    ec = LastError();
    return false;
  }
  return true;
}

bool MakeDirectoryAt(const Directory& dir, std::string_view name,
                     std::error_code& ec) noexcept {
  return MakeDirectory(Join(dir, name), ec);
}

bool TryLockExclusive(PlatformFile file, std::error_code& ec) noexcept {
  ec.clear();
  if (!FileValid(file)) {
    ec = std::make_error_code(std::errc::bad_file_descriptor);
    return false;
  }
  OVERLAPPED overlapped{};
  if (!LockFileEx(static_cast<HANDLE>(file),
                  LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0,
                  &overlapped)) {
    ec = LastError();
    return false;
  }
  return true;
}

bool SyncFile(PlatformFile file, std::error_code& ec) noexcept {
  ec.clear();
  if (!FileValid(file)) {
    ec = std::make_error_code(std::errc::bad_file_descriptor);
    return false;
  }
  if (!FlushFileBuffers(static_cast<HANDLE>(file))) {
    ec = LastError();
    return false;
  }
  return true;
}

bool SyncDirectory(const Directory&, std::error_code& ec) noexcept {
  // No portable directory flush on Windows; entry durability follows the
  // file-level flushes.
  ec.clear();
  return true;
}

bool SyncAncestors(const Directory&, std::error_code& ec) noexcept {
  ec.clear();
  return true;
}

void Advise(PlatformFile, AdviseHint, std::uint64_t, std::uint64_t) noexcept {
  // Windows exposes no page-cache hint for a file handle.
}

bool ListDirectory(
    const Directory& dir,
    const std::function<void(std::string_view, const FileInfo&)>& visit,
    std::error_code& ec) {
  ec.clear();
  WIN32_FIND_DATAW data{};
  HANDLE find = FindFirstFileExW((dir.Path() / L"*").c_str(), FindExInfoBasic,
                                 &data, FindExSearchNameMatch, nullptr, 0);
  if (find == INVALID_HANDLE_VALUE) {
    ec = LastError();
    return false;
  }
  do {
    const auto wide = static_cast<int>(wcslen(data.cFileName));
    std::string name(WideCharToMultiByte(CP_UTF8, 0, data.cFileName, wide,
                                         nullptr, 0, nullptr, nullptr),
                     '\0');
    WideCharToMultiByte(CP_UTF8, 0, data.cFileName, wide, name.data(),
                        static_cast<int>(name.size()), nullptr, nullptr);
    if (name == "." || name == "..")
      continue;
    FileInfo info{};
    if (data.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) {
      // Report a link like POSIX lstat: present, never regular, so callers
      // skip it; a link carries no file identity of its own.
      info.size = (static_cast<std::uint64_t>(data.nFileSizeHigh) << 32) |
                  data.nFileSizeLow;
    } else if (!StatAt(dir, name, false, &info, ec)) {
      // File identity needs a handle; directory enumeration cannot supply
      // it for regular entries.
      FindClose(find);
      return false;
    }
    visit(name, info);
  } while (FindNextFileW(find, &data));
  const DWORD code = GetLastError();
  FindClose(find);
  if (code != ERROR_NO_MORE_FILES) {
    ec = Win32Error(code);
    return false;
  }
  return true;
}

std::filesystem::path UserCacheDirectory() noexcept {
  const char* base = std::getenv("LOCALAPPDATA");
  if (base == nullptr || *base == '\0')
    return {};
  return base;
}

std::size_t Write(PlatformFile file, const void* data, std::size_t bytes,
                  std::error_code& ec) noexcept {
  ec.clear();
  if (!FileValid(file)) {
    ec = std::make_error_code(std::errc::bad_file_descriptor);
    return 0;
  }
  DWORD written = 0;
  const DWORD request = bytes > kMaxTransfer ? static_cast<DWORD>(kMaxTransfer)
                                             : static_cast<DWORD>(bytes);
  if (!WriteFile(static_cast<HANDLE>(file), data, request, &written, nullptr)) {
    ec = LastError();
    return 0;
  }
  return written;
}

std::size_t Read(PlatformFile file, void* output, std::size_t bytes,
                 std::error_code& ec) noexcept {
  ec.clear();
  if (!FileValid(file)) {
    ec = std::make_error_code(std::errc::bad_file_descriptor);
    return 0;
  }
  // No OVERLAPPED: the call reads from and advances the file position.
  DWORD read_bytes = 0;
  const DWORD request = bytes > kMaxTransfer ? static_cast<DWORD>(kMaxTransfer)
                                             : static_cast<DWORD>(bytes);
  if (!ReadFile(static_cast<HANDLE>(file), output, request, &read_bytes,
                nullptr)) {
    const DWORD code = GetLastError();
    if (code == ERROR_HANDLE_EOF)
      return 0;  // Clean end of file.
    ec = Win32Error(code);
    return 0;
  }
  return read_bytes;
}

std::size_t ReadAt(PlatformFile file, std::uint64_t offset, void* output,
                   std::size_t bytes, std::error_code& ec) noexcept {
  ec.clear();
  if (!FileValid(file)) {
    ec = std::make_error_code(std::errc::bad_file_descriptor);
    return 0;
  }
  OVERLAPPED overlapped{};
  overlapped.Offset = static_cast<DWORD>(offset);
  overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
  DWORD read_bytes = 0;
  const DWORD request = bytes > kMaxTransfer ? static_cast<DWORD>(kMaxTransfer)
                                             : static_cast<DWORD>(bytes);
  if (!ReadFile(static_cast<HANDLE>(file), output, request, &read_bytes,
                &overlapped)) {
    const DWORD code = GetLastError();
    if (code == ERROR_HANDLE_EOF)
      return 0;  // Clean end of file.
    ec = Win32Error(code);
    return 0;
  }
  return read_bytes;
}

bool OwnerOnly(const FileInfo&, std::uint32_t, std::uint32_t) noexcept {
  // No owner model until ACL checks exist.
  return true;
}

PlatformFile OpenForAppend(const std::filesystem::path& path,
                           std::error_code& ec) noexcept {
  ec.clear();
  // FILE_WRITE_DATA lets Truncate cut a torn record back; callers seek to
  // the end before every write, so appends still land at the end.
  HANDLE file = CreateFileW(path.c_str(), FILE_APPEND_DATA | FILE_WRITE_DATA,
                            kDirectoryShare, nullptr, OPEN_ALWAYS,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    ec = LastError();
    return kInvalidFile;
  }
  return file;
}

std::uint64_t SeekToEnd(PlatformFile file, std::error_code& ec) noexcept {
  ec.clear();
  LARGE_INTEGER distance{};
  LARGE_INTEGER end{};
  if (!SetFilePointerEx(file, distance, &end, FILE_END)) {
    ec = LastError();
    return 0;
  }
  return static_cast<std::uint64_t>(end.QuadPart);
}

bool SeekTo(PlatformFile file, std::uint64_t offset,
            std::error_code& ec) noexcept {
  ec.clear();
  LARGE_INTEGER distance{};
  distance.QuadPart = static_cast<LONGLONG>(offset);
  if (!SetFilePointerEx(file, distance, nullptr, FILE_BEGIN)) {
    ec = LastError();
    return false;
  }
  return true;
}

bool TouchAt(const Directory& dir, std::string_view name) noexcept {
  HANDLE handle = OpenNoFollow(Join(dir, name), FILE_WRITE_ATTRIBUTES, 0);
  if (handle == INVALID_HANDLE_VALUE)
    return false;
  if (IsReparsePoint(handle)) {
    CloseHandle(handle);
    return false;
  }
  FILETIME now{};
  GetSystemTimeAsFileTime(&now);
  const bool ok = SetFileTime(handle, nullptr, &now, &now) != 0;
  CloseHandle(handle);
  return ok;
}

bool LockDirectoryShared(const Directory& dir) noexcept {
  return dir.valid();
}

bool TryLockDirectoryExclusive(const Directory& dir) noexcept {
  return dir.valid();
}

void UnlockDirectory(const Directory&) noexcept {}

std::FILE* OpenMemoryWriteStream(void*, std::size_t) noexcept {
  return std::tmpfile();
}

bool FinishMemoryWriteStream(std::FILE* stream, void* buffer,
                             std::size_t expected_bytes) noexcept {
  if (stream == nullptr)
    return false;
  const bool ok =
      std::fseek(stream, 0, SEEK_SET) == 0 &&
      std::fread(buffer, 1, expected_bytes, stream) == expected_bytes;
  const bool closed = std::fclose(stream) == 0;
  return ok && closed;
}

std::FILE* OpenMemoryReadStream(const void* buffer,
                                std::size_t bytes) noexcept {
  std::FILE* stream = std::tmpfile();
  if (stream == nullptr)
    return nullptr;
  if (std::fwrite(buffer, 1, bytes, stream) != bytes ||
      std::fseek(stream, 0, SEEK_SET) != 0) {
    std::fclose(stream);
    return nullptr;
  }
  return stream;
}

bool Truncate(PlatformFile file, std::uint64_t bytes,
              std::error_code& ec) noexcept {
  ec.clear();
  LARGE_INTEGER distance{};
  distance.QuadPart = static_cast<LONGLONG>(bytes);
  if (!SetFilePointerEx(file, distance, nullptr, FILE_BEGIN) ||
      !SetEndOfFile(file)) {
    ec = LastError();
    return false;
  }
  return true;
}

bool RestrictToOwner(const std::filesystem::path&) noexcept {
  // No-op: access follows the directory's inherited ACLs.
  return true;
}

}  // namespace gufo::platform
