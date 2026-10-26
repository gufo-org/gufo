#include "src/core/platform/mapped_file.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace gufo::platform {
namespace {

#if defined(_WIN32)
std::string FormatWindowsError(const char* what, DWORD code) {
  return std::string(what) + " failed with Windows error " +
         std::to_string(code);
}

HANDLE OpenForRead(const std::filesystem::path& path, DWORD flags) {
  return CreateFileW(path.c_str(), GENERIC_READ,
                     FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                     nullptr, OPEN_EXISTING, flags, nullptr);
}
#else
std::string FormatErrno(const char* what, int code) {
  return std::string(what) + ": " + std::strerror(code);
}
#endif

}  // namespace

ReadOnlyMappedFile::ReadOnlyMappedFile(ReadOnlyMappedFile&& other) noexcept
    : data_(other.data_),
      size_(other.size_),
#if defined(_WIN32)
      handle_(other.handle_),
      mapping_(other.mapping_)
#else
      fd_(other.fd_)
#endif
{
  other.Reset();
}

ReadOnlyMappedFile& ReadOnlyMappedFile::operator=(
    ReadOnlyMappedFile&& other) noexcept {
  if (this != &other) {
    Close();
    data_ = other.data_;
    size_ = other.size_;
#if defined(_WIN32)
    handle_ = other.handle_;
    mapping_ = other.mapping_;
#else
    fd_ = other.fd_;
#endif
    other.Reset();
  }
  return *this;
}

ReadOnlyMappedFile::~ReadOnlyMappedFile() {
  Close();
}

PlatformFile ReadOnlyMappedFile::PlatformHandle() const noexcept {
#if defined(_WIN32)
  return handle_;
#else
  return fd_;
#endif
}

void ReadOnlyMappedFile::Reset() noexcept {
  data_ = nullptr;
  size_ = 0;
#if defined(_WIN32)
  handle_ = nullptr;
  mapping_ = nullptr;
#else
  fd_ = -1;
#endif
}

void ReadOnlyMappedFile::Close() noexcept {
#if defined(_WIN32)
  if (data_ != nullptr)
    UnmapViewOfFile(data_);
  if (mapping_ != nullptr)
    CloseHandle(static_cast<HANDLE>(mapping_));
  if (handle_ != nullptr)
    CloseHandle(static_cast<HANDLE>(handle_));
#else
  if (data_ != nullptr)
    munmap(const_cast<std::uint8_t*>(data_), size_);
  if (fd_ >= 0)
    ::close(fd_);
#endif
  Reset();
}

ReadOnlyMappedFile ReadOnlyMappedFile::Open(const std::filesystem::path& path,
                                            std::string* error) {
  const auto fail = [&](std::string message) {
    if (error != nullptr)
      *error = std::move(message);
    return ReadOnlyMappedFile();
  };
#if defined(_WIN32)
  const HANDLE file = OpenForRead(path, FILE_FLAG_SEQUENTIAL_SCAN);
  if (file == INVALID_HANDLE_VALUE)
    return fail(FormatWindowsError("CreateFileW", GetLastError()));
  LARGE_INTEGER file_size{};
  if (!GetFileSizeEx(file, &file_size) || file_size.QuadPart < 0) {
    const DWORD code = GetLastError();
    CloseHandle(file);
    return fail(FormatWindowsError("GetFileSizeEx", code));
  }
  if (file_size.QuadPart == 0) {
    CloseHandle(file);
    return fail("file is empty");
  }
  const HANDLE mapping =
      CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (mapping == nullptr) {
    const DWORD code = GetLastError();
    CloseHandle(file);
    return fail(FormatWindowsError("CreateFileMappingW", code));
  }
  void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  if (view == nullptr) {
    const DWORD code = GetLastError();
    CloseHandle(mapping);
    CloseHandle(file);
    return fail(FormatWindowsError("MapViewOfFile", code));
  }
  ReadOnlyMappedFile result;
  result.data_ = static_cast<const std::uint8_t*>(view);
  result.size_ = static_cast<std::size_t>(file_size.QuadPart);
  result.handle_ = file;
  result.mapping_ = mapping;
  return result;
#else
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return fail(FormatErrno("cannot open", errno));
  struct stat st{};
  if (::fstat(fd, &st) != 0 || st.st_size < 0) {
    const int code = errno;
    ::close(fd);
    return fail(FormatErrno("cannot stat", code));
  }
  if (st.st_size == 0) {
    ::close(fd);
    return fail("file is empty");
  }
  const auto size = static_cast<std::size_t>(st.st_size);
  void* view = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (view == MAP_FAILED) {
    const int code = errno;
    ::close(fd);
    return fail(FormatErrno("cannot mmap", code));
  }
  (void)::madvise(view, size, MADV_SEQUENTIAL);
  ReadOnlyMappedFile result;
  result.data_ = static_cast<const std::uint8_t*>(view);
  result.size_ = size;
  result.fd_ = fd;
  return result;
#endif
}

std::size_t SystemPageSize() noexcept {
#if defined(_WIN32)
  SYSTEM_INFO info{};
  GetSystemInfo(&info);
  return info.dwPageSize != 0 ? static_cast<std::size_t>(info.dwPageSize)
                              : static_cast<std::size_t>(4096);
#else
  const long page_size = ::sysconf(_SC_PAGESIZE);
  return page_size > 0 ? static_cast<std::size_t>(page_size)
                       : static_cast<std::size_t>(4096);
#endif
}

bool RawMappedFile::Open(const std::filesystem::path& path, MapHint hint,
                         std::string* error) noexcept {
  const auto fail = [&](const char* what, std::uint64_t code) {
    if (error != nullptr) {
      if (code == 0)
        *error = what;
      else
#if defined(_WIN32)
        *error = FormatWindowsError(what, static_cast<DWORD>(code));
#else
        *error = FormatErrno(what, static_cast<int>(code));
#endif
    }
    return false;
  };
#if defined(_WIN32)
  const DWORD attributes = hint == MapHint::kSequential
                               ? FILE_FLAG_SEQUENTIAL_SCAN
                               : FILE_ATTRIBUTE_NORMAL;
  const HANDLE file = OpenForRead(path, attributes);
  if (file == INVALID_HANDLE_VALUE)
    return fail("cannot open", GetLastError());
  LARGE_INTEGER file_size{};
  if (!GetFileSizeEx(file, &file_size)) {
    const DWORD code = GetLastError();
    CloseHandle(file);
    return fail("cannot stat", code);
  }
  if (file_size.QuadPart <= 0) {
    CloseHandle(file);
    return fail("file is empty", 0);
  }
  const HANDLE section =
      CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (section == nullptr) {
    const DWORD code = GetLastError();
    CloseHandle(file);
    return fail("cannot create mapping", code);
  }
  void* view = MapViewOfFile(section, FILE_MAP_READ, 0, 0, 0);
  if (view == nullptr) {
    const DWORD code = GetLastError();
    CloseHandle(section);
    CloseHandle(file);
    return fail("cannot map", code);
  }
  data = view;
  size = static_cast<std::size_t>(file_size.QuadPart);
  view_base = view;
  view_size = size;
  this->file = file;
  this->section = section;
  return true;
#else
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return fail("cannot open", errno);
  struct stat status{};
  if (::fstat(fd, &status) != 0) {
    const int code = errno;
    ::close(fd);
    return fail("cannot stat", code);
  }
  if (status.st_size <= 0) {
    ::close(fd);
    return fail("file is empty", 0);
  }
  const auto file_size = static_cast<std::size_t>(status.st_size);
  void* mapping = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapping == MAP_FAILED) {
    const int code = errno;
    ::close(fd);
    return fail("cannot mmap", code);
  }
  ::close(fd);
  if (hint == MapHint::kSequential)
    (void)::madvise(mapping, file_size, MADV_SEQUENTIAL);
  data = mapping;
  size = file_size;
  view_base = mapping;
  view_size = file_size;
  return true;
#endif
}
bool RawMappedFile::OpenRange(const std::filesystem::path& path,
                              std::uint64_t offset, std::size_t length,
                              MapHint hint, std::string* error) noexcept {
  const auto fail = [&](const char* what, std::uint64_t code) {
    if (error != nullptr) {
      if (code == 0)
        *error = what;
      else
#if defined(_WIN32)
        *error = FormatWindowsError(what, static_cast<DWORD>(code));
#else
        *error = FormatErrno(what, static_cast<int>(code));
#endif
    }
    return false;
  };
  if (length == 0)
    return fail("zero length", 0);
#if defined(_WIN32)
  SYSTEM_INFO info{};
  GetSystemInfo(&info);
  const std::uint64_t granularity =
      info.dwAllocationGranularity != 0 ? info.dwAllocationGranularity : 65536;
#else
  const std::uint64_t granularity = SystemPageSize();
#endif
  const std::uint64_t base = offset - (offset % granularity);
  const std::uint64_t span = offset - base + length;
  if (span > (std::numeric_limits<std::size_t>::max)())
    return fail("range overflow", 0);
#if defined(_WIN32)
  const DWORD attributes = hint == MapHint::kSequential
                               ? FILE_FLAG_SEQUENTIAL_SCAN
                               : FILE_ATTRIBUTE_NORMAL;
  const HANDLE file = OpenForRead(path, attributes);
  if (file == INVALID_HANDLE_VALUE)
    return fail("cannot open", GetLastError());
  const HANDLE section =
      CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (section == nullptr) {
    const DWORD code = GetLastError();
    CloseHandle(file);
    return fail("cannot create mapping", code);
  }
  void* view =
      MapViewOfFile(section, FILE_MAP_READ, static_cast<DWORD>(base >> 32),
                    static_cast<DWORD>(base), static_cast<std::size_t>(span));
  if (view == nullptr) {
    const DWORD code = GetLastError();
    CloseHandle(section);
    CloseHandle(file);
    return fail("cannot map", code);
  }
  this->file = file;
  this->section = section;
#else
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return fail("cannot open", errno);
  void* view = ::mmap(nullptr, static_cast<std::size_t>(span), PROT_READ,
                      MAP_PRIVATE, fd, static_cast<off_t>(base));
  if (view == MAP_FAILED) {
    const int code = errno;
    ::close(fd);
    return fail("cannot mmap", code);
  }
  ::close(fd);
  if (hint == MapHint::kSequential)
    (void)::madvise(view, static_cast<std::size_t>(span), MADV_SEQUENTIAL);
#endif
  data = static_cast<std::uint8_t*>(view) + (offset - base);
  size = length;
  view_base = view;
  view_size = static_cast<std::size_t>(span);
  return true;
}

void RawMappedFile::Close() noexcept {
#if defined(_WIN32)
  if (view_base != nullptr)
    UnmapViewOfFile(view_base);
  if (section != nullptr)
    CloseHandle(static_cast<HANDLE>(section));
  if (file != nullptr)
    CloseHandle(static_cast<HANDLE>(file));
  file = nullptr;
  section = nullptr;
#else
  if (view_base != nullptr)
    munmap(view_base, view_size);
#endif
  data = nullptr;
  size = 0;
  view_base = nullptr;
  view_size = 0;
}

PlatformFile OpenReadFile(const std::filesystem::path& path,
                          std::string* error) noexcept {
#if defined(_WIN32)
  const HANDLE file = OpenForRead(path, FILE_ATTRIBUTE_NORMAL);
  if (file == INVALID_HANDLE_VALUE && error != nullptr)
    *error = FormatWindowsError("CreateFileW", GetLastError());
  return file;
#else
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0 && error != nullptr)
    *error = FormatErrno("cannot open", errno);
  return fd;
#endif
}

bool FileValid(PlatformFile file) noexcept {
#if defined(_WIN32)
  return file != nullptr && file != INVALID_HANDLE_VALUE;
#else
  return file >= 0;
#endif
}

void CloseFile(PlatformFile file) noexcept {
  if (!FileValid(file))
    return;
#if defined(_WIN32)
  CloseHandle(static_cast<HANDLE>(file));
#else
  ::close(file);
#endif
}

std::size_t ReadFileAt(PlatformFile file, void* output, std::size_t bytes,
                       std::uint64_t offset) noexcept {
#if defined(_WIN32)
  OVERLAPPED overlapped{};
  overlapped.Offset = static_cast<DWORD>(offset);
  overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
  // ReadFile takes a DWORD count; larger requests return a short read.
  const DWORD request = bytes < MAXDWORD ? static_cast<DWORD>(bytes) : MAXDWORD;
  DWORD read_bytes = 0;
  if (!ReadFile(static_cast<HANDLE>(file), output, request, &read_bytes,
                &overlapped)) {
    return 0;
  }
  return read_bytes;
#else
  const ssize_t read_bytes =
      ::pread(file, output, bytes, static_cast<off_t>(offset));
  return read_bytes > 0 ? static_cast<std::size_t>(read_bytes) : 0;
#endif
}

std::optional<std::uint64_t> FileSizeBytes(PlatformFile file) noexcept {
#if defined(_WIN32)
  LARGE_INTEGER size{};
  if (!GetFileSizeEx(static_cast<HANDLE>(file), &size) || size.QuadPart < 0)
    return std::nullopt;
  return static_cast<std::uint64_t>(size.QuadPart);
#else
  struct stat status{};
  if (::fstat(file, &status) != 0 || status.st_size < 0)
    return std::nullopt;
  return static_cast<std::uint64_t>(status.st_size);
#endif
}

PlatformFile DuplicateFile(PlatformFile file) noexcept {
  if (!FileValid(file))
    return kInvalidFile;
#if defined(_WIN32)
  HANDLE copy = nullptr;
  if (!DuplicateHandle(GetCurrentProcess(), static_cast<HANDLE>(file),
                       GetCurrentProcess(), &copy, 0, FALSE,
                       DUPLICATE_SAME_ACCESS))
    return kInvalidFile;
  return copy;
#else
  return ::fcntl(file, F_DUPFD_CLOEXEC, 0);
#endif
}

PlatformFile ReopenUnbuffered(PlatformFile file) noexcept {
  if (!FileValid(file))
    return kInvalidFile;
#if defined(_WIN32)
  // ReOpenFile fails when the desired share mode conflicts with the original
  // open; callers fall back to buffered I/O.
  HANDLE reopened =
      ReOpenFile(static_cast<HANDLE>(file), GENERIC_READ,
                 FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                 FILE_FLAG_NO_BUFFERING);
  return reopened == INVALID_HANDLE_VALUE ? kInvalidFile : reopened;
#else
  // Reopening the retained descriptor creates an independent O_DIRECT file
  // description without resolving the original, replaceable pathname.
  const auto path = "/proc/self/fd/" + std::to_string(file);
  return ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
#endif
}

}  // namespace gufo::platform
