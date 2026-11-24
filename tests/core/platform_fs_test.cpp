#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>

#include "src/core/platform/fs.hpp"

namespace {

// Checks stay active under NDEBUG: several conditions carry the call under
// test.
void Require(bool ok, const char* condition, int line) {
  if (!ok) {
    std::fprintf(stderr, "FAIL line %d: %s\n", line, condition);
    std::exit(1);
  }
}
#define CHECK(condition) Require((condition), #condition, __LINE__)

}  // namespace

int main() {
  namespace fs = std::filesystem;
  namespace platform = gufo::platform;

  const auto root =
      fs::temp_directory_path() /
      ("gufo_fs_test_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(root);

  // Directory handles: Open refuses symlinks; OpenDirectoryAt resolves ".".
  platform::Directory dir;
  std::error_code ec;
  CHECK(dir.Open(root, ec));
  CHECK(dir.valid());
  {
    platform::Directory missing;
    CHECK(!missing.Open(root / "absent", ec));
    CHECK(ec == std::errc::no_such_file_or_directory);
  }
  {
    auto self = platform::OpenDirectoryAt(dir, ".", ec);
    CHECK(self.valid());
  }

  // OpenAt: create-new refuses a second creation; open-or-create reuses.
  const auto name = "payload.bin";
  platform::PlatformFile created =
      platform::OpenAt(dir, name, platform::FileMode::kCreateNew, ec);
  CHECK(platform::FileValid(created));
  {
    platform::PlatformFile again =
        platform::OpenAt(dir, name, platform::FileMode::kCreateNew, ec);
    CHECK(!platform::FileValid(again));
    CHECK(ec == std::errc::file_exists);
    platform::CloseFile(again);
  }
  {
    platform::PlatformFile reopened =
        platform::OpenAt(dir, name, platform::FileMode::kOpenOrCreate, ec);
    CHECK(platform::FileValid(reopened));
    platform::CloseFile(reopened);
  }
  {
    platform::PlatformFile missing =
        platform::OpenAt(dir, "absent.bin", platform::FileMode::kRead, ec);
    CHECK(!platform::FileValid(missing));
    CHECK(ec == std::errc::no_such_file_or_directory);
  }

  // Write / Read / ReadAt round trip, including the INT_MAX clamp path.
  const std::string payload = "gufo filesystem leaf";
  CHECK(platform::Write(created, payload.data(), payload.size(), ec) ==
        payload.size());
  {
    std::error_code read_ec;
    platform::PlatformFile reader =
        platform::OpenAt(dir, name, platform::FileMode::kRead, ec);
    CHECK(platform::FileValid(reader));
    char buffer[64]{};
    const auto got = platform::Read(reader, buffer, sizeof(buffer), read_ec);
    CHECK(got == payload.size());
    CHECK(std::string(buffer, got) == payload);
    // Position sits at end of file now; the next read reports clean EOF.
    CHECK(platform::Read(reader, buffer, sizeof(buffer), read_ec) == 0);
    CHECK(!read_ec);
    char at[32]{};
    CHECK(platform::ReadAt(reader, 5, at, 4, read_ec) == 4);
    CHECK(std::string(at, 4) == "file");
    platform::CloseFile(reader);
  }

  // Stat reports the identity; it moves on a rewrite.
  platform::FileInfo before{};
  CHECK(platform::Stat(created, &before, ec));
  CHECK(before.regular);
  CHECK(before.links == 1);
  CHECK(before.size == payload.size());
  CHECK(before.inode != 0);
  platform::CloseFile(created);
  {
    // A fresh file under the same name keeps the inode but the size moves,
    // and the identity comparison tracks every field.
    platform::FileInfo after{};
    platform::PlatformFile file =
        platform::OpenAt(dir, name, platform::FileMode::kOpenOrCreate, ec);
    CHECK(platform::FileValid(file));
    CHECK(platform::Stat(file, &after, ec));
    CHECK(after.device == before.device);
    CHECK(after.inode == before.inode);
    CHECK(after.size == before.size);
    const std::string longer = payload + " with more bytes";
    CHECK(platform::Write(file, longer.data(), longer.size(), ec) ==
          longer.size());
    platform::FileInfo grown{};
    CHECK(platform::Stat(file, &grown, ec));
    CHECK(grown.size == longer.size());
    CHECK(grown.modified_ns != 0);
    CHECK(!(grown == before));
    platform::CloseFile(file);
  }

  // StatAt without following symlinks reports ENOENT for a dangling path;
  // regular lookups agree with Stat.
  {
    platform::FileInfo info{};
    CHECK(platform::StatAt(dir, name, false, &info, ec));
    CHECK(info.size == payload.size() + sizeof(" with more bytes") - 1);
    platform::FileInfo by_handle{};
    platform::PlatformFile file =
        platform::OpenAt(dir, name, platform::FileMode::kRead, ec);
    CHECK(platform::FileValid(file));
    CHECK(platform::Stat(file, &by_handle, ec));
    CHECK(by_handle == info);
    platform::CloseFile(file);
    CHECK(!platform::StatAt(dir, "absent.bin", false, &info, ec));
    CHECK(ec == std::errc::no_such_file_or_directory);
  }

  // RenameNoReplace keeps an existing target and reports it; Rename replaces.
  {
    platform::Directory source;
    CHECK(platform::MakeDirectoryAt(dir, "from", ec));
    CHECK(source.Open(dir.Path() / "from", ec));
    platform::PlatformFile staged = platform::OpenAt(
        source, "staged.bin", platform::FileMode::kCreateNew, ec);
    CHECK(platform::FileValid(staged));
    const std::string contents = "staged";
    CHECK(platform::Write(staged, contents.data(), contents.size(), ec) ==
          contents.size());
    platform::CloseFile(staged);
    CHECK(platform::RenameNoReplace(source, "staged.bin", dir, "kept.bin",
                                    ec) == platform::RenameResult::kRenamed);
    // Restage, then refuse to replace the target that now exists.
    staged = platform::OpenAt(source, "staged.bin",
                              platform::FileMode::kCreateNew, ec);
    CHECK(platform::FileValid(staged));
    platform::CloseFile(staged);
    CHECK(
        platform::RenameNoReplace(source, "staged.bin", dir, "kept.bin", ec) ==
        platform::RenameResult::kTargetExists);
    CHECK(ec == std::errc::file_exists);
    {
      platform::FileInfo kept{};
      CHECK(platform::StatAt(dir, "kept.bin", false, &kept, ec));
      CHECK(kept.size == contents.size());  // The target survived untouched.
    }
    CHECK(platform::Rename(source, "absent.bin", dir, "kept.bin", ec) == false);
    CHECK(ec == std::errc::no_such_file_or_directory);
  }

  // Directory creation, listing, and unlink.
  {
    CHECK(platform::MakeDirectoryAt(dir, "listing", ec));
    CHECK(platform::MakeDirectoryAt(dir, "listing", ec));  // EEXIST tolerated.
    platform::Directory listing;
    CHECK(listing.Open(dir.Path() / "listing", ec));
    for (const char* entry : {"a.bin", "b.bin", "c.bin"}) {
      platform::PlatformFile file =
          platform::OpenAt(listing, entry, platform::FileMode::kCreateNew, ec);
      CHECK(platform::FileValid(file));
      platform::CloseFile(file);
    }
    std::vector<std::string> names;
    CHECK(platform::ListDirectory(
        listing,
        [&](std::string_view name, const platform::FileInfo& info) {
          names.emplace_back(name);
          CHECK(info.regular);
        },
        ec));
    CHECK(names.size() == 3);
    CHECK(platform::Unlink(listing, "a.bin", ec));
    CHECK(!platform::Unlink(listing, "a.bin", ec));
    CHECK(ec == std::errc::no_such_file_or_directory);
  }

  // MakeDirectory on a fresh path, twice.
  {
    CHECK(platform::MakeDirectory(root / "created", ec));
    CHECK(platform::MakeDirectory(root / "created", ec));
  }

  // Exclusive lock: the second taker fails.
  {
    platform::PlatformFile lock =
        platform::OpenAt(dir, "LOCK", platform::FileMode::kOpenOrCreate, ec);
    CHECK(platform::FileValid(lock));
    CHECK(platform::TryLockExclusive(lock, ec));
    platform::PlatformFile second =
        platform::OpenAt(dir, "LOCK", platform::FileMode::kOpenOrCreate, ec);
    CHECK(platform::FileValid(second));
    CHECK(!platform::TryLockExclusive(second, ec));
    platform::CloseFile(second);
    platform::CloseFile(lock);
  }

  // Sync helpers: a writable file and its directory sync cleanly.
  {
    platform::PlatformFile file =
        platform::OpenAt(dir, name, platform::FileMode::kOpenOrCreate, ec);
    CHECK(platform::FileValid(file));
    CHECK(platform::SyncFile(file, ec));
    platform::CloseFile(file);
    CHECK(platform::SyncDirectory(dir, ec));
    CHECK(platform::SyncAncestors(dir, ec));
  }

  // OwnerOnly policy is exercised on POSIX; the user_id/permissions fields
  // are documented zero on other platforms.
  {
    platform::FileInfo info{};
    CHECK(platform::StatAt(dir, name, false, &info, ec));
    CHECK(platform::OwnerOnly(info, 0022) ||
          (info.user_id == 0 && info.permissions == 0));
  }

  // UserCacheDirectory: empty when the platform has none, else non-empty.
  (void)platform::UserCacheDirectory();

#if !defined(_WIN32)
  // Symlink refusal: neither a directory handle nor a file opens through a
  // symlink final component, and a link lists as a non-regular entry.
  {
    platform::Directory target;
    CHECK(platform::MakeDirectory(root / "linked-target", ec));
    CHECK(target.Open(root / "linked-target", ec));
    platform::PlatformFile seeded = platform::OpenAt(
        target, "inner.bin", platform::FileMode::kCreateNew, ec);
    CHECK(platform::FileValid(seeded));
    platform::CloseFile(seeded);
    fs::create_directory_symlink(root / "linked-target", root / "dir-link");
    fs::create_symlink("inner.bin", root / "linked-target" / "file-link");
    platform::Directory through_link;
    CHECK(!through_link.Open(root / "dir-link", ec));
    // POSIX allows either code for O_DIRECTORY | O_NOFOLLOW on a link; Linux
    // reports ENOTDIR.
    CHECK(ec == std::errc::too_many_symbolic_link_levels ||
          ec == std::errc::not_a_directory);
    platform::PlatformFile file_link =
        platform::OpenAt(target, "file-link", platform::FileMode::kRead, ec);
    CHECK(!platform::FileValid(file_link));
    CHECK(ec == std::errc::too_many_symbolic_link_levels);
    std::size_t entries = 0, regular = 0;
    CHECK(platform::ListDirectory(
        target,
        [&](std::string_view, const platform::FileInfo& info) {
          ++entries;
          regular += info.regular ? 1 : 0;
        },
        ec));
    CHECK(entries == 2);  // inner.bin and file-link; the link is not fatal...
    CHECK(regular == 1);  // ...and only inner.bin is regular.
    // A listing that contains a directory link succeeds too.
    std::size_t seen = 0, link_entries = 0;
    CHECK(platform::ListDirectory(
        dir,
        [&](std::string_view name, const platform::FileInfo& info) {
          ++seen;
          if (name == "dir-link")
            ++link_entries;
          (void)info;
        },
        ec));
    CHECK(seen >= 2);
    CHECK(link_entries == 1);
  }
#endif

  // Advise is best effort and must not crash on any range.
  {
    platform::PlatformFile file =
        platform::OpenAt(dir, name, platform::FileMode::kRead, ec);
    CHECK(platform::FileValid(file));
    platform::Advise(file, platform::AdviseHint::kWillNeed, 0, 16);
    platform::Advise(file, platform::AdviseHint::kDontNeed, 0, 0);
    platform::CloseFile(file);
  }

  // CloneFile: a reflink holds the same bytes in its own file; where the file
  // system cannot clone, nothing is left behind. It never replaces a file.
  {
    const auto source = root / "clone-source";
    const auto clone = root / "clone";
    std::ofstream(source, std::ios::binary) << "shared extents";
    if (platform::CloneFile(source, clone, ec)) {
      std::ifstream in(clone, std::ios::binary);
      std::string text((std::istreambuf_iterator<char>(in)), {});
      CHECK(text == "shared extents");
      CHECK(!fs::equivalent(source, clone));
      std::error_code again;
      CHECK(!platform::CloneFile(source, clone, again));
      CHECK(again == std::errc::file_exists);
    } else {
      CHECK(static_cast<bool>(ec));
      CHECK(!fs::exists(clone));
    }
    std::error_code missing;
    CHECK(!platform::CloneFile(root / "absent", root / "never", missing));
    CHECK(!fs::exists(root / "never"));
  }

  std::error_code cleanup;
  fs::remove_all(root, cleanup);
  return 0;
}
