// Test-only gate: hold a private cache write before its durable publication.
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {
bool MatchesManifest(int descriptor) {
  if (std::getenv("GUFO_TEST_DISK_MANIFEST_ONLY") == nullptr)
    return true;
  char path[64]{};
  std::snprintf(path, sizeof(path), "/proc/self/fd/%d", descriptor);
  const int input = open(path, O_RDONLY | O_CLOEXEC);
  if (input < 0)
    return false;
  char magic[8]{};
  const auto count = pread(input, magic, sizeof(magic), 0);
  close(input);
  return count == 8 && std::memcmp(magic, "GUFOMNF2", 8) == 0;
}
}  // namespace

extern "C" int fsync(int descriptor) {
  using Function = int (*)(int);
  static const auto original =
      reinterpret_cast<Function>(dlsym(RTLD_NEXT, "fsync"));
  static std::atomic<bool> held{false};
  const char* gate = std::getenv("GUFO_TEST_DISK_GATE");
  const char* directory = std::getenv("GUFO_TEST_DISK_DIRECTORY");
  if (gate != nullptr && directory != nullptr && !held.load()) {
    struct stat status{};
    char descriptor_path[64]{};
    char filename[4096]{};
    std::snprintf(descriptor_path, sizeof(descriptor_path), "/proc/self/fd/%d",
                  descriptor);
    const auto length =
        readlink(descriptor_path, filename, sizeof(filename) - 1);
    const std::string root = std::string(directory) + "/";
    const std::string control = gate;
    if (length > 0 && fstat(descriptor, &status) == 0 &&
        S_ISREG(status.st_mode) &&
        std::strncmp(filename, root.c_str(), root.size()) == 0 &&
        access((control + "/armed").c_str(), F_OK) == 0 &&
        MatchesManifest(descriptor) && !held.exchange(true)) {
      const int marker = open((control + "/blocked").c_str(),
                              O_WRONLY | O_CREAT | O_EXCL, 0600);
      if (marker >= 0) {
        const auto written =
            write(marker, filename, static_cast<std::size_t>(length));
        close(marker);
        if (written != length) {
          unlink((control + "/blocked").c_str());
          return original(descriptor);
        }
        // The harness releases this gate or kills only its own child process.
        // Publication-wait controls hold beyond the scheduler's 120s ceiling.
        for (unsigned attempt = 0; attempt < 300000; ++attempt) {
          if (access((control + "/release").c_str(), F_OK) == 0)
            break;
          usleep(1000);
        }
      }
    }
  }
  return original(descriptor);
}
