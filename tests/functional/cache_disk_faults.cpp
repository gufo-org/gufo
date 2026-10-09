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
        !held.exchange(true)) {
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
        for (unsigned attempt = 0; attempt < 120000; ++attempt) {
          if (access((control + "/release").c_str(), F_OK) == 0)
            break;
          usleep(1000);
        }
      }
    }
  }
  return original(descriptor);
}
