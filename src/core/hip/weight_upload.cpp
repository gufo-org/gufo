#include "src/core/hip/weight_upload.hpp"

#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/hip/host_allocations.hpp"

namespace gufo::hip {
namespace {

constexpr std::size_t kAlignment = 4096;
constexpr std::size_t kChunkBytes = 16ULL << 20;
constexpr std::size_t kReaders = 16;

bool ReadFully(int fd, void* buffer, std::uint64_t offset, std::size_t length,
               std::size_t required) {
  std::size_t got = 0;
  while (got < required) {
    const auto n = ::pread(fd, static_cast<char*>(buffer) + got, length - got,
                           static_cast<off_t>(offset + got));
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n <= 0) {
      if (n == 0) {
        errno = EIO;
      }
      return false;
    }
    got += static_cast<std::size_t>(n);
  }
  return true;
}

}  // namespace

struct WeightUpload::State {
  struct Shard {
    int fd{-1};
    int direct_fd{-1};
    std::uint64_t size{0};
  };
  struct Slot {
    void* buffer{nullptr};
    hipStream_t stream{nullptr};
  };
  struct Task {
    std::uint32_t shard;
    std::uint64_t offset;
    std::size_t size;
    void* destination;
  };

  std::vector<Shard> shards;
  HostAllocations host_allocations;
  std::array<Slot, kReaders> slots{};
  std::vector<std::jthread> readers;
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<Task> queue;
  std::size_t pending{0};
  bool stopping{false};
  std::string failure;

  ~State() {
    {
      std::lock_guard lock(mutex);
      stopping = true;
    }
    changed.notify_all();
    // Workers finish queued and in-flight copies before their buffers or the
    // caller's destinations can be freed, even on a failed upload.
    readers.clear();
    for (auto& slot : slots) {
      if (slot.stream != nullptr) {
        (void)hipStreamDestroy(slot.stream);
      }
    }
    for (const auto& shard : shards) {
      if (shard.fd >= 0) {
        ::close(shard.fd);
      }
      if (shard.direct_fd >= 0) {
        ::close(shard.direct_fd);
      }
    }
  }

  bool Status(std::string* error) const {
    if (!failure.empty() && error != nullptr) {
      *error = failure;
    }
    return failure.empty();
  }

  void Fail(std::string message) {
    // Caller holds mutex (or workers have not started yet).
    if (failure.empty()) {
      failure = std::move(message);
    }
    pending -= queue.size();
    queue.clear();
    changed.notify_all();
  }

  std::string Transfer(const Task& task, Slot& slot) {
    const auto& shard = shards[task.shard];
    const void* payload = slot.buffer;
    bool read = false;
    if (shard.direct_fd >= 0) {
      const auto begin = task.offset & ~(kAlignment - 1);
      const auto skip = static_cast<std::size_t>(task.offset - begin);
      const auto length =
          (skip + task.size + kAlignment - 1) & ~(kAlignment - 1);
      read = ReadFully(shard.direct_fd, slot.buffer, begin, length,
                       skip + task.size);
      payload = static_cast<const char*>(slot.buffer) + skip;
      if (!read && errno != EINVAL && errno != EOPNOTSUPP) {
        return "shard read failed: " + std::string(std::strerror(errno));
      }
    }
    if (!read) {
      // Some filesystems accept O_DIRECT at open but reject aligned reads.
      if (!ReadFully(shard.fd, slot.buffer, task.offset, task.size,
                     task.size)) {
        return "shard read failed: " + std::string(std::strerror(errno));
      }
      payload = slot.buffer;
      (void)::posix_fadvise(shard.fd, static_cast<off_t>(task.offset),
                            static_cast<off_t>(task.size), POSIX_FADV_DONTNEED);
    }
    auto status = hipMemcpyAsync(task.destination, payload, task.size,
                                 hipMemcpyHostToDevice, slot.stream);
    if (status == hipSuccess) {
      status = hipStreamSynchronize(slot.stream);
    }
    if (status != hipSuccess) {
      return "weight upload failed: " + std::string(hipGetErrorString(status));
    }
    return {};
  }

  void Run(Slot& slot) {
    for (;;) {
      Task task{};
      {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return stopping || !queue.empty(); });
        if (queue.empty()) {
          return;
        }
        task = queue.front();
        queue.pop_front();
        changed.notify_all();
      }
      auto error = Transfer(task, slot);
      {
        std::lock_guard lock(mutex);
        --pending;
        if (!error.empty()) {
          Fail(std::move(error));
        }
      }
      changed.notify_all();
    }
  }
};

WeightUpload::WeightUpload() : state_(std::make_unique<State>()) {}
WeightUpload::~WeightUpload() = default;

std::unique_ptr<WeightUpload> WeightUpload::Create(
    std::span<const core::GgufMappedRegion> regions, std::string* error) {
  std::unique_ptr<WeightUpload> uploader(new WeightUpload());
  auto& s = *uploader->state_;
  for (const auto& region : regions) {
    auto& shard = s.shards.emplace_back();
    shard.fd = ::fcntl(region.file_descriptor, F_DUPFD_CLOEXEC, 0);
    struct stat info{};
    if (shard.fd < 0 || ::fstat(shard.fd, &info) != 0 || info.st_size <= 0 ||
        static_cast<std::uint64_t>(info.st_size) != region.size) {
      s.Fail("cannot read the mapped GGUF shard");
      s.Status(error);
      return nullptr;
    }
    shard.size = static_cast<std::uint64_t>(info.st_size);
    // Reopening the retained descriptor creates an independent O_DIRECT file
    // description without resolving the original, replaceable pathname.
    const auto path = "/proc/self/fd/" + std::to_string(shard.fd);
    shard.direct_fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
  }
  for (auto& slot : s.slots) {
    if (s.host_allocations.Allocate(&slot.buffer, kChunkBytes + kAlignment) !=
            hipSuccess ||
        hipStreamCreateWithFlags(&slot.stream, hipStreamNonBlocking) !=
            hipSuccess) {
      s.Fail("weight upload staging allocation failed");
      s.Status(error);
      return nullptr;
    }
  }
  try {
    for (auto& slot : s.slots) {
      s.readers.emplace_back([&s, &slot] { s.Run(slot); });
    }
  } catch (const std::system_error& e) {
    std::lock_guard lock(s.mutex);
    s.Fail("weight upload worker creation failed: " + std::string(e.what()));
    s.Status(error);
    return nullptr;
  }
  return uploader;
}

bool WeightUpload::Copy(std::uint32_t shard, std::uint64_t offset,
                        std::size_t size, void* device, std::string* error) {
  auto& s = *state_;
  std::unique_lock lock(s.mutex);
  if (shard >= s.shards.size() || offset > s.shards[shard].size ||
      size > s.shards[shard].size - offset || (size != 0 && !device)) {
    s.Fail("weight upload range is outside its shard");
  }
  std::size_t done = 0;
  while (s.Status(error) && done < size) {
    s.changed.wait(lock, [&] {
      return !s.failure.empty() || s.queue.size() < 2 * kReaders;
    });
    if (!s.Status(error)) {
      break;
    }
    const auto count = std::min(size - done, kChunkBytes);
    s.queue.push_back(
        {shard, offset + done, count, static_cast<char*>(device) + done});
    ++s.pending;
    done += count;
    s.changed.notify_all();
  }
  return s.Status(error);
}

bool WeightUpload::Finish(std::string* error) {
  auto& s = *state_;
  std::unique_lock lock(s.mutex);
  s.changed.wait(lock, [&] { return s.pending == 0; });
  return s.Status(error);
}

}  // namespace gufo::hip
