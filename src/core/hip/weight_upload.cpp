#include "src/core/hip/weight_upload.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/hip/host_allocations.hpp"
#include "src/core/platform/fs.hpp"

namespace gufo::hip {
namespace {

constexpr std::size_t kAlignment = 4096;
constexpr std::size_t kChunkBytes = 16ULL << 20;
constexpr std::size_t kReaders = 16;

bool ReadFully(platform::PlatformFile fd, void* buffer, std::uint64_t offset,
               std::size_t length, std::size_t required, std::error_code& ec) {
  std::size_t got = 0;
  while (got < required) {
    const auto n = platform::ReadAt(
        fd, offset + got, static_cast<char*>(buffer) + got, length - got, ec);
    if (n == 0 && ec == std::errc::interrupted) {
      continue;
    }
    if (n == 0) {
      if (!ec) {
        // End of file before the required bytes.
        ec = std::make_error_code(std::errc::io_error);
      }
      return false;
    }
    got += n;
  }
  return true;
}

}  // namespace

struct WeightUpload::State {
  struct Shard {
    platform::PlatformFile fd{platform::kInvalidFile};
    std::uint64_t size{0};
    const void* mapping{nullptr};
  };
  struct Slot {
    void* buffer{nullptr};
    hipStream_t stream{nullptr};
    // One unbuffered description per reader, reopened when the reader moves
    // to another shard: independent file objects keep concurrent readers off
    // one synchronous handle.
    platform::PlatformFile direct{platform::kInvalidFile};
    std::uint32_t direct_shard{0xffffffffu};
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
      platform::CloseFile(slot.direct);
      if (slot.stream != nullptr) {
        (void)hipStreamDestroy(slot.stream);
      }
    }
    for (const auto& shard : shards) {
      platform::CloseFile(shard.fd);
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
    std::error_code ec;
    bool read = false;
    // Each reader keeps an unbuffered description of the shard it reads, so
    // concurrent readers never queue on one synchronous handle: Windows
    // serializes synchronous I/O per file object, not per process.
    if (slot.direct_shard != task.shard) {
      platform::CloseFile(slot.direct);
      slot.direct = platform::kInvalidFile;
      slot.direct_shard = task.shard;
    }
    if (!platform::FileValid(slot.direct)) {
      slot.direct = platform::ReopenUnbuffered(shard.fd);
    }
    if (platform::FileValid(slot.direct)) {
      const auto begin = task.offset & ~(kAlignment - 1);
      const auto skip = static_cast<std::size_t>(task.offset - begin);
      const auto length =
          (skip + task.size + kAlignment - 1) & ~(kAlignment - 1);
      read = ReadFully(slot.direct, slot.buffer, begin, length,
                       skip + task.size, ec);
      payload = static_cast<const char*>(slot.buffer) + skip;
    }
    if (!read) {
      // A failed unbuffered read retries through the buffered descriptor,
      // which reports its own error. Some filesystems accept unbuffered opens
      // but reject aligned reads, and Windows refuses an unbuffered read with
      // ERROR_NO_SYSTEM_RESOURCES when it cannot lock the pages in memory.
      ec = {};
      if (!ReadFully(shard.fd, slot.buffer, task.offset, task.size, task.size,
                     ec)) {
        return "shard read failed: " + ec.message();
      }
      payload = slot.buffer;
      platform::Advise(shard.fd, platform::AdviseHint::kDontNeed, task.offset,
                       task.size);
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
    shard.fd = platform::DuplicateFile(region.file_descriptor);
    std::error_code ec;
    platform::FileInfo info{};
    if (!platform::FileValid(shard.fd) ||
        !platform::Stat(shard.fd, &info, ec) || info.size == 0 ||
        info.size != region.size) {
      s.Fail("cannot read the mapped GGUF shard");
      s.Status(error);
      return nullptr;
    }
    shard.size = info.size;
    shard.mapping = region.data;
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

bool WeightUpload::CopyMapped(const void* source, std::size_t size,
                              void* device, std::string* error) {
  const auto address = reinterpret_cast<std::uintptr_t>(source);
  for (std::size_t i = 0; i < state_->shards.size(); ++i) {
    const auto& shard = state_->shards[i];
    const auto base = reinterpret_cast<std::uintptr_t>(shard.mapping);
    if (shard.mapping != nullptr && address >= base &&
        address - base < shard.size) {
      return Copy(static_cast<std::uint32_t>(i), address - base, size, device,
                  error);
    }
  }
  if (error)
    *error = "weight upload source is outside its mappings";
  return false;
}

bool WeightUpload::Finish(std::string* error) {
  auto& s = *state_;
  std::unique_lock lock(s.mutex);
  s.changed.wait(lock, [&] { return s.pending == 0; });
  return s.Status(error);
}

}  // namespace gufo::hip
