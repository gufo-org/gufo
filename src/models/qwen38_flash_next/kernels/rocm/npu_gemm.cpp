#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace gufo::models::qwen38_flash_next::rocm {

namespace {

constexpr unsigned kOpcode = 3;
constexpr unsigned kDefaultRunTimeoutMs = 8000;  // a dispatch takes 1-15 ms
constexpr std::size_t kBytesPer8 = 9;
constexpr std::int64_t kHeartbeatQuietMs = 2000;
constexpr std::size_t kMaxInFlight = 4;  // the driver's stock queue depth
constexpr unsigned kCopyHelpers = 3;
constexpr std::size_t kParallelCopyMin = std::size_t{1} << 20;

std::int64_t NowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// GUFO_NPU_TRACE=<path>: one line per NPU-path event, written through O_DSYNC
// so the last line survives a hang or a freeze.
void Trace(const char* what, long a = 0, long b = 0) {
  static const int fd = [] {
    const char* path = std::getenv("GUFO_NPU_TRACE");
    return path != nullptr && path[0] != '\0'
               ? open(path, O_WRONLY | O_CREAT | O_APPEND | O_DSYNC, 0644)
               : -1;
  }();
  if (fd < 0) {
    return;
  }
  char line[160];
  const int n = std::snprintf(line, sizeof line, "%lld %s %ld %ld\n",
                              static_cast<long long>(NowMs()), what, a, b);
  if (n > 0) {
    (void)!write(fd, line, static_cast<std::size_t>(n));
  }
}

struct Call {
  void* impl;
  int a_id;
  std::vector<int> b_ids;
  std::vector<int> c_ids;
};

class CopyPool {
public:
  explicit CopyPool(unsigned helpers) : parts_(helpers + 1), next_(parts_) {
    for (unsigned i = 0; i < helpers; ++i) {
      threads_.emplace_back([this] { Run(); });
    }
  }

  ~CopyPool() {
    {
      const std::lock_guard lock(mutex_);
      stop_ = true;
    }
    wake_.notify_all();
    for (auto& t : threads_) {
      t.join();
    }
  }

  void Copy(void* dst, const void* src, std::size_t bytes) {
    if (bytes < kParallelCopyMin) {
      std::memcpy(dst, src, bytes);
      return;
    }
    const std::lock_guard serial(call_mutex_);
    {
      const std::lock_guard lock(mutex_);
      dst_ = static_cast<char*>(dst);
      src_ = static_cast<const char*>(src);
      bytes_ = bytes;
      next_ = 0;
      remaining_ = parts_;
    }
    wake_.notify_all();
    Work();
    std::unique_lock lock(mutex_);
    done_.wait(lock, [this] { return remaining_ == 0; });
  }

private:
  void Work() {
    for (;;) {
      unsigned part = 0;
      char* dst = nullptr;
      const char* src = nullptr;
      std::size_t begin = 0;
      std::size_t end = 0;
      {
        const std::lock_guard lock(mutex_);
        if (next_ >= parts_) {
          return;
        }
        part = next_++;
        const std::size_t chunk = (bytes_ + parts_ - 1) / parts_;
        begin = std::min<std::size_t>(part * chunk, bytes_);
        end = std::min(begin + chunk, bytes_);
        dst = dst_;
        src = src_;
      }
      std::memcpy(dst + begin, src + begin, end - begin);
      const std::lock_guard lock(mutex_);
      if (--remaining_ == 0) {
        done_.notify_all();
      }
    }
  }

  void Run() {
    for (;;) {
      {
        std::unique_lock lock(mutex_);
        wake_.wait(lock, [this] { return stop_ || next_ < parts_; });
        if (stop_) {
          return;
        }
      }
      Work();
    }
  }

  const unsigned parts_;
  std::mutex call_mutex_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::condition_variable done_;
  char* dst_{nullptr};
  const char* src_{nullptr};
  std::size_t bytes_{0};
  unsigned next_;
  unsigned remaining_{0};
  bool stop_{false};
  std::vector<std::thread> threads_;
};

}  // namespace

struct NpuGemm::Impl {
  Config config;
  xrt::device device;
  xrt::hw_context context;
  xrt::kernel kernel;
  xrt::bo instructions;
  std::size_t instruction_words{0};
  int group_a{0};

  struct Buffer {
    void* host{nullptr};    // GPU-visible pinned memory
    void* device{nullptr};  // the same memory as a GPU pointer
    std::size_t bytes{0};
    xrt::bo shared_bo;   // userptr import of `host` (normal mode)
    xrt::bo private_bo;  // NPU-only pages (isolate mode)
    void* private_host{nullptr};
  };
  std::vector<Buffer> buffers;
  const bool isolate{std::getenv("GUFO_NPU_ISOLATE") != nullptr};

  CopyPool copy_pool{kCopyHelpers};

  std::mutex mutex;
  std::deque<std::pair<xrt::run, int>> pending;  // run, its C buffer

  // Submissions run on a worker so the stream only blocks in Wait.
  std::thread worker;
  std::mutex queue_mutex;
  std::condition_variable queue_wake;
  std::condition_variable queue_idle;
  std::deque<Call> queue;
  std::uint64_t enqueued{0};
  std::uint64_t completed{0};
  bool worker_stop{false};
  std::atomic<bool> failed{false};
  std::atomic<std::int64_t> last_submit_ms{0};  // steady clock, ms
  std::chrono::milliseconds run_timeout{kDefaultRunTimeoutMs};
  std::vector<xrt::run>
      abandoned;  // timed-out runs stay alive: never destroyed under the NPU
  std::string last_error;

  // GUFO_NPU_STATS=1: wall-clock accounting printed every 36 waits.
  const bool stats{std::getenv("GUFO_NPU_STATS") != nullptr};
  double submit_us{0};
  // Phase totals of the worker (GUFO_NPU_STATS): where the NPU path spends
  // time.
  double copy_a_us{0};    // A host->private copy + sync
  double launch_us{0};    // kernel(...) submissions
  double run_wait_us{0};  // waiting for the NPU itself
  double copy_c_us{0};    // C sync + private->host copy
  double wait_us{0};
  std::uint64_t waits{0};

  // Keepalive: the driver suspends the NPU after 5 s without a job, and waking
  // it re-initializes the SMU, PSP and mailbox while the GPU may be running
  // flat out. A tiny job every second keeps it powered for the object's life.
  std::thread keepalive;
  std::mutex keepalive_mutex;
  std::condition_variable keepalive_wake;
  bool keepalive_stop{false};
  int heartbeat_a{-1};
  int heartbeat_b{-1};
  int heartbeat_c{-1};

  ~Impl() {
    StopKeepalive();
    StopWorker();
    if (failed) {
      // A timed-out run may still be doing DMA: abort it, then never free the
      // pages or the run objects it uses (leak them until the process exits).
      for (auto& r : abandoned) {
        try {
          r.abort();
        } catch (const std::exception&) {
        }
      }
      new std::vector<xrt::run>(std::move(abandoned));
      new std::vector<Buffer>(std::move(buffers));
      new std::deque<std::pair<xrt::run, int>>(std::move(pending));
      return;
    }
    pending.clear();
    for (auto& b : buffers) {
      b.shared_bo = xrt::bo();
      b.private_bo = xrt::bo();
      if (b.host != nullptr) {
        (void)hipHostFree(b.host);
      }
    }
  }

  xrt::bo& NpuBo(int id) {
    Buffer& b = buffers[id];
    return isolate ? b.private_bo : b.shared_bo;
  }

  static double Since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now() - t)
        .count();
  }

  void CopyToNpu(int id) {
    const auto t0 = std::chrono::steady_clock::now();
    Buffer& b = buffers[id];
    if (isolate) {
      copy_pool.Copy(b.private_host, b.host, b.bytes);
    }
    NpuBo(id).sync(XCL_BO_SYNC_BO_TO_DEVICE);
    copy_a_us += Since(t0);
  }

  void CopyFromNpu(int id) {
    const auto t0 = std::chrono::steady_clock::now();
    Buffer& b = buffers[id];
    NpuBo(id).sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (isolate) {
      copy_pool.Copy(b.host, b.private_host, b.bytes);
    }
    copy_c_us += Since(t0);
  }

  void Fail(const std::exception& e) {
    failed = true;
    last_error = e.what();
  }

  // Waits for `run` with a bound; a timeout poisons this object instead of
  // blocking the worker (and through it the HIP stream) forever.
  bool WaitRun(xrt::run& run) {
    const ert_cmd_state state = run.wait(run_timeout);
    if (state == ERT_CMD_STATE_COMPLETED) {
      return true;
    }
    failed = true;
    last_error = state == ERT_CMD_STATE_TIMEOUT
                     ? "NPU run timed out after " +
                           std::to_string(run_timeout.count()) + " ms"
                     : "NPU run did not complete";
    std::fprintf(stderr, "NPU: %s\n", last_error.c_str());
    return false;
  }

  // Waits for the oldest outstanding run (caller holds `mutex`).
  void ReapOldest() {
    auto& [run, c] = pending.front();
    Trace("reap_begin", c);
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = WaitRun(run);
    run_wait_us += Since(t0);
    if (!ok) {
      abandoned.push_back(std::move(run));
    }
    pending.pop_front();
    Trace("reap_done", c, ok);
    if (ok) {
      CopyFromNpu(c);
    }
  }

  bool Heartbeat() {
    const std::lock_guard lock(mutex);
    if (failed || heartbeat_a < 0) {
      return false;
    }
    if (!pending.empty()) {
      return true;  // real jobs are outstanding: the device is awake
    }
    // A prefill submits every layer; the heartbeat only matters in a real lull.
    if (NowMs() - last_submit_ms.load() < kHeartbeatQuietMs) {
      return true;
    }
    try {
      Trace("heartbeat_fire");
      auto run =
          kernel(kOpcode, instructions, instruction_words, NpuBo(heartbeat_a),
                 NpuBo(heartbeat_b), NpuBo(heartbeat_c));
      if (WaitRun(run)) {
        return true;
      }
      abandoned.push_back(std::move(run));
      return false;
    } catch (const std::exception& e) {
      Fail(e);
      return false;
    }
  }

  void StopKeepalive() {
    {
      const std::lock_guard lock(keepalive_mutex);
      keepalive_stop = true;
    }
    keepalive_wake.notify_all();
    if (keepalive.joinable()) {
      keepalive.join();
    }
  }

  void StopWorker() {
    {
      const std::lock_guard lock(queue_mutex);
      worker_stop = true;
    }
    queue_wake.notify_all();
    if (worker.joinable()) {
      worker.join();
    }
  }

  void WorkerLoop() {
    std::unique_lock lock(queue_mutex);
    for (;;) {
      queue_wake.wait(lock, [this] { return worker_stop || !queue.empty(); });
      if (queue.empty()) {
        return;
      }
      Call call = std::move(queue.front());
      queue.pop_front();
      lock.unlock();
      Process(call);
      lock.lock();
      ++completed;
      queue_idle.notify_all();
    }
  }

  void Enqueue(Call call) {
    Trace("enqueue", call.a_id);
    {
      const std::lock_guard lock(queue_mutex);
      queue.push_back(std::move(call));
      ++enqueued;
    }
    queue_wake.notify_one();
  }

  void Process(const Call& call) {
    Trace("process_wait_lock", call.a_id);
    const std::lock_guard lock(mutex);
    Trace("process_begin", call.a_id, static_cast<long>(call.b_ids.size()));
    const auto t0 = std::chrono::steady_clock::now();
    last_submit_ms = NowMs();
    if (failed) {
      return;
    }
    try {
      CopyToNpu(call.a_id);
      for (std::size_t i = 0; i < call.b_ids.size() && !failed; ++i) {
        while (pending.size() >= kMaxInFlight) {
          ReapOldest();
        }
        const auto t_launch = std::chrono::steady_clock::now();
        pending.emplace_back(
            kernel(kOpcode, instructions, instruction_words, NpuBo(call.a_id),
                   NpuBo(call.b_ids[i]), NpuBo(call.c_ids[i])),
            call.c_ids[i]);
        launch_us += Since(t_launch);
      }
      while (!pending.empty()) {
        ReapOldest();
      }
    } catch (const std::exception& e) {
      Fail(e);
      pending.clear();
    }
    submit_us += std::chrono::duration<double, std::micro>(
                     std::chrono::steady_clock::now() - t0)
                     .count();
    Trace("process_end", call.a_id);
  }

  void WaitNow() {
    Trace("wait_begin");
    const auto t0 = std::chrono::steady_clock::now();
    {
      std::unique_lock lock(queue_mutex);
      queue_idle.wait(lock, [this] { return completed == enqueued; });
    }
    wait_us += std::chrono::duration<double, std::micro>(
                   std::chrono::steady_clock::now() - t0)
                   .count();
    Trace("wait_end");
    if (stats && ++waits % 36 == 0) {
      std::fprintf(stderr,
                   "npu stats: %llu layers | worker %.0f us (A copy %.0f, "
                   "launch %.0f, NPU wait %.0f, C copy %.0f), stream blocked "
                   "in wait %.0f us (per layer averages)\n",
                   static_cast<unsigned long long>(waits), submit_us / waits,
                   copy_a_us / waits, launch_us / waits, run_wait_us / waits,
                   copy_c_us / waits, wait_us / waits);
    }
  }
};

NpuGemm::NpuGemm() = default;
NpuGemm::~NpuGemm() = default;

std::unique_ptr<NpuGemm> NpuGemm::Create(const Config& config,
                                         std::string* error) {
  auto fail = [&](const std::string& message) -> std::unique_ptr<NpuGemm> {
    if (error != nullptr) {
      *error = message;
    }
    return nullptr;
  };
  std::unique_ptr<NpuGemm> gemm(new NpuGemm());
  gemm->impl_ = std::make_unique<Impl>();
  Impl& i = *gemm->impl_;
  i.config = config;
  try {
    i.device = xrt::device(0u);
    const xrt::xclbin xclbin{std::string(config.xclbin)};
    i.device.register_xclbin(xclbin);
    i.context = xrt::hw_context(i.device, xclbin.get_uuid());
    i.kernel = xrt::kernel(i.context, xclbin.get_kernels().at(0).get_name());
    std::ifstream file(config.insts, std::ios::binary | std::ios::ate);
    if (!file) {
      return fail("NPU instruction file not readable: " + config.insts);
    }
    const auto size = static_cast<std::size_t>(file.tellg());
    file.seekg(0);
    std::vector<char> words(size);
    file.read(words.data(), static_cast<std::streamsize>(size));
    i.instruction_words = size / sizeof(int);
    i.instructions =
        xrt::bo(i.device, size, XCL_BO_FLAGS_CACHEABLE, i.kernel.group_id(1));
    std::memcpy(i.instructions.map<void*>(), words.data(), size);
    i.instructions.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    i.group_a = i.kernel.group_id(3);
  } catch (const std::exception& e) {
    return fail(std::string("NPU setup failed: ") + e.what());
  }
  // Wake the NPU now, while the GPU is still light, and keep it awake.
  i.heartbeat_a = gemm->NewBuffer(gemm->ABytes(), error);
  i.heartbeat_b = gemm->NewBuffer(gemm->BBytes(), error);
  i.heartbeat_c = gemm->NewBuffer(gemm->CBytes(), error);
  if (i.heartbeat_a < 0 || i.heartbeat_b < 0 || i.heartbeat_c < 0) {
    return nullptr;
  }
  for (int id : {i.heartbeat_a, i.heartbeat_b}) {
    std::memset(i.buffers[id].host, 0, i.buffers[id].bytes);
    i.CopyToNpu(id);
  }
  if (!i.Heartbeat()) {
    return fail("NPU did not run its first job: " + i.last_error);
  }
  Impl* impl = &i;
  if (const char* t = std::getenv("GUFO_NPU_TIMEOUT_MS"); t != nullptr) {
    i.run_timeout = std::chrono::milliseconds(std::max(1, std::atoi(t)));
  }
  i.worker = std::thread([impl] { impl->WorkerLoop(); });
  const char* keepalive_env = std::getenv("GUFO_NPU_KEEPALIVE");
  if (keepalive_env != nullptr && keepalive_env[0] == '0') {
    return gemm;
  }
  i.keepalive = std::thread([impl] {
    std::unique_lock lock(impl->keepalive_mutex);
    while (!impl->keepalive_stop) {
      impl->keepalive_wake.wait_for(lock, std::chrono::milliseconds(1000));
      if (impl->keepalive_stop) {
        break;
      }
      lock.unlock();
      impl->Heartbeat();
      lock.lock();
    }
  });
  return gemm;
}

std::size_t NpuGemm::ABytes() const noexcept {
  return std::size_t{impl_->config.tokens} * impl_->config.k / 8 * kBytesPer8;
}
std::size_t NpuGemm::BBytes() const noexcept {
  return std::size_t{impl_->config.k} * impl_->config.n / 8 * kBytesPer8;
}
std::size_t NpuGemm::CBytes() const noexcept {
  return std::size_t{impl_->config.tokens} * impl_->config.n / 8 * kBytesPer8;
}
std::uint32_t NpuGemm::tokens() const noexcept {
  return impl_->config.tokens;
}
std::uint32_t NpuGemm::k() const noexcept {
  return impl_->config.k;
}
std::uint32_t NpuGemm::n() const noexcept {
  return impl_->config.n;
}

int NpuGemm::NewBuffer(std::size_t bytes, std::string* error) {
  Impl& i = *impl_;
  void* host = nullptr;
  // Isolate mode wants ordinary cached pages on the CPU side; the GPU path is
  // flushed by kernel boundaries.
  const unsigned flags =
      hipHostMallocMapped | (i.isolate ? hipHostMallocNonCoherent : 0U);
  if (hipHostMalloc(&host, bytes, flags) != hipSuccess) {
    if (error != nullptr) {
      *error = "hipHostMalloc failed for an NPU buffer";
    }
    return -1;
  }
  try {
    Impl::Buffer buffer;
    buffer.host = host;
    buffer.bytes = bytes;
    if (hipHostGetDevicePointer(&buffer.device, host, 0) != hipSuccess) {
      throw std::runtime_error("hipHostGetDevicePointer failed");
    }
    if (i.isolate) {
      buffer.private_bo =
          xrt::bo(i.device, bytes, XRT_BO_FLAGS_HOST_ONLY, i.group_a);
      buffer.private_host = buffer.private_bo.map<void*>();
    } else {
      buffer.shared_bo =
          xrt::bo(i.device, host, bytes, XRT_BO_FLAGS_HOST_ONLY, i.group_a);
    }
    i.buffers.push_back(std::move(buffer));
  } catch (const std::exception& e) {
    (void)hipHostFree(host);
    if (error != nullptr) {
      *error = std::string("NPU buffer setup failed: ") + e.what();
    }
    return -1;
  }
  return static_cast<int>(i.buffers.size()) - 1;
}

bool NpuGemm::Healthy() const noexcept {
  return !impl_->failed;
}

void* NpuGemm::Host(int id) const {
  return impl_->buffers.at(id).host;
}

void* NpuGemm::Device(int id) const {
  return impl_->buffers.at(id).device;
}

void NpuGemm::FinalizeWeights(int id) {
  const std::lock_guard lock(impl_->mutex);
  impl_->CopyToNpu(id);
}

bool NpuGemm::Submit(hipStream_t stream, int a_id, const int* b_ids,
                     const int* c_ids, int count) {
  Trace("submit_host", a_id, count);
  auto* call = new Call{
      impl_.get(), a_id, {b_ids, b_ids + count}, {c_ids, c_ids + count}};
  const hipError_t status = hipLaunchHostFunc(
      stream,
      [](void* data) {
        std::unique_ptr<Call> c(static_cast<Call*>(data));
        static_cast<Impl*>(c->impl)->Enqueue(std::move(*c));
      },
      call);
  if (status != hipSuccess) {
    delete call;
    return false;
  }
  return true;
}

void* NpuGemm::Staged(hipStream_t /*stream*/, int id) {
  return impl_->buffers.at(id).device;
}

bool NpuGemm::Wait(hipStream_t stream) {
  Trace("wait_host");
  return hipLaunchHostFunc(
             stream, [](void* data) { static_cast<Impl*>(data)->WaitNow(); },
             impl_.get()) == hipSuccess;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
