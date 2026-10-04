// Process-local fault injection for device_loss.py. Never resets the GPU.
// HIP's stable C ABI uses opaque pointers and int error codes; keeping the
// hooks header-free lets the CPU test preset build this opt-in helper.
#include <dlfcn.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <string_view>
#include <thread>

namespace {
constexpr int kLaunchFailure = 719;
constexpr int kNotReady = 600;
std::atomic<bool> failed{false};
std::atomic<void*> probe_stream{nullptr};

std::string Path(std::string_view name) {
  const auto* directory = std::getenv("GUFO_TEST_DEVICE_LOSS_DIR");
  return directory ? std::string(directory) + "/" + std::string(name) : "";
}
bool Exists(std::string_view name) {
  const auto path = Path(name);
  return !path.empty() && ::access(path.c_str(), F_OK) == 0;
}
void Mark(std::string_view name) {
  std::ofstream file(Path(name));
  file << "observed\n";
}
bool Loss() {
  return Exists("loss");
}
bool Inject() {
  if (!Exists("armed") || failed.exchange(true))
    return false;
  Mark("injected");
  return true;
}
[[noreturn]] void Block() {
  for (;;)
    std::this_thread::sleep_for(std::chrono::seconds(1));
}
void Cleanup() {
  if (failed.load() && Loss()) {
    Mark("cleanup_attempted");
    // Isolate listener observation from cleanup ordering in the writer case.
    if (!Exists("block_writer"))
      Block();
  }
}
template<class Function>
Function Next(const char* name) {
  const auto address = ::dlsym(RTLD_NEXT, name);
  if (!address)
    ::_exit(126);
  return reinterpret_cast<Function>(address);
}
}  // namespace

extern "C" int hipMemcpyAsync(void* dst, const void* src, std::size_t bytes,
                              int kind, void* stream) {
  // Input/sampling uploads are generation work. Snapshot copies are optional
  // best-effort work whose failures may be swallowed before the scheduler;
  // racing their worker must not consume this single injected model error.
  constexpr int kHostToDevice = 1;
  if (kind == kHostToDevice && Inject())
    return kLaunchFailure;
  return Next<int (*)(void*, const void*, std::size_t, int, void*)>(
      "hipMemcpyAsync")(dst, src, bytes, kind, stream);
}
extern "C" int hipMemsetAsync(void* dst, int value, std::size_t bytes,
                              void* stream) {
  if (failed.load() && Exists("armed") && bytes == sizeof(std::uint32_t)) {
    Mark("probed");
    probe_stream = stream;
    if (Loss())
      return kLaunchFailure;
  } else {
    Cleanup();
  }
  return Next<int (*)(void*, int, std::size_t, void*)>("hipMemsetAsync")(
      dst, value, bytes, stream);
}
extern "C" int hipStreamQuery(void* stream) {
  if (stream == probe_stream.load() && Exists("armed") && Exists("timeout"))
    return kNotReady;
  return Next<int (*)(void*)>("hipStreamQuery")(stream);
}
extern "C" int hipMemset(void* dst, int value, std::size_t bytes) {
  Cleanup();
  return Next<int (*)(void*, int, std::size_t)>("hipMemset")(dst, value, bytes);
}
extern "C" int hipFree(void* pointer) {
  Cleanup();
  return Next<int (*)(void*)>("hipFree")(pointer);
}
extern "C" int hipGraphExecDestroy(void* graph) {
  Cleanup();
  return Next<int (*)(void*)>("hipGraphExecDestroy")(graph);
}
extern "C" int hipGraphDestroy(void* graph) {
  Cleanup();
  return Next<int (*)(void*)>("hipGraphDestroy")(graph);
}
extern "C" ssize_t send(int fd, const void* buffer, std::size_t bytes,
                        int flags) {
  const std::string_view data(static_cast<const char*>(buffer), bytes);
  if (Exists("block_writer") &&
      (data.starts_with("data: ") || data.starts_with("event: "))) {
    Mark("writer_blocked");
    Block();
  }
  return Next<ssize_t (*)(int, const void*, std::size_t, int)>("send")(
      fd, buffer, bytes, flags);
}
