#include "src/cli/serve/trace.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <utility>

#include "src/cli/serve/logging.hpp"
#include "src/core/utf8.hpp"

namespace gufo::server {
namespace {

std::mutex& TraceMutex() {
  static std::mutex mutex;
  return mutex;
}

// Guarded by TraceMutex. `Enabled()` reads the flag below instead, so call
// sites that only ask whether to build a record never contend for the lock.
int& TraceFd() {
  static int fd = -1;
  return fd;
}

std::atomic<bool>& TraceEnabled() {
  static std::atomic<bool> enabled{false};
  return enabled;
}

thread_local std::string current_request;

// A trace failure never changes a request's outcome. Only the first is
// reported: a full disk would otherwise warn once per request.
void ReportWriteFailure(std::string_view reason) {
  static std::atomic<bool> reported{false};
  if (!reported.exchange(true, std::memory_order_relaxed)) {
    Logger::Warn("trace",
                 "event=trace_write_failed reason=" + std::string(reason));
  }
}

// Local time with milliseconds: the log timestamp plus enough resolution to
// order a request's records.
std::string CurrentTimestamp() {
  const auto now = std::chrono::system_clock::now();
  const auto seconds = std::chrono::system_clock::to_time_t(now);
  const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now.time_since_epoch())
                          .count() %
                      1000;
  std::tm calendar{};
  ::localtime_r(&seconds, &calendar);
  std::array<char, 40> buffer{};
  const auto written = std::strftime(buffer.data(), buffer.size(),
                                     "%Y-%m-%d %H:%M:%S", &calendar);
  const int suffix =
      std::snprintf(buffer.data() + written, buffer.size() - written, ".%03d",
                    static_cast<int>(millis));
  return {buffer.data(), written + static_cast<std::size_t>(suffix)};
}

}  // namespace

std::optional<std::string> Trace::Open(const std::string& path) {
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC,
                        S_IRUSR | S_IWUSR);
  if (fd < 0) {
    return std::string(std::strerror(errno));
  }
  const std::lock_guard lock(TraceMutex());
  if (TraceFd() >= 0) {
    ::close(TraceFd());
  }
  TraceFd() = fd;
  TraceEnabled().store(true, std::memory_order_release);
  return std::nullopt;
}

void Trace::Close() {
  const std::lock_guard lock(TraceMutex());
  TraceEnabled().store(false, std::memory_order_release);
  if (TraceFd() >= 0) {
    ::close(TraceFd());
    TraceFd() = -1;
  }
}

bool Trace::Enabled() {
  return TraceEnabled().load(std::memory_order_acquire);
}

json::Value Trace::Record(std::string_view event, std::string_view request_id) {
  json::Value record = json::Value::object();
  record["time"] = CurrentTimestamp();
  record["event"] = std::string(event);
  if (!request_id.empty()) {
    record["request"] = std::string(request_id);
  }
  return record;
}

void Trace::Write(const json::Value& record) {
  if (!Enabled()) {
    return;
  }
  std::string text;
  try {
    text = record.dump();
  } catch (const std::exception& error) {
    ReportWriteFailure(error.what());
    return;
  }
  text += '\n';
  // One locked write per record keeps lines whole when requests finish
  // concurrently. A failed write drops the record rather than stalling a
  // request on a full disk.
  const std::lock_guard lock(TraceMutex());
  const int fd = TraceFd();
  if (fd < 0) {
    return;
  }
  // Every writer holds the lock, so the record starts at the current end. A
  // pipe or terminal cannot seek, and a torn record there cannot be undone.
  const off_t start = ::lseek(fd, 0, SEEK_END);
  std::string_view pending = text;
  while (!pending.empty()) {
    const ssize_t written = ::write(fd, pending.data(), pending.size());
    if (written < 0 && errno == EINTR) {
      continue;
    }
    if (written <= 0) {
      const std::string reason =
          written < 0 ? std::strerror(errno) : "short write";
      // A torn prefix would join the next record into one invalid line, so it
      // is cut back to the record's start. A sink that cannot be cut back
      // would corrupt every later record, and is closed instead.
      if (pending.size() == text.size() ||
          (start >= 0 && ::ftruncate(fd, start) == 0)) {
        ReportWriteFailure(reason);
        return;
      }
      ::close(fd);
      TraceFd() = -1;
      TraceEnabled().store(false, std::memory_order_release);
      Logger::Warn("trace", "event=trace_disabled reason=" + reason);
      return;
    }
    pending.remove_prefix(static_cast<std::size_t>(written));
  }
}

std::string Trace::Text(std::string_view bytes) {
  return core::Utf8Decoder{}.Push(bytes, true);
}

Trace::RequestScope::RequestScope(std::string_view request_id)
    : previous_(std::exchange(current_request, std::string(request_id))) {}

Trace::RequestScope::~RequestScope() {
  current_request = std::move(previous_);
}

std::string Trace::CurrentRequest() {
  return current_request;
}

}  // namespace gufo::server
