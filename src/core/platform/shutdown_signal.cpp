#include "src/core/platform/shutdown_signal.hpp"

#include <atomic>
#include <cerrno>
#include <csignal>
#include <stdexcept>
#include <system_error>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>

#include <cstring>
#endif

namespace gufo::platform {
namespace {

#if defined(_WIN32)
using Handler = ShutdownSignalGuard::Handler;
// Read by the console-handler thread; the atomic plus the local-copy
// pattern below keep an in-flight callback racing teardown well-defined:
// a callback that loaded the pointer before teardown completes its single
// atomic-store side effect on process-lifetime state, and one that loads
// after the null store returns without calling.
std::atomic<Handler> g_active_handler{nullptr};
std::atomic<bool> g_cleanup_complete{false};

BOOL WINAPI ConsoleEventRouter(DWORD event) noexcept {
  const Handler handler = g_active_handler.load(std::memory_order_acquire);
  if (handler == nullptr)
    return FALSE;
  // Map console events onto their closest POSIX signal numbers so the
  // server sees one uniform "shutdown requested" value.
  const int mapped =
      (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT) ? SIGINT : SIGTERM;
  handler(mapped);
  if (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT)
    return TRUE;
  // CTRL_CLOSE/CTRL_LOGOFF/CTRL_SHUTDOWN terminate the process when this
  // handler returns. Hold it while the server drains its workers, under
  // the OS deadline (5 s for close events).
  g_cleanup_complete.store(false, std::memory_order_release);
  for (int waited_ms = 0; waited_ms < 4000; waited_ms += 10) {
    if (g_cleanup_complete.load(std::memory_order_acquire))
      return TRUE;
    Sleep(10);
  }
  return TRUE;
}
#endif

}  // namespace

ShutdownSignalGuard::ShutdownSignalGuard(ShutdownSignalGuard&& other) noexcept
    :
#if !defined(_WIN32)
      previous_interrupt_(other.previous_interrupt_),
      previous_terminate_(other.previous_terminate_),
#endif
      installed_(std::exchange(other.installed_, false)) {
}

ShutdownSignalGuard& ShutdownSignalGuard::operator=(
    ShutdownSignalGuard&& other) noexcept {
  if (this != &other) {
    this->~ShutdownSignalGuard();
    new (this) ShutdownSignalGuard(std::move(other));
  }
  return *this;
}

ShutdownSignalGuard::ShutdownSignalGuard(Handler handler) {
#if defined(_WIN32)
  g_active_handler.store(handler, std::memory_order_release);
  if (!SetConsoleCtrlHandler(ConsoleEventRouter, TRUE)) {
    const DWORD error = GetLastError();
    g_active_handler = nullptr;
    throw std::system_error(static_cast<int>(error), std::system_category(),
                            "install console ctrl handler");
  }
  installed_ = true;
#else
  struct sigaction action{};
  action.sa_handler = handler;
  ::sigemptyset(&action.sa_mask);
  if (::sigaction(SIGINT, &action, &previous_interrupt_) != 0)
    throw std::system_error(errno, std::generic_category(),
                            "install SIGINT handler");
  if (::sigaction(SIGTERM, &action, &previous_terminate_) != 0) {
    const int error = errno;
    (void)::sigaction(SIGINT, &previous_interrupt_, nullptr);
    throw std::system_error(error, std::generic_category(),
                            "install SIGTERM handler");
  }
  installed_ = true;
#endif
}

ShutdownSignalGuard::~ShutdownSignalGuard() {
#if defined(_WIN32)
  if (installed_) {
    g_active_handler.store(nullptr, std::memory_order_release);
    SetConsoleCtrlHandler(ConsoleEventRouter, FALSE);
  }
#else
  if (installed_) {
    (void)::sigaction(SIGTERM, &previous_terminate_, nullptr);
    (void)::sigaction(SIGINT, &previous_interrupt_, nullptr);
  }
#endif
}

void NotifyShutdownHandled() noexcept {
#if defined(_WIN32)
  g_cleanup_complete.store(true, std::memory_order_release);
#endif
}

#if !defined(_WIN32)
namespace {

extern "C" void ReportFatalSignal(int number) {
  static constexpr char kPrefix[] =
      "[ERROR] [server] event=fatal_signal signal=";
  char digits[8];
  std::size_t length = 0;
  int value = number;
  if (value <= 0) {
    digits[length++] = '0';
  } else {
    char reversed[8];
    std::size_t count = 0;
    while (value > 0 && count < sizeof(reversed)) {
      reversed[count++] = static_cast<char>('0' + (value % 10));
      value /= 10;
    }
    while (count > 0) {
      digits[length++] = reversed[--count];
    }
  }
  const auto wrote_prefix =
      ::write(STDERR_FILENO, kPrefix, sizeof(kPrefix) - 1);
  const auto wrote_digits = ::write(STDERR_FILENO, digits, length);
  const auto wrote_nl = ::write(STDERR_FILENO, "\n", 1);
  (void)wrote_prefix;
  (void)wrote_digits;
  (void)wrote_nl;
  // Restore the default action and re-raise, so the exit status and any core
  // dump still describe the original fault.
  struct sigaction restore{};
  restore.sa_handler = SIG_DFL;
  (void)::sigemptyset(&restore.sa_mask);
  (void)::sigaction(number, &restore, nullptr);
  (void)::raise(number);
}

}  // namespace
#endif

void InstallFatalSignalReporter() noexcept {
#if !defined(_WIN32)
  struct sigaction action{};
  action.sa_handler = ReportFatalSignal;
  (void)::sigemptyset(&action.sa_mask);
  action.sa_flags = SA_RESETHAND;
  for (const int number : {SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGFPE}) {
    (void)::sigaction(number, &action, nullptr);
  }
#endif
}

}  // namespace gufo::platform
