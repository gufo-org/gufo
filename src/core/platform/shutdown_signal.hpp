#ifndef GUFO_CORE_PLATFORM_SHUTDOWN_SIGNAL_HPP_
#define GUFO_CORE_PLATFORM_SHUTDOWN_SIGNAL_HPP_

#include <csignal>

#if !defined(_WIN32)
#include <cstring>
#endif

namespace gufo::platform {

/// RAII install of a process shutdown handler (SIGINT/SIGTERM on POSIX,
/// CTRL_C/CTRL_CLOSE/CTRL_LOGOFF/CTRL_SHUTDOWN console events on Windows).
///
/// The handler receives the POSIX signal number on Linux; Windows console
/// events are mapped onto the matching constants (SIGINT/SIGTERM). Handlers
/// must be async-signal-safe: an atomic store and nothing more.
class ShutdownSignalGuard {
public:
  using Handler = void (*)(int);

  ShutdownSignalGuard() = default;
  explicit ShutdownSignalGuard(Handler handler);
  ~ShutdownSignalGuard();
  ShutdownSignalGuard(const ShutdownSignalGuard&) = delete;
  ShutdownSignalGuard& operator=(const ShutdownSignalGuard&) = delete;
  ShutdownSignalGuard(ShutdownSignalGuard&& other) noexcept;
  ShutdownSignalGuard& operator=(ShutdownSignalGuard&& other) noexcept;

private:
#if !defined(_WIN32)
  struct sigaction previous_interrupt_{};
  struct sigaction previous_terminate_{};
  bool installed_{false};
#else
  bool installed_{false};
#endif
};

/// Called by the server once its shutdown path finished draining workers.
/// Releases the close/logoff/shutdown event handler's bounded wait so the
/// process can exit before the OS deadline expires. No-op on POSIX.
void NotifyShutdownHandled() noexcept;

/// Ignore SIGPIPE for the process (POSIX); a no-op on Windows, where Winsock
/// reports WSAECONNRESET instead of raising a signal.
inline void IgnoreSocketPipeSignal() noexcept {
#if !defined(_WIN32)
  std::signal(SIGPIPE, SIG_IGN);
#endif
}

/// Installs an async-signal-safe fatal signal reporter on POSIX (no-op on
/// Windows).
void InstallFatalSignalReporter() noexcept;

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_SHUTDOWN_SIGNAL_HPP_
