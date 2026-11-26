#ifndef GUFO_CORE_PLATFORM_NET_HPP_
#define GUFO_CORE_PLATFORM_NET_HPP_

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
// clang-format off
// winsock2.h must precede windows.h (which would pull in winsock 1), and
// iphlpapi.h needs windows.h first; keep the chain unsorted. The macro
// guards keep windows.h's min/max macros out of std:: calls downstream.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <iphlpapi.h>
// clang-format on
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace gufo::net {

// Blocking TCP client sockets for the in-process HTTP and WebSocket tests.

#if defined(_WIN32)
using Socket = SOCKET;

/// Length type for socket address calls: int on Windows.
using SocklenT = int;
#else
using Socket = int;

/// Length type for socket address calls: socklen_t on POSIX.
using SocklenT = socklen_t;
#endif

/// Returned by CreateStreamSocket on failure.
#if defined(_WIN32)
inline constexpr Socket kInvalidSocket = INVALID_SOCKET;
#else
inline constexpr Socket kInvalidSocket = -1;
#endif

/// Close a socket: closesocket on Windows, close on POSIX.
void CloseSocket(Socket socket) noexcept;

/// Create a blocking TCP stream socket for the address family (AF_INET or
/// AF_INET6). Returns kInvalidSocket on failure.
Socket CreateStreamSocket(int address_family) noexcept;

/// Send that never raises SIGPIPE (MSG_NOSIGNAL on POSIX; default on Winsock).
std::intptr_t SendNoSignal(Socket socket, const void* data, std::size_t bytes);

/// Receive up to `bytes` from the socket into `data`, with optional message
/// flags. Returns the byte count, 0 when the peer closed, -1 on error.
std::intptr_t Receive(Socket socket, void* data, std::size_t bytes,
                      int flags = 0) noexcept;

/// How ShutdownSocket half-closes a connected socket: stop receiving, stop
/// sending, or stop both directions.
enum class ShutdownHow { kRead, kWrite, kReadWrite };

/// Half-close a connected socket; returns false on failure.
bool ShutdownSocket(Socket socket, ShutdownHow how) noexcept;

/// Set receive/send timeouts in milliseconds; returns false on failure.
bool SetSocketTimeouts(Socket socket, int timeout_ms) noexcept;

}  // namespace gufo::net

#endif  // GUFO_CORE_PLATFORM_NET_HPP_
