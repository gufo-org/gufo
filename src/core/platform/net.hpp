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

/// The one-time WinSock startup the HTTP server performs before opening its
/// listener; a no-op on POSIX. Throws std::system_error on failure.
void EnsureSocketRuntimeStarted();

/// Close a socket: closesocket on Windows, close on POSIX.
void CloseSocket(Socket socket) noexcept;

/// Create a blocking TCP stream socket for the address family (AF_INET or
/// AF_INET6). Returns kInvalidSocket on failure.
Socket CreateStreamSocket(int address_family) noexcept;

/// poll()/WSAPoll() with the platform's native timeout unit.
int PollSocket(void* descriptors, std::uint32_t count, int timeout_ms);

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

/// Accept a client socket from `listener`, ensuring the accepted socket is in
/// blocking mode on platforms where non-blocking state is inherited.
Socket AcceptClientSocket(Socket listener) noexcept;

/// Poll a single socket for readability. Returns >0 if readable or
/// closed/error, 0 on timeout, and <0 on poll error.
int PollReadable(Socket socket, int timeout_ms) noexcept;

/// Non-blocking check whether the remote peer has closed or reset the socket.
bool IsPeerDisconnected(Socket socket) noexcept;

/// Last socket error code on the calling thread (errno on POSIX,
/// WSAGetLastError() on Windows).
int LastSocketError() noexcept;

/// Whether the last socket operation failed due to signal interruption.
bool IsSocketInterrupted() noexcept;

/// Whether the last socket operation timed out or would block.
bool IsSocketWouldBlockOrTimeout() noexcept;

/// Whether a failed accept() call is transient and the accept loop should
/// retry.
bool IsSocketAcceptRetryable() noexcept;

/// Enable SO_REUSEADDR on `socket`.
bool SetReuseAddr(Socket socket) noexcept;

/// Configure address reuse for a temporary holding listener (`SO_REUSEADDR` on
/// POSIX; `SO_EXCLUSIVEADDRUSE` on Windows so a second server cannot
/// double-bind the held port before the holder closes it).
bool SetExclusiveListenerReuse(Socket socket) noexcept;

/// Disable IPV6_V6ONLY on an AF_INET6 socket so dual-stack binds work when
/// supported by the OS.
bool ClearIpv6Only(Socket socket) noexcept;

/// Mark a socket non-blocking; returns false on failure.
bool SetNonBlocking(Socket socket) noexcept;

/// Mark a socket blocking; returns false on failure. Required for sockets
/// accepted from a non-blocking listener on Windows, where they inherit the
/// listener's mode (POSIX accepted sockets never inherit O_NONBLOCK).
bool SetBlocking(Socket socket) noexcept;

/// Disable Nagle on a connected stream socket; returns false on failure.
bool SetNoDelay(Socket socket) noexcept;

/// Set receive/send timeouts in milliseconds; returns false on failure.
bool SetSocketTimeouts(Socket socket, int timeout_ms) noexcept;

/// Set only the send timeout in milliseconds; the receive timeout stays at
/// the caller's idle limit. Returns false on failure.
bool SetSendTimeout(Socket socket, int timeout_ms) noexcept;

/// Read the receive timeout back in milliseconds; 0 means no timeout is
/// set, -1 means the query failed.
long long SocketRecvTimeoutMs(Socket socket) noexcept;

}  // namespace gufo::net

#endif  // GUFO_CORE_PLATFORM_NET_HPP_
