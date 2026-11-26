#include "src/core/platform/net.hpp"

#include <algorithm>
#include <limits>

namespace gufo::net {

void CloseSocket(Socket socket) noexcept {
#if defined(_WIN32)
  closesocket(socket);
#else
  ::close(socket);
#endif
}

Socket CreateStreamSocket(int address_family) noexcept {
  return ::socket(address_family, SOCK_STREAM, 0);
}

std::intptr_t SendNoSignal(Socket socket, const void* data, std::size_t bytes) {
#if defined(_WIN32)
  // Winsock's send takes an int length; clamp each call so oversized buffers
  // narrow safely. The send loops handle partial writes.
  const int chunk = static_cast<int>(std::min<std::size_t>(
      bytes, static_cast<std::size_t>(std::numeric_limits<int>::max())));
  return ::send(socket, static_cast<const char*>(data), chunk, 0);
#else
#ifdef MSG_NOSIGNAL
  return ::send(socket, data, bytes, MSG_NOSIGNAL);
#else
  return ::send(socket, data, bytes, 0);
#endif
#endif
}

std::intptr_t Receive(Socket socket, void* data, std::size_t bytes,
                      int flags) noexcept {
#if defined(_WIN32)
  // Winsock's recv takes an int length; clamp each call so oversized reads
  // narrow safely.
  const int chunk = static_cast<int>(std::min<std::size_t>(
      bytes, static_cast<std::size_t>(std::numeric_limits<int>::max())));
  return ::recv(socket, static_cast<char*>(data), chunk, flags);
#else
  return ::recv(socket, data, bytes, flags);
#endif
}

bool ShutdownSocket(Socket socket, ShutdownHow how) noexcept {
#if defined(_WIN32)
  const int mode = how == ShutdownHow::kRead
                       ? SD_RECEIVE
                       : (how == ShutdownHow::kWrite ? SD_SEND : SD_BOTH);
  return ::shutdown(socket, mode) == 0;
#else
  const int mode = how == ShutdownHow::kRead
                       ? SHUT_RD
                       : (how == ShutdownHow::kWrite ? SHUT_WR : SHUT_RDWR);
  return ::shutdown(socket, mode) == 0;
#endif
}

bool SetSocketTimeouts(Socket socket, int timeout_ms) noexcept {
#if defined(_WIN32)
  const DWORD ms = static_cast<DWORD>(timeout_ms);
  const bool received =
      setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO,
                 reinterpret_cast<const char*>(&ms), sizeof(ms)) == 0;
  const bool sent =
      setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO,
                 reinterpret_cast<const char*>(&ms), sizeof(ms)) == 0;
  return received && sent;
#else
  timeval tv{};
  tv.tv_sec = timeout_ms / 1000;
  tv.tv_usec = (timeout_ms % 1000) * 1000;
  const bool received =
      ::setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) == 0;
  const bool sent =
      ::setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) == 0;
  return received && sent;
#endif
}

}  // namespace gufo::net
