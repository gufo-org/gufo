#include "src/cli/serve/websocket.hpp"

#include <gufo/http.h>
#include <gufo/websocket.h>
#include <openssl/evp.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>

#include "src/cli/serve/audio_stream.hpp"
#include "src/core/platform/net.hpp"

namespace gufo::server {
namespace {
constexpr std::size_t kMaximumMessage = 4U << 20;
constexpr std::size_t kMaximumQueue = 8U << 20;

// Whether the comma-separated header value lists the lowercase token.
bool HasToken(std::string_view value, std::string_view token) {
  const int found = gufo_http_has_token(
      reinterpret_cast<const std::uint8_t*>(value.data()), value.size(),
      reinterpret_cast<const std::uint8_t*>(token.data()), token.size());
  if (found < 0)
    throw std::logic_error("Rust HTTP token check failed internally");
  return found != 0;
}
}  // namespace

bool IsWebSocketUpgrade(const HttpRequest& request) {
  return HasToken(request.header("Upgrade"), "websocket");
}

HttpResponse UpgradeWebSocket(const HttpRequest& request,
                              std::function<void(WebSocket&)> handler) {
  std::string decoded;
  const auto key = request.header("Sec-WebSocket-Key");
  if (request.method != "GET" || !IsWebSocketUpgrade(request) ||
      !HasToken(request.header("Connection"), "upgrade") ||
      request.header("Sec-WebSocket-Version") != "13" ||
      (!request.header("Content-Length").empty() &&
       request.header("Content-Length") != "0") ||
      !DecodeAudioBase64(key, 16, &decoded) || decoded.size() != 16) {
    return {
        .status = 400,
        .reason = "Bad Request",
        .body =
            R"({"error":{"message":"invalid WebSocket handshake","code":"invalid_websocket"}})"};
  }
  const auto input = key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned size = 0;
  if (EVP_Digest(input.data(), input.size(), digest.data(), &size, EVP_sha1(),
                 nullptr) != 1)
    throw std::runtime_error("WebSocket handshake digest failed");
  HttpResponse response;
  response.status = 101;
  response.reason = "Switching Protocols";
  response.headers = {
      {"Upgrade", "websocket"},
      {"Connection", "Upgrade"},
      {"Sec-WebSocket-Accept",
       EncodeAudioBase64(std::string_view(
           reinterpret_cast<const char*>(digest.data()), size))}};
  // Browser Realtime clients offer "realtime" (and carry their API key as a
  // second protocol); the handshake must select one they offered.
  if (HasToken(request.header("Sec-WebSocket-Protocol"), "realtime"))
    response.headers.emplace_back("Sec-WebSocket-Protocol", "realtime");
  response.websocket = std::move(handler);
  return response;
}

WebSocket::WebSocket(int fd, std::string buffered)
    : fd_(fd), buffered_(std::move(buffered)) {
  // Frame headers and audio payloads are separate writes; do not wait for a
  // delayed TCP acknowledgement before sending the payload.
  (void)net::SetNoDelay(fd_);
  // Bound backpressure from a client that stops consuming audio. Read timeout
  // stays at the server's idle timeout.
  (void)net::SetSendTimeout(fd_, 5'000);
  if (const long long recv_timeout_ms = net::SocketRecvTimeoutMs(fd_);
      recv_timeout_ms >= 0) {
    idle_ = std::chrono::milliseconds(recv_timeout_ms);
  }
  last_send_ = std::chrono::steady_clock::now().time_since_epoch().count();
  reader_ = std::jthread([this] { ReadLoop(); });
}

WebSocket::~WebSocket() {
  Close();
  // Cancellation can wake the consumer before the reader finishes its close
  // reply. Let that writer complete before shutting down the send half.
  reader_.join();
  (void)net::ShutdownSocket(fd_, net::ShutdownHow::kReadWrite);
}

bool WebSocket::Read(char* data, std::size_t size) {
  while (size != 0 && !closed_.load()) {
    if (offset_ < buffered_.size()) {
      const auto count = std::min(size, buffered_.size() - offset_);
      std::memcpy(data, buffered_.data() + offset_, count);
      offset_ += count;
      data += count;
      size -= count;
      continue;
    }
    const auto count = net::Receive(fd_, data, size, 0);
    if (count < 0 && net::IsSocketInterrupted())
      continue;
    if (count < 0 && net::IsSocketWouldBlockOrTimeout()) {
      // A client that only listens to streamed output is not idle.
      const std::chrono::steady_clock::duration since_send(
          std::chrono::steady_clock::now().time_since_epoch().count() -
          last_send_.load());
      if (since_send < idle_)
        continue;
      Close(1001);
      return false;
    }
    if (count <= 0)
      return false;
    data += count;
    size -= static_cast<std::size_t>(count);
  }
  return size == 0;
}

bool WebSocket::Send(std::uint8_t opcode, std::string_view bytes) {
  const std::lock_guard lock(send_mutex_);
  if (closed_.load() && opcode != 8)
    return false;
  std::string header(1, static_cast<char>(0x80 | opcode));
  if (bytes.size() < 126) {
    header.push_back(static_cast<char>(bytes.size()));
  } else {
    const auto count = bytes.size() <= 65535 ? 2U : 8U;
    header.push_back(static_cast<char>(count == 2 ? 126 : 127));
    const auto length = static_cast<std::uint64_t>(bytes.size());
    for (unsigned i = count; i != 0; --i)
      header.push_back(static_cast<char>((length >> ((i - 1) * 8)) & 255));
  }
  for (auto part : {std::string_view(header), bytes}) {
    while (!part.empty()) {
      const auto count = static_cast<long long>(
          net::SendNoSignal(fd_, part.data(), part.size()));
      if (count < 0 && net::IsSocketInterrupted())
        continue;
      if (count <= 0) {
        MarkClosed();
        (void)net::ShutdownSocket(fd_, net::ShutdownHow::kReadWrite);
        return false;
      }
      part.remove_prefix(static_cast<std::size_t>(count));
      last_send_ = std::chrono::steady_clock::now().time_since_epoch().count();
    }
  }
  return true;
}

bool WebSocket::SendText(std::string_view text) {
  return Send(1, text);
}
bool WebSocket::SendBinary(std::string_view bytes) {
  return Send(2, bytes);
}

void WebSocket::Close(std::uint16_t code) {
  if (!MarkClosed())
    return;
  const std::array<char, 2> payload{static_cast<char>(code >> 8),
                                    static_cast<char>(code & 255)};
  (void)Send(8, std::string_view(payload.data(), payload.size()));
  // Drain unread client bytes before half-closing: on Windows, shutting down
  // the receive direction with unread data queued aborts the connection with
  // RST and discards the close frame just queued above.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
  while (true) {
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now())
            .count();
    if (remaining <= 0 ||
        net::PollReadable(fd_, static_cast<int>(remaining)) <= 0)
      break;
    char sink[2048];
    if (net::Receive(fd_, sink, sizeof(sink), 0) <= 0)
      break;
  }
  (void)net::ShutdownSocket(fd_, net::ShutdownHow::kReadWrite);
}

bool WebSocket::MarkClosed() {
  bool changed;
  {
    // Receive checks this predicate under queue_mutex_. Changing it under the
    // same lock prevents a close notification racing between its check and
    // wait.
    const std::lock_guard lock(queue_mutex_);
    changed = !closed_.exchange(true);
  }
  ready_.notify_all();
  return changed;
}

bool WebSocket::Receive(std::string* text) {
  std::unique_lock lock(queue_mutex_);
  ready_.wait(lock, [&] { return closed_.load() || !messages_.empty(); });
  if (closed_.load())
    return false;
  *text = std::move(messages_.front());
  messages_.pop_front();
  queued_bytes_ -= text->size();
  return true;
}

void WebSocket::ReadLoop() {
  // Rust parses the frames; an exception from Read is rethrown here.
  struct ReadContext {
    WebSocket* socket;
    std::exception_ptr failure;
  } context{this, nullptr};
  const GufoWsReadFn read = [](void* opaque, std::uint8_t* data,
                               std::size_t size) noexcept -> int {
    auto& context = *static_cast<ReadContext*>(opaque);
    try {
      return context.socket->Read(reinterpret_cast<char*>(data), size) ? 1 : 0;
    } catch (...) {
      context.failure = std::current_exception();
      return 0;
    }
  };
  const std::unique_ptr<GufoWsReader, decltype(&gufo_ws_reader_free)> reader(
      gufo_ws_reader_new(kMaximumMessage), gufo_ws_reader_free);
  try {
    if (!reader)
      throw std::bad_alloc();
    while (!closed_.load()) {
      GufoWsEvent event{};
      gufo_ws_reader_next(reader.get(), read, &context, &event);
      if (context.failure)
        std::rethrow_exception(context.failure);
      const std::string_view payload(reinterpret_cast<const char*>(event.data),
                                     event.len);
      if (event.kind == GUFO_WS_END)
        break;
      if (event.kind == GUFO_WS_ERROR) {
        Close(event.code);
        return;
      }
      if (event.kind == GUFO_WS_CLOSE) {
        if (MarkClosed())
          (void)Send(8, payload);
        break;
      }
      if (event.kind == GUFO_WS_PING) {
        if (!Send(10, payload))
          break;
        continue;
      }
      if (event.kind != GUFO_WS_MESSAGE)
        throw std::logic_error("Rust WebSocket reader failed internally");
      {
        std::unique_lock lock(queue_mutex_);
        if (messages_.size() >= 128 ||
            payload.size() > kMaximumQueue - queued_bytes_) {
          lock.unlock();
          Close(1009);
          return;
        }
        queued_bytes_ += payload.size();
        messages_.emplace_back(payload);
      }
      ready_.notify_one();
    }
  } catch (const std::exception&) {
    Close(1011);
  }
  MarkClosed();
}

}  // namespace gufo::server
