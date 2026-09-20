#ifndef GUFO_SERVER_WEBSOCKET_HPP_
#define GUFO_SERVER_WEBSOCKET_HPP_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

#include "src/cli/serve/http_server.hpp"

namespace gufo::server {

// One reader owns frame parsing; inference consumes a bounded message queue.
// The reader handles close/ping even while inference or socket output is busy.
class WebSocket {
public:
  WebSocket(int fd, std::string buffered);
  ~WebSocket();
  WebSocket(const WebSocket&) = delete;
  WebSocket& operator=(const WebSocket&) = delete;
  bool Receive(std::string* text);
  bool SendText(std::string_view text);
  bool SendBinary(std::string_view bytes);
  void Close(std::uint16_t code = 1000);
  bool cancelled() const { return closed_.load(); }

private:
  bool MarkClosed();
  void ReadLoop();
  bool Read(char* data, std::size_t size);
  bool Send(std::uint8_t opcode, std::string_view bytes);
  int fd_;
  std::string buffered_;
  std::size_t offset_{0};
  std::atomic<bool> closed_{false};
  std::mutex send_mutex_;
  std::mutex queue_mutex_;
  std::condition_variable ready_;
  std::deque<std::string> messages_;
  std::size_t queued_bytes_{0};
  std::jthread reader_;
};

bool IsWebSocketUpgrade(const HttpRequest& request);
HttpResponse UpgradeWebSocket(const HttpRequest& request,
                              std::function<void(WebSocket&)> handler);

}  // namespace gufo::server
#endif  // GUFO_SERVER_WEBSOCKET_HPP_
