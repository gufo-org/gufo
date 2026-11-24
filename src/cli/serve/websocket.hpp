#ifndef GUFO_SERVER_WEBSOCKET_HPP_
#define GUFO_SERVER_WEBSOCKET_HPP_

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

#include "src/cli/serve/http_server.hpp"

struct GufoWsSession;

namespace gufo::server {

// Thin C++ wrapper over the Rust `GufoWsSession` driven by hyper's upgraded
// stream and `gufo-websocket`'s RFC 6455 frame reader.
class WebSocket {
public:
  explicit WebSocket(GufoWsSession* session);
  ~WebSocket();
  WebSocket(const WebSocket&) = delete;
  WebSocket& operator=(const WebSocket&) = delete;
  bool Receive(std::string* text);
  bool SendText(std::string_view text);
  bool SendBinary(std::string_view bytes);
  void Close(std::uint16_t code = 1000);
  bool cancelled() const;

private:
  GufoWsSession* session_;
};

bool IsWebSocketUpgrade(const HttpRequest& request);
HttpResponse UpgradeWebSocket(const HttpRequest& request,
                              std::function<void(WebSocket&)> handler);

}  // namespace gufo::server
#endif  // GUFO_SERVER_WEBSOCKET_HPP_
