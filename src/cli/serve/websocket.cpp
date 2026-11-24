#include "src/cli/serve/websocket.hpp"

#include <gufo/http_server.h>
#include <openssl/evp.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "src/cli/serve/audio_stream.hpp"

namespace gufo::server {

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

WebSocket::WebSocket(GufoWsSession* session) : session_(session) {}

WebSocket::~WebSocket() {
  Close();
}

bool WebSocket::Receive(std::string* text) {
  if (session_ == nullptr || text == nullptr) {
    return false;
  }
  text->clear();
  const auto sink = [](void* context, const std::uint8_t* data,
                       std::size_t len) noexcept {
    auto* out = static_cast<std::string*>(context);
    out->assign(reinterpret_cast<const char*>(data), len);
  };
  return gufo_ws_session_receive(session_, sink, text) != 0;
}

bool WebSocket::SendText(std::string_view text) {
  if (session_ == nullptr) {
    return false;
  }
  return gufo_ws_session_send_text(
             session_, reinterpret_cast<const std::uint8_t*>(text.data()),
             text.size()) != 0;
}

bool WebSocket::SendBinary(std::string_view bytes) {
  if (session_ == nullptr) {
    return false;
  }
  return gufo_ws_session_send_binary(
             session_, reinterpret_cast<const std::uint8_t*>(bytes.data()),
             bytes.size()) != 0;
}

void WebSocket::Close(std::uint16_t code) {
  if (session_ != nullptr) {
    gufo_ws_session_close(session_, code);
  }
}

bool WebSocket::cancelled() const {
  return session_ == nullptr || gufo_ws_session_is_closed(session_) != 0;
}

}  // namespace gufo::server
