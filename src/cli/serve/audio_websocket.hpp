#ifndef GUFO_SERVER_AUDIO_WEBSOCKET_HPP_
#define GUFO_SERVER_AUDIO_WEBSOCKET_HPP_

#include "src/cli/serve/http_server.hpp"

namespace gufo::server {
HttpResponse HandleTtsWebSocket(const HttpRequest& request,
                                TtsService& service);
HttpResponse HandleAsrWebSocket(const HttpRequest& request,
                                AsrService& service);
}  // namespace gufo::server

#endif  // GUFO_SERVER_AUDIO_WEBSOCKET_HPP_
