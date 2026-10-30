#ifndef GUFO_SERVER_AUDIO_WEBSOCKET_HPP_
#define GUFO_SERVER_AUDIO_WEBSOCKET_HPP_

#include "src/cli/serve/http_server.hpp"

namespace gufo::server {
HttpResponse HandleTtsWebSocket(const HttpRequest& request,
                                TtsService& service);
HttpResponse HandleAsrWebSocket(const HttpRequest& request,
                                AsrService& service);
// A spoken conversation (OpenAI Realtime "realtime" sessions) with a model
// that hears and speaks; see docs/models/lfm3-audio.
HttpResponse HandleVoiceWebSocket(const HttpRequest& request,
                                  AsrService& service);
}  // namespace gufo::server

#endif  // GUFO_SERVER_AUDIO_WEBSOCKET_HPP_
