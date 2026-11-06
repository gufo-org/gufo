#ifndef GUFO_SERVER_REALTIME_AUDIO_HPP_
#define GUFO_SERVER_REALTIME_AUDIO_HPP_

#include "gufo/ext/asr.hpp"

namespace gufo::server {

// Realtime streams are extension contracts; models implement them.
using LiveTranscription = ext::LiveTranscription;
using VoiceConversation = ext::VoiceConversation;

}  // namespace gufo::server

#endif  // GUFO_SERVER_REALTIME_AUDIO_HPP_
