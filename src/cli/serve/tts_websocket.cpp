#include <algorithm>

#include "src/cli/serve/audio_tts_api.hpp"
#include "src/cli/serve/audio_websocket.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/websocket.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen3_tts/text_stream.hpp"

namespace gufo::server {

HttpResponse HandleTtsWebSocket(const HttpRequest& request,
                                TtsService& service) {
  return UpgradeWebSocket(request, [&service, id = request.request_id](
                                       WebSocket& socket) {
    auto config = json::Value::object();
    bool configured = false;
    bool in_utterance = false;
    bool streaming = false;
    std::string granularity = "none";
    std::string pending;
    std::size_t input_bytes = 0;
    std::size_t utterance = 0;
    std::size_t sentence = 0;
    const auto error = [&](std::string message) {
      auto event = json::Value::object();
      event["type"] = "error";
      event["message"] = std::move(message);
      return socket.SendText(event.dump());
    };
    const auto synthesize = [&](std::string text) {
      if (text.find_first_not_of(" \r\n\t") == std::string::npos)
        return true;
      if (sentence >= 64) {
        (void)error("utterance exceeds 64 speech segments");
        return false;
      }
      auto body = config;
      body["input"] = text;
      HttpRequest speech{.method = "POST",
                         .path = "/v1/audio/speech",
                         .body = body.dump(),
                         .is_cancelled = [&] { return socket.cancelled(); }};
      auto validation = ValidateAudioTtsRequest(speech, service);
      if (validation.status != 200) {
        (void)error(validation.body);
        return false;
      }
      auto start = json::Value::object();
      start["type"] = "audio.start";
      start["utterance_index"] = static_cast<std::int64_t>(utterance);
      start["sentence_index"] = static_cast<std::int64_t>(sentence);
      start["sentence_text"] = std::move(text);
      start["format"] = body.member_str("response_format");
      start["sample_rate"] = 24000;
      if (!socket.SendText(start.dump()))
        return false;
      auto response = HandleAudioTtsApiRequest(speech, service);
      if (response.status != 200) {
        (void)error(response.body);
        return false;
      }
      if (response.streaming_body) {
        std::string buffered;
        response.streaming_body([&](std::string_view bytes) {
          if (streaming)
            return socket.SendBinary(bytes);
          if (socket.cancelled())
            return false;
          buffered.append(bytes);
          return true;
        });
        if (response.stream_log && !response.stream_log->error_code.empty()) {
          (void)error(response.stream_log->error_code);
          return false;
        }
        if (!streaming && !socket.SendBinary(buffered))
          return false;
      } else if (!socket.SendBinary(response.body))
        return false;
      auto done = json::Value::object();
      done["type"] = "audio.done";
      done["utterance_index"] = static_cast<std::int64_t>(utterance);
      done["sentence_index"] = static_cast<std::int64_t>(sentence++);
      return socket.SendText(done.dump());
    };
    std::string raw;
    while (socket.Receive(&raw)) {
      try {
        const auto message = json::parse(raw);
        if (!message.is_object())
          throw std::invalid_argument("message must be an object");
        const auto type = message.member_str("type");
        if (type != "session.config") {
          for (const auto& [name, unused] : message.members()) {
            (void)unused;
            if (name != "type" && !(type == "input.text" && name == "text"))
              throw std::invalid_argument("unsupported speech stream field: " +
                                          name);
          }
        }
        if (type == "session.close")
          break;
        if (type == "session.config") {
          if (in_utterance)
            throw std::invalid_argument(
                "finish the utterance before reconfiguring");
          auto next = json::Value::object();
          std::string split = "none";
          bool stream_audio = false;
          for (const auto& [name, value] : message.members()) {
            if (name == "type")
              continue;
            if (name == "split_granularity") {
              if (!value.is_string() ||
                  (value.str() != "none" && value.str() != "sentence" &&
                   value.str() != "clause"))
                throw std::invalid_argument(
                    "split_granularity must be none, sentence or clause");
              split = value.str();
            } else if (name == "stream_audio") {
              if (!value.is_bool())
                throw std::invalid_argument("stream_audio must be boolean");
              stream_audio = value.as_bool();
            } else if (name == "word_timestamps") {
              if (!value.is_bool() || value.as_bool())
                throw std::invalid_argument("timestamps are not supported");
            } else if (name == "input" || name == "stream_format") {
              throw std::invalid_argument("use input.text and stream_audio");
            } else if (name == "ref_audio") {
              next["reference_audio"] = value;
            } else if (name == "ref_text") {
              next["reference_text"] = value;
            } else if (name == "x_vector_only_mode") {
              if (!value.is_bool())
                throw std::invalid_argument(
                    "x_vector_only_mode must be boolean");
              next["voice_clone_mode"] =
                  value.as_bool() ? "speaker_embedding_only" : "icl";
            } else if (name == "task_type") {
              const auto variant = service.variant();
              const std::string expected =
                  variant == models::qwen3_tts::ModelVariant::kBase ? "Base"
                  : variant == models::qwen3_tts::ModelVariant::kVoiceDesign
                      ? "VoiceDesign"
                      : "CustomVoice";
              if (!value.is_string() || value.str() != expected)
                throw std::invalid_argument(
                    "task_type does not match the loaded checkpoint");
            } else {
              next[name] = value;
            }
          }
          if (!next.find("model"))
            next["model"] = service.model_id();
          if (!next.find("response_format"))
            next["response_format"] = "wav";
          if (stream_audio && next.member_str("response_format") != "pcm")
            throw std::invalid_argument(
                "stream_audio requires response_format pcm");
          next["input"] = "validate";
          auto validation =
              ValidateAudioTtsRequest(HttpRequest{.method = "POST",
                                                  .path = "/v1/audio/speech",
                                                  .body = next.dump()},
                                      service);
          if (validation.status != 200)
            throw std::invalid_argument(validation.body);
          config = std::move(next);
          granularity = split;
          streaming = stream_audio;
          configured = true;
        } else if (type == "input.text") {
          if (!configured)
            throw std::invalid_argument("send session.config first");
          const auto* text = message.find("text");
          if (!text || !text->is_string())
            throw std::invalid_argument("input.text requires text");
          if (text->str().size() > (16U << 10) - input_bytes)
            throw std::length_error("utterance exceeds 16384 UTF-8 bytes");
          input_bytes += text->str().size();
          in_utterance = true;
          pending.append(text->str());
          for (auto& part : models::qwen3_tts::TakeSpeechSegments(
                   &pending, granularity, false))
            if (!synthesize(std::move(part))) {
              socket.Close(1011);
              return;
            }
        } else if (type == "input.done") {
          if (!configured)
            throw std::invalid_argument("send session.config first");
          for (auto& part : models::qwen3_tts::TakeSpeechSegments(
                   &pending, granularity, true))
            if (!synthesize(std::move(part))) {
              socket.Close(1011);
              return;
            }
          auto done = json::Value::object();
          done["type"] = "session.done";
          done["utterance_index"] = static_cast<std::int64_t>(utterance++);
          done["total_sentences"] = static_cast<std::int64_t>(sentence);
          if (!socket.SendText(done.dump()))
            return;
          Logger::Info("tts",
                       "request=" + id + " event=utterance_complete segments=" +
                           std::to_string(sentence) +
                           " input_bytes=" + std::to_string(input_bytes));
          sentence = 0;
          input_bytes = 0;
          in_utterance = false;
        } else {
          throw std::invalid_argument("unsupported speech stream event");
        }
      } catch (const std::exception& exception) {
        if (!error(exception.what()))
          return;
      }
    }
  });
}

}  // namespace gufo::server
