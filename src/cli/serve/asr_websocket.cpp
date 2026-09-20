#include <optional>

#include "src/cli/serve/asr_service.hpp"
#include "src/cli/serve/audio_stream.hpp"
#include "src/cli/serve/audio_websocket.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/websocket.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen3_asr/audio.hpp"

namespace gufo::server {
namespace {
void Members(const json::Value& object,
             std::initializer_list<std::string_view> allowed) {
  if (!object.is_object())
    throw std::invalid_argument("configuration must be an object");
  for (const auto& [name, unused] : object.members()) {
    (void)unused;
    if (std::find(allowed.begin(), allowed.end(), name) == allowed.end())
      throw std::invalid_argument("unsupported transcription field: " + name);
  }
}
}  // namespace

HttpResponse HandleAsrWebSocket(const HttpRequest& request,
                                AsrService& service) {
  const auto model = request.query_param("model");
  const auto intent = request.query_param("intent");
  if ((!model.empty() && model != service.model_id() && model != "qwen3-asr") ||
      (!intent.empty() && intent != "transcription"))
    return {
        .status = 400,
        .reason = "Bad Request",
        .body =
            R"({"error":{"message":"unsupported realtime model or intent","code":"invalid_model"}})"};
  return UpgradeWebSocket(request, [&service, id = request.request_id](
                                       WebSocket& socket) {
    std::size_t sequence = 0;
    std::size_t item_sequence = 0;
    std::string previous_item;
    std::string audio_bytes;
    std::optional<std::string> language;
    std::string prompt;
    constexpr std::size_t maximum_bytes = 120U * 24000 * 2;
    auto config = json::parse(
        R"({"type":"transcription","audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"transcription":{},"turn_detection":null,"noise_reduction":null}}})");
    config["id"] = "sess_" + id;
    config["audio"]["input"]["transcription"]["model"] = service.model_id();
    const auto send = [&](json::Value event) {
      event["event_id"] = "event_" + id + "_" + std::to_string(++sequence);
      return socket.SendText(event.dump());
    };
    const auto session_event = [&](std::string type) {
      auto event = json::Value::object();
      event["type"] = std::move(type);
      event["session"] = config;
      return send(std::move(event));
    };
    const auto error = [&](std::string message, std::string client_event = {}) {
      auto event = json::Value::object();
      event["type"] = "error";
      event["error"] = json::Value::object();
      event["error"]["type"] = "invalid_request_error";
      event["error"]["code"] = "invalid_audio_event";
      event["error"]["message"] = std::move(message);
      if (!client_event.empty())
        event["error"]["event_id"] = std::move(client_event);
      return send(std::move(event));
    };
    if (!session_event("session.created"))
      return;
    std::string raw;
    while (socket.Receive(&raw)) {
      std::string client_event;
      try {
        const auto message = json::parse(raw);
        if (!message.is_object())
          throw std::invalid_argument("event must be an object");
        client_event = message.member_str("event_id");
        const auto type = message.member_str("type");
        if (type == "session.update") {
          Members(message, {"type", "event_id", "session"});
          if (!audio_bytes.empty())
            throw std::invalid_argument(
                "commit or clear audio before session.update");
          const auto* session = message.find("session");
          if (!session)
            throw std::invalid_argument("session.update requires session");
          Members(*session, {"type", "audio", "include"});
          if (const auto* kind = session->find("type");
              kind && (!kind->is_string() || kind->str() != "transcription"))
            throw std::invalid_argument(
                "only transcription sessions are supported");
          if (const auto* include = session->find("include");
              include && (!include->is_array() || !include->items().empty()))
            throw std::invalid_argument(
                "transcription logprobs are not supported");
          auto next = config;
          auto next_language = language;
          auto next_prompt = prompt;
          if (const auto* audio = session->find("audio")) {
            Members(*audio, {"input"});
            if (const auto* input = audio->find("input")) {
              Members(*input, {"format", "transcription", "turn_detection",
                               "noise_reduction"});
              if (const auto* format = input->find("format")) {
                Members(*format, {"type", "rate"});
                if (format->member_str("type") != "audio/pcm" ||
                    (format->find("rate") &&
                     (!format->find("rate")->is_number() ||
                      format->find("rate")->as_double() != 24000)))
                  throw std::invalid_argument(
                      "realtime input requires mono PCM16 at 24000 Hz");
              }
              for (const auto name : {"turn_detection", "noise_reduction"})
                if (const auto* option = input->find(name);
                    option && !option->is_null())
                  throw std::invalid_argument(std::string(name) +
                                              " must be null");
              if (const auto* transcription = input->find("transcription")) {
                Members(*transcription, {"model", "language", "prompt"});
                for (const auto& [name, value] : transcription->members()) {
                  if (!value.is_string())
                    throw std::invalid_argument(name + " must be a string");
                  if (name == "model" && value.str() != service.model_id() &&
                      value.str() != "qwen3-asr")
                    throw std::invalid_argument(
                        "unsupported transcription model");
                  if (name == "language")
                    next_language =
                        value.str().empty()
                            ? std::nullopt
                            : std::optional<std::string>{value.str()};
                  if (name == "prompt")
                    next_prompt = value.str();
                  next["audio"]["input"]["transcription"][name] = value;
                }
              }
            }
          }
          if (next_prompt.size() > 16384 ||
              (next_language && next_language->size() > 64))
            throw std::length_error(
                "transcription configuration exceeds its text limit");
          config = std::move(next);
          prompt = std::move(next_prompt);
          language = std::move(next_language);
          if (!session_event("session.updated"))
            return;
        } else if (type == "input_audio_buffer.append") {
          Members(message, {"type", "event_id", "audio"});
          const auto* audio = message.find("audio");
          std::string bytes;
          if (!audio || !audio->is_string() ||
              !DecodeAudioBase64(audio->str(),
                                 maximum_bytes - audio_bytes.size(), &bytes) ||
              bytes.size() % 2 != 0)
            throw std::invalid_argument(
                "invalid PCM16 audio or 120-second input buffer exceeded");
          audio_bytes.append(bytes);
        } else if (type == "input_audio_buffer.clear") {
          Members(message, {"type", "event_id"});
          audio_bytes.clear();
          auto event = json::Value::object();
          event["type"] = "input_audio_buffer.cleared";
          if (!send(std::move(event)))
            return;
        } else if (type == "input_audio_buffer.commit") {
          Members(message, {"type", "event_id"});
          if (audio_bytes.size() < 2400 * 2)
            throw std::invalid_argument(
                "commit requires at least 100 ms of audio");
          const auto item =
              "item_" + id + "_" + std::to_string(++item_sequence);
          auto committed = json::Value::object();
          committed["type"] = "input_audio_buffer.committed";
          committed["item_id"] = item;
          committed["previous_item_id"] = previous_item.empty()
                                              ? json::Value{}
                                              : json::Value(previous_item);
          if (!send(std::move(committed)))
            return;
          auto created = json::Value::object();
          created["type"] = "conversation.item.created";
          created["previous_item_id"] = previous_item.empty()
                                            ? json::Value{}
                                            : json::Value(previous_item);
          created["item"] = json::parse(
              R"({"type":"message","role":"user","status":"completed","content":[{"type":"input_audio","transcript":null}]})");
          created["item"]["id"] = item;
          if (!send(std::move(created)))
            return;
          previous_item = item;
          models::qwen3_asr::AudioBuffer audio{.sample_rate = 24000,
                                               .channels = 1};
          audio.samples.resize(audio_bytes.size() / 2);
          for (std::size_t i = 0; i < audio.samples.size(); ++i) {
            const auto bits = static_cast<std::uint16_t>(
                static_cast<unsigned char>(audio_bytes[2 * i]) |
                (static_cast<unsigned char>(audio_bytes[2 * i + 1]) << 8));
            audio.samples[i] = static_cast<std::int16_t>(bits) / 32768.0F;
          }
          audio_bytes.clear();
          const auto pcm = models::qwen3_asr::ResampleMono16k(audio);
          models::qwen3_asr::TranscriptionRequest transcription{
              .context = prompt, .language = language, .pcm16k = pcm};
          std::string published;
          transcription.on_text = [&](std::string_view text) {
            if (!text.starts_with(published))
              return false;
            if (text.size() == published.size())
              return !socket.cancelled();
            auto delta = json::Value::object();
            delta["type"] = "conversation.item.input_audio_transcription.delta";
            delta["item_id"] = item;
            delta["content_index"] = 0;
            delta["delta"] = std::string(text.substr(published.size()));
            published = text;
            return send(std::move(delta));
          };
          models::qwen3_asr::TranscriptionResult result;
          std::string failure;
          const bool ok = service.Transcribe(
              transcription, [&] { return socket.cancelled(); }, &result,
              &failure);
          if (socket.cancelled())
            return;
          auto completed = json::Value::object();
          completed["type"] =
              ok ? "conversation.item.input_audio_transcription.completed"
                 : "conversation.item.input_audio_transcription.failed";
          completed["item_id"] = item;
          completed["content_index"] = 0;
          if (ok)
            completed["transcript"] = result.text;
          else {
            completed["error"] = json::Value::object();
            completed["error"]["type"] = "server_error";
            completed["error"]["code"] = "transcription_failed";
            completed["error"]["message"] = failure;
          }
          if (!send(std::move(completed)))
            return;
          Logger::Info("asr",
                       "request=" + id + " event=transcription_complete item=" +
                           item + " chunks=" + std::to_string(result.chunks) +
                           " audio_samples=" + std::to_string(pcm.size()) +
                           " outcome=" + (ok ? "completed" : "failed"));
        } else {
          throw std::invalid_argument(
              "unsupported realtime transcription event");
        }
      } catch (const std::exception& exception) {
        if (!error(exception.what(), client_event))
          return;
      }
    }
  });
}

}  // namespace gufo::server
