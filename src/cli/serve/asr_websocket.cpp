#include <algorithm>
#include <cctype>
#include <memory>
#include <optional>

#include "src/cli/serve/asr_service.hpp"
#include "src/cli/serve/audio_stream.hpp"
#include "src/cli/serve/audio_websocket.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/voice_activity.hpp"
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
// Mono floats of little-endian PCM16 bytes.
std::vector<float> Pcm16(std::string_view bytes) {
  std::vector<float> samples(bytes.size() / 2);
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const auto bits = static_cast<std::uint16_t>(
        static_cast<unsigned char>(bytes[2 * i]) |
        (static_cast<unsigned char>(bytes[2 * i + 1]) << 8));
    samples[i] = static_cast<std::int16_t>(bits) / 32768.0F;
  }
  return samples;
}
// Live transcription models are English-only.
bool English(const std::optional<std::string>& language) {
  if (!language)
    return true;
  std::string lower = *language;
  for (auto& c : lower)
    c = char(std::tolower(static_cast<unsigned char>(c)));
  return lower == "en" || lower == "english";
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
  // Without an intent, models that hear and speak hold a conversation.
  if (intent.empty() && service.supports_voice())
    return HandleVoiceWebSocket(request, service);
  return UpgradeWebSocket(request, [&service, id = request.request_id](
                                       WebSocket& socket) {
    std::size_t sequence = 0;
    std::size_t item_sequence = 0;
    std::string previous_item;
    std::string audio_bytes;
    // Live transcription decodes text while audio arrives; the deltas belong
    // to the item the next commit creates. live_bytes counts its input.
    std::unique_ptr<LiveTranscription> live;
    std::size_t live_bytes = 0;
    std::optional<std::string> language;
    std::string prompt;
    // Server turn detection: speech commits itself after its trailing
    // silence; audio before speech waits in ring for the prefix padding.
    std::optional<VoiceActivity> vad;
    std::string ring;
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
    // Text arrives while audio is pushed when the session transcribes live;
    // otherwise the audio waits for the commit.
    const auto push = [&](const std::string& bytes) -> bool {
      if (bytes.empty())
        return true;
      if (!live && audio_bytes.empty() && service.supports_live() &&
          prompt.empty() && English(language)) {
        std::string failure;
        live = service.CreateLiveTranscription(24000, &failure);
        if (!live)
          throw std::runtime_error(failure);
      }
      if (live) {
        const auto item =
            "item_" + id + "_" + std::to_string(item_sequence + 1);
        live_bytes += bytes.size();
        std::string failure;
        bool sent = true;
        if (!live->Push(
                Pcm16(bytes),
                [&](std::string_view text) {
                  auto delta = json::Value::object();
                  delta["type"] =
                      "conversation.item.input_audio_transcription.delta";
                  delta["item_id"] = item;
                  delta["content_index"] = 0;
                  delta["delta"] = std::string(text);
                  return sent = send(std::move(delta));
                },
                &failure)) {
          live.reset();
          live_bytes = 0;
          if (!sent)
            return false;
          throw std::runtime_error("live transcription failed: " + failure);
        }
      } else
        audio_bytes.append(bytes);
      return true;
    };
    // Closes the buffered utterance as the next item and transcribes it.
    const auto commit = [&]() -> bool {
      const auto item = "item_" + id + "_" + std::to_string(++item_sequence);
      auto committed = json::Value::object();
      committed["type"] = "input_audio_buffer.committed";
      committed["item_id"] = item;
      committed["previous_item_id"] =
          previous_item.empty() ? json::Value{} : json::Value(previous_item);
      if (!send(std::move(committed)))
        return false;
      auto created = json::Value::object();
      created["type"] = "conversation.item.created";
      created["previous_item_id"] =
          previous_item.empty() ? json::Value{} : json::Value(previous_item);
      created["item"] = json::parse(
          R"({"type":"message","role":"user","status":"completed","content":[{"type":"input_audio","transcript":null}]})");
      created["item"]["id"] = item;
      if (!send(std::move(created)))
        return false;
      previous_item = item;
      if (live) {
        auto stream = std::move(live);
        const auto samples = live_bytes / 2;
        live_bytes = 0;
        std::string failure;
        bool sent = true;
        const bool ok = stream->Finish(
            [&](std::string_view text) {
              auto delta = json::Value::object();
              delta["type"] =
                  "conversation.item.input_audio_transcription.delta";
              delta["item_id"] = item;
              delta["content_index"] = 0;
              delta["delta"] = std::string(text);
              return sent = send(std::move(delta));
            },
            &failure);
        if (!sent || socket.cancelled())
          return false;
        auto completed = json::Value::object();
        completed["type"] =
            ok ? "conversation.item.input_audio_transcription.completed"
               : "conversation.item.input_audio_transcription.failed";
        completed["item_id"] = item;
        completed["content_index"] = 0;
        if (ok)
          completed["transcript"] = stream->text();
        else {
          completed["error"] = json::Value::object();
          completed["error"]["type"] = "server_error";
          completed["error"]["code"] = "transcription_failed";
          completed["error"]["message"] = failure;
        }
        if (!send(std::move(completed)))
          return false;
        Logger::Info("asr",
                     "request=" + id +
                         " event=transcription_complete item=" + item +
                         " live=true audio_samples=" + std::to_string(samples) +
                         " outcome=" + (ok ? "completed" : "failed"));
        return true;
      }
      models::qwen3_asr::AudioBuffer audio{.sample_rate = 24000, .channels = 1};
      audio.samples = Pcm16(audio_bytes);
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
          transcription, [&] { return socket.cancelled(); }, &result, &failure);
      if (socket.cancelled())
        return false;
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
        return false;
      Logger::Info("asr", "request=" + id +
                              " event=transcription_complete item=" + item +
                              " chunks=" + std::to_string(result.chunks) +
                              " audio_samples=" + std::to_string(pcm.size()) +
                              " outcome=" + (ok ? "completed" : "failed"));
      return true;
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
          if (!audio_bytes.empty() || live)
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
          std::optional<VadSettings> next_vad;
          bool vad_changed = false;
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
              if (const auto* option = input->find("noise_reduction");
                  option && !option->is_null())
                throw std::invalid_argument("noise_reduction must be null");
              if (const auto* detection = input->find("turn_detection")) {
                next_vad = ParseTurnDetection(*detection);
                vad_changed = true;
                next["audio"]["input"]["turn_detection"] =
                    TurnDetectionJson(next_vad);
              }
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
          if (vad_changed) {
            vad.reset();
            ring.clear();
            if (next_vad)
              vad.emplace(*next_vad, 24000);
          }
          if (!session_event("session.updated"))
            return;
        } else if (type == "input_audio_buffer.append") {
          Members(message, {"type", "event_id", "audio"});
          const auto* audio = message.find("audio");
          std::string bytes;
          if (!audio || !audio->is_string() ||
              !DecodeAudioBase64(
                  audio->str(), maximum_bytes - audio_bytes.size() - live_bytes,
                  &bytes) ||
              bytes.size() % 2 != 0)
            throw std::invalid_argument(
                "invalid PCM16 audio or 120-second input buffer exceeded");
          if (!vad) {
            if (!push(bytes))
              return;
            continue;
          }
          const auto events = vad->Push(Pcm16(bytes));
          std::size_t cursor = 0;
          for (const auto& event : events) {
            const auto at = std::min(bytes.size(), event.offset * 2);
            auto out = json::Value::object();
            out["item_id"] =
                "item_" + id + "_" + std::to_string(item_sequence + 1);
            if (event.start) {
              out["type"] = "input_audio_buffer.speech_started";
              out["audio_start_ms"] = std::int64_t(event.sample * 1000 / 24000);
              if (!send(std::move(out)))
                return;
              ring.append(bytes, cursor, at - cursor);
              if (!push(ring))
                return;
              ring.clear();
            } else {
              if (!push(bytes.substr(cursor, at - cursor)))
                return;
              out["type"] = "input_audio_buffer.speech_stopped";
              out["audio_end_ms"] = std::int64_t(event.sample * 1000 / 24000);
              if (!send(std::move(out)))
                return;
              if (event.voiced_ms < VoiceActivity::kMinSpeechMs) {
                // Too little speech is noise: drop the utterance.
                audio_bytes.clear();
                live.reset();
                live_bytes = 0;
                auto cleared = json::Value::object();
                cleared["type"] = "input_audio_buffer.cleared";
                if (!send(std::move(cleared)))
                  return;
              } else if (audio_bytes.size() + live_bytes >= 2400 * 2 &&
                         !commit())
                return;
            }
            cursor = at;
          }
          if (vad->speaking()) {
            if (!push(bytes.substr(cursor)))
              return;
          } else {
            // Prefix padding plus the 60 ms that confirm a start.
            ring.append(bytes, cursor);
            const auto keep =
                std::size_t(vad->settings().prefix_padding_ms + 60) * 24000 /
                1000 * 2;
            if (ring.size() > keep)
              ring.erase(0, ring.size() - keep);
          }
        } else if (type == "input_audio_buffer.clear") {
          Members(message, {"type", "event_id"});
          audio_bytes.clear();
          live.reset();
          live_bytes = 0;
          ring.clear();
          if (vad)
            vad.emplace(vad->settings(), 24000);
          auto event = json::Value::object();
          event["type"] = "input_audio_buffer.cleared";
          if (!send(std::move(event)))
            return;
        } else if (type == "input_audio_buffer.commit") {
          Members(message, {"type", "event_id"});
          if (audio_bytes.size() + live_bytes < 2400 * 2)
            throw std::invalid_argument(
                "commit requires at least 100 ms of audio");
          if (!commit())
            return;
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
