// Spoken conversations over the OpenAI Realtime WebSocket protocol: user
// audio is prefilled into the model while it arrives, replies stream text
// and speech, and function tools are called between turns. The protocol state
// machine lives in Rust (gufo-realtime-voice); this file owns the WebSocket
// threads, audio buffers, voice activity detection and VoiceConversation calls.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "gufo/realtime_voice.h"
#include "src/cli/serve/asr_service.hpp"
#include "src/cli/serve/audio_buffer.hpp"
#include "src/cli/serve/audio_stream.hpp"
#include "src/cli/serve/audio_websocket.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/voice_activity.hpp"
#include "src/cli/serve/websocket.hpp"

namespace gufo::server {
namespace {
constexpr std::uint32_t kRate = 24000;
// Queued client events while a reply runs: one minute of 24 kHz PCM16 in
// base64, plus framing.
constexpr std::size_t kQueueBytes = 8U << 20;
// Transcript tokens of a user turn (at most two minutes of speech); models
// that transcribe in windows may apply it to each window.
constexpr std::size_t kTranscriptTokens = 1024;

std::string_view View(const std::uint8_t* data, std::size_t len) noexcept {
  return len == 0 ? std::string_view{}
                  : std::string_view(reinterpret_cast<const char*>(data), len);
}

std::string_view View(GufoRealtimeVoiceBytes bytes) noexcept {
  return View(bytes.data, bytes.len);
}

GufoRealtimeVoiceBytes Bytes(std::string_view text) noexcept {
  return {text.empty() ? nullptr
                       : reinterpret_cast<const std::uint8_t*>(text.data()),
          text.size()};
}

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

VoiceConversation::Settings FromCSettings(
    const GufoRealtimeVoiceSettings& raw) {
  VoiceConversation::Settings settings;
  settings.instructions = std::string(View(raw.instructions));
  settings.audio_output = raw.audio_output;
  settings.tools.reserve(raw.tools_len);
  for (std::size_t i = 0; i < raw.tools_len; ++i)
    settings.tools.emplace_back(View(raw.tools[i]));
  settings.max_output_tokens = raw.max_output_tokens;
  settings.reasoning = raw.reasoning;
  settings.voice = std::string(View(raw.voice));
  settings.response_instructions = std::string(View(raw.response_instructions));
  settings.call_tools = raw.call_tools;
  settings.out_of_band = raw.out_of_band;
  return settings;
}

bool SendOnSocket(void* ctx, const std::uint8_t* data,
                  std::size_t len) noexcept {
  auto* socket = static_cast<WebSocket*>(ctx);
  try {
    return socket->SendText(View(data, len));
  } catch (...) {
    return false;
  }
}

void CopyBytes(void* ctx, const std::uint8_t* data, std::size_t len) noexcept {
  auto* out = static_cast<std::string*>(ctx);
  try {
    out->assign(View(data, len));
  } catch (...) {
    out->clear();
  }
}

void SetError(GufoRealtimeVoiceErrorSetFn set_error, void* error_ctx,
              std::string_view message) noexcept {
  const auto b = Bytes(message);
  set_error(error_ctx, b.data, b.len);
}

class Session {
public:
  Session(WebSocket& socket, AsrService& service, std::string id,
          std::unique_ptr<VoiceConversation> conversation)
      : socket_(socket),
        service_(service),
        id_(std::move(id)),
        conversation_(std::move(conversation)),
        protocol_(nullptr, &gufo_realtime_voice_free) {
    const auto id_b = Bytes(id_);
    const auto model = service.model_id();
    const auto model_b = Bytes(model);
    protocol_.reset(gufo_realtime_voice_new(id_b.data, id_b.len, model_b.data,
                                            model_b.len));
  }

  void Run() {
    if (!protocol_)
      return;
    std::thread worker([this] { Work(); });
    if (gufo_realtime_voice_session_created(protocol_.get(), &SendOnSocket,
                                            &socket_) ==
        GUFO_REALTIME_VOICE_OK) {
      Read();
    }
    {
      const std::lock_guard lock(mutex_);
      closing_ = true;
    }
    stopping_ = true;
    cancel_ = true;
    ready_.notify_all();
    worker.join();
  }

private:
  struct QueueItem {
    enum class Kind { kClient, kVadAudio, kVadClear, kVadCommit };
    Kind kind;
    std::string payload;
    bool respond{false};
    std::size_t bytes{0};
  };

  bool SendError(std::string_view message, std::string_view client_event = {},
                 std::string_view code = "invalid_request") {
    const auto msg_b = Bytes(message);
    const auto ev_b = Bytes(client_event);
    const auto code_b = Bytes(code);
    return gufo_realtime_voice_error(protocol_.get(), msg_b.data, msg_b.len,
                                     ev_b.data, ev_b.len, code_b.data,
                                     code_b.len, &SendOnSocket,
                                     &socket_) == GUFO_REALTIME_VOICE_OK;
  }

  // Reader: everything but response.cancel goes to the worker in order.
  void Read() {
    std::string raw;
    while (socket_.Receive(&raw)) {
      std::optional<VadSettings> vad_settings;
      {
        const std::lock_guard lock(vad_mutex_);
        vad_settings = vad_settings_;
        if (vad_generation_ != reader_generation_) {
          reader_generation_ = vad_generation_;
          vad_.reset();
          interrupting_ = false;
        }
      }
      if (!vad_settings) {
        vad_.reset();
        interrupting_ = false;
      }
      GufoRealtimeVoiceDecision decision = GUFO_REALTIME_VOICE_HANDLED;
      std::string pcm;
      const auto raw_b = Bytes(raw);
      const auto status = gufo_realtime_voice_classify_client_event(
          protocol_.get(), raw_b.data, raw_b.len, vad_settings.has_value(),
          responding_.load(), &decision, &CopyBytes, &pcm, &SendOnSocket,
          &socket_);
      if (status == GUFO_REALTIME_VOICE_CLOSED)
        return;
      if (status != GUFO_REALTIME_VOICE_OK) {
        if (!SendError("internal protocol error"))
          return;
        continue;
      }
      if (decision == GUFO_REALTIME_VOICE_HANDLED)
        continue;
      if (decision == GUFO_REALTIME_VOICE_CANCEL) {
        cancel_ = true;
        continue;
      }
      if (decision == GUFO_REALTIME_VOICE_VAD_AUDIO) {
        if (!Detect(*vad_settings, pcm))
          return;
        continue;
      }
      if (!Enqueue(
              {QueueItem::Kind::kClient, std::move(raw), false, raw_b.len})) {
        if (!SendError("too many events queued during a response"))
          return;
      }
    }
  }

  bool Enqueue(QueueItem item) {
    std::unique_lock lock(mutex_);
    if (queued_bytes_ + item.bytes > kQueueBytes)
      return false;
    queued_bytes_ += item.bytes;
    queue_.push_back(std::move(item));
    lock.unlock();
    ready_.notify_one();
    return true;
  }

  bool EnqueueAudio(std::string_view pcm) {
    if (pcm.empty())
      return true;
    return Enqueue({QueueItem::Kind::kVadAudio, std::string(pcm), false,
                    pcm.size() * 4 / 3});
  }

  // Server turn detection on the reader thread, so an interruption cancels
  // the reply at once.
  bool Detect(const VadSettings& settings, std::string_view bytes) {
    if (!vad_) {
      vad_.emplace(settings, kRate);
      ring_.clear();
    }
    // The client may still play the reply: its echo must not start a turn.
    vad_->Guard(Now() < playing_until_.load() + 300'000);
    const auto events = vad_->Push(Pcm16(bytes));
    std::size_t cursor = 0;
    const auto ms = [](std::uint64_t sample) {
      return std::int64_t(sample * 1000 / kRate);
    };
    for (const auto& event : events) {
      const auto at = std::min(bytes.size(), event.offset * 2);
      if (event.start) {
        // The client hears of the speech before the reply it interrupts ends.
        if (gufo_realtime_voice_vad_speech_started(
                protocol_.get(), ms(event.sample), &CopyBytes, &vad_item_,
                &SendOnSocket, &socket_) != GUFO_REALTIME_VOICE_OK) {
          return false;
        }
        interrupting_ = settings.interrupt_response;
        ring_.append(bytes, cursor, at - cursor);
        const bool queued = EnqueueAudio(ring_);
        ring_.clear();
        if (!queued)
          return SendError("too many events queued during a response");
      } else {
        interrupting_ = false;
        if (!EnqueueAudio(bytes.substr(cursor, at - cursor)))
          return SendError("too many events queued during a response");
        const auto id_b = Bytes(vad_item_);
        if (gufo_realtime_voice_vad_speech_stopped(
                protocol_.get(), id_b.data, id_b.len, ms(event.sample),
                &SendOnSocket, &socket_) != GUFO_REALTIME_VOICE_OK) {
          return false;
        }
        // Too little speech is noise (a click, a breath): no turn.
        const auto kind = event.voiced_ms < VoiceActivity::kMinSpeechMs
                              ? QueueItem::Kind::kVadClear
                              : QueueItem::Kind::kVadCommit;
        if (!Enqueue({kind, vad_item_, settings.create_response, 64}))
          return SendError("too many events queued during a response");
      }
      cursor = at;
    }
    const auto rest = bytes.substr(cursor);
    if (vad_->speaking()) {
      if (!EnqueueAudio(rest))
        return SendError("too many events queued during a response");
    } else {
      // Prefix padding plus the 60 ms that confirm a start.
      ring_.append(rest);
      const auto keep =
          std::size_t(settings.prefix_padding_ms + 60) * kRate / 1000 * 2;
      if (ring_.size() > keep)
        ring_.erase(0, ring_.size() - keep);
    }
    return true;
  }

  GufoRealtimeVoiceHost MakeHost() noexcept {
    return GufoRealtimeVoiceHost{
        .context = this,
        .send_event = &HostSendEvent,
        .configure = &HostConfigure,
        .on_vad_settings = &HostOnVadSettings,
        .append_audio = &HostAppendAudio,
        .commit_audio = &HostCommitAudio,
        .clear_audio = &HostClearAudio,
        .truncate_reply = &HostTruncateReply,
        .add_user_text = &HostAddUserText,
        .add_tool_outputs = &HostAddToolOutputs,
        .serves_model = &HostServesModel,
        .transcribe_turn = &HostTranscribeTurn,
        .delete_user_turn = &HostDeleteUserTurn,
        .supports_response_options = &HostSupportsResponseOptions,
    };
  }

  static bool HostSendEvent(void* ctx, const std::uint8_t* data,
                            std::size_t len) noexcept {
    auto* self = static_cast<Session*>(ctx);
    return SendOnSocket(&self->socket_, data, len);
  }

  static bool HostConfigure(void* ctx,
                            const GufoRealtimeVoiceSettings* settings,
                            GufoRealtimeVoiceErrorSetFn set_error,
                            void* error_ctx) noexcept {
    auto* self = static_cast<Session*>(ctx);
    try {
      auto next = FromCSettings(*settings);
      if (!next.voice.empty()) {
        const auto voices = self->conversation_->voices();
        if (std::ranges::find(voices, next.voice) == voices.end()) {
          std::string message = "audio.output.voice " + next.voice + " is ";
          if (voices.empty()) {
            message += "not available: this model speaks its own voice";
          } else {
            message += "not one of";
            for (const auto& voice : voices)
              message += " " + voice;
          }
          SetError(set_error, error_ctx, message);
          return false;
        }
      }
      std::string failure;
      if (self->conversation_->Configure(next, &failure))
        return true;
      SetError(set_error, error_ctx, failure);
      return false;
    } catch (const std::exception& e) {
      SetError(set_error, error_ctx, e.what());
      return false;
    } catch (...) {
      SetError(set_error, error_ctx, "internal error");
      return false;
    }
  }

  static void HostOnVadSettings(
      void* ctx, bool has_vad,
      const GufoRealtimeVoiceVadSettings* vad) noexcept {
    auto* self = static_cast<Session*>(ctx);
    const std::lock_guard lock(self->vad_mutex_);
    if (has_vad && vad) {
      self->vad_settings_ = VadSettings{
          .threshold = vad->threshold,
          .prefix_padding_ms = vad->prefix_padding_ms,
          .silence_duration_ms = vad->silence_duration_ms,
          .create_response = vad->create_response,
          .interrupt_response = vad->interrupt_response,
      };
    } else {
      self->vad_settings_.reset();
    }
    ++self->vad_generation_;
  }

  static bool HostAppendAudio(void* ctx, const std::uint8_t* pcm,
                              std::size_t len, bool record,
                              GufoRealtimeVoiceErrorSetFn set_error,
                              void* error_ctx) noexcept {
    auto* self = static_cast<Session*>(ctx);
    try {
      std::string failure;
      if (self->conversation_->AppendAudio(Pcm16(View(pcm, len)), &failure)) {
        if (record)
          self->turn_pcm_.append(View(pcm, len));
        return true;
      }
      SetError(set_error, error_ctx, failure);
      return false;
    } catch (const std::exception& e) {
      SetError(set_error, error_ctx, e.what());
      return false;
    } catch (...) {
      SetError(set_error, error_ctx, "internal error");
      return false;
    }
  }

  static bool HostCommitAudio(void* ctx, bool* committed,
                              GufoRealtimeVoiceErrorSetFn set_error,
                              void* error_ctx) noexcept {
    auto* self = static_cast<Session*>(ctx);
    try {
      std::size_t samples = 0;
      std::string failure;
      if (self->conversation_->CommitAudio(&samples, &failure)) {
        // The committed turn's recorded audio stays for its transcript.
        self->committed_pcm_ = std::move(self->turn_pcm_);
        self->turn_pcm_.clear();
        *committed = true;
        return true;
      }
      // A failed commit keeps the turn open with its audio.
      if (failure.empty())
        self->turn_pcm_.clear();
      if (!failure.empty()) {
        SetError(set_error, error_ctx, failure);
        return false;
      }
      *committed = false;
      return true;
    } catch (const std::exception& e) {
      SetError(set_error, error_ctx, e.what());
      return false;
    } catch (...) {
      SetError(set_error, error_ctx, "internal error");
      return false;
    }
  }

  static bool HostClearAudio(void* ctx, GufoRealtimeVoiceErrorSetFn set_error,
                             void* error_ctx) noexcept {
    auto* self = static_cast<Session*>(ctx);
    try {
      std::string failure;
      self->turn_pcm_.clear();
      if (self->conversation_->ClearAudio(&failure))
        return true;
      SetError(set_error, error_ctx, failure);
      return false;
    } catch (const std::exception& e) {
      SetError(set_error, error_ctx, e.what());
      return false;
    } catch (...) {
      SetError(set_error, error_ctx, "internal error");
      return false;
    }
  }

  static bool HostTruncateReply(void* ctx, std::size_t samples,
                                GufoRealtimeVoiceErrorSetFn set_error,
                                void* error_ctx) noexcept {
    auto* self = static_cast<Session*>(ctx);
    try {
      std::string failure;
      if (self->conversation_->TruncateReply(samples, &failure)) {
        self->playing_until_ = Now();  // the client stopped playing
        return true;
      }
      SetError(set_error, error_ctx, failure);
      return false;
    } catch (const std::exception& e) {
      SetError(set_error, error_ctx, e.what());
      return false;
    } catch (...) {
      SetError(set_error, error_ctx, "internal error");
      return false;
    }
  }

  static bool HostAddUserText(void* ctx, const std::uint8_t* text,
                              std::size_t len,
                              GufoRealtimeVoiceErrorSetFn set_error,
                              void* error_ctx) noexcept {
    auto* self = static_cast<Session*>(ctx);
    try {
      std::string failure;
      if (self->conversation_->AddUserText(View(text, len), &failure))
        return true;
      SetError(set_error, error_ctx, failure);
      return false;
    } catch (const std::exception& e) {
      SetError(set_error, error_ctx, e.what());
      return false;
    } catch (...) {
      SetError(set_error, error_ctx, "internal error");
      return false;
    }
  }

  static bool HostAddToolOutputs(void* ctx,
                                 const GufoRealtimeVoiceBytes* outputs,
                                 std::size_t count,
                                 GufoRealtimeVoiceErrorSetFn set_error,
                                 void* error_ctx) noexcept {
    auto* self = static_cast<Session*>(ctx);
    try {
      std::vector<std::string> items;
      items.reserve(count);
      for (std::size_t i = 0; i < count; ++i)
        items.emplace_back(View(outputs[i]));
      std::string failure;
      if (self->conversation_->AddToolOutputs(items, &failure))
        return true;
      SetError(set_error, error_ctx, failure);
      return false;
    } catch (const std::exception& e) {
      SetError(set_error, error_ctx, e.what());
      return false;
    } catch (...) {
      SetError(set_error, error_ctx, "internal error");
      return false;
    }
  }

  static bool HostServesModel(void* ctx, const std::uint8_t* name,
                              std::size_t len) noexcept {
    auto* self = static_cast<Session*>(ctx);
    try {
      return self->service_.serves_model(View(name, len));
    } catch (...) {
      return false;
    }
  }

  // The served model transcribes the committed turn, as a transcription
  // session would.
  static bool HostTranscribeTurn(void* ctx, GufoRealtimeVoiceBytes language,
                                 GufoRealtimeVoiceBytes prompt,
                                 GufoRealtimeVoiceBytesFn on_transcript,
                                 void* transcript_ctx,
                                 GufoRealtimeVoiceErrorSetFn set_error,
                                 void* error_ctx) noexcept {
    auto* self = static_cast<Session*>(ctx);
    try {
      AudioBuffer audio{.sample_rate = kRate, .channels = 1, .samples = {}};
      audio.samples = Pcm16(self->committed_pcm_);
      self->committed_pcm_.clear();
      const auto pcm = ResampleMono16k(audio);
      const auto lang = View(language);
      const ext::TranscriptionRequest request{
          .wav = {},
          .context = std::string(View(prompt)),
          .language =
              lang.empty() ? std::nullopt : std::optional<std::string>(lang),
          .max_new_tokens = kTranscriptTokens,
          .pcm16k = pcm,
          .on_text = {}};
      ext::TranscriptionResult result;
      std::string failure;
      const bool ok = self->service_.Transcribe(
          request,
          [self] {
            return self->stopping_.load() || self->socket_.cancelled();
          },
          &result, &failure);
      Logger::Info("asr", "request=" + self->id_ +
                              " event=turn_transcribed audio_samples=" +
                              std::to_string(pcm.size()) +
                              " outcome=" + (ok ? "completed" : "failed"));
      if (!ok) {
        SetError(set_error, error_ctx,
                 failure.empty() ? "transcription failed" : failure);
        return false;
      }
      const auto text = Bytes(result.text);
      on_transcript(transcript_ctx, text.data, text.len);
      return true;
    } catch (const std::exception& e) {
      SetError(set_error, error_ctx, e.what());
      return false;
    } catch (...) {
      SetError(set_error, error_ctx, "internal error");
      return false;
    }
  }

  static bool HostSupportsResponseOptions(void* ctx) noexcept {
    return static_cast<Session*>(ctx)
        ->conversation_->supports_response_options();
  }

  static bool HostDeleteUserTurn(void* ctx,
                                 GufoRealtimeVoiceErrorSetFn set_error,
                                 void* error_ctx) noexcept {
    auto* self = static_cast<Session*>(ctx);
    try {
      std::string failure;
      if (self->conversation_->DeleteUserTurn(&failure))
        return true;
      SetError(set_error, error_ctx, failure);
      return false;
    } catch (const std::exception& e) {
      SetError(set_error, error_ctx, e.what());
      return false;
    } catch (...) {
      SetError(set_error, error_ctx, "internal error");
      return false;
    }
  }

  void Work() {
    const auto host = MakeHost();
    while (true) {
      QueueItem item;
      {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, [&] { return closing_ || !queue_.empty(); });
        if (closing_)
          return;
        item = std::move(queue_.front());
        queued_bytes_ -= item.bytes;
        queue_.pop_front();
      }
      bool start_response = false;
      GufoRealtimeVoiceStatus status = GUFO_REALTIME_VOICE_OK;
      const auto payload_b = Bytes(item.payload);
      switch (item.kind) {
        case QueueItem::Kind::kClient:
          status = gufo_realtime_voice_handle_client_event(
              protocol_.get(), payload_b.data, payload_b.len,
              conversation_->audio_open(), &host, &start_response);
          break;
        case QueueItem::Kind::kVadAudio:
          status = gufo_realtime_voice_handle_vad_audio(
              protocol_.get(), payload_b.data, payload_b.len,
              conversation_->audio_open(), &host);
          break;
        case QueueItem::Kind::kVadClear:
          status = gufo_realtime_voice_handle_vad_clear(protocol_.get(), &host);
          break;
        case QueueItem::Kind::kVadCommit:
          status = gufo_realtime_voice_handle_vad_commit(
              protocol_.get(), payload_b.data, payload_b.len, item.respond,
              interrupting_.load(), &host, &start_response);
          break;
      }
      if (status != GUFO_REALTIME_VOICE_OK) {
        if (status == GUFO_REALTIME_VOICE_INTERNAL)
          socket_.Close(1011);
        return;
      }
      if (start_response && !RunResponse())
        return;
    }
  }

  bool RunResponse() {
    if (stopping_.load())
      return false;
    struct ActiveInfo {
      VoiceConversation::Settings settings;
      std::string response_id;
      bool ok{false};
    };
    ActiveInfo info;
    const auto capture =
        [](void* ctx, const GufoRealtimeVoiceSettings* settings,
           const std::uint8_t* id, std::size_t id_len) noexcept {
          auto* dst = static_cast<ActiveInfo*>(ctx);
          try {
            dst->settings = FromCSettings(*settings);
            dst->response_id = std::string(View(id, id_len));
            dst->ok = true;
          } catch (...) {
            dst->ok = false;
          }
        };
    // Active before it is announced: a client may cancel on response.created.
    cancel_ = false;
    responding_ = true;
    if (gufo_realtime_voice_response_start(protocol_.get(), capture, &info,
                                           &SendOnSocket, &socket_) !=
            GUFO_REALTIME_VOICE_OK ||
        !info.ok) {
      responding_ = false;
      return false;
    }

    bool sent = true;
    VoiceConversation::Callbacks callbacks;
    callbacks.cancelled = [&] {
      return stopping_.load() || cancel_.load() || !sent ||
             interrupting_.load();
    };
    callbacks.on_reasoning = [&](std::string_view text) {
      const auto b = Bytes(text);
      return sent = gufo_realtime_voice_response_reasoning(
                        protocol_.get(), b.data, b.len, &SendOnSocket,
                        &socket_) == GUFO_REALTIME_VOICE_OK;
    };
    callbacks.on_text = [&](std::string_view text) {
      const auto b = Bytes(text);
      return sent = gufo_realtime_voice_response_text(
                        protocol_.get(), b.data, b.len, &SendOnSocket,
                        &socket_) == GUFO_REALTIME_VOICE_OK;
    };
    callbacks.on_audio = [&](std::span<const float> samples) {
      // Playback on the client ends no sooner than this audio's duration
      // after it arrives (or after the audio before it).
      const auto now = Now();
      playing_until_ = std::max(playing_until_.load(), now) +
                       std::int64_t(samples.size()) * 1'000'000 / kRate;
      const auto pcm = EncodePcm16(samples);
      const auto b = Bytes(pcm);
      return sent = gufo_realtime_voice_response_audio(
                        protocol_.get(), b.data, b.len, &SendOnSocket,
                        &socket_) == GUFO_REALTIME_VOICE_OK;
    };

    VoiceConversation::Reply reply;
    std::string failure;
    const bool ok =
        conversation_->Respond(info.settings, callbacks, &reply, &failure);
    responding_ = false;
    cancel_ = false;
    if (!sent || socket_.cancelled())
      return false;

    const char* status_str = !ok                ? "failed"
                             : reply.cancelled  ? "cancelled"
                             : reply.incomplete ? "incomplete"
                                                : "completed";
    Logger::Info("asr",
                 "request=" + id_ + " event=response_done response=" +
                     info.response_id + " status=" + status_str +
                     " audio_samples=" + std::to_string(reply.audio_samples) +
                     " calls=" + std::to_string(reply.calls.size()) +
                     " first_text_ms=" + std::to_string(reply.first_text_ms) +
                     " first_audio_ms=" + std::to_string(reply.first_audio_ms) +
                     " total_ms=" + std::to_string(reply.total_ms));

    std::vector<GufoRealtimeVoiceFunctionCall> c_calls;
    c_calls.reserve(reply.calls.size());
    for (const auto& call : reply.calls)
      c_calls.push_back({Bytes(call.name), Bytes(call.arguments)});
    const GufoRealtimeVoiceResponseOutcome outcome{
        .ok = ok,
        .cancelled = reply.cancelled,
        .incomplete = reply.incomplete,
        .input_tokens = static_cast<std::int64_t>(reply.input_tokens),
        .output_tokens = static_cast<std::int64_t>(reply.output_tokens),
        .calls = c_calls.empty() ? nullptr : c_calls.data(),
        .calls_len = c_calls.size(),
        .failure = Bytes(failure),
    };
    return gufo_realtime_voice_response_finish(protocol_.get(), &outcome,
                                               &SendOnSocket, &socket_) ==
           GUFO_REALTIME_VOICE_OK;
  }

  WebSocket& socket_;
  AsrService& service_;
  std::string id_;
  std::unique_ptr<VoiceConversation> conversation_;
  std::unique_ptr<GufoRealtimeVoice, void (*)(GufoRealtimeVoice*)> protocol_;
  std::atomic<bool> cancel_{false}, responding_{false}, stopping_{false};
  // The server hears the user speak (turn detection) and is configured to
  // interrupt a reply.
  std::atomic<bool> interrupting_{false};
  // Estimated end of the reply's playback on the client (Now() clock).
  std::atomic<std::int64_t> playing_until_{0};
  static std::int64_t Now() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }
  // Worker state: PCM16 of the open user turn and of the turn committed
  // last, which its transcript reads.
  std::string turn_pcm_, committed_pcm_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<QueueItem> queue_;
  std::size_t queued_bytes_{0};
  bool closing_{false};
  // Turn detection: settings from session.update (worker) for the reader.
  std::mutex vad_mutex_;
  std::optional<VadSettings> vad_settings_;
  std::size_t vad_generation_{0};
  // Reader state.
  std::size_t reader_generation_{0};
  std::optional<VoiceActivity> vad_;
  std::string ring_, vad_item_;
};
}  // namespace

HttpResponse HandleVoiceWebSocket(const HttpRequest& request,
                                  AsrService& service) {
  return UpgradeWebSocket(
      request, [&service, id = request.request_id](WebSocket& socket) {
        std::string failure;
        auto conversation = service.CreateVoiceConversation(kRate, &failure);
        if (!conversation) {
          const auto b = Bytes(failure);
          (void)gufo_realtime_voice_session_unavailable(b.data, b.len,
                                                        &SendOnSocket, &socket);
          socket.Close(1013);
          return;
        }
        Logger::Info("asr", "request=" + id + " event=conversation_open");
        Session(socket, service, id, std::move(conversation)).Run();
        Logger::Info("asr", "request=" + id + " event=conversation_closed");
      });
}

}  // namespace gufo::server
