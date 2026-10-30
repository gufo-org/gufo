// Spoken conversations over the OpenAI Realtime WebSocket protocol: user
// audio is prefilled into the model while it arrives, replies stream text
// and speech, and function tools are called between turns. One worker owns
// the conversation and handles events in order; the reader keeps receiving,
// so response.cancel interrupts a reply within one generation step.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>

#include "src/cli/serve/asr_service.hpp"
#include "src/cli/serve/audio_stream.hpp"
#include "src/cli/serve/audio_websocket.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/voice_activity.hpp"
#include "src/cli/serve/websocket.hpp"
#include "src/core/json.hpp"

namespace gufo::server {
namespace {
constexpr std::uint32_t kRate = 24000;
// Queued client events while a reply runs: one minute of 24 kHz PCM16 in
// base64, plus framing.
constexpr std::size_t kQueueBytes = 8U << 20;
// One user turn holds at most two minutes of audio.
constexpr std::size_t kTurnSamples = 120U * kRate;

void Members(const json::Value& object,
             std::initializer_list<std::string_view> allowed,
             std::string_view what) {
  if (!object.is_object())
    throw std::invalid_argument(std::string(what) + " must be an object");
  for (const auto& [name, unused] : object.members()) {
    (void)unused;
    if (std::find(allowed.begin(), allowed.end(), name) == allowed.end())
      throw std::invalid_argument("unsupported " + std::string(what) +
                                  " field: " + name);
  }
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
std::vector<float> DecodeAudio(const json::Value* audio) {
  std::string bytes;
  if (!audio || !audio->is_string() ||
      !DecodeAudioBase64(audio->str(), kTurnSamples * 2, &bytes) ||
      bytes.size() % 2 != 0)
    throw std::invalid_argument("audio must be base64 PCM16 (mono, 24 kHz)");
  return Pcm16(bytes);
}
std::size_t OutputTokens(const json::Value& value) {
  if (value.is_string() && value.str() == "inf")
    return 4096;
  if (!value.is_number() || value.as_double() < 1 || value.as_double() > 4096 ||
      value.as_double() != std::floor(value.as_double()))
    throw std::invalid_argument(
        "max_output_tokens must be an integer in [1, 4096] or \"inf\"");
  return std::size_t(value.as_double());
}
// Validates modalities: ["audio"] (speech with its transcript) or ["text"].
bool AudioModality(const json::Value& value) {
  if (!value.is_array() || value.items().size() != 1 ||
      !value.items()[0].is_string() ||
      (value.items()[0].str() != "audio" && value.items()[0].str() != "text"))
    throw std::invalid_argument(
        "output_modalities must be [\"audio\"] or [\"text\"]");
  return value.items()[0].str() == "audio";
}
// A Realtime function tool ({type, name, description, parameters}) as the
// function object the model's template lists.
std::string Tool(const json::Value& tool) {
  Members(tool, {"type", "name", "description", "parameters"}, "tool");
  if (tool.member_str("type") != "function" || tool.member_str("name").empty())
    throw std::invalid_argument("tools must be named functions");
  auto function = json::Value::object();
  for (const auto& [key, value] : tool.members())
    if (key != "type")
      function[key] = value;
  return function.dump();
}

class Session {
public:
  Session(WebSocket& socket, AsrService& service, std::string id,
          std::unique_ptr<VoiceConversation> conversation)
      : socket_(socket),
        service_(service),
        id_(std::move(id)),
        conversation_(std::move(conversation)) {
    config_ = json::parse(
        R"({"type":"realtime","object":"realtime.session","output_modalities":["audio"],"instructions":"","tools":[],"tool_choice":"auto","max_output_tokens":"inf","audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"transcription":null,"turn_detection":null,"noise_reduction":null},"output":{"format":{"type":"audio/pcm","rate":24000}}}})");
    config_["id"] = "sess_" + id_;
    config_["model"] = service_.model_id();
  }
  void Run() {
    std::thread worker([this] { Work(); });
    if (Event("session.created", "session", config_))
      Read();
    {
      const std::lock_guard lock(mutex_);
      closing_ = true;
    }
    cancel_ = true;
    ready_.notify_all();
    worker.join();
  }

private:
  bool Send(json::Value event) {
    event["event_id"] = "event_" + id_ + "_" + std::to_string(++sequence_);
    return socket_.SendText(event.dump());
  }
  bool Event(std::string type, std::string_view key = {},
             json::Value value = {}) {
    auto event = json::Value::object();
    event["type"] = std::move(type);
    if (!key.empty())
      event[std::string(key)] = std::move(value);
    return Send(std::move(event));
  }
  bool Error(std::string message, std::string client_event = {},
             std::string code = "invalid_request") {
    auto error = json::Value::object();
    error["type"] = "invalid_request_error";
    error["code"] = std::move(code);
    error["message"] = std::move(message);
    if (!client_event.empty())
      error["event_id"] = std::move(client_event);
    return Event("error", "error", std::move(error));
  }
  // Reader: everything but response.cancel goes to the worker in order.
  void Read() {
    std::string raw;
    while (socket_.Receive(&raw)) {
      std::string client_event;
      try {
        auto message = json::parse(raw);
        if (!message.is_object())
          throw std::invalid_argument("event must be an object");
        client_event = message.member_str("event_id");
        const auto type = message.member_str("type");
        if (type.starts_with("_"))
          throw std::invalid_argument(
              "unsupported realtime conversation "
              "event: " +
              type);
        if (type == "input_audio_buffer.append" && Detect(message)) {
          continue;
        }
        if (type == "response.cancel") {
          Members(message, {"type", "event_id", "response_id"}, "event");
          if (!responding_) {
            Error("no response is in progress", client_event,
                  "response_cancel_not_active");
            continue;
          }
          cancel_ = true;
          continue;
        }
        Enqueue(std::move(message), raw.size());
      } catch (const std::exception& e) {
        if (!Error(e.what(), client_event))
          return;
      }
    }
  }
  void Enqueue(json::Value message, std::size_t bytes) {
    std::unique_lock lock(mutex_);
    if (queued_bytes_ + bytes > kQueueBytes)
      throw std::length_error("too many events queued during a response");
    queued_bytes_ += bytes;
    queue_.emplace_back(std::move(message), bytes);
    lock.unlock();
    ready_.notify_one();
  }
  void EnqueueAudio(std::string_view pcm) {
    if (pcm.empty())
      return;
    auto event = json::Value::object();
    event["type"] = "_vad.audio";
    event["audio"] = EncodeAudioBase64(pcm);
    Enqueue(std::move(event), pcm.size() * 4 / 3);
  }
  // Server turn detection on the reader thread, so an interruption cancels
  // the reply at once. Speech (with its prefix padding and trailing
  // silence) goes to the worker as internal audio events, and the end of a
  // turn as a commit. False when the session has no turn detection.
  bool Detect(const json::Value& message) {
    std::optional<VadSettings> settings;
    {
      const std::lock_guard lock(vad_mutex_);
      settings = vad_settings_;
      if (vad_generation_ != reader_generation_) {
        reader_generation_ = vad_generation_;
        vad_.reset();
      }
    }
    if (!settings) {
      vad_.reset();
      speaking_ = false;
      return false;
    }
    Members(message, {"type", "event_id", "audio"}, "event");
    const auto* audio = message.find("audio");
    std::string bytes;
    if (!audio || !audio->is_string() ||
        !DecodeAudioBase64(audio->str(), kTurnSamples * 2, &bytes) ||
        bytes.size() % 2 != 0)
      throw std::invalid_argument("audio must be base64 PCM16 (mono, 24 kHz)");
    if (!vad_) {
      vad_.emplace(*settings, kRate);
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
      auto out = json::Value::object();
      out["item_id"] = vad_item_;
      if (event.start) {
        vad_item_ = NextItem();
        out["item_id"] = vad_item_;
        out["type"] = "input_audio_buffer.speech_started";
        out["audio_start_ms"] = ms(event.sample);
        // The client hears of the speech before the reply it interrupts ends.
        if (!Send(std::move(out)))
          return true;
        speaking_ = true;
        interrupt_ = settings->interrupt_response;
        ring_.append(bytes, cursor, at - cursor);
        EnqueueAudio(ring_);
        ring_.clear();
      } else {
        EnqueueAudio(std::string_view(bytes).substr(cursor, at - cursor));
        speaking_ = false;
        out["type"] = "input_audio_buffer.speech_stopped";
        out["audio_end_ms"] = ms(event.sample);
        if (!Send(std::move(out)))
          return true;
        // Too little speech is noise (a click, a breath): no turn.
        auto commit = json::Value::object();
        commit["type"] = event.voiced_ms < VoiceActivity::kMinSpeechMs
                             ? "_vad.clear"
                             : "_vad.commit";
        commit["item_id"] = vad_item_;
        commit["respond"] = settings->create_response;
        Enqueue(std::move(commit), 64);
      }
      cursor = at;
    }
    const auto rest = std::string_view(bytes).substr(cursor);
    if (vad_->speaking())
      EnqueueAudio(rest);
    else {
      // Prefix padding plus the 60 ms that confirm a start.
      ring_.append(rest);
      const auto keep =
          std::size_t(settings->prefix_padding_ms + 60) * kRate / 1000 * 2;
      if (ring_.size() > keep)
        ring_.erase(0, ring_.size() - keep);
    }
    return true;
  }
  void Work() {
    while (true) {
      json::Value message;
      {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, [&] { return closing_ || !queue_.empty(); });
        if (closing_)
          return;
        message = std::move(queue_.front().first);
        queued_bytes_ -= queue_.front().second;
        queue_.pop_front();
      }
      const auto client_event = message.member_str("event_id");
      try {
        if (!Handle(message))
          return;
      } catch (const std::exception& e) {
        if (!Error(e.what(), client_event))
          return;
      }
    }
  }
  std::string NextItem() {
    return "item_" + id_ + "_" + std::to_string(++items_);
  }
  // Closes the user audio turn as item; false when nothing was committed.
  bool Commit(const std::string& item, bool* sent) {
    std::string failure;
    std::size_t samples = 0;
    *sent = true;
    if (!conversation_->CommitAudio(&samples, &failure)) {
      if (!failure.empty())
        throw std::runtime_error(failure);
      return false;
    }
    turn_samples_ = 0;
    auto committed = json::Value::object();
    committed["type"] = "input_audio_buffer.committed";
    committed["item_id"] = item;
    committed["previous_item_id"] = PreviousItem();
    if (!Send(std::move(committed)))
      return *sent = false;
    auto created = json::parse(
        R"({"type":"message","object":"realtime.item","role":"user","status":"completed","content":[{"type":"input_audio","transcript":null}]})");
    created["id"] = item;
    *sent = Created(std::move(created));
    return true;
  }
  json::Value PreviousItem() const {
    return previous_item_.empty() ? json::Value{} : json::Value(previous_item_);
  }
  bool Created(json::Value item) {
    auto event = json::Value::object();
    event["type"] = "conversation.item.created";
    event["previous_item_id"] = PreviousItem();
    previous_item_ = item.member_str("id");
    event["item"] = std::move(item);
    return Send(std::move(event));
  }
  // Pending function results enter the conversation as one tool turn.
  void FlushOutputs() {
    if (outputs_.empty())
      return;
    std::string failure;
    const auto outputs = std::move(outputs_);
    outputs_.clear();
    if (!conversation_->AddToolOutputs(outputs, &failure))
      throw std::runtime_error(failure);
  }
  bool Handle(const json::Value& message) {
    const auto type = message.member_str("type");
    std::string failure;
    if (type == "session.update") {
      Members(message, {"type", "event_id", "session"}, "event");
      const auto* session = message.find("session");
      if (!session)
        throw std::invalid_argument("session.update requires session");
      Members(*session,
              {"type", "instructions", "output_modalities", "tools",
               "tool_choice", "max_output_tokens", "audio", "reasoning"},
              "session");
      auto next = config_;
      auto settings = settings_;
      std::optional<VadSettings> vad;
      bool vad_changed = false;
      for (const auto& [name, value] : session->members()) {
        if (name == "type") {
          if (!value.is_string() || value.str() != "realtime")
            throw std::invalid_argument(
                "conversation sessions have type realtime");
        } else if (name == "instructions") {
          if (!value.is_string() || value.str().size() > 65536)
            throw std::invalid_argument(
                "instructions must be a string of at most 64 KiB");
          settings.instructions = value.str();
        } else if (name == "output_modalities") {
          settings.audio_output = AudioModality(value);
        } else if (name == "tools") {
          if (!value.is_array() || value.items().size() > 64)
            throw std::invalid_argument("tools must be an array of at most 64");
          settings.tools.clear();
          for (const auto& tool : value.items())
            settings.tools.push_back(Tool(tool));
        } else if (name == "tool_choice") {
          if (!value.is_string() || value.str() != "auto")
            throw std::invalid_argument("tool_choice must be auto");
        } else if (name == "max_output_tokens") {
          settings.max_output_tokens = OutputTokens(value);
        } else if (name == "reasoning") {
          // Gufo extension for text answers: {"effort": "none" | "low" |
          // "medium" | "high"}; the model reasons or not.
          Members(value, {"effort"}, "reasoning");
          const auto effort = value.member_str("effort");
          if (effort != "none" && effort != "low" && effort != "medium" &&
              effort != "high")
            throw std::invalid_argument(
                "reasoning.effort must be none, low, medium or high");
          settings.reasoning = effort != "none";
        } else if (name == "audio") {
          Members(value, {"input", "output"}, "audio");
          if (const auto* input = value.find("input")) {
            Members(*input,
                    {"format", "transcription", "turn_detection",
                     "noise_reduction"},
                    "audio.input");
            if (const auto* format = input->find("format"))
              Format(*format);
            for (const auto key : {"transcription", "noise_reduction"})
              if (const auto* option = input->find(key);
                  option && !option->is_null())
                throw std::invalid_argument(std::string("audio.input.") + key +
                                            " must be null");
            if (const auto* detection = input->find("turn_detection")) {
              vad = ParseTurnDetection(*detection);
              vad_changed = true;
            }
          }
          if (const auto* output = value.find("output")) {
            Members(*output, {"format"}, "audio.output");
            if (const auto* format = output->find("format"))
              Format(*format);
          }
          continue;
        }
        next[name] = value;
      }
      if (settings.reasoning && settings.audio_output)
        throw std::invalid_argument(
            "reasoning applies to text output_modalities");
      if (conversation_->audio_open())
        throw std::invalid_argument(
            "commit or clear the audio buffer before session.update");
      if (!conversation_->Configure(settings, &failure))
        throw std::runtime_error(failure);
      settings_ = std::move(settings);
      if (vad_changed) {
        next["audio"]["input"]["turn_detection"] = TurnDetectionJson(vad);
        const std::lock_guard lock(vad_mutex_);
        vad_settings_ = vad;
        ++vad_generation_;
      }
      config_ = std::move(next);
      return Event("session.updated", "session", config_);
    }
    if (type == "_vad.clear") {
      if (!conversation_->ClearAudio(&failure))
        throw std::runtime_error(failure);
      turn_samples_ = 0;
      return Event("input_audio_buffer.cleared");
    }
    if (type == "_vad.commit") {
      bool sent = true;
      if (!Commit(message.member_str("item_id"), &sent))
        return true;  // nothing was heard: no turn
      if (!sent)
        return false;
      // A user who already speaks again gets no reply to this turn.
      if (message.find("respond") && message.find("respond")->as_bool() &&
          !(speaking_ && interrupt_))
        return Respond(settings_);
      return true;
    }
    if (type == "conversation.item.truncate") {
      Members(message,
              {"type", "event_id", "item_id", "content_index", "audio_end_ms"},
              "event");
      const auto* end = message.find("audio_end_ms");
      if (message.member_str("item_id").empty() ||
          message.member_str("item_id") != last_reply_item_)
        throw std::invalid_argument(
            "only the latest assistant message can be truncated");
      if (!end || !end->is_number() || end->as_double() < 0)
        throw std::invalid_argument("audio_end_ms must be a number");
      if (!conversation_->TruncateReply(
              std::size_t(end->as_double() * kRate / 1000), &failure))
        throw std::runtime_error(failure);
      playing_until_ = Now();  // the client stopped playing
      auto event = json::Value::object();
      event["type"] = "conversation.item.truncated";
      event["item_id"] = last_reply_item_;
      event["content_index"] = 0;
      event["audio_end_ms"] = std::int64_t(end->as_double());
      return Send(std::move(event));
    }
    if (type == "input_audio_buffer.append" || type == "_vad.audio") {
      if (type == "input_audio_buffer.append")
        Members(message, {"type", "event_id", "audio"}, "event");
      const auto samples = DecodeAudio(message.find("audio"));
      if (turn_samples_ + samples.size() > kTurnSamples)
        throw std::length_error("a user turn holds at most 120 s of audio");
      if (!conversation_->audio_open())
        FlushOutputs();
      if (!conversation_->AppendAudio(samples, &failure))
        throw std::runtime_error(failure);
      turn_samples_ += samples.size();
      return true;
    }
    if (type == "input_audio_buffer.clear") {
      Members(message, {"type", "event_id"}, "event");
      if (!conversation_->ClearAudio(&failure))
        throw std::runtime_error(failure);
      turn_samples_ = 0;
      return Event("input_audio_buffer.cleared");
    }
    if (type == "input_audio_buffer.commit") {
      Members(message, {"type", "event_id"}, "event");
      if (turn_samples_ < kRate / 10)
        throw std::invalid_argument("commit requires at least 100 ms of audio");
      bool sent = true;
      if (!Commit(NextItem(), &sent))
        throw std::invalid_argument("the audio buffer is empty");
      return sent;
    }
    if (type == "conversation.item.create") {
      Members(message, {"type", "event_id", "previous_item_id", "item"},
              "event");
      const auto* item = message.find("item");
      if (!item)
        throw std::invalid_argument("conversation.item.create requires item");
      if (const auto* previous = message.find("previous_item_id");
          previous && !previous->is_null() &&
          (!previous->is_string() || previous->str() != previous_item_))
        throw std::invalid_argument("items can only be appended");
      if (conversation_->audio_open())
        throw std::invalid_argument(
            "commit or clear the audio buffer before adding an item");
      const auto kind = item->member_str("type");
      auto echo = *item;
      if (!item->find("id"))
        echo["id"] = NextItem();
      echo["object"] = "realtime.item";
      echo["status"] = "completed";
      if (kind == "function_call_output") {
        Members(*item, {"type", "id", "call_id", "output"}, "item");
        if (item->member_str("call_id").empty() ||
            !item->find("output")->is_string())
          throw std::invalid_argument(
              "function_call_output needs call_id and a string output");
        outputs_.push_back(item->member_str("output"));
        return Created(std::move(echo));
      }
      if (kind != "message" || item->member_str("role") != "user")
        throw std::invalid_argument(
            "items are user messages or function_call_output");
      Members(*item, {"type", "id", "role", "content"}, "item");
      const auto* content = item->find("content");
      if (!content || !content->is_array() || content->items().empty())
        throw std::invalid_argument("message content must be a list");
      FlushOutputs();
      // A message of text parts, or of one spoken part.
      std::string text;
      std::optional<std::vector<float>> spoken;
      for (const auto& part : content->items()) {
        Members(part, {"type", "text", "audio"}, "content part");
        if (part.member_str("type") == "input_text")
          text += part.member_str("text");
        else if (part.member_str("type") == "input_audio" && !spoken)
          spoken = DecodeAudio(part.find("audio"));
        else
          throw std::invalid_argument(
              "content parts are input_text, or one input_audio");
      }
      if (spoken && !text.empty())
        throw std::invalid_argument("a message is text or speech, not both");
      if (spoken) {
        std::size_t samples = 0;
        if (!conversation_->AppendAudio(*spoken, &failure) ||
            !conversation_->CommitAudio(&samples, &failure)) {
          std::string ignored;
          (void)conversation_->ClearAudio(&ignored);
          throw std::runtime_error(failure.empty() ? "empty input_audio"
                                                   : failure);
        }
      } else if (!conversation_->AddUserText(text, &failure))
        throw std::runtime_error(failure);
      return Created(std::move(echo));
    }
    if (type == "response.create") {
      Members(message, {"type", "event_id", "response"}, "event");
      auto settings = settings_;
      if (const auto* response = message.find("response")) {
        Members(*response, {"output_modalities", "max_output_tokens"},
                "response");
        if (const auto* modalities = response->find("output_modalities"))
          settings.audio_output = AudioModality(*modalities);
        if (const auto* tokens = response->find("max_output_tokens"))
          settings.max_output_tokens = OutputTokens(*tokens);
        if (settings.reasoning && settings.audio_output)
          throw std::invalid_argument(
              "reasoning applies to text output_modalities");
      }
      FlushOutputs();
      return Respond(settings);
    }
    throw std::invalid_argument("unsupported realtime conversation event: " +
                                type);
  }
  static void Format(const json::Value& format) {
    Members(format, {"type", "rate"}, "audio format");
    if (format.member_str("type") != "audio/pcm" ||
        (format.find("rate") && (!format.find("rate")->is_number() ||
                                 format.find("rate")->as_double() != kRate)))
      throw std::invalid_argument("audio is mono PCM16 at 24000 Hz");
  }
  bool Respond(const VoiceConversation::Settings& settings) {
    const auto response_id = "resp_" + id_ + "_" + std::to_string(++responses_);
    last_reply_item_.clear();
    auto response = json::Value::object();
    response["id"] = response_id;
    response["object"] = "realtime.response";
    response["status"] = "in_progress";
    response["output"] = json::Value::array();
    response["output_modalities"] = json::Value::array();
    response["output_modalities"].push_back(settings.audio_output ? "audio"
                                                                  : "text");
    if (!Event("response.created", "response", response))
      return false;
    // The message item opens with the first text or audio.
    std::string item;
    std::string transcript, reasoning;
    json::Value output = json::Value::array();
    bool sent = true;
    const auto open = [&]() {
      if (!item.empty())
        return true;
      item = NextItem();
      auto added = json::Value::object();
      added["type"] = "response.output_item.added";
      added["response_id"] = response_id;
      added["output_index"] = 0;
      added["item"] = json::parse(
          R"({"object":"realtime.item","type":"message","role":"assistant","status":"in_progress","content":[]})");
      added["item"]["id"] = item;
      if (!Send(std::move(added)))
        return false;
      auto part = json::Value::object();
      part["type"] = "response.content_part.added";
      part["response_id"] = response_id;
      part["item_id"] = item;
      part["output_index"] = 0;
      part["content_index"] = 0;
      part["part"] = json::Value::object();
      part["part"]["type"] = settings.audio_output ? "audio" : "text";
      part["part"][settings.audio_output ? "transcript" : "text"] = "";
      return Send(std::move(part));
    };
    const auto delta = [&](std::string type, std::string text) {
      auto event = json::Value::object();
      event["type"] = std::move(type);
      event["response_id"] = response_id;
      event["item_id"] = item;
      event["output_index"] = 0;
      event["content_index"] = 0;
      event["delta"] = std::move(text);
      return Send(std::move(event));
    };
    VoiceConversation::Callbacks callbacks;
    // Speech detected by the server interrupts the reply.
    callbacks.cancelled = [&] {
      return cancel_.load() || !sent || (speaking_.load() && interrupt_.load());
    };
    callbacks.on_reasoning = [&](std::string_view text) {
      reasoning += text;
      return sent = open() &&
                    delta("response.reasoning_text.delta", std::string(text));
    };
    callbacks.on_text = [&](std::string_view text) {
      transcript += text;
      return sent =
                 open() && delta(settings.audio_output
                                     ? "response.output_audio_transcript.delta"
                                     : "response.output_text.delta",
                                 std::string(text));
    };
    callbacks.on_audio = [&](std::span<const float> samples) {
      // Playback on the client ends no sooner than this audio's duration
      // after it arrives (or after the audio before it).
      const auto now = Now();
      playing_until_ = std::max(playing_until_.load(), now) +
                       std::int64_t(samples.size()) * 1'000'000 / kRate;
      const auto pcm = EncodePcm16(samples);
      return sent = open() && delta("response.output_audio.delta",
                                    EncodeAudioBase64(pcm));
    };
    VoiceConversation::Reply reply;
    std::string failure;
    cancel_ = false;
    responding_ = true;
    const bool ok =
        conversation_->Respond(settings, callbacks, &reply, &failure);
    responding_ = false;
    cancel_ = false;
    if (!sent || socket_.cancelled())
      return false;
    std::string status = !ok                ? "failed"
                         : reply.cancelled  ? "cancelled"
                         : reply.incomplete ? "incomplete"
                                            : "completed";
    if (!item.empty()) {
      const char* kind = settings.audio_output ? "audio" : "text";
      if (settings.audio_output) {
        auto done = json::Value::object();
        done["type"] = "response.output_audio.done";
        done["response_id"] = response_id;
        done["item_id"] = item;
        done["output_index"] = 0;
        done["content_index"] = 0;
        if (!Send(std::move(done)))
          return false;
      }
      auto text_done = json::Value::object();
      text_done["type"] = settings.audio_output
                              ? "response.output_audio_transcript.done"
                              : "response.output_text.done";
      text_done["response_id"] = response_id;
      text_done["item_id"] = item;
      text_done["output_index"] = 0;
      text_done["content_index"] = 0;
      text_done[settings.audio_output ? "transcript" : "text"] = transcript;
      if (!Send(std::move(text_done)))
        return false;
      auto part = json::Value::object();
      part["type"] = kind;
      part[settings.audio_output ? "transcript" : "text"] = transcript;
      auto part_done = json::Value::object();
      part_done["type"] = "response.content_part.done";
      part_done["response_id"] = response_id;
      part_done["item_id"] = item;
      part_done["output_index"] = 0;
      part_done["content_index"] = 0;
      part_done["part"] = part;
      if (!Send(std::move(part_done)))
        return false;
      auto message = json::parse(
          R"({"object":"realtime.item","type":"message","role":"assistant"})");
      message["id"] = item;
      message["status"] = status == "completed" ? "completed" : "incomplete";
      message["content"] = json::Value::array();
      message["content"].push_back(part);
      auto item_done = json::Value::object();
      item_done["type"] = "response.output_item.done";
      item_done["response_id"] = response_id;
      item_done["output_index"] = 0;
      item_done["item"] = message;
      if (!Send(std::move(item_done)))
        return false;
      previous_item_ = item;
      last_reply_item_ = item;
      output.push_back(std::move(message));
    }
    for (const auto& call : reply.calls) {
      const auto call_item = NextItem();
      const auto index = std::int64_t(output.items().size());
      auto entry = json::Value::object();
      entry["id"] = call_item;
      entry["object"] = "realtime.item";
      entry["type"] = "function_call";
      entry["status"] = "in_progress";
      entry["call_id"] = "call_" + call_item;
      entry["name"] = call.name;
      entry["arguments"] = "";
      auto added = json::Value::object();
      added["type"] = "response.output_item.added";
      added["response_id"] = response_id;
      added["output_index"] = index;
      added["item"] = entry;
      if (!Send(std::move(added)))
        return false;
      auto arguments = json::Value::object();
      arguments["type"] = "response.function_call_arguments.done";
      arguments["response_id"] = response_id;
      arguments["item_id"] = call_item;
      arguments["output_index"] = index;
      arguments["call_id"] = "call_" + call_item;
      arguments["name"] = call.name;
      arguments["arguments"] = call.arguments;
      if (!Send(std::move(arguments)))
        return false;
      entry["status"] = "completed";
      entry["arguments"] = call.arguments;
      auto done = json::Value::object();
      done["type"] = "response.output_item.done";
      done["response_id"] = response_id;
      done["output_index"] = index;
      done["item"] = entry;
      if (!Send(std::move(done)))
        return false;
      previous_item_ = call_item;
      output.push_back(std::move(entry));
    }
    response["status"] = status;
    response["output"] = std::move(output);
    if (!ok) {
      response["status_details"] = json::Value::object();
      response["status_details"]["type"] = "failed";
      response["status_details"]["error"] = json::Value::object();
      response["status_details"]["error"]["type"] = "server_error";
      response["status_details"]["error"]["message"] = failure;
    } else {
      auto usage = json::Value::object();
      usage["input_tokens"] = reply.input_tokens;
      usage["output_tokens"] = reply.output_tokens;
      usage["total_tokens"] = reply.input_tokens + reply.output_tokens;
      response["usage"] = std::move(usage);
      if (!reasoning.empty())
        response["reasoning"] = reasoning;
    }
    Logger::Info("asr",
                 "request=" + id_ + " event=response_done response=" +
                     response_id + " status=" + status +
                     " audio_samples=" + std::to_string(reply.audio_samples) +
                     " calls=" + std::to_string(reply.calls.size()) +
                     " first_text_ms=" + std::to_string(reply.first_text_ms) +
                     " first_audio_ms=" + std::to_string(reply.first_audio_ms) +
                     " total_ms=" + std::to_string(reply.total_ms));
    return Event("response.done", "response", std::move(response));
  }

  WebSocket& socket_;
  AsrService& service_;
  std::string id_;
  std::unique_ptr<VoiceConversation> conversation_;
  json::Value config_;
  VoiceConversation::Settings settings_;
  std::atomic<std::size_t> sequence_{0};
  std::atomic<bool> cancel_{false}, responding_{false};
  // The server hears the user speak (turn detection), and whether that
  // interrupts a reply.
  std::atomic<bool> speaking_{false}, interrupt_{false};
  // Estimated end of the reply's playback on the client (Now() clock).
  std::atomic<std::int64_t> playing_until_{0};
  static std::int64_t Now() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }
  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<std::pair<json::Value, std::size_t>> queue_;
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
  // Worker state; items are numbered by both threads.
  std::atomic<std::size_t> items_{0};
  std::size_t responses_{0}, turn_samples_{0};
  std::string previous_item_, last_reply_item_;
  std::vector<std::string> outputs_;
};
}  // namespace

HttpResponse HandleVoiceWebSocket(const HttpRequest& request,
                                  AsrService& service) {
  return UpgradeWebSocket(
      request, [&service, id = request.request_id](WebSocket& socket) {
        std::string failure;
        auto conversation = service.CreateVoiceConversation(kRate, &failure);
        if (!conversation) {
          auto event = json::Value::object();
          event["type"] = "error";
          event["error"] = json::Value::object();
          event["error"]["type"] = "server_error";
          event["error"]["code"] = "session_unavailable";
          event["error"]["message"] = failure;
          (void)socket.SendText(event.dump());
          socket.Close(1013);
          return;
        }
        Logger::Info("asr", "request=" + id + " event=conversation_open");
        Session(socket, service, id, std::move(conversation)).Run();
        Logger::Info("asr", "request=" + id + " event=conversation_closed");
      });
}

}  // namespace gufo::server
