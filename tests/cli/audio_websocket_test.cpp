#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "src/cli/serve/asr_service.hpp"
#include "src/cli/serve/audio_stream.hpp"
#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/realtime_audio.hpp"
#include "src/cli/serve/tts_service.hpp"
#include "src/cli/serve/voice_activity.hpp"
#include "src/cli/serve/websocket.hpp"
#include "src/core/json.hpp"
#include "src/core/platform/net.hpp"

namespace {
namespace net = gufo::net;
using namespace gufo::server;
namespace json = gufo::json;

class Client {
public:
  Client(int port, std::string_view path, std::string first = {},
         std::string_view key = "dGhlIHNhbXBsZSBub25jZQ==",
         std::string_view authorization = "", std::string_view extra = {}) {
    fd = ::socket(AF_INET, SOCK_STREAM, 0);
    assert(fd >= 0);
    assert(net::SetSocketTimeouts(fd, 3'000));
    sockaddr_in address{.sin_family = AF_INET, .sin_port = htons(port)};
    assert(::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr) == 1);
    assert(::connect(fd, reinterpret_cast<sockaddr*>(&address),
                     sizeof(address)) == 0);
    Send("GET " + std::string(path) +
         " HTTP/1.1\r\nHost: localhost\r\nUpgrade: other,\twebsocket\r\n"
         "Connection: keep-alive,\tUpgrade\r\nSec-WebSocket-Version: "
         "13\r\nSec-WebSocket-Key: " +
         std::string(key) + "\r\n" +
         (authorization.empty()
              ? std::string()
              : "Authorization: " + std::string(authorization) + "\r\n") +
         std::string(extra) + "\r\n" + first);
    while (!headers.ends_with("\r\n\r\n")) {
      char c;
      Read(&c, 1);
      headers += c;
    }
  }
  ~Client() {
    ::shutdown(fd, SHUT_RDWR);
    net::CloseSocket(fd);
  }
  void Send(std::string_view bytes) {
    while (!bytes.empty()) {
      const auto count = static_cast<ssize_t>(
          net::SendNoSignal(fd, bytes.data(), bytes.size()));
      assert(count > 0);
      bytes.remove_prefix(static_cast<std::size_t>(count));
    }
  }
  static std::string Frame(std::string_view payload, unsigned opcode = 1,
                           bool final = true, bool masked = true) {
    std::string frame(1, static_cast<char>((final ? 128 : 0) | opcode));
    const unsigned mask = masked ? 128 : 0;
    if (payload.size() < 126)
      frame += static_cast<char>(mask | payload.size());
    else {
      assert(payload.size() <= 65535);
      frame += static_cast<char>(mask | 126);
      frame += static_cast<char>(payload.size() >> 8);
      frame += static_cast<char>(payload.size() & 255);
    }
    if (masked)
      frame += "abcd";
    for (std::size_t i = 0; i < payload.size(); ++i)
      frame += static_cast<char>(payload[i] ^ (masked ? "abcd"[i % 4] : 0));
    return frame;
  }
  void Text(std::string_view text) { Send(Frame(text)); }
  std::pair<unsigned, std::string> Receive() {
    std::array<unsigned char, 2> head{};
    Read(reinterpret_cast<char*>(head.data()), 2);
    assert((head[0] & 128) && !(head[1] & 128));
    std::size_t size = head[1] & 127;
    if (size >= 126) {
      const int count = size == 126 ? 2 : 8;
      size = 0;
      for (int i = 0; i < count; ++i) {
        unsigned char byte;
        Read(reinterpret_cast<char*>(&byte), 1);
        size = size * 256 + byte;
      }
    }
    assert(size < (4U << 20));
    std::string payload(size, '\0');
    Read(payload.data(), payload.size());
    return {head[0] & 15, std::move(payload)};
  }
  std::string Next() {
    auto [opcode, payload] = Receive();
    assert(opcode == 1);
    return json::parse(payload).member_str("type");
  }
  json::Value Event(std::string_view type) {
    auto [opcode, payload] = Receive();
    assert(opcode == 1);
    auto event = json::parse(payload);
    if (event.member_str("type") != type)
      std::fprintf(stderr, "expected %s, got %s\n", std::string(type).c_str(),
                   payload.c_str());
    assert(event.member_str("type") == type);
    return event;
  }
  int fd;
  std::string headers;

private:
  void Read(char* bytes, std::size_t size) {
    while (size) {
      const auto count = ::recv(fd, bytes, size, 0);
      assert(count > 0);
      bytes += count;
      size -= static_cast<std::size_t>(count);
    }
  }
};

// The raw response to a JSON POST.
std::string HttpPost(int port, std::string_view path, std::string_view body) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  assert(fd >= 0 && net::SetSocketTimeouts(fd, 3'000));
  sockaddr_in address{.sin_family = AF_INET, .sin_port = htons(port)};
  assert(::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr) == 1);
  assert(::connect(fd, reinterpret_cast<sockaddr*>(&address),
                   sizeof(address)) == 0);
  const std::string request =
      "POST " + std::string(path) +
      " HTTP/1.1\r\nHost: localhost\r\n"
      "Content-Type: application/json\r\nContent-Length: " +
      std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" +
      std::string(body);
  assert(net::SendNoSignal(fd, request.data(), request.size()) ==
         static_cast<long>(request.size()));
  std::string response;
  char buffer[4096];
  for (ssize_t n; (n = ::recv(fd, buffer, sizeof(buffer), 0)) > 0;)
    response.append(buffer, std::size_t(n));
  net::CloseSocket(fd);
  return response;
}

// The raw response to a plain GET.
std::string HttpGet(int port, std::string_view path) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  assert(fd >= 0 && net::SetSocketTimeouts(fd, 3'000));
  sockaddr_in address{.sin_family = AF_INET, .sin_port = htons(port)};
  assert(::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr) == 1);
  assert(::connect(fd, reinterpret_cast<sockaddr*>(&address),
                   sizeof(address)) == 0);
  const std::string request = "GET " + std::string(path) +
                              " HTTP/1.1\r\nHost: localhost\r\nConnection: "
                              "close\r\n\r\n";
  assert(net::SendNoSignal(fd, request.data(), request.size()) ==
         static_cast<long>(request.size()));
  std::string response;
  char buffer[4096];
  for (ssize_t n; (n = ::recv(fd, buffer, sizeof(buffer), 0)) > 0;)
    response.append(buffer, std::size_t(n));
  net::CloseSocket(fd);
  return response;
}
// Text of every pushed piece of nonzero audio; the transcript of a stream is
// its pieces' text.
class FakeLive final : public LiveTranscription {
public:
  bool Push(std::span<const float> samples, const Text& on_text,
            std::string*) override {
    assert(!samples.empty());
    text_ += "a";
    return on_text("a");
  }
  bool Finish(const Text& on_text, std::string*) override {
    text_ += ".";
    return on_text(".");
  }
  const std::string& text() const override { return text_; }

private:
  std::string text_;
};
// Replies "Hi" with three samples of speech; "call" asks for a tool and
// "wait" runs until cancelled.
// With `deleted`, it lists voices and deletes turns, counting them there;
// without, it keeps the interface's defaults.
class FakeVoice final : public VoiceConversation {
public:
  explicit FakeVoice(std::atomic<int>* deleted = nullptr) : deleted_(deleted) {}
  bool Configure(const Settings& settings, std::string*) override {
    settings_ = settings;
    return true;
  }
  bool AppendAudio(std::span<const float> samples, std::string*) override {
    audio_ += samples.size();
    return true;
  }
  bool CommitAudio(std::size_t* samples, std::string*) override {
    *samples = audio_;
    audio_ = 0;
    last_ = "audio";
    return *samples > 0;
  }
  bool ClearAudio(std::string*) override {
    audio_ = 0;
    return true;
  }
  bool audio_open() const override { return audio_ > 0; }
  bool TruncateReply(std::size_t samples, std::string*) override {
    truncated = samples;
    return true;
  }
  std::size_t truncated{0};
  std::vector<std::string> voices() const override {
    if (!deleted_)
      return {};
    return {"female", "male"};
  }
  bool supports_response_options() const override { return deleted_; }
  bool DeleteUserTurn(std::string* error) override {
    if (!deleted_)
      return VoiceConversation::DeleteUserTurn(error);
    ++*deleted_;
    return true;
  }
  bool AddUserText(std::string_view text, std::string*) override {
    last_ = text;
    return true;
  }
  bool AddToolOutputs(std::span<const std::string> outputs,
                      std::string*) override {
    assert(outputs.size() == 1 && outputs[0] == "{\"ok\":true}");
    last_ = "tool";
    return true;
  }
  bool Respond(const Settings& settings, const Callbacks& callbacks,
               Reply* reply, std::string*) override {
    *reply = {};
    if (last_ == "call") {
      assert(settings_.tools.size() == 1 &&
             settings_.tools[0] ==
                 R"({"name":"set_volume","parameters":{"type":"object"}})");
      reply->calls.push_back({"set_volume", "{\"volume\":30}"});
      return true;
    }
    if (last_ == "wait") {
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (!callbacks.cancelled() &&
             std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
      reply->cancelled = callbacks.cancelled();
      return true;
    }
    assert(!deleted_ || settings.voice == "female");
    const std::vector<float> speech{-.5F, 0, .5F};
    reply->text = "Hi";
    reply->audio_samples = speech.size();
    return callbacks.on_text("Hi") &&
           (!settings.audio_output || callbacks.on_audio(speech));
  }

private:
  std::atomic<int>* deleted_;
  Settings settings_;
  std::size_t audio_{0};
  std::string last_;
};

// A client that only listens to streamed output stays connected while the
// server sends binary frames.
void ListeningClientIsNotIdle() {
  HttpServer server("127.0.0.1", 0, nullptr);
  server.add("GET", "/ws-stream", [](const HttpRequest& req, auto&) {
    return UpgradeWebSocket(req, [](WebSocket& socket) {
      for (int i = 0; i < 10; ++i) {
        assert(socket.SendBinary("audio"));
        assert(!socket.cancelled());
        if (i + 1 < 10)
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
      socket.Close(1000);
    });
  });
  std::string error;
  assert(server.start(&error));
  std::jthread thread([&] { server.run(); });

  Client client(server.port(), "/ws-stream");
  assert(client.headers.find("101 ") != std::string::npos);
  for (int i = 0; i < 10; ++i) {
    const auto [opcode, payload] = client.Receive();
    assert(opcode == 2 && payload == "audio");
  }
  server.stop();
}

// An upgraded session keeps its connection slot until it ends, so open
// WebSockets count toward the connection limit like any request in progress.
void WebSocketHoldsConnectionSlot() {
  HttpServer server("127.0.0.1", 0, nullptr, nullptr, nullptr, nullptr,
                    {.max_connections = 1});
  server.add("GET", "/ws-echo", [](const HttpRequest& req, auto&) {
    return UpgradeWebSocket(req, [](WebSocket& socket) {
      std::string text;
      while (socket.Receive(&text) && text != "bye")
        assert(socket.SendText(text));
    });
  });
  std::string error;
  assert(server.start(&error));
  std::jthread thread([&] { server.run(); });
  {
    Client client(server.port(), "/ws-echo");
    assert(client.headers.starts_with("HTTP/1.1 101"));
    client.Text("ping");
    const auto [opcode, payload] = client.Receive();
    assert(opcode == 1 && payload == "ping");
    assert(HttpGet(server.port(), "/missing").starts_with("HTTP/1.1 503"));
    client.Text("bye");
    assert(client.Receive().first == 8);
  }
  // The slot frees once the session has closed.
  bool admitted = false;
  for (int attempt = 0; attempt < 200 && !admitted; ++attempt) {
    admitted = HttpGet(server.port(), "/missing").starts_with("HTTP/1.1 404");
    if (!admitted)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  assert(admitted);
  server.stop();
}

// 24 kHz PCM16 of silence (quiet noise) or a speech-loud tone.
std::string Pcm(double seconds, bool loud) {
  std::vector<float> samples(std::size_t(seconds * 24000));
  std::uint32_t state = 7;
  for (std::size_t i = 0; i < samples.size(); ++i) {
    state = state * 1664525U + 1013904223U;
    const float noise = (float(state >> 8) / float(1U << 24) - 0.5F) * 0.002F;
    samples[i] = noise + (loud ? 0.3F * std::sin(0.11F * float(i)) : 0.0F);
  }
  return EncodePcm16(samples);
}
void VoiceActivityDetects() {
  VoiceActivity vad({.threshold = 0.5, .silence_duration_ms = 300}, 24000);
  auto samples = [](const std::string& pcm) {
    std::vector<float> out(pcm.size() / 2);
    for (std::size_t i = 0; i < out.size(); ++i)
      out[i] = float(std::int16_t(std::uint16_t(
                   static_cast<unsigned char>(pcm[2 * i]) |
                   (static_cast<unsigned char>(pcm[2 * i + 1]) << 8)))) /
               32768.0F;
    return out;
  };
  assert(vad.Push(samples(Pcm(1.0, false))).empty());
  const auto start = vad.Push(samples(Pcm(0.5, true)));
  assert(start.size() == 1 && start[0].start);
  // The first voiced frame, within one frame of the tone's onset.
  assert(start[0].sample >= 24000 && start[0].sample <= 24000 + 480);
  assert(vad.speaking());
  const auto stop = vad.Push(samples(Pcm(0.5, false)));
  assert(stop.size() == 1 && !stop[0].start);
  assert(stop[0].sample >= 36000 + 7200 && stop[0].sample <= 36000 + 7680);
  assert(!vad.speaking());
  // A 60 ms click starts nothing; 140 ms of speech is too little for a turn.
  assert(vad.Push(samples(Pcm(0.06, true) + Pcm(0.5, false))).empty());
  const auto burst = vad.Push(samples(Pcm(0.14, true) + Pcm(0.5, false)));
  assert(burst.size() == 2 && burst[1].voiced_ms < VoiceActivity::kMinSpeechMs);
  // While the reply plays, quiet speech (its echo) starts no turn; loud
  // speech does.
  vad.Guard(true);
  std::vector<float> quiet(12000);
  for (std::size_t i = 0; i < quiet.size(); ++i)
    quiet[i] = 0.01F * std::sin(0.11F * float(i));
  assert(vad.Push(quiet).empty());
  assert(vad.Push(samples(Pcm(0.3, true))).size() == 1);
  // Steady loud noise is the floor, not speech.
  VoiceActivity steady({}, 24000);
  bool started = false;
  for (int i = 0; i < 20; ++i)
    for (const auto& event : steady.Push(samples(Pcm(0.25, true))))
      started = started || (event.start && event.sample > 24000 * 4);
  assert(!started);
  bool rejected = false;
  try {
    (void)ParseTurnDetection(json::parse(R"({"type":"semantic_vad"})"));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}
// Live transcription and spoken conversations of a model that hears and
// speaks, against fake streams.
void RealtimeStreams() {
  auto asr = std::make_shared<AsrService>(AsrServiceOptions{
      .validate_model = false,
      .model_id = "voice",
      .runner = [](const auto&, const auto&, auto*, auto*) { return false; },
      .live =
          [](std::uint32_t rate, std::string*) {
            assert(rate == 24000);
            return std::unique_ptr<LiveTranscription>(new FakeLive);
          },
      .voice =
          [](std::uint32_t rate, std::string*) {
            assert(rate == 24000);
            return std::unique_ptr<VoiceConversation>(new FakeVoice);
          }});
  HttpServer server("127.0.0.1", 0, nullptr, nullptr, nullptr, asr, {});
  std::string error;
  assert(server.start(&error));
  std::jthread worker([&] { server.run(); });
  const auto append = [](std::size_t bytes) {
    return R"({"type":"input_audio_buffer.append","audio":")" +
           EncodeAudioBase64(std::string(bytes, '\1')) + "\"}";
  };
  {
    // Text arrives while audio is appended, for the item commit creates.
    Client client(server.port(), "/v1/realtime?intent=transcription");
    client.Event("session.created");
    client.Text(append(4800));
    auto delta =
        client.Event("conversation.item.input_audio_transcription.delta");
    assert(delta.member_str("delta") == "a");
    const auto item = delta.member_str("item_id");
    client.Text(R"({"type":"input_audio_buffer.commit"})");
    assert(client.Event("input_audio_buffer.committed").member_str("item_id") ==
           item);
    client.Event("conversation.item.created");
    assert(client.Event("conversation.item.input_audio_transcription.delta")
               .member_str("delta") == ".");
    auto done =
        client.Event("conversation.item.input_audio_transcription.completed");
    assert(done.member_str("transcript") == "a." &&
           done.member_str("item_id") == item);
    // Cleared audio leaves no transcript behind.
    client.Text(append(4800));
    client.Event("conversation.item.input_audio_transcription.delta");
    client.Text(R"({"type":"input_audio_buffer.clear"})");
    client.Event("input_audio_buffer.cleared");
    client.Text(append(4800));
    client.Event("conversation.item.input_audio_transcription.delta");
    client.Text(R"({"type":"input_audio_buffer.commit"})");
    client.Event("input_audio_buffer.committed");
    client.Event("conversation.item.created");
    client.Event("conversation.item.input_audio_transcription.delta");
    assert(client.Event("conversation.item.input_audio_transcription.completed")
               .member_str("transcript") == "a.");
  }
  {
    // Without an intent the session is a conversation.
    Client client(server.port(), "/v1/realtime");
    assert(client.Event("session.created")["session"].member_str("type") ==
           "realtime");
    client.Text(append(4800));
    client.Text(R"({"type":"input_audio_buffer.commit"})");
    client.Event("input_audio_buffer.committed");
    auto user = client.Event("conversation.item.created");
    assert(user["item"].member_str("role") == "user");
    client.Text(R"({"type":"response.create"})");
    auto created = client.Event("response.created");
    const auto id = created["response"].member_str("id");
    auto added = client.Event("response.output_item.added");
    assert(added["item"].member_str("type") == "message");
    client.Event("response.content_part.added");
    assert(client.Event("response.output_audio_transcript.delta")
               .member_str("delta") == "Hi");
    assert(client.Event("response.output_audio.delta").member_str("delta") ==
           EncodeAudioBase64(EncodePcm16(std::vector<float>{-.5F, 0, .5F})));
    client.Event("response.output_audio.done");
    assert(client.Event("response.output_audio_transcript.done")
               .member_str("transcript") == "Hi");
    client.Event("response.content_part.done");
    client.Event("response.output_item.done");
    auto done = client.Event("response.done");
    assert(done["response"].member_str("id") == id &&
           done["response"].member_str("status") == "completed");
    // Text replies; tools are listed without their Realtime type.
    client.Text(
        R"({"type":"session.update","session":{"output_modalities":["text"],"tools":[{"type":"function","name":"set_volume","parameters":{"type":"object"}}]}})");
    client.Event("session.updated");
    client.Text(
        R"({"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"call"}]}})");
    client.Event("conversation.item.created");
    client.Text(R"({"type":"response.create"})");
    client.Event("response.created");
    auto call = client.Event("response.output_item.added");
    assert(call["item"].member_str("type") == "function_call" &&
           call["item"].member_str("name") == "set_volume");
    auto arguments = client.Event("response.function_call_arguments.done");
    assert(arguments.member_str("arguments") == R"({"volume":30})");
    client.Event("response.output_item.done");
    done = client.Event("response.done");
    assert(done["response"]["output"].items().size() == 1);
    client.Text(
        R"({"type":"conversation.item.create","item":{"type":"function_call_output","call_id":")" +
        arguments.member_str("call_id") + R"(","output":"{\"ok\":true}"}})");
    client.Event("conversation.item.created");
    client.Text(R"({"type":"response.create"})");
    client.Event("response.created");
    client.Event("response.output_item.added");
    client.Event("response.content_part.added");
    assert(client.Event("response.output_text.delta").member_str("delta") ==
           "Hi");
    client.Event("response.output_text.done");
    client.Event("response.content_part.done");
    client.Event("response.output_item.done");
    client.Event("response.done");
    // response.cancel interrupts a running reply.
    client.Text(
        R"({"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"wait"}]}})");
    client.Event("conversation.item.created");
    client.Text(R"({"type":"response.create"})");
    client.Event("response.created");
    client.Text(R"({"type":"response.cancel"})");
    assert(client.Event("response.done")["response"].member_str("status") ==
           "cancelled");
    client.Text(R"({"type":"response.cancel"})");
    assert(client.Event("error")["error"].member_str("code") ==
           "response_cancel_not_active");
    client.Text(
        R"({"type":"session.update","session":{"audio":{"input":{"turn_detection":{"type":"semantic_vad"}}}}})");
    client.Event("error");
    // Conversations without voices or deletion keep rejecting both.
    client.Text(
        R"({"type":"session.update","session":{"audio":{"output":{"voice":"female"}}}})");
    assert(client.Event("error")["error"].member_str("message") ==
           "audio.output.voice female is not available: this model speaks "
           "its own voice");
    client.Text(
        R"({"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"x"}]}})");
    const auto text_item =
        client.Event("conversation.item.created")["item"].member_str("id");
    client.Text(R"({"type":"conversation.item.delete","item_id":")" +
                text_item + R"("})");
    assert(client.Event("error")["error"].member_str("message") ==
           "this conversation cannot delete items");
    // Server turn detection: speech commits itself and gets a reply.
    client.Text(
        R"({"type":"session.update","session":{"output_modalities":["audio"],"tools":[],"audio":{"input":{"turn_detection":{"type":"server_vad","silence_duration_ms":200}}}}})");
    auto updated = client.Event("session.updated");
    assert(updated["session"]["audio"]["input"]["turn_detection"].member_str(
               "type") == "server_vad");
    const auto speak = [&](const std::string& pcm) {
      for (std::size_t at = 0; at < pcm.size(); at += 4800)
        client.Text(R"({"type":"input_audio_buffer.append","audio":")" +
                    EncodeAudioBase64(pcm.substr(at, 4800)) + "\"}");
    };
    speak(Pcm(0.5, false) + Pcm(0.6, true) + Pcm(0.4, false));
    auto started = client.Event("input_audio_buffer.speech_started");
    assert(started.find("audio_start_ms")->as_double() >= 480 &&
           started.find("audio_start_ms")->as_double() <= 520);
    const auto spoken = started.member_str("item_id");
    client.Event("input_audio_buffer.speech_stopped");
    assert(client.Event("input_audio_buffer.committed").member_str("item_id") ==
           spoken);
    client.Event("conversation.item.created");
    client.Event("response.created");
    auto message = client.Event("response.output_item.added");
    const auto reply_item = message["item"].member_str("id");
    for (const auto* type :
         {"response.content_part.added",
          "response.output_audio_transcript.delta",
          "response.output_audio.delta", "response.output_audio.done",
          "response.output_audio_transcript.done", "response.content_part.done",
          "response.output_item.done"})
      client.Event(type);
    assert(client.Event("response.done")["response"].member_str("status") ==
           "completed");
    client.Text(R"({"type":"conversation.item.truncate","item_id":")" +
                reply_item + R"(","content_index":0,"audio_end_ms":40})");
    assert(client.Event("conversation.item.truncated")
               .find("audio_end_ms")
               ->as_double() == 40);
    // Speech during a reply interrupts it.
    client.Text(
        R"({"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"wait"}]}})");
    client.Event("conversation.item.created");
    client.Text(R"({"type":"response.create"})");
    client.Event("response.created");
    speak(Pcm(0.6, true));
    client.Event("input_audio_buffer.speech_started");
    assert(client.Event("response.done")["response"].member_str("status") ==
           "cancelled");
    // Wait for cancellation before sending silence: the response worker and
    // incoming audio run independently, so their events can otherwise race.
    speak(Pcm(0.4, false));
    client.Event("input_audio_buffer.speech_stopped");
    client.Event("input_audio_buffer.committed");
    client.Event("conversation.item.created");
    client.Event("response.created");
  }
  {
    // Transcription with server turn detection commits by itself.
    Client client(server.port(), "/v1/realtime?intent=transcription");
    client.Event("session.created");
    client.Text(
        R"({"type":"session.update","session":{"type":"transcription","audio":{"input":{"turn_detection":{"type":"server_vad","silence_duration_ms":200}}}}})");
    client.Event("session.updated");
    const auto pcm = Pcm(0.3, false) + Pcm(0.6, true) + Pcm(0.4, false);
    for (std::size_t at = 0; at < pcm.size(); at += 4800)
      client.Text(R"({"type":"input_audio_buffer.append","audio":")" +
                  EncodeAudioBase64(pcm.substr(at, 4800)) + "\"}");
    const auto item =
        client.Event("input_audio_buffer.speech_started").member_str("item_id");
    std::string type;
    while ((type = client.Next()) ==
           "conversation.item.input_audio_transcription.delta") {
    }
    assert(type == "input_audio_buffer.speech_stopped");
    assert(client.Event("input_audio_buffer.committed").member_str("item_id") ==
           item);
    client.Event("conversation.item.created");
    while ((type = client.Next()) ==
           "conversation.item.input_audio_transcription.delta") {
    }
    assert(type == "conversation.item.input_audio_transcription.completed");
  }
  {
    // Browsers may keep offering the legacy key subprotocol; the handshake
    // selects "realtime" and ignores the rest.
    Client browser(server.port(), "/v1/realtime", {},
                   "dGhlIHNhbXBsZSBub25jZQ==", "",
                   "Sec-WebSocket-Protocol: realtime, "
                   "openai-insecure-api-key.test\r\n");
    assert(browser.headers.starts_with("HTTP/1.1 101"));
    auto lower_headers = browser.headers;
    for (char& c : lower_headers)
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    assert(lower_headers.find("sec-websocket-protocol: realtime") !=
           std::string::npos);
    browser.Event("session.created");
    const auto page = HttpGet(server.port(), "/voice");
    assert(page.starts_with("HTTP/1.1 200") &&
           page.find("text/html") != std::string::npos &&
           page.find("/v1/realtime") != std::string::npos);
  }
  server.stop();
  worker.join();
}
// Open WebUI 0.13 voice calls: the session.update it sends, a transcript
// for each turn before the client asks for the reply, and deletion of turns
// without one. The llama-swap playground asks for both output modalities.
void OpenWebUiVoiceCall() {
  std::atomic<int> turns{0}, deleted{0};
  auto asr = std::make_shared<AsrService>(AsrServiceOptions{
      .validate_model = false,
      .model_id = "voice",
      .runner =
          [&](const auto& request, const auto&, auto* result, auto* error) {
            // The turn from its prefix padding to the end of the silence
            // that ends it, at 16 kHz.
            assert(request.wav.empty() && request.pcm16k.size() >= 14400 &&
                   request.pcm16k.size() <= 30400);
            assert(request.language == std::optional<std::string>("en"));
            if (turns++ > 0) {
              *error = "decoder failed";
              return false;
            }
            result->text = "Hello there.";
            return true;
          },
      .voice =
          [&](std::uint32_t, std::string*) {
            return std::unique_ptr<VoiceConversation>(new FakeVoice(&deleted));
          }});
  HttpServer server("127.0.0.1", 0, nullptr, nullptr, nullptr, asr, {});
  std::string error;
  assert(server.start(&error));
  std::jthread worker([&] { server.run(); });
  Client client(server.port(), "/v1/realtime?model=voice");
  client.Event("session.created");
  client.Text(
      R"({"type":"session.update","session":{"type":"realtime","output_modalities":["audio"],"instructions":"Be brief.","audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"transcription":{"model":"voice","language":"en"},"turn_detection":{"type":"server_vad","interrupt_response":false,"create_response":false}},"output":{"format":{"type":"audio/pcm","rate":24000},"voice":"female"}},"tools":[{"type":"function","name":"generate_chat_completion","parameters":{"type":"object"}}],"tool_choice":"auto"}})");
  auto updated = client.Event("session.updated");
  assert(updated["session"]["audio"]["output"].member_str("voice") == "female");
  assert(updated["session"]["audio"]["input"]["transcription"].member_str(
             "model") == "voice");
  const auto speak = [&](const std::string& pcm) {
    for (std::size_t at = 0; at < pcm.size(); at += 4800)
      client.Text(R"({"type":"input_audio_buffer.append","audio":")" +
                  EncodeAudioBase64(pcm.substr(at, 4800)) + "\"}");
  };
  const auto turn = [&] {
    speak(Pcm(0.5, false) + Pcm(0.6, true) + Pcm(0.8, false));
    const auto item =
        client.Event("input_audio_buffer.speech_started").member_str("item_id");
    client.Event("input_audio_buffer.speech_stopped");
    assert(client.Event("input_audio_buffer.committed").member_str("item_id") ==
           item);
    client.Event("conversation.item.created");
    return item;
  };
  // Chat snapshots: system items that replace each other.
  const auto snapshot = [&](int n) {
    client.Text(
        R"({"type":"conversation.item.create","item":{"id":"chat_context_)" +
        std::to_string(n) +
        R"(","type":"message","role":"system","content":[{"type":"input_text","text":"Snapshot."}]}})");
    client.Event("conversation.item.created");
  };
  snapshot(1);
  snapshot(2);
  client.Text(
      R"({"type":"conversation.item.delete","item_id":"chat_context_1"})");
  client.Event("conversation.item.deleted");
  auto item = turn();
  assert(client.Event("conversation.item.input_audio_transcription.delta")
             .member_str("delta") == "Hello there.");
  auto done =
      client.Event("conversation.item.input_audio_transcription.completed");
  assert(done.member_str("item_id") == item &&
         done.member_str("transcript") == "Hello there.");
  // No reply starts by itself: the next event answers the next request.
  client.Text(R"({"type":"conversation.item.delete","item_id":"missing"})");
  client.Event("error");
  client.Text(
      R"({"type":"response.create","response":{"metadata":{"input_item_id":")" +
      item + R"("},"tools":[],"tool_choice":"none"}})");
  assert(client.Event("response.created")["response"]["metadata"].member_str(
             "input_item_id") == item);
  for (const auto* type :
       {"response.output_item.added", "response.content_part.added",
        "response.output_audio_transcript.delta", "response.output_audio.delta",
        "response.output_audio.done", "response.output_audio_transcript.done",
        "response.content_part.done", "response.output_item.done"})
    client.Event(type);
  assert(client.Event("response.done")["response"].member_str("status") ==
         "completed");
  // A failed transcript; the client deletes the turn.
  item = turn();
  auto failed =
      client.Event("conversation.item.input_audio_transcription.failed");
  assert(failed.member_str("item_id") == item &&
         failed["error"].member_str("message") == "decoder failed");
  client.Text(R"({"type":"conversation.item.delete","item_id":")" + item +
              R"("})");
  assert(client.Event("conversation.item.deleted").member_str("item_id") ==
         item);
  assert(deleted == 1);
  // An out-of-band status line.
  client.Text(
      R"({"type":"response.create","response":{"conversation":"none","input":[],"tools":[],"tool_choice":"none","instructions":"Say: still working.","metadata":{"status":"working"}}})");
  auto status = client.Event("response.created")["response"];
  assert(status["metadata"].member_str("status") == "working" &&
         status["conversation_id"].is_null());
  std::string kind;
  while ((kind = client.Next()) != "response.done") {
  }
  // Voices are the conversation's; another model cannot transcribe.
  client.Text(
      R"({"type":"session.update","session":{"audio":{"output":{"voice":"robot"}}}})");
  assert(client.Event("error")["error"].member_str("message") ==
         "audio.output.voice robot is not one of female male");
  client.Text(
      R"({"type":"session.update","session":{"audio":{"input":{"transcription":{"model":"whisper-1"}}}}})");
  client.Event("error");
  client.Text(
      R"({"type":"session.update","session":{"output_modalities":["audio","text"]}})");
  client.Event("session.updated");
  server.stop();
  worker.join();
}
}  // namespace

int main() {
  ListeningClientIsNotIdle();
  WebSocketHoldsConnectionSlot();
  VoiceActivityDetects();
  std::atomic<int> tts_calls{0}, asr_calls{0};
  std::atomic<bool> entered{false}, cancelled{false};
  auto tts = std::make_shared<TtsService>(TtsServiceOptions{
      .validate_model = false,
      .model_aliases = {"tts-alias"},
      .voices = {"vivian"},
      .runner = [&](const auto& request, const auto& check, auto* result,
                    auto*) {
        ++tts_calls;
        if (request.text == "wait") {
          entered = true;
          const auto deadline =
              std::chrono::steady_clock::now() + std::chrono::seconds(2);
          while (!check() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
          cancelled = check();
          return false;
        }
        result->sample_rate = 24000;
        result->code_groups = 16;
        result->samples = {-.5F, 0, .5F};
        if (request.on_audio)
          return request.on_audio(result->samples) &&
                 request.on_audio(result->samples);
        return true;
      }});
  auto asr = std::make_shared<AsrService>(AsrServiceOptions{
      .validate_model = false,
      .model_aliases = {"asr-alias"},
      .runner = [&](const auto& request, const auto&, auto* result, auto*) {
        ++asr_calls;
        assert(request.pcm16k.size() == 1600);
        assert(request.language == "en");
        result->text = "Hello world.";
        result->chunks = 1;
        return !request.on_text ||
               (request.on_text("Hello") && request.on_text(result->text));
      }});
  HttpServer server("127.0.0.1", 0, nullptr, nullptr, tts, asr, {});
  std::string error;
  assert(server.start(&error));
  std::jthread worker([&] { server.run(); });
  {
    Client invalid(server.port(), "/v1/realtime", {}, "bad");
    assert(invalid.headers.starts_with("HTTP/1.1 400"));
  }
  {
    Client client(
        server.port(), "/v1/realtime?intent=transcription",
        Client::Frame(
            R"({"type":"session.update","session":{"type":"transcription","audio":{"input":{"transcription":{"model":"qwen3-asr","language":"en"},"turn_detection":null}}}})"));
    assert(client.headers.starts_with("HTTP/1.1 101"));
    assert(client.headers.find("s3pPLMBiTxaQ9kYGzzhZRbK+xOo=") !=
           std::string::npos);
    assert(client.headers.find("Connection: close") == std::string::npos);
    client.Event("session.created");
    client.Event("session.updated");
    std::string previous;
    for (int turn = 0; turn < 2; ++turn) {
      const auto encoded = EncodeAudioBase64(std::string(4800, '\0'));
      client.Text(R"({"type":"input_audio_buffer.append","audio":")" + encoded +
                  "\"}");
      client.Text(R"({"type":"input_audio_buffer.commit"})");
      const auto committed = client.Event("input_audio_buffer.committed");
      assert(committed.member_str("previous_item_id") == previous);
      previous = committed.member_str("item_id");
      client.Event("conversation.item.created");
      assert(client.Event("conversation.item.input_audio_transcription.delta")
                 .member_str("delta") == "Hello");
      assert(client.Event("conversation.item.input_audio_transcription.delta")
                 .member_str("delta") == " world.");
      assert(
          client.Event("conversation.item.input_audio_transcription.completed")
              .member_str("transcript") == "Hello world.");
    }
    client.Text(
        R"({"type":"session.update","session":{"audio":{"input":{"turn_detection":{"type":"semantic_vad"}}}}})");
    client.Event("error");
    client.Text(R"({"type":"input_audio_buffer.commit"})");
    client.Event("error");
    assert(asr_calls == 2);
  }
  {
    Client client(server.port(), "/v1/audio/speech/stream");
    client.Text(
        R"({"type":"session.config","voice":"vivian","response_format":"pcm","stream_audio":true,"split_granularity":"sentence"})");
    const std::string message = R"({"type":"input.text","text":"Hello é. "})";
    const auto split = message.find("é") + 1;
    client.Send(
        Client::Frame(std::string_view(message).substr(0, split), 1, false));
    client.Send(Client::Frame("ping", 9));
    assert(client.Receive() == std::make_pair(10U, std::string("ping")));
    client.Send(Client::Frame(std::string_view(message).substr(split), 0));
    auto first_start = client.Event("audio.start");
    assert(first_start.member_str("sentence_text") == "Hello é.");
    assert(first_start.find("sample_rate")->as_double() == 24000);
    const auto expected = EncodePcm16(std::vector<float>{-.5F, 0, .5F});
    assert(client.Receive() == std::make_pair(2U, expected));
    assert(client.Receive() == std::make_pair(2U, expected));
    client.Event("audio.done");
    client.Text(R"({"type":"input.done"})");
    client.Event("session.done");
    client.Text(
        R"({"type":"session.config","voice":"vivian","response_format":"pcm","stream_audio":true})");
    client.Text(R"({"type":"input.text","text":"Next utterance"})");
    client.Text(R"({"type":"input.done"})");
    client.Event("audio.start");
    client.Receive();
    client.Receive();
    client.Event("audio.done");
    client.Event("session.done");
    assert(tts_calls == 2);
    client.Send(Client::Frame("unmasked", 1, true, false));
    auto [opcode, close] = client.Receive();
    assert(opcode == 8 && close == std::string("\x03\xEA", 2));
  }
  {
    Client client(server.port(), "/v1/audio/speech/stream");
    client.Text(
        R"({"type":"session.config","voice":"vivian","response_format":"pcm","stream_audio":true})");
    client.Text(R"({"type":"input.text","text":"wait"})");
    client.Text(R"({"type":"input.done"})");
    client.Event("audio.start");
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!entered && std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    assert(entered);
    client.Send(Client::Frame(std::string("\x03\xE8", 2), 8));
    assert(client.Receive().first == 8);
    while (!cancelled && std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    assert(cancelled);
  }
  // Exercise closure before/while the consumer waits for its first message.
  // Every connection must release its worker and admission slot.
  for (int i = 0; i < 32; ++i) {
    Client client(server.port(), "/v1/audio/speech/stream");
    client.Send(Client::Frame(std::string("\x03\xE8", 2), 8));
    const auto reply = client.Receive();
    assert(reply.first == 8 && reply.second == std::string("\x03\xE8", 2));
  }
  {
    // Request-level aliases (#957): a configured alias synthesizes, an
    // unserved name is refused, both aliases are listed, and the realtime
    // handshake accepts the ASR alias.
    const auto aliased =
        HttpPost(server.port(), "/v1/audio/speech",
                 R"({"model":"tts-alias","input":"Hi","voice":"vivian"})");
    assert(aliased.starts_with("HTTP/1.1 200"));
    assert(HttpPost(server.port(), "/v1/audio/speech",
                    R"({"model":"tts-typo","input":"Hi","voice":"vivian"})")
               .starts_with("HTTP/1.1 400"));
    const auto models = HttpGet(server.port(), "/v1/models");
    assert(models.find("\"id\":\"tts-alias\"") != std::string::npos);
    assert(models.find("\"id\":\"asr-alias\"") != std::string::npos);
    const auto asr_alias =
        HttpGet(server.port(), "/v1/realtime?model=asr-alias");
    // Without the upgrade headers the handshake cannot complete, but the
    // model check must not reject the configured alias up front.
    assert(asr_alias.find("invalid_model") == std::string::npos);
    assert(HttpGet(server.port(), "/v1/realtime?model=asr-typo")
               .find("invalid_model") != std::string::npos);
  }
  server.stop();
  worker.join();
  {
    bool mismatch_rate = false;
    auto tts_44k = std::make_shared<TtsService>(TtsServiceOptions{
        .validate_model = false,
        .sample_rate = 44100,
        .model_id = "fish-drama-3",
        .voices = {"default"},
        .runner = [&](const auto& request, const auto&, auto* result, auto*) {
          result->sample_rate = mismatch_rate ? 24000 : 44100;
          result->code_groups = 10;
          result->samples = {-.25F, 0.0F, .25F};
          return !request.on_audio || request.on_audio(result->samples);
        }});
    HttpServer server_44k("127.0.0.1", 0, nullptr, nullptr, tts_44k, nullptr,
                          {});
    assert(server_44k.start(&error));
    std::jthread worker_44k([&] { server_44k.run(); });
    {
      Client client(server_44k.port(), "/v1/audio/speech/stream");
      client.Text(
          R"({"type":"session.config","model":"fish-drama-3","voice":"default","response_format":"pcm","stream_audio":true})");
      client.Text(R"({"type":"input.text","text":"Forty-four one."})");
      client.Text(R"({"type":"input.done"})");
      auto start = client.Event("audio.start");
      assert(start.find("sample_rate")->as_double() == 44100);
      const auto expected = EncodePcm16(std::vector<float>{-.25F, 0.0F, .25F});
      assert(client.Receive() == std::make_pair(2U, expected));
      client.Event("audio.done");
      client.Event("session.done");

      mismatch_rate = true;
      client.Text(R"({"type":"input.text","text":"Mismatch."})");
      client.Text(R"({"type":"input.done"})");
      assert(client.Event("audio.start").find("sample_rate")->as_double() ==
             44100);
      assert(client.Receive() == std::make_pair(2U, expected));
      assert(client.Event("error").member_str("message") == "invalid_audio");
    }
    server_44k.stop();
    worker_44k.join();
  }
  {
    constexpr std::string_view kReferenceWav =
        "UklGRiwAAABXQVZFZm10IBAAAAABAAEAwF0AAIC7AAACABAAZGF0YQgAAAAAAOgD"
        "GPwAAA==";
    gufo::ext::SynthesisRequest last_request;
    auto hybrid_tts = std::make_shared<TtsService>(TtsServiceOptions{
        .validate_model = false,
        .variant = gufo::ext::TtsVariant::kBase,
        .conditioning =
            gufo::ext::TtsConditioningCapabilities{
                .unconditioned = true,
                .require_voice = false,
                .default_voice = "default",
                .reference_audio = gufo::ext::TtsConditioningMode::kOptional,
                .reference_presets = true,
                .instructions = gufo::ext::TtsConditioningMode::kOptional,
            },
        .model_id = "fish-drama-3",
        .voices = {"default", "voice-clone"},
        .voice_presets = {{"narrator",
                           {.reference_audio = {.sample_rate = 24000,
                                                .channels = 1,
                                                .samples = {0.2F, -0.2F}},
                            .reference_text = "Narrator ref."}}},
        .runner = [&](const auto& request, const auto&, auto* result, auto*) {
          last_request = request;
          last_request.on_audio = {};
          result->sample_rate = 24000;
          result->code_groups = 10;
          result->samples = {-.25F, 0.0F, .25F};
          return !request.on_audio || request.on_audio(result->samples);
        }});
    HttpServer hybrid_server("127.0.0.1", 0, nullptr, nullptr, hybrid_tts,
                             nullptr, {});
    assert(hybrid_server.start(&error));
    std::jthread hybrid_worker([&] { hybrid_server.run(); });
    {
      Client client(hybrid_server.port(), "/v1/audio/speech/stream");
      client.Text(
          R"({"type":"session.config","model":"fish-drama-3","task_type":"CustomVoice","response_format":"pcm","stream_audio":true})");
      client.Text(R"({"type":"input.text","text":"Default voice."})");
      client.Text(R"({"type":"input.done"})");
      client.Event("audio.start");
      client.Receive();
      client.Event("audio.done");
      client.Event("session.done");
      assert(last_request.speaker == "default" &&
             last_request.reference_audio.samples.empty());

      client.Text(
          std::string(
              R"({"type":"session.config","model":"fish-drama-3","task_type":"Base","voice":"voice-clone","ref_audio":")") +
          std::string(kReferenceWav) +
          R"(","ref_text":"Clone ref.","response_format":"pcm","stream_audio":true})");
      client.Text(R"({"type":"input.text","text":"Cloned voice."})");
      client.Text(R"({"type":"input.done"})");
      client.Event("audio.start");
      client.Receive();
      client.Event("audio.done");
      client.Event("session.done");
      assert(last_request.speaker == "voice-clone" &&
             last_request.reference_audio.samples.size() == 4 &&
             last_request.reference_text == "Clone ref.");

      client.Text(
          R"({"type":"session.config","model":"fish-drama-3","voice":"narrator","response_format":"pcm","stream_audio":true})");
      client.Text(R"({"type":"input.text","text":"Preset voice."})");
      client.Text(R"({"type":"input.done"})");
      client.Event("audio.start");
      client.Receive();
      client.Event("audio.done");
      client.Event("session.done");
      assert(last_request.speaker == "narrator" &&
             last_request.reference_audio.samples.size() == 2 &&
             last_request.reference_text == "Narrator ref.");

      client.Text(
          std::string(
              R"({"type":"session.config","model":"fish-drama-3","voice":"narrator","ref_audio":")") +
          std::string(kReferenceWav) + R"(","ref_text":"conflict"})");
      assert(client.Event("error").member_str("message").find(
                 "unsupported_field") != std::string::npos);

      client.Text(
          R"({"type":"session.config","model":"fish-drama-3","task_type":"VoiceDesign","instructions":"warm"})");
      assert(client.Event("error").member_str("message").find("task_type") !=
             std::string::npos);
    }
    hybrid_server.stop();
    hybrid_worker.join();
  }
  RealtimeStreams();
  OpenWebUiVoiceCall();
}
