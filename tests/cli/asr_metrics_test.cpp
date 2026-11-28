#include <atomic>
#include <cassert>
#include <chrono>
#include <memory>
#include <semaphore>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "src/cli/serve/asr_service.hpp"
#include "tests/cli/http_server_test_support.hpp"
#include "tests/cli/metrics_scrape.hpp"

namespace {

using gufo::server::AsrService;
using gufo::server::AsrServiceOptions;
using gufo::server::LiveTranscription;
using gufo::server::VoiceConversation;
using gufo::server::testing::Expect;
using gufo::server::testing::Sample;

enum class Mode { kOk, kError, kBlock, kThrow, kInvalid };

class FakeLive final : public LiveTranscription {
public:
  bool Push(std::span<const float>, const Text&, std::string*) override {
    return true;
  }
  bool Finish(const Text&, std::string*) override { return true; }
  const std::string& text() const override { return text_; }

private:
  std::string text_;
};

class FakeVoice final : public VoiceConversation {
public:
  bool Configure(const Settings&, std::string*) override { return true; }
  bool AppendAudio(std::span<const float>, std::string*) override {
    return true;
  }
  bool CommitAudio(std::size_t*, std::string*) override { return true; }
  bool ClearAudio(std::string*) override { return true; }
  bool audio_open() const override { return false; }
  bool AddUserText(std::string_view, std::string*) override { return true; }
  bool AddToolOutputs(std::span<const std::string>, std::string*) override {
    return true;
  }
  bool Respond(const Settings&, const Callbacks&, Reply*,
               std::string*) override {
    return true;
  }
  bool TruncateReply(std::size_t, std::string*) override { return true; }
};

class AsrServer {
public:
  AsrServer()
      : service(std::make_shared<AsrService>(AsrServiceOptions{
            .validate_model = false,
            .model_id = "asr-test",
            .runner =
                [this](const gufo::ext::TranscriptionRequest&,
                       const gufo::ext::CancellationCheck& is_cancelled,
                       gufo::ext::TranscriptionResult* result,
                       std::string* error) {
                  if (mode.load() == Mode::kBlock) {
                    ++entered;
                    release.acquire();
                  }
                  std::this_thread::sleep_for(delay.load());
                  if (mode.load() == Mode::kThrow)
                    throw std::runtime_error("injected throw");
                  if (mode.load() == Mode::kInvalid) {
                    result->invalid_audio = true;
                    *error = "undecodable";
                    return false;
                  }
                  if (mode.load() == Mode::kError || is_cancelled()) {
                    *error = "injected";
                    return false;
                  }
                  result->audio_samples = 32000;
                  result->sample_rate = 16000;
                  result->timings.total_ms = processing_ms.load();
                  return true;
                },
            .live = [this](std::uint32_t,
                           std::string*) -> std::unique_ptr<LiveTranscription> {
              if (refuse_sessions)
                return nullptr;
              return std::make_unique<FakeLive>();
            },
            .voice = [this](std::uint32_t, std::string*)
                -> std::unique_ptr<VoiceConversation> {
              if (refuse_sessions)
                return nullptr;
              return std::make_unique<FakeVoice>();
            },
        })),
        server("127.0.0.1", 0, nullptr, nullptr, nullptr, service) {
    std::string error;
    const bool started = server.start(&error);
    assert(started);
    (void)started;
    worker = std::jthread([this] { server.run(false); });
  }
  ~AsrServer() {
    server.stop();
    worker.join();
  }

  bool Transcribe(bool cancelled = false) {
    gufo::ext::TranscriptionResult result;
    std::string error;
    return service->Transcribe(
        {}, [cancelled] { return cancelled; }, &result, &error);
  }

  std::string Scrape(std::string_view path = "/metrics") {
    return gufo::server::testing::Scrape(server, path);
  }

  std::atomic<Mode> mode{Mode::kOk};
  std::atomic<std::chrono::milliseconds> delay{std::chrono::milliseconds(0)};
  std::atomic<double> processing_ms{0.0};
  std::atomic<int> entered{0};
  std::counting_semaphore<> release{0};
  std::atomic<bool> refuse_sessions{false};
  std::shared_ptr<AsrService> service;
  HttpServer server;
  std::jthread worker;
};

/// The service admits a second request while the first holds the device
/// gate, so the gauge reaches 2 with one runner call inside.
void WaitUntilInFlight(AsrServer& server, std::string_view count) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (server.entered.load() < 1 ||
         Sample(server.Scrape(), "gufo_asr_requests_in_flight") != count) {
    assert(std::chrono::steady_clock::now() < deadline);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

std::uint64_t Bucket(const std::string& body, std::string_view family,
                     std::string_view bound) {
  return std::stoull(Sample(body, std::string(family) + "_bucket{le=\"" +
                                      std::string(bound) + "\"}"));
}

}  // namespace

int main() {
  AsrServer server;

  const auto fresh = server.Scrape();
  Expect(fresh, "gufo_asr_requests_total{status=\"ok\"}", "0");
  Expect(fresh, "gufo_asr_requests_in_flight", "0");
  Expect(fresh, "gufo_asr_request_seconds_count", "0");
  Expect(fresh, "gufo_asr_live_sessions", "0");
  Expect(fresh, "gufo_asr_voice_sessions", "0");
  assert(fresh.find("llamacpp:") == std::string::npos);
  assert(fresh.find("gufo_embedding_") == std::string::npos);
  assert(fresh.find("gufo_device_lost_total") == std::string::npos);
  assert(server.Scrape("/v1/metrics") == fresh);

  assert(server.Transcribe());
  assert(server.Transcribe());
  auto body = server.Scrape();
  Expect(body, "gufo_asr_requests_total{status=\"ok\"}", "2");
  Expect(body, "gufo_asr_audio_seconds_total{kind=\"batch\"}", "4");
  Expect(body, "gufo_asr_request_seconds_count", "2");
  Expect(body, "gufo_asr_real_time_factor_count", "2");

  assert(!server.Transcribe(true));
  server.mode = Mode::kError;
  assert(!server.Transcribe());
  body = server.Scrape();
  Expect(body, "gufo_asr_requests_total{status=\"cancelled\"}", "1");
  Expect(body, "gufo_asr_requests_total{status=\"error\"}", "1");
  Expect(body, "gufo_asr_requests_total{status=\"ok\"}", "2");
  Expect(body, "gufo_asr_audio_seconds_total{kind=\"batch\"}", "4");
  Expect(body, "gufo_asr_request_seconds_count", "4");
  Expect(body, "gufo_asr_real_time_factor_count", "2");

  server.mode = Mode::kInvalid;
  assert(!server.Transcribe());
  server.mode = Mode::kThrow;
  bool threw = false;
  try {
    (void)server.Transcribe();
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);
  body = server.Scrape();
  Expect(body, "gufo_asr_requests_total{status=\"invalid\"}", "1");
  Expect(body, "gufo_asr_requests_total{status=\"error\"}", "2");
  Expect(body, "gufo_asr_requests_in_flight", "0");
  Expect(body, "gufo_asr_request_seconds_count", "6");
  Expect(body, "gufo_asr_real_time_factor_count", "2");

  server.mode = Mode::kBlock;
  std::jthread first([&] {
    const bool ok = server.Transcribe();
    assert(ok);
    (void)ok;
  });
  std::jthread second([&] {
    const bool ok = server.Transcribe();
    assert(ok);
    (void)ok;
  });
  WaitUntilInFlight(server, "2");
  server.mode = Mode::kOk;
  server.release.release(2);
  first.join();
  second.join();
  body = server.Scrape();
  Expect(body, "gufo_asr_requests_in_flight", "0");
  Expect(body, "gufo_asr_requests_total{status=\"ok\"}", "4");

  const auto seconds_before = Bucket(body, "gufo_asr_request_seconds", "0.1");
  const auto factor_before = Bucket(body, "gufo_asr_real_time_factor", "0.2");
  const auto factor_half = Bucket(body, "gufo_asr_real_time_factor", "0.5");
  const auto seconds_all = Bucket(body, "gufo_asr_request_seconds", "+Inf");
  server.delay = std::chrono::milliseconds(120);
  server.processing_ms = 500.0;
  assert(server.Transcribe());
  server.delay = std::chrono::milliseconds(0);
  body = server.Scrape();
  assert(Bucket(body, "gufo_asr_request_seconds", "0.1") == seconds_before);
  assert(Bucket(body, "gufo_asr_request_seconds", "+Inf") == seconds_all + 1);
  assert(Bucket(body, "gufo_asr_real_time_factor", "0.2") == factor_before);
  assert(Bucket(body, "gufo_asr_real_time_factor", "0.5") == factor_half + 1);

  std::string error;
  auto live = server.service->CreateLiveTranscription(16000, &error);
  auto voice = server.service->CreateVoiceConversation(24000, &error);
  auto second_voice = server.service->CreateVoiceConversation(24000, &error);
  assert(live && voice && second_voice);
  const std::vector<float> samples(8000);
  assert(live->Push(samples, {}, &error));
  assert(live->Push(samples, {}, &error));
  body = server.Scrape();
  Expect(body, "gufo_asr_live_sessions", "1");
  Expect(body, "gufo_asr_voice_sessions", "2");
  Expect(body, "gufo_asr_sessions_total{kind=\"live\"}", "1");
  Expect(body, "gufo_asr_sessions_total{kind=\"voice\"}", "2");
  Expect(body, "gufo_asr_audio_seconds_total{kind=\"live\"}", "1");

  live.reset();
  voice.reset();
  body = server.Scrape();
  Expect(body, "gufo_asr_live_sessions", "0");
  Expect(body, "gufo_asr_voice_sessions", "1");
  Expect(body, "gufo_asr_audio_seconds_total{kind=\"live\"}", "1");
  second_voice.reset();
  Expect(server.Scrape(), "gufo_asr_voice_sessions", "0");

  server.refuse_sessions = true;
  assert(!server.service->CreateLiveTranscription(16000, &error));
  assert(!server.service->CreateVoiceConversation(24000, &error));
  body = server.Scrape();
  Expect(body, "gufo_asr_sessions_total{kind=\"live\"}", "1");
  Expect(body, "gufo_asr_sessions_total{kind=\"voice\"}", "2");
  Expect(body, "gufo_asr_live_sessions", "0");
  assert(body.find("llamacpp:") == std::string::npos);
  assert(body.find("gufo_embedding_") == std::string::npos);

  auto orphaned = std::make_unique<AsrService>(AsrServiceOptions{
      .validate_model = false,
      .model_id = "asr-orphan",
      .runner = [](const gufo::ext::TranscriptionRequest&,
                   const gufo::ext::CancellationCheck&,
                   gufo::ext::TranscriptionResult*,
                   std::string*) { return true; },
      .live = [](std::uint32_t,
                 std::string*) -> std::unique_ptr<LiveTranscription> {
        return std::make_unique<FakeLive>();
      },
      .voice = [](std::uint32_t,
                  std::string*) -> std::unique_ptr<VoiceConversation> {
        return std::make_unique<FakeVoice>();
      },
  });
  auto orphan_live = orphaned->CreateLiveTranscription(16000, &error);
  auto orphan_voice = orphaned->CreateVoiceConversation(24000, &error);
  assert(orphan_live && orphan_voice);
  const auto open = orphaned->RenderMetrics();
  Expect(open, "gufo_asr_live_sessions", "1");
  Expect(open, "gufo_asr_voice_sessions", "1");
  orphaned.reset();
  assert(orphan_live->Push(samples, {}, &error));
  orphan_live.reset();
  orphan_voice.reset();
}
