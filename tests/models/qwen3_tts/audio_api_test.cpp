#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <ranges>
#include <semaphore>
#include <string>
#include <utility>

#include "src/cli/serve/text_stream.hpp"
#include "src/models/qwen3_tts/synthesis.hpp"
#include "tests/support/model_api_test_support.hpp"

namespace {

namespace qwen3_tts = gufo::models::qwen3_tts;
using gufo::server::HandleAudioTtsApiRequest;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;
using gufo::server::TtsService;
using gufo::server::TtsServiceOptions;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL qwen3_tts_audio_api_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

class FakeRunner {
public:
  bool Run(const qwen3_tts::SynthesisRequest& request,
           const qwen3_tts::CancellationCheck&,
           qwen3_tts::SynthesisResult* result, std::string* error) {
    ++calls;
    latest = request;
    latest.on_audio = {};
    result->sample_rate = 24000;
    result->code_groups = 16;
    result->codes.resize(32);
    result->samples = {-1.0F, -0.5F, 0.0F, 0.5F, 1.0F};
    if (request.on_audio) {
      for (const auto sample : result->samples) {
        if (!request.on_audio(std::span<const float>(&sample, 1))) {
          if (error)
            *error = "cancelled";
          return false;
        }
      }
    }
    return true;
  }

  int calls{0};
  qwen3_tts::SynthesisRequest latest;
};

TtsService MakeService(FakeRunner* runner) {
  return TtsService(TtsServiceOptions{
      .model_root = {},
      .validate_model = false,
      .model_id = "qwen3-tts-12hz-1.7b-customvoice",
      .voices = {"vivian", "ryan"},
      .runner =
          [runner](const qwen3_tts::SynthesisRequest& request,
                   const qwen3_tts::CancellationCheck& cancelled,
                   qwen3_tts::SynthesisResult* result, std::string* error) {
            return runner->Run(request, cancelled, result, error);
          },
  });
}

TtsService MakeVoiceDesignService(FakeRunner* runner) {
  return TtsService(TtsServiceOptions{
      .model_root = {},
      .validate_model = false,
      .variant = gufo::ext::TtsVariant::kVoiceDesign,
      .model_id = {},
      .voices = {},
      .runner =
          [runner](const qwen3_tts::SynthesisRequest& request,
                   const qwen3_tts::CancellationCheck& cancelled,
                   qwen3_tts::SynthesisResult* result, std::string* error) {
            return runner->Run(request, cancelled, result, error);
          },
  });
}

TtsService MakeBaseService(FakeRunner* runner) {
  return TtsService(TtsServiceOptions{
      .model_root = {},
      .validate_model = false,
      .variant = gufo::ext::TtsVariant::kBase,
      .model_id = {},
      .voices = {},
      .runner =
          [runner](const qwen3_tts::SynthesisRequest& request,
                   const qwen3_tts::CancellationCheck& cancelled,
                   qwen3_tts::SynthesisResult* result, std::string* error) {
            return runner->Run(request, cancelled, result, error);
          },
  });
}

HttpResponse Send(TtsService& service, std::string method, std::string path,
                  std::string body = {}) {
  return HandleAudioTtsApiRequest(
      HttpRequest{
          .method = std::move(method),
          .path = std::move(path),
          .query = {},
          .body = std::move(body),
          .headers = {},
          .is_cancelled = {},
      },
      service);
}

bool HasHeader(const HttpResponse& response, std::string_view name,
               std::string_view value) {
  return std::ranges::any_of(response.headers, [&](const auto& header) {
    return header.first == name && header.second == value;
  });
}

void TestSpeech() {
  FakeRunner runner;
  TtsService service = MakeService(&runner);
  Check(service.ready(), "injected service is ready");

  const HttpResponse response = Send(
      service, "POST", "/v1/audio/speech",
      R"({"model":"qwen3-tts","input":"The boy who lived.","voice":"vivian","language":"english","instructions":"calm","seed":7,"max_new_tokens":12,"greedy":true,"response_format":"wav","speed":1.0})");
  Check(response.status == 200, "valid request succeeds");
  Check(
      response.log_details.find("codec_steps=2") != std::string::npos &&
          response.log_details.find("synthesis_ms=") != std::string::npos &&
          response.log_details.find("The boy who lived.") == std::string::npos,
      "speech diagnostics report work without input text");
  Check(response.body.size() == 54 && response.body.substr(0, 4) == "RIFF" &&
            response.body.substr(8, 8) == "WAVEfmt ",
        "response is a mono PCM16 WAV");
  Check(HasHeader(response, "Content-Type", "audio/wav"),
        "WAV content type is explicit");
  Check(HasHeader(response, "X-Gufo-Codec-Steps", "2"),
        "codec step count is exposed");
  Check(runner.calls == 1 && runner.latest.text == "The boy who lived." &&
            runner.latest.speaker == "vivian" &&
            runner.latest.language == "english" &&
            runner.latest.instruct == "calm" &&
            runner.latest.sampling.seed == 7 &&
            runner.latest.max_new_tokens == 12 &&
            !runner.latest.sampling.sample &&
            !runner.latest.sampling.predictor_sample,
        "API fields reach the model contract without drift");

  const HttpResponse sampled = Send(
      service, "POST", "/v1/audio/speech",
      R"({"model":"qwen3-tts","input":"The boy who lived.","voice":"vivian"})");
  Check(sampled.status == 200 && runner.calls == 2 &&
            runner.latest.sampling.sample &&
            runner.latest.sampling.predictor_sample,
        "checkpoint-quality sampling is the API default");
}

void TestSampling() {
  FakeRunner runner;
  TtsService service = MakeService(&runner);
  const auto response = Send(
      service, "POST", "/v1/audio/speech",
      R"({"model":"qwen3-tts","input":"x","voice":"vivian","temperature":0.5,"top_k":0,"top_p":0.8,"repetition_penalty":1.2,"subtalker_dosample":false,"subtalker_temperature":0.7,"subtalker_top_k":13,"subtalker_top_p":0.9})");
  const auto& sampling = runner.latest.sampling;
  Check(response.status == 200 && sampling.temperature == 0.5F &&
            sampling.top_k == 0 && sampling.top_p == 0.8F &&
            sampling.repetition_penalty == 1.2F && !sampling.predictor_sample &&
            sampling.predictor_temperature == 0.7F &&
            sampling.predictor_top_k == 13 && sampling.predictor_top_p == 0.9F,
        "main and predictor sampling controls reach their own distributions");
  for (const std::string field :
       {R"("top_p":0)", R"("top_p":1.1)", R"("temperature":0)",
        R"("temperature":1e100)", R"("top_k":-1)", R"("top_k":1.5)",
        R"("subtalker_top_p":2)", R"("subtalker_dosample":1)",
        R"("repetition_penalty":0)"}) {
    const auto invalid = Send(
        service, "POST", "/v1/audio/speech",
        R"({"model":"qwen3-tts","input":"x","voice":"vivian",)" + field + "}");
    Check(invalid.status == 400 &&
              invalid.body.find("invalid_sampling") != std::string::npos,
          "invalid sampling rejected: " + field);
  }
  Check(runner.calls == 1, "invalid sampling never invokes inference");

  std::mt19937 random(42);
  for (int iteration = 0; iteration < 100; ++iteration) {
    std::vector<qwen3_tts::TokenScore> logits{{0, 0}, {1, -1}, {2, -2}};
    Check(qwen3_tts::SampleCodec(logits, 0, 0.8F, 0.5F, &random) == 0,
          "temperature is applied before nucleus filtering");
  }
  bool saw_tied = false;
  for (int iteration = 0; iteration < 100; ++iteration) {
    std::vector<qwen3_tts::TokenScore> logits{{0, 0}, {1, 0}, {2, -10}};
    saw_tied |= qwen3_tts::SampleCodec(logits, 1, 1.0F, 1.0F, &random) == 1;
  }
  Check(saw_tied, "top-k preserves all logits tied at the threshold");
  std::vector<qwen3_tts::TokenScore> invalid{
      {0, std::numeric_limits<float>::quiet_NaN()}};
  bool rejected = false;
  try {
    (void)qwen3_tts::SampleCodec(invalid, 1, 1, 1, &random);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  Check(rejected, "nonfinite codec logits fail closed");
}

void TestStreaming() {
  FakeRunner runner;
  TtsService service = MakeService(&runner);
  auto response = Send(
      service, "POST", "/v1/audio/speech",
      R"({"model":"qwen3-tts","input":"x","voice":"vivian","response_format":"pcm"})");
  Check(response.status == 200 && response.streaming_body && runner.calls == 0,
        "PCM synthesis starts when the response is consumed");
  std::string pcm;
  response.streaming_body([&](std::string_view bytes) {
    pcm.append(bytes);
    return true;
  });
  Check(
      pcm == gufo::server::EncodePcm16(std::vector<float>{-1, -.5F, 0, .5F, 1}),
      "PCM chunks retain buffered sample encoding");
  auto sse = Send(
      service, "POST", "/v1/audio/speech",
      R"({"model":"qwen3-tts","input":"x","voice":"vivian","stream_format":"sse"})");
  std::string events;
  sse.streaming_body([&](std::string_view bytes) {
    events.append(bytes);
    return true;
  });
  Check(events.find("speech.audio.delta") != std::string::npos &&
            events.find("\"audio\":") != std::string::npos &&
            events.find("speech.audio.done") != std::string::npos,
        "SSE uses the OpenAI audio field and terminal event");
  std::size_t writes = 0;
  response.streaming_body([&](std::string_view) {
    ++writes;
    return false;
  });
  Check(writes == 1 && response.stream_log->error_code == "cancelled",
        "output disconnect cancels native synthesis immediately");
  for (std::string value : {R"("bogus")", "true"}) {
    Check(
        Send(
            service, "POST", "/v1/audio/speech",
            R"({"model":"qwen3-tts","input":"x","voice":"vivian","stream_format":)" +
                value + "}")
                .status == 400,
        "unsupported stream formats fail closed");
  }
  std::string decoded;
  Check(gufo::server::DecodeAudioBase64(gufo::server::EncodeAudioBase64(pcm),
                                        pcm.size(), &decoded) &&
            decoded == pcm,
        "audio base64 round trip");
  Check(!gufo::server::DecodeAudioBase64("AB==", 16, &decoded) &&
            !gufo::server::DecodeAudioBase64("YQ==AAAA", 16, &decoded) &&
            !gufo::server::DecodeAudioBase64("AAAA", 2, &decoded),
        "base64 rejects noncanonical padding and oversized audio");
  const std::string text = "Value 3.14 is fine. Next phrase! Then 中文。";
  std::string pending = text;
  const auto expected =
      gufo::server::TakeSpeechSegments(&pending, "sentence", true);
  for (std::size_t split = 0; split <= text.size(); ++split) {
    pending = text.substr(0, split);
    auto actual = gufo::server::TakeSpeechSegments(&pending, "sentence", false);
    pending += text.substr(split);
    auto tail = gufo::server::TakeSpeechSegments(&pending, "sentence", true);
    actual.insert(actual.end(), tail.begin(), tail.end());
    Check(actual == expected,
          "text splitting is independent of packet boundaries");
  }
}

void TestQueuedCancellation() {
  std::binary_semaphore entered{0}, release{0};
  std::atomic<int> calls{0};
  TtsService service(TtsServiceOptions{
      .validate_model = false,
      .runner = [&](const auto&, const auto&, auto* result, auto*) {
        ++calls;
        entered.release();
        release.acquire();
        result->sample_rate = 24000;
        return true;
      }});
  auto active = std::async(std::launch::async, [&] {
    qwen3_tts::SynthesisResult output;
    std::string error;
    return service.Synthesize({}, {}, &output, &error);
  });
  entered.acquire();
  std::atomic<bool> cancel{false};
  auto queued = std::async(std::launch::async, [&] {
    qwen3_tts::SynthesisResult output;
    output.samples = {1};
    std::string error;
    Check(!service.Synthesize(
              {}, [&] { return cancel.load(); }, &output, &error) &&
              output.samples.empty(),
          "queued cancellation clears the result");
  });
  cancel = true;
  Check(queued.wait_for(std::chrono::seconds(1)) == std::future_status::ready,
        "queued cancellation does not wait for active inference");
  queued.get();
  Check(calls == 1, "cancelled waiter never enters the mutable runtime");
  release.release();
  Check(active.get(),
        "queued cancellation does not invalidate the active request");
}

void TestValidationAndVoices() {
  FakeRunner runner;
  TtsService service = MakeService(&runner);

  const HttpResponse voices = Send(service, "GET", "/v1/audio/voices");
  Check(voices.status == 200 &&
            voices.body.find("\"id\":\"vivian\"") != std::string::npos &&
            voices.body.find("\"id\":\"ryan\"") != std::string::npos,
        "voice discovery lists configured CustomVoice speakers");

  Check(Send(service, "POST", "/v1/audio/speech",
             R"({"model":"qwen3-tts","input":"x","voice":"unknown"})")
                .status == 400,
        "unknown voice fails closed");
  Check(
      Send(
          service, "POST", "/v1/audio/speech",
          R"({"model":"qwen3-tts","input":"x","voice":"vivian","response_format":"mp3"})")
              .status == 400,
      "unsupported format fails closed");
  Check(Send(service, "POST", "/v1/audio/speech",
             R"({"model":"qwen3-tts","input":"x","voice":"vivian","seed":1.5})")
                .status == 400,
        "fractional seed fails closed");
  Check(runner.calls == 0, "invalid requests never invoke inference");
  Check(
      Send(service, "POST", "/v1/audio/speech",
           R"({"model":"qwen3-tts","input":"x","voice":"vivian","typo":true})")
                  .status == 200 &&
          runner.calls == 1,
      "unknown fields are ignored");
}

void TestVoiceDesignContract() {
  FakeRunner runner;
  TtsService service = MakeVoiceDesignService(&runner);
  Check(service.ready() &&
            service.model_id() == "qwen3-tts-12hz-1.7b-voice-design",
        "VoiceDesign service derives its model identity");

  const HttpResponse response = Send(
      service, "POST", "/v1/audio/speech",
      R"({"model":"qwen3-tts","input":"A lantern in the rain.","language":"english","instructions":"A calm, warm adult voice."})");
  Check(response.status == 200 && runner.calls == 1 &&
            runner.latest.speaker == "voice-design" &&
            runner.latest.instruct == "A calm, warm adult voice.",
        "VoiceDesign uses instructions without CustomVoice speaker semantics");

  Check(Send(service, "POST", "/v1/audio/speech",
             R"({"model":"qwen3-tts","input":"x"})")
                .status == 400,
        "VoiceDesign rejects a missing instruction");
  Check(
      Send(
          service, "POST", "/v1/audio/speech",
          R"({"model":"qwen3-tts","input":"x","voice":"vivian","instructions":"calm"})")
              .status == 400,
      "VoiceDesign rejects CustomVoice speaker names");
}

void TestBaseContract() {
  constexpr std::string_view kReferenceWav =
      "UklGRiwAAABXQVZFZm10IBAAAAABAAEAwF0AAIC7AAACABAAZGF0YQgAAAAAAOgD"
      "GPwAAA==";
  FakeRunner runner;
  TtsService service = MakeBaseService(&runner);
  Check(service.ready() && service.model_id() == "qwen3-tts-12hz-1.7b-base",
        "Base service derives its model identity");

  const HttpResponse response = Send(
      service, "POST", "/v1/audio/speech",
      std::string(
          R"({"model":"qwen3-tts","input":"Clone me.","reference_audio":")") +
          std::string(kReferenceWav) +
          R"(","reference_text":"Reference phrase.","language":"auto"})");
  Check(response.status == 200 && runner.calls == 1 &&
            runner.latest.speaker == "voice-clone" &&
            runner.latest.reference_audio.sample_rate == 24000 &&
            runner.latest.reference_audio.channels == 1 &&
            runner.latest.reference_audio.samples.size() == 4 &&
            runner.latest.reference_text == "Reference phrase." &&
            !runner.latest.speaker_embedding_only,
        "Base ICL reference audio reaches the synthesis contract");

  Check(Send(service, "POST", "/v1/audio/speech",
             R"({"model":"qwen3-tts","input":"x"})")
                .status == 400,
        "Base rejects a missing reference WAV");
  Check(Send(service, "POST", "/v1/audio/speech",
             std::string(
                 R"({"model":"qwen3-tts","input":"x","reference_audio":")") +
                 std::string(kReferenceWav) + R"("})")
                .status == 400,
        "Base ICL rejects missing reference text");
  Check(Send(service, "POST", "/v1/audio/speech",
             std::string(
                 R"({"model":"qwen3-tts","input":"x","reference_audio":")") +
                 std::string(kReferenceWav) +
                 R"(","voice_clone_mode":"speaker_embedding_only"})")
                    .status == 200 &&
            runner.latest.speaker_embedding_only,
        "Base speaker-embedding-only mode does not require reference text");
}

void TestInvalidWaveform() {
  for (const float sample : {std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()}) {
    TtsService service(TtsServiceOptions{
        .validate_model = false,
        .voices = {"vivian"},
        .runner =
            [sample](const auto& request, const auto&, auto* result, auto*) {
              result->sample_rate = 24000;
              result->samples = {0, sample};
              return !request.on_audio || request.on_audio(result->samples);
            },
    });
    const auto wav =
        Send(service, "POST", "/v1/audio/speech",
             R"({"model":"qwen3-tts","input":"x","voice":"vivian"})");
    Check(wav.status == 500 &&
              wav.body.find("invalid_audio") != std::string::npos,
          "nonfinite WAV samples fail before conversion");
    for (const auto format : {"audio", "sse"}) {
      auto response = Send(
          service, "POST", "/v1/audio/speech",
          std::string(
              R"({"model":"qwen3-tts","input":"x","voice":"vivian","response_format":"pcm","stream_format":")") +
              format + "\"}");
      std::string body;
      response.streaming_body([&](std::string_view bytes) {
        body.append(bytes);
        return true;
      });
      Check(response.stream_log->error_code == "invalid_audio",
            "nonfinite streaming samples report invalid audio");
      Check(body.find("speech.audio.delta") == std::string::npos &&
                body.find("speech.audio.done") == std::string::npos &&
                (format == std::string_view("sse")
                     ? body.find("event: error") != std::string::npos
                     : body.empty()),
            "invalid audio never becomes PCM or a successful terminal event");
    }
  }
}

void TestSampleRateAndDuration() {
  struct CustomRateRunner final : gufo::ext::TtsRunner {
    explicit CustomRateRunner(std::uint32_t rate) : rate_(rate) {}
    bool Synthesize(const gufo::ext::SynthesisRequest& request,
                    const gufo::ext::CancellationCheck&,
                    gufo::ext::SynthesisResult* result, std::string*) override {
      result->sample_rate = rate_;
      result->samples.assign(22050, 0.1F);
      return !request.on_audio || request.on_audio(result->samples);
    }
    std::vector<std::string> Voices() const override { return {"default"}; }
    std::uint32_t SampleRate() const override { return rate_; }
    std::string ModelId() const override { return "custom-rate-tts"; }
    std::uint32_t rate_;
  };
  static std::uint32_t provider_rate = 44100;
  gufo::ext::RegisterTtsProvider({
      .id = "test-custom-rate-provider",
      .probe =
          [](std::string_view root) {
            return root == "/virtual/custom-rate-tts";
          },
      .resolve = [](std::string_view, std::string*) { return true; },
      .create = [](const std::string&, std::size_t,
                   std::string*) -> std::unique_ptr<gufo::ext::TtsRunner> {
        return std::make_unique<CustomRateRunner>(provider_rate);
      },
  });
  {
    TtsService from_provider(TtsServiceOptions{
        .model_root = "/virtual/custom-rate-tts",
        .validate_model = true,
    });
    Check(from_provider.ready() && from_provider.sample_rate() == 44100 &&
              from_provider.model_id() == "custom-rate-tts",
          "provider TtsRunner::SampleRate() populates service sample_rate");
  }
  for (const std::uint32_t bad_rate : {0U, 192001U}) {
    provider_rate = bad_rate;
    TtsService invalid_provider(TtsServiceOptions{
        .model_root = "/virtual/custom-rate-tts",
        .validate_model = true,
    });
    Check(!invalid_provider.ready(),
          "invalid TtsRunner::SampleRate() rejected at startup");
  }

  for (const std::uint32_t bad_rate : {0U, 192001U}) {
    TtsService invalid(TtsServiceOptions{
        .validate_model = false,
        .sample_rate = bad_rate,
        .voices = {"vivian"},
        .runner = [](const auto&, const auto&, auto*, auto*) { return true; },
    });
    Check(!invalid.ready(), "invalid output sample rate rejected at startup");
    qwen3_tts::SynthesisResult out;
    std::string error;
    Check(!invalid.Synthesize({}, {}, &out, &error) && !error.empty(),
          "unready service rejects synthesis");
  }

  for (const auto& [rate, half_sec_samples] :
       {std::pair{24000U, std::size_t{12000}},
        std::pair{44100U, std::size_t{22050}}}) {
    TtsService service(TtsServiceOptions{
        .validate_model = false,
        .sample_rate = rate,
        .model_id = "test-tts",
        .voices = {"default"},
        .runner =
            [rate, half_sec_samples](const auto& request, const auto&,
                                     auto* result, auto*) {
              result->sample_rate = rate;
              result->code_groups = 10;
              result->codes.resize(100);
              result->samples.assign(half_sec_samples, 0.25F);
              return !request.on_audio || request.on_audio(result->samples);
            },
    });
    Check(service.ready() && service.sample_rate() == rate,
          "custom sample rate service is ready");

    const auto wav =
        Send(service, "POST", "/v1/audio/speech",
             R"({"model":"test-tts","input":"half second","voice":"default"})");
    Check(wav.status == 200 &&
              wav.log_details.find("audio_ms=500") != std::string::npos,
          "buffered WAV duration uses exact sample rate");
    std::uint32_t wav_rate = 0;
    std::memcpy(&wav_rate, wav.body.data() + 24, sizeof(wav_rate));
    Check(wav_rate == rate, "WAV header carries provider sample rate");

    for (const auto format : {"audio", "sse"}) {
      auto stream = Send(
          service, "POST", "/v1/audio/speech",
          std::string(
              R"({"model":"test-tts","input":"half second","voice":"default","response_format":"pcm","stream_format":")") +
              format + "\"}");
      std::string payload;
      stream.streaming_body([&](std::string_view bytes) {
        payload.append(bytes);
        return true;
      });
      Check(stream.stream_log->error_code.empty() &&
                stream.stream_log->details.find("audio_ms=500") !=
                    std::string::npos,
            "streaming duration uses exact provider sample rate");
    }
  }

  TtsService mismatched(TtsServiceOptions{
      .validate_model = false,
      .sample_rate = 44100,
      .model_id = "test-tts",
      .voices = {"default"},
      .runner =
          [](const auto& request, const auto&, auto* result, auto*) {
            result->sample_rate = 24000;
            result->samples = {0.0F, 0.25F};
            return !request.on_audio || request.on_audio(result->samples);
          },
  });
  const auto bad_wav =
      Send(mismatched, "POST", "/v1/audio/speech",
           R"({"model":"test-tts","input":"x","voice":"default"})");
  Check(bad_wav.status == 500 &&
            bad_wav.body.find("invalid_audio") != std::string::npos,
        "mismatched result sample rate fails closed for buffered WAV");
  for (const auto format : {"audio", "sse"}) {
    auto stream = Send(
        mismatched, "POST", "/v1/audio/speech",
        std::string(
            R"({"model":"test-tts","input":"x","voice":"default","response_format":"pcm","stream_format":")") +
            format + "\"}");
    std::string body;
    stream.streaming_body([&](std::string_view bytes) {
      body.append(bytes);
      return true;
    });
    Check(stream.stream_log->error_code == "invalid_audio",
          "mismatched result sample rate marks streaming log as invalid_audio");
    if (format == std::string_view("sse")) {
      Check(
          body.find("event: error") != std::string::npos &&
              body.find("speech.audio.done") == std::string::npos,
          "SSE reports an error instead of speech.audio.done on rate mismatch");
    }
  }
}

void TestConditioningCapabilities() {
  constexpr std::string_view kReferenceWav =
      "UklGRiwAAABXQVZFZm10IBAAAAABAAEAwF0AAIC7AAACABAAZGF0YQgAAAAAAOgD"
      "GPwAAA==";

  for (const auto variant : {gufo::ext::TtsVariant::kCustomVoice,
                             gufo::ext::TtsVariant::kVoiceDesign}) {
    FakeRunner runner;
    TtsService rejected(TtsServiceOptions{
        .validate_model = false,
        .variant = variant,
        .voices = {"vivian"},
        .voice_presets = {{"preset",
                           {.reference_audio = {.sample_rate = 24000,
                                                .channels = 1,
                                                .samples = {0.1F}},
                            .reference_text = "hi"}}},
        .runner = [&](const auto&, const auto&, auto*, auto*) { return true; },
    });
    Check(!rejected.ready(),
          "startup rejects voice presets when reference_presets is false");
  }

  for (const auto& bad_cond : {
           gufo::ext::TtsConditioningCapabilities{
               .unconditioned = false,
               .require_voice = false,
               .default_voice = {},
               .reference_audio = gufo::ext::TtsConditioningMode::kUnsupported,
               .reference_presets = false,
               .instructions = gufo::ext::TtsConditioningMode::kOptional,
           },
           gufo::ext::TtsConditioningCapabilities{
               .unconditioned = true,
               .require_voice = false,
               .default_voice = "default",
               .reference_audio = gufo::ext::TtsConditioningMode::kRequired,
               .reference_presets = false,
               .instructions = gufo::ext::TtsConditioningMode::kOptional,
           },
       }) {
    TtsService invalid(TtsServiceOptions{
        .validate_model = false,
        .conditioning = bad_cond,
        .voices = {"default"},
        .runner = [](const auto&, const auto&, auto*, auto*) { return true; },
    });
    Check(!invalid.ready(),
          "contradictory conditioning capabilities rejected at startup");
  }

  FakeRunner runner;
  TtsService hybrid(TtsServiceOptions{
      .validate_model = false,
      .variant = gufo::ext::TtsVariant::kBase,
      .conditioning =
          gufo::ext::TtsConditioningCapabilities{
              .unconditioned = true,
              .require_voice = false,
              .default_voice = "default",
              .reference_audio = gufo::ext::TtsConditioningMode::kOptional,
              .reference_presets = true,
              .instructions = gufo::ext::TtsConditioningMode::kUnsupported,
          },
      .model_id = "fish-drama-3",
      .voices = {"default", "voice-clone"},
      .voice_presets = {{"aaa_preset",
                         {.reference_audio = {.sample_rate = 24000,
                                              .channels = 1,
                                              .samples = {0.25F, -0.25F}},
                          .reference_text = "Preset transcript.",
                          .speaker_embedding_only = false,
                          .language = "italian"}}},
      .runner =
          [&runner](const qwen3_tts::SynthesisRequest& request,
                    const qwen3_tts::CancellationCheck& cancelled,
                    qwen3_tts::SynthesisResult* result, std::string* error) {
            return runner.Run(request, cancelled, result, error);
          },
  });
  Check(hybrid.ready(), "hybrid conditioning service with presets is ready");

  const auto unconditioned =
      Send(hybrid, "POST", "/v1/audio/speech",
           R"({"model":"fish-drama-3","input":"Unconditioned hello."})");
  Check(
      unconditioned.status == 200 && runner.calls == 1 &&
          runner.latest.speaker == "default" &&
          runner.latest.reference_audio.samples.empty() &&
          runner.latest.reference_text.empty(),
      "hybrid runner synthesizes without reference audio using default_voice");

  const auto cloned = Send(
      hybrid, "POST", "/v1/audio/speech",
      std::string(
          R"({"model":"fish-drama-3","input":"Cloned hello.","voice":"voice-clone","reference_audio":")") +
          std::string(kReferenceWav) +
          R"(","reference_text":"Reference phrase."})");
  Check(cloned.status == 200 && runner.calls == 2 &&
            runner.latest.speaker == "voice-clone" &&
            runner.latest.reference_audio.samples.size() == 4 &&
            runner.latest.reference_text == "Reference phrase.",
        "hybrid runner accepts explicit per-request reference cloning");

  const auto preset = Send(
      hybrid, "POST", "/v1/audio/speech",
      R"({"model":"fish-drama-3","input":"Preset hello.","voice":"aaa_preset"})");
  Check(
      preset.status == 200 && runner.calls == 3 &&
          runner.latest.speaker == "aaa_preset" &&
          runner.latest.reference_audio.samples.size() == 2 &&
          runner.latest.reference_text == "Preset transcript." &&
          runner.latest.language == "italian",
      "hybrid runner applies registered voice preset and its default language");

  Check(
      Send(
          hybrid, "POST", "/v1/audio/speech",
          std::string(
              R"({"model":"fish-drama-3","input":"x","voice":"aaa_preset","reference_audio":")") +
              std::string(kReferenceWav) + R"(","reference_text":"conflict"})")
              .status == 400,
      "preset rejects conflicting per-request reference audio");
  Check(
      Send(hybrid, "POST", "/v1/audio/speech",
           R"({"model":"fish-drama-3","input":"x","reference_text":"orphan"})")
              .status == 400,
      "orphan reference_text without reference_audio fails closed");
  Check(Send(hybrid, "POST", "/v1/audio/speech",
             std::string(
                 R"({"model":"fish-drama-3","input":"x","reference_audio":")") +
                 std::string(kReferenceWav) + R"("})")
                .status == 400,
        "ICL reference_audio without reference_text fails closed");
  Check(
      Send(
          hybrid, "POST", "/v1/audio/speech",
          R"({"model":"fish-drama-3","input":"x","reference_audio":"not-wav","reference_text":"hi"})")
              .status == 400,
      "malformed reference_audio fails closed");
  Check(Send(hybrid, "POST", "/v1/audio/speech",
             R"({"model":"fish-drama-3","input":"x","instructions":"whisper"})")
                .status == 400,
        "unsupported instructions field fails closed");
}

}  // namespace

int main() {
  TestSpeech();
  TestSampling();
  TestStreaming();
  TestQueuedCancellation();
  TestValidationAndVoices();
  TestVoiceDesignContract();
  TestBaseContract();
  TestInvalidWaveform();
  TestSampleRateAndDuration();
  TestConditioningCapabilities();
  std::cout << "PASS qwen3_tts_audio_api_test\n";
  return 0;
}
