#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <ranges>
#include <string>
#include <utility>

#include "src/cli/serve/audio_tts_api.hpp"
#include "src/cli/serve/tts_service.hpp"

namespace {

namespace qwen3_tts = strix::models::qwen3_tts;
using strix::server::HandleAudioTtsApiRequest;
using strix::server::HttpRequest;
using strix::server::HttpResponse;
using strix::server::TtsService;
using strix::server::TtsServiceOptions;

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
           qwen3_tts::SynthesisResult* result, std::string*) {
    ++calls;
    latest = request;
    result->sample_rate = 24000;
    result->code_groups = 16;
    result->codes.resize(32);
    result->samples = {-1.0F, -0.5F, 0.0F, 0.5F, 1.0F};
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
      .variant = qwen3_tts::ModelVariant::kVoiceDesign,
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
      .variant = qwen3_tts::ModelVariant::kBase,
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
      R"({"model":"qwen3-tts","input":"The boy who lived.","voice":"vivian","language":"english","instruct":"calm","seed":7,"max_new_tokens":12,"greedy":true,"response_format":"wav","speed":1.0})");
  Check(response.status == 200, "valid request succeeds");
  Check(response.body.size() == 54 && response.body.substr(0, 4) == "RIFF" &&
            response.body.substr(8, 8) == "WAVEfmt ",
        "response is a mono PCM16 WAV");
  Check(HasHeader(response, "Content-Type", "audio/wav"),
        "WAV content type is explicit");
  Check(HasHeader(response, "X-Strix-Codec-Steps", "2"),
        "codec step count is exposed");
  Check(runner.calls == 1 && runner.latest.text == "The boy who lived." &&
            runner.latest.speaker == "vivian" &&
            runner.latest.language == "english" &&
            runner.latest.instruct == "calm" && runner.latest.seed == 7 &&
            runner.latest.max_new_tokens == 12 && runner.latest.greedy,
        "API fields reach the model contract without drift");

  const HttpResponse sampled = Send(
      service, "POST", "/v1/audio/speech",
      R"({"model":"qwen3-tts","input":"The boy who lived.","voice":"vivian"})");
  Check(sampled.status == 200 && runner.calls == 2 && !runner.latest.greedy,
        "checkpoint-quality sampling is the API default");
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
  Check(
      Send(service, "POST", "/v1/audio/speech",
           R"({"model":"qwen3-tts","input":"x","voice":"vivian","typo":true})")
              .status == 400,
      "unknown fields fail closed");
  Check(runner.calls == 0, "invalid requests never invoke inference");
}

void TestVoiceDesignContract() {
  FakeRunner runner;
  TtsService service = MakeVoiceDesignService(&runner);
  Check(service.ready() &&
            service.model_id() == "qwen3-tts-12hz-1.7b-voice-design",
        "VoiceDesign service derives its model identity");

  const HttpResponse response = Send(
      service, "POST", "/v1/audio/speech",
      R"({"model":"qwen3-tts","input":"A lantern in the rain.","language":"english","instruct":"A calm, warm adult voice."})");
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
          R"({"model":"qwen3-tts","input":"x","voice":"vivian","instruct":"calm"})")
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

}  // namespace

int main() {
  TestSpeech();
  TestValidationAndVoices();
  TestVoiceDesignContract();
  TestBaseContract();
  std::cout << "PASS qwen3_tts_audio_api_test\n";
  return 0;
}
