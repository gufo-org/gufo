#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "src/models/qwen3_asr/audio.hpp"
#include "src/models/qwen3_asr/hip/transcription_runtime.hpp"

namespace {

namespace qwen3_asr = gufo::models::qwen3_asr;
namespace qwen3_asr_hip = gufo::models::qwen3_asr::hip;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL qwen3_asr_transcription_hip_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

std::vector<std::byte> ReadBytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    Fail("cannot open " + path.string());
  }
  std::vector<char> chars((std::istreambuf_iterator<char>(input)),
                          std::istreambuf_iterator<char>());
  std::vector<std::byte> result(chars.size());
  for (std::size_t index = 0; index < chars.size(); ++index) {
    result[index] =
        static_cast<std::byte>(static_cast<unsigned char>(chars[index]));
  }
  return result;
}

}  // namespace

int main() {
  const char* configured_model = std::getenv("QWEN3_ASR_MODEL_ROOT");
  const std::filesystem::path model_root =
      configured_model != nullptr ? configured_model
                                  : "/var/llms/huggingface/hub/"
                                    "models--Qwen--Qwen3-ASR-1.7B/snapshots/"
                                    "7278e1e70fe206f11671096ffdd38061171dd6e5";
  const char* configured_wav = std::getenv("QWEN3_ASR_TEST_WAV");
  const std::filesystem::path wav_path =
      configured_wav != nullptr ? configured_wav : "/tmp/qwen3-asr-en.wav";
  if (!std::filesystem::is_regular_file(model_root / "config.json") ||
      !std::filesystem::is_regular_file(wav_path)) {
    std::cerr << "SKIP qwen3_asr_transcription_hip_test: "
                 "model or WAV unavailable\n";
    return 77;
  }

  const std::vector<std::byte> wav = ReadBytes(wav_path);
  std::string error;
  auto runtime = qwen3_asr_hip::TranscriptionHipRuntime::Create(
      model_root.string(), 512U, &error);
  Check(runtime != nullptr, error);
  qwen3_asr::TranscriptionResult result;
  Check(runtime->Transcribe(
            qwen3_asr::TranscriptionRequest{
                .wav = wav,
                .context = {},
                .language = std::nullopt,
                .max_new_tokens = 128,
            },
            {}, &result, &error),
        error);
  constexpr std::string_view kExpected =
      "Uh huh. Oh yeah, yeah. He wasn't even that big when I started "
      "listening to him, but and his solo music didn't do overly well, but he "
      "did very well when he started writing for other people.";
  Check(result.language == "English", "language must match official output");
  Check(result.text == kExpected, "transcript must match official output");
  Check(result.generated_ids.size() == 49U,
        "generated token count must match official output");
  Check(result.audio_tokens == 196U && result.prompt_tokens == 211U,
        "audio and prompt shape must match official output");
  qwen3_asr::AudioBuffer audio;
  Check(qwen3_asr::DecodeWav(wav, &audio, &error), error);
  const auto mono = qwen3_asr::ResampleMono16k(audio);
  std::vector<float> long_audio;
  for (int i = 0; i < 4; ++i) {
    long_audio.insert(long_audio.end(), mono.begin(), mono.end());
  }
  qwen3_asr::TranscriptionResult longer;
  std::string streamed;
  Check(runtime->Transcribe({.max_new_tokens = 128,
                             .pcm16k = long_audio,
                             .on_text =
                                 [&](std::string_view text) {
                                   Check(text.starts_with(streamed),
                                         "streamed transcript only appends");
                                   streamed = text;
                                   return true;
                                 }},
                            {}, &longer, &error),
        "long recording: " + error);
  Check(longer.chunks > 1 && streamed == longer.text &&
            longer.audio_samples == long_audio.size(),
        "long recording streams without exceeding the context");
  const auto chunks = qwen3_asr::SplitAudio(
      long_audio, qwen3_asr::AudioSamplesForTokenBudget(512 - 15 - 128));
  std::string expected_long;
  std::vector<std::uint32_t> expected_ids;
  for (const auto& chunk : chunks) {
    qwen3_asr::TranscriptionResult part;
    Check(runtime->Transcribe(
              {.max_new_tokens = 128,
               .pcm16k = std::span<const float>(long_audio)
                             .subspan(chunk.offset, chunk.samples)},
              {}, &part, &error),
          "independent chunk: " + error);
    expected_long += part.text;
    expected_ids.insert(expected_ids.end(), part.generated_ids.begin(),
                        part.generated_ids.end());
  }
  Check(longer.text == expected_long && longer.generated_ids == expected_ids,
        "long recording equals isolated chunk inference and merge");
  qwen3_asr::TranscriptionResult cancelled;
  int updates = 0;
  Check(!runtime->Transcribe(
            {.wav = wav,
             .max_new_tokens = 128,
             .on_text = [&](std::string_view) { return ++updates < 3; }},
            {}, &cancelled, &error) &&
            cancelled.generated_ids.empty() && updates == 3,
        "cancellation interrupts decode and returns no successful partial "
        "result");
  qwen3_asr::TranscriptionResult recovered;
  Check(runtime->Transcribe({.wav = wav, .max_new_tokens = 128}, {}, &recovered,
                            &error) &&
            recovered.generated_ids == result.generated_ids,
        "runtime remains reproducible after cancellation");
  const auto concurrent = [&](std::string context) {
    qwen3_asr::TranscriptionResult output;
    std::string failure;
    Check(runtime->Transcribe(
              {.wav = wav, .context = context, .max_new_tokens = 128}, {},
              &output, &failure),
          "concurrent transcription: " + failure);
    return output.generated_ids;
  };
  auto first = std::async(std::launch::async, concurrent, "");
  auto second = std::async(std::launch::async, concurrent, "Names: Gufo.");
  const auto first_ids = first.get();
  const auto second_ids = second.get();
  Check(
      first_ids == result.generated_ids &&
          second_ids == concurrent("Names: Gufo."),
      "concurrent frontend/chunk scheduling preserves independent transcripts");
  std::cout << "PASS qwen3_asr_transcription_hip_test"
            << " total_ms=" << result.timings.total_ms
            << " features_ms=" << result.timings.feature_extraction_ms
            << " audio_encoder_ms=" << result.timings.audio_encoder_ms
            << " text_decoder_ms=" << result.timings.text_decoder_ms
            << " transcript=\"" << result.text << "\"\n";
  return 0;
}
