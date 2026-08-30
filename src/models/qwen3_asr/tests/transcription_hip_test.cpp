#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

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
  std::cout << "PASS qwen3_asr_transcription_hip_test"
            << " total_ms=" << result.timings.total_ms
            << " features_ms=" << result.timings.feature_extraction_ms
            << " audio_encoder_ms=" << result.timings.audio_encoder_ms
            << " text_decoder_ms=" << result.timings.text_decoder_ms
            << " transcript=\"" << result.text << "\"\n";
  return 0;
}
