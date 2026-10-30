#ifndef GUFO_SERVER_LFM3_AUDIO_RUNNER_HPP_
#define GUFO_SERVER_LFM3_AUDIO_RUNNER_HPP_
#include <filesystem>
#include <memory>
#include <string>

#include "src/cli/serve/realtime_audio.hpp"
#include "src/models/qwen3_asr/transcription.hpp"
#include "src/models/qwen3_tts/synthesis.hpp"
namespace gufo::cli {
// The LFM3-Audio mmproj for a backbone GGUF: explicit when given, otherwise
// the single mmproj*.gguf with LFM3-Audio towers beside the backbone. An
// empty result with an empty error means the model is not LFM3-Audio.
std::filesystem::path FindLfm3AudioMmproj(const std::filesystem::path& model,
                                          const std::filesystem::path& mmproj,
                                          std::string* error);
// Serializes requests over one loaded LFM3-Audio model, exposing it through
// the transcription and synthesis service contracts.
class Lfm3AudioRuntime {
public:
  static std::shared_ptr<Lfm3AudioRuntime> Create(
      const std::filesystem::path& model, const std::filesystem::path& mmproj,
      std::uint32_t context, std::string* error);
  ~Lfm3AudioRuntime();
  bool Transcribe(const models::qwen3_asr::TranscriptionRequest& request,
                  const models::qwen3_asr::CancellationCheck& cancelled,
                  models::qwen3_asr::TranscriptionResult* result,
                  std::string* error);
  // speaker is the voice (female or male); language a code or name.
  bool Synthesize(const models::qwen3_tts::SynthesisRequest& request,
                  const models::qwen3_tts::CancellationCheck& cancelled,
                  models::qwen3_tts::SynthesisResult* result,
                  std::string* error);
  // Realtime streams have their own backbone contexts; at most
  // kRealtimeSessions exist at once. rate is the input sample rate.
  static constexpr int kRealtimeSessions = 4;
  std::unique_ptr<server::LiveTranscription> CreateLiveTranscription(
      std::uint32_t rate, std::string* error);
  std::unique_ptr<server::VoiceConversation> CreateVoiceConversation(
      std::uint32_t rate, std::string* error);

private:
  Lfm3AudioRuntime();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gufo::cli
#endif
