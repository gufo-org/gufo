#ifndef GUFO_SERVER_ASR_SERVICE_HPP_
#define GUFO_SERVER_ASR_SERVICE_HPP_

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

#include "src/cli/serve/realtime_audio.hpp"
#include "src/models/qwen3_asr/transcription.hpp"

namespace gufo::server {

struct AsrServiceOptions {
  using Runner = std::function<bool(
      const models::qwen3_asr::TranscriptionRequest&,
      const models::qwen3_asr::CancellationCheck&,
      models::qwen3_asr::TranscriptionResult*, std::string*)>;

  std::filesystem::path model_root;
  std::size_t native_context_tokens{1024};
  bool validate_model{true};
  std::string model_id;
  Runner runner;
  /// Optional realtime streams of models that support them; rate is the
  /// input sample rate.
  std::function<std::unique_ptr<LiveTranscription>(std::uint32_t, std::string*)>
      live;
  std::function<std::unique_ptr<VoiceConversation>(std::uint32_t, std::string*)>
      voice;
};

/// Synchronous, bounded native HIP Qwen3-ASR-1.7B service.
class AsrService {
public:
  explicit AsrService(AsrServiceOptions options);
  ~AsrService();

  AsrService(const AsrService&) = delete;
  AsrService& operator=(const AsrService&) = delete;
  AsrService(AsrService&&) = delete;
  AsrService& operator=(AsrService&&) = delete;

  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] std::string initialization_error() const;
  [[nodiscard]] std::string model_id() const;
  [[nodiscard]] std::string backend_name() const;

  [[nodiscard]] bool Transcribe(
      const models::qwen3_asr::TranscriptionRequest& request,
      const models::qwen3_asr::CancellationCheck& is_cancelled,
      models::qwen3_asr::TranscriptionResult* result,
      std::string* error = nullptr);

  /// Live transcription while audio arrives, when the model supports it.
  [[nodiscard]] bool supports_live() const noexcept;
  [[nodiscard]] std::unique_ptr<LiveTranscription> CreateLiveTranscription(
      std::uint32_t rate, std::string* error);
  /// Spoken conversations, when the model hears and speaks.
  [[nodiscard]] bool supports_voice() const noexcept;
  [[nodiscard]] std::unique_ptr<VoiceConversation> CreateVoiceConversation(
      std::uint32_t rate, std::string* error);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_ASR_SERVICE_HPP_
