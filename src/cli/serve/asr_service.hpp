#ifndef GUFO_SERVER_ASR_SERVICE_HPP_
#define GUFO_SERVER_ASR_SERVICE_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "gufo/ext/asr.hpp"
#include "src/cli/serve/realtime_audio.hpp"

namespace gufo::server {

struct AsrServiceOptions {
  using Runner = std::function<bool(const ext::TranscriptionRequest&,
                                    const ext::CancellationCheck&,
                                    ext::TranscriptionResult*, std::string*)>;

  std::filesystem::path model_root;
  /// Explicit model sidecar, such as an audio mmproj GGUF.
  std::filesystem::path projector_path;
  std::size_t native_context_tokens{1024};
  bool validate_model{true};
  std::string model_id;
  /// Additional IDs accepted from requests; `model_id` stays primary in
  /// listings and responses (#957).
  std::vector<std::string> model_aliases;
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
  /// True for the primary ID, a built-in alias, or a configured alias.
  [[nodiscard]] bool serves_model(std::string_view name) const;
  [[nodiscard]] const std::vector<std::string>& model_aliases() const;
  [[nodiscard]] std::string backend_name() const;

  [[nodiscard]] bool Transcribe(const ext::TranscriptionRequest& request,
                                const ext::CancellationCheck& is_cancelled,
                                ext::TranscriptionResult* result,
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
