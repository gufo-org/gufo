#ifndef GUFO_SERVER_TTS_SERVICE_HPP_
#define GUFO_SERVER_TTS_SERVICE_HPP_

#include <cstddef>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "src/models/qwen3_tts/audio.hpp"
#include "src/models/qwen3_tts/config.hpp"
#include "src/models/qwen3_tts/synthesis.hpp"

namespace gufo::server {

/// An operator-registered named voice: reference audio the server holds on
/// behalf of clients, so a Base checkpoint can be driven by voice name alone
/// instead of a base64 WAV on every request.
struct TtsVoicePreset {
  models::qwen3_tts::AudioBuffer reference_audio;
  std::string reference_text;
  bool speaker_embedding_only{false};
  /// Language this voice speaks. Supplies the request default when set, so a
  /// caller naming the voice does not have to repeat it; an explicit request
  /// `language` still wins. Empty means the global default applies.
  std::string language;
};

struct TtsServiceOptions {
  using Runner =
      std::function<bool(const models::qwen3_tts::SynthesisRequest&,
                         const models::qwen3_tts::CancellationCheck&,
                         models::qwen3_tts::SynthesisResult*, std::string*)>;

  std::filesystem::path model_root;
  std::size_t native_context_tokens{4096};
  bool validate_model{true};
  models::qwen3_tts::ModelVariant variant{
      models::qwen3_tts::ModelVariant::kCustomVoice};
  std::string model_id;
  std::vector<std::string> voices;
  std::map<std::string, TtsVoicePreset> voice_presets;
  Runner runner;
};

/// Synchronous, bounded native HIP TTS service (supports Qwen3-TTS and Fish
/// Drama-3).
class TtsService {
public:
  explicit TtsService(TtsServiceOptions options);
  ~TtsService();

  TtsService(const TtsService&) = delete;
  TtsService& operator=(const TtsService&) = delete;
  TtsService(TtsService&&) = delete;
  TtsService& operator=(TtsService&&) = delete;

  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] std::string initialization_error() const;
  [[nodiscard]] std::string model_id() const;
  [[nodiscard]] std::string backend_name() const;
  [[nodiscard]] std::vector<std::string> voices() const;
  /// Registered preset for `name`, or nullptr when `name` is not a preset.
  [[nodiscard]] const TtsVoicePreset* voice_preset(
      const std::string& name) const;
  [[nodiscard]] models::qwen3_tts::ModelVariant variant() const noexcept;
  [[nodiscard]] bool is_fish_drama3() const noexcept;

  [[nodiscard]] bool Synthesize(
      const models::qwen3_tts::SynthesisRequest& request,
      const models::qwen3_tts::CancellationCheck& is_cancelled,
      models::qwen3_tts::SynthesisResult* result, std::string* error = nullptr);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_TTS_SERVICE_HPP_
