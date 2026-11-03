#ifndef GUFO_SERVER_TTS_SERVICE_HPP_
#define GUFO_SERVER_TTS_SERVICE_HPP_

#include <cstddef>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gufo/ext/tts.hpp"
#include "src/cli/serve/audio_buffer.hpp"

namespace gufo::server {

/// Exact `error` text set by `TtsService::Synthesize()` when it rejects the
/// runner's output (null result, or `sample_rate` disagreeing with the rate
/// the service announced). HTTP handlers map this to `invalid_audio`.
inline constexpr std::string_view kInvalidSynthesisResultError =
    "TTS provider returned an invalid synthesis result";

/// An operator-registered named voice: reference audio the server holds on
/// behalf of clients, so a Base checkpoint can be driven by voice name alone
/// instead of a base64 WAV on every request.
struct TtsVoicePreset {
  AudioBuffer reference_audio;
  std::string reference_text;
  bool speaker_embedding_only{false};
  /// Language this voice speaks. Supplies the request default when set, so a
  /// caller naming the voice does not have to repeat it; an explicit request
  /// `language` still wins. Empty means the global default applies.
  std::string language;
};

struct TtsServiceOptions {
  using Runner = std::function<bool(const ext::SynthesisRequest&,
                                    const ext::CancellationCheck&,
                                    ext::SynthesisResult*, std::string*)>;

  std::filesystem::path model_root;
  /// Explicit model sidecar, such as an audio mmproj GGUF.
  std::filesystem::path projector_path;
  std::size_t native_context_tokens{4096};
  bool validate_model{true};
  std::uint32_t sample_rate{24000};
  ext::TtsVariant variant{ext::TtsVariant::kCustomVoice};
  std::optional<ext::TtsConditioningCapabilities> conditioning;
  std::string model_id;
  /// Additional IDs accepted from requests; `model_id` stays primary in
  /// listings and responses (#957).
  std::vector<std::string> model_aliases;
  std::vector<std::string> voices;
  std::map<std::string, TtsVoicePreset> voice_presets;
  Runner runner;
  /// The runner admits concurrent requests itself (for example by batching
  /// them); requests then skip the service's one-at-a-time gate.
  bool concurrent{false};
};

/// Synchronous, bounded native HIP Qwen3-TTS service.
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
  /// True for the primary ID, a built-in alias, or a configured alias.
  [[nodiscard]] bool serves_model(std::string_view name) const;
  [[nodiscard]] const std::vector<std::string>& model_aliases() const;
  [[nodiscard]] std::string backend_name() const;
  [[nodiscard]] std::vector<std::string> voices() const;
  /// Registered preset for `name`, or nullptr when `name` is not a preset.
  [[nodiscard]] const TtsVoicePreset* voice_preset(
      const std::string& name) const;
  [[nodiscard]] std::uint32_t sample_rate() const noexcept;
  [[nodiscard]] ext::TtsVariant variant() const noexcept;
  [[nodiscard]] const ext::TtsConditioningCapabilities& conditioning()
      const noexcept;

  [[nodiscard]] bool Synthesize(const ext::SynthesisRequest& request,
                                const ext::CancellationCheck& is_cancelled,
                                ext::SynthesisResult* result,
                                std::string* error = nullptr);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_TTS_SERVICE_HPP_
