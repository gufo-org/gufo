#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::ext {

enum class TtsVariant : std::uint8_t {
  kBase,
  kVoiceDesign,
  kCustomVoice,
  kUnsupported,
};

struct TtsSamplingOptions {
  bool sample{true};
  std::uint32_t seed{42};
  std::size_t top_k{50};
  float temperature{0.9F};
  float top_p{1.0F};
  bool predictor_sample{true};
  std::size_t predictor_top_k{50};
  float predictor_temperature{0.9F};
  float predictor_top_p{1.0F};
  float repetition_penalty{1.05F};
};

struct TtsAudioBuffer {
  std::uint32_t sample_rate{0};
  std::uint32_t channels{0};
  std::vector<float> samples;
};

struct SynthesisRequest {
  std::string text;
  std::string speaker{"vivian"};
  std::string language{"english"};
  std::string instruct;
  TtsAudioBuffer reference_audio;
  std::string reference_text;
  bool speaker_embedding_only{false};
  std::size_t max_new_tokens{3000};
  TtsSamplingOptions sampling;
  /// Called with newly generated mono audio at 24 kHz. Returning false cancels
  /// generation. Streaming results retain codes but do not buffer the waveform.
  std::function<bool(std::span<const float>)> on_audio{};
};

struct SynthesisResult {
  std::uint32_t sample_rate{0};
  std::uint32_t code_groups{0};
  std::vector<float> samples;
  // Frame-major [frames, code_groups].
  std::vector<std::int32_t> codes;
  std::size_t sample_count{0};
};

using CancellationCheck = std::function<bool()>;

/// Abstract capability runner for speech synthesis models.
class TtsRunner {
public:
  virtual ~TtsRunner() = default;

  [[nodiscard]] virtual bool Synthesize(const SynthesisRequest& request,
                                        const CancellationCheck& is_cancelled,
                                        SynthesisResult* result,
                                        std::string* error = nullptr) = 0;

  [[nodiscard]] virtual std::vector<std::string> Voices() const { return {}; }
  [[nodiscard]] virtual TtsVariant Variant() const {
    return TtsVariant::kCustomVoice;
  }
};

using TtsProbe = bool (*)(std::string_view model_root);
using TtsArtifactResolver = bool (*)(std::string_view model_root,
                                     std::string* error);
using TtsRunnerFactory = std::unique_ptr<TtsRunner> (*)(
    const std::string& model_root, std::size_t maximum_context_tokens,
    std::string* error);

struct TtsProvider {
  std::string id;
  TtsProbe probe = nullptr;
  TtsArtifactResolver resolve = nullptr;
  TtsRunnerFactory create = nullptr;
};

void RegisterTtsProvider(TtsProvider provider);
const TtsProvider* FindTtsProvider(std::string_view id);
const TtsProvider* ProbeTtsProvider(std::string_view model_root);

}  // namespace gufo::ext
