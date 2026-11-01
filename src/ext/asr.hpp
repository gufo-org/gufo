#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::ext {

struct TranscriptionTimings {
  double decode_audio_ms{0.0};
  double feature_extraction_ms{0.0};
  double audio_encoder_ms{0.0};
  double text_decoder_ms{0.0};
  double total_ms{0.0};
};

struct TranscriptionRequest {
  std::span<const std::byte> wav;
  std::string context;
  std::optional<std::string> language;
  std::size_t max_new_tokens{256};
  /// Alternative to WAV for live sessions: normalized mono audio at 16 kHz.
  /// Exactly one audio source must be supplied.
  std::span<const float> pcm16k;
  /// Cumulative UTF-8 transcript, called during generation. Returning false
  /// cancels the operation; the view is valid only during the callback.
  std::function<bool(std::string_view)> on_text;
};

struct TranscriptionResult {
  std::string language;
  std::string text;
  std::string decoded;
  std::vector<std::uint32_t> generated_ids;
  std::uint32_t sample_rate{16000};
  std::size_t audio_samples{0};
  std::size_t mel_frames{0};
  std::size_t audio_tokens{0};
  std::size_t prompt_tokens{0};
  std::size_t chunks{0};
  TranscriptionTimings timings;
};

using CancellationCheck = std::function<bool()>;

/// Abstract capability runner for speech recognition models.
class AsrRunner {
public:
  virtual ~AsrRunner() = default;

  [[nodiscard]] virtual bool Transcribe(const TranscriptionRequest& request,
                                        const CancellationCheck& is_cancelled,
                                        TranscriptionResult* result,
                                        std::string* error = nullptr) = 0;
};

using AsrProbe = bool (*)(std::string_view model_root);
using AsrArtifactResolver = bool (*)(std::string_view model_root,
                                     std::string* error);
using AsrRunnerFactory = std::unique_ptr<AsrRunner> (*)(
    const std::string& model_root, std::size_t maximum_context_tokens,
    std::string* error);

struct AsrProvider {
  std::string id;
  AsrProbe probe = nullptr;
  AsrArtifactResolver resolve = nullptr;
  AsrRunnerFactory create = nullptr;
};

void RegisterAsrProvider(AsrProvider provider);
const AsrProvider* FindAsrProvider(std::string_view id);
const AsrProvider* ProbeAsrProvider(std::string_view model_root);

}  // namespace gufo::ext
