#ifndef GUFO_MODELS_QWEN3_ASR_TRANSCRIPTION_HPP_
#define GUFO_MODELS_QWEN3_ASR_TRANSCRIPTION_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::qwen3_asr {

struct TranscriptionRequest {
  std::span<const std::byte> wav;
  std::string context;
  std::optional<std::string> language;
  std::size_t max_new_tokens{256};
};

struct TranscriptionTimings {
  double decode_audio_ms{0.0};
  double feature_extraction_ms{0.0};
  double audio_encoder_ms{0.0};
  double text_decoder_ms{0.0};
  double total_ms{0.0};
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
  TranscriptionTimings timings;
};

using CancellationCheck = std::function<bool()>;

}  // namespace gufo::models::qwen3_asr

#endif  // GUFO_MODELS_QWEN3_ASR_TRANSCRIPTION_HPP_
