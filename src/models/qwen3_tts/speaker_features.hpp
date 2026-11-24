#ifndef GUFO_MODELS_QWEN3_TTS_SPEAKER_FEATURES_HPP_
#define GUFO_MODELS_QWEN3_TTS_SPEAKER_FEATURES_HPP_

// The Base speaker encoder's input: 128-bin log-mel frames of the reference
// audio at 24 kHz (1024-point STFT, hop 256, reflect padding), shared by the
// GPU and NPU encoders.

#include <cstddef>
#include <vector>

#include "src/models/qwen3_tts/audio.hpp"

namespace gufo::models::qwen3_tts {

inline constexpr std::size_t kSpeakerMelBins = 128;

struct SpeakerFeatures {
  std::vector<float> values;  // [frames][128]
  std::size_t frames{0};
};

/// Throws std::invalid_argument for empty or too short audio.
[[nodiscard]] SpeakerFeatures ComputeSpeakerFeatures(const AudioBuffer& audio);

}  // namespace gufo::models::qwen3_tts

#endif  // GUFO_MODELS_QWEN3_TTS_SPEAKER_FEATURES_HPP_
