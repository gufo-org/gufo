#ifndef GUFO_MODELS_QWEN3_ASR_AUDIO_HPP_
#define GUFO_MODELS_QWEN3_ASR_AUDIO_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::qwen3_asr {

inline constexpr std::uint32_t kAudioSampleRate = 16000;
inline constexpr std::size_t kAudioMelBins = 128;
inline constexpr std::size_t kAudioFftSize = 400;
inline constexpr std::size_t kAudioHopLength = 160;

struct AudioBuffer {
  std::uint32_t sample_rate{0};
  std::uint32_t channels{0};
  // Interleaved float samples.
  std::vector<float> samples;
};

struct LogMelFeatures {
  // Mel-major [kAudioMelBins, frames], matching WhisperFeatureExtractor.
  std::vector<float> values;
  std::size_t frames{0};
};

[[nodiscard]] bool DecodeWav(std::span<const std::byte> bytes,
                             AudioBuffer* output, std::string* error = nullptr);

/// Mixes to mono and resamples to 16 kHz. Input already at 16 kHz is copied
/// exactly, preserving the official-reference waveform.
[[nodiscard]] std::vector<float> ResampleMono16k(const AudioBuffer& audio);

/// Exact model frontend: centered periodic-Hann 400-point STFT, 160-sample
/// hop, squared magnitude, 128-bin Slaney mel bank, and Whisper log scaling.
[[nodiscard]] LogMelFeatures ComputeLogMelFeatures(
    std::span<const float> waveform);

/// Returns the number of audio embeddings produced by the fixed 1.7B
/// convolutional frontend for a log-mel frame count.
[[nodiscard]] std::size_t AudioEmbeddingTokenCount(std::size_t feature_frames);

struct AudioChunk {
  std::size_t offset;
  std::size_t samples;
};

/// Maximum 16-kHz samples fitting the audio-token budget, capped at the
/// upstream 20-minute inference window.
[[nodiscard]] std::size_t AudioSamplesForTokenBudget(std::size_t tokens);

/// Low-energy cuts following the upstream 100-ms absolute-amplitude window,
/// with the boundary search constrained by the runtime's hard capacity.
/// Chunks partition the original samples exactly, without gaps or overlap.
[[nodiscard]] std::vector<AudioChunk> SplitAudio(
    std::span<const float> waveform, std::size_t maximum_samples);

}  // namespace gufo::models::qwen3_asr

#endif  // GUFO_MODELS_QWEN3_ASR_AUDIO_HPP_
