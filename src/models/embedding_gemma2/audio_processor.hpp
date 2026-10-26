#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_AUDIO_PROCESSOR_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_AUDIO_PROCESSOR_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::embedding_gemma2 {

inline constexpr std::size_t kAudioSampleRate = 16000;
inline constexpr std::size_t kMaxAudioSamples = 480000;
inline constexpr std::size_t kAudioMelBins = 128;

struct AudioFeatures {
  // Time-major [frames, 128]. Invalid frames are zero, including partial
  // frames.
  std::vector<float> values;
  std::vector<std::uint8_t> mask;
  [[nodiscard]] std::size_t soft_tokens() const noexcept;
};

struct SequenceAudio {
  std::size_t offset{0};
  AudioFeatures features;
};

[[nodiscard]] bool ValidateAudioProcessor(const std::string& model_dir,
                                          std::string* error = nullptr);

// The pinned Gemma4 frontend, rather than Whisper/ASR features: semicausal
// padding, 321-sample analysis windows, periodic Hann, magnitude (not power),
// HTK mel filters and natural log. Waveform is mono float32 at 16 kHz.
// Match the official extractor's per-batch waveform padding to 128 samples.
// A longer padding_samples reproduces multi-clip batches without changing
// which features or subsampled tokens are valid.
[[nodiscard]] AudioFeatures ProcessAudio(
    std::span<const float> waveform, std::size_t padding_samples = 0,
    const std::function<bool()>& is_cancelled = {});

}  // namespace gufo::models::embedding_gemma2
#endif
