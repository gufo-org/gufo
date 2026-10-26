#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_AUDIO_DECODER_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_AUDIO_DECODER_HPP_
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>
namespace gufo::models::embedding_gemma2 {
// Pin transformers' librosa decoder backend: libsndfile float32, arithmetic
// channel mean, then SOXR_HQ resampling and ceil-length zero padding.
// Bounded to 30 seconds, 8 channels, 192 kHz and 20 MiB encoded bytes.
[[nodiscard]] std::vector<float> DecodeAudio(
    std::span<const std::uint8_t> bytes,
    const std::function<bool()>& is_cancelled = {});
}  // namespace gufo::models::embedding_gemma2
#endif
