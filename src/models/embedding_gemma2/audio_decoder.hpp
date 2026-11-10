#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_AUDIO_DECODER_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_AUDIO_DECODER_HPP_
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>
namespace gufo::models::embedding_gemma2 {
// Follow transformers' librosa decoder backend: libsndfile's float32
// conversion and arithmetic channel mean (core::DecodeAudio with
// downmix_mono), resampling with SOXR_HQ's response (core::Resample), then
// zero padding to ceil(frames * 16000 / rate) samples. Bounded to 30
// seconds, 8 channels, 8–192 kHz and 20 MiB encoded bytes.
[[nodiscard]] std::vector<float> DecodeAudio(
    std::span<const std::uint8_t> bytes,
    const std::function<bool()>& is_cancelled = {});
}  // namespace gufo::models::embedding_gemma2
#endif
