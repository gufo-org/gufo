#ifndef GUFO_MODELS_QWEN3_TTS_AUDIO_HPP_
#define GUFO_MODELS_QWEN3_TTS_AUDIO_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/ext/tts.hpp"

namespace gufo::models::qwen3_tts {

using AudioBuffer = ext::TtsAudioBuffer;

[[nodiscard]] bool DecodeWav(std::span<const std::byte> bytes,
                             AudioBuffer* output, std::string* error = nullptr);

[[nodiscard]] std::vector<float> ResampleMono(const AudioBuffer& audio,
                                              std::uint32_t output_sample_rate);

}  // namespace gufo::models::qwen3_tts

#endif  // GUFO_MODELS_QWEN3_TTS_AUDIO_HPP_
