#ifndef GUFO_SERVER_AUDIO_BUFFER_HPP_
#define GUFO_SERVER_AUDIO_BUFFER_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::server {

struct AudioBuffer {
  std::uint32_t sample_rate{0};
  std::uint32_t channels{0};
  // Interleaved float samples [-1.0, 1.0].
  std::vector<float> samples;
};

[[nodiscard]] bool DecodeWav(std::span<const std::byte> bytes,
                             AudioBuffer* output, std::string* error = nullptr);

[[nodiscard]] bool DecodeBase64Wav(std::string_view encoded,
                                   AudioBuffer* output,
                                   std::string* error = nullptr);

// Decodes an uploaded transcription file: WAV, FLAC, MP3, Ogg Vorbis, Ogg
// Opus or WebM (Opus or Vorbis), at most 30 minutes and 8 channels. PCM16,
// PCM24, PCM32 and float32 WAV keep every channel, as Qwen3-ASR has always
// read them; other inputs are mixed down to mono while decoding. Returns
// false with an English message when the input cannot be decoded, or when
// should_stop returns true.
[[nodiscard]] bool DecodeTranscriptionAudio(
    std::span<const std::byte> bytes, const std::function<bool()>& should_stop,
    AudioBuffer* output, std::string* error);

[[nodiscard]] std::vector<float> ResampleMono(const AudioBuffer& audio,
                                              std::uint32_t output_sample_rate);

[[nodiscard]] std::vector<float> ResampleMono16k(const AudioBuffer& audio);

}  // namespace gufo::server

#endif  // GUFO_SERVER_AUDIO_BUFFER_HPP_
