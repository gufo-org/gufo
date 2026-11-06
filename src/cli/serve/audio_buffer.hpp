#ifndef GUFO_SERVER_AUDIO_BUFFER_HPP_
#define GUFO_SERVER_AUDIO_BUFFER_HPP_

#include <cstddef>
#include <cstdint>
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

[[nodiscard]] std::vector<float> ResampleMono(const AudioBuffer& audio,
                                              std::uint32_t output_sample_rate);

[[nodiscard]] std::vector<float> ResampleMono16k(const AudioBuffer& audio);

}  // namespace gufo::server

#endif  // GUFO_SERVER_AUDIO_BUFFER_HPP_
