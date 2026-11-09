#include "src/core/wav.hpp"

#include <gufo/wav_decode.h>

#include <stdexcept>
#include <string>

namespace gufo::core {
namespace {

[[noreturn]] void Throw(GufoWavStatus status, const WavMessages& messages) {
  switch (status) {
    case GUFO_WAV_NOT_RIFF:
      throw std::invalid_argument(std::string(messages.not_riff));
    case GUFO_WAV_TRUNCATED:
      throw std::invalid_argument(std::string(messages.truncated));
    case GUFO_WAV_INVALID_FMT:
      throw std::invalid_argument(std::string(messages.invalid_fmt));
    case GUFO_WAV_UNSUPPORTED_FORMAT:
      throw std::invalid_argument(std::string(messages.unsupported_format));
    case GUFO_WAV_UNSUPPORTED_LAYOUT:
      throw std::invalid_argument(std::string(messages.unsupported_layout));
    case GUFO_WAV_TOO_LONG:
      throw std::length_error(std::string(messages.too_long));
    case GUFO_WAV_NON_FINITE:
      throw std::invalid_argument(std::string(messages.non_finite));
    case GUFO_WAV_OK:
    case GUFO_WAV_INTERNAL:
      break;
  }
  throw std::logic_error("Rust WAV decoder failed internally");
}

}  // namespace

WavAudio DecodeWav(std::span<const std::byte> bytes, const WavLimits& limits,
                   const WavMessages& messages) {
  const auto* data = reinterpret_cast<const std::uint8_t*>(bytes.data());
  const GufoWavLimits rust_limits{
      .max_channels = limits.max_channels,
      .max_sample_rate = limits.max_sample_rate,
      .max_seconds = limits.max_seconds,
      .max_pcm_bits = limits.max_pcm_bits,
  };
  GufoWavInfo info{};
  if (const GufoWavStatus status =
          gufo_wav_read_info(data, bytes.size(), &rust_limits, &info);
      status != GUFO_WAV_OK) {
    Throw(status, messages);
  }
  WavAudio output{
      .sample_rate = info.sample_rate,
      .channels = info.channels,
  };
  output.samples.resize(static_cast<std::size_t>(info.frames) * info.channels);
  if (const GufoWavStatus status =
          gufo_wav_decode_f32(data, bytes.size(), &rust_limits,
                              output.samples.data(), output.samples.size());
      status != GUFO_WAV_OK) {
    Throw(status, messages);
  }
  return output;
}

}  // namespace gufo::core
