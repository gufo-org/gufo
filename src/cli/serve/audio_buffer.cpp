#include "src/cli/serve/audio_buffer.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/core/wav.hpp"

namespace gufo::server {
namespace {

constexpr std::size_t kMaximumEncodedBytes = 12U << 20U;
constexpr std::int64_t kResamplerRadius = 24;
constexpr std::size_t kResamplerTaps =
    2U * static_cast<std::size_t>(kResamplerRadius);

struct ResamplerPhase {
  std::array<double, kResamplerTaps> weights{};
  std::array<std::int64_t, kResamplerTaps> offsets{};
  std::size_t taps{0U};
  double normalization{0.0};
};

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

int Base64Value(unsigned char character) {
  if (character >= 'A' && character <= 'Z') {
    return character - 'A';
  }
  if (character >= 'a' && character <= 'z') {
    return character - 'a' + 26;
  }
  if (character >= '0' && character <= '9') {
    return character - '0' + 52;
  }
  if (character == '+') {
    return 62;
  }
  if (character == '/') {
    return 63;
  }
  return -1;
}

std::int64_t ReflectIndex(std::int64_t index, std::int64_t length) {
  if (length <= 1) {
    return 0;
  }
  while (index < 0 || index >= length) {
    index = index < 0 ? -index : 2 * length - index - 2;
  }
  return index;
}

std::vector<std::byte> DecodeBase64(std::string_view encoded) {
  constexpr std::string_view prefix = "data:audio/wav;base64,";
  if (encoded.starts_with(prefix)) {
    encoded.remove_prefix(prefix.size());
  }
  if (encoded.empty() || encoded.size() > kMaximumEncodedBytes ||
      encoded.size() % 4 != 0) {
    throw std::invalid_argument("reference audio base64 is invalid");
  }
  std::vector<std::byte> output;
  output.reserve((encoded.size() / 4) * 3);
  for (std::size_t offset = 0; offset < encoded.size(); offset += 4) {
    std::uint32_t block = 0;
    unsigned padding = 0;
    for (unsigned index = 0; index < 4; ++index) {
      const unsigned char character =
          static_cast<unsigned char>(encoded[offset + index]);
      if (character == '=') {
        if (index < 2 || offset + 4 != encoded.size()) {
          throw std::invalid_argument(
              "reference audio base64 padding is invalid");
        }
        ++padding;
        block <<= 6U;
      } else {
        if (padding != 0) {
          throw std::invalid_argument(
              "reference audio base64 padding is invalid");
        }
        const int value = Base64Value(character);
        if (value < 0) {
          throw std::invalid_argument("reference audio base64 is invalid");
        }
        block = (block << 6U) | static_cast<std::uint32_t>(value);
      }
    }
    output.push_back(static_cast<std::byte>((block >> 16U) & 0xFFU));
    if (padding < 2) {
      output.push_back(static_cast<std::byte>((block >> 8U) & 0xFFU));
    }
    if (padding == 0) {
      output.push_back(static_cast<std::byte>(block & 0xFFU));
    }
  }
  return output;
}

constexpr core::WavMessages kWavMessages{
    .not_riff = "reference audio must be a RIFF WAV",
    .truncated = "reference WAV is truncated",
    .invalid_fmt = "reference WAV fmt is invalid",
    .unsupported_format = "reference WAV format is unsupported",
    .unsupported_layout = "supports PCM16 or float32 interleaved WAV",
    .too_long = "reference WAV must be at most 30 seconds",
    .non_finite = "reference WAV contains non-finite samples",
};

AudioBuffer DecodeWavBytes(std::span<const std::byte> bytes) {
  core::WavAudio wav = core::DecodeWav(bytes, core::WavLimits{}, kWavMessages);
  return AudioBuffer{
      .sample_rate = wav.sample_rate,
      .channels = wav.channels,
      .samples = std::move(wav.samples),
  };
}

}  // namespace

bool DecodeWav(std::span<const std::byte> bytes, AudioBuffer* output,
               std::string* error) {
  if (output == nullptr) {
    SetError(error, "audio output must not be null");
    return false;
  }
  *output = {};
  try {
    *output = DecodeWavBytes(bytes);
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool DecodeBase64Wav(std::string_view encoded, AudioBuffer* output,
                     std::string* error) {
  if (output == nullptr) {
    SetError(error, "reference audio output must not be null");
    return false;
  }
  *output = {};
  try {
    const std::vector<std::byte> bytes = DecodeBase64(encoded);
    *output = DecodeWavBytes(bytes);
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

std::vector<float> ResampleMono(const AudioBuffer& audio,
                                std::uint32_t output_sample_rate) {
  if (audio.sample_rate == 0 || audio.channels == 0 || audio.samples.empty() ||
      audio.samples.size() % audio.channels != 0 || output_sample_rate == 0) {
    throw std::invalid_argument("audio buffer is invalid");
  }
  const std::size_t input_frames = audio.samples.size() / audio.channels;
  std::vector<float> mono(input_frames, 0.0F);
  for (std::size_t frame = 0; frame < input_frames; ++frame) {
    double sum = 0.0;
    for (std::uint32_t channel = 0; channel < audio.channels; ++channel) {
      sum += audio.samples[frame * audio.channels + channel];
    }
    mono[frame] = static_cast<float>(sum / static_cast<double>(audio.channels));
  }
  if (audio.sample_rate == output_sample_rate) {
    return mono;
  }
  const std::size_t output_frames = static_cast<std::size_t>(
      (static_cast<std::uint64_t>(input_frames) * output_sample_rate +
       audio.sample_rate - 1) /
      audio.sample_rate);
  std::vector<float> result(output_frames, 0.0F);
  const double scale = static_cast<double>(audio.sample_rate) /
                       static_cast<double>(output_sample_rate);
  const double cutoff =
      std::min(1.0, static_cast<double>(output_sample_rate) /
                        static_cast<double>(audio.sample_rate));
  std::unordered_map<std::uint64_t, ResamplerPhase> phases;
  for (std::size_t frame = 0; frame < output_frames; ++frame) {
    const double position = static_cast<double>(frame) * scale;
    const auto center = static_cast<std::int64_t>(std::floor(position));
    const double fraction = position - static_cast<double>(center);
    const std::uint64_t key = std::bit_cast<std::uint64_t>(fraction);

    auto entry = phases.find(key);
    if (entry == phases.end()) {
      ResamplerPhase phase_taps;
      for (std::int64_t tap = -kResamplerRadius + 1; tap <= kResamplerRadius;
           ++tap) {
        const double distance = position - static_cast<double>(center + tap);
        const double window_position =
            std::abs(distance) / static_cast<double>(kResamplerRadius);
        if (window_position >= 1.0) {
          continue;
        }
        const double phase = std::numbers::pi * cutoff * distance;
        const double sinc = phase == 0.0 ? 1.0 : std::sin(phase) / phase;
        const double window =
            0.5 + 0.5 * std::cos(std::numbers::pi * window_position);
        const double weight = cutoff * sinc * window;
        phase_taps.offsets[phase_taps.taps] = tap;
        phase_taps.weights[phase_taps.taps] = weight;
        ++phase_taps.taps;
        phase_taps.normalization += weight;
      }
      entry = phases.emplace(key, phase_taps).first;
    }

    const ResamplerPhase& phase_taps = entry->second;
    double weighted = 0.0;
    for (std::size_t tap = 0; tap < phase_taps.taps; ++tap) {
      const std::int64_t reflected =
          ReflectIndex(center + phase_taps.offsets[tap],
                       static_cast<std::int64_t>(input_frames));
      weighted +=
          static_cast<double>(mono[static_cast<std::size_t>(reflected)]) *
          phase_taps.weights[tap];
    }
    result[frame] =
        phase_taps.normalization == 0.0
            ? 0.0F
            : static_cast<float>(weighted / phase_taps.normalization);
  }
  return result;
}

std::vector<float> ResampleMono16k(const AudioBuffer& audio) {
  return ResampleMono(audio, 16000);
}

}  // namespace gufo::server
