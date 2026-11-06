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

namespace gufo::server {
namespace {

constexpr std::size_t kMaximumEncodedBytes = 12U << 20U;
constexpr std::uint32_t kMaximumChannels = 8;
constexpr std::uint32_t kMaximumSampleRate = 192000;
constexpr std::uint64_t kMaximumFrames = 30ULL * kMaximumSampleRate;
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

std::uint16_t ReadU16(std::span<const std::byte> bytes, std::size_t offset) {
  return static_cast<std::uint16_t>(
      std::to_integer<unsigned char>(bytes[offset]) |
      (std::to_integer<unsigned char>(bytes[offset + 1]) << 8U));
}

std::uint32_t ReadU32(std::span<const std::byte> bytes, std::size_t offset) {
  std::uint32_t result = 0;
  for (unsigned index = 0; index < 4; ++index) {
    result |= static_cast<std::uint32_t>(
                  std::to_integer<unsigned char>(bytes[offset + index]))
              << (index * 8U);
  }
  return result;
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

AudioBuffer DecodeWavBytes(std::span<const std::byte> bytes) {
  if (bytes.size() < 44 || std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
      std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
    throw std::invalid_argument("reference audio must be a RIFF WAV");
  }
  std::uint16_t format = 0;
  std::uint16_t channels = 0;
  std::uint32_t sample_rate = 0;
  std::uint16_t block_align = 0;
  std::uint16_t bits_per_sample = 0;
  std::span<const std::byte> data;
  for (std::size_t offset = 12; offset + 8 <= bytes.size();) {
    const std::uint32_t chunk_size = ReadU32(bytes, offset + 4);
    const std::size_t data_offset = offset + 8;
    if (chunk_size > bytes.size() - data_offset) {
      throw std::invalid_argument("reference WAV is truncated");
    }
    if (std::memcmp(bytes.data() + offset, "fmt ", 4) == 0) {
      if (chunk_size < 16) {
        throw std::invalid_argument("reference WAV fmt is invalid");
      }
      format = ReadU16(bytes, data_offset);
      channels = ReadU16(bytes, data_offset + 2);
      sample_rate = ReadU32(bytes, data_offset + 4);
      block_align = ReadU16(bytes, data_offset + 12);
      bits_per_sample = ReadU16(bytes, data_offset + 14);
    } else if (std::memcmp(bytes.data() + offset, "data", 4) == 0) {
      data = bytes.subspan(data_offset, chunk_size);
    }
    const std::size_t padded = chunk_size + (chunk_size & 1U);
    if (padded > bytes.size() - data_offset) {
      break;
    }
    offset = data_offset + padded;
  }
  if ((format != 1 && format != 3) || channels == 0 ||
      channels > kMaximumChannels || sample_rate == 0 ||
      sample_rate > kMaximumSampleRate || data.empty()) {
    throw std::invalid_argument("reference WAV format is unsupported");
  }
  const std::size_t bytes_per_sample = bits_per_sample / 8U;
  if ((format == 1 && bits_per_sample != 16) ||
      (format == 3 && bits_per_sample != 32) ||
      block_align != channels * bytes_per_sample ||
      data.size() % block_align != 0) {
    throw std::invalid_argument("supports PCM16 or float32 interleaved WAV");
  }
  const std::uint64_t frames = data.size() / block_align;
  if (frames == 0 || frames > kMaximumFrames || frames > 30ULL * sample_rate) {
    throw std::length_error("reference WAV must be at most 30 seconds");
  }
  AudioBuffer output{
      .sample_rate = sample_rate,
      .channels = channels,
  };
  output.samples.resize(static_cast<std::size_t>(frames) * channels);
  for (std::size_t index = 0; index < output.samples.size(); ++index) {
    const std::size_t offset = index * bytes_per_sample;
    if (format == 1) {
      const std::uint16_t raw = ReadU16(data, offset);
      const auto pcm = static_cast<std::int16_t>(raw);
      output.samples[index] =
          static_cast<float>(pcm) / static_cast<float>(32768);
    } else {
      std::uint32_t raw = ReadU32(data, offset);
      float value = 0.0F;
      std::memcpy(&value, &raw, sizeof(value));
      if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "reference WAV contains non-finite samples");
      }
      output.samples[index] = std::clamp(value, -1.0F, 1.0F);
    }
  }
  return output;
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
