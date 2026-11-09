// DecodeWavBytes and its helpers from src/cli/serve/audio_buffer.cpp,
// src/models/qwen3_asr/audio.cpp and src/models/qwen3_tts/audio.cpp in gufo
// 0.32.0, before the Rust WAV decoder replaced them. Only the namespaces and
// the AudioBuffer alias are new.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>

#include "src/core/wav.hpp"

namespace gufo::core::wav_reference {
namespace serve {

using AudioBuffer = WavAudio;

constexpr std::uint32_t kMaximumChannels = 8;
constexpr std::uint32_t kMaximumSampleRate = 192000;
constexpr std::uint64_t kMaximumFrames = 30ULL * kMaximumSampleRate;

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

}  // namespace serve

namespace asr {

using AudioBuffer = WavAudio;

constexpr std::uint32_t kMaximumChannels = 8;
constexpr std::uint32_t kMaximumSampleRate = 192000;
constexpr std::uint64_t kMaximumSeconds = static_cast<std::uint64_t>(30U) * 60U;

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

AudioBuffer DecodeWavBytes(std::span<const std::byte> bytes) {
  if (bytes.size() < 44U || std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
      std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
    throw std::invalid_argument("Qwen3-ASR input must be a RIFF WAV");
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
      throw std::invalid_argument("Qwen3-ASR WAV is truncated");
    }
    if (std::memcmp(bytes.data() + offset, "fmt ", 4) == 0) {
      if (chunk_size < 16U) {
        throw std::invalid_argument("Qwen3-ASR WAV fmt chunk is invalid");
      }
      format = ReadU16(bytes, data_offset);
      channels = ReadU16(bytes, data_offset + 2);
      sample_rate = ReadU32(bytes, data_offset + 4);
      block_align = ReadU16(bytes, data_offset + 12);
      bits_per_sample = ReadU16(bytes, data_offset + 14);
    } else if (std::memcmp(bytes.data() + offset, "data", 4) == 0) {
      data = bytes.subspan(data_offset, chunk_size);
    }
    const std::size_t padded_size = chunk_size + (chunk_size & 1U);
    if (padded_size > bytes.size() - data_offset) {
      break;
    }
    offset = data_offset + padded_size;
  }

  if ((format != 1U && format != 3U) || channels == 0U ||
      channels > kMaximumChannels || sample_rate == 0U ||
      sample_rate > kMaximumSampleRate || data.empty()) {
    throw std::invalid_argument("Qwen3-ASR WAV format is unsupported");
  }
  const std::size_t bytes_per_sample = bits_per_sample / 8U;
  if ((format == 1U && bits_per_sample != 16U && bits_per_sample != 24U &&
       bits_per_sample != 32U) ||
      (format == 3U && bits_per_sample != 32U) ||
      block_align != channels * bytes_per_sample ||
      data.size() % block_align != 0U) {
    throw std::invalid_argument(
        "Qwen3-ASR supports PCM16/24/32 or float32 interleaved WAV");
  }
  const std::uint64_t frames = data.size() / block_align;
  if (frames == 0U || frames > kMaximumSeconds * sample_rate) {
    throw std::length_error("Qwen3-ASR WAV must be at most 30 minutes");
  }

  AudioBuffer output{
      .sample_rate = sample_rate,
      .channels = channels,
  };
  output.samples.resize(static_cast<std::size_t>(frames) * channels);
  for (std::size_t index = 0; index < output.samples.size(); ++index) {
    const std::size_t offset = index * bytes_per_sample;
    if (format == 1U) {
      if (bits_per_sample == 16U) {
        const auto pcm = static_cast<std::int16_t>(ReadU16(data, offset));
        output.samples[index] =
            static_cast<float>(pcm) / static_cast<float>(32768);
      } else if (bits_per_sample == 24U) {
        std::int32_t pcm =
            static_cast<std::int32_t>(
                std::to_integer<unsigned char>(data[offset])) |
            (static_cast<std::int32_t>(
                 std::to_integer<unsigned char>(data[offset + 1U]))
             << 8U) |
            (static_cast<std::int32_t>(
                 std::to_integer<unsigned char>(data[offset + 2U]))
             << 16U);
        if ((pcm & 0x00800000) != 0) {
          pcm |= static_cast<std::int32_t>(0xFF000000U);
        }
        output.samples[index] =
            static_cast<float>(pcm) / static_cast<float>(8388608);
      } else {
        const auto pcm = static_cast<std::int32_t>(ReadU32(data, offset));
        output.samples[index] =
            static_cast<float>(pcm) / static_cast<float>(2147483648.0);
      }
    } else {
      const std::uint32_t raw = ReadU32(data, offset);
      float value = 0.0F;
      static_assert(sizeof(value) == sizeof(raw));
      std::memcpy(&value, &raw, sizeof(value));
      if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "Qwen3-ASR WAV contains non-finite samples");
      }
      output.samples[index] = std::clamp(value, -1.0F, 1.0F);
    }
  }
  return output;
}

}  // namespace asr

namespace tts {

using AudioBuffer = WavAudio;

constexpr std::uint32_t kMaximumChannels = 8;
constexpr std::uint32_t kMaximumSampleRate = 192000;
constexpr std::uint64_t kMaximumFrames = 30ULL * kMaximumSampleRate;

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

AudioBuffer DecodeWavBytes(std::span<const std::byte> bytes) {
  if (bytes.size() < 44 || std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
      std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
    throw std::invalid_argument("Qwen3-TTS reference audio must be a RIFF WAV");
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
      throw std::invalid_argument("Qwen3-TTS reference WAV is truncated");
    }
    if (std::memcmp(bytes.data() + offset, "fmt ", 4) == 0) {
      if (chunk_size < 16) {
        throw std::invalid_argument("Qwen3-TTS reference WAV fmt is invalid");
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
    throw std::invalid_argument(
        "Qwen3-TTS reference WAV format is unsupported");
  }
  const std::size_t bytes_per_sample = bits_per_sample / 8U;
  if ((format == 1 && bits_per_sample != 16) ||
      (format == 3 && bits_per_sample != 32) ||
      block_align != channels * bytes_per_sample ||
      data.size() % block_align != 0) {
    throw std::invalid_argument(
        "Qwen3-TTS supports PCM16 or float32 interleaved WAV");
  }
  const std::uint64_t frames = data.size() / block_align;
  if (frames == 0 || frames > kMaximumFrames || frames > 30ULL * sample_rate) {
    throw std::length_error(
        "Qwen3-TTS reference WAV must be at most 30 seconds");
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
            "Qwen3-TTS reference WAV contains non-finite samples");
      }
      output.samples[index] = std::clamp(value, -1.0F, 1.0F);
    }
  }
  return output;
}

}  // namespace tts

}  // namespace gufo::core::wav_reference
