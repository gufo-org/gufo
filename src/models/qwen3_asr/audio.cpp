#include "src/models/qwen3_asr/audio.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gufo::models::qwen3_asr {
namespace {

constexpr std::uint32_t kMaximumChannels = 8;
constexpr std::uint32_t kMaximumSampleRate = 192000;
constexpr std::uint64_t kMaximumSeconds = static_cast<std::uint64_t>(30U) * 60U;
constexpr std::size_t kFrequencyBins = kAudioFftSize / 2U + 1U;
constexpr std::size_t kCenterPadding = kAudioFftSize / 2U;
constexpr std::size_t kAudioChunkFrames = 100U;
constexpr std::size_t kAudioChunkTokens = 13U;

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

std::int64_t ReflectIndex(std::int64_t index, std::int64_t length) {
  if (length <= 1) {
    return 0;
  }
  while (index < 0 || index >= length) {
    index = index < 0 ? -index : 2 * length - index - 2;
  }
  return index;
}

double HertzToSlaneyMel(double hertz) {
  constexpr double kMinLogHertz = 1000.0;
  constexpr double kMinLogMel = 15.0;
  const double kLogStep = 27.0 / std::log(6.4);
  if (hertz >= kMinLogHertz) {
    return kMinLogMel + std::log(hertz / kMinLogHertz) * kLogStep;
  }
  return 3.0 * hertz / 200.0;
}

double SlaneyMelToHertz(double mel) {
  constexpr double kMinLogHertz = 1000.0;
  constexpr double kMinLogMel = 15.0;
  const double kLogStep = std::log(6.4) / 27.0;
  if (mel >= kMinLogMel) {
    return kMinLogHertz * std::exp(kLogStep * (mel - kMinLogMel));
  }
  return 200.0 * mel / 3.0;
}

using MelFilterbank = std::array<float, kFrequencyBins * kAudioMelBins>;

MelFilterbank MakeMelFilterbank() {
  std::array<double, kAudioMelBins + 2U> filter_frequencies{};
  const double minimum_mel = HertzToSlaneyMel(0.0);
  const double maximum_mel =
      HertzToSlaneyMel(static_cast<double>(kAudioSampleRate) / 2.0);
  for (std::size_t index = 0; index < filter_frequencies.size(); ++index) {
    const double mel =
        minimum_mel + (maximum_mel - minimum_mel) * static_cast<double>(index) /
                          static_cast<double>(kAudioMelBins + 1U);
    filter_frequencies[index] = SlaneyMelToHertz(mel);
  }

  MelFilterbank result{};
  for (std::size_t bin = 0; bin < kFrequencyBins; ++bin) {
    const double frequency = (static_cast<double>(kAudioSampleRate) / 2.0) *
                             static_cast<double>(bin) /
                             static_cast<double>(kFrequencyBins - 1U);
    for (std::size_t mel = 0; mel < kAudioMelBins; ++mel) {
      const double left = filter_frequencies[mel];
      const double center = filter_frequencies[mel + 1U];
      const double right = filter_frequencies[mel + 2U];
      const double lower = (frequency - left) / (center - left);
      const double upper = (right - frequency) / (right - center);
      const double triangle = std::max(0.0, std::min(lower, upper));
      const double slaney_normalization = 2.0 / (right - left);
      result[bin * kAudioMelBins + mel] =
          static_cast<float>(triangle * slaney_normalization);
    }
  }
  return result;
}

struct DftTable {
  std::array<float, kFrequencyBins * kAudioFftSize> real{};
  std::array<float, kFrequencyBins * kAudioFftSize> imaginary{};
};

DftTable MakeDftTable() {
  DftTable result;
  for (std::size_t bin = 0; bin < kFrequencyBins; ++bin) {
    for (std::size_t sample = 0; sample < kAudioFftSize; ++sample) {
      const double window =
          0.5 -
          0.5 * std::cos(2.0 * std::numbers::pi * static_cast<double>(sample) /
                         static_cast<double>(kAudioFftSize));
      const double phase = -2.0 * std::numbers::pi *
                           static_cast<double>(bin * sample) /
                           static_cast<double>(kAudioFftSize);
      const std::size_t index = bin * kAudioFftSize + sample;
      result.real[index] = static_cast<float>(window * std::cos(phase));
      result.imaginary[index] = static_cast<float>(window * std::sin(phase));
    }
  }
  return result;
}

}  // namespace

bool DecodeWav(std::span<const std::byte> bytes, AudioBuffer* output,
               std::string* error) {
  if (output == nullptr) {
    SetError(error, "Qwen3-ASR audio output must not be null");
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

std::vector<float> ResampleMono16k(const AudioBuffer& audio) {
  if (audio.sample_rate == 0U || audio.channels == 0U ||
      audio.samples.empty() || audio.samples.size() % audio.channels != 0U) {
    throw std::invalid_argument("Qwen3-ASR audio buffer is invalid");
  }
  const std::size_t input_frames = audio.samples.size() / audio.channels;
  std::vector<float> mono(input_frames, 0.0F);
  for (std::size_t frame = 0; frame < input_frames; ++frame) {
    double sum = 0.0;
    for (std::uint32_t channel = 0; channel < audio.channels; ++channel) {
      sum += audio.samples[frame * audio.channels + channel];
    }
    mono[frame] = static_cast<float>(sum / audio.channels);
  }
  if (audio.sample_rate == kAudioSampleRate) {
    return mono;
  }

  // A compact windowed-sinc converter avoids folding ultrasonic energy into
  // the speech band when validating 24 kHz Qwen3-TTS output. The 16 kHz
  // reference path above intentionally remains an exact copy.
  constexpr std::int64_t kRadius = 24;
  const std::size_t output_frames = static_cast<std::size_t>(
      (static_cast<std::uint64_t>(input_frames) * kAudioSampleRate +
       audio.sample_rate - 1U) /
      audio.sample_rate);
  std::vector<float> result(output_frames, 0.0F);
  const double scale = static_cast<double>(audio.sample_rate) /
                       static_cast<double>(kAudioSampleRate);
  const double cutoff =
      std::min(1.0, static_cast<double>(kAudioSampleRate) /
                        static_cast<double>(audio.sample_rate));
  for (std::size_t frame = 0; frame < output_frames; ++frame) {
    const double position = static_cast<double>(frame) * scale;
    const auto center = static_cast<std::int64_t>(std::floor(position));
    double weighted = 0.0;
    double normalization = 0.0;
    for (std::int64_t tap = -kRadius + 1; tap <= kRadius; ++tap) {
      const std::int64_t source = center + tap;
      const double distance = position - static_cast<double>(source);
      const double phase = std::numbers::pi * cutoff * distance;
      const double sinc = phase == 0.0 ? 1.0 : std::sin(phase) / phase;
      const double window_position =
          std::abs(distance) / static_cast<double>(kRadius);
      if (window_position >= 1.0) {
        continue;
      }
      const double window =
          0.5 + 0.5 * std::cos(std::numbers::pi * window_position);
      const double weight = cutoff * sinc * window;
      const std::int64_t reflected =
          ReflectIndex(source, static_cast<std::int64_t>(input_frames));
      weighted +=
          static_cast<double>(mono[static_cast<std::size_t>(reflected)]) *
          weight;
      normalization += weight;
    }
    result[frame] = normalization == 0.0
                        ? 0.0F
                        : static_cast<float>(weighted / normalization);
  }
  return result;
}

LogMelFeatures ComputeLogMelFeatures(std::span<const float> waveform) {
  if (waveform.size() <= kCenterPadding) {
    throw std::invalid_argument(
        "Qwen3-ASR input is too short for reflected STFT padding");
  }
  if (waveform.size() >
      static_cast<std::size_t>(kMaximumSeconds) * kAudioSampleRate) {
    throw std::length_error("Qwen3-ASR input must be at most 30 minutes");
  }
  if (!std::all_of(waveform.begin(), waveform.end(),
                   [](float value) { return std::isfinite(value); })) {
    throw std::invalid_argument("Qwen3-ASR input contains non-finite samples");
  }

  std::vector<float> padded(waveform.size() + 2U * kCenterPadding);
  for (std::size_t index = 0; index < padded.size(); ++index) {
    const std::int64_t source =
        ReflectIndex(static_cast<std::int64_t>(index) -
                         static_cast<std::int64_t>(kCenterPadding),
                     static_cast<std::int64_t>(waveform.size()));
    padded[index] = waveform[static_cast<std::size_t>(source)];
  }

  const std::size_t stft_frames =
      1U + (padded.size() - kAudioFftSize) / kAudioHopLength;
  // WhisperFeatureExtractor intentionally drops the final centered STFT bin.
  const std::size_t frames = stft_frames - 1U;
  std::vector<float> power(kFrequencyBins * frames, 0.0F);
  static const DftTable dft = MakeDftTable();
  for (std::size_t frame = 0; frame < frames; ++frame) {
    const float* samples = padded.data() + frame * kAudioHopLength;
    for (std::size_t bin = 0; bin < kFrequencyBins; ++bin) {
      double real = 0.0;
      double imaginary = 0.0;
      const std::size_t table_offset = bin * kAudioFftSize;
      for (std::size_t sample = 0; sample < kAudioFftSize; ++sample) {
        real += static_cast<double>(samples[sample]) *
                dft.real[table_offset + sample];
        imaginary += static_cast<double>(samples[sample]) *
                     dft.imaginary[table_offset + sample];
      }
      power[bin * frames + frame] =
          static_cast<float>(real * real + imaginary * imaginary);
    }
  }

  static const MelFilterbank filters = MakeMelFilterbank();
  LogMelFeatures result;
  result.frames = frames;
  result.values.resize(kAudioMelBins * frames);
  float maximum = -std::numeric_limits<float>::infinity();
  for (std::size_t mel = 0; mel < kAudioMelBins; ++mel) {
    for (std::size_t frame = 0; frame < frames; ++frame) {
      float sum = 0.0F;
      for (std::size_t bin = 0; bin < kFrequencyBins; ++bin) {
        sum += filters[bin * kAudioMelBins + mel] * power[bin * frames + frame];
      }
      const float value = std::log10(std::max(sum, 1.0e-10F));
      result.values[mel * frames + frame] = value;
      maximum = std::max(maximum, value);
    }
  }
  const float floor = maximum - 8.0F;
  for (float& value : result.values) {
    value = (std::max(value, floor) + 4.0F) / 4.0F;
  }
  return result;
}

std::size_t AudioEmbeddingTokenCount(std::size_t feature_frames) {
  if (feature_frames == 0U) {
    throw std::invalid_argument(
        "Qwen3-ASR audio token count requires log-mel frames");
  }
  const std::size_t chunks = 1U + (feature_frames - 1U) / kAudioChunkFrames;
  const std::size_t tail_frames =
      feature_frames - (chunks - 1U) * kAudioChunkFrames;
  const std::size_t tail_tokens = (tail_frames + 7U) / 8U;
  if (chunks - 1U > (std::numeric_limits<std::size_t>::max() - tail_tokens) /
                        kAudioChunkTokens) {
    throw std::overflow_error("Qwen3-ASR audio token count overflow");
  }
  return (chunks - 1U) * kAudioChunkTokens + tail_tokens;
}

}  // namespace gufo::models::qwen3_asr
