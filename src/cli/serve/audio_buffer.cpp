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

#include "src/core/audio.hpp"
#include "src/core/base64.hpp"

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
  // Nonzero bits under the padding are accepted here.
  std::vector<std::byte> output;
  switch (core::DecodeBase64(encoded, {.canonical = false}, output)) {
    case core::Base64Status::kOk:
      return output;
    case core::Base64Status::kPadding:
    case core::Base64Status::kAfterPadding:
      throw std::invalid_argument("reference audio base64 padding is invalid");
    case core::Base64Status::kLength:
    case core::Base64Status::kCharacter:
    case core::Base64Status::kNonCanonical:
    case core::Base64Status::kTooLong:
      break;
  }
  throw std::invalid_argument("reference audio base64 is invalid");
}

std::string_view WavMessage(core::AudioStatus status) {
  switch (status) {
    case core::AudioStatus::kUnrecognized:
      return "reference audio must be a RIFF WAV";
    case core::AudioStatus::kMalformedHeader:
      return "reference WAV fmt is invalid";
    case core::AudioStatus::kUnsupportedCodec:
    case core::AudioStatus::kOutOfBounds:
    case core::AudioStatus::kNoAudio:
      return "reference WAV format is unsupported";
    case core::AudioStatus::kUnsupportedLayout:
      return "supports PCM16 or float32 interleaved WAV";
    case core::AudioStatus::kTooLong:
      return "reference WAV must be at most 30 seconds";
    case core::AudioStatus::kTruncated:
      return "reference WAV is truncated";
    case core::AudioStatus::kNonFinite:
      return "reference WAV contains non-finite samples";
    case core::AudioStatus::kStopped:
      break;
  }
  return "reference WAV could not be decoded";
}

AudioBuffer DecodeWavBytes(std::span<const std::byte> bytes) {
  try {
    core::DecodedAudio wav = core::DecodeAudio(bytes, core::AudioLimits{});
    return AudioBuffer{
        .sample_rate = wav.sample_rate,
        .channels = wav.channels,
        .samples = std::move(wav.samples),
    };
  } catch (const core::AudioError& error) {
    throw std::invalid_argument(std::string(WavMessage(error.status())));
  }
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
