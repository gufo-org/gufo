#include "audio_decoder.hpp"

#include <chrono>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "audio_processor.hpp"
#include "gufo/ext/audio.hpp"

namespace gufo::models::embedding_gemma2 {
namespace {
constexpr core::AudioLimits kLimits{
    .formats = core::AudioFormats::kAll,
    .max_channels = 8,
    .min_sample_rate = 8000,
    .max_sample_rate = 192000,
    .max_seconds = 30,
    // Mixed down block by block: no buffer holds every channel.
    .downmix_mono = true,
};

constexpr std::string_view kStopped =
    "embedding audio decoding cancelled or timed out";

std::string_view Message(core::AudioStatus status) {
  switch (status) {
    case core::AudioStatus::kUnsupportedCodec:
      return "embedding audio supports WAV, FLAC, Ogg and MP3";
    case core::AudioStatus::kOutOfBounds:
      return "embedding audio exceeds 30 seconds, 8 channels or 8–192 kHz "
             "bounds";
    case core::AudioStatus::kTooLong:
      return "embedding audio exceeds 30 seconds";
    case core::AudioStatus::kNonFinite:
      return "embedding audio has nonfinite samples";
    case core::AudioStatus::kNoAudio:
    case core::AudioStatus::kTruncated:
      return "embedding audio is truncated or decoding failed";
    case core::AudioStatus::kStopped:
      return kStopped;
    case core::AudioStatus::kUnrecognized:
    case core::AudioStatus::kMalformedHeader:
    case core::AudioStatus::kUnsupportedLayout:
      break;
  }
  return "embedding audio could not be decoded";
}
}  // namespace

std::vector<float> DecodeAudio(std::span<const std::uint8_t> bytes,
                               const std::function<bool()>& is_cancelled) {
  if (bytes.empty() || bytes.size() > 20 * 1024 * 1024)
    throw std::invalid_argument("embedding audio is empty or exceeds 20 MiB");
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(15);
  const auto stop = [&] {
    return (is_cancelled && is_cancelled()) ||
           std::chrono::steady_clock::now() > deadline;
  };
  core::DecodedAudio audio;
  try {
    audio = core::DecodeAudio(std::as_bytes(bytes), kLimits, stop);
  } catch (const core::AudioError& error) {
    const std::string message(Message(error.status()));
    if (error.status() == core::AudioStatus::kStopped)
      throw std::runtime_error(message);
    throw std::invalid_argument(message);
  }
  const std::vector<float>& mono = audio.samples;
  std::vector<float> output = core::Resample(
      mono, audio.sample_rate, static_cast<std::uint32_t>(kAudioSampleRate));
  if (stop())
    throw std::runtime_error(std::string(kStopped));
  // Resample returns soxr's rounded length; librosa's fix_length pads with
  // zeros to the ceiling.
  output.resize((mono.size() * kAudioSampleRate + audio.sample_rate - 1) /
                audio.sample_rate);
  return output;
}
}  // namespace gufo::models::embedding_gemma2
