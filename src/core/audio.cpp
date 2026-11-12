#include "src/core/audio.hpp"

#include <gufo/audio_decode.h>

#include <exception>
#include <string>

namespace gufo::core {
namespace {

const char* Describe(AudioStatus status) {
  switch (status) {
    case AudioStatus::kUnrecognized:
      return "audio format is not recognized";
    case AudioStatus::kMalformedHeader:
      return "audio header is malformed";
    case AudioStatus::kUnsupportedCodec:
      return "audio codec is not supported";
    case AudioStatus::kUnsupportedLayout:
      return "audio sample layout is not supported";
    case AudioStatus::kOutOfBounds:
      return "audio channels or sample rate are out of bounds";
    case AudioStatus::kNoAudio:
      return "audio has no samples";
    case AudioStatus::kTooLong:
      return "audio is too long";
    case AudioStatus::kTruncated:
      return "audio is truncated or fails to decode";
    case AudioStatus::kNonFinite:
      return "audio has non-finite samples";
    case AudioStatus::kStopped:
      return "audio decoding was stopped";
  }
  return "audio decoding failed";
}

// The mean of each frame's channels into `mono`, one value per frame.
void Mix(std::span<const float> interleaved, std::uint32_t channels,
         std::span<float> mono) {
  for (std::size_t frame = 0; frame < mono.size(); ++frame) {
    float sum = 0;
    for (std::uint32_t channel = 0; channel < channels; ++channel) {
      sum += interleaved[frame * channels + channel];
    }
    mono[frame] = sum / static_cast<float>(channels);
  }
}

struct Context {
  const std::function<bool()>* should_stop;
  DecodedAudio* output;
  bool downmix_mono;
  // An exception from should_stop or the copy, rethrown once Rust returns.
  std::exception_ptr failure;
};

int Stop(void* opaque) noexcept {
  auto& context = *static_cast<Context*>(opaque);
  try {
    return *context.should_stop && (*context.should_stop)() ? 1 : 0;
  } catch (...) {
    context.failure = std::current_exception();
    return 1;
  }
}

int Append(void* opaque, const GufoAudioInfo* info, const float* samples,
           std::size_t len) noexcept {
  auto& context = *static_cast<Context*>(opaque);
  try {
    DecodedAudio& output = *context.output;
    if (output.samples.empty()) {
      output.sample_rate = info->sample_rate;
      output.channels = context.downmix_mono ? 1 : info->channels;
      // The declared length, within the caller's limits: a forged header
      // reserves address space that stays untouched.
      output.samples.reserve(
          static_cast<std::size_t>(info->frames * output.channels));
    }
    if (!context.downmix_mono) {
      output.samples.insert(output.samples.end(), samples, samples + len);
      return 0;
    }
    // Rust hands over whole frames.
    const std::size_t frames = output.samples.size();
    output.samples.resize(frames + len / info->channels);
    Mix(std::span(samples, len), info->channels,
        std::span(output.samples).subspan(frames));
    return 0;
  } catch (...) {
    context.failure = std::current_exception();
    return 1;
  }
}

AudioStatus FromRust(GufoAudioStatus status) {
  switch (status) {
    case GUFO_AUDIO_UNRECOGNIZED:
      return AudioStatus::kUnrecognized;
    case GUFO_AUDIO_MALFORMED_HEADER:
      return AudioStatus::kMalformedHeader;
    case GUFO_AUDIO_UNSUPPORTED_CODEC:
      return AudioStatus::kUnsupportedCodec;
    case GUFO_AUDIO_UNSUPPORTED_LAYOUT:
      return AudioStatus::kUnsupportedLayout;
    case GUFO_AUDIO_OUT_OF_BOUNDS:
      return AudioStatus::kOutOfBounds;
    case GUFO_AUDIO_NO_AUDIO:
      return AudioStatus::kNoAudio;
    case GUFO_AUDIO_TOO_LONG:
      return AudioStatus::kTooLong;
    case GUFO_AUDIO_TRUNCATED:
      return AudioStatus::kTruncated;
    case GUFO_AUDIO_NON_FINITE:
      return AudioStatus::kNonFinite;
    case GUFO_AUDIO_STOPPED:
      return AudioStatus::kStopped;
    case GUFO_AUDIO_OK:
    case GUFO_AUDIO_INTERNAL:
      break;
  }
  throw std::logic_error("Rust audio decoder failed internally");
}

}  // namespace

AudioError::AudioError(AudioStatus status)
    : std::runtime_error(Describe(status)), status_(status) {}

DecodedAudio DecodeAudio(std::span<const std::byte> bytes,
                         const AudioLimits& limits,
                         const std::function<bool()>& should_stop) {
  const GufoAudioLimits rust_limits{
      .formats = limits.formats == AudioFormats::kAll
                     ? GUFO_AUDIO_FORMATS_ALL
                     : GUFO_AUDIO_FORMATS_PCM_WAV,
      .max_channels = limits.max_channels,
      .min_sample_rate = limits.min_sample_rate,
      .max_sample_rate = limits.max_sample_rate,
      .max_seconds = limits.max_seconds,
      .max_pcm_bits = limits.max_pcm_bits,
  };
  DecodedAudio output;
  Context context{&should_stop, &output, limits.downmix_mono, nullptr};
  const GufoAudioStatus status =
      gufo_audio_decode(reinterpret_cast<const std::uint8_t*>(bytes.data()),
                        bytes.size(), &rust_limits, Stop, Append, &context);
  if (context.failure) {
    std::rethrow_exception(context.failure);
  }
  if (status != GUFO_AUDIO_OK) {
    throw AudioError(FromRust(status));
  }
  return output;
}

std::vector<float> DownmixMono(std::span<const float> interleaved,
                               std::uint32_t channels) {
  if (channels == 0 || interleaved.size() % channels != 0) {
    throw std::invalid_argument(
        "DownmixMono: channels must divide the sample count");
  }
  std::vector<float> mono(interleaved.size() / channels);
  Mix(interleaved, channels, mono);
  return mono;
}

std::vector<float> Resample(std::span<const float> samples,
                            std::uint32_t from_rate, std::uint32_t to_rate) {
  DecodedAudio output;
  const std::function<bool()> never;
  Context context{&never, &output, false, nullptr};
  const GufoAudioStatus status = gufo_audio_resample(
      samples.data(), samples.size(), from_rate, to_rate, Append, &context);
  if (context.failure) {
    std::rethrow_exception(context.failure);
  }
  if (status == GUFO_AUDIO_OUT_OF_BOUNDS) {
    throw std::invalid_argument(
        "Resample: a rate is 0 or more than 64 times the other");
  }
  if (status != GUFO_AUDIO_OK) {
    throw std::logic_error("Rust resampler failed internally");
  }
  return std::move(output.samples);
}

}  // namespace gufo::core
