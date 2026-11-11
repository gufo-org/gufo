#ifndef GUFO_CORE_AUDIO_HPP_
#define GUFO_CORE_AUDIO_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <vector>

namespace gufo::core {

// The inputs DecodeAudio accepts.
enum class AudioFormats : std::uint8_t {
  // RIFF WAV with interleaved integer PCM (16 bits up to max_pcm_bits) or
  // 32-bit float, read strictly: unknown chunks are skipped, the last fmt and
  // data chunks win, and a chunk past the end is an error. Floats are clamped
  // to [-1, 1]. The input contract of serve, Qwen3-ASR and Qwen3-TTS.
  kPcmWav,
  // WAV (integer PCM, float, A-law, mu-law, IMA and Microsoft ADPCM), FLAC,
  // MPEG audio layers I to III, Ogg Vorbis and Ogg Opus, decoded to floats as
  // libsndfile does. Ogg Opus decodes at the lowest Opus rate (8, 12, 16, 24
  // or 48 kHz) that is at least the header's input rate.
  kAll,
};

// What a caller accepts, and whether it wants one channel.
struct AudioLimits {
  AudioFormats formats{AudioFormats::kPcmWav};
  std::uint32_t max_channels{8};
  std::uint32_t min_sample_rate{1};
  std::uint32_t max_sample_rate{192000};
  std::uint64_t max_seconds{30};
  // kPcmWav only: the widest integer PCM accepted, 16, 24 or 32 bits.
  std::uint32_t max_pcm_bits{16};
  // Mix the channels down while decoding, block by block, to the samples
  // DownmixMono returns, without holding every channel of the stream.
  bool downmix_mono{false};
};

// Why DecodeAudio rejected an input.
enum class AudioStatus : std::uint8_t {
  // Not a container that `formats` reads, or no audio track in it.
  kUnrecognized,
  // A malformed format description, such as a WAV fmt chunk under 16 bytes.
  kMalformedHeader,
  // A codec, or WAV format tag, that `formats` does not decode.
  kUnsupportedCodec,
  // kPcmWav: a sample width other than PCM16 up to max_pcm_bits or float32,
  // or a block alignment that does not match it.
  kUnsupportedLayout,
  // Channels or sample rate outside the limits, or zero.
  kOutOfBounds,
  // No samples: a WAV without a data chunk, or a stream that decodes to none.
  kNoAudio,
  // Longer than max_seconds, as declared or as decoded.
  kTooLong,
  // The data ends early, a packet fails to decode, or fewer frames decode
  // than the container declares.
  kTruncated,
  // A NaN or infinite sample.
  kNonFinite,
  // should_stop returned true.
  kStopped,
};

// The exception DecodeAudio throws for every rejected input. what() describes
// the status in English; callers that keep their own wording map status().
class AudioError : public std::runtime_error {
public:
  explicit AudioError(AudioStatus status);
  [[nodiscard]] AudioStatus status() const noexcept { return status_; }

private:
  AudioStatus status_;
};

struct DecodedAudio {
  std::uint32_t sample_rate{0};
  // The stream's channels, or 1 with downmix_mono.
  std::uint32_t channels{0};
  // Interleaved float samples: frames * channels.
  std::vector<float> samples;
};

// Decodes `bytes` as one of limits.formats. An MPEG stream keeps the frames
// that decode; every other input must hold the frames it declares.
// should_stop is polled between packets (between blocks of frames for
// kPcmWav); an exception it throws propagates. Throws AudioError.
[[nodiscard]] DecodedAudio DecodeAudio(
    std::span<const std::byte> bytes, const AudioLimits& limits,
    const std::function<bool()>& should_stop = {});

// The mean of each frame's channels: a float sum in channel order divided by
// the channel count. Throws std::invalid_argument when channels is 0 or does
// not divide the sample count.
[[nodiscard]] std::vector<float> DownmixMono(std::span<const float> interleaved,
                                             std::uint32_t channels);

// Resamples one channel with the response of soxr's HQ preset: flat to 0.913
// of the lower Nyquist frequency, linear phase, no delay. Returns
// samples.size() * to_rate / from_rate samples rounded half up, as
// soxr_oneshot does; equal rates return a copy. Not interruptible: 30 s from
// 192 kHz to 16 kHz takes about 10 ms on Strix Halo. Throws
// std::invalid_argument when a rate is 0 or one rate exceeds 64 times the
// other.
[[nodiscard]] std::vector<float> Resample(std::span<const float> samples,
                                          std::uint32_t from_rate,
                                          std::uint32_t to_rate);

}  // namespace gufo::core

#endif  // GUFO_CORE_AUDIO_HPP_
