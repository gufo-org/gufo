# One audio decoding surface

Breaking: series 0.3 starts here, and an extension must request `>= 0.3.0`.
Audio decoding has one public surface, `gufo/ext/audio.hpp`
(`src/core/audio.hpp`):

- `core::DecodeAudio` decodes bytes into `core::DecodedAudio`: the sample
  rate, the channel count and interleaved float samples. `core::AudioLimits`
  bounds channels, sample rate and duration, and its `formats` selects the
  reader. `AudioFormats::kPcmWav` is the strict RIFF WAV reader: interleaved
  integer PCM from 16 bits up to `max_pcm_bits`, or 32-bit float clamped to
  [-1, 1]. `AudioFormats::kAll` decodes WAV (integer PCM, float, A-law,
  mu-law, IMA and Microsoft ADPCM), FLAC, MPEG audio layers I to III, Ogg
  Vorbis and Ogg Opus to floats as libsndfile does. An optional
  `should_stop` is polled between packets.
- Every rejected input throws `core::AudioError`. Its `status()` is a
  `core::AudioStatus`, which callers map to their own messages.
- `core::DownmixMono` averages interleaved channels. With
  `AudioLimits::downmix_mono`, `DecodeAudio` returns the same mean, mixed
  block by block while decoding, without holding every channel of the
  stream. `core::Resample` resamples one channel with the response of soxr's
  HQ preset and returns `soxr_oneshot`'s rounded length.

Removed: `gufo/ext/wav.hpp` with `core::DecodeWav`, `core::WavLimits`,
`core::WavMessages` and `core::WavAudio`. A caller of `DecodeWav` passes
`AudioLimits` with `AudioFormats::kPcmWav` and its channel, rate, duration and
PCM-width limits, and maps `AudioError::status()` to its messages; the decoded
samples are identical.
