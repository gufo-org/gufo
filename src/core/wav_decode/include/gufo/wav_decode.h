#ifndef GUFO_WAV_DECODE_H_
#define GUFO_WAV_DECODE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum GufoWavStatus {
  GUFO_WAV_OK = 0,
  /* Shorter than 44 bytes, or no RIFF/WAVE header. */
  GUFO_WAV_NOT_RIFF = 1,
  /* A chunk claims more bytes than follow its header. */
  GUFO_WAV_TRUNCATED = 2,
  /* A fmt chunk is shorter than 16 bytes. */
  GUFO_WAV_INVALID_FMT = 3,
  /* Neither integer PCM nor IEEE float, no channels or more than
     max_channels, a sample rate of 0 or above max_sample_rate, or no data. */
  GUFO_WAV_UNSUPPORTED_FORMAT = 4,
  /* A sample width the limits do not accept, or a block alignment that is not
     one interleaved frame or does not divide the data. */
  GUFO_WAV_UNSUPPORTED_LAYOUT = 5,
  /* No frames, or more than max_seconds of them. */
  GUFO_WAV_TOO_LONG = 6,
  /* A float sample is NaN or infinite. */
  GUFO_WAV_NON_FINITE = 7,
  /* A caller bug (null pointer, wrong output length) or a decoder bug. */
  GUFO_WAV_INTERNAL = 8,
} GufoWavStatus;

/* What the caller accepts. Integer PCM is decoded from 16 bits up to
   max_pcm_bits (16, 24 or 32); 32-bit float always. */
typedef struct GufoWavLimits {
  uint32_t max_channels;
  uint32_t max_sample_rate;
  uint64_t max_seconds;
  uint32_t max_pcm_bits;
} GufoWavLimits;

typedef struct GufoWavInfo {
  uint32_t sample_rate;
  uint32_t channels;
  uint64_t frames;
} GufoWavInfo;

/* Reads the chunks and checks the format against *limits. Unknown chunks are
   skipped and the last fmt and data chunks win. On GUFO_WAV_OK, *info holds
   the format; otherwise it is unchanged. */
GufoWavStatus gufo_wav_read_info(const uint8_t* bytes, size_t len,
                                 const GufoWavLimits* limits,
                                 GufoWavInfo* info);

/* Decodes the interleaved samples to floats in [-1, 1]: integer PCM divided by
   2^(bits - 1), floats clamped. samples_len must be frames * channels from
   gufo_wav_read_info with the same bytes and limits, and samples must not
   overlap the bytes. The output is unspecified unless the result is
   GUFO_WAV_OK. */
GufoWavStatus gufo_wav_decode_f32(const uint8_t* bytes, size_t len,
                                  const GufoWavLimits* limits, float* samples,
                                  size_t samples_len);

#ifdef __cplusplus
}
#endif

#endif
