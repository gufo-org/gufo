#ifndef GUFO_AUDIO_DECODE_H_
#define GUFO_AUDIO_DECODE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The C interface of the Rust audio decoder and resampler. src/core/audio.hpp
   wraps it for C++; see there for the meaning of each status. */
typedef enum GufoAudioStatus {
  GUFO_AUDIO_OK = 0,
  GUFO_AUDIO_UNRECOGNIZED = 1,
  GUFO_AUDIO_MALFORMED_HEADER = 2,
  GUFO_AUDIO_UNSUPPORTED_CODEC = 3,
  GUFO_AUDIO_UNSUPPORTED_LAYOUT = 4,
  /* Also a resampling rate of 0, or one rate above 64 times the other. */
  GUFO_AUDIO_OUT_OF_BOUNDS = 5,
  GUFO_AUDIO_NO_AUDIO = 6,
  GUFO_AUDIO_TOO_LONG = 7,
  GUFO_AUDIO_TRUNCATED = 8,
  GUFO_AUDIO_NON_FINITE = 9,
  /* stop returned nonzero. */
  GUFO_AUDIO_STOPPED = 10,
  /* A caller bug (null pointer, unknown formats), a failing sink or a decoder
     bug. */
  GUFO_AUDIO_INTERNAL = 11,
} GufoAudioStatus;

/* GufoAudioLimits.formats: interleaved integer PCM or float32 WAV, read
   strictly. */
#define GUFO_AUDIO_FORMATS_PCM_WAV 0u
/* GufoAudioLimits.formats: every container and codec symphonia decodes. */
#define GUFO_AUDIO_FORMATS_ALL 1u

typedef struct GufoAudioLimits {
  uint32_t formats;
  uint32_t max_channels;
  uint32_t min_sample_rate;
  uint32_t max_sample_rate;
  uint64_t max_seconds;
  /* GUFO_AUDIO_FORMATS_PCM_WAV only: the widest integer PCM, 16, 24 or 32. */
  uint32_t max_pcm_bits;
} GufoAudioLimits;

/* While decoding, frames is the declared length when the container states one
   within the limits (0 otherwise), for reserving. When resampling, it is the
   output length. */
typedef struct GufoAudioInfo {
  uint32_t sample_rate;
  uint32_t channels;
  uint64_t frames;
} GufoAudioInfo;

/* Returns nonzero to stop decoding. */
typedef int (*GufoAudioStopFn)(void* context);
/* Takes the next interleaved samples, valid during the call only; returns
   nonzero if it could not. */
typedef int (*GufoAudioSinkFn)(void* context, const GufoAudioInfo* info,
                               const float* samples, size_t len);

/* Decodes bytes and hands the interleaved samples to sink in order, polling
   stop (which may be null) before each packet or block. On an error, the
   samples already handed over are to be discarded. */
GufoAudioStatus gufo_audio_decode(const uint8_t* bytes, size_t len,
                                  const GufoAudioLimits* limits,
                                  GufoAudioStopFn stop, GufoAudioSinkFn sink,
                                  void* context);

/* Resamples one channel and hands the whole result to sink in one call. */
GufoAudioStatus gufo_audio_resample(const float* samples, size_t len,
                                    uint32_t from_rate, uint32_t to_rate,
                                    GufoAudioSinkFn sink, void* context);

#ifdef __cplusplus
}
#endif

#endif /* GUFO_AUDIO_DECODE_H_ */
