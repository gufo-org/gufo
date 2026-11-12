# WAV decoding

Publish the shared WAV decoder (`src/core/wav.hpp`, reached through
`gufo/ext/wav.hpp`): `core::DecodeWav` decodes a RIFF WAV with interleaved
integer PCM or 32-bit float samples into `core::WavAudio` (sample rate,
channels, interleaved floats in [-1, 1]). `core::WavLimits` sets the accepted
channels, sample rate, duration and integer PCM width (16 up to 24 or 32 bits);
`core::WavMessages` gives the message of each rejection, thrown as
`std::length_error` for audio without frames or over the duration limit and as
`std::invalid_argument` otherwise. Additive.
