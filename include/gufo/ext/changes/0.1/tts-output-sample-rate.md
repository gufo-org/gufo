# TTS provider output sample rate

Speech synthesis (`src/ext/tts.hpp`): `TtsRunner::SampleRate()` declares the
runner's fixed output waveform sample rate in Hz (default `24000`, valid range
`[1, 192000]`). `SynthesisRequest::on_audio` delivers mono chunks at
`SampleRate()`, and `SynthesisResult::sample_rate` must match `SampleRate()`.
The announced rate is fixed for the whole stream, so a disagreeing result is
detectable only when synthesis completes; the service then fails the request
(reported as `invalid_audio`) instead of succeeding with misleading metadata.
The TTS service announces `SampleRate()` in WebSocket `audio.start` metadata
and computes streaming duration metrics from that rate. Additive; existing
24 kHz runners retain the default without changes.
