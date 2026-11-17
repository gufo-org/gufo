# Invalid ASR audio

ASR (`src/ext/asr.hpp`):
- `TranscriptionResult::invalid_audio` is set by a failed `Transcribe` when
  the request's audio could not be decoded. `POST /v1/audio/transcriptions`
  then answers `400 invalid_audio_input` (an `invalid_audio_input` error event
  when streaming) instead of `500 transcription_failed`.

The member defaults to false, so existing runners keep compiling and their
failures stay server errors. Runners must be rebuilt against the new header.
