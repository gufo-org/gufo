# Decoded transcription uploads

ASR (`src/ext/asr.hpp`):
- `POST /v1/audio/transcriptions` decodes the uploaded file in serve and calls
  `Transcribe` with `TranscriptionRequest::pcm16k` (16 kHz mono) instead of
  `wav`. Every provider now receives WAV, FLAC, MP3, Ogg Vorbis, Ogg Opus and
  WebM uploads; undecodable files get `400 invalid_audio_input` before the
  provider runs. Runners must accept `pcm16k`, as live sessions already
  require; `wav` remains for direct callers.

Audio (`src/core/audio.hpp`):
- `AudioFormats::kAll` also reads WebM (Matroska) Vorbis and Opus. WebM keeps
  the encoder's end padding, which its DiscardPadding marks.
