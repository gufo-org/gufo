# TTS conditioning capability descriptor

Speech synthesis (`src/ext/tts.hpp`): `TtsConditioningMode`,
`TtsConditioningCapabilities`, `DefaultTtsConditioningCapabilities(TtsVariant)`
and `TtsRunner::Conditioning()` declare whether a TTS runner supports
unconditioned synthesis, required vs optional `voice` selection, a provider
`default_voice`, per-request `reference_audio` cloning, `--voice` reference
presets, and optional vs required `instructions`. The TTS service validates
startup presets, HTTP `/v1/audio/speech` requests and WebSocket sessions
against `Conditioning()`, allowing a single checkpoint to support both
unconditioned default synthesis and reference-audio voice cloning while
preserving existing `TtsVariant` behavior by default. Additive.
