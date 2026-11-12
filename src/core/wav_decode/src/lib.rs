//! The strict WAV reader behind `gufo_audio_decode` for callers that accept
//! only interleaved integer PCM or 32-bit float WAV (serve, Qwen3-ASR and
//! Qwen3-TTS). It has no C interface of its own.

#![forbid(unsafe_code)]

mod decode;

pub use decode::{Error, Limits, Wav, parse};
