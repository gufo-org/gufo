#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

#[cfg(panic = "abort")]
compile_error!("guarded() catches panics at the C boundary, so build with panic = \"unwind\"");

mod decode;
mod opus;
mod pcm;

use std::ffi::{c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};

use decode::{Error, Formats, Info, Limits};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum GufoAudioStatus {
    Ok = 0,
    Unrecognized = 1,
    MalformedHeader = 2,
    UnsupportedCodec = 3,
    UnsupportedLayout = 4,
    OutOfBounds = 5,
    NoAudio = 6,
    TooLong = 7,
    Truncated = 8,
    NonFinite = 9,
    Stopped = 10,
    Internal = 11,
}

/// `GufoAudioLimits::formats`: the strict WAV reader.
pub const GUFO_AUDIO_FORMATS_PCM_WAV: u32 = 0;
/// `GufoAudioLimits::formats`: every container and codec symphonia decodes.
pub const GUFO_AUDIO_FORMATS_ALL: u32 = 1;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoAudioLimits {
    pub formats: u32,
    pub max_channels: u32,
    pub min_sample_rate: u32,
    pub max_sample_rate: u32,
    pub max_seconds: u64,
    pub max_pcm_bits: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoAudioInfo {
    pub sample_rate: u32,
    pub channels: u32,
    pub frames: u64,
}

/// Returns nonzero to stop decoding.
pub type GufoAudioStopFn = unsafe extern "C" fn(context: *mut c_void) -> c_int;
/// Takes the next interleaved samples, valid during the call only, of the
/// stream `info` describes; returns nonzero if it could not.
pub type GufoAudioSinkFn = unsafe extern "C" fn(
    context: *mut c_void,
    info: *const GufoAudioInfo,
    samples: *const f32,
    len: usize,
) -> c_int;

fn status(error: Error) -> GufoAudioStatus {
    match error {
        Error::Unrecognized => GufoAudioStatus::Unrecognized,
        Error::MalformedHeader => GufoAudioStatus::MalformedHeader,
        Error::UnsupportedCodec => GufoAudioStatus::UnsupportedCodec,
        Error::UnsupportedLayout => GufoAudioStatus::UnsupportedLayout,
        Error::OutOfBounds => GufoAudioStatus::OutOfBounds,
        Error::NoAudio => GufoAudioStatus::NoAudio,
        Error::TooLong => GufoAudioStatus::TooLong,
        Error::Truncated => GufoAudioStatus::Truncated,
        Error::NonFinite => GufoAudioStatus::NonFinite,
        Error::Stopped => GufoAudioStatus::Stopped,
    }
}

fn c_info(info: &Info) -> GufoAudioInfo {
    GufoAudioInfo {
        sample_rate: info.sample_rate,
        channels: info.channels,
        frames: info.frames,
    }
}

/// A panic must not unwind into C++; the decoders have no known panics, so
/// one here is a bug and reports `Internal`.
fn guarded(call: impl FnOnce() -> GufoAudioStatus) -> GufoAudioStatus {
    catch_unwind(AssertUnwindSafe(call)).unwrap_or(GufoAudioStatus::Internal)
}

/// # Safety
/// `data` points to `len` readable values, or `len` is 0.
#[allow(unsafe_code)]
unsafe fn input<'a, T>(data: *const T, len: usize) -> Option<&'a [T]> {
    match (data.is_null(), len) {
        (_, 0) => Some(&[]),
        (true, _) => None,
        // SAFETY: the caller guarantees `len` readable values at a non-null
        // `data`.
        (false, _) => Some(unsafe { std::slice::from_raw_parts(data, len) }),
    }
}

/// Decodes `bytes` and hands the interleaved samples to `sink` in order.
/// On an error, the samples already handed over are to be discarded.
///
/// # Safety
/// `bytes` points to `len` readable bytes, or `len` is 0. `limits` points to
/// a readable value. `stop`, if not null, and `sink` may be called with
/// `context` from this thread until this function returns.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_audio_decode(
    bytes: *const u8,
    len: usize,
    limits: *const GufoAudioLimits,
    stop: Option<GufoAudioStopFn>,
    sink: Option<GufoAudioSinkFn>,
    context: *mut c_void,
) -> GufoAudioStatus {
    guarded(|| {
        // SAFETY: the caller upholds the contract above; `as_ref` rejects null.
        let (Some(limits), Some(sink)) = (unsafe { limits.as_ref() }, sink) else {
            return GufoAudioStatus::Internal;
        };
        // SAFETY: the caller guarantees `len` readable bytes.
        let Some(bytes) = (unsafe { input(bytes, len) }) else {
            return GufoAudioStatus::Internal;
        };
        let formats = match limits.formats {
            GUFO_AUDIO_FORMATS_PCM_WAV => Formats::PcmWav,
            GUFO_AUDIO_FORMATS_ALL => Formats::All,
            _ => return GufoAudioStatus::Internal,
        };
        let limits = Limits {
            formats,
            max_channels: limits.max_channels,
            min_sample_rate: limits.min_sample_rate,
            max_sample_rate: limits.max_sample_rate,
            max_seconds: limits.max_seconds,
            max_pcm_bits: limits.max_pcm_bits,
        };
        // SAFETY: the caller allows calling `stop` with `context` here.
        let mut stop = || stop.is_some_and(|stop| unsafe { stop(context) } != 0);
        let mut refused = false;
        let mut sink = |info: &Info, samples: &[f32]| {
            let info = c_info(info);
            // SAFETY: the caller allows calling `sink` with `context` here;
            // `info` and `samples` stay valid during the call.
            refused = unsafe { sink(context, &info, samples.as_ptr(), samples.len()) } != 0;
            !refused
        };
        match decode::decode(bytes, &limits, &mut stop, &mut sink) {
            Ok(_) => GufoAudioStatus::Ok,
            Err(_) if refused => GufoAudioStatus::Internal,
            Err(error) => status(error),
        }
    })
}

/// Resamples `len` samples of one channel from `from_rate` to `to_rate` and
/// hands the result to `sink` in one call, with `frames` the output length.
/// Returns `OutOfBounds` when a rate is 0 or one rate exceeds
/// `gufo_resample::MAX_RATIO` times the other.
///
/// # Safety
/// `samples` points to `len` readable floats, or `len` is 0. `sink` may be
/// called with `context` from this thread until this function returns.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_audio_resample(
    samples: *const f32,
    len: usize,
    from_rate: u32,
    to_rate: u32,
    sink: Option<GufoAudioSinkFn>,
    context: *mut c_void,
) -> GufoAudioStatus {
    guarded(|| {
        let Some(sink) = sink else {
            return GufoAudioStatus::Internal;
        };
        // SAFETY: the caller guarantees `len` readable floats.
        let Some(samples) = (unsafe { input(samples, len) }) else {
            return GufoAudioStatus::Internal;
        };
        let Ok(output) = gufo_resample::resample(samples, from_rate, to_rate) else {
            return GufoAudioStatus::OutOfBounds;
        };
        let info = GufoAudioInfo {
            sample_rate: to_rate,
            channels: 1,
            frames: output.len() as u64,
        };
        // SAFETY: the caller allows calling `sink` with `context` here;
        // `info` and `output` stay valid during the call.
        match unsafe { sink(context, &info, output.as_ptr(), output.len()) } {
            0 => GufoAudioStatus::Ok,
            _ => GufoAudioStatus::Internal,
        }
    })
}
