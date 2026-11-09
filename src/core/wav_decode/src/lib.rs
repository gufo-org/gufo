#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

#[cfg(panic = "abort")]
compile_error!("guarded() catches panics at the C boundary, so build with panic = \"unwind\"");

mod decode;

use std::panic::{AssertUnwindSafe, catch_unwind};

use decode::{Error, Limits};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum GufoWavStatus {
    Ok = 0,
    NotRiff = 1,
    Truncated = 2,
    InvalidFmt = 3,
    UnsupportedFormat = 4,
    UnsupportedLayout = 5,
    TooLong = 6,
    NonFinite = 7,
    Internal = 8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoWavLimits {
    pub max_channels: u32,
    pub max_sample_rate: u32,
    pub max_seconds: u64,
    pub max_pcm_bits: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoWavInfo {
    pub sample_rate: u32,
    pub channels: u32,
    pub frames: u64,
}

fn status(result: Result<(), Error>) -> GufoWavStatus {
    match result {
        Ok(()) => GufoWavStatus::Ok,
        Err(Error::NotRiff) => GufoWavStatus::NotRiff,
        Err(Error::Truncated) => GufoWavStatus::Truncated,
        Err(Error::InvalidFmt) => GufoWavStatus::InvalidFmt,
        Err(Error::UnsupportedFormat) => GufoWavStatus::UnsupportedFormat,
        Err(Error::UnsupportedLayout) => GufoWavStatus::UnsupportedLayout,
        Err(Error::TooLong) => GufoWavStatus::TooLong,
        Err(Error::NonFinite) => GufoWavStatus::NonFinite,
    }
}

/// A panic must not unwind into C++; the decoder has no known panics, so one
/// here is a decoder bug and reports `Internal`.
fn guarded(call: impl FnOnce() -> GufoWavStatus) -> GufoWavStatus {
    catch_unwind(AssertUnwindSafe(call)).unwrap_or(GufoWavStatus::Internal)
}

fn limits(limits: &GufoWavLimits) -> Limits {
    Limits {
        max_channels: limits.max_channels,
        max_sample_rate: limits.max_sample_rate,
        max_seconds: limits.max_seconds,
        max_pcm_bits: limits.max_pcm_bits,
    }
}

/// # Safety
/// `bytes` points to `len` readable bytes, or `len` is 0. `limits` points to a
/// readable value and `info` to a writable one.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_wav_read_info(
    bytes: *const u8,
    len: usize,
    limits: *const GufoWavLimits,
    info: *mut GufoWavInfo,
) -> GufoWavStatus {
    guarded(|| {
        // SAFETY: the caller upholds the contract above; `as_ref`/`as_mut`
        // reject nulls.
        let (Some(limits), Some(info)) = (unsafe { limits.as_ref() }, unsafe { info.as_mut() })
        else {
            return GufoWavStatus::Internal;
        };
        // SAFETY: the caller guarantees `len` readable bytes.
        let Some(bytes) = (unsafe { input(bytes, len) }) else {
            return GufoWavStatus::Internal;
        };
        status(decode::parse(bytes, &self::limits(limits)).map(|wav| {
            *info = GufoWavInfo {
                sample_rate: wav.sample_rate,
                channels: wav.channels,
                frames: wav.frames,
            };
        }))
    })
}

/// # Safety
/// `bytes` points to `len` readable bytes, or `len` is 0. `limits` points to a
/// readable value. `samples` points to `samples_len` writable floats, or
/// `samples_len` is 0, and does not overlap the WAV bytes.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_wav_decode_f32(
    bytes: *const u8,
    len: usize,
    limits: *const GufoWavLimits,
    samples: *mut f32,
    samples_len: usize,
) -> GufoWavStatus {
    guarded(|| {
        // SAFETY: the caller upholds the contract above; `as_ref` rejects null.
        let Some(limits) = (unsafe { limits.as_ref() }) else {
            return GufoWavStatus::Internal;
        };
        // SAFETY: the caller guarantees `len` readable bytes.
        let Some(bytes) = (unsafe { input(bytes, len) }) else {
            return GufoWavStatus::Internal;
        };
        let wav = match decode::parse(bytes, &self::limits(limits)) {
            Ok(wav) => wav,
            Err(error) => return status(Err(error)),
        };
        if wav.samples() != samples_len {
            return GufoWavStatus::Internal;
        }
        let out = match (samples.is_null(), samples_len) {
            (_, 0) => &mut [],
            // SAFETY: the caller upholds the contract above, and `samples` is
            // not null.
            (false, _) => unsafe { std::slice::from_raw_parts_mut(samples, samples_len) },
            (true, _) => return GufoWavStatus::Internal,
        };
        status(wav.decode(out))
    })
}

/// # Safety
/// `bytes` points to `len` readable bytes, or `len` is 0.
#[allow(unsafe_code)]
unsafe fn input<'a>(bytes: *const u8, len: usize) -> Option<&'a [u8]> {
    match (bytes.is_null(), len) {
        (_, 0) => Some(&[]),
        (true, _) => None,
        // SAFETY: the caller guarantees `len` readable bytes at a non-null `bytes`.
        (false, _) => Some(unsafe { std::slice::from_raw_parts(bytes, len) }),
    }
}
