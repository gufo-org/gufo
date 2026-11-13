#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod codec;

use gufo_ffi::guard;

use codec::Error;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum GufoBase64Status {
    Ok = 0,
    Length = 1,
    Padding = 2,
    AfterPadding = 3,
    Character = 4,
    NonCanonical = 5,
    TooLong = 6,
    Internal = 7,
}

fn status(error: Error) -> GufoBase64Status {
    match error {
        Error::Length => GufoBase64Status::Length,
        Error::Padding => GufoBase64Status::Padding,
        Error::AfterPadding => GufoBase64Status::AfterPadding,
        Error::Character => GufoBase64Status::Character,
        Error::NonCanonical => GufoBase64Status::NonCanonical,
        Error::TooLong => GufoBase64Status::TooLong,
    }
}

/// Encodes `len` bytes into `out_len` characters, which must be
/// `ceil(len / 3) * 4`.
///
/// # Safety
/// `bytes` points to `len` readable bytes, or `len` is 0. `out` points to
/// `out_len` writable bytes that do not overlap them, or `out_len` is 0.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_base64_encode(
    bytes: *const u8,
    len: usize,
    out: *mut u8,
    out_len: usize,
) -> GufoBase64Status {
    guard(GufoBase64Status::Internal, || {
        if codec::encoded_len(len) != Some(out_len) {
            return GufoBase64Status::Internal;
        }
        // SAFETY: the caller guarantees `len` readable bytes.
        let Some(bytes) = (unsafe { gufo_ffi::input(bytes, len) }) else {
            return GufoBase64Status::Internal;
        };
        // SAFETY: the caller guarantees `out_len` writable bytes apart from
        // `bytes`.
        let Some(out) = (unsafe { gufo_ffi::output(out, out_len) }) else {
            return GufoBase64Status::Internal;
        };
        codec::encode(bytes, out);
        GufoBase64Status::Ok
    })
}

/// Decodes `len` characters into `out`, which holds at least `len / 4 * 3`
/// bytes, and stores the decoded length in `*written` (0 on an error).
///
/// # Safety
/// `text` points to `len` readable bytes, or `len` is 0. `out` points to
/// `out_len` writable bytes that do not overlap them, or `out_len` is 0.
/// `written` points to a writable value.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_base64_decode(
    text: *const u8,
    len: usize,
    canonical: bool,
    max_decoded: usize,
    out: *mut u8,
    out_len: usize,
    written: *mut usize,
) -> GufoBase64Status {
    guard(GufoBase64Status::Internal, || {
        // SAFETY: the caller upholds the contract above; `as_mut` rejects null.
        let Some(written) = (unsafe { written.as_mut() }) else {
            return GufoBase64Status::Internal;
        };
        *written = 0;
        if out_len < codec::decoded_capacity(len) {
            return GufoBase64Status::Internal;
        }
        // SAFETY: the caller guarantees `len` readable bytes.
        let Some(text) = (unsafe { gufo_ffi::input(text, len) }) else {
            return GufoBase64Status::Internal;
        };
        // SAFETY: the caller guarantees `out_len` writable bytes apart from
        // `text`.
        let Some(out) = (unsafe { gufo_ffi::output(out, out_len) }) else {
            return GufoBase64Status::Internal;
        };
        match codec::decode(text, canonical, max_decoded, out) {
            Ok(len) => {
                *written = len;
                GufoBase64Status::Ok
            }
            Err(error) => status(error),
        }
    })
}
