#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

#[cfg(panic = "abort")]
compile_error!("guarded() catches panics at the C boundary, so build with panic = \"unwind\"");

mod decode;

use std::panic::{AssertUnwindSafe, catch_unwind};

use decode::Error;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum GufoPngStatus {
    Ok = 0,
    Invalid = 1,
    Internal = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoPngDimensions {
    pub width: u32,
    pub height: u32,
}

fn status(result: Result<(), Error>) -> GufoPngStatus {
    match result {
        Ok(()) => GufoPngStatus::Ok,
        Err(Error::Invalid) => GufoPngStatus::Invalid,
        Err(Error::Internal) => GufoPngStatus::Internal,
    }
}

/// The png crate panics on some malformed input, such as a PLTE whose length
/// is not a multiple of 3. A panic reports an invalid image and must not unwind
/// into C++.
fn guarded(call: impl FnOnce() -> GufoPngStatus) -> GufoPngStatus {
    catch_unwind(AssertUnwindSafe(call)).unwrap_or(GufoPngStatus::Invalid)
}

/// # Safety
/// `bytes` points to `len` readable bytes, or `len` is 0. `out` points to a
/// writable `GufoPngDimensions`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_png_dimensions(
    bytes: *const u8,
    len: usize,
    out: *mut GufoPngDimensions,
) -> GufoPngStatus {
    guarded(|| {
        // SAFETY: the caller upholds the contract above; `as_mut` rejects a null `out`.
        let (Some(bytes), Some(out)) = (unsafe { input(bytes, len) }, unsafe { out.as_mut() })
        else {
            return GufoPngStatus::Internal;
        };
        status(decode::dimensions(bytes).map(|(width, height)| {
            *out = GufoPngDimensions { width, height };
        }))
    })
}

/// # Safety
/// `bytes` points to `len` readable bytes, or `len` is 0. `rgb` points to
/// `rgb_len` writable bytes, or `rgb_len` is 0. The two ranges do not overlap.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_png_decode_rgb8(
    bytes: *const u8,
    len: usize,
    rgb: *mut u8,
    rgb_len: usize,
) -> GufoPngStatus {
    guarded(|| {
        let rgb = match (rgb.is_null(), rgb_len) {
            (_, 0) => &mut [],
            // SAFETY: the caller upholds the contract above, and `rgb` is not null.
            (false, _) => unsafe { std::slice::from_raw_parts_mut(rgb, rgb_len) },
            (true, _) => return GufoPngStatus::Internal,
        };
        // SAFETY: the caller upholds the contract above.
        let Some(bytes) = (unsafe { input(bytes, len) }) else {
            return GufoPngStatus::Internal;
        };
        status(decode::decode_rgb8(bytes, rgb))
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
