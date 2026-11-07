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

/// A panic must not unwind into C++. The decoder rejects the malformed input
/// known to panic in the png crate before decoding, so a panic here is a
/// decoder bug and reports `Internal`.
fn guarded(call: impl FnOnce() -> GufoPngStatus) -> GufoPngStatus {
    catch_unwind(AssertUnwindSafe(call)).unwrap_or(GufoPngStatus::Internal)
}

/// Opaque to C. Holds a PNG opened by `gufo_png_open` until it is decoded.
pub struct GufoPngDecoder {
    png: Option<decode::Png<'static>>,
}

/// # Safety
/// `bytes` points to `len` readable bytes, or `len` is 0, and they stay
/// readable and unchanged until the decoder stored in `decoder` is closed.
/// `decoder` and `out` point to writable values.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_png_open(
    bytes: *const u8,
    len: usize,
    decoder: *mut *mut GufoPngDecoder,
    out: *mut GufoPngDimensions,
) -> GufoPngStatus {
    guarded(|| {
        // SAFETY: the caller upholds the contract above; `as_mut` rejects nulls.
        let (Some(decoder), Some(out)) = (unsafe { decoder.as_mut() }, unsafe { out.as_mut() })
        else {
            return GufoPngStatus::Internal;
        };
        *decoder = std::ptr::null_mut();
        // SAFETY: the caller keeps the bytes readable and unchanged until the
        // decoder that borrows them is closed, so `'static` never outlives them.
        let Some(bytes) = (unsafe { input(bytes, len) }) else {
            return GufoPngStatus::Internal;
        };
        status(decode::open(bytes).map(|png| {
            let (width, height) = png.size();
            *out = GufoPngDimensions { width, height };
            *decoder = Box::into_raw(Box::new(GufoPngDecoder { png: Some(png) }));
        }))
    })
}

/// # Safety
/// `decoder` comes from `gufo_png_open` and is not closed. `rgb` points to
/// `rgb_len` writable bytes, or `rgb_len` is 0, and does not overlap the PNG
/// bytes.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_png_decode_rgb8(
    decoder: *mut GufoPngDecoder,
    rgb: *mut u8,
    rgb_len: usize,
) -> GufoPngStatus {
    guarded(|| {
        // SAFETY: the caller upholds the contract above; `as_mut` rejects null.
        let Some(decoder) = (unsafe { decoder.as_mut() }) else {
            return GufoPngStatus::Internal;
        };
        // A decoder decodes once.
        let Some(png) = decoder.png.take() else {
            return GufoPngStatus::Internal;
        };
        let rgb = match (rgb.is_null(), rgb_len) {
            (_, 0) => &mut [],
            // SAFETY: the caller upholds the contract above, and `rgb` is not null.
            (false, _) => unsafe { std::slice::from_raw_parts_mut(rgb, rgb_len) },
            (true, _) => return GufoPngStatus::Internal,
        };
        status(png.decode_rgb8(rgb))
    })
}

/// # Safety
/// `decoder` is null or comes from `gufo_png_open` and is not yet closed.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_png_close(decoder: *mut GufoPngDecoder) {
    if !decoder.is_null() {
        // SAFETY: `gufo_png_open` made `decoder` with `Box::into_raw`, and the
        // caller closes it once.
        drop(unsafe { Box::from_raw(decoder) });
    }
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
