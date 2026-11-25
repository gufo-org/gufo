#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod decode;
mod encode;
mod rgba;

use std::ffi::c_void;

use gufo_ffi::guard;

use decode::{Error, Format};

/// `GUFO_PNG_RGB8` and `GUFO_PNG_RGBA8`.
const FORMAT_RGB8: u32 = 0;
const FORMAT_RGBA8: u32 = 1;

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

/// Opaque to C. Holds a PNG opened by `gufo_png_open` until it is decoded.
pub struct GufoPngDecoder {
    png: Option<decode::Png<'static>>,
}

/// Receives the encoded PNG, borrowed for the call.
pub type GufoPngBytesFn = unsafe extern "C" fn(context: *mut c_void, data: *const u8, len: usize);

/// # Safety
/// `bytes` points to `len` readable bytes, or `len` is 0, and they stay
/// readable and unchanged until the decoder stored in `decoder` is closed.
/// `decoder` and `out` point to writable values.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_png_open(
    bytes: *const u8,
    len: usize,
    format: u32,
    decoder: *mut *mut GufoPngDecoder,
    out: *mut GufoPngDimensions,
) -> GufoPngStatus {
    guard(GufoPngStatus::Internal, || {
        // SAFETY: the caller upholds the contract above; `as_mut` rejects nulls.
        let (Some(decoder), Some(out)) = (unsafe { decoder.as_mut() }, unsafe { out.as_mut() })
        else {
            return GufoPngStatus::Internal;
        };
        *decoder = std::ptr::null_mut();
        // SAFETY: the caller keeps the bytes readable and unchanged until the
        // decoder that borrows them is closed, so `'static` never outlives them.
        let Some(bytes) = (unsafe { gufo_ffi::input(bytes, len) }) else {
            return GufoPngStatus::Internal;
        };
        let format = match format {
            FORMAT_RGB8 => Format::Rgb8,
            FORMAT_RGBA8 => Format::Rgba8,
            _ => return GufoPngStatus::Internal,
        };
        status(decode::open(bytes, format).map(|png| {
            let (width, height) = png.size();
            *out = GufoPngDimensions { width, height };
            *decoder = Box::into_raw(Box::new(GufoPngDecoder { png: Some(png) }));
        }))
    })
}

/// # Safety
/// `decoder` comes from `gufo_png_open` and is not closed. `pixels` points to
/// `len` writable bytes, or `len` is 0, and does not overlap the PNG bytes.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_png_decode(
    decoder: *mut GufoPngDecoder,
    pixels: *mut u8,
    len: usize,
) -> GufoPngStatus {
    guard(GufoPngStatus::Internal, || {
        // SAFETY: the caller upholds the contract above; `as_mut` rejects null.
        let Some(decoder) = (unsafe { decoder.as_mut() }) else {
            return GufoPngStatus::Internal;
        };
        // A decoder decodes once.
        let Some(png) = decoder.png.take() else {
            return GufoPngStatus::Internal;
        };
        // SAFETY: the caller supplies an exclusive writable output buffer.
        let Some(pixels) = (unsafe { gufo_ffi::output(pixels, len) }) else {
            return GufoPngStatus::Internal;
        };
        status(png.decode(pixels))
    })
}

/// # Safety
/// `decoder` is null or comes from `gufo_png_open` and is not yet closed.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_png_close(decoder: *mut GufoPngDecoder) {
    guard((), || {
        if !decoder.is_null() {
            // SAFETY: `gufo_png_open` made `decoder` with `Box::into_raw`, and the
            // caller closes it once.
            drop(unsafe { Box::from_raw(decoder) });
        }
    });
}

/// # Safety
/// `rgba` points to `len` readable bytes, or `len` is 0. `bytes` reads the
/// encoded PNG only during the call and does not unwind.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_png_encode_rgba8(
    rgba: *const u8,
    len: usize,
    width: u32,
    height: u32,
    bytes: Option<GufoPngBytesFn>,
    context: *mut c_void,
) -> GufoPngStatus {
    guard(GufoPngStatus::Internal, || {
        // SAFETY: the caller supplies `len` readable bytes.
        let (Some(rgba), Some(bytes)) = (unsafe { gufo_ffi::input(rgba, len) }, bytes) else {
            return GufoPngStatus::Internal;
        };
        let png = match encode::encode_rgba8(rgba, width, height) {
            Ok(png) => png,
            Err(error) => return status(Err(error)),
        };
        // SAFETY: `png` outlives the call, and the callback follows the
        // contract above.
        unsafe { bytes(context, png.as_ptr(), png.len()) };
        GufoPngStatus::Ok
    })
}
