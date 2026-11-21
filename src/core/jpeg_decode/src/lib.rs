#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod decode;
mod orientation;

use gufo_ffi::guard;

use decode::Error;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum GufoJpegStatus {
    Ok = 0,
    Invalid = 1,
    Internal = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoJpegInfo {
    pub width: u32,
    pub height: u32,
    pub orientation: u32,
}

fn status(result: Result<(), Error>) -> GufoJpegStatus {
    match result {
        Ok(()) => GufoJpegStatus::Ok,
        Err(Error::Invalid) => GufoJpegStatus::Invalid,
        Err(Error::Internal) => GufoJpegStatus::Internal,
    }
}

/// Opaque to C. Holds a JPEG opened by `gufo_jpeg_open` until it is decoded.
pub struct GufoJpegDecoder {
    jpeg: Option<decode::Jpeg<'static>>,
}

/// # Safety
/// `bytes` points to `len` readable bytes, or `len` is 0, and they stay
/// readable and unchanged until the decoder stored in `decoder` is closed.
/// `decoder` and `out` point to writable values.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_jpeg_open(
    bytes: *const u8,
    len: usize,
    decoder: *mut *mut GufoJpegDecoder,
    out: *mut GufoJpegInfo,
) -> GufoJpegStatus {
    guard(GufoJpegStatus::Internal, || {
        // SAFETY: the caller upholds the contract above; `as_mut` rejects nulls.
        let (Some(decoder), Some(out)) = (unsafe { decoder.as_mut() }, unsafe { out.as_mut() })
        else {
            return GufoJpegStatus::Internal;
        };
        *decoder = std::ptr::null_mut();
        // SAFETY: the caller keeps the bytes readable and unchanged until the
        // decoder that borrows them is closed, so `'static` never outlives them.
        let Some(bytes) = (unsafe { gufo_ffi::input(bytes, len) }) else {
            return GufoJpegStatus::Internal;
        };
        status(decode::open(bytes).map(|jpeg| {
            let (width, height) = jpeg.size();
            *out = GufoJpegInfo {
                width,
                height,
                orientation: orientation::of(bytes),
            };
            *decoder = Box::into_raw(Box::new(GufoJpegDecoder { jpeg: Some(jpeg) }));
        }))
    })
}

/// # Safety
/// `decoder` comes from `gufo_jpeg_open` and is not closed. `rgb` points to
/// `rgb_len` writable bytes, or `rgb_len` is 0, and does not overlap the JPEG
/// bytes.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_jpeg_decode_rgb8(
    decoder: *mut GufoJpegDecoder,
    rgb: *mut u8,
    rgb_len: usize,
) -> GufoJpegStatus {
    guard(GufoJpegStatus::Internal, || {
        // SAFETY: the caller upholds the contract above; `as_mut` rejects null.
        let Some(decoder) = (unsafe { decoder.as_mut() }) else {
            return GufoJpegStatus::Internal;
        };
        // A decoder decodes once.
        let Some(jpeg) = decoder.jpeg.take() else {
            return GufoJpegStatus::Internal;
        };
        // SAFETY: the caller supplies an exclusive writable output buffer.
        let Some(rgb) = (unsafe { gufo_ffi::output(rgb, rgb_len) }) else {
            return GufoJpegStatus::Internal;
        };
        status(jpeg.decode_rgb8(rgb))
    })
}

/// # Safety
/// `decoder` is null or comes from `gufo_jpeg_open` and is not yet closed.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_jpeg_close(decoder: *mut GufoJpegDecoder) {
    guard((), || {
        if !decoder.is_null() {
            // SAFETY: `gufo_jpeg_open` made `decoder` with `Box::into_raw`, and
            // the caller closes it once.
            drop(unsafe { Box::from_raw(decoder) });
        }
    });
}
