//! The safetensors header reader and its C interface, which
//! `gufo::safetensors` (src/core/safetensors.cpp) wraps for the model
//! loaders. Only the header is read; the tensor data stays in the caller's
//! file or mapping.
#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod dtype;
mod entry;
mod header;
mod layout;

pub use dtype::Dtype;
pub use entry::{Entry, MAX_DIMENSION, MAX_RANK};
pub use header::{Failure, Header, METADATA, Rejection, parse};
pub use layout::{LENGTH_BYTES, MAX_HEADER_BYTES, header_length};

use gufo_ffi::guard;
use std::ffi::{c_int, c_void};

/// Why the reader rejects a file: it breaks the format, or one of Gufo's
/// limits, the strictest of the loaders the reader replaced (a header of at
/// most 32 MiB with at least one tensor, at most 5 dimensions, each at most
/// 2^31 - 1). The values are those of `GufoSafetensorsReason`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// The file is shorter than the length prefix.
    File = 1,
    /// The header length is 0, above `MAX_HEADER_BYTES` (a Gufo limit) or
    /// past the file.
    Length = 2,
    /// The header is not JSON.
    Json = 3,
    /// The header is not a JSON object.
    NotObject = 4,
    /// `__metadata__` is neither null nor an object of strings.
    Metadata = 5,
    /// A tensor entry is not an object.
    Entry = 6,
    /// A type is missing, not a string or not a type the format names.
    Dtype = 7,
    /// `shape` or `data_offsets` is missing or not an array, the shape has
    /// more than `MAX_RANK` dimensions (a Gufo limit) or the offsets are not
    /// two.
    Layout = 8,
    /// A dimension is not an integer from 0 to `MAX_DIMENSION` (a Gufo
    /// limit: the format allows larger), the shape holds more bytes than the
    /// payload, or, without elements, its dimensions multiplied in order
    /// overflow 64 bits before the first 0.
    Dimension = 9,
    /// An offset is not an integer, or the offsets do not span exactly the
    /// tensor's bytes within the payload.
    Extent = 10,
    /// Two tensors share bytes, or a tensor without elements lies inside
    /// another's bytes.
    Overlap = 11,
    /// Payload bytes before a tensor belong to none.
    Gap = 12,
    /// Payload bytes after the last tensor belong to none.
    Trailing = 13,
    /// The header holds no tensor.
    Empty = 14,
}

/// A fault: why, and what is wrong in words, for the rejection's message.
pub type Problem = (Reason, &'static str);

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GufoSafetensorsParse {
    Complete = 0,
    Rejected = 1,
    NoMemory = 2,
    /// A callback returned nonzero.
    Stopped = 3,
    Internal = 4,
}

/// Bytes valid during a callback only.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoSafetensorsSlice {
    pub data: *const u8,
    pub len: usize,
}

impl GufoSafetensorsSlice {
    fn of(bytes: &[u8]) -> Self {
        Self {
            data: bytes.as_ptr(),
            len: bytes.len(),
        }
    }
}

/// A rejected file: why, the tensor or metadata key that fails (empty for
/// the file or the header as a whole) and what is wrong, in words. Callers
/// put "invalid safetensors header: " before the name and message, or
/// "invalid safetensors file: " for `File`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoSafetensorsRejection {
    pub reason: Reason,
    pub name: GufoSafetensorsSlice,
    pub message: GufoSafetensorsSlice,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoSafetensorsMetadata {
    pub key: GufoSafetensorsSlice,
    pub value: GufoSafetensorsSlice,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoSafetensorsTensor {
    pub name: GufoSafetensorsSlice,
    pub dtype: Dtype,
    pub shape: *const u64,
    pub rank: usize,
    pub begin: u64,
    pub end: u64,
}

/// Takes the rejection.
pub type GufoSafetensorsRejectFn =
    unsafe extern "C" fn(context: *mut c_void, rejection: *const GufoSafetensorsRejection);
/// Takes one metadata entry; returns nonzero to stop.
pub type GufoSafetensorsMetadataFn =
    unsafe extern "C" fn(context: *mut c_void, entry: *const GufoSafetensorsMetadata) -> c_int;
/// Takes one tensor; returns nonzero to stop.
pub type GufoSafetensorsTensorFn =
    unsafe extern "C" fn(context: *mut c_void, tensor: *const GufoSafetensorsTensor) -> c_int;

/// Passes `rejection` to `reject`, when there is one.
#[allow(unsafe_code)]
fn report(reject: Option<GufoSafetensorsRejectFn>, context: *mut c_void, rejection: &Rejection) {
    let Some(reject) = reject else {
        return;
    };
    let rejection = GufoSafetensorsRejection {
        reason: rejection.reason,
        name: GufoSafetensorsSlice::of(rejection.name.as_bytes()),
        message: GufoSafetensorsSlice::of(rejection.message.as_bytes()),
    };
    // SAFETY: the caller allows calling `reject` with `context`;
    // `rejection` and the bytes it points to live until it returns.
    unsafe { reject(context, &rejection) };
}

/// The spelling of the type whose value is `dtype`, such as "BF16"; empty
/// for a value that names no type. The bytes live as long as the program.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn gufo_safetensors_dtype_name(dtype: u32) -> GufoSafetensorsSlice {
    GufoSafetensorsSlice::of(Dtype::from_value(dtype).map_or("", Dtype::name).as_bytes())
}

/// The size in bits of one element of the type whose value is `dtype`; 0
/// for a value that names no type.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn gufo_safetensors_dtype_bits(dtype: u32) -> u32 {
    // Every size is at most 64 bits.
    Dtype::from_value(dtype).map_or(0, |dtype| dtype.bits() as u32)
}

/// Reads the header length from the first `len` bytes of a file of
/// `file_size` bytes and checks it against the file. On `Rejected`,
/// `reject` (which may be null) got the reason.
///
/// # Safety
/// `prefix` points to `len` readable bytes, or `len` is 0; `header_len`
/// points to a writable value; `reject` may be called with `context` until
/// this returns.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_safetensors_header_length(
    prefix: *const u8,
    len: usize,
    file_size: u64,
    reject: Option<GufoSafetensorsRejectFn>,
    context: *mut c_void,
    header_len: *mut u64,
) -> GufoSafetensorsParse {
    guard(GufoSafetensorsParse::Internal, || {
        // SAFETY: the caller guarantees `len` readable bytes.
        let Some(prefix) = (unsafe { gufo_ffi::input(prefix, len) }) else {
            return GufoSafetensorsParse::Internal;
        };
        // SAFETY: the caller passes a writable `header_len`; `as_mut`
        // rejects null.
        let Some(out) = (unsafe { header_len.as_mut() }) else {
            return GufoSafetensorsParse::Internal;
        };
        match header_length(prefix, file_size) {
            Ok(length) => {
                *out = length;
                GufoSafetensorsParse::Complete
            }
            Err((reason, message)) => {
                let rejection = Rejection {
                    reason,
                    name: String::new(),
                    message: message.to_owned(),
                };
                report(reject, context, &rejection);
                GufoSafetensorsParse::Rejected
            }
        }
    })
}

/// Checks the header in `header[0, len)`, which follows the length prefix
/// of a file of `file_size` bytes, then passes its `__metadata__` entries to
/// `metadata` and its tensors to `tensor`, both in header order. Offsets are
/// in the payload, which starts `8 + len` bytes into the file. On
/// `Rejected`, `reject` (which may be null) got the reason, and no entry was
/// passed.
///
/// # Safety
/// `header` points to `len` readable bytes that stay unchanged during the
/// call, or `len` is 0; each callback may be called with `context` until
/// this returns.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_safetensors_parse_header(
    header: *const u8,
    len: usize,
    file_size: u64,
    metadata: Option<GufoSafetensorsMetadataFn>,
    tensor: Option<GufoSafetensorsTensorFn>,
    reject: Option<GufoSafetensorsRejectFn>,
    context: *mut c_void,
) -> GufoSafetensorsParse {
    guard(GufoSafetensorsParse::Internal, || {
        // SAFETY: the caller guarantees `len` readable bytes.
        let Some(bytes) = (unsafe { gufo_ffi::input(header, len) }) else {
            return GufoSafetensorsParse::Internal;
        };
        let (Some(metadata), Some(tensor)) = (metadata, tensor) else {
            return GufoSafetensorsParse::Internal;
        };
        let parsed = match parse(bytes, file_size) {
            Ok(parsed) => parsed,
            Err(Failure::Rejected(rejection)) => {
                report(reject, context, &rejection);
                return GufoSafetensorsParse::Rejected;
            }
            Err(Failure::NoMemory) => return GufoSafetensorsParse::NoMemory,
            Err(Failure::Internal) => return GufoSafetensorsParse::Internal,
        };
        for (key, value) in parsed.metadata() {
            let entry = GufoSafetensorsMetadata {
                key: GufoSafetensorsSlice::of(key.as_bytes()),
                value: GufoSafetensorsSlice::of(value.as_bytes()),
            };
            // SAFETY: the caller allows this call; `entry` and the bytes it
            // points to live until it returns.
            if unsafe { metadata(context, &entry) } != 0 {
                return GufoSafetensorsParse::Stopped;
            }
        }
        for (name, entry) in parsed.tensors() {
            let shape = entry.shape();
            let parsed = GufoSafetensorsTensor {
                name: GufoSafetensorsSlice::of(name.as_bytes()),
                dtype: entry.dtype,
                shape: shape.as_ptr(),
                rank: shape.len(),
                begin: entry.begin,
                end: entry.end,
            };
            // SAFETY: the caller allows this call; `parsed` and what it
            // points to live until it returns.
            if unsafe { tensor(context, &parsed) } != 0 {
                return GufoSafetensorsParse::Stopped;
            }
        }
        GufoSafetensorsParse::Complete
    })
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;

    type SeenTensor = (Vec<u8>, Dtype, Vec<u64>, u64, u64);

    #[derive(Default)]
    struct Seen {
        metadata: Vec<(Vec<u8>, Vec<u8>)>,
        tensors: Vec<SeenTensor>,
        rejection: Option<(Reason, Vec<u8>, Vec<u8>)>,
        stop_after: Option<usize>,
    }

    fn seen<'a>(context: *mut c_void) -> &'a mut Seen {
        // SAFETY: every test passes a live `Seen` as the context.
        unsafe { &mut *context.cast::<Seen>() }
    }

    fn bytes(slice: GufoSafetensorsSlice) -> Vec<u8> {
        // SAFETY: the reader passes slices valid during the callback.
        unsafe { gufo_ffi::input(slice.data, slice.len) }
            .unwrap()
            .to_vec()
    }

    unsafe extern "C" fn take_metadata(
        context: *mut c_void,
        entry: *const GufoSafetensorsMetadata,
    ) -> c_int {
        // SAFETY: the reader passes a live entry.
        let entry = unsafe { &*entry };
        seen(context)
            .metadata
            .push((bytes(entry.key), bytes(entry.value)));
        0
    }

    unsafe extern "C" fn take_tensor(
        context: *mut c_void,
        tensor: *const GufoSafetensorsTensor,
    ) -> c_int {
        // SAFETY: the reader passes a live tensor.
        let tensor = unsafe { &*tensor };
        // SAFETY: `shape` holds `rank` dimensions during the callback.
        let shape = unsafe { gufo_ffi::input(tensor.shape, tensor.rank) }
            .unwrap()
            .to_vec();
        let seen = seen(context);
        seen.tensors.push((
            bytes(tensor.name),
            tensor.dtype,
            shape,
            tensor.begin,
            tensor.end,
        ));
        c_int::from(seen.stop_after == Some(seen.tensors.len()))
    }

    unsafe extern "C" fn take_rejection(
        context: *mut c_void,
        rejection: *const GufoSafetensorsRejection,
    ) {
        // SAFETY: the reader passes a live rejection.
        let rejection = unsafe { &*rejection };
        seen(context).rejection = Some((
            rejection.reason,
            bytes(rejection.name),
            bytes(rejection.message),
        ));
    }

    fn run(header: &[u8], file_size: u64, seen: &mut Seen) -> GufoSafetensorsParse {
        let context = (seen as *mut Seen).cast();
        // SAFETY: valid buffer and callbacks; `seen` outlives the call.
        unsafe {
            gufo_safetensors_parse_header(
                header.as_ptr(),
                header.len(),
                file_size,
                Some(take_metadata),
                Some(take_tensor),
                Some(take_rejection),
                context,
            )
        }
    }

    const HEADER: &[u8] = br#"{"a":{"dtype":"BF16","shape":[1,2],"data_offsets":[0,4]},"__metadata__":{"k":"v"},"b":{"dtype":"U8","shape":[2],"data_offsets":[4,6]}}"#;

    #[test]
    fn passes_metadata_then_tensors() {
        let mut seen = Seen::default();
        let size = 8 + HEADER.len() as u64 + 6;
        assert_eq!(run(HEADER, size, &mut seen), GufoSafetensorsParse::Complete);
        assert_eq!(seen.metadata, [(b"k".to_vec(), b"v".to_vec())]);
        assert_eq!(
            seen.tensors,
            [
                (b"a".to_vec(), Dtype::BF16, vec![1, 2], 0, 4),
                (b"b".to_vec(), Dtype::U8, vec![2], 4, 6)
            ]
        );
        assert_eq!(seen.rejection, None);
    }

    #[test]
    fn reports_rejections() {
        let mut seen = Seen::default();
        let size = 8 + HEADER.len() as u64 + 7;
        assert_eq!(run(HEADER, size, &mut seen), GufoSafetensorsParse::Rejected);
        assert_eq!(
            seen.rejection,
            Some((
                Reason::Trailing,
                Vec::new(),
                b"payload continues after the last tensor".to_vec()
            ))
        );
        assert!(seen.tensors.is_empty() && seen.metadata.is_empty());

        let mut seen = Seen::default();
        assert_eq!(run(b"{", 9, &mut seen), GufoSafetensorsParse::Rejected);
        let (reason, name, message) = seen.rejection.unwrap();
        assert_eq!((reason, name.is_empty()), (Reason::Json, true));
        assert!(message.starts_with(b"not JSON: "));

        // Without a sink, a rejection is still reported by its status.
        // SAFETY: valid buffer and callbacks; `seen` outlives the call.
        let status = unsafe {
            gufo_safetensors_parse_header(
                b"[]".as_ptr(),
                2,
                10,
                Some(take_metadata),
                Some(take_tensor),
                None,
                (&mut Seen::default() as *mut Seen).cast(),
            )
        };
        assert_eq!(status, GufoSafetensorsParse::Rejected);
    }

    #[test]
    fn stops_when_asked() {
        let mut seen = Seen {
            stop_after: Some(1),
            ..Seen::default()
        };
        let size = 8 + HEADER.len() as u64 + 6;
        assert_eq!(run(HEADER, size, &mut seen), GufoSafetensorsParse::Stopped);
        assert_eq!(seen.tensors.len(), 1);
    }

    #[test]
    fn rejects_caller_errors() {
        // SAFETY: null with a nonzero length and missing callbacks are
        // caller errors the reader reports instead of reading.
        unsafe {
            let parse = |data: *const u8, len, metadata, tensor| {
                gufo_safetensors_parse_header(
                    data,
                    len,
                    100,
                    metadata,
                    tensor,
                    None,
                    std::ptr::null_mut(),
                )
            };
            assert_eq!(
                parse(std::ptr::null(), 1, Some(take_metadata), Some(take_tensor)),
                GufoSafetensorsParse::Internal
            );
            assert_eq!(
                parse(HEADER.as_ptr(), HEADER.len(), None, Some(take_tensor)),
                GufoSafetensorsParse::Internal
            );
            assert_eq!(
                parse(HEADER.as_ptr(), HEADER.len(), Some(take_metadata), None),
                GufoSafetensorsParse::Internal
            );
            let mut length = 0;
            assert_eq!(
                gufo_safetensors_header_length(
                    std::ptr::null(),
                    8,
                    100,
                    None,
                    std::ptr::null_mut(),
                    &mut length
                ),
                GufoSafetensorsParse::Internal
            );
            assert_eq!(
                gufo_safetensors_header_length(
                    2u64.to_le_bytes().as_ptr(),
                    8,
                    10,
                    None,
                    std::ptr::null_mut(),
                    std::ptr::null_mut()
                ),
                GufoSafetensorsParse::Internal
            );
        }
    }

    #[test]
    fn reads_lengths() {
        let mut length = 0;
        let mut seen = Seen::default();
        let context = (&mut seen as *mut Seen).cast();
        // SAFETY: valid buffers and callbacks; `seen` outlives the calls.
        unsafe {
            assert_eq!(
                gufo_safetensors_header_length(
                    2u64.to_le_bytes().as_ptr(),
                    8,
                    10,
                    Some(take_rejection),
                    context,
                    &mut length
                ),
                GufoSafetensorsParse::Complete
            );
            assert_eq!(length, 2);
            assert_eq!(
                gufo_safetensors_header_length(
                    3u64.to_le_bytes().as_ptr(),
                    8,
                    10,
                    Some(take_rejection),
                    context,
                    &mut length
                ),
                GufoSafetensorsParse::Rejected
            );
        }
        assert_eq!(
            seen.rejection,
            Some((
                Reason::Length,
                Vec::new(),
                b"runs past the end of the file".to_vec()
            ))
        );
    }
}
