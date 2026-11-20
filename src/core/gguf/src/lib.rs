//! The C interface of the GGUF header reader. `GgufReader`
//! (src/core/gguf_reader.cpp) parses the header of every file it opens with
//! it; the tensor data stays in the caller's mapping.
#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod header;

use gufo_ffi::guard;
use std::ffi::{c_int, c_void};

use header::{ArrayKind, Failure, Items, Limits, Sink, Stopped, Tensor, Value};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GufoGgufParse {
    Complete = 0,
    Rejected = 1,
    NoMemory = 2,
    Stopped = 3,
    Internal = 4,
}

/// Bytes inside the parsed buffer, or items the parser holds during a call.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoGgufSlice {
    pub data: *const u8,
    pub len: usize,
}

impl GufoGgufSlice {
    const EMPTY: Self = Self {
        data: std::ptr::null(),
        len: 0,
    };

    fn of(bytes: &[u8]) -> Self {
        Self {
            data: bytes.as_ptr(),
            len: bytes.len(),
        }
    }
}

/// Which field of `GufoGgufMetadata` holds the value.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GufoGgufKind {
    Unsigned = 0,
    Signed = 1,
    Float = 2,
    Bool = 3,
    String = 4,
    Strings = 5,
    Unsigneds = 6,
    Signeds = 7,
    Floats = 8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoGgufMetadata {
    pub key: GufoGgufSlice,
    pub value_type: u32,
    pub kind: GufoGgufKind,
    pub unsigned_value: u64,
    pub signed_value: i64,
    pub float_value: f64,
    pub string: GufoGgufSlice,
    pub count: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoGgufItems {
    pub kind: GufoGgufKind,
    pub strings: *const GufoGgufSlice,
    pub unsigneds: *const u64,
    pub signeds: *const i64,
    pub floats: *const f64,
    pub count: usize,
    pub first: usize,
    pub len: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoGgufTensor {
    pub name: GufoGgufSlice,
    pub dimensions: *const u64,
    pub dimension_count: usize,
    pub ggml_type: u32,
    pub sized: c_int,
    pub offset: u64,
    pub start: usize,
    pub size: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoGgufLimits {
    pub descriptor_budget: usize,
    pub metadata_entry_bytes: usize,
    pub tensor_entry_bytes: usize,
    pub string_item_bytes: usize,
    pub allow_unsized_tensors: c_int,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GufoGgufHeader {
    pub version: u32,
    pub alignment: u64,
}

/// Returns the encoded bytes of a tensor, or 0 when they cannot be sized.
pub type GufoGgufTensorBytesFn = unsafe extern "C" fn(
    context: *mut c_void,
    ggml_type: u32,
    elements: u64,
    first_dimension: u64,
) -> usize;

/// Takes the entry counts; returns nonzero to stop.
pub type GufoGgufCountsFn =
    unsafe extern "C" fn(context: *mut c_void, metadata_count: usize, tensor_count: usize) -> c_int;

/// Takes a block of array items; returns nonzero to stop.
pub type GufoGgufItemsFn =
    unsafe extern "C" fn(context: *mut c_void, items: *const GufoGgufItems) -> c_int;

/// Takes one metadata entry; returns nonzero to stop.
pub type GufoGgufMetadataFn =
    unsafe extern "C" fn(context: *mut c_void, entry: *const GufoGgufMetadata) -> c_int;

/// Takes one tensor; returns nonzero to stop.
pub type GufoGgufTensorFn =
    unsafe extern "C" fn(context: *mut c_void, tensor: *const GufoGgufTensor) -> c_int;

/// Takes bytes valid during the call only.
pub type GufoGgufBytesFn = unsafe extern "C" fn(context: *mut c_void, data: *const u8, len: usize);

/// The caller's callbacks, which it allows to be called with `context` until
/// the parse returns.
struct Callbacks {
    context: *mut c_void,
    tensor_bytes: GufoGgufTensorBytesFn,
    counts: GufoGgufCountsFn,
    items: GufoGgufItemsFn,
    metadata: GufoGgufMetadataFn,
    tensor: GufoGgufTensorFn,
}

impl<'a> Sink<'a> for Callbacks {
    type Str = GufoGgufSlice;

    fn string(bytes: &'a [u8]) -> GufoGgufSlice {
        GufoGgufSlice::of(bytes)
    }

    #[allow(unsafe_code)]
    fn tensor_bytes(&mut self, ggml_type: u32, elements: u64, first_dimension: u64) -> usize {
        // SAFETY: the caller allows this call (see `Callbacks`).
        unsafe { (self.tensor_bytes)(self.context, ggml_type, elements, first_dimension) }
    }

    #[allow(unsafe_code)]
    fn counts(&mut self, metadata: usize, tensors: usize) -> Result<(), Stopped> {
        // SAFETY: the caller allows this call (see `Callbacks`).
        match unsafe { (self.counts)(self.context, metadata, tensors) } {
            0 => Ok(()),
            _ => Err(Stopped),
        }
    }

    #[allow(unsafe_code)]
    fn items(
        &mut self,
        items: Items<'_, GufoGgufSlice>,
        first: usize,
        len: usize,
    ) -> Result<(), Stopped> {
        let mut block = GufoGgufItems {
            kind: GufoGgufKind::Strings,
            strings: std::ptr::null(),
            unsigneds: std::ptr::null(),
            signeds: std::ptr::null(),
            floats: std::ptr::null(),
            count: 0,
            first,
            len,
        };
        match items {
            Items::Strings(items) => {
                block.strings = items.as_ptr();
                block.count = items.len();
            }
            Items::Unsigneds(items) => {
                block.kind = GufoGgufKind::Unsigneds;
                block.unsigneds = items.as_ptr();
                block.count = items.len();
            }
            Items::Signeds(items) => {
                block.kind = GufoGgufKind::Signeds;
                block.signeds = items.as_ptr();
                block.count = items.len();
            }
            Items::Floats(items) => {
                block.kind = GufoGgufKind::Floats;
                block.floats = items.as_ptr();
                block.count = items.len();
            }
        }
        // SAFETY: the caller allows this call (see `Callbacks`); `block` and
        // the items it points to live until it returns.
        match unsafe { (self.items)(self.context, &block) } {
            0 => Ok(()),
            _ => Err(Stopped),
        }
    }

    #[allow(unsafe_code)]
    fn metadata(
        &mut self,
        key: &'a [u8],
        value_type: u32,
        value: Value<'a>,
    ) -> Result<(), Stopped> {
        let mut entry = GufoGgufMetadata {
            key: GufoGgufSlice::of(key),
            value_type,
            kind: GufoGgufKind::Unsigned,
            unsigned_value: 0,
            signed_value: 0,
            float_value: 0.0,
            string: GufoGgufSlice::EMPTY,
            count: 0,
        };
        match value {
            Value::Unsigned(value) => entry.unsigned_value = value,
            Value::Signed(value) => {
                entry.kind = GufoGgufKind::Signed;
                entry.signed_value = value;
            }
            Value::Float(value) => {
                entry.kind = GufoGgufKind::Float;
                entry.float_value = value;
            }
            Value::Bool(value) => {
                entry.kind = GufoGgufKind::Bool;
                entry.unsigned_value = value.into();
            }
            Value::String(value) => {
                entry.kind = GufoGgufKind::String;
                entry.string = GufoGgufSlice::of(value);
            }
            Value::Array(kind, len) => {
                entry.kind = match kind {
                    ArrayKind::Strings => GufoGgufKind::Strings,
                    ArrayKind::Unsigneds => GufoGgufKind::Unsigneds,
                    ArrayKind::Signeds => GufoGgufKind::Signeds,
                    ArrayKind::Floats => GufoGgufKind::Floats,
                };
                entry.count = len;
            }
        }
        // SAFETY: the caller allows this call (see `Callbacks`); `entry` lives
        // until it returns.
        match unsafe { (self.metadata)(self.context, &entry) } {
            0 => Ok(()),
            _ => Err(Stopped),
        }
    }

    #[allow(unsafe_code)]
    fn tensor(&mut self, tensor: &Tensor<'a>) -> Result<(), Stopped> {
        let dimensions = tensor.dimensions();
        let parsed = GufoGgufTensor {
            name: GufoGgufSlice::of(tensor.name),
            dimensions: dimensions.as_ptr(),
            dimension_count: dimensions.len(),
            ggml_type: tensor.ggml_type,
            sized: tensor.sized.into(),
            offset: tensor.offset,
            start: tensor.start,
            size: tensor.size,
        };
        // SAFETY: the caller allows this call (see `Callbacks`); `parsed`
        // and its dimensions live until it returns.
        match unsafe { (self.tensor)(self.context, &parsed) } {
            0 => Ok(()),
            _ => Err(Stopped),
        }
    }
}

/// Parses the header of the GGUF file in `data`: `counts` takes the entry
/// counts once they passed the limits, `items` the items of each array in
/// blocks before its entry, `metadata` each entry in file order as its checks
/// pass, `tensor` each tensor once all passed.
/// On `Complete`, `*header` holds the version and the data alignment; on
/// `Rejected`, `error` got the message unless the file is merely malformed
/// or truncated. Entries taken before a failure are void.
///
/// # Safety
/// `data` points to `len` readable bytes that stay unchanged during the
/// call, or `len` is 0; `limits` points to a readable value and `header` to a
/// writable one; each callback may be called with `context` until this
/// returns.
#[allow(unsafe_code)]
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_gguf_parse_header(
    data: *const u8,
    len: usize,
    limits: *const GufoGgufLimits,
    tensor_bytes: Option<GufoGgufTensorBytesFn>,
    counts: Option<GufoGgufCountsFn>,
    items: Option<GufoGgufItemsFn>,
    metadata: Option<GufoGgufMetadataFn>,
    tensor: Option<GufoGgufTensorFn>,
    error: Option<GufoGgufBytesFn>,
    context: *mut c_void,
    header: *mut GufoGgufHeader,
) -> GufoGgufParse {
    guard(GufoGgufParse::Internal, || {
        // SAFETY: the caller guarantees `len` readable bytes.
        let Some(bytes) = (unsafe { gufo_ffi::input(data, len) }) else {
            return GufoGgufParse::Internal;
        };
        // SAFETY: the caller passes a readable `limits`; `as_ref` rejects null.
        let Some(limits) = (unsafe { limits.as_ref() }) else {
            return GufoGgufParse::Internal;
        };
        // SAFETY: the caller passes a writable `header`; `as_mut` rejects null.
        let Some(out) = (unsafe { header.as_mut() }) else {
            return GufoGgufParse::Internal;
        };
        let (Some(tensor_bytes), Some(counts), Some(items), Some(metadata), Some(tensor)) =
            (tensor_bytes, counts, items, metadata, tensor)
        else {
            return GufoGgufParse::Internal;
        };
        let limits = Limits {
            budget: limits.descriptor_budget,
            metadata_entry_bytes: limits.metadata_entry_bytes,
            tensor_entry_bytes: limits.tensor_entry_bytes,
            string_item_bytes: limits.string_item_bytes,
            allow_unsized: limits.allow_unsized_tensors != 0,
        };
        let mut callbacks = Callbacks {
            context,
            tensor_bytes,
            counts,
            items,
            metadata,
            tensor,
        };
        match header::parse(bytes, &limits, &mut callbacks) {
            Ok(parsed) => {
                *out = GufoGgufHeader {
                    version: parsed.version,
                    alignment: parsed.alignment,
                };
                GufoGgufParse::Complete
            }
            Err(Failure::Rejected(rejection)) => {
                let Some(error) = error else {
                    return GufoGgufParse::Rejected;
                };
                match rejection.message() {
                    Ok(Some(message)) => {
                        // SAFETY: the caller allows calling `error` with
                        // `context`; `message` lives until it returns.
                        unsafe { error(context, message.as_ptr(), message.len()) };
                        GufoGgufParse::Rejected
                    }
                    Ok(None) => GufoGgufParse::Rejected,
                    Err(_) => GufoGgufParse::NoMemory,
                }
            }
            Err(Failure::NoMemory) => GufoGgufParse::NoMemory,
            Err(Failure::Stopped) => GufoGgufParse::Stopped,
        }
    })
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Seen {
        keys: Vec<Vec<u8>>,
        strings: Vec<Vec<u8>>,
        tensors: Vec<(Vec<u8>, Vec<u64>, usize, usize)>,
        error: Vec<u8>,
        counts: Option<(usize, usize)>,
        stop: bool,
    }

    fn seen<'a>(context: *mut c_void) -> &'a mut Seen {
        // SAFETY: every test passes a live `Seen` as the context.
        unsafe { &mut *context.cast::<Seen>() }
    }

    unsafe extern "C" fn f32_bytes(_: *mut c_void, ggml_type: u32, elements: u64, _: u64) -> usize {
        if ggml_type == 0 {
            elements as usize * 4
        } else {
            0
        }
    }

    unsafe extern "C" fn take_counts(
        context: *mut c_void,
        metadata: usize,
        tensors: usize,
    ) -> c_int {
        seen(context).counts = Some((metadata, tensors));
        0
    }

    unsafe extern "C" fn take_metadata(
        context: *mut c_void,
        entry: *const GufoGgufMetadata,
    ) -> c_int {
        let seen = seen(context);
        // SAFETY: the parser passes a live entry whose slices are valid now.
        let entry = unsafe { &*entry };
        // SAFETY: as above.
        let key = unsafe { std::slice::from_raw_parts(entry.key.data, entry.key.len) };
        seen.keys.push(key.to_vec());
        if entry.kind == GufoGgufKind::Strings {
            assert_eq!(entry.count, seen.strings.len());
        }
        c_int::from(seen.stop)
    }

    unsafe extern "C" fn take_items(context: *mut c_void, items: *const GufoGgufItems) -> c_int {
        let seen = seen(context);
        // SAFETY: the parser passes a live block whose slices are valid now.
        let items = unsafe { &*items };
        assert_eq!(items.kind, GufoGgufKind::Strings);
        assert_eq!(items.first, seen.strings.len());
        // SAFETY: as above.
        let strings = unsafe { std::slice::from_raw_parts(items.strings, items.count) };
        for string in strings {
            // SAFETY: as above.
            seen.strings
                .push(unsafe { std::slice::from_raw_parts(string.data, string.len) }.to_vec());
        }
        0
    }

    unsafe extern "C" fn take_tensor(context: *mut c_void, tensor: *const GufoGgufTensor) -> c_int {
        // SAFETY: the parser passes a live tensor whose slices are valid now.
        let tensor = unsafe { &*tensor };
        // SAFETY: as above.
        let name = unsafe { std::slice::from_raw_parts(tensor.name.data, tensor.name.len) };
        // SAFETY: as above.
        let dimensions =
            unsafe { std::slice::from_raw_parts(tensor.dimensions, tensor.dimension_count) };
        seen(context).tensors.push((
            name.to_vec(),
            dimensions.to_vec(),
            tensor.start,
            tensor.size,
        ));
        0
    }

    unsafe extern "C" fn take_error(context: *mut c_void, data: *const u8, len: usize) {
        // SAFETY: the parser passes `len` readable bytes.
        seen(context).error = unsafe { std::slice::from_raw_parts(data, len) }.to_vec();
    }

    const LIMITS: GufoGgufLimits = GufoGgufLimits {
        descriptor_budget: 1 << 20,
        metadata_entry_bytes: 104,
        tensor_entry_bytes: 208,
        string_item_bytes: 16,
        allow_unsized_tensors: 0,
    };

    fn call(bytes: &[u8], seen: &mut Seen, header: &mut GufoGgufHeader) -> GufoGgufParse {
        // SAFETY: every pointer is live for the call.
        unsafe {
            gufo_gguf_parse_header(
                bytes.as_ptr(),
                bytes.len(),
                &LIMITS,
                Some(f32_bytes),
                Some(take_counts),
                Some(take_items),
                Some(take_metadata),
                Some(take_tensor),
                Some(take_error),
                (seen as *mut Seen).cast(),
                header,
            )
        }
    }

    /// One string array and one F32 tensor of two elements at offset 0.
    fn file() -> Vec<u8> {
        let mut bytes = b"GGUF".to_vec();
        for value in [
            3u32.to_le_bytes().as_slice(),
            &1u64.to_le_bytes(),
            &1u64.to_le_bytes(),
        ] {
            bytes.extend_from_slice(value);
        }
        let string = |bytes: &mut Vec<u8>, value: &[u8]| {
            bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            bytes.extend_from_slice(value);
        };
        string(&mut bytes, b"tokens");
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        string(&mut bytes, b"a");
        string(&mut bytes, b"bc");
        string(&mut bytes, b"w");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.resize(bytes.len().next_multiple_of(32) + 8, 0);
        bytes
    }

    #[test]
    fn delivers_entries_and_tensors() {
        let bytes = file();
        let mut seen = Seen::default();
        let mut header = GufoGgufHeader::default();
        assert_eq!(
            call(&bytes, &mut seen, &mut header),
            GufoGgufParse::Complete
        );
        assert_eq!((header.version, header.alignment), (3, 32));
        assert_eq!(seen.counts, Some((1, 1)));
        assert_eq!(seen.keys, [b"tokens".to_vec()]);
        assert_eq!(seen.strings, [b"a".to_vec(), b"bc".to_vec()]);
        assert_eq!(seen.tensors, [(b"w".to_vec(), vec![2], bytes.len() - 8, 8)]);
    }

    #[test]
    fn reports_rejections_and_stops() {
        let mut bytes = file();
        bytes.pop();
        let mut seen = Seen::default();
        let mut header = GufoGgufHeader::default();
        assert_eq!(
            call(&bytes, &mut seen, &mut header),
            GufoGgufParse::Rejected
        );
        assert_eq!(seen.error, b"Tensor extent or alignment is invalid: w");
        assert!(seen.tensors.is_empty());
        assert_eq!(header.version, 0);
        // Without an error callback, the rejection has no message.
        let mut seen = Seen::default();
        // SAFETY: every pointer is live for the call.
        let parsed = unsafe {
            gufo_gguf_parse_header(
                bytes.as_ptr(),
                bytes.len(),
                &LIMITS,
                Some(f32_bytes),
                Some(take_counts),
                Some(take_items),
                Some(take_metadata),
                Some(take_tensor),
                None,
                (&mut seen as *mut Seen).cast(),
                &mut header,
            )
        };
        assert_eq!(parsed, GufoGgufParse::Rejected);

        let mut seen = Seen {
            stop: true,
            ..Seen::default()
        };
        assert_eq!(
            call(&file(), &mut seen, &mut header),
            GufoGgufParse::Stopped
        );
    }

    #[test]
    fn rejects_caller_errors() {
        let mut header = GufoGgufHeader::default();
        // SAFETY: null pointers and missing callbacks are reported.
        unsafe {
            assert_eq!(
                gufo_gguf_parse_header(
                    std::ptr::null(),
                    1,
                    &LIMITS,
                    Some(f32_bytes),
                    Some(take_counts),
                    Some(take_items),
                    Some(take_metadata),
                    Some(take_tensor),
                    None,
                    std::ptr::null_mut(),
                    &mut header,
                ),
                GufoGgufParse::Internal
            );
            let bytes = file();
            assert_eq!(
                gufo_gguf_parse_header(
                    bytes.as_ptr(),
                    bytes.len(),
                    &LIMITS,
                    Some(f32_bytes),
                    Some(take_counts),
                    Some(take_items),
                    None,
                    Some(take_tensor),
                    None,
                    std::ptr::null_mut(),
                    &mut header,
                ),
                GufoGgufParse::Internal
            );
        }
    }
}
