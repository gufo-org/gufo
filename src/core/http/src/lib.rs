#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

#[cfg(panic = "abort")]
compile_error!("the C functions catch panics at the boundary, so build with panic = \"unwind\"");

mod request;

use std::ffi::{c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};

use request::{Lookup, Parsed};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GufoHttpParse {
    Complete = 0,
    Partial = 1,
    Malformed = 2,
    Internal = 3,
    TooManyFields = 4,
}

/// Bytes inside the caller's buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoHttpSlice {
    pub data: *const u8,
    pub len: usize,
}

impl GufoHttpSlice {
    fn of(bytes: &[u8]) -> Self {
        Self {
            data: bytes.as_ptr(),
            len: bytes.len(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoHttpHead {
    pub len: usize,
    pub method: GufoHttpSlice,
    pub path: GufoHttpSlice,
    pub query: GufoHttpSlice,
    pub http11: c_int,
    pub framing_valid: c_int,
    pub content_length: u64,
    pub expect_continue: c_int,
}

/// Takes one header field, in order; returns nonzero to give up.
pub type GufoHttpFieldFn = unsafe extern "C" fn(
    context: *mut c_void,
    name: *const u8,
    name_len: usize,
    value: *const u8,
    value_len: usize,
) -> c_int;

/// Takes bytes valid during the call only.
pub type GufoHttpBytesFn = unsafe extern "C" fn(context: *mut c_void, data: *const u8, len: usize);

/// # Safety
/// `data` points to `len` readable bytes, or `len` is 0.
#[allow(unsafe_code)]
unsafe fn input<'a>(data: *const u8, len: usize) -> Option<&'a [u8]> {
    match (data.is_null(), len) {
        (_, 0) => Some(&[]),
        (true, _) => None,
        // SAFETY: the caller guarantees `len` readable bytes at a non-null
        // `data`.
        (false, _) => Some(unsafe { std::slice::from_raw_parts(data, len) }),
    }
}

/// Parses the request head at the start of `buffer`. `known_partial` is a
/// length of the same buffer that already parsed as `Partial` (0 at first),
/// so that only the new bytes are searched for the end of the head. On
/// `Complete`, `*head` describes it, its slices point into `buffer`, and
/// `field` was called for each header field in order; otherwise neither is
/// touched.
///
/// # Safety
/// `buffer` points to `len` readable bytes, or `len` is 0; `head` points to a
/// writable value; `field` may be called with `context` until this returns.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_http_parse_head(
    buffer: *const u8,
    len: usize,
    known_partial: usize,
    head: *mut GufoHttpHead,
    field: Option<GufoHttpFieldFn>,
    context: *mut c_void,
) -> GufoHttpParse {
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the caller guarantees `len` readable bytes.
        let Some(buffer) = (unsafe { input(buffer, len) }) else {
            return GufoHttpParse::Internal;
        };
        // SAFETY: the caller passes a writable `head`; `as_mut` rejects null.
        let (Some(out), Some(field)) = (unsafe { head.as_mut() }, field) else {
            return GufoHttpParse::Internal;
        };
        let parsed = match request::parse_head(buffer, known_partial) {
            Parsed::Partial => return GufoHttpParse::Partial,
            Parsed::Malformed => return GufoHttpParse::Malformed,
            Parsed::TooManyFields => return GufoHttpParse::TooManyFields,
            Parsed::NoMemory => return GufoHttpParse::Internal,
            Parsed::Complete(parsed) => parsed,
        };
        for httparse::Header { name, value } in &parsed.headers {
            // SAFETY: the caller allows calling `field` with `context` here;
            // both slices live in `buffer` during the call.
            let stop = unsafe {
                field(
                    context,
                    name.as_ptr(),
                    name.len(),
                    value.as_ptr(),
                    value.len(),
                )
            };
            if stop != 0 {
                return GufoHttpParse::Internal;
            }
        }
        *out = GufoHttpHead {
            len: parsed.len,
            method: GufoHttpSlice::of(parsed.method.as_bytes()),
            path: GufoHttpSlice::of(parsed.path.as_bytes()),
            query: GufoHttpSlice::of(parsed.query.as_bytes()),
            http11: c_int::from(parsed.http11),
            framing_valid: c_int::from(parsed.content_length.is_some()),
            content_length: parsed.content_length.unwrap_or(0),
            expect_continue: c_int::from(parsed.expect_continue),
        };
        GufoHttpParse::Complete
    }))
    .unwrap_or(GufoHttpParse::Internal)
}

/// Finds the first parameter of `query` whose decoded name is `key`, hands
/// its decoded value to `sink` and returns 1; returns 0 if there is none, -1
/// on a caller or library bug or when memory runs out.
///
/// # Safety
/// `query` and `key` point to their lengths of readable bytes, or the length
/// is 0; `sink` may be called with `context` until this returns.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_http_query_param(
    query: *const u8,
    query_len: usize,
    key: *const u8,
    key_len: usize,
    sink: Option<GufoHttpBytesFn>,
    context: *mut c_void,
) -> c_int {
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the caller guarantees `query_len` readable bytes.
        let Some(query) = (unsafe { input(query, query_len) }) else {
            return -1;
        };
        // SAFETY: the caller guarantees `key_len` readable bytes.
        let (Some(key), Some(sink)) = (unsafe { input(key, key_len) }, sink) else {
            return -1;
        };
        match request::query_param(query, key) {
            Lookup::Found(value) => {
                // SAFETY: the caller allows calling `sink` with `context`;
                // `value` lives during the call.
                unsafe { sink(context, value.as_ptr(), value.len()) };
                1
            }
            Lookup::Missing => 0,
            Lookup::NoMemory => -1,
        }
    }))
    .unwrap_or(-1)
}

/// 1 if the comma-separated `value` lists the lowercase `token`, ignoring
/// ASCII case and spaces and tabs around each item; 0 if not; -1 on a caller
/// or library bug.
///
/// # Safety
/// `value` and `token` point to their lengths of readable bytes, or the
/// length is 0.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_http_has_token(
    value: *const u8,
    value_len: usize,
    token: *const u8,
    token_len: usize,
) -> c_int {
    catch_unwind(|| {
        // SAFETY: the caller guarantees `value_len` readable bytes.
        let Some(value) = (unsafe { input(value, value_len) }) else {
            return -1;
        };
        // SAFETY: the caller guarantees `token_len` readable bytes.
        let Some(token) = (unsafe { input(token, token_len) }) else {
            return -1;
        };
        c_int::from(request::has_token(value, token))
    })
    .unwrap_or(-1)
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;

    unsafe extern "C" fn keep_field(
        context: *mut c_void,
        _name: *const u8,
        _name_len: usize,
        _value: *const u8,
        _value_len: usize,
    ) -> c_int {
        // SAFETY: the tests pass a `usize` counter as the context.
        let count = unsafe { &mut *context.cast::<usize>() };
        *count += 1;
        0
    }

    unsafe extern "C" fn refuse_field(
        _context: *mut c_void,
        _name: *const u8,
        _name_len: usize,
        _value: *const u8,
        _value_len: usize,
    ) -> c_int {
        1
    }

    unsafe extern "C" fn keep_bytes(context: *mut c_void, data: *const u8, len: usize) {
        // SAFETY: the tests pass a `Vec<u8>` as the context, and `data`
        // holds `len` bytes during the call.
        unsafe {
            (*context.cast::<Vec<u8>>()).extend_from_slice(std::slice::from_raw_parts(data, len))
        };
    }

    fn parse(
        text: &[u8],
        field: Option<GufoHttpFieldFn>,
        head: *mut GufoHttpHead,
    ) -> (GufoHttpParse, usize) {
        let mut count = 0usize;
        // SAFETY: `text` is readable; `head` is null or writable; the
        // callbacks take the counter passed as the context.
        let parsed = unsafe {
            gufo_http_parse_head(
                text.as_ptr(),
                text.len(),
                0,
                head,
                field,
                (&raw mut count).cast(),
            )
        };
        (parsed, count)
    }

    #[test]
    fn head_results_and_caller_errors() {
        let text = b"GET /a?b HTTP/1.1\r\nHost: h\r\nX: y\r\n\r\n";
        let mut head = GufoHttpHead {
            len: 0,
            method: GufoHttpSlice::of(b""),
            path: GufoHttpSlice::of(b""),
            query: GufoHttpSlice::of(b""),
            http11: 0,
            framing_valid: 0,
            content_length: 0,
            expect_continue: 0,
        };
        assert_eq!(
            parse(text, Some(keep_field), &raw mut head),
            (GufoHttpParse::Complete, 2)
        );
        assert_eq!(
            (head.len, head.http11, head.framing_valid),
            (text.len(), 1, 1)
        );
        assert_eq!(
            parse(text, Some(refuse_field), &raw mut head).0,
            GufoHttpParse::Internal
        );
        assert_eq!(parse(text, None, &raw mut head).0, GufoHttpParse::Internal);
        assert_eq!(
            parse(text, Some(keep_field), std::ptr::null_mut()).0,
            GufoHttpParse::Internal
        );
        let many = [
            &b"GET / HTTP/1.1\r\n"[..],
            &b"X: y\r\n".repeat(101),
            b"\r\n",
        ]
        .concat();
        assert_eq!(
            parse(&many, Some(keep_field), &raw mut head),
            (GufoHttpParse::TooManyFields, 0)
        );
        // SAFETY: a null buffer with a length is the caller error under test.
        let null = unsafe {
            gufo_http_parse_head(
                std::ptr::null(),
                4,
                0,
                &raw mut head,
                Some(keep_field),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(null, GufoHttpParse::Internal);
    }

    #[test]
    fn query_lookup_results() {
        let lookup = |query: &[u8], key: &[u8]| {
            let mut value = Vec::new();
            // SAFETY: both inputs are readable; `keep_bytes` takes the `Vec`
            // passed as the context.
            let found = unsafe {
                gufo_http_query_param(
                    query.as_ptr(),
                    query.len(),
                    key.as_ptr(),
                    key.len(),
                    Some(keep_bytes),
                    (&raw mut value).cast(),
                )
            };
            (found, value)
        };
        assert_eq!(lookup(b"model=a%20b", b"model"), (1, b"a b".to_vec()));
        assert_eq!(lookup(b"", b""), (0, Vec::new()));
        assert_eq!(lookup(b"a&", b""), (0, Vec::new()));
        assert_eq!(lookup(b"a&&b", b""), (1, Vec::new()));
        assert_eq!(lookup(b"=x", b""), (1, b"x".to_vec()));
        // SAFETY: a missing sink is the caller error under test.
        let no_sink = unsafe {
            gufo_http_query_param(
                b"a".as_ptr(),
                1,
                b"a".as_ptr(),
                1,
                None,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(no_sink, -1);
        // SAFETY: a null key with a length is the caller error under test.
        let null = unsafe {
            gufo_http_query_param(
                b"a".as_ptr(),
                1,
                std::ptr::null(),
                1,
                Some(keep_bytes),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(null, -1);
    }

    #[test]
    fn token_caller_errors() {
        // SAFETY: both inputs are readable.
        let listed = unsafe { gufo_http_has_token(b"a, b".as_ptr(), 4, b"b".as_ptr(), 1) };
        // SAFETY: a null value with a length is the caller error under test.
        let null = unsafe { gufo_http_has_token(std::ptr::null(), 4, b"b".as_ptr(), 1) };
        assert_eq!((listed, null), (1, -1));
    }
}
