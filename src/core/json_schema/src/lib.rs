//! The C interface of the JSON Schema compiler: string `pattern`s to DFA
//! tables, the string keywords and formats, and the exact number lexeme.
//! `src/core/json_schema_regex.cpp` and `src/core/json_schema_lexeme.cpp`
//! step over the results.
#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod chars;
mod decimal;
mod error;
mod number;
mod regex;
mod string;
mod value;

use std::ffi::{c_int, c_void};

use chars::Chars;
use error::Failure;
use gufo_ffi::guard;
use value::Value;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GufoJsonSchemaStatus {
    Complete = 0,
    Rejected = 1,
    Empty = 2,
    NoMemory = 3,
    Stopped = 4,
    Internal = 5,
}

pub const GUFO_JSON_SCHEMA_ABSENT: c_int = 0;
pub const GUFO_JSON_SCHEMA_NUMBER: c_int = 1;
pub const GUFO_JSON_SCHEMA_STRING: c_int = 2;
pub const GUFO_JSON_SCHEMA_OTHER: c_int = 3;

/// A keyword's value: see `include/gufo/json_schema.h`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoJsonSchemaValue {
    pub kind: c_int,
    pub number: f64,
    pub text: *const u8,
    pub len: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoJsonSchemaBytes {
    pub data: *const u8,
    pub len: usize,
}

pub type GufoJsonSchemaBytesFn =
    unsafe extern "C" fn(context: *mut c_void, data: *const u8, len: usize);
pub type GufoJsonSchemaRangeFn = unsafe extern "C" fn(sink: *mut c_void, first: u32, last: u32);
pub type GufoJsonSchemaStringFn =
    unsafe extern "C" fn(sink: *mut c_void, codepoints: *const u32, len: usize);
pub type GufoJsonSchemaSetFn = unsafe extern "C" fn(
    context: *mut c_void,
    pattern: *const u8,
    len: usize,
    add: GufoJsonSchemaRangeFn,
    add_string: GufoJsonSchemaStringFn,
    sink: *mut c_void,
) -> c_int;

pub const GUFO_JSON_SCHEMA_SET_RESOLVED: c_int = 0;
pub const GUFO_JSON_SCHEMA_SET_REJECTED: c_int = 1;
pub const GUFO_JSON_SCHEMA_SET_FAILED: c_int = 2;

/// The tables of a compiled pattern, borrowed from its handle.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoJsonSchemaRegexTables {
    pub states: u32,
    pub classes: u32,
    pub next: *const u32,
    pub accepting: *const u8,
    pub distance: *const u32,
    pub successor_offsets: *const u32,
    pub successors: *const u32,
    pub maximum_suffix: u32,
    pub lookup_len: usize,
    pub lookup_starts: *const u32,
    pub lookup_classes: *const u32,
    pub class_offsets: *const u32,
    pub class_ranges: *const u32,
}

pub struct GufoJsonSchemaRegex {
    dfa: regex::Dfa,
    class_offsets: Vec<u32>,
    class_ranges: Vec<u32>,
}

pub struct GufoJsonSchemaNumber(number::Number);

/// The caller's callbacks, which it allows to be called with `context` until
/// the call returns.
struct Callbacks {
    context: *mut c_void,
    error: Option<GufoJsonSchemaBytesFn>,
    sets: Option<GufoJsonSchemaSetFn>,
}

impl Callbacks {
    fn new(context: *mut c_void, error: Option<GufoJsonSchemaBytesFn>) -> Self {
        Self {
            context,
            error,
            sets: None,
        }
    }

    #[allow(unsafe_code)]
    fn bytes(&self, call: Option<GufoJsonSchemaBytesFn>, bytes: &[u8]) {
        if let Some(call) = call {
            // SAFETY: the caller allows calling its callbacks with `context`;
            // `bytes` lives until the call returns.
            unsafe { call(self.context, bytes.as_ptr(), bytes.len()) };
        }
    }

    fn status<T>(&self, result: error::Result<T>) -> (GufoJsonSchemaStatus, Option<T>) {
        match result {
            Ok(value) => (GufoJsonSchemaStatus::Complete, Some(value)),
            Err(Failure::Rejected(reason)) => {
                self.bytes(self.error, reason.as_bytes());
                (GufoJsonSchemaStatus::Rejected, None)
            }
            Err(Failure::Empty(reason)) => {
                self.bytes(self.error, reason.as_bytes());
                (GufoJsonSchemaStatus::Empty, None)
            }
            Err(Failure::NoMemory) => (GufoJsonSchemaStatus::NoMemory, None),
            Err(Failure::Stopped) => (GufoJsonSchemaStatus::Stopped, None),
        }
    }
}

#[derive(Default)]
struct RangeSink {
    chars: Chars,
    failed: bool,
}

#[allow(unsafe_code)]
unsafe extern "C" fn add_range(sink: *mut c_void, first: u32, last: u32) {
    // SAFETY: `resolve` passes its live `RangeSink`, which nothing else
    // borrows during the callback.
    let sink = unsafe { &mut *sink.cast::<RangeSink>() };
    if !guard(false, || {
        sink.chars.add(first, last);
        true
    }) {
        sink.failed = true;
    }
}

#[allow(unsafe_code)]
unsafe extern "C" fn add_string(sink: *mut c_void, codepoints: *const u32, len: usize) {
    // SAFETY: as in `add_range`.
    let sink = unsafe { &mut *sink.cast::<RangeSink>() };
    // SAFETY: the callback passes `len` readable codepoints.
    let Some(string) = (unsafe { gufo_ffi::input(codepoints, len) }) else {
        sink.failed = true;
        return;
    };
    if !guard(false, || {
        sink.chars.add_string(string);
        true
    }) {
        sink.failed = true;
    }
}

impl regex::UnicodeSets for Callbacks {
    #[allow(unsafe_code)]
    fn resolve(&mut self, pattern: &str) -> error::Result<Option<Chars>> {
        let Some(sets) = self.sets else {
            return Err(Failure::Stopped);
        };
        let mut sink = RangeSink::default();
        // SAFETY: the caller allows calling `sets` with `context`; `pattern`
        // and `sink` live until it returns.
        let status = unsafe {
            sets(
                self.context,
                pattern.as_ptr(),
                pattern.len(),
                add_range,
                add_string,
                std::ptr::from_mut(&mut sink).cast(),
            )
        };
        match status {
            GUFO_JSON_SCHEMA_SET_RESOLVED if !sink.failed => Ok(Some(sink.chars)),
            GUFO_JSON_SCHEMA_SET_REJECTED => Ok(None),
            _ => Err(Failure::Stopped),
        }
    }
}

/// Borrows a keyword value, or `None` for a caller error.
///
/// # Safety
/// `value.text` points to `value.len` readable bytes for `'a`, or `len` is 0.
#[allow(unsafe_code)]
unsafe fn borrow<'a>(value: &GufoJsonSchemaValue) -> Option<Value<'a>> {
    // SAFETY: the caller guarantees `len` readable bytes.
    let text = unsafe { gufo_ffi::input(value.text, value.len) }?;
    Some(match value.kind {
        GUFO_JSON_SCHEMA_ABSENT => Value::Absent,
        GUFO_JSON_SCHEMA_NUMBER => Value::Number {
            value: value.number,
            text,
        },
        GUFO_JSON_SCHEMA_STRING => Value::String(text),
        GUFO_JSON_SCHEMA_OTHER => Value::Other,
        _ => return None,
    })
}

/// Borrows `N` keyword values.
///
/// # Safety
/// `values` points to `N` values, each valid as `borrow` requires.
#[allow(unsafe_code)]
unsafe fn borrow_all<'a, const N: usize>(
    values: *const GufoJsonSchemaValue,
) -> Option<[Value<'a>; N]> {
    // SAFETY: the caller guarantees `N` readable values.
    let values = unsafe { gufo_ffi::input(values, N) }?;
    let mut result = [Value::Absent; N];
    for (slot, value) in result.iter_mut().zip(values) {
        // SAFETY: the caller guarantees each value's text.
        *slot = unsafe { borrow(value) }?;
    }
    Some(result)
}

/// Compiles the intersection of `patterns` into a DFA admitting at most
/// `maximum` codepoints. `sets` resolves ICU UnicodeSet patterns.
///
/// # Safety
/// `patterns` points to `count` entries, each naming `len` readable bytes
/// (or `len` 0); `out` is writable; each callback may be called with
/// `context` until this returns.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_json_schema_regex_compile(
    patterns: *const GufoJsonSchemaBytes,
    count: usize,
    maximum: u32,
    sets: Option<GufoJsonSchemaSetFn>,
    error: Option<GufoJsonSchemaBytesFn>,
    context: *mut c_void,
    out: *mut *mut GufoJsonSchemaRegex,
) -> GufoJsonSchemaStatus {
    if out.is_null() {
        return GufoJsonSchemaStatus::Internal;
    }
    // SAFETY: `out` is writable.
    unsafe { out.write(std::ptr::null_mut()) };
    guard(GufoJsonSchemaStatus::Internal, || {
        // SAFETY: the caller guarantees `count` entries.
        let Some(entries) = (unsafe { gufo_ffi::input(patterns, count) }) else {
            return GufoJsonSchemaStatus::Internal;
        };
        let mut texts = Vec::new();
        if texts.try_reserve(entries.len()).is_err() {
            return GufoJsonSchemaStatus::NoMemory;
        }
        for entry in entries {
            // SAFETY: the caller guarantees each entry's bytes.
            let Some(text) = (unsafe { gufo_ffi::input(entry.data, entry.len) }) else {
                return GufoJsonSchemaStatus::Internal;
            };
            texts.push(text);
        }
        let mut callbacks = Callbacks::new(context, error);
        callbacks.sets = sets;
        let result = regex::compile(&texts, maximum, &mut callbacks);
        let (status, dfa) = callbacks.status(result);
        if let Some(dfa) = dfa {
            let mut class_offsets = Vec::with_capacity(dfa.alphabet.len() + 1);
            let mut class_ranges = Vec::new();
            class_offsets.push(0);
            for chars in &dfa.alphabet {
                for &(first, last) in chars.ranges() {
                    class_ranges.extend([first, last]);
                }
                class_offsets.push((class_ranges.len() / 2) as u32);
            }
            let handle = Box::new(GufoJsonSchemaRegex {
                dfa,
                class_offsets,
                class_ranges,
            });
            // SAFETY: `out` is writable.
            unsafe { out.write(Box::into_raw(handle)) };
        }
        status
    })
}

/// Fills `out` with the tables of `regex`, valid until it is freed.
///
/// # Safety
/// `regex` is a live handle and `out` is writable.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_json_schema_regex_tables(
    regex: *const GufoJsonSchemaRegex,
    out: *mut GufoJsonSchemaRegexTables,
) -> GufoJsonSchemaStatus {
    guard(GufoJsonSchemaStatus::Internal, || {
        if regex.is_null() || out.is_null() {
            return GufoJsonSchemaStatus::Internal;
        }
        // SAFETY: the caller passes a live handle.
        let regex = unsafe { &*regex };
        let dfa = &regex.dfa;
        let tables = GufoJsonSchemaRegexTables {
            states: dfa.states() as u32,
            classes: dfa.alphabet.len() as u32,
            next: dfa.next.as_ptr(),
            accepting: dfa.accepting.as_ptr(),
            distance: dfa.distance.as_ptr(),
            successor_offsets: dfa.successor_offsets.as_ptr(),
            successors: dfa.successors.as_ptr(),
            maximum_suffix: dfa.maximum_suffix,
            lookup_len: dfa.lookup_classes.len(),
            lookup_starts: dfa.lookup_starts.as_ptr(),
            lookup_classes: dfa.lookup_classes.as_ptr(),
            class_offsets: regex.class_offsets.as_ptr(),
            class_ranges: regex.class_ranges.as_ptr(),
        };
        // SAFETY: `out` is writable.
        unsafe { out.write(tables) };
        GufoJsonSchemaStatus::Complete
    })
}

/// Frees a compiled pattern; null is ignored.
///
/// # Safety
/// `regex` came from `gufo_json_schema_regex_compile` and is not used again.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_json_schema_regex_free(regex: *mut GufoJsonSchemaRegex) {
    guard((), || {
        if !regex.is_null() {
            // SAFETY: the caller transfers a handle from `Box::into_raw`.
            drop(unsafe { Box::from_raw(regex) });
        }
    });
}

/// Checks the string keywords `minLength`, `maxLength`, `pattern` and
/// `format` (in that order in `keywords`), writes the length bounds and
/// reports each pattern the value must match to `pattern`.
///
/// # Safety
/// `keywords` points to 4 values; `minimum` and `maximum` are writable; each
/// callback may be called with `context` until this returns.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_json_schema_string(
    keywords: *const GufoJsonSchemaValue,
    minimum: *mut u32,
    maximum: *mut u32,
    pattern: Option<GufoJsonSchemaBytesFn>,
    error: Option<GufoJsonSchemaBytesFn>,
    context: *mut c_void,
) -> GufoJsonSchemaStatus {
    guard(GufoJsonSchemaStatus::Internal, || {
        if minimum.is_null() || maximum.is_null() {
            return GufoJsonSchemaStatus::Internal;
        }
        // SAFETY: the caller guarantees 4 values.
        let Some(keywords) = (unsafe { borrow_all::<4>(keywords) }) else {
            return GufoJsonSchemaStatus::Internal;
        };
        let callbacks = Callbacks::new(context, error);
        let (status, plan) = callbacks.status(string::plan(&keywords));
        if let Some(plan) = plan {
            // SAFETY: both are writable.
            unsafe {
                minimum.write(plan.minimum);
                maximum.write(plan.maximum);
            }
            for text in &plan.patterns {
                callbacks.bytes(pattern, text);
            }
        }
        status
    })
}

/// Reports the pattern of string format `name` to `pattern` and writes its
/// length cap (`UINT32_MAX` for none) to `maximum`.
///
/// # Safety
/// `name` points to `len` readable bytes (or `len` is 0); `maximum` is
/// writable; each callback may be called with `context` until this returns.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_json_schema_format(
    name: *const u8,
    len: usize,
    maximum: *mut u32,
    pattern: Option<GufoJsonSchemaBytesFn>,
    error: Option<GufoJsonSchemaBytesFn>,
    context: *mut c_void,
) -> GufoJsonSchemaStatus {
    guard(GufoJsonSchemaStatus::Internal, || {
        // SAFETY: the caller guarantees `len` readable bytes.
        let Some(name) = (unsafe { gufo_ffi::input(name, len) }) else {
            return GufoJsonSchemaStatus::Internal;
        };
        if maximum.is_null() {
            return GufoJsonSchemaStatus::Internal;
        }
        let callbacks = Callbacks::new(context, error);
        let (status, format) = callbacks.status(string::format(name));
        if let Some((text, cap)) = format {
            // SAFETY: `maximum` is writable.
            unsafe { maximum.write(cap) };
            callbacks.bytes(pattern, text.as_bytes());
        }
        status
    })
}

/// Compiles the number lexeme from `minimum`, `maximum`,
/// `exclusiveMinimum`, `exclusiveMaximum` and `multipleOf` (in that order in
/// `keywords`). Numbers have up to `digits` integral and fraction digits,
/// more where a schema number needs them.
///
/// # Safety
/// `keywords` points to 5 values; `out` is writable; each callback may be
/// called with `context` until this returns.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_json_schema_number_new(
    keywords: *const GufoJsonSchemaValue,
    integer: c_int,
    digits: u32,
    error: Option<GufoJsonSchemaBytesFn>,
    context: *mut c_void,
    out: *mut *mut GufoJsonSchemaNumber,
) -> GufoJsonSchemaStatus {
    if out.is_null() {
        return GufoJsonSchemaStatus::Internal;
    }
    // SAFETY: `out` is writable.
    unsafe { out.write(std::ptr::null_mut()) };
    guard(GufoJsonSchemaStatus::Internal, || {
        // SAFETY: the caller guarantees 5 values.
        let Some(keywords) = (unsafe { borrow_all::<5>(keywords) }) else {
            return GufoJsonSchemaStatus::Internal;
        };
        let callbacks = Callbacks::new(context, error);
        let result = number::Number::new(&keywords, integer != 0, i64::from(digits));
        let (status, number) = callbacks.status(result);
        if let Some(number) = number {
            let handle = Box::into_raw(Box::new(GufoJsonSchemaNumber(number)));
            // SAFETY: `out` is writable.
            unsafe { out.write(handle) };
        }
        status
    })
}

pub const GUFO_JSON_SCHEMA_PREFIX: u8 = 1;
pub const GUFO_JSON_SCHEMA_MATCH: u8 = 2;

/// Writes to `match` whether `bytes` can continue to an accepted number
/// (`GUFO_JSON_SCHEMA_PREFIX`) and whether it is one (`GUFO_JSON_SCHEMA_MATCH`).
///
/// # Safety
/// `number` is a live handle; `bytes` points to `len` readable bytes (or
/// `len` is 0); `out` is writable.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_json_schema_number_check(
    number: *const GufoJsonSchemaNumber,
    bytes: *const u8,
    len: usize,
    out: *mut u8,
) -> GufoJsonSchemaStatus {
    guard(GufoJsonSchemaStatus::Internal, || {
        // SAFETY: the caller guarantees `len` readable bytes.
        let Some(bytes) = (unsafe { gufo_ffi::input(bytes, len) }) else {
            return GufoJsonSchemaStatus::Internal;
        };
        if number.is_null() || out.is_null() {
            return GufoJsonSchemaStatus::Internal;
        }
        // SAFETY: the caller passes a live handle.
        let found = unsafe { &*number }.0.check(bytes);
        let bits = if found.prefix {
            GUFO_JSON_SCHEMA_PREFIX
        } else {
            0
        } | if found.complete {
            GUFO_JSON_SCHEMA_MATCH
        } else {
            0
        };
        // SAFETY: `out` is writable.
        unsafe { out.write(bits) };
        GufoJsonSchemaStatus::Complete
    })
}

/// Writes to `accepted` whether the JSON number spelled `text` is accepted.
///
/// # Safety
/// As `gufo_json_schema_number_check`; each callback may be called with
/// `context` until this returns.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_json_schema_number_accept(
    number: *const GufoJsonSchemaNumber,
    text: *const u8,
    len: usize,
    accepted: *mut u8,
    error: Option<GufoJsonSchemaBytesFn>,
    context: *mut c_void,
) -> GufoJsonSchemaStatus {
    guard(GufoJsonSchemaStatus::Internal, || {
        // SAFETY: the caller guarantees `len` readable bytes.
        let Some(text) = (unsafe { gufo_ffi::input(text, len) }) else {
            return GufoJsonSchemaStatus::Internal;
        };
        if number.is_null() || accepted.is_null() {
            return GufoJsonSchemaStatus::Internal;
        }
        let callbacks = Callbacks::new(context, error);
        // SAFETY: the caller passes a live handle.
        let (status, found) = callbacks.status(unsafe { &*number }.0.accept(text));
        if let Some(found) = found {
            // SAFETY: `accepted` is writable.
            unsafe { accepted.write(u8::from(found)) };
        }
        status
    })
}

/// Frees a number lexeme; null is ignored.
///
/// # Safety
/// `number` came from `gufo_json_schema_number_new` and is not used again.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_json_schema_number_free(number: *mut GufoJsonSchemaNumber) {
    guard((), || {
        if !number.is_null() {
            // SAFETY: the caller transfers a handle from `Box::into_raw`.
            drop(unsafe { Box::from_raw(number) });
        }
    });
}

/// Reports the exact least common multiple of two `multipleOf` values to
/// `result`, as `digits` `e` `exponent`.
///
/// # Safety
/// `a` and `b` each point to a value; each callback may be called with
/// `context` until this returns.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_json_schema_multiple_of(
    a: *const GufoJsonSchemaValue,
    b: *const GufoJsonSchemaValue,
    result: Option<GufoJsonSchemaBytesFn>,
    error: Option<GufoJsonSchemaBytesFn>,
    context: *mut c_void,
) -> GufoJsonSchemaStatus {
    guard(GufoJsonSchemaStatus::Internal, || {
        // SAFETY: the caller guarantees one value each.
        let (Some([a]), Some([b])) = (unsafe { borrow_all::<1>(a) }, unsafe { borrow_all::<1>(b) })
        else {
            return GufoJsonSchemaStatus::Internal;
        };
        let callbacks = Callbacks::new(context, error);
        let (status, text) = callbacks.status(number::intersect_multiple_of(&a, &b));
        if let Some(text) = text {
            callbacks.bytes(result, &text);
        }
        status
    })
}

/// Writes to `equal` whether two JSON numbers have the same exact value.
///
/// # Safety
/// `a` and `b` point to `a_len` and `b_len` readable bytes (or the length is
/// 0); `equal` is writable; each callback may be called with `context` until
/// this returns.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_json_schema_number_equal(
    a: *const u8,
    a_len: usize,
    b: *const u8,
    b_len: usize,
    equal: *mut u8,
    error: Option<GufoJsonSchemaBytesFn>,
    context: *mut c_void,
) -> GufoJsonSchemaStatus {
    guard(GufoJsonSchemaStatus::Internal, || {
        // SAFETY: the caller guarantees the readable bytes.
        let (Some(a), Some(b)) = (unsafe { gufo_ffi::input(a, a_len) }, unsafe {
            gufo_ffi::input(b, b_len)
        }) else {
            return GufoJsonSchemaStatus::Internal;
        };
        if equal.is_null() {
            return GufoJsonSchemaStatus::Internal;
        }
        let callbacks = Callbacks::new(context, error);
        let (status, found) = callbacks.status(number::equal(a, b));
        if let Some(found) = found {
            // SAFETY: `equal` is writable.
            unsafe { equal.write(u8::from(found)) };
        }
        status
    })
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests;
