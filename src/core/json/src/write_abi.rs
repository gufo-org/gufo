//! The C interface of the writer, through which json_write_test compares it
//! with C++ `json::Value::dump` and `tojson`. Gufo itself does not call it.

use crate::value::{Limits, ValueError, parse_value};
use crate::write::{Numbers, Spacing, Style, write};
use crate::{GufoJsonBytesFn, GufoJsonParse};
use gufo_ffi::guard;
use std::ffi::c_void;

/// Writes tokens as parsed (`Numbers::AsParsed`) instead of canonically.
pub const GUFO_JSON_DUMP_AS_PARSED: u32 = 1;
/// Writes `, ` and `: ` (`tojson`) instead of `,` and `:` (`dump`).
pub const GUFO_JSON_DUMP_SPACED: u32 = 2;

/// Parses `data` as `gufo_json_parse` does into a value tree, and passes it
/// to `out` as written in the style `flags` selects. On `Rejected`, `error`
/// got the reason unless it is null.
///
/// # Safety
/// `data` points to `len` readable bytes that stay unchanged during the
/// call, or `len` is 0; each callback may be called with `context` until this
/// returns.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_json_dump(
    data: *const u8,
    len: usize,
    flags: u32,
    out: Option<GufoJsonBytesFn>,
    error: Option<GufoJsonBytesFn>,
    context: *mut c_void,
) -> GufoJsonParse {
    guard(GufoJsonParse::Internal, || {
        // SAFETY: the caller guarantees `len` readable bytes.
        let Some(text) = (unsafe { gufo_ffi::input(data, len) }) else {
            return GufoJsonParse::Internal;
        };
        let Some(out) = out else {
            return GufoJsonParse::Internal;
        };
        if flags & !(GUFO_JSON_DUMP_AS_PARSED | GUFO_JSON_DUMP_SPACED) != 0 {
            return GufoJsonParse::Internal;
        }
        let style = Style {
            numbers: if flags & GUFO_JSON_DUMP_AS_PARSED != 0 {
                Numbers::AsParsed
            } else {
                Numbers::Canonical
            },
            spacing: if flags & GUFO_JSON_DUMP_SPACED != 0 {
                Spacing::Spaced
            } else {
                Spacing::Compact
            },
        };
        let value = match parse_value(text, Limits::NONE) {
            Ok(value) => value,
            Err(failure) => {
                if let (ValueError::Rejected(reason), Some(error)) = (failure, error) {
                    // SAFETY: the caller allows calling `error` with
                    // `context`; `reason` is static.
                    unsafe { error(context, reason.as_ptr(), reason.len()) };
                }
                return status(failure);
            }
        };
        let mut written = String::new();
        if write(&value, style, &mut written).is_err() {
            return GufoJsonParse::NoMemory;
        }
        // SAFETY: the caller allows calling `out` with `context`; `written`
        // lives until it returns.
        unsafe { out(context, written.as_ptr(), written.len()) };
        GufoJsonParse::Complete
    })
}

/// The status for a failed parse. No limits are set, so `TooLarge` is a bug
/// too.
fn status(failure: ValueError) -> GufoJsonParse {
    match failure {
        ValueError::Rejected(_) => GufoJsonParse::Rejected,
        ValueError::NoMemory => GufoJsonParse::NoMemory,
        ValueError::TooLarge | ValueError::Internal => GufoJsonParse::Internal,
    }
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;

    unsafe extern "C" fn keep(context: *mut c_void, data: *const u8, len: usize) {
        // SAFETY: tests pass a live `Vec<u8>` and the writer readable bytes.
        unsafe {
            (*context.cast::<Vec<u8>>()).extend_from_slice(std::slice::from_raw_parts(data, len))
        };
    }

    fn run(text: &[u8], flags: u32) -> (GufoJsonParse, String) {
        let mut seen = Vec::new();
        let context = std::ptr::from_mut(&mut seen).cast();
        // SAFETY: `text` and `seen` outlive the call.
        let status = unsafe {
            gufo_json_dump(
                text.as_ptr(),
                text.len(),
                flags,
                Some(keep),
                Some(keep),
                context,
            )
        };
        (status, String::from_utf8(seen).unwrap())
    }

    #[test]
    fn failures_map_to_statuses() {
        assert_eq!(status(ValueError::Rejected("x")), GufoJsonParse::Rejected);
        assert_eq!(status(ValueError::NoMemory), GufoJsonParse::NoMemory);
        assert_eq!(status(ValueError::Internal), GufoJsonParse::Internal);
        assert_eq!(status(ValueError::TooLarge), GufoJsonParse::Internal);
    }

    #[test]
    fn styles_cross_the_boundary() {
        let text = br#"{"a": [1.0, 2]}"#;
        assert_eq!(
            run(text, 0),
            (GufoJsonParse::Complete, r#"{"a":[1,2]}"#.into())
        );
        assert_eq!(
            run(text, GUFO_JSON_DUMP_AS_PARSED | GUFO_JSON_DUMP_SPACED),
            (GufoJsonParse::Complete, r#"{"a": [1.0, 2]}"#.into())
        );
        assert_eq!(
            run(b"[1,]", 0),
            (GufoJsonParse::Rejected, "bad number".into())
        );
        assert_eq!(run(b"1", 4).0, GufoJsonParse::Internal);
        // SAFETY: the null pointer comes with a nonzero length, which the
        // function rejects before reading.
        let status = unsafe {
            gufo_json_dump(
                std::ptr::null(),
                1,
                0,
                Some(keep),
                None,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(status, GufoJsonParse::Internal);
    }
}
