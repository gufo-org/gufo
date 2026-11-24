//! The JSON parser, a value tree built from its events and a writer that
//! spells values as C++ `json::Value::dump` does, for Rust callers; and the
//! C interface of the parser, from whose events `gufo::json::parse`
//! (src/core/json.cpp) builds its values.
#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod parse;
mod value;
mod write;
mod write_abi;

pub use parse::{Event, Failure, MAX_DEPTH, Sink, Stopped, parse};
pub use value::{Limits, Number, Object, Value, ValueError, parse_value};
pub use write::{Numbers, Spacing, Style, dump, tojson, write};
pub use write_abi::{GUFO_JSON_DUMP_AS_PARSED, GUFO_JSON_DUMP_SPACED, gufo_json_dump};

use gufo_ffi::guard;
use std::ffi::{c_int, c_void};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GufoJsonParse {
    Complete = 0,
    Rejected = 1,
    NoMemory = 2,
    Stopped = 3,
    Internal = 4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GufoJsonKind {
    Null = 0,
    False = 1,
    True = 2,
    Number = 3,
    String = 4,
    Key = 5,
    BeginArray = 6,
    BeginObject = 7,
    End = 8,
}

/// One event, valid during the callback only; `text` is null when `len` is 0.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoJsonEvent {
    pub kind: GufoJsonKind,
    pub number: f64,
    pub text: *const u8,
    pub len: usize,
}

/// Takes one event; returns nonzero to stop.
pub type GufoJsonEventFn =
    unsafe extern "C" fn(context: *mut c_void, event: *const GufoJsonEvent) -> c_int;
pub type GufoJsonBytesFn = unsafe extern "C" fn(context: *mut c_void, data: *const u8, len: usize);

/// The caller's event callback, which it allows to be called with `context`
/// until the parse returns.
struct Callback {
    context: *mut c_void,
    event: GufoJsonEventFn,
}

impl Sink for Callback {
    #[allow(unsafe_code)]
    fn event(&mut self, event: Event<'_>) -> Result<(), Stopped> {
        let (kind, number, text): (_, _, &[u8]) = match event {
            Event::Null => (GufoJsonKind::Null, 0.0, &[]),
            Event::Bool(false) => (GufoJsonKind::False, 0.0, &[]),
            Event::Bool(true) => (GufoJsonKind::True, 0.0, &[]),
            Event::Number(value, token) => (GufoJsonKind::Number, value, token),
            Event::String(bytes) => (GufoJsonKind::String, 0.0, bytes),
            Event::Key(bytes) => (GufoJsonKind::Key, 0.0, bytes),
            Event::BeginArray => (GufoJsonKind::BeginArray, 0.0, &[]),
            Event::BeginObject => (GufoJsonKind::BeginObject, 0.0, &[]),
            Event::End => (GufoJsonKind::End, 0.0, &[]),
        };
        let event = GufoJsonEvent {
            kind,
            number,
            // An empty buffer's pointer may dangle; C++ gets null instead.
            text: if text.is_empty() {
                std::ptr::null()
            } else {
                text.as_ptr()
            },
            len: text.len(),
        };
        // SAFETY: the caller allows this call (see `Callback`); `event` and
        // the bytes it points to live until it returns.
        match unsafe { (self.event)(self.context, &event) } {
            0 => Ok(()),
            _ => Err(Stopped),
        }
    }
}

/// Parses the JSON text in `data` and reports its values to `event` in
/// document order. On `Rejected`, `error` got the reason unless it is null.
/// Events taken before a failure are void.
///
/// # Safety
/// `data` points to `len` readable bytes that stay unchanged during the
/// call, or `len` is 0; each callback may be called with `context` until this
/// returns.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_json_parse(
    data: *const u8,
    len: usize,
    event: Option<GufoJsonEventFn>,
    error: Option<GufoJsonBytesFn>,
    context: *mut c_void,
) -> GufoJsonParse {
    guard(GufoJsonParse::Internal, || {
        // SAFETY: the caller guarantees `len` readable bytes.
        let Some(text) = (unsafe { gufo_ffi::input(data, len) }) else {
            return GufoJsonParse::Internal;
        };
        let Some(event) = event else {
            return GufoJsonParse::Internal;
        };
        match parse::parse(text, &mut Callback { context, event }) {
            Ok(()) => GufoJsonParse::Complete,
            Err(Failure::Rejected(reason)) => {
                if let Some(error) = error {
                    // SAFETY: the caller allows calling `error` with
                    // `context`; `reason` is static.
                    unsafe { error(context, reason.as_ptr(), reason.len()) };
                }
                GufoJsonParse::Rejected
            }
            Err(Failure::NoMemory) => GufoJsonParse::NoMemory,
            Err(Failure::Stopped) => GufoJsonParse::Stopped,
        }
    })
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Seen {
        kinds: Vec<GufoJsonKind>,
        texts: Vec<Vec<u8>>,
        numbers: Vec<f64>,
        null_text: usize,
        error: Vec<u8>,
        stop_at: Option<usize>,
    }

    fn seen<'a>(context: *mut c_void) -> &'a mut Seen {
        // SAFETY: every test passes a live `Seen` as the context.
        unsafe { &mut *context.cast::<Seen>() }
    }

    unsafe extern "C" fn record(context: *mut c_void, event: *const GufoJsonEvent) -> c_int {
        let seen = seen(context);
        if seen.stop_at == Some(seen.kinds.len()) {
            return 1;
        }
        // SAFETY: the parser passes a live event.
        let event = unsafe { &*event };
        seen.kinds.push(event.kind);
        seen.numbers.push(event.number);
        if event.text.is_null() {
            seen.null_text += 1;
            assert_eq!(event.len, 0);
        } else {
            assert_ne!(event.len, 0);
            // SAFETY: the event's text is readable during the call.
            seen.texts
                .push(unsafe { std::slice::from_raw_parts(event.text, event.len) }.to_vec());
        }
        0
    }

    unsafe extern "C" fn error(context: *mut c_void, data: *const u8, len: usize) {
        // SAFETY: the parser passes readable bytes.
        seen(context).error = unsafe { std::slice::from_raw_parts(data, len) }.to_vec();
    }

    fn run(text: &[u8], seen: &mut Seen) -> GufoJsonParse {
        let context = std::ptr::from_mut(seen).cast();
        // SAFETY: `text` and `seen` outlive the call.
        unsafe {
            gufo_json_parse(
                text.as_ptr(),
                text.len(),
                Some(record),
                Some(error),
                context,
            )
        }
    }

    #[test]
    fn events_cross_the_boundary() {
        let mut seen = Seen::default();
        assert_eq!(
            run(br#"{"a":[1.5,"",true,false,null,"\n"]}"#, &mut seen),
            GufoJsonParse::Complete
        );
        use GufoJsonKind::*;
        assert_eq!(
            seen.kinds,
            [
                BeginObject,
                Key,
                BeginArray,
                Number,
                String,
                True,
                False,
                Null,
                String,
                End,
                End
            ]
        );
        assert_eq!(seen.texts, [b"a".to_vec(), b"1.5".to_vec(), b"\n".to_vec()]);
        assert_eq!(seen.numbers[3], 1.5);
        // Containers, literals and the empty string carry no text.
        assert_eq!(seen.null_text, 8);
    }

    #[test]
    fn rejection_reason_and_stop() {
        let mut seen = Seen::default();
        assert_eq!(run(b"[1,]", &mut seen), GufoJsonParse::Rejected);
        assert_eq!(seen.error, b"bad number");
        let mut seen = Seen {
            stop_at: Some(1),
            ..Seen::default()
        };
        assert_eq!(run(b"[1]", &mut seen), GufoJsonParse::Stopped);
        assert!(seen.error.is_empty());
        assert_eq!(seen.kinds, [GufoJsonKind::BeginArray]);
    }

    #[test]
    fn decoded_texts_arrive_intact() {
        // Escaped strings and keys share the parser's scratch buffer; plain
        // ones point into the text. Each must arrive intact.
        let mut text = String::from("[");
        let mut expected = Vec::new();
        for i in 0..1000 {
            if i > 0 {
                text.push(',');
            }
            if i % 3 == 0 {
                text.push_str(&format!("\"plain {i}\""));
                expected.push(format!("plain {i}"));
            } else {
                text.push_str(&format!("{{\"k\\u0031{i}\":\"e\\n{i}\"}}"));
                expected.push(format!("k1{i}"));
                expected.push(format!("e\n{i}"));
            }
        }
        text.push(']');
        let mut seen = Seen::default();
        assert_eq!(run(text.as_bytes(), &mut seen), GufoJsonParse::Complete);
        let texts: Vec<String> = seen
            .texts
            .iter()
            .map(|t| String::from_utf8(t.clone()).unwrap())
            .collect();
        assert_eq!(texts, expected);
    }

    #[test]
    fn caller_errors_are_internal() {
        let mut seen = Seen::default();
        let context = std::ptr::from_mut(&mut seen).cast();
        // SAFETY: null is passed only with a nonzero length or no callback,
        // which the function rejects before reading.
        unsafe {
            assert_eq!(
                gufo_json_parse(std::ptr::null(), 1, Some(record), None, context),
                GufoJsonParse::Internal
            );
            assert_eq!(
                gufo_json_parse(b"1".as_ptr(), 1, None, None, context),
                GufoJsonParse::Internal
            );
            // An empty text without a pointer is a valid, if empty, input.
            assert_eq!(
                gufo_json_parse(std::ptr::null(), 0, Some(record), None, context),
                GufoJsonParse::Rejected
            );
        }
    }
}
