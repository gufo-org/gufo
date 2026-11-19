//! The JSON text `gufo::json::parse` accepts, as `src/core/json.hpp` read it
//! in Gufo 0.35.0: RFC 8259 with strict UTF-8, paired surrogate escapes,
//! unique object keys, at most 128 nested arrays and objects, and finite
//! numbers that `std::from_chars` reads without leaving the range of a
//! double. Rejections carry that parser's reasons; the first error in
//! document order wins.

use std::borrow::Cow;
use std::collections::HashSet;
use std::collections::TryReserveError;

/// Arrays and objects nested deeper than this are rejected.
pub const MAX_DEPTH: usize = 128;

/// A value or container boundary, in document order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Event<'a> {
    Null,
    Bool(bool),
    /// The value and the token it was read from.
    Number(f64, &'a [u8]),
    /// Decoded UTF-8.
    String(&'a [u8]),
    /// An object member's decoded key; its value follows.
    Key(&'a [u8]),
    BeginArray,
    BeginObject,
    /// Ends the innermost array or object.
    End,
}

/// The sink asked to stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stopped;

pub trait Sink {
    /// Takes the next event, whose text is valid during the call only.
    fn event(&mut self, event: Event<'_>) -> Result<(), Stopped>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Failure {
    /// The text is not JSON; the reason as the C++ parser gave it.
    Rejected(&'static str),
    NoMemory,
    Stopped,
}

impl From<Stopped> for Failure {
    fn from(_: Stopped) -> Self {
        Self::Stopped
    }
}

impl From<TryReserveError> for Failure {
    fn from(_: TryReserveError) -> Self {
        Self::NoMemory
    }
}

fn reject<T>(reason: &'static str) -> Result<T, Failure> {
    Err(Failure::Rejected(reason))
}

/// Parses `text` and reports its values to `sink`.
pub fn parse(text: &[u8], sink: &mut impl Sink) -> Result<(), Failure> {
    let mut parser = Parser {
        s: text,
        i: 0,
        scratch: Vec::new(),
        keys: Vec::new(),
        objects: 0,
    };
    parser.value(sink, 0)?;
    parser.skip_ws();
    if parser.i != text.len() {
        return reject("trailing input");
    }
    Ok(())
}

/// Where a decoded string is: in the text when it has no escapes, else in
/// the parser's scratch buffer until the next string.
#[derive(Debug, Clone, Copy)]
enum Text {
    Input(usize, usize),
    Scratch,
}

/// Up to this many keys, an object's keys are compared one by one.
const FEW_KEYS: usize = 16;

/// The keys of one open object, to reject duplicates. Keys without escapes
/// borrow the text.
enum Keys<'a> {
    Few(Vec<Cow<'a, [u8]>>),
    Many(HashSet<Cow<'a, [u8]>>),
}

impl<'a> Keys<'a> {
    /// Adds `key`; false when the object already has it.
    fn insert(&mut self, key: Cow<'a, [u8]>) -> Result<bool, Failure> {
        match self {
            Self::Few(keys) => {
                if keys.contains(&key) {
                    return Ok(false);
                }
                if keys.len() < FEW_KEYS {
                    keys.try_reserve(1)?;
                    keys.push(key);
                    return Ok(true);
                }
                let mut set = HashSet::new();
                set.try_reserve(2 * FEW_KEYS)?;
                set.extend(keys.drain(..));
                set.insert(key);
                *self = Self::Many(set);
                Ok(true)
            }
            Self::Many(set) => {
                set.try_reserve(1)?;
                Ok(set.insert(key))
            }
        }
    }

    /// Forgets the keys; small buffers are kept for the next object.
    fn clear(&mut self) {
        match self {
            Self::Few(keys) => keys.clear(),
            Self::Many(_) => *self = Self::Few(Vec::new()),
        }
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
    scratch: Vec<u8>,
    /// The keys of each open object, outermost first; entries past `objects`
    /// are cleared buffers kept for reuse.
    keys: Vec<Keys<'a>>,
    objects: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn skip_ws(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.peek() {
            self.i += 1;
        }
    }

    fn text(&self, text: Text) -> &[u8] {
        match text {
            Text::Input(start, end) => &self.s[start..end],
            Text::Scratch => &self.scratch,
        }
    }

    fn value(&mut self, sink: &mut impl Sink, depth: usize) -> Result<(), Failure> {
        self.skip_ws();
        let Some(c) = self.peek() else {
            return reject("unexpected end of input");
        };
        if (c == b'{' || c == b'[') && depth >= MAX_DEPTH {
            return reject("maximum nesting depth exceeded");
        }
        match c {
            b'{' => self.object(sink, depth + 1),
            b'[' => self.array(sink, depth + 1),
            b'"' => {
                let text = self.string()?;
                Ok(sink.event(Event::String(self.text(text)))?)
            }
            b't' | b'f' => self.boolean(sink),
            b'n' => self.null(sink),
            _ => self.number(sink),
        }
    }

    fn object(&mut self, sink: &mut impl Sink, depth: usize) -> Result<(), Failure> {
        self.i += 1;
        sink.event(Event::BeginObject)?;
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(sink.event(Event::End)?);
        }
        let level = self.objects;
        if self.keys.len() == level {
            self.keys.try_reserve(1)?;
            self.keys.push(Keys::Few(Vec::new()));
        }
        self.objects += 1;
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return reject("expected object key string");
            }
            let key = self.string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return reject("expected ':' after key");
            }
            self.i += 1;
            let s: &'a [u8] = self.s;
            let unique = match key {
                Text::Input(start, end) => Cow::Borrowed(&s[start..end]),
                Text::Scratch => {
                    let mut owned = Vec::new();
                    owned.try_reserve_exact(self.scratch.len())?;
                    owned.extend_from_slice(&self.scratch);
                    Cow::Owned(owned)
                }
            };
            if !self.keys[level].insert(unique)? {
                return reject("duplicate object key");
            }
            sink.event(Event::Key(self.text(key)))?;
            self.value(sink, depth)?;
            self.skip_ws();
            match self.peek() {
                None => return reject("unterminated object"),
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    break;
                }
                Some(_) => return reject("expected ',' or '}' in object"),
            }
        }
        self.objects -= 1;
        self.keys[level].clear();
        Ok(sink.event(Event::End)?)
    }

    fn array(&mut self, sink: &mut impl Sink, depth: usize) -> Result<(), Failure> {
        self.i += 1;
        sink.event(Event::BeginArray)?;
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(sink.event(Event::End)?);
        }
        loop {
            self.value(sink, depth)?;
            self.skip_ws();
            match self.peek() {
                None => return reject("unterminated array"),
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(sink.event(Event::End)?);
                }
                Some(_) => return reject("expected ',' or ']' in array"),
            }
        }
    }

    /// The string at the opening quote. Bytes up to the first escape stay in
    /// the text; from there on they are decoded into the scratch buffer.
    fn string(&mut self) -> Result<Text, Failure> {
        self.i += 1;
        let start = self.i;
        let (end, stop) = self.plain(start)?;
        match stop {
            b'"' => {
                self.i = end + 1;
                return Ok(Text::Input(start, end));
            }
            b'\\' => {}
            _ => return reject("unescaped control character"),
        }
        self.scratch.clear();
        append(&mut self.scratch, &self.s[start..end])?;
        self.i = end;
        loop {
            self.i += 1;
            self.escape()?;
            let from = self.i;
            let (end, stop) = self.plain(from)?;
            append(&mut self.scratch, &self.s[from..end])?;
            match stop {
                b'"' => {
                    self.i = end + 1;
                    return Ok(Text::Scratch);
                }
                b'\\' => self.i = end,
                _ => return reject("unescaped control character"),
            }
        }
    }

    /// The end of the unescaped run from `from`, at a quote, backslash or
    /// control byte, and that byte, once the run passed as UTF-8. A byte
    /// sequence cut short by the stop byte or the end of the text is invalid
    /// UTF-8, as it was when the C++ parser reached it first.
    fn plain(&self, from: usize) -> Result<(usize, u8), Failure> {
        let s = self.s;
        // Most keys and strings are short: their first bytes are checked one
        // by one, then the rest of a long run eight bytes at a time.
        let mut i = from;
        let short_end = s.len().min(from + SHORT_RUN);
        while i < short_end {
            let c = s[i];
            if c == b'"' || c == b'\\' || c < 0x20 {
                return Ok((i, c));
            }
            i += if c < 0x80 { 1 } else { utf8_sequence(&s[i..])? };
        }
        let rest = &s[i..];
        let found = special(rest);
        if std::str::from_utf8(&rest[..found.unwrap_or(rest.len())]).is_err() {
            return reject("invalid UTF-8");
        }
        match found {
            Some(at) => Ok((i + at, rest[at])),
            None => reject("unterminated string"),
        }
    }

    /// Decodes the escape after a backslash into the scratch buffer.
    fn escape(&mut self) -> Result<(), Failure> {
        let Some(e) = self.peek() else {
            return reject("bad escape");
        };
        self.i += 1;
        let byte = match e {
            b'"' => b'"',
            b'\\' => b'\\',
            b'/' => b'/',
            b'b' => 8,
            b'f' => 12,
            b'n' => b'\n',
            b'r' => b'\r',
            b't' => b'\t',
            b'u' => {
                let mut code = self.hex_quad()?;
                if (0xD800..=0xDBFF).contains(&code) {
                    if self.i + 2 > self.s.len()
                        || self.s[self.i] != b'\\'
                        || self.s[self.i + 1] != b'u'
                    {
                        return reject("unpaired high surrogate");
                    }
                    self.i += 2;
                    let low = self.hex_quad()?;
                    if !(0xDC00..=0xDFFF).contains(&low) {
                        return reject("invalid low surrogate");
                    }
                    code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                } else if (0xDC00..=0xDFFF).contains(&code) {
                    return reject("unpaired low surrogate");
                }
                // Surrogates are paired or rejected above, so `code` is a
                // scalar value.
                let Some(scalar) = char::from_u32(code) else {
                    return reject("bad \\u");
                };
                let mut buffer = [0; 4];
                return append(
                    &mut self.scratch,
                    scalar.encode_utf8(&mut buffer).as_bytes(),
                );
            }
            _ => return reject("bad escape"),
        };
        append(&mut self.scratch, &[byte])
    }

    fn hex_quad(&mut self) -> Result<u32, Failure> {
        if self.i + 4 > self.s.len() {
            return reject("bad \\u");
        }
        let mut code = 0;
        for _ in 0..4 {
            let digit = match self.s[self.i] {
                h @ b'0'..=b'9' => h - b'0',
                h @ b'a'..=b'f' => h - b'a' + 10,
                h @ b'A'..=b'F' => h - b'A' + 10,
                _ => return reject("bad hex"),
            };
            self.i += 1;
            code = (code << 4) | u32::from(digit);
        }
        Ok(code)
    }

    fn boolean(&mut self, sink: &mut impl Sink) -> Result<(), Failure> {
        let rest = &self.s[self.i..];
        let value = if rest.starts_with(b"true") {
            true
        } else if rest.starts_with(b"false") {
            false
        } else {
            return reject("bad literal");
        };
        self.i += if value { 4 } else { 5 };
        Ok(sink.event(Event::Bool(value))?)
    }

    fn null(&mut self, sink: &mut impl Sink) -> Result<(), Failure> {
        if !self.s[self.i..].starts_with(b"null") {
            return reject("bad literal");
        }
        self.i += 4;
        Ok(sink.event(Event::Null)?)
    }

    fn digits(&mut self) {
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.i += 1;
        }
    }

    fn number(&mut self, sink: &mut impl Sink) -> Result<(), Failure> {
        let start = self.i;
        let negative = self.peek() == Some(b'-');
        if negative {
            self.i += 1;
        }
        let digits = self.i;
        match self.peek() {
            Some(b'0') => {
                self.i += 1;
                if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    return reject("bad number");
                }
            }
            Some(b'1'..=b'9') => self.digits(),
            _ => return reject("bad number"),
        }
        let integer_end = self.i;
        if self.peek() == Some(b'.') {
            self.i += 1;
            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return reject("bad number");
            }
            self.digits();
        }
        let mantissa_end = self.i;
        if let Some(b'e' | b'E') = self.peek() {
            self.i += 1;
            if let Some(b'-' | b'+') = self.peek() {
                self.i += 1;
            }
            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return reject("bad number");
            }
            self.digits();
        }
        let s: &'a [u8] = self.s;
        let token = &s[start..self.i];
        // Integers of at most 15 digits are below 2^53 and convert exactly.
        if self.i == integer_end && integer_end - digits <= 15 {
            let magnitude = s[digits..integer_end]
                .iter()
                .fold(0u64, |n, c| n * 10 + u64::from(c - b'0')) as f64;
            let value = if negative { -magnitude } else { magnitude };
            return Ok(sink.event(Event::Number(value, token))?);
        }
        let nonzero = s[start..mantissa_end]
            .iter()
            .any(|c| (b'1'..=b'9').contains(c));
        // Both parsers round correctly, but Rust reads an exponent exactly up
        // to 655,359 only and `std::from_chars` up to 2^28, while both count
        // every mantissa digit. A long mantissa can offset a larger exponent,
        // so such a token is read as 0.<digits from the first nonzero>e<shift>.
        let mut scaled = Vec::new();
        let mut text = token;
        if nonzero && mantissa_end < self.i {
            let mut exponent = 0i64;
            let mut sign = 1;
            for &c in &s[mantissa_end + 1..self.i] {
                match c {
                    b'-' => sign = -1,
                    b'+' => {}
                    // The saturation of `std::from_chars` (fast_float).
                    _ if exponent < 0x1000_0000 => exponent = 10 * exponent + i64::from(c - b'0'),
                    _ => {}
                }
            }
            if exponent > 0xffff {
                let mantissa = s[digits..mantissa_end]
                    .iter()
                    .filter(|c| c.is_ascii_digit());
                let zeros = mantissa.clone().take_while(|&&c| c == b'0').count();
                let shift = (integer_end - digits) as i64 - zeros as i64 + sign * exponent;
                // Then the value is at least 10^65535 or below 10^-65536.
                if shift.abs() > 0xffff {
                    return reject("bad number");
                }
                // The token's sign and digits, "0." and "e-" with five
                // exponent digits fit, so nothing below allocates.
                scaled.try_reserve(mantissa_end - start + 16)?;
                if negative {
                    scaled.push(b'-');
                }
                scaled.extend_from_slice(b"0.");
                scaled.extend(mantissa.skip(zeros));
                scaled.extend_from_slice(if shift < 0 { b"e-" } else { b"e" });
                let magnitude = shift.unsigned_abs();
                scaled.extend(
                    [10_000, 1_000, 100, 10, 1].map(|unit| b'0' + (magnitude / unit % 10) as u8),
                );
                text = &scaled;
            }
        }
        // The grammar above admits ASCII only. `std::from_chars` also reports
        // overflow and underflow to zero as out of range, while subnormal
        // results stay.
        let Some(value) = std::str::from_utf8(text)
            .ok()
            .and_then(|text| text.parse::<f64>().ok())
        else {
            return reject("bad number");
        };
        if !value.is_finite() || (value == 0.0 && nonzero) {
            return reject("bad number");
        }
        Ok(sink.event(Event::Number(value, token))?)
    }
}

/// Up to this many bytes of an unescaped run are checked one by one.
const SHORT_RUN: usize = 32;

/// The length of the UTF-8 sequence at the start of `bytes`, whose first byte
/// is not ASCII, by the C++ parser's rules (the well-formed sequences of
/// Unicode).
fn utf8_sequence(bytes: &[u8]) -> Result<usize, Failure> {
    let lead = bytes[0];
    let length = match lead {
        0xC2..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF4 => 4,
        _ => return reject("invalid UTF-8"),
    };
    if length > bytes.len() {
        return reject("invalid UTF-8");
    }
    let (low, high) = match lead {
        0xE0 => (0xA0, 0xBF),
        0xED => (0x80, 0x9F),
        0xF0 => (0x90, 0xBF),
        0xF4 => (0x80, 0x8F),
        _ => (0x80, 0xBF),
    };
    if !(low..=high).contains(&bytes[1])
        || bytes[2..length]
            .iter()
            .any(|&c| !(0x80..=0xBF).contains(&c))
    {
        return reject("invalid UTF-8");
    }
    Ok(length)
}

fn append(buffer: &mut Vec<u8>, bytes: &[u8]) -> Result<(), Failure> {
    buffer.try_reserve(bytes.len())?;
    buffer.extend_from_slice(bytes);
    Ok(())
}

/// The index of the first quote, backslash or control byte. Eight bytes are
/// tested at once: the lowest byte each test flags is a true match, since a
/// borrow only carries past a matching byte.
fn special(bytes: &[u8]) -> Option<usize> {
    const ONES: u64 = 0x0101_0101_0101_0101;
    const HIGH: u64 = 0x8080_8080_8080_8080;
    let mut chunks = bytes.chunks_exact(8);
    let mut offset = 0;
    for chunk in &mut chunks {
        let mut word = [0; 8];
        word.copy_from_slice(chunk);
        let word = u64::from_le_bytes(word);
        let quote = word ^ (ONES * u64::from(b'"'));
        let backslash = word ^ (ONES * u64::from(b'\\'));
        let found = ((quote.wrapping_sub(ONES) & !quote)
            | (backslash.wrapping_sub(ONES) & !backslash)
            | (word.wrapping_sub(ONES * 0x20) & !word))
            & HIGH;
        if found != 0 {
            return Some(offset + (found.trailing_zeros() / 8) as usize);
        }
        offset += 8;
    }
    chunks
        .remainder()
        .iter()
        .position(|&c| c == b'"' || c == b'\\' || c < 0x20)
        .map(|at| offset + at)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records events as text: `n`, `t`, `f`, numbers as `#value|token`,
    /// strings as `"…"`, keys as `k…`, `[`, `{` and `)`.
    #[derive(Default)]
    struct Record {
        events: Vec<String>,
        stop_after: Option<usize>,
    }

    impl Sink for Record {
        fn event(&mut self, event: Event<'_>) -> Result<(), Stopped> {
            if self.stop_after == Some(self.events.len()) {
                return Err(Stopped);
            }
            let text = |bytes: &[u8]| String::from_utf8(bytes.to_vec()).unwrap();
            self.events.push(match event {
                Event::Null => "n".into(),
                Event::Bool(true) => "t".into(),
                Event::Bool(false) => "f".into(),
                Event::Number(value, token) => format!("#{value:?}|{}", text(token)),
                Event::String(bytes) => format!("\"{}", text(bytes)),
                Event::Key(bytes) => format!("k{}", text(bytes)),
                Event::BeginArray => "[".into(),
                Event::BeginObject => "{".into(),
                Event::End => ")".into(),
            });
            Ok(())
        }
    }

    fn events(text: &[u8]) -> Result<Vec<String>, Failure> {
        let mut record = Record::default();
        parse(text, &mut record).map(|()| record.events)
    }

    fn rejected(text: &[u8]) -> &'static str {
        match events(text) {
            Err(Failure::Rejected(reason)) => reason,
            other => panic!("{:?} gave {other:?}", String::from_utf8_lossy(text)),
        }
    }

    #[test]
    fn values_in_document_order() {
        assert_eq!(
            events(b" {\"a\": [1, -2.5e1, true, false, null], \"b\": {}, \"c\": \"x\"} ").unwrap(),
            [
                "{",
                "ka",
                "[",
                "#1.0|1",
                "#-25.0|-2.5e1",
                "t",
                "f",
                "n",
                ")",
                "kb",
                "{",
                ")",
                "kc",
                "\"x",
                ")"
            ]
        );
        assert_eq!(events(b"[]").unwrap(), ["[", ")"]);
        assert_eq!(events(b"\"\"").unwrap(), ["\""]);
        assert_eq!(events(b"-0").unwrap(), ["#-0.0|-0"]);
    }

    #[test]
    fn escapes_decode() {
        assert_eq!(
            events(br#""a\"b\\c\/d\b\f\n\r\t\u00e9\u20ac\ud83d\ude00\u0000z""#).unwrap(),
            ["\"a\"b\\c/d\u{8}\u{c}\n\r\t\u{e9}\u{20ac}\u{1f600}\u{0}z"]
        );
        // A key decoded into the scratch buffer survives the value's string.
        assert_eq!(
            events(br#"{"k\u0031": "v\n"}"#).unwrap(),
            ["{", "kk1", "\"v\n", ")"]
        );
    }

    #[test]
    fn every_reason() {
        let cases: &[(&[u8], &str)] = &[
            (b"", "unexpected end of input"),
            (b"  ", "unexpected end of input"),
            (b"[1,", "unexpected end of input"),
            (b"1 2", "trailing input"),
            (b"{1:2}", "expected object key string"),
            (b"{\"a\" 1}", "expected ':' after key"),
            (b"{\"a\":1,\"a\":2}", "duplicate object key"),
            (b"{\"a\":1,\"\\u0061\":2}", "duplicate object key"),
            (b"{\"a\":1", "unterminated object"),
            (b"{\"a\":1]", "expected ',' or '}' in object"),
            (b"[1", "unterminated array"),
            (b"[1}", "expected ',' or ']' in array"),
            (b"\"\\u12", "bad \\u"),
            (b"\"\\u12g4\"", "bad hex"),
            (b"\"\\", "bad escape"),
            (b"\"\\x\"", "bad escape"),
            (b"\"\\ud800\"", "unpaired high surrogate"),
            (b"\"\\ud800\\u0041\"", "invalid low surrogate"),
            (b"\"\\udc00\"", "unpaired low surrogate"),
            (b"\"a\x01\"", "unescaped control character"),
            (b"\"\xc0\x80\"", "invalid UTF-8"),
            (b"\"\xe2\x82\"", "invalid UTF-8"),
            (b"\"\xed\xa0\x80\"", "invalid UTF-8"),
            (b"\"\xf4\x90\x80\x80\"", "invalid UTF-8"),
            (b"\"\xe2", "invalid UTF-8"),
            (b"\"abc", "unterminated string"),
            (b"tru", "bad literal"),
            (b"nul", "bad literal"),
            (b"-", "bad number"),
            (b"01", "bad number"),
            (b"1.", "bad number"),
            (b"1e", "bad number"),
            (b"1e+", "bad number"),
            (b"+1", "bad number"),
            (b"}", "bad number"),
            (b"1e400", "bad number"),
            (b"-1e400", "bad number"),
            (b"1e-400", "bad number"),
            (b"2.4703282292062327e-324", "bad number"),
        ];
        for &(text, reason) in cases {
            assert_eq!(
                rejected(text),
                reason,
                "{:?}",
                String::from_utf8_lossy(text)
            );
        }
    }

    #[test]
    fn range_matches_from_chars() {
        // Subnormal results and zeros with any exponent stay.
        assert_eq!(events(b"5e-324").unwrap(), ["#5e-324|5e-324"]);
        assert_eq!(events(b"1e-310").unwrap(), ["#1e-310|1e-310"]);
        assert_eq!(events(b"0e-400").unwrap(), ["#0.0|0e-400"]);
        assert_eq!(events(b"0.000e999").unwrap(), ["#0.0|0.000e999"]);
        assert_eq!(
            events(b"1.7976931348623158e308").unwrap(),
            ["#1.7976931348623157e308|1.7976931348623158e308"]
        );
        assert_eq!(rejected(b"1.7976931348623159e308"), "bad number");
        // Long mantissas offset exponents that Rust alone reads saturated.
        let zeros = "0".repeat(700_000);
        for (token, value) in [
            (format!("1{zeros}e-700001"), 0.1),
            (format!("-1{zeros}e-700001"), -0.1),
            (format!("0.{zeros}1e700001"), 1.0),
            (format!("1{}e-655360", "0".repeat(655_300)), 1e-60),
            (format!("0.{zeros}e999999"), 0.0),
            ("0e4294967296".into(), 0.0),
        ] {
            assert_eq!(
                events(token.as_bytes()).unwrap(),
                [format!("#{value:?}|{token}")]
            );
        }
        for token in [
            format!("1{zeros}e-1000000"),
            format!("1{zeros}e-600000"),
            "1e4294967296".into(),
            "1e-4294967296".into(),
        ] {
            assert_eq!(rejected(token.as_bytes()), "bad number");
        }
    }

    #[test]
    fn short_integers_match_the_general_path() {
        let mut tokens = vec![
            "0".to_string(),
            "-0".into(),
            "999999999999999".into(),
            "-999999999999999".into(),
            "9999999999999999".into(),
            "9007199254740993".into(),
        ];
        let mut n: u64 = 1;
        for _ in 0..19 {
            tokens.push(n.to_string());
            tokens.push((n - 1).to_string());
            n *= 10;
        }
        for token in tokens {
            let mut record = Record::default();
            parse(token.as_bytes(), &mut record).unwrap();
            let expected: f64 = token.parse().unwrap();
            assert_eq!(record.events, [format!("#{expected:?}|{token}")], "{token}");
        }
        assert_eq!(
            events(b"-0").unwrap(),
            [format!("#{:?}|-0", -0.0f64)],
            "the sign of zero stays"
        );
    }

    #[test]
    fn nesting_limit() {
        let deep = |n: usize| [vec![b'['; n], vec![b']'; n]].concat();
        assert!(events(&deep(MAX_DEPTH)).is_ok());
        assert_eq!(
            rejected(&deep(MAX_DEPTH + 1)),
            "maximum nesting depth exceeded"
        );
        let objects = |n: usize| {
            let mut text = b"{\"a\":".repeat(n);
            text.extend_from_slice(b"1");
            text.extend(std::iter::repeat_n(b'}', n));
            text
        };
        assert!(events(&objects(MAX_DEPTH)).is_ok());
        assert_eq!(
            rejected(&objects(MAX_DEPTH + 1)),
            "maximum nesting depth exceeded"
        );
    }

    #[test]
    fn many_keys_switch_to_a_set() {
        let keys = (0..100)
            .map(|k| format!("\"k{k}\":{k}"))
            .collect::<Vec<_>>();
        let mut text = format!("{{{}}}", keys.join(","));
        assert!(events(text.as_bytes()).is_ok());
        text.insert_str(text.len() - 1, ",\"k\\u00357\":0");
        assert_eq!(rejected(text.as_bytes()), "duplicate object key");
        // Sibling objects keep their own keys.
        assert!(events(b"[{\"a\":{\"a\":1}},{\"a\":2}]").is_ok());
    }

    #[test]
    fn stop_is_reported() {
        let mut record = Record {
            stop_after: Some(2),
            ..Record::default()
        };
        assert_eq!(parse(b"[1,2,3]", &mut record), Err(Failure::Stopped));
        assert_eq!(record.events, ["[", "#1.0|1"]);
    }

    thread_local! {
        /// Allocations on this thread that succeed before one fails;
        /// negative never fails.
        static ALLOCATIONS_LEFT: std::cell::Cell<i64> = const { std::cell::Cell::new(-1) };
    }

    struct FailingAllocator;

    #[allow(unsafe_code)]
    // SAFETY: forwards to the system allocator, or reports a failed
    // allocation with null.
    unsafe impl std::alloc::GlobalAlloc for FailingAllocator {
        unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
            let fail = ALLOCATIONS_LEFT
                .try_with(|left| {
                    let n = left.get();
                    if n >= 0 {
                        left.set(n - 1);
                    }
                    n == 0
                })
                .unwrap_or(false);
            if fail {
                return std::ptr::null_mut();
            }
            // SAFETY: the caller's layout, as `GlobalAlloc::alloc` requires.
            unsafe { std::alloc::System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
            // SAFETY: `ptr` came from `System.alloc` with `layout`.
            unsafe { std::alloc::System.dealloc(ptr, layout) }
        }
    }

    #[global_allocator]
    static ALLOCATOR: FailingAllocator = FailingAllocator;

    /// Every allocation of the parser may fail and is reported as
    /// `NoMemory`; one that cannot fail aborts the test instead.
    #[test]
    fn failed_allocations_report_no_memory() {
        struct Ignore;
        impl Sink for Ignore {
            fn event(&mut self, _: Event<'_>) -> Result<(), Stopped> {
                Ok(())
            }
        }
        // Escaped strings and keys, nested objects, the switch to a key set
        // and a rescaled number.
        let mut text = String::from(r#"{"é":"a\nb","o":{"p":[1]},"#);
        for k in 0..20 {
            text.push_str(&format!("\"k{k}\":{k},"));
        }
        text.push_str(&format!("\"x\":1{}e-70001}}", "0".repeat(70_000)));
        let mut failures = 0;
        loop {
            ALLOCATIONS_LEFT.with(|left| left.set(failures));
            let result = parse(text.as_bytes(), &mut Ignore);
            let unused = ALLOCATIONS_LEFT.with(|left| left.replace(-1)) >= 0;
            if unused {
                assert_eq!(result, Ok(()));
                break;
            }
            assert_eq!(result, Err(Failure::NoMemory), "allocation {failures}");
            failures += 1;
        }
        assert!(failures >= 6, "{failures} allocations");
    }

    #[test]
    fn utf8_sequence_matches_std() {
        let tails = [0x00, 0x7f, 0x80, 0x8f, 0x90, 0x9f, 0xa0, 0xbf, 0xc0, 0xff];
        for lead in 0x80..=0xffu8 {
            for second in 0..=0xffu8 {
                for &third in &tails {
                    for &fourth in &tails {
                        let bytes = [lead, second, third, fourth];
                        for cut in 1..=4 {
                            let bytes = &bytes[..cut];
                            let expected = (2..=bytes.len()).find(|&n| {
                                std::str::from_utf8(&bytes[..n])
                                    .is_ok_and(|text| text.chars().count() == 1)
                            });
                            assert_eq!(utf8_sequence(bytes).ok(), expected, "{bytes:x?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn special_finds_the_first_match() {
        for len in 0..40 {
            for at in 0..len {
                for stop in [b'"', b'\\', 0, 0x1f] {
                    let mut bytes = vec![b'a'; len];
                    // Bytes that only resemble a match must not stop it.
                    for (k, byte) in bytes.iter_mut().enumerate() {
                        *byte = [b'a', 0x20, 0x7f, 0x80, 0xa2, 0xdc, 0xff, b'#'][k % 8];
                    }
                    bytes[at] = stop;
                    if at + 1 < len {
                        bytes[at + 1] = b'"';
                    }
                    assert_eq!(special(&bytes), Some(at), "{bytes:?}");
                }
            }
            assert_eq!(special(&vec![b'x'; len]), None);
        }
    }
}
