//! Writes a value as C++ `json::Value::dump` (src/core/json.hpp) does, byte
//! for byte: `Spacing::Compact` is `dump`, `Spacing::Spaced` is `tojson`.
//! Strings escape `"`, `\`, `\b`, `\f`, `\n`, `\r` and `\t` by name and other
//! bytes below 0x20 as `\u00xx`; DEL and non-ASCII pass through unchanged.
//! Canonical numbers spell integers up to 2^53 in magnitude in decimal (except
//! -0) and other values as `std::to_chars` writes the shortest round-trip
//! form.

use crate::value::{Number, Value};
use std::collections::TryReserveError;
use std::fmt::Write as _;

/// How numbers read by the parser are written; C++ `Value::Numbers`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Numbers {
    /// Every number from its value (`kCanonical`).
    #[default]
    Canonical,
    /// The token the parser read; numbers built from values stay canonical
    /// (`kAsParsed`).
    AsParsed,
}

/// The separators between items and after keys.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Spacing {
    /// `,` and `:`, as C++ `dump` writes them.
    #[default]
    Compact,
    /// `, ` and `: `, as C++ `tojson` (Python's `json.dumps`) writes them.
    Spaced,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Style {
    pub numbers: Numbers,
    pub spacing: Spacing,
}

/// C++ `value.dump(numbers)`.
pub fn dump(value: &Value, numbers: Numbers) -> Result<String, TryReserveError> {
    to_string(
        value,
        Style {
            numbers,
            spacing: Spacing::Compact,
        },
    )
}

/// C++ `value.tojson(numbers)`.
pub fn tojson(value: &Value, numbers: Numbers) -> Result<String, TryReserveError> {
    to_string(
        value,
        Style {
            numbers,
            spacing: Spacing::Spaced,
        },
    )
}

fn to_string(value: &Value, style: Style) -> Result<String, TryReserveError> {
    let mut out = String::new();
    write(value, style, &mut out)?;
    Ok(out)
}

fn put(out: &mut String, text: &str) -> Result<(), TryReserveError> {
    out.try_reserve(text.len())?;
    out.push_str(text);
    Ok(())
}

/// A container being written: the items it has left and whether one was
/// written, which the next follows after a comma.
struct Frame<'a> {
    items: Items<'a>,
    started: bool,
}

enum Items<'a> {
    Array(std::slice::Iter<'a, Value>),
    Object(std::slice::Iter<'a, (String, Value)>),
}

/// Appends `value` in `style` to `out`. Nesting is written with an explicit
/// stack, so deep values built in code cannot overflow the call stack.
pub fn write(value: &Value, style: Style, out: &mut String) -> Result<(), TryReserveError> {
    let (comma, colon) = match style.spacing {
        Spacing::Compact => (",", ":"),
        Spacing::Spaced => (", ", ": "),
    };
    let mut stack: Vec<Frame<'_>> = Vec::new();
    let mut next = value;
    loop {
        let items = match next {
            Value::Null => put(out, "null").map(|()| None)?,
            Value::Bool(true) => put(out, "true").map(|()| None)?,
            Value::Bool(false) => put(out, "false").map(|()| None)?,
            Value::Number(value) => number(value, style.numbers, out).map(|()| None)?,
            Value::String(text) => string(text, out).map(|()| None)?,
            Value::Array(items) => {
                put(out, "[")?;
                Some(Items::Array(items.iter()))
            }
            Value::Object(object) => {
                put(out, "{")?;
                Some(Items::Object(object.members().iter()))
            }
        };
        if let Some(items) = items {
            stack.try_reserve(1)?;
            stack.push(Frame {
                items,
                started: false,
            });
        }
        // The innermost container's next value; finished ones close.
        loop {
            let Some(frame) = stack.last_mut() else {
                return Ok(());
            };
            let (key, value) = match &mut frame.items {
                Items::Array(items) => (None, items.next()),
                Items::Object(members) => match members.next() {
                    Some((key, value)) => (Some(key), Some(value)),
                    None => (None, None),
                },
            };
            let Some(value) = value else {
                let close = match frame.items {
                    Items::Array(_) => "]",
                    Items::Object(_) => "}",
                };
                put(out, close)?;
                stack.pop();
                continue;
            };
            if frame.started {
                put(out, comma)?;
            }
            frame.started = true;
            if let Some(key) = key {
                string(key, out)?;
                put(out, colon)?;
            }
            next = value;
            break;
        }
    }
}

/// A quoted string, escaped as C++ `Value::escape` does.
fn string(text: &str, out: &mut String) -> Result<(), TryReserveError> {
    out.try_reserve(text.len() + 2)?;
    out.push('"');
    let mut plain = 0;
    for (i, byte) in text.bytes().enumerate() {
        let named = match byte {
            b'"' => "\\\"",
            b'\\' => "\\\\",
            0x08 => "\\b",
            0x0c => "\\f",
            b'\n' => "\\n",
            b'\r' => "\\r",
            b'\t' => "\\t",
            0..0x20 => "",
            _ => continue,
        };
        put(out, &text[plain..i])?;
        if named.is_empty() {
            const HEX: &[u8; 16] = b"0123456789abcdef";
            put(out, "\\u00")?;
            out.try_reserve(2)?;
            out.push(char::from(HEX[usize::from(byte >> 4)]));
            out.push(char::from(HEX[usize::from(byte & 15)]));
        } else {
            put(out, named)?;
        }
        plain = i + 1;
    }
    put(out, &text[plain..])?;
    put(out, "\"")
}

fn number(number: &Number, numbers: Numbers, out: &mut String) -> Result<(), TryReserveError> {
    match (numbers, number.token()) {
        (Numbers::AsParsed, Some(token)) if !token.is_empty() => put(out, token),
        _ => canonical(number.value(), out),
    }
}

/// Integers whose magnitude is at most this keep their decimal spelling.
const EXACT_INTEGERS: f64 = 9_007_199_254_740_992.0; // 2^53

/// One formatted number. The longest spelling of a double, or of an integer
/// below 2^128, is 40 bytes.
struct Buffer {
    bytes: [u8; 48],
    len: usize,
}

impl Buffer {
    fn new(args: std::fmt::Arguments<'_>) -> Self {
        let mut buffer = Self {
            bytes: [0; 48],
            len: 0,
        };
        // No spelling exceeds the buffer.
        let _ = buffer.write_fmt(args);
        buffer
    }

    fn text(&self) -> &str {
        std::str::from_utf8(&self.bytes[..self.len]).unwrap_or("")
    }
}

impl std::fmt::Write for Buffer {
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        let end = self.len + text.len();
        let slot = self.bytes.get_mut(self.len..end).ok_or(std::fmt::Error)?;
        slot.copy_from_slice(text.as_bytes());
        self.len = end;
        Ok(())
    }
}

/// `value` as C++ `dump` writes it with `Numbers::kCanonical`.
fn canonical(value: f64, out: &mut String) -> Result<(), TryReserveError> {
    let integral = value.trunc() == value
        && value.abs() <= EXACT_INTEGERS
        && !(value == 0.0 && value.is_sign_negative());
    if integral {
        return put(out, Buffer::new(format_args!("{}", value as i64)).text());
    }
    shortest(value, out)
}

/// The shortest digits that read back as `value`, closest to it, in Rust's
/// `d[.ddd]e[-]x` spelling. Rust's `{:e}` finds the shortest count, but when
/// `value` lies exactly halfway between two such spellings it rounds up,
/// where `std::to_chars` (Ryu) rounds to even: 2.49695587158203125 is
/// 2.4969558715820312. Rounding `value` to that count of digits rounds ties
/// to even; it is the answer unless it no longer reads back, which happens
/// below a power of two, whose lower neighbour is nearer.
fn digits(value: f64) -> Buffer {
    let shortest = Buffer::new(format_args!("{value:e}"));
    let text = shortest.text();
    let mantissa = text.split_once('e').map_or(text, |(m, _)| m);
    let count = mantissa.bytes().filter(u8::is_ascii_digit).count();
    let rounded = Buffer::new(format_args!("{value:.*e}", count.saturating_sub(1)));
    if rounded.text().parse::<f64>() == Ok(value) {
        rounded
    } else {
        shortest
    }
}

/// `value` as `std::to_chars` writes it without a format: the fewest
/// characters that read back as `value`, fixed notation on a tie with
/// scientific (`d[.ddd]e±xx`), and the closest such spelling.
fn shortest(value: f64, out: &mut String) -> Result<(), TryReserveError> {
    let scientific = digits(value);
    let scientific = scientific.text();
    let (sign, rest) = match scientific.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", scientific),
    };
    let (mantissa, exponent) = rest.split_once('e').unwrap_or((rest, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let (lead, tail) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = lead.len() + tail.len();
    let magnitude = exponent.unsigned_abs();
    let exponent_digits = if magnitude < 100 { 2 } else { 3 };
    let scientific_len = digits + usize::from(!tail.is_empty()) + 2 + exponent_digits;
    let fixed_len = if exponent >= 0 {
        let whole = exponent as usize + 1;
        if digits <= whole { whole } else { digits + 1 }
    } else {
        // 0.000ddd
        digits + 1 + magnitude as usize
    };
    put(out, sign)?;
    if scientific_len < fixed_len {
        put(out, lead)?;
        if !tail.is_empty() {
            put(out, ".")?;
            put(out, tail)?;
        }
        put(out, if exponent < 0 { "e-" } else { "e+" })?;
        if magnitude < 10 {
            put(out, "0")?;
        }
        return put(out, Buffer::new(format_args!("{magnitude}")).text());
    }
    if exponent < 0 {
        put(out, "0.")?;
        for _ in 1..magnitude {
            put(out, "0")?;
        }
        put(out, lead)?;
        return put(out, tail);
    }
    let whole = exponent as usize + 1;
    if digits <= whole {
        // Of the spellings this long, the exact integer is the closest, where
        // padding the digits with zeros would not be: 2^55 is
        // 36028797018963968, not 36028797018963970. Fixed notation wins only
        // below 10^22, so the integer fits a u128.
        return put(
            out,
            Buffer::new(format_args!("{}", value.abs() as u128)).text(),
        );
    }
    put(out, lead)?;
    put(out, &tail[..whole - 1])?;
    put(out, ".")?;
    put(out, &tail[whole - 1..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{Limits, Object, parse_value};

    fn parsed(text: &str) -> Value {
        parse_value(text.as_bytes(), Limits::NONE).unwrap()
    }

    fn number(value: f64) -> String {
        dump(
            &Value::Number(Number::new(value).unwrap()),
            Numbers::Canonical,
        )
        .unwrap()
    }

    #[test]
    fn canonical_numbers_match_cpp() {
        for (value, want) in [
            (0.0, "0"),
            (-0.0, "-0"),
            (1.0, "1"),
            (-1.0, "-1"),
            (100000.0, "100000"),
            (1e15, "1000000000000000"),
            (9007199254740992.0, "9007199254740992"),
            (-9007199254740992.0, "-9007199254740992"),
            // Past 2^53, integers take the shortest form.
            (9007199254740994.0, "9007199254740994"),
            // Fixed notation spells the exact integer, not padded digits.
            (36028797018963968.0, "36028797018963968"),
            (1e21, "1e+21"),
            (1.2e21, "1.2e+21"),
            (1e16, "1e+16"),
            (1.5e17, "1.5e+17"),
            (1e21, "1e+21"),
            (123456789012345680000.0, "123456789012345683968"),
            (1.2345678901234567e21, "1234567890123456774144"),
            // An exact tie between two shortest spellings rounds to even.
            (2.4969558715820312, "2.4969558715820312"),
            // Below a power of two, the nearer tie-free spelling.
            (7.120236347223045e-307, "7.120236347223045e-307"),
            (1.2345678901234567e30, "1.2345678901234567e+30"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (0.5, "0.5"),
            (-2.5, "-2.5"),
            (0.1 + 0.2, "0.30000000000000004"),
            (0.001, "0.001"),
            (0.0001, "1e-04"),
            (0.00012, "0.00012"),
            (1.5e-7, "1.5e-07"),
            (5e-324, "5e-324"),
            (2.2250738585072014e-308, "2.2250738585072014e-308"),
            (123.456, "123.456"),
            (1e100, "1e+100"),
        ] {
            assert_eq!(number(value), want, "{value:e}");
        }
    }

    #[test]
    fn as_parsed_writes_tokens() {
        let value = parsed("[1.50, 1E2, -0, 9007199254740993, 1e300]");
        assert_eq!(
            dump(&value, Numbers::AsParsed).unwrap(),
            "[1.50,1E2,-0,9007199254740993,1e300]"
        );
        assert_eq!(
            dump(&value, Numbers::Canonical).unwrap(),
            "[1.5,100,-0,9007199254740992,1e+300]"
        );
        // Built numbers have no token.
        let built = Value::Number(Number::new(2.5).unwrap());
        assert_eq!(dump(&built, Numbers::AsParsed).unwrap(), "2.5");
    }

    #[test]
    fn strings_escape_as_cpp() {
        let text = "\"\\/\u{8}\u{c}\n\r\t\u{0}\u{1f}\u{7f}é€😀";
        assert_eq!(
            dump(&Value::String(text.into()), Numbers::Canonical).unwrap(),
            "\"\\\"\\\\/\\b\\f\\n\\r\\t\\u0000\\u001f\u{7f}é€😀\""
        );
    }

    #[test]
    fn containers_and_spacing() {
        let value = parsed(
            r#" { "a" : [ 1 , [ ] , { } , null , true , false ] , "b\n" : { "c" : "d" } } "#,
        );
        assert_eq!(
            dump(&value, Numbers::Canonical).unwrap(),
            r#"{"a":[1,[],{},null,true,false],"b\n":{"c":"d"}}"#
        );
        assert_eq!(
            tojson(&value, Numbers::Canonical).unwrap(),
            r#"{"a": [1, [], {}, null, true, false], "b\n": {"c": "d"}}"#
        );
        assert_eq!(dump(&parsed("[]"), Numbers::Canonical).unwrap(), "[]");
        assert_eq!(
            dump(&parsed("[[[]]]"), Numbers::Canonical).unwrap(),
            "[[[]]]"
        );
        let mut duplicates = Object::new();
        duplicates.push("k".into(), Value::Null).unwrap();
        duplicates.push("k".into(), Value::Bool(true)).unwrap();
        assert_eq!(
            dump(&Value::Object(duplicates), Numbers::Canonical).unwrap(),
            r#"{"k":null,"k":true}"#
        );
    }

    #[test]
    fn deep_values_use_no_call_stack() {
        let mut value = Value::Null;
        for _ in 0..100_000 {
            value = Value::Array(vec![value]);
        }
        let text = dump(&value, Numbers::Canonical).unwrap();
        assert_eq!(text.len(), 200_004);
        // Dropping the tree recurses; unwind it by hand.
        while let Value::Array(mut items) = value {
            value = items.pop().unwrap_or(Value::Null);
        }
    }

    #[test]
    fn round_trips_through_the_parser() {
        let text = r#"{"a":[1.5,"x\u0001",{"b":null}],"c":-0,"d":1e+21}"#;
        assert_eq!(dump(&parsed(text), Numbers::Canonical).unwrap(), text);
        assert_eq!(dump(&parsed(text), Numbers::AsParsed).unwrap(), text);
    }
}
