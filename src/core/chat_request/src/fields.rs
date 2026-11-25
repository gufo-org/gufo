//! `json::Value` accessors as the C++ decoders used them, and fallible copies
//! of input-sized text.

use crate::error::Outcome;
use gufo_json::{Numbers, Spacing, Style, Value, write};

/// `member_str(key)`: the member's string, or "" when it is missing or not a
/// string (also on non-objects).
pub fn member_str<'a>(value: &'a Value, key: &str) -> &'a str {
    member_str_or(value, key, "")
}

/// `member_str(key, def)`.
pub fn member_str_or<'a>(value: &'a Value, key: &str, def: &'a str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap_or(def)
}

/// A member that is present and not null.
pub fn non_null<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value.get(key).filter(|value| !matches!(value, Value::Null))
}

/// `as_double()`, or `None` for non-numbers.
pub fn number(value: &Value) -> Option<f64> {
    value.as_number().map(|number| number.value())
}

pub fn copy(text: &str) -> Outcome<String> {
    let mut out = String::new();
    out.try_reserve_exact(text.len())?;
    out.push_str(text);
    Ok(out)
}

pub fn append(out: &mut String, text: &str) -> Outcome<()> {
    out.try_reserve(text.len())?;
    out.push_str(text);
    Ok(())
}

pub fn push<T>(items: &mut Vec<T>, item: T) -> Outcome<()> {
    items.try_reserve(1)?;
    items.push(item);
    Ok(())
}

/// Moves a string out of `value`, leaving null; "" for other values.
pub fn take_text(value: &mut Value) -> String {
    match value.take() {
        Value::String(text) => text,
        _ => String::new(),
    }
}

pub fn string(text: &str) -> Outcome<Value> {
    Ok(Value::String(copy(text)?))
}

/// `value.dump()`.
pub fn dump(value: &Value) -> Outcome<String> {
    Ok(gufo_json::dump(value, Numbers::Canonical)?)
}

/// `value.dump(kAsParsed)`, which `json::parse` reads back to the same value.
pub fn dump_as_parsed(value: &Value) -> Outcome<String> {
    Ok(gufo_json::dump(value, Numbers::AsParsed)?)
}

/// `value` written compactly, as `dump` writes it.
pub fn write_value(value: &Value, numbers: Numbers, out: &mut String) -> Outcome<()> {
    let style = Style {
        numbers,
        spacing: Spacing::Compact,
    };
    Ok(write(value, style, out)?)
}

/// An object key and its colon.
pub fn write_key(key: &str, out: &mut String) -> Outcome<()> {
    write_value(&Value::String(copy(key)?), Numbers::Canonical, out)?;
    append(out, ":")
}

/// Text joined from parts, some of them the client's.
pub fn concat(parts: &[&str]) -> Outcome<String> {
    let mut out = String::new();
    out.try_reserve_exact(parts.iter().map(|part| part.len()).sum())?;
    for part in parts {
        out.push_str(part);
    }
    Ok(out)
}

/// Members whose values the C++ helpers never read; the body view sends
/// null in their place (`GufoChatHostField`).
const ELIDED: [&str; 5] = ["messages", "input", "tools", "system", "instructions"];

/// The body view the helpers that read the whole body get: the body with
/// the values of `ELIDED` members null, or null when it is not an object.
pub fn body_view(body: &Value) -> Outcome<String> {
    let Some(object) = body.as_object() else {
        return copy("null");
    };
    let style = Style {
        numbers: Numbers::AsParsed,
        spacing: Spacing::Compact,
    };
    let mut out = String::new();
    append(&mut out, "{")?;
    for (index, (key, value)) in object.members().iter().enumerate() {
        if index > 0 {
            append(&mut out, ",")?;
        }
        write(&Value::String(copy(key)?), style, &mut out)?;
        append(&mut out, ":")?;
        if ELIDED.contains(&key.as_str()) {
            append(&mut out, "null")?;
        } else {
            write(value, style, &mut out)?;
        }
    }
    append(&mut out, "}")?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gufo_json::{Limits, parse_value};

    fn parse(text: &str) -> Value {
        parse_value(text.as_bytes(), Limits::NONE).unwrap()
    }

    #[test]
    fn accessors_follow_the_cpp_defaults() {
        let value = parse(r#"{"a":"x","b":1,"c":null}"#);
        assert_eq!(member_str(&value, "a"), "x");
        assert_eq!(member_str(&value, "b"), "");
        assert_eq!(member_str_or(&value, "z", "d"), "d");
        assert_eq!(member_str_or(&value, "b", "d"), "d");
        assert_eq!(member_str(&parse("[1]"), "a"), "");
        assert!(non_null(&value, "c").is_none());
        assert!(non_null(&value, "b").is_some());
    }

    #[test]
    fn the_body_view_nulls_decoded_members() {
        let body =
            parse(r#"{"model":"m","messages":[1],"temperature":1.50,"tools":{},"k\u0000":"x"}"#);
        assert_eq!(
            body_view(&body).unwrap(),
            r#"{"model":"m","messages":null,"temperature":1.50,"tools":null,"k\u0000":"x"}"#
        );
        assert_eq!(body_view(&parse("[1]")).unwrap(), "null");
    }
}
