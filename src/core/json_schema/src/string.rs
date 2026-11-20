//! The string lexeme's keywords: lengths, `pattern` and `format`. Formats
//! follow JSON Schema's RFC-backed string formats. The grammar validates
//! decoded strings, so an escaped character has exactly the same meaning as
//! its UTF-8 spelling.

use crate::error::{Result, empty, invalid};
use crate::value::Value;

pub const KEYWORDS: [&str; 4] = ["minLength", "maxLength", "pattern", "format"];
const MAX_LENGTH: f64 = 1048576.0;
/// RFC 1123 limits a hostname to 253 characters.
const HOSTNAME_LENGTH: u32 = 253;

/// A string format's pattern, and its length cap (`u32::MAX` for none).
pub fn format(name: &[u8]) -> Result<(String, u32)> {
    let leap = "(?:[0-9]{2}(?:0[48]|[2468][048]|[13579][26])|(?:[02468][048]|[13579][26])00)";
    let date = format!(
        "(?:[0-9]{{4}}-(?:(?:0[13578]|1[02])-(?:0[1-9]|[12][0-9]|3[01])|(?:0[469]|11)-(?:0[1-9]|[12][0-9]|30)|02-(?:0[1-9]|1[0-9]|2[0-8]))|{leap}-02-29)"
    );
    let time = "(?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)(?:\\.[0-9]+)?(?:[zZ]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])";
    let ipv4 = "(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])";
    let pattern = match name {
        b"date" => format!("^{date}$"),
        b"time" => format!("^{time}$"),
        b"date-time" => format!("^{date}[tT]{time}$"),
        b"uuid" => "^[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$".into(),
        b"ipv4" => format!("^{ipv4}$"),
        b"hostname" => "^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*\\.?$".into(),
        b"email" => "^[^@\\s]+@[^@\\s]+$".into(),
        b"duration" => "^P(?:[0-9]+W|(?=[0-9]|T[0-9])(?:[0-9]+Y)?(?:[0-9]+M)?(?:[0-9]+D)?(?:T(?=[0-9])(?:[0-9]+H)?(?:[0-9]+M)?(?:[0-9]+(?:\\.[0-9]+)?S)?)?)$".into(),
        b"ipv6" => ipv6(ipv4),
        _ => return invalid("unsupported string format"),
    };
    let cap = if name == b"hostname" {
        HOSTNAME_LENGTH
    } else {
        u32::MAX
    };
    Ok((pattern, cap))
}

fn ipv6(ipv4: &str) -> String {
    let groups = |count: usize| vec!["[0-9a-fA-F]{1,4}"; count].join(":");
    let mut result = format!("^(?:{}", groups(8));
    for left in 0..8 {
        for right in 0..8 - left {
            result += &format!("|{}::{}", groups(left), groups(right));
        }
    }
    result += &format!("|{}:{ipv4}", groups(6));
    for left in 0..6 {
        for right in 0..6 - left {
            let separator = if right != 0 { ":" } else { "" };
            result += &format!("|{}::{}{separator}{ipv4}", groups(left), groups(right));
        }
    }
    result + ")$"
}

/// The checked string keywords: length bounds and the patterns, in keyword
/// order, whose intersection the value must match.
#[derive(Debug, PartialEq, Eq)]
pub struct Plan {
    pub minimum: u32,
    pub maximum: u32,
    pub patterns: Vec<Vec<u8>>,
}

/// `keywords` holds the values of `KEYWORDS`, in order.
pub fn plan(keywords: &[Value<'_>; 4]) -> Result<Plan> {
    let mut lengths = [0u32, u32::MAX];
    for (length, value) in lengths.iter_mut().zip(&keywords[..2]) {
        match *value {
            Value::Absent => {}
            Value::Number { value, .. }
                if (0.0..=MAX_LENGTH).contains(&value) && value.floor() == value =>
            {
                *length = value as u32;
            }
            _ => return invalid("string lengths must be nonnegative integers up to 1048576"),
        }
    }
    let [minimum, mut maximum] = lengths;
    if minimum > maximum {
        return empty("minLength exceeds maxLength");
    }
    let mut patterns = Vec::new();
    for (key, value) in KEYWORDS[2..].iter().zip(&keywords[2..]) {
        let text = match *value {
            Value::Absent => continue,
            Value::String(text) => text,
            _ => return invalid(format!("{key} must be a string")),
        };
        if *key == "format" {
            let (pattern, cap) = format(text)?;
            maximum = maximum.min(cap);
            patterns.push(pattern.into_bytes());
        } else {
            patterns.push(text.to_vec());
        }
    }
    Ok(Plan {
        minimum,
        maximum,
        patterns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Failure;

    fn number(value: f64) -> Value<'static> {
        Value::Number { value, text: b"" }
    }

    #[test]
    fn keywords() {
        let plan = plan(&[
            number(2.0),
            number(400.0),
            Value::String(b"a+"),
            Value::String(b"hostname"),
        ])
        .unwrap();
        assert_eq!((plan.minimum, plan.maximum), (2, 253));
        assert_eq!(plan.patterns[0], b"a+");
        assert_eq!(plan.patterns[1], format(b"hostname").unwrap().0.as_bytes());
        let lengths = "string lengths must be nonnegative integers up to 1048576";
        for bad in [-1.0, 1.5, 1048577.0, f64::NAN, f64::INFINITY] {
            let keywords = [number(bad), Value::Absent, Value::Absent, Value::Absent];
            assert_eq!(
                super::plan(&keywords),
                Err(Failure::Rejected(lengths.into()))
            );
        }
        let keywords = [number(3.0), number(2.0), Value::Absent, Value::Absent];
        assert_eq!(
            super::plan(&keywords),
            Err(Failure::Empty("minLength exceeds maxLength".into()))
        );
        let keywords = [Value::Absent, Value::Absent, Value::Absent, number(1.0)];
        assert_eq!(
            super::plan(&keywords),
            Err(Failure::Rejected("format must be a string".into()))
        );
        assert_eq!(
            format(b"color"),
            Err(Failure::Rejected("unsupported string format".into()))
        );
    }

    #[test]
    fn ipv6_alternatives() {
        let (pattern, cap) = format(b"ipv6").unwrap();
        assert_eq!(cap, u32::MAX);
        // The full form, then every compressed form, with and without an
        // IPv4 suffix.
        let group = "[0-9a-fA-F]{1,4}";
        assert!(pattern.starts_with(&format!("^(?:{group}:{group}:")));
        assert!(pattern.contains(&format!("|{group}::{group}|")));
        assert!(pattern.contains("|::|"));
        assert!(pattern.ends_with("[1-9]?[0-9]))$"));
    }
}
