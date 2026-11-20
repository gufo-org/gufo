//! A schema keyword's value, as C++ describes it from its `json::Value`.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value<'a> {
    Absent,
    /// `text` is the canonical spelling (`json::Value::dump()`) when `value`
    /// is finite, and empty otherwise.
    Number {
        value: f64,
        text: &'a [u8],
    },
    String(&'a [u8]),
    Other,
}

impl<'a> Value<'a> {
    pub fn finite_number(&self) -> Option<&'a [u8]> {
        match *self {
            Value::Number { value, text } if value.is_finite() => Some(text),
            _ => None,
        }
    }

    pub fn positive_number(&self) -> Option<&'a [u8]> {
        match *self {
            Value::Number { value, text } if value.is_finite() && value > 0.0 => Some(text),
            _ => None,
        }
    }
}
