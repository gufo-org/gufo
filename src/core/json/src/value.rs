//! A JSON value tree built from the parser's events, as `gufo::json::parse`
//! (src/core/json.cpp) builds `json::Value`: members keep their order,
//! numbers keep the token they were read from, and lookups take the first
//! member with a key. The parser rejects duplicate keys, so only values
//! built with `Object::push` can hold them.

use crate::parse::{self, Event, Failure, Sink, Stopped};
use std::collections::TryReserveError;

/// A finite number and, when parsed, the token it was read from.
#[derive(Debug, Clone, PartialEq)]
pub struct Number {
    value: f64,
    token: Option<Box<str>>,
}

impl Number {
    /// A number built from a value, which `Numbers::AsParsed` still writes
    /// canonically; `None` when `value` is not finite, which JSON cannot
    /// spell (C++ `dump` throws on one).
    pub fn new(value: f64) -> Option<Self> {
        value.is_finite().then_some(Self { value, token: None })
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    /// The token the parser read, which `Numbers::AsParsed` writes.
    pub fn token(&self) -> Option<&str> {
        self.token.as_deref()
    }
}

/// An object's members in document order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Object {
    members: Vec<(String, Value)>,
}

impl Object {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn members(&self) -> &[(String, Value)] {
        &self.members
    }

    /// The value of the first member named `key`, as C++ `find` returns it.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.members.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Appends a member, even when one has the same key, as C++
    /// `append_member` does.
    pub fn push(&mut self, key: String, value: Value) -> Result<(), TryReserveError> {
        self.members.try_reserve(1)?;
        self.members.push((key, value));
        Ok(())
    }

    /// The value of the first member named `key`, as `get` finds it.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        self.members
            .iter_mut()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    /// Removes the first member named `key` and returns its value, keeping
    /// the others in order; a later duplicate then becomes the first.
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        let at = self.members.iter().position(|(k, _)| k == key)?;
        Some(self.members.remove(at).1)
    }

    /// The members in order, moved out without copying.
    pub fn into_members(self) -> Vec<(String, Value)> {
        self.members
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}

impl IntoIterator for Object {
    type Item = (String, Value);
    type IntoIter = std::vec::IntoIter<(String, Value)>;

    fn into_iter(self) -> Self::IntoIter {
        self.members.into_iter()
    }
}

impl<'a> IntoIterator for &'a Object {
    type Item = &'a (String, Value);
    type IntoIter = std::slice::Iter<'a, (String, Value)>;

    fn into_iter(self) -> Self::IntoIter {
        self.members.iter()
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub enum Value {
    #[default]
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<Value>),
    Object(Object),
}

impl Value {
    /// The first member named `key` of an object; `None` for other values.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Self::Object(object) => object.get(key),
            _ => None,
        }
    }

    /// The first member named `key` of an object, mutably.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        match self {
            Self::Object(object) => object.get_mut(key),
            _ => None,
        }
    }

    /// Moves the value out, leaving `Null`.
    pub fn take(&mut self) -> Value {
        std::mem::take(self)
    }

    pub fn into_string(self) -> Option<String> {
        match self {
            Self::String(text) => Some(text),
            _ => None,
        }
    }

    pub fn into_array(self) -> Option<Vec<Value>> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    pub fn into_object(self) -> Option<Object> {
        match self {
            Self::Object(object) => Some(object),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_number(&self) -> Option<&Number> {
        match self {
            Self::Number(number) => Some(number),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(text) => Some(text),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&Object> {
        match self {
            Self::Object(object) => Some(object),
            _ => None,
        }
    }
}

/// What a parse may build. Each value, container and member counts once
/// toward `values`, checked before it is allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// The longest text accepted, checked before parsing.
    pub bytes: usize,
    pub values: usize,
}

impl Limits {
    pub const NONE: Self = Self {
        bytes: usize::MAX,
        values: usize::MAX,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueError {
    /// The text is not JSON; the reason as `gufo::json::parse` gives it
    /// without its "JSON parse error: " prefix.
    Rejected(&'static str),
    /// The text or its values exceed the caller's `Limits`.
    TooLarge,
    /// An allocation failed.
    NoMemory,
    /// The parser broke its event contract (a key outside an object, an
    /// unmatched end, text that is not UTF-8): a bug, not bad input.
    Internal,
}

/// Parses `text` into a value tree within `limits`.
pub fn parse_value(text: &[u8], limits: Limits) -> Result<Value, ValueError> {
    if text.len() > limits.bytes {
        return Err(ValueError::TooLarge);
    }
    let mut builder = Builder {
        open: Vec::new(),
        root: None,
        left: limits.values,
        error: None,
    };
    match parse::parse(text, &mut builder) {
        Ok(()) => {}
        Err(Failure::Rejected(reason)) => return Err(ValueError::Rejected(reason)),
        Err(Failure::NoMemory) => return Err(ValueError::NoMemory),
        // Only the builder stops, after recording why.
        Err(Failure::Stopped) => return Err(builder.error.unwrap_or(ValueError::Internal)),
    }
    // A complete parse closed every container and gave one root.
    builder.root.ok_or(ValueError::Internal)
}

/// An open container; an object's newest key waits for its value.
enum Open {
    Array(Vec<Value>),
    Object(Object, Option<String>),
}

struct Builder {
    open: Vec<Open>,
    root: Option<Value>,
    left: usize,
    error: Option<ValueError>,
}

fn owned(bytes: &[u8]) -> Result<String, ValueError> {
    // The parser reports strict UTF-8; anything else is a parser bug.
    let text = std::str::from_utf8(bytes).map_err(|_| ValueError::Internal)?;
    let mut out = String::new();
    out.try_reserve_exact(text.len())
        .map_err(|_| ValueError::NoMemory)?;
    out.push_str(text);
    Ok(out)
}

impl Builder {
    fn count(&mut self) -> Result<(), ValueError> {
        self.left = self.left.checked_sub(1).ok_or(ValueError::TooLarge)?;
        Ok(())
    }

    /// Puts a finished value into the innermost container or the root.
    fn place(&mut self, value: Value) -> Result<(), ValueError> {
        let no_memory = |_| ValueError::NoMemory;
        match self.open.last_mut() {
            None => self.root = Some(value),
            Some(Open::Array(items)) => {
                items.try_reserve(1).map_err(no_memory)?;
                items.push(value);
            }
            Some(Open::Object(object, key)) => {
                let key = key.take().ok_or(ValueError::Internal)?;
                object.push(key, value).map_err(no_memory)?;
            }
        }
        Ok(())
    }

    fn take(&mut self, event: Event<'_>) -> Result<(), ValueError> {
        if !matches!(event, Event::End) {
            self.count()?;
        }
        let value = match event {
            Event::Null => Value::Null,
            Event::Bool(value) => Value::Bool(value),
            Event::Number(value, token) => Value::Number(Number {
                value,
                token: Some(owned(token)?.into_boxed_str()),
            }),
            Event::String(bytes) => Value::String(owned(bytes)?),
            Event::Key(bytes) => {
                let key = owned(bytes)?;
                return match self.open.last_mut() {
                    Some(Open::Object(_, slot @ None)) => {
                        *slot = Some(key);
                        Ok(())
                    }
                    _ => Err(ValueError::Internal),
                };
            }
            Event::BeginArray | Event::BeginObject => {
                self.open.try_reserve(1).map_err(|_| ValueError::NoMemory)?;
                self.open.push(match event {
                    Event::BeginArray => Open::Array(Vec::new()),
                    _ => Open::Object(Object::new(), None),
                });
                return Ok(());
            }
            Event::End => match self.open.pop() {
                Some(Open::Array(items)) => Value::Array(items),
                Some(Open::Object(object, None)) => Value::Object(object),
                _ => return Err(ValueError::Internal),
            },
        };
        self.place(value)
    }
}

impl Sink for Builder {
    fn event(&mut self, event: Event<'_>) -> Result<(), Stopped> {
        self.take(event).map_err(|error| {
            self.error = Some(error);
            Stopped
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Value {
        parse_value(text.as_bytes(), Limits::NONE).unwrap()
    }

    #[test]
    fn builds_the_tree_in_order() {
        let value = parse(r#" {"b":[1, 2.50, -0], "a":{"x":"é"}, "c":null, "d":true} "#);
        let object = value.as_object().unwrap();
        let keys: Vec<&str> = object.members().iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["b", "a", "c", "d"]);
        let items = value.get("b").unwrap().as_array().unwrap();
        let tokens: Vec<_> = items
            .iter()
            .map(|v| v.as_number().unwrap().token().unwrap())
            .collect();
        assert_eq!(tokens, ["1", "2.50", "-0"]);
        assert_eq!(items[1].as_number().unwrap().value(), 2.5);
        assert!(items[2].as_number().unwrap().value().is_sign_negative());
        assert_eq!(
            value.get("a").unwrap().get("x").unwrap().as_str(),
            Some("é")
        );
        assert_eq!(value.get("c"), Some(&Value::Null));
        assert_eq!(value.get("d").unwrap().as_bool(), Some(true));
        assert_eq!(value.get("missing"), None);
        assert_eq!(items[0].get("b"), None);
    }

    #[test]
    fn scalar_roots_and_rejections() {
        assert_eq!(parse("\"x\""), Value::String("x".into()));
        assert_eq!(parse("[]"), Value::Array(Vec::new()));
        assert_eq!(parse("{}"), Value::Object(Object::new()));
        assert_eq!(
            parse_value(b"[1,]", Limits::NONE),
            Err(ValueError::Rejected("bad number"))
        );
        assert_eq!(
            parse_value(br#"{"a":1,"a":2}"#, Limits::NONE),
            Err(ValueError::Rejected("duplicate object key"))
        );
    }

    #[test]
    fn lookup_takes_the_first_duplicate() {
        let mut object = Object::new();
        object.push("a".into(), Value::Bool(true)).unwrap();
        object.push("a".into(), Value::Bool(false)).unwrap();
        assert_eq!(object.len(), 2);
        assert_eq!(object.get("a"), Some(&Value::Bool(true)));
    }

    #[test]
    fn limits_are_checked_before_allocating() {
        let limits = |bytes, values| Limits { bytes, values };
        assert_eq!(
            parse_value(b"[1, 2]", limits(5, 9)),
            Err(ValueError::TooLarge)
        );
        assert!(parse_value(b"[1, 2]", limits(6, 3)).is_ok());
        assert_eq!(
            parse_value(b"[1, 2]", limits(6, 2)),
            Err(ValueError::TooLarge)
        );
        // A member's key counts too.
        assert!(parse_value(br#"{"a":1}"#, limits(7, 3)).is_ok());
        assert_eq!(
            parse_value(br#"{"a":1}"#, limits(7, 2)),
            Err(ValueError::TooLarge)
        );
        // Rejection still wins when the text is bad before the limit.
        assert_eq!(
            parse_value(b"[x, 1, 2]", limits(9, 2)),
            Err(ValueError::Rejected("bad number"))
        );
    }

    #[test]
    fn members_move_out_without_copying() {
        let mut value = parse(r#"{"a":"payload","b":[1]}"#);
        let pointer = value.get("a").unwrap().as_str().unwrap().as_ptr();
        // `take` and `into_string` move the string's buffer.
        let taken = value.get_mut("a").unwrap().take().into_string().unwrap();
        assert_eq!(taken.as_ptr(), pointer);
        assert_eq!(value.get("a"), Some(&Value::Null));
        let mut object = value.into_object().unwrap();
        *object.get_mut("b").unwrap() = Value::Bool(true);
        assert_eq!(object.remove("a"), Some(Value::Null));
        assert_eq!(object.remove("a"), None);
        assert_eq!(object.members().len(), 1);
        let members: Vec<_> = (&object).into_iter().map(|(k, _)| k.clone()).collect();
        assert_eq!(members, ["b"]);
        let (key, value) = object.into_iter().next().unwrap();
        assert_eq!((key.as_str(), value), ("b", Value::Bool(true)));

        let mut duplicates = Object::new();
        duplicates.push("k".into(), Value::Bool(true)).unwrap();
        duplicates.push("j".into(), Value::Null).unwrap();
        duplicates.push("k".into(), Value::Bool(false)).unwrap();
        // `remove` takes the first; the later duplicate becomes visible.
        assert_eq!(duplicates.remove("k"), Some(Value::Bool(true)));
        assert_eq!(duplicates.get("k"), Some(&Value::Bool(false)));
        let keys: Vec<String> = duplicates
            .into_members()
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        assert_eq!(keys, ["j", "k"]);
        assert_eq!(Value::Null.into_array(), None);
        assert_eq!(Value::Bool(true).into_string(), None);
        assert_eq!(parse("[2]").into_array().unwrap().len(), 1);
    }

    #[test]
    fn broken_event_contracts_are_internal() {
        let builder = || Builder {
            open: Vec::new(),
            root: None,
            left: usize::MAX,
            error: None,
        };
        // A key outside an object, an unmatched end, an end with a key
        // waiting, text that is not UTF-8 and a second key in a row.
        let mut b = builder();
        assert_eq!(b.take(Event::Key(b"k")), Err(ValueError::Internal));
        let mut b = builder();
        assert_eq!(b.take(Event::End), Err(ValueError::Internal));
        let mut b = builder();
        b.take(Event::BeginObject).unwrap();
        b.take(Event::Key(b"k")).unwrap();
        assert_eq!(b.take(Event::End), Err(ValueError::Internal));
        let mut b = builder();
        assert_eq!(b.take(Event::String(b"\xff")), Err(ValueError::Internal));
        let mut b = builder();
        b.take(Event::BeginObject).unwrap();
        b.take(Event::Key(b"k")).unwrap();
        assert_eq!(b.take(Event::Key(b"j")), Err(ValueError::Internal));
        // A value in an object without its key.
        let mut b = builder();
        b.take(Event::BeginObject).unwrap();
        assert_eq!(b.take(Event::Null), Err(ValueError::Internal));
        // Through the sink, the recorded error stops the parse.
        let mut b = builder();
        b.take(Event::BeginObject).unwrap();
        assert_eq!(b.event(Event::Null), Err(Stopped));
        assert_eq!(b.error, Some(ValueError::Internal));
    }

    #[test]
    fn number_new_rejects_non_finite() {
        assert!(Number::new(f64::NAN).is_none());
        assert!(Number::new(f64::INFINITY).is_none());
        let number = Number::new(-0.0).unwrap();
        assert_eq!(number.token(), None);
    }
}
