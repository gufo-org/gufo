//! The parts of a tool's JSON schema that decide how an untyped value reads.
//! Qwen's XML-like arguments carry no type marker, so the advertised schema
//! is needed to distinguish a string such as 42 from the JSON number 42.

use gufo_json::Value;

/// The value a local JSON pointer (`#` or `#/...`) names, as
/// `JsonConstraint::ResolveReference` reads it; `None` when it is not one or
/// names nothing.
fn resolve_reference<'a>(root: &'a Value, reference: &Value) -> Option<&'a Value> {
    let pointer = reference.as_str()?;
    if pointer != "#" && !pointer.starts_with("#/") {
        return None;
    }
    let mut target = root;
    let mut pointer = &pointer.as_bytes()[1..];
    while !pointer.is_empty() {
        pointer = &pointer[1..];
        let slash = pointer.iter().position(|&byte| byte == b'/');
        let segment = &pointer[..slash.unwrap_or(pointer.len())];
        let mut decoded = Vec::with_capacity(segment.len());
        let mut i = 0;
        while i < segment.len() {
            if segment[i] == b'~' {
                i += 1;
                match segment.get(i) {
                    Some(b'0') => decoded.push(b'~'),
                    Some(b'1') => decoded.push(b'/'),
                    _ => return None,
                }
            } else {
                decoded.push(segment[i]);
            }
            i += 1;
        }
        target = match target {
            Value::Array(items) => index(&decoded).and_then(|index| items.get(index))?,
            _ => target.get(std::str::from_utf8(&decoded).ok()?)?,
        };
        match slash {
            None => break,
            Some(slash) => pointer = &pointer[slash..],
        }
    }
    Some(target)
}

/// A whole decimal array index, as `std::from_chars` reads one into
/// `size_t`.
fn index(digits: &[u8]) -> Option<usize> {
    if digits.is_empty() || !digits.iter().all(u8::is_ascii_digit) {
        return None;
    }
    digits.iter().try_fold(0usize, |value, &digit| {
        value
            .checked_mul(10)?
            .checked_add(usize::from(digit - b'0'))
    })
}

/// `schema` after following its local `$ref`s; `None` for a cycle, a
/// reference to the whole tool schema (recursion the native best-effort
/// grammar cannot enforce, so no definite scalar type is inferred from the
/// outer parameter container) or one that does not resolve.
pub fn resolve<'a>(root: &'a Value, schema: &'a Value) -> Option<&'a Value> {
    let mut target = schema;
    let mut seen: Vec<*const Value> = Vec::new();
    while let Some(reference) = target.get("$ref") {
        if seen.contains(&(target as *const Value)) {
            return None;
        }
        seen.push(target);
        target = resolve_reference(root, reference)?;
        if std::ptr::eq(target, root) {
            return None;
        }
    }
    Some(target)
}

fn same_kind(candidate: &Value, value: &Value) -> bool {
    std::mem::discriminant(candidate) == std::mem::discriminant(value)
}

fn is_type(kind: &str, value: &Value) -> bool {
    match (kind, value) {
        ("string", Value::String(_))
        | ("number", Value::Number(_))
        | ("boolean", Value::Bool(_))
        | ("null", Value::Null)
        | ("array", Value::Array(_))
        | ("object", Value::Object(_)) => true,
        ("integer", Value::Number(number)) => number.value().floor() == number.value(),
        _ => false,
    }
}

/// `original` admits `value`, as far as the kinds llama.cpp's value_types()
/// matches go. Unknown or unresolvable schemas admit anything.
pub fn accepts(root: &Value, original: &Value, value: &Value, depth: usize) -> bool {
    let Some(schema) = resolve(root, original) else {
        return true;
    };
    if depth > 16 {
        return true; // Retain text for unknown non-strict argument types.
    }
    // Match value kinds rather than guessing from the lexeme.
    // {"const":"42"} is text even without an explicit type.
    if let Some(constant) = schema.get("const") {
        return same_kind(constant, value);
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        return values.iter().any(|candidate| same_kind(candidate, value));
    }
    if let Some(kind) = schema.get("type") {
        return match kind {
            Value::String(kind) => is_type(kind, value),
            Value::Array(kinds) => kinds
                .iter()
                .any(|item| item.as_str().is_some_and(|kind| is_type(kind, value))),
            _ => false,
        };
    }
    for name in ["anyOf", "oneOf"] {
        if let Some(choices) = schema.get(name).and_then(Value::as_array) {
            return choices
                .iter()
                .any(|item| accepts(root, item, value, depth + 1));
        }
    }
    if schema.get("properties").is_some()
        || schema
            .get("additionalProperties")
            .is_some_and(|extra| extra.as_bool() != Some(true))
    {
        return matches!(value, Value::Object(_));
    }
    if let Some(parts) = schema.get("allOf").and_then(Value::as_array) {
        return parts
            .iter()
            .all(|item| accepts(root, item, value, depth + 1));
    }
    if schema.get("items").is_some() || schema.get("prefixItems").is_some() {
        return matches!(value, Value::Array(_));
    }
    if schema.get("pattern").is_some()
        || schema.get("minLength").is_some()
        || schema.get("maxLength").is_some()
    {
        return matches!(value, Value::String(_));
    }
    true
}

/// `schema` admits only strings.
pub fn string_only(root: &Value, schema: &Value, depth: u32) -> bool {
    if depth > 32 {
        return false;
    }
    let Some(resolved) = resolve(root, schema) else {
        return false;
    };
    if !std::ptr::eq(resolved, schema) {
        return string_only(root, resolved, depth + 1);
    }
    if let Some(Value::String(kind)) = schema.get("type") {
        return kind == "string";
    }
    if let Some(value) = schema.get("const") {
        return matches!(value, Value::String(_));
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        return !values.is_empty() && values.iter().all(|value| matches!(value, Value::String(_)));
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(values) = schema.get(key).and_then(Value::as_array) {
            return !values.is_empty()
                && values
                    .iter()
                    .all(|value| string_only(root, value, depth + 1));
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use gufo_json::{Limits, parse_value};

    fn json(text: &str) -> Value {
        parse_value(text.as_bytes(), Limits::NONE).unwrap()
    }

    #[test]
    fn local_references_resolve_and_cycles_do_not() {
        let root = json(
            r##"{"$defs":{"a":{"$ref":"#/$defs/b"},"b":{"type":"integer"},"c":{"$ref":"#/$defs/c"},"t~/":{"type":"string"},"l":[{"type":"null"}]},"properties":{"x":{"$ref":"#"}}}"##,
        );
        let a = json(r##"{"$ref":"#/$defs/a"}"##);
        assert_eq!(
            resolve(&root, &a).and_then(|s| s.get("type")),
            Some(&Value::String("integer".into()))
        );
        assert!(resolve(&root, &json(r##"{"$ref":"#/$defs/c"}"##)).is_none());
        assert!(resolve(&root, &json(r##"{"$ref":"#"}"##)).is_none());
        assert!(resolve(&root, &json(r##"{"$ref":"other.json"}"##)).is_none());
        assert!(resolve(&root, &json(r##"{"$ref":"#/$defs/t~0~1"}"##)).is_some());
        assert!(resolve(&root, &json(r##"{"$ref":"#/$defs/t~2"}"##)).is_none());
        assert!(resolve(&root, &json(r##"{"$ref":"#/$defs/l/0"}"##)).is_some());
        assert!(resolve(&root, &json(r##"{"$ref":"#/$defs/l/1"}"##)).is_none());
        assert!(resolve(&root, &json(r##"{"$ref":"#/$defs/l/-0"}"##)).is_none());
    }

    #[test]
    fn kinds_decide_acceptance() {
        let root = json("{}");
        let integer = json(r#"{"type":"integer"}"#);
        assert!(accepts(&root, &integer, &json("4"), 0));
        assert!(!accepts(&root, &integer, &json("4.5"), 0));
        assert!(!accepts(&root, &json(r#"{"const":"42"}"#), &json("42"), 0));
        assert!(accepts(
            &root,
            &json(r#"{"anyOf":[{"type":"null"},{"type":"string"}]}"#),
            &json("null"),
            0
        ));
        assert!(!accepts(
            &root,
            &json(r#"{"additionalProperties":false}"#),
            &json("1"),
            0
        ));
        assert!(accepts(
            &root,
            &json(r#"{"additionalProperties":true}"#),
            &json("1"),
            0
        ));
        assert!(string_only(&root, &json(r#"{"enum":["a","b"]}"#), 0));
        assert!(!string_only(&root, &json(r#"{"enum":[]}"#), 0));
    }
}
