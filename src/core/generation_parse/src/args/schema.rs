//! The parts of a tool's JSON schema that decide how an untyped value reads.
//! Qwen's XML-like arguments carry no type marker, so the advertised schema
//! is needed to distinguish a string such as 42 from the JSON number 42.

use crate::config::Error;
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

/// Results already computed for a union or intersection during one top-level
/// check, keyed by the resolved schema node and the depth it was reached at.
/// The value is fixed for the whole check and `root` owns every node, so each
/// result is a function of the key alone. A schema node reachable along many
/// paths (an `allOf` or `anyOf` naming itself or a shared definition more
/// than once) is then evaluated once per depth instead of once per path,
/// which keeps the work polynomial in the schema size while returning the
/// same answer as the unmemoised recursion, including its depth cap.
type Memo = std::collections::HashMap<(*const Value, usize), bool>;

fn remember(memo: &mut Memo, key: (*const Value, usize), result: bool) -> Result<bool, Error> {
    memo.try_reserve(1)?;
    memo.insert(key, result);
    Ok(result)
}

/// The deepest union or intersection `accepts` descends into; deeper parts
/// admit anything.
const ACCEPTS_DEPTH: usize = 16;

/// `original` admits `value`, as far as the kinds llama.cpp's value_types()
/// matches go. Unknown or unresolvable schemas admit anything.
pub fn accepts(root: &Value, original: &Value, value: &Value) -> Result<bool, Error> {
    accepts_at(root, original, value, 0, &mut Memo::new())
}

fn accepts_at(
    root: &Value,
    original: &Value,
    value: &Value,
    depth: usize,
    memo: &mut Memo,
) -> Result<bool, Error> {
    let Some(schema) = resolve(root, original) else {
        return Ok(true);
    };
    if depth > ACCEPTS_DEPTH {
        return Ok(true); // Retain text for unknown non-strict argument types.
    }
    // Match value kinds rather than guessing from the lexeme.
    // {"const":"42"} is text even without an explicit type.
    if let Some(constant) = schema.get("const") {
        return Ok(same_kind(constant, value));
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        return Ok(values.iter().any(|candidate| same_kind(candidate, value)));
    }
    if let Some(kind) = schema.get("type") {
        return Ok(match kind {
            Value::String(kind) => is_type(kind, value),
            Value::Array(kinds) => kinds
                .iter()
                .any(|item| item.as_str().is_some_and(|kind| is_type(kind, value))),
            _ => false,
        });
    }
    let key = (schema as *const Value, depth);
    for name in ["anyOf", "oneOf"] {
        if let Some(choices) = schema.get(name).and_then(Value::as_array) {
            if let Some(&known) = memo.get(&key) {
                return Ok(known);
            }
            let mut any = false;
            for item in choices {
                if accepts_at(root, item, value, depth + 1, memo)? {
                    any = true;
                    break;
                }
            }
            return remember(memo, key, any);
        }
    }
    if schema.get("properties").is_some()
        || schema
            .get("additionalProperties")
            .is_some_and(|extra| extra.as_bool() != Some(true))
    {
        return Ok(matches!(value, Value::Object(_)));
    }
    if let Some(parts) = schema.get("allOf").and_then(Value::as_array) {
        if let Some(&known) = memo.get(&key) {
            return Ok(known);
        }
        let mut all = true;
        for item in parts {
            if !accepts_at(root, item, value, depth + 1, memo)? {
                all = false;
                break;
            }
        }
        return remember(memo, key, all);
    }
    if schema.get("items").is_some() || schema.get("prefixItems").is_some() {
        return Ok(matches!(value, Value::Array(_)));
    }
    if schema.get("pattern").is_some()
        || schema.get("minLength").is_some()
        || schema.get("maxLength").is_some()
    {
        return Ok(matches!(value, Value::String(_)));
    }
    Ok(true)
}

/// The deepest reference or union `string_only` follows; deeper schemas are
/// not string-only.
const STRING_ONLY_DEPTH: usize = 32;

/// `schema` admits only strings.
pub fn string_only(root: &Value, schema: &Value) -> Result<bool, Error> {
    string_only_at(root, schema, 0, &mut Memo::new())
}

fn string_only_at(
    root: &Value,
    schema: &Value,
    depth: usize,
    memo: &mut Memo,
) -> Result<bool, Error> {
    if depth > STRING_ONLY_DEPTH {
        return Ok(false);
    }
    let Some(resolved) = resolve(root, schema) else {
        return Ok(false);
    };
    if !std::ptr::eq(resolved, schema) {
        return string_only_at(root, resolved, depth + 1, memo);
    }
    if let Some(Value::String(kind)) = schema.get("type") {
        return Ok(kind == "string");
    }
    if let Some(value) = schema.get("const") {
        return Ok(matches!(value, Value::String(_)));
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        return Ok(
            !values.is_empty() && values.iter().all(|value| matches!(value, Value::String(_)))
        );
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(values) = schema.get(key).and_then(Value::as_array) {
            let key = (schema as *const Value, depth);
            if let Some(&known) = memo.get(&key) {
                return Ok(known);
            }
            let mut all = !values.is_empty();
            for value in values {
                if !string_only_at(root, value, depth + 1, memo)? {
                    all = false;
                    break;
                }
            }
            return remember(memo, key, all);
        }
    }
    Ok(false)
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

    fn accepted(root: &Value, schema: &Value, value: &Value) -> bool {
        accepts(root, schema, value).unwrap()
    }

    #[test]
    fn kinds_decide_acceptance() {
        let root = json("{}");
        let integer = json(r#"{"type":"integer"}"#);
        assert!(accepted(&root, &integer, &json("4")));
        assert!(!accepted(&root, &integer, &json("4.5")));
        assert!(!accepted(&root, &json(r#"{"const":"42"}"#), &json("42")));
        assert!(accepted(
            &root,
            &json(r#"{"anyOf":[{"type":"null"},{"type":"string"}]}"#),
            &json("null"),
        ));
        assert!(!accepted(
            &root,
            &json(r#"{"additionalProperties":false}"#),
            &json("1"),
        ));
        assert!(accepted(
            &root,
            &json(r#"{"additionalProperties":true}"#),
            &json("1"),
        ));
        assert!(string_only(&root, &json(r#"{"enum":["a","b"]}"#)).unwrap());
        assert!(!string_only(&root, &json(r#"{"enum":[]}"#)).unwrap());
    }

    /// The recursion before memoisation, kept to show it gives the same
    /// answers.
    fn unmemoised_accepts(root: &Value, original: &Value, value: &Value, depth: usize) -> bool {
        let Some(schema) = resolve(root, original) else {
            return true;
        };
        if depth > 16 {
            return true;
        }
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
                    .any(|item| unmemoised_accepts(root, item, value, depth + 1));
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
                .all(|item| unmemoised_accepts(root, item, value, depth + 1));
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

    fn unmemoised_string_only(root: &Value, schema: &Value, depth: u32) -> bool {
        if depth > 32 {
            return false;
        }
        let Some(resolved) = resolve(root, schema) else {
            return false;
        };
        if !std::ptr::eq(resolved, schema) {
            return unmemoised_string_only(root, resolved, depth + 1);
        }
        if let Some(Value::String(kind)) = schema.get("type") {
            return kind == "string";
        }
        if let Some(value) = schema.get("const") {
            return matches!(value, Value::String(_));
        }
        if let Some(values) = schema.get("enum").and_then(Value::as_array) {
            return !values.is_empty()
                && values.iter().all(|value| matches!(value, Value::String(_)));
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(values) = schema.get(key).and_then(Value::as_array) {
                return !values.is_empty()
                    && values
                        .iter()
                        .all(|value| unmemoised_string_only(root, value, depth + 1));
            }
        }
        false
    }

    const LEAVES: [&str; 12] = [
        r#"{"type":"string"}"#,
        r#"{"type":"integer"}"#,
        r#"{"type":["null","number"]}"#,
        r#"{"const":"42"}"#,
        r#"{"enum":[1,"a"]}"#,
        r#"{"properties":{}}"#,
        r#"{"items":{}}"#,
        r#"{"maxLength":3}"#,
        r#"{"additionalProperties":true}"#,
        r##"{"$ref":"#"}"##,
        r##"{"$ref":"#/$defs/missing"}"##,
        r#"{}"#,
    ];

    const VALUES: [&str; 7] = ["\"42\"", "42", "4.5", "null", "true", "[1]", "{\"a\":1}"];

    /// The shapes `generated_root` builds; each keeps the plain recursion
    /// cheap enough to serve as the reference.
    #[derive(Clone, Copy)]
    enum Shape {
        /// References only to later definitions, up to three per part list,
        /// so shared definitions are reached along many paths.
        Shared,
        /// One reference per definition, to any definition including itself,
        /// so every cycle runs into the depth cap.
        Cyclic,
        /// A long chain whose definitions also skip ahead, so one definition
        /// is reached at several depths on both sides of the caps.
        Ladder,
    }

    /// Deterministic `$defs` of `shape`, mixing leaves, unions,
    /// intersections and reference aliases.
    fn generated_root(seed: u64, shape: Shape, definitions: usize) -> Value {
        let mut state = seed;
        let mut next = |bound: usize| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as usize) % bound
        };
        let reference = |index: usize| format!(r##"{{"$ref":"#/$defs/d{index}"}}"##);
        let mut defs = Vec::new();
        for index in 0..definitions {
            let later = definitions - index - 1;
            let mut parts = Vec::new();
            match shape {
                Shape::Shared => {
                    for _ in 0..1 + next(3) {
                        parts.push(if later == 0 || next(3) == 0 {
                            LEAVES[next(LEAVES.len())].to_string()
                        } else {
                            reference(index + 1 + next(later))
                        });
                    }
                }
                Shape::Cyclic => {
                    parts.push(reference(next(definitions)));
                    for _ in 0..next(3) {
                        parts.push(LEAVES[next(LEAVES.len())].to_string());
                    }
                }
                Shape::Ladder => {
                    if later == 0 {
                        parts.push(LEAVES[next(LEAVES.len())].to_string());
                    } else {
                        parts.push(reference(index + 1));
                        parts.push(if later > 1 && next(2) == 0 {
                            reference(index + 2 + next(later.min(3) - 1))
                        } else {
                            LEAVES[next(LEAVES.len())].to_string()
                        });
                    }
                }
            }
            let rotate = next(parts.len());
            parts.rotate_left(rotate);
            let combinator = ["anyOf", "oneOf", "allOf"][next(3)];
            let body = if !matches!(shape, Shape::Shared) && next(6) == 0 {
                // A reference alias, which string_only follows with depth.
                reference(match shape {
                    Shape::Cyclic => next(definitions),
                    _ => (index + 1).min(definitions - 1),
                })
            } else {
                format!(r#"{{"{combinator}":[{}]}}"#, parts.join(","))
            };
            defs.push(format!(r#""d{index}":{body}"#));
        }
        json(&format!(r#"{{"$defs":{{{}}}}}"#, defs.join(",")))
    }

    #[test]
    fn memoised_answers_match_the_plain_recursion() {
        let values: Vec<Value> = VALUES.iter().map(|value| json(value)).collect();
        let mut checked = 0;
        let mut accepted_count = 0;
        for seed in 0..120 {
            let (shape, definitions, starts) = match seed % 3 {
                0 => (Shape::Shared, 2 + seed as usize % 5, 3),
                1 => (Shape::Cyclic, 1 + seed as usize % 6, 3),
                _ => (Shape::Ladder, 14 + seed as usize % 8, 1),
            };
            let root = generated_root(seed, shape, definitions);
            for index in 0..starts.min(definitions) {
                let start = json(&format!(r##"{{"$ref":"#/$defs/d{index}"}}"##));
                let inside = resolve(&root, &start);
                // A copy lives outside root, as a property schema can.
                let outside = inside.cloned();
                for schema in [Some(&start), inside, outside.as_ref()]
                    .into_iter()
                    .flatten()
                {
                    let expected = unmemoised_string_only(&root, schema, 0);
                    assert_eq!(string_only(&root, schema).unwrap(), expected, "seed {seed}");
                    for value in &values {
                        let expected = unmemoised_accepts(&root, schema, value, 0);
                        assert_eq!(accepted(&root, schema, value), expected, "seed {seed}");
                        checked += 1;
                        accepted_count += usize::from(expected);
                    }
                }
            }
        }
        // Both answers occur often enough for the comparison to mean something.
        assert!(checked > 2000);
        assert!(accepted_count > checked / 5 && accepted_count < checked * 4 / 5);
    }

    #[test]
    fn depth_cap_answers_are_kept() {
        // A chain one level past the cap admits anything; one level short of
        // it still decides by kind.
        let chain = |levels: usize| {
            let mut defs = Vec::new();
            for index in 0..levels {
                defs.push(format!(
                    r##""d{index}":{{"allOf":[{{"$ref":"#/$defs/d{}"}}]}}"##,
                    index + 1
                ));
            }
            defs.push(format!(r#""d{levels}":{{"type":"integer"}}"#));
            json(&format!(r#"{{"$defs":{{{}}}}}"#, defs.join(",")))
        };
        let start = json(r##"{"$ref":"#/$defs/d0"}"##);
        assert!(!accepted(&chain(16), &start, &json("\"x\"")));
        assert!(accepted(&chain(17), &start, &json("\"x\"")));
    }

    #[test]
    fn recursive_intersections_finish_quickly() {
        // Each definition intersects itself several times: without
        // memoisation the check visits branches^17 nodes.
        for branches in [2, 3, 8] {
            let parts = vec![r##"{"$ref":"#/$defs/a"}"##; branches].join(",");
            let root = json(&format!(
                r##"{{"type":"object","$defs":{{"a":{{"allOf":[{parts}]}},"b":{{"anyOf":[{parts},{{"type":"null"}}]}}}},"properties":{{"x":{{"$ref":"#/$defs/a"}},"y":{{"$ref":"#/$defs/b"}}}}}}"##
            ));
            let properties = root.get("properties").unwrap();
            let start = std::time::Instant::now();
            for value in ["1", "\"1\"", "null"] {
                // The cap admits anything, so the cycle accepts every value,
                // as it did before.
                assert!(accepted(&root, properties.get("x").unwrap(), &json(value)));
                assert!(accepted(&root, properties.get("y").unwrap(), &json(value)));
            }
            let strings = json(&format!(
                r##"{{"$defs":{{"s":{{"anyOf":[{}, {{"type":"string"}}]}}}}}}"##,
                vec![r##"{"$ref":"#/$defs/s"}"##; branches].join(",")
            ));
            let s = json(r##"{"$ref":"#/$defs/s"}"##);
            assert!(!string_only(&strings, &s).unwrap());
            assert!(start.elapsed() < std::time::Duration::from_secs(1));
        }
    }
}
