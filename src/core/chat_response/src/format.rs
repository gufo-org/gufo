//! Shared JSON formatting, SSE framing, and one-shot error/utility response encoders.

use std::fmt::Write;

use gufo_json::{
    Limits, Number, Numbers, Object, Spacing, Style, Value, dump, parse_value, write as json_write,
};

pub const SSE_DONE: &str = "data: [DONE]\n\n";
const COMPACT_STYLE: Style = Style {
    numbers: Numbers::Canonical,
    spacing: Spacing::Compact,
};

pub fn dump_json(value: &Value) -> String {
    dump(value, Numbers::Canonical).expect("JSON serialization out of memory")
}

const JSON_LIMITS: Limits = Limits {
    bytes: 64 << 20,
    values: 1_000_000,
};

pub fn parse_json(raw: &str) -> Option<Value> {
    if raw.is_empty() {
        return None;
    }
    parse_value(raw.as_bytes(), JSON_LIMITS).ok()
}

pub fn push_member(obj: &mut Object, key: &str, value: Value) {
    let _ = obj.push(key.to_string(), value);
}

pub fn set_member(obj: &mut Object, key: &str, value: Value) {
    if let Some(slot) = obj.get_mut(key) {
        *slot = value;
    } else {
        let _ = obj.push(key.to_string(), value);
    }
}

pub fn num_usize(v: usize) -> Value {
    Value::Number(Number::new(v as f64).expect("usize is finite"))
}

pub fn num_i64(v: i64) -> Value {
    Value::Number(Number::new(v as f64).expect("i64 is finite"))
}

pub fn num_f64(v: f64) -> Value {
    Value::Number(Number::new(if v.is_finite() { v } else { 0.0 }).expect("finite"))
}

pub fn sse_frame(value: &Value) -> String {
    let mut out = String::with_capacity(256);
    out.push_str("data: ");
    json_write(value, COMPACT_STYLE, &mut out).expect("JSON serialization out of memory");
    out.push_str("\n\n");
    out
}

pub fn named_sse_frame(event_type: &str, value: &Value) -> String {
    let mut out = String::with_capacity(event_type.len() + 256);
    out.push_str("event: ");
    out.push_str(event_type);
    out.push_str("\ndata: ");
    json_write(value, COMPACT_STYLE, &mut out).expect("JSON serialization out of memory");
    out.push_str("\n\n");
    out
}

pub fn push_json_escaped(out: &mut String, text: &str) {
    for ch in text.chars() {
        match ch {
            '"' => out.push_str(r#"\""#),
            '\\' => out.push_str(r#"\\"#),
            '\x08' => out.push_str(r#"\b"#),
            '\x0c' => out.push_str(r#"\f"#),
            '\n' => out.push_str(r#"\n"#),
            '\r' => out.push_str(r#"\r"#),
            '\t' => out.push_str(r#"\t"#),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
}

pub fn error_response_json(message: &str, error_type: &str, code: &str) -> String {
    let mut detail = Object::new();
    push_member(&mut detail, "message", Value::String(message.to_string()));
    push_member(&mut detail, "type", Value::String(error_type.to_string()));
    push_member(&mut detail, "code", Value::String(code.to_string()));
    let mut root = Object::new();
    push_member(&mut root, "error", Value::Object(detail));
    dump_json(&Value::Object(root))
}

pub fn text_generation_error_json(
    message: &str,
    code: &str,
    required_bytes: Option<usize>,
    remaining_bytes: Option<usize>,
) -> String {
    let mut detail = Object::new();
    push_member(&mut detail, "message", Value::String(message.to_string()));
    push_member(
        &mut detail,
        "type",
        Value::String("server_error".to_string()),
    );
    push_member(&mut detail, "code", Value::String(code.to_string()));
    if let Some(req) = required_bytes {
        push_member(&mut detail, "required_bytes", num_usize(req));
    }
    if let Some(rem) = remaining_bytes {
        push_member(&mut detail, "remaining_bytes", num_usize(rem));
    }
    let mut root = Object::new();
    push_member(&mut root, "error", Value::Object(detail));
    if required_bytes.is_some() || remaining_bytes.is_some() {
        let mut gufo = Object::new();
        push_member(&mut gufo, "code", Value::String(code.to_string()));
        if let Some(req) = required_bytes {
            push_member(&mut gufo, "required_bytes", num_usize(req));
        }
        if let Some(rem) = remaining_bytes {
            push_member(&mut gufo, "remaining_bytes", num_usize(rem));
        }
        let mut usage = Object::new();
        push_member(&mut usage, "gufo", Value::Object(gufo));
        push_member(&mut root, "usage", Value::Object(usage));
    }
    dump_json(&Value::Object(root))
}

pub fn input_tokens_json(input_tokens: usize) -> String {
    let mut root = Object::new();
    push_member(
        &mut root,
        "object",
        Value::String("response.input_tokens".to_string()),
    );
    push_member(&mut root, "input_tokens", num_usize(input_tokens));
    dump_json(&Value::Object(root))
}

pub fn rendered_prompt_json(prompt: &str) -> String {
    let mut root = Object::new();
    push_member(&mut root, "prompt", Value::String(prompt.to_string()));
    dump_json(&Value::Object(root))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_and_utility_encoders() {
        let err = error_response_json("bad input", "invalid_request_error", "invalid_prompt");
        assert!(err.contains(r#""code":"invalid_prompt""#));
        assert!(err.contains(r#""type":"invalid_request_error""#));

        let tg = text_generation_error_json("oom", "out_of_memory", Some(1024), Some(512));
        assert!(tg.contains(r#""required_bytes":1024"#));
        assert!(tg.contains(r#""remaining_bytes":512"#));
        assert!(tg.contains(r#""gufo":{"code":"out_of_memory""#));

        assert_eq!(
            input_tokens_json(42),
            r#"{"object":"response.input_tokens","input_tokens":42}"#
        );
        assert_eq!(rendered_prompt_json("hi\n"), r#"{"prompt":"hi\n"}"#);
    }
}
