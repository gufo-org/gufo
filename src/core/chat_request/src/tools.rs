//! Function tool declarations: the name rule, Responses namespaces and
//! hosted tools, strict schemas and the JSON the templates render.

use crate::draft::{Tool, Tools};
use crate::error::{Outcome, invalid};
use crate::fields::{append, copy, member_str, push, write_key, write_value};
use crate::host::Host;
use gufo_json::{Numbers, Object, Value};
use std::collections::HashSet;

/// Declared names reach Qwen and DeepSeek unescaped inside "<function=NAME>"
/// and "name=\"NAME\"", so the characters that frame a call are excluded, as
/// are non-ASCII bytes. Historical names are not checked.
pub const TOOL_NAME_RULE: &str = "function names require 1-64 printable ASCII characters other than \
     spaces, '<', '>', '\"' and '\\'";

/// `RenderableToolName`.
pub fn renderable(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|c| c > 0x20 && c < 0x7F && !matches!(c, b'<' | b'>' | b'"' | b'\\'))
}

/// Adds `name` to the set; false when it was there.
fn insert_name(names: &mut HashSet<Vec<u8>>, name: &[u8]) -> Outcome<bool> {
    if names.contains(name) {
        return Ok(false);
    }
    names.try_reserve(1)?;
    let mut owned = Vec::new();
    owned.try_reserve_exact(name.len())?;
    owned.extend_from_slice(name);
    names.insert(owned);
    Ok(true)
}

/// `ParseTools`: appends the function tools of `tools` to the request's,
/// whose names stay unique. With `responses`, namespaces flatten and hosted
/// tool types are skipped.
pub fn parse_tools(
    value: Option<&Value>,
    tools: &mut Tools,
    responses: bool,
    host: &mut impl Host,
) -> Outcome<()> {
    let mut names: HashSet<Vec<u8>> = HashSet::new();
    for name in tools.names() {
        insert_name(&mut names, name)?;
    }
    parse_list(value, tools, &mut names, responses, host)
}

/// `ParseToolList`.
fn parse_list(
    value: Option<&Value>,
    tools: &mut Tools,
    names: &mut HashSet<Vec<u8>>,
    responses: bool,
    host: &mut impl Host,
) -> Outcome<()> {
    let items = match value {
        None | Some(Value::Null) => return Ok(()),
        Some(Value::Array(items)) => items,
        Some(_) => return invalid("'tools' must be an array"),
    };
    for item in items {
        if !matches!(item, Value::Object(_)) {
            return invalid("'tools' entries must be objects");
        }
        let kind = member_str(item, "type");
        if responses && kind == "namespace" {
            // A namespace only groups client-executed functions; calls
            // replay by the plain name, so uniqueness rejects ambiguity.
            let nested = item
                .get("tools")
                .filter(|nested| matches!(nested, Value::Array(_)));
            let Some(nested) = nested else {
                return invalid("namespace tools require a tools array");
            };
            parse_list(Some(nested), tools, names, responses, host)?;
            continue;
        }
        if kind != "function" {
            // Hosted Responses tools only the provider can execute.
            if responses {
                continue;
            }
            return invalid("only function tools are supported");
        }
        let tool = function_tool(item, names, host)?;
        push(&mut tools.added, tool)?;
    }
    Ok(())
}

fn function_tool(
    item: &Value,
    names: &mut HashSet<Vec<u8>>,
    host: &mut impl Host,
) -> Outcome<Tool> {
    let function = item.get("function");
    if function.is_some_and(|function| !matches!(function, Value::Object(_))) {
        return invalid("'function' must be an object");
    }
    let nested = function.is_some();
    let source = function.unwrap_or(item);
    if source
        .get("strict")
        .is_some_and(|strict| !matches!(strict, Value::Null | Value::Bool(_)))
    {
        return invalid("function strict must be a boolean or null");
    }
    // OpenAI uses "parameters"; some agent clients send parametersJsonSchema.
    let parameters = match source.get("parameters") {
        None | Some(Value::Null) => source.get("parametersJsonSchema"),
        found => found,
    };
    let name = member_str(source, "name");
    if !renderable(name) {
        return invalid(TOOL_NAME_RULE);
    }
    if !insert_name(names, name.as_bytes())? {
        return invalid("function names must be unique");
    }
    if parameters.is_some_and(|parameters| !matches!(parameters, Value::Null | Value::Object(_))) {
        return invalid("function tools require an object parameters schema");
    }
    // Keep nested definitions and their field order; a flat tool's body
    // moves under "function", strict included. Both are written straight
    // from the request, as C++ dumped the copies it built.
    let strict = source.get("strict").and_then(Value::as_bool) == Some(true);
    let schema = match parameters {
        Some(schema @ Value::Object(_)) => Schema::Sent(schema),
        // Omitting parameters defines a strict function with no arguments.
        None | Some(Value::Null) if strict => Schema::Strict,
        _ => Schema::Empty,
    };
    if strict {
        let mut text = String::new();
        schema.write(Numbers::AsParsed, &mut text)?;
        host.compile_schema(text.as_bytes())?;
    }
    let mut parameters_json = String::new();
    schema.write(Numbers::Canonical, &mut parameters_json)?;
    let body = Body {
        members: source.as_object().map_or(&[][..], Object::members),
        nested,
        schema,
    };
    let mut definition = String::new();
    if nested {
        let members = item.as_object().map_or(&[][..], Object::members);
        let (mut kind, mut function) = (false, false);
        append(&mut definition, "{")?;
        for (index, (key, value)) in members.iter().enumerate() {
            if index > 0 {
                append(&mut definition, ",")?;
            }
            write_key(key, &mut definition)?;
            if key == "type" && !kind {
                kind = true;
                append(&mut definition, "\"function\"")?;
            } else if key == "function" && !function {
                function = true;
                body.write(&mut definition)?;
            } else {
                write_value(value, Numbers::Canonical, &mut definition)?;
            }
        }
        if !kind {
            append(&mut definition, if members.is_empty() { "" } else { "," })?;
            append(&mut definition, "\"type\":\"function\"")?;
        }
        if !function {
            append(&mut definition, ",\"function\":")?;
            body.write(&mut definition)?;
        }
        append(&mut definition, "}")?;
    } else {
        append(&mut definition, "{\"type\":\"function\",\"function\":")?;
        body.write(&mut definition)?;
        append(&mut definition, "}")?;
    }
    Ok(Tool {
        name: copy(name)?,
        description: copy(member_str(source, "description"))?,
        parameters: parameters_json,
        definition,
    })
}

/// A function's parameters: the request's schema, an empty one, or the
/// schema of a strict function that declares none.
#[derive(Clone, Copy)]
enum Schema<'a> {
    Sent(&'a Value),
    Empty,
    Strict,
}

const STRICT_SCHEMA: &str =
    r#"{"type":"object","properties":{},"required":[],"additionalProperties":false}"#;

impl Schema<'_> {
    fn write(self, numbers: Numbers, out: &mut String) -> Outcome<()> {
        match self {
            Self::Sent(schema) => write_value(schema, numbers, out),
            Self::Empty => append(out, "{}"),
            Self::Strict => append(out, STRICT_SCHEMA),
        }
    }
}

/// The function object C++ built: the source's members without
/// `parametersJsonSchema` (and, for a flat tool, `type`), its first
/// `parameters` replaced by the schema or the schema appended.
struct Body<'a> {
    members: &'a [(String, Value)],
    nested: bool,
    schema: Schema<'a>,
}

impl Body<'_> {
    fn write(&self, out: &mut String) -> Outcome<()> {
        append(out, "{")?;
        let mut first = true;
        let mut replaced = false;
        for (key, value) in self.members {
            if key == "parametersJsonSchema" || (!self.nested && key == "type") {
                continue;
            }
            if !first {
                append(out, ",")?;
            }
            first = false;
            write_key(key, out)?;
            if key == "parameters" && !replaced {
                replaced = true;
                self.schema.write(Numbers::Canonical, out)?;
            } else {
                write_value(value, Numbers::Canonical, out)?;
            }
        }
        if !replaced {
            append(
                out,
                if first {
                    "\"parameters\":"
                } else {
                    ",\"parameters\":"
                },
            )?;
            self.schema.write(Numbers::Canonical, out)?;
        }
        append(out, "}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_name_rule() {
        assert!(renderable("mcp__srv__t1.a:b/c"));
        assert!(!renderable(""));
        assert!(!renderable(&"n".repeat(65)));
        assert!(renderable(&"n".repeat(64)));
        for bad in ["a b", "a<b", "a>b", "a\"b", "a\\b", "a\u{7f}", "é"] {
            assert!(!renderable(bad), "{bad:?}");
        }
    }
}
