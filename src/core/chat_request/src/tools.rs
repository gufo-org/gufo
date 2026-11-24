//! Function tool declarations: the name rule, Responses namespaces and
//! hosted tools, strict schemas and the JSON the templates render.

use crate::draft::{Tool, Tools};
use crate::error::{Failure, Outcome, invalid};
use crate::fields::{assign, copy, dump, dump_as_parsed, member_str, push, string};
use crate::host::Host;
use gufo_json::{Object, Value};
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

/// The parameters a strict function without them gets.
fn empty_strict_schema() -> Outcome<Value> {
    let mut schema = Object::new();
    schema.push(copy("type")?, string("object")?)?;
    schema.push(copy("properties")?, Value::Object(Object::new()))?;
    schema.push(copy("required")?, Value::Array(Vec::new()))?;
    schema.push(copy("additionalProperties")?, Value::Bool(false))?;
    Ok(Value::Object(schema))
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
        names.insert(name.to_vec());
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
    if !names.insert(name.as_bytes().to_vec()) {
        return invalid("function names must be unique");
    }
    if parameters.is_some_and(|parameters| !matches!(parameters, Value::Null | Value::Object(_))) {
        return invalid("function tools require an object parameters schema");
    }
    // Keep nested definitions and their field order; a flat tool's body
    // moves under "function", strict included.
    let mut body = Object::new();
    let members = source.as_object().map_or(&[][..], Object::members);
    for (key, value) in members {
        if key == "parametersJsonSchema" || (!nested && key == "type") {
            continue;
        }
        body.push(copy(key)?, value.clone())?;
    }
    let schema = match parameters {
        Some(schema @ Value::Object(_)) => schema.clone(),
        _ => Value::Object(Object::new()),
    };
    assign(&mut body, "parameters", schema)?;
    if body.get("strict").and_then(Value::as_bool) == Some(true) {
        // Omitting parameters defines a function with no arguments.
        if matches!(parameters, None | Some(Value::Null)) {
            assign(&mut body, "parameters", empty_strict_schema()?)?;
        }
        let schema = body.get("parameters").ok_or(Failure::Internal)?;
        host.compile_schema(dump_as_parsed(schema)?.as_bytes())?;
    }
    let parameters_json = dump(body.get("parameters").ok_or(Failure::Internal)?)?;
    let mut definition = Object::new();
    if nested {
        let members = item.as_object().map_or(&[][..], Object::members);
        for (key, value) in members {
            if key != "function" {
                definition.push(copy(key)?, value.clone())?;
            } else {
                // Placeholder; `assign` below fills the first one in place.
                definition.push(copy(key)?, Value::Null)?;
            }
        }
    }
    assign(&mut definition, "type", string("function")?)?;
    let description = copy(member_str(source, "description"))?;
    let name = copy(name)?;
    assign(&mut definition, "function", Value::Object(body))?;
    Ok(Tool {
        name,
        description,
        parameters: parameters_json,
        definition: dump(&Value::Object(definition))?,
    })
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
