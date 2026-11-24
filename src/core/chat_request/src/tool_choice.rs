//! `tool_choice`, `parallel_tool_calls` and constrained tools.

use crate::draft::{ToolChoice, Tools};
use crate::error::{Outcome, bad_request, invalid};
use crate::fields::{copy, member_str};
use crate::host::Host;
use crate::tools::parse_tools;
use gufo_json::Value;

/// `ParseToolChoice`.
fn parse_choice(value: Option<&Value>, tools: &mut Tools) -> Outcome<()> {
    let Some(value) = value.filter(|value| !matches!(value, Value::Null)) else {
        return Ok(());
    };
    if let Value::String(choice) = value {
        tools.forced_name = Some(String::new());
        let choice = match choice.as_str() {
            "auto" => Some(ToolChoice::Auto),
            "none" => Some(ToolChoice::None),
            "required" => Some(ToolChoice::Required),
            _ => None,
        };
        if let Some(choice) = choice {
            tools.set_choice(choice);
            return Ok(());
        }
    }
    if matches!(value, Value::Object(_)) && member_str(value, "type") == "function" {
        let name = match value.get("function") {
            Some(function) => member_str(function, "name"),
            None => member_str(value, "name"),
        };
        if tools.select(name) {
            tools.set_choice(ToolChoice::Required);
            tools.forced_name = Some(copy(name)?);
            return Ok(());
        }
        return invalid("'tool_choice' must name a declared function");
    }
    invalid("'tool_choice' must be auto, none, required, or a declared function")
}

/// `ParseToolControls` on `body`: `nullable_parallel` lets
/// `parallel_tool_calls` be null and `responses` reads Responses tools.
pub fn parse_tool_controls(
    body: &Value,
    tools: &mut Tools,
    nullable_parallel: bool,
    responses: bool,
    host: &mut impl Host,
) -> Outcome<()> {
    parse_tools(body.get("tools"), tools, responses, host)
        .and_then(|()| parse_choice(body.get("tool_choice"), tools))
        .map_err(|failure| failure.with(400, "invalid_tools"))?;
    if tools.choice == ToolChoice::Required && tools.is_empty() {
        return Err(bad_request(
            "invalid_tool_choice",
            "'tool_choice' cannot be required without tools",
        ));
    }
    if let Some(parallel) = body.get("parallel_tool_calls")
        && !(nullable_parallel && matches!(parallel, Value::Null))
    {
        let Some(parallel) = parallel.as_bool() else {
            return Err(bad_request(
                "invalid_tools",
                "'parallel_tool_calls' must be a boolean",
            ));
        };
        tools.parallel = Some(parallel);
    }
    if matches!(body.get("tool_choice"), Some(Value::Object(_))) {
        // Forced functions execute exactly once.
        tools.parallel = Some(false);
    }
    tools.constrained = Some(!tools.is_empty() && tools.choice != ToolChoice::None);
    Ok(())
}
