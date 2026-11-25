//! A Messages body (`ReadAnthropicChat`): its tools
//! (`ParseAnthropicToolControls`), content blocks
//! (`ParseAnthropicToolMessage` and the Messages branch of
//! `ReadTextMessages`), `system` and `messages`.

use crate::Defaults;
use crate::chat::{field, parse_body, thinking_argument};
use crate::content::{anthropic_image, read_text_content};
use crate::draft::{Draft, Message, Role, ToolChoice, Tools};
use crate::error::{Failure, Outcome, bad_request, compatibility, invalid};
use crate::fields::{append, body_view, copy, dump, member_str, member_str_or, push, string};
use crate::host::{Host, HostField};
use crate::message::{historical, role_from};
use crate::reasoning;
use crate::responses::compatible;
use crate::tool_choice::parse_tool_controls;
use gufo_json::{Object, Value};

/// `ParseAnthropicToolControls`: the Messages declarations rewritten as
/// Chat tools, so validation, schemas and framing stay on the Chat path. The
/// declarations move out of `body`, which nothing reads after them.
pub fn tool_controls(body: &mut Value, draft: &mut Draft, host: &mut impl Host) -> Outcome<()> {
    if body.get("tools").is_none() && body.get("tool_choice").is_none() {
        return Ok(());
    }
    let mut converted = Object::new();
    if let Some(tools) = body.get_mut("tools") {
        let Value::Array(items) = tools else {
            return Err(bad_request("invalid_tools", "'tools' must be an array"));
        };
        let mut functions = Vec::new();
        functions.try_reserve_exact(items.len())?;
        for tool in items.iter_mut() {
            let schema = tool.get("input_schema");
            let (true, "custom", true) = (
                matches!(tool, Value::Object(_)),
                member_str_or(tool, "type", "custom"),
                schema.is_some(),
            ) else {
                return Err(bad_request(
                    "invalid_tools",
                    "only custom tools with an input_schema are supported",
                ));
            };
            let mut function = Object::new();
            for key in ["name", "description"] {
                if let Some(value) = tool.get_mut(key) {
                    function.push(copy(key)?, value.take())?;
                }
            }
            let schema = tool.get_mut("input_schema").ok_or(Failure::Internal)?;
            function.push(copy("parameters")?, schema.take())?;
            let mut wrapped = Object::new();
            wrapped.push(copy("type")?, string("function")?)?;
            wrapped.push(copy("function")?, Value::Object(function))?;
            functions.push(Value::Object(wrapped));
        }
        converted.push(copy("tools")?, Value::Array(functions))?;
    }
    if let Some(choice) = body.get_mut("tool_choice") {
        let kind = match choice {
            Value::Object(_) => member_str(choice, "type"),
            _ => "",
        };
        // A static copy of the type, so the forced name can move out.
        let kind: &'static str = match kind {
            "auto" => "auto",
            "none" => "none",
            "any" => "any",
            "tool" => "tool",
            _ => "",
        };
        let converted_choice = match kind {
            "auto" | "none" => string(kind)?,
            "any" => string("required")?,
            "tool" => {
                let mut function = Object::new();
                if let Some(name) = choice.get_mut("name") {
                    function.push(copy("name")?, name.take())?;
                }
                let mut forced = Object::new();
                forced.push(copy("type")?, string("function")?)?;
                forced.push(copy("function")?, Value::Object(function))?;
                Value::Object(forced)
            }
            _ => {
                return Err(bad_request(
                    "invalid_tool_choice",
                    "'tool_choice.type' must be auto, any, tool or none",
                ));
            }
        };
        converted.push(copy("tool_choice")?, converted_choice)?;
        if let Some(single) = choice.get("disable_parallel_tool_use") {
            let Some(single) = single.as_bool() else {
                return Err(bad_request(
                    "invalid_tools",
                    "'disable_parallel_tool_use' must be a boolean",
                ));
            };
            if kind != "tool" {
                converted.push(copy("parallel_tool_calls")?, Value::Bool(!single))?;
            }
        }
    }
    parse_tool_controls(
        &Value::Object(converted),
        &mut draft.tools,
        false,
        false,
        host,
    )
}

/// A text block of a tool turn.
fn read_text(part: &Value, out: &mut String) -> Outcome<()> {
    match part.get("text").and_then(Value::as_str) {
        Some(text) if member_str(part, "type") == "text" => append(out, text),
        _ => invalid("unsupported Messages content block"),
    }
}

/// A user text, image or tool result block.
fn read_part(part: &Value, target: &mut Message, host: &mut impl Host) -> Outcome<()> {
    if matches!(part, Value::Object(_)) && member_str(part, "type") == "image" {
        return anthropic_image(part, target, host);
    }
    read_text(part, &mut target.content)
}

/// `ParseAnthropicToolMessage`: each tool_result becomes a tool message,
/// with its images; the text and images of a user turn follow as a user
/// message.
fn tool_message(item: &Value, messages: &mut Vec<Message>, host: &mut impl Host) -> Outcome<()> {
    let role = member_str(item, "role");
    let content = item.get("content").and_then(Value::as_array);
    let (true, "user" | "assistant", Some(content)) =
        (matches!(item, Value::Object(_)), role, content)
    else {
        return invalid("tool blocks require a user or assistant content array");
    };
    if role == "assistant" {
        let mut message = Message::new(Role::Assistant);
        for part in content {
            let kind = match part {
                Value::Object(_) => member_str(part, "type"),
                _ => "",
            };
            match kind {
                "thinking" => {
                    let Some(thinking) = part.get("thinking").and_then(Value::as_str) else {
                        return invalid("thinking blocks require thinking text");
                    };
                    append(&mut message.thought, thinking)?;
                }
                "tool_use" => {
                    let id = part
                        .get("id")
                        .and_then(Value::as_str)
                        .filter(|id| !id.is_empty());
                    let input = part
                        .get("input")
                        .filter(|input| matches!(input, Value::Object(_)));
                    let (Some(id), Some(input)) = (id, input) else {
                        return invalid(
                            "tool_use blocks require a non-empty id and an object input",
                        );
                    };
                    let arguments = Value::String(dump(input)?);
                    let call = historical(copy(id)?, part.get("name"), Some(&arguments))?;
                    push(&mut message.calls, call)?;
                }
                _ => read_text(part, &mut message.content)?,
            }
        }
        return push(messages, message);
    }
    // One user turn carries the results of the previous calls, then text.
    let mut text: Option<Message> = None;
    for part in content {
        if !matches!(part, Value::Object(_)) || member_str(part, "type") != "tool_result" {
            let target = text.get_or_insert_with(|| Message::new(Role::User));
            read_part(part, target, host)?;
            continue;
        }
        if let Some(message) = text.take() {
            push(messages, message)?;
        }
        let id = part
            .get("tool_use_id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty());
        let failed = part.get("is_error");
        let Some(id) = id.filter(|_| failed.is_none_or(|failed| matches!(failed, Value::Bool(_))))
        else {
            return invalid("tool_result blocks require a non-empty tool_use_id");
        };
        let mut result = Message::new(Role::Tool);
        result.tool_call_id = copy(id)?;
        match part.get("content") {
            None => {}
            Some(Value::String(output)) => result.content = copy(output)?,
            Some(Value::Array(blocks)) => {
                for block in blocks {
                    read_part(block, &mut result, host)?;
                }
            }
            Some(_) => {
                return invalid("tool_result content must be a string, text or image blocks");
            }
        }
        push(messages, result)?;
    }
    if let Some(message) = text {
        push(messages, message)?;
    }
    Ok(())
}

fn has_tool_blocks(item: &Value) -> bool {
    let parts = item.get("content").and_then(Value::as_array);
    matches!(item, Value::Object(_))
        && parts.is_some_and(|parts| {
            parts.iter().any(|part| {
                matches!(part, Value::Object(_))
                    && matches!(member_str(part, "type"), "tool_use" | "tool_result")
            })
        })
}

/// The Messages branch of `ReadTextMessages`; `Failure::Invalid` with an
/// empty message where it returned false without one.
fn messages(
    input: Option<&mut Value>,
    out: &mut Vec<Message>,
    host: &mut impl Host,
) -> Outcome<()> {
    let items = match input {
        Some(Value::Array(items)) if !items.is_empty() => items,
        _ => return invalid(""),
    };
    for item in items.iter_mut() {
        if has_tool_blocks(item) {
            tool_message(item, out, host)?;
            continue;
        }
        let role = member_str(item, "role");
        if !matches!(item, Value::Object(_))
            || !matches!(role, "user" | "assistant" | "system" | "developer")
            || member_str_or(item, "type", "message") != "message"
            || item.get("tool_calls").is_some()
        {
            return invalid("");
        }
        let role = role_from(role);
        let mut message = Message::new(role);
        let thought = role == Role::Assistant;
        let images = role == Role::User;
        if !read_text_content(item.get_mut("content"), &mut message, thought, images, host)? {
            return invalid("");
        }
        push(out, message)?;
    }
    Ok(())
}

/// `ReadAnthropicChat`, from the server's reasoning defaults.
pub fn decode(text: &[u8], defaults: &Defaults, host: &mut impl Host) -> Outcome<Draft> {
    let mut body = parse_body(text).map_err(compatible)?;
    let mut draft = Draft::new(defaults.reasoning, Tools::new(Vec::new(), ToolChoice::Auto));
    if matches!(body, Value::Object(_)) {
        reasoning::messages_thinking(&body, &mut draft.reasoning)?;
        reasoning::messages_output_config(&body, &mut draft.reasoning)?;
    }
    let view = body_view(&body)?;
    let thinking = thinking_argument(&draft.reasoning);
    host.field(HostField::MessagesOptions, thinking, Some(view.as_bytes()))?;
    tool_controls(&mut body, &mut draft, host)?;

    if let Some(system) = body.get_mut("system") {
        let mut message = Message::new(Role::System);
        match read_text_content(Some(system), &mut message, false, false, host) {
            Ok(true) => {}
            Ok(false) | Err(Failure::Invalid(_)) => {
                return Err(compatibility("'system' must contain only text"));
            }
            Err(other) => return Err(other),
        }
        push(&mut draft.messages, message)?;
    }
    messages(body.get_mut("messages"), &mut draft.messages, host).map_err(
        |failure| match failure {
            Failure::Invalid(error) if !error.is_empty() => compatibility(error),
            Failure::Invalid(_) => compatibility(
                "'messages' must contain text, image, thinking, tool_use or tool_result blocks",
            ),
            other => other,
        },
    )?;
    field(
        host,
        HostField::StopSequences,
        0,
        body.get("stop_sequences"),
    )?;
    draft.stream = Some(body.get("stream").and_then(Value::as_bool) == Some(true));
    if let Some(model) = body.get("model").and_then(Value::as_str) {
        draft.model = Some(copy(model)?);
    }
    Ok(draft)
}

/// `GUFO_CHAT_FRAGMENT_ANTHROPIC_TOOLS`.
pub fn decode_tools(text: &[u8], draft: &mut Draft, host: &mut impl Host) -> Outcome<()> {
    let mut body = parse_body(text)?;
    tool_controls(&mut body, draft, host)
}
