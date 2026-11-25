//! A Responses body: its controls (`ParseOpenAiResponseControls`), items
//! (`ParseOpenAiResponseMessage`) and `input` (`ReadOpenAiResponseChat` and
//! the Responses branch of `ReadTextMessages`).

use crate::Defaults;
use crate::chat::{field, parse_body, thinking_argument};
use crate::content::responses_parts;
use crate::draft::{Draft, Message, Role, ToolChoice, Tools};
use crate::error::{Failure, Outcome, bad_request, compatibility, invalid};
use crate::fields::{
    append, body_view, concat, copy, member_str, member_str_or, non_null, push, take_text,
    write_key, write_value,
};
use crate::host::{Host, HostField};
use crate::message::{Content, historical, message};
use crate::reasoning;
use crate::tool_choice::parse_tool_controls;
use gufo_json::{Numbers, Object, Value};

/// `cache_prompt`, `cache_persist` and `return_progress`: null keeps the
/// default.
fn flag(body: &Value, key: &str, code: &'static str) -> Outcome<Option<bool>> {
    match non_null(body, key) {
        None => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(bad_request(code, format!("'{key}' must be a boolean"))),
    }
}

/// `ParseOpenAiResponseControls`: hosted tools are skipped and standard
/// members with no native effect tolerated.
pub fn controls(body: &Value, draft: &mut Draft, host: &mut impl Host) -> Outcome<()> {
    parse_tool_controls(body, &mut draft.tools, true, true, host)?;
    // Clients route namespaced calls by namespace and name.
    if let Some(items) = body.get("tools").and_then(Value::as_array) {
        for item in items {
            let name = member_str(item, "name");
            if member_str(item, "type") != "namespace" || name.is_empty() {
                continue;
            }
            let nested = item
                .get("tools")
                .and_then(Value::as_array)
                .ok_or(Failure::Internal)?;
            for tool in nested {
                if member_str(tool, "type") == "function" {
                    let pair = (copy(member_str(tool, "name"))?, copy(name)?);
                    push(&mut draft.tools.namespaces, pair)?;
                }
            }
        }
    }
    if let Some(value) = flag(body, "return_progress", "invalid_return_progress")? {
        draft.return_progress = Some(value);
    }
    if let Some(value) = flag(body, "cache_prompt", "invalid_cache_prompt")? {
        draft.cache_prompt = Some(value);
    }
    if let Some(value) = flag(body, "cache_persist", "invalid_cache_persist")? {
        draft.cache_persist = Some(value);
    }
    reasoning::responses(body, &mut draft.reasoning)?;
    reasoning::budget(body, &mut draft.reasoning)?;
    if let Some(text) = non_null(body, "text") {
        let Some(object) = text.as_object() else {
            return Err(bad_request(
                "invalid_response_format",
                "'text' must be an object",
            ));
        };
        for (key, value) in object.members() {
            if key == "verbosity" {
                continue; // No native equivalent.
            }
            if key != "format" {
                return Err(bad_request(
                    "invalid_response_format",
                    concat(&["unsupported text member: ", key])?,
                ));
            }
            field(host, HostField::TextFormat, 0, Some(value))?;
        }
    }
    Ok(())
}

fn is_function_item(kind: &str) -> bool {
    matches!(
        kind,
        "function_call" | "function_call_output" | "custom_tool_call" | "custom_tool_call_output"
    )
}

/// `ParseOpenAiResponseMessage` on a fresh message.
pub fn item(item: &mut Value, host: &mut impl Host) -> Outcome<Message> {
    let kind = member_str(item, "type");
    if is_function_item(kind) {
        let Some(id) = item
            .get("call_id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
        else {
            return invalid("function items require a nonempty call_id");
        };
        let id = copy(id)?;
        if kind == "function_call" || kind == "custom_tool_call" {
            let call = if kind == "custom_tool_call" {
                let Some(input) = item
                    .get("input")
                    .filter(|input| matches!(input, Value::String(_)))
                else {
                    return invalid("historical custom tool calls require string input");
                };
                // One free-form input, kept as literal argument data:
                // {"input":...} as C++ dumped it.
                let mut arguments = String::new();
                append(&mut arguments, "{")?;
                write_key("input", &mut arguments)?;
                write_value(input, Numbers::Canonical, &mut arguments)?;
                append(&mut arguments, "}")?;
                let arguments = Value::String(arguments);
                historical(id, item.get("name"), Some(&arguments))?
            } else {
                historical(id, item.get("name"), item.get("arguments"))?
            };
            let mut message = Message::new(Role::Assistant);
            push(&mut message.calls, call)?;
            return Ok(message);
        }
        let output = item
            .get_mut("output")
            .filter(|output| matches!(output, Value::String(_) | Value::Array(_)));
        let Some(output) = output else {
            return invalid("function output requires a string or content array");
        };
        let mut converted = Object::new();
        converted.push(copy("role")?, Value::String(copy("tool")?))?;
        converted.push(copy("tool_call_id")?, Value::String(id))?;
        converted.push(copy("content")?, output.take())?;
        return rewritten_message(&mut Value::Object(converted), host);
    }
    rewritten_message(item, host)
}

fn rewritten_message(converted: &mut Value, host: &mut impl Host) -> Outcome<Message> {
    let content = match converted.get("content").and_then(Value::as_array) {
        Some(parts) => Content::Rewritten(responses_parts(parts)?),
        None => Content::Chat,
    };
    message(converted, content, host)
}

/// The Responses branch of `ReadTextMessages`; `Failure::Invalid` with an
/// empty message where it returned false without one.
fn input_messages(
    input: Option<&mut Value>,
    messages: &mut Vec<Message>,
    host: &mut impl Host,
) -> Outcome<()> {
    let items = match input {
        Some(Value::Array(items)) if !items.is_empty() => items,
        _ => return invalid(""),
    };
    // One assistant turn may arrive as several items; a system or developer
    // item between them waits for the next message.
    let mut deferred: Vec<Message> = Vec::new();
    let flush =
        |messages: &mut Vec<Message>, deferred: &mut Vec<Message>, message| -> Outcome<()> {
            messages.try_reserve(deferred.len() + 1)?;
            messages.append(deferred);
            messages.push(message);
            Ok(())
        };
    for value in items.iter_mut() {
        let kind = member_str(value, "type");
        if is_function_item(kind) {
            let message = item(value, host)?;
            match messages.last_mut() {
                Some(last) if message.role == Role::Assistant && last.role == Role::Assistant => {
                    last.calls.try_reserve(message.calls.len())?;
                    last.calls.extend(message.calls);
                }
                _ => flush(messages, &mut deferred, message)?,
            }
            continue;
        }
        if kind == "reasoning" {
            let summary = value.get("summary").and_then(Value::as_array);
            // Codex replays null content and encrypted_content.
            let Some(summary) = summary.filter(|_| {
                non_null(value, "encrypted_content").is_none()
                    && non_null(value, "content").is_none()
            }) else {
                return invalid("");
            };
            let mut reasoning = Message::new(Role::Assistant);
            for part in summary {
                let text = part.get("text").and_then(Value::as_str);
                match text {
                    Some(text) if member_str(part, "type") == "summary_text" => {
                        append(&mut reasoning.thought, text)?;
                    }
                    _ => return invalid(""),
                }
            }
            flush(messages, &mut deferred, reasoning)?;
            continue;
        }
        let role = member_str(value, "role");
        if !matches!(value, Value::Object(_))
            || !matches!(role, "user" | "assistant" | "system" | "developer")
            || member_str_or(value, "type", "message") != "message"
            || value.get("tool_calls").is_some()
        {
            return invalid("");
        }
        let message = item(value, host)?;
        let after_assistant = messages
            .last()
            .is_some_and(|last| last.role == Role::Assistant);
        if matches!(message.role, Role::System | Role::Developer) && after_assistant {
            push(&mut deferred, message)?;
            continue;
        }
        match messages.last_mut() {
            Some(last)
                if message.role == Role::Assistant
                    && last.role == Role::Assistant
                    && last.content.is_empty()
                    && !last.thought.is_empty() =>
            {
                last.content = message.content;
            }
            _ => flush(messages, &mut deferred, message)?,
        }
    }
    messages.try_reserve(deferred.len())?;
    messages.append(&mut deferred);
    Ok(())
}

/// `ReadOpenAiResponseChat`, from the server's reasoning defaults.
pub fn decode(text: &[u8], defaults: &Defaults, host: &mut impl Host) -> Outcome<Draft> {
    let mut body = parse_body(text).map_err(compatible)?;
    let mut draft = Draft::new(defaults.reasoning, Tools::new(Vec::new(), ToolChoice::Auto));
    controls(&body, &mut draft, host)?;
    let view = body_view(&body)?;
    let thinking = thinking_argument(&draft.reasoning);
    host.field(HostField::ResponsesOptions, thinking, Some(view.as_bytes()))?;

    if let Some(instructions) = body.get_mut("instructions") {
        if !matches!(instructions, Value::String(_)) {
            return Err(compatibility("'instructions' must be a string"));
        }
        let mut system = Message::new(Role::System);
        system.content = take_text(instructions);
        push(&mut draft.messages, system)?;
    }
    match body.get_mut("input") {
        Some(input @ Value::String(_)) if input.as_str().is_some_and(|text| !text.is_empty()) => {
            let mut user = Message::new(Role::User);
            user.content = take_text(input);
            push(&mut draft.messages, user)?;
        }
        input => {
            input_messages(input, &mut draft.messages, host).map_err(|failure| match failure {
                Failure::Invalid(error) if !error.is_empty() => compatibility(error),
                Failure::Invalid(_) => compatibility(
                    "'input' must contain text, message items with text/images, reasoning items, \
                 function calls or function outputs",
                ),
                other => other,
            })?
        }
    }
    draft.stream = Some(body.get("stream").and_then(Value::as_bool) == Some(true));
    if let Some(model) = body.get("model").and_then(Value::as_str) {
        draft.model = Some(copy(model)?);
    }
    Ok(draft)
}

/// A parse error as `Err` spells it: its message ends at a NUL.
pub fn compatible(failure: Failure) -> Failure {
    match failure {
        Failure::Rejected(rejection) if rejection.code == "parse_error" => {
            let mut rejection = rejection;
            if let Some(end) = rejection.message.find('\0') {
                rejection.message.truncate(end);
            }
            Failure::Rejected(rejection)
        }
        other => other,
    }
}

/// `GUFO_CHAT_FRAGMENT_RESPONSE_CONTROLS`.
pub fn decode_controls(text: &[u8], draft: &mut Draft, host: &mut impl Host) -> Outcome<()> {
    let body = parse_body(text)?;
    controls(&body, draft, host)
}

/// `GUFO_CHAT_FRAGMENT_RESPONSE_ITEM`.
pub fn decode_item(text: &[u8], draft: &mut Draft, host: &mut impl Host) -> Outcome<()> {
    let mut value = parse_body(text)?;
    let message = item(&mut value, host).map_err(|f| f.with(400, "invalid_messages"))?;
    push(&mut draft.messages, message)
}
