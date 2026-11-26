//! Chat messages: roles, content, reasoning and historical tool calls.

use crate::content::{Rewrite, chat_content, parts, rewritten};
use crate::draft::{Argument, Call, Message, Part, Role};
use crate::error::{Failure, Outcome, invalid};
use crate::fields::{copy, member_str, member_str_or, push, take_text};
use crate::host::Host;
use gufo_json::{Limits, Numbers, Value, ValueError, parse_value};

/// `IsKnownRole` and `ParseRole`.
pub fn known_role(role: &str) -> Option<Role> {
    match role {
        "system" => Some(Role::System),
        "developer" => Some(Role::Developer),
        "user" => Some(Role::User),
        "assistant" => Some(Role::Assistant),
        "tool" => Some(Role::Tool),
        _ => None,
    }
}

/// `RoleFrom` (src/cli/serve/http_server.cpp): unknown roles are user.
pub fn role_from(role: &str) -> Role {
    known_role(role).unwrap_or(Role::User)
}

/// `ParseArguments`: a JSON object's members, strings as they are and other
/// values as the reference templates' tojson writes them, numbers as the
/// client spelled them.
pub fn arguments(text: &str) -> Outcome<Vec<Argument>> {
    let parsed = match parse_value(text.as_bytes(), Limits::NONE) {
        Ok(value) => value,
        Err(ValueError::Rejected(reason)) => {
            return invalid(format!(
                "tool arguments are not valid JSON: JSON parse error: {reason}"
            ));
        }
        Err(ValueError::NoMemory) => return Err(Failure::NoMemory),
        Err(ValueError::TooLarge | ValueError::Internal) => return Err(Failure::Internal),
    };
    let Some(object) = parsed.into_object() else {
        return invalid("tool arguments must encode a JSON object");
    };
    let mut out = Vec::new();
    out.try_reserve_exact(object.len())?;
    for (name, value) in object.into_members() {
        let argument = match value {
            Value::String(value) => Argument {
                name,
                value,
                is_string: true,
            },
            other => Argument {
                name,
                value: gufo_json::tojson(&other, Numbers::AsParsed)?,
                is_string: false,
            },
        };
        out.push(argument);
    }
    Ok(out)
}

/// `ParseHistoricalFunction` on a function's name and arguments members.
pub fn historical(
    id: String,
    name: Option<&Value>,
    arguments_value: Option<&Value>,
) -> Outcome<Call> {
    let name = name.and_then(Value::as_str).filter(|name| !name.is_empty());
    let (Some(name), Some(Value::String(text))) = (name, arguments_value) else {
        return invalid("historical function calls require a non-empty name and string arguments");
    };
    // History is a record, not a declaration; NUL cannot pass the DeepSeek
    // tokenizer's C-string interface.
    if name.contains('\0') {
        return invalid("historical function names cannot contain NUL");
    }
    Ok(Call {
        id,
        name: copy(name)?,
        arguments: arguments(text)?,
    })
}

/// How the content of a message is read: as Chat parts, or as the Responses
/// rewrite of each part (see `content::responses_parts`).
pub enum Content {
    Chat,
    Rewritten(Vec<Rewrite>),
}

/// `ParseMessage` on a fresh message.
pub fn message(value: &mut Value, content: Content, host: &mut impl Host) -> Outcome<Message> {
    if !matches!(value, Value::Object(_)) {
        return invalid("each message must be an object");
    }
    let Some(role) = known_role(member_str(value, "role")) else {
        return invalid("message role must be system, developer, user, assistant, or tool");
    };
    let mut message = Message::new(role);
    message.name = copy(member_str(value, "name"))?;
    message.tool_call_id = copy(member_str(value, "tool_call_id"))?;
    match content {
        Content::Chat => chat_content(value.get_mut("content"), &mut message, host)?,
        Content::Rewritten(kinds) => {
            let items = value
                .get("content")
                .and_then(Value::as_array)
                .ok_or(Failure::Internal)?;
            let views = kinds
                .iter()
                .zip(items)
                .map(|(kind, part)| rewritten(*kind, part));
            parts(views, &mut message, host)?;
        }
    }
    if let Some(reasoning) = value
        .get_mut("reasoning_content")
        .filter(|reasoning| !matches!(reasoning, Value::Null))
    {
        if role != Role::Assistant || !matches!(reasoning, Value::String(_)) {
            return invalid("'reasoning_content' is only valid as a string on assistant messages");
        }
        let mut thought = take_text(reasoning);
        if !message.parts.is_empty() && !thought.is_empty() {
            // It is the first thought, before the content's parts.
            let shift = thought.len();
            for part in &mut message.parts {
                if part.thought {
                    part.end += shift;
                }
            }
            message.parts.try_reserve(1)?;
            message.parts.insert(
                0,
                Part {
                    thought: true,
                    end: shift,
                },
            );
            thought.try_reserve(message.thought.len())?;
            thought.push_str(&message.thought);
        }
        message.thought = thought;
    }

    let Some(tool_calls) = value.get("tool_calls") else {
        return Ok(message);
    };
    let (Role::Assistant, Some(items)) = (role, tool_calls.as_array()) else {
        return invalid("'tool_calls' is only valid as an array on assistant messages");
    };
    for item in items {
        if !matches!(item, Value::Object(_))
            || member_str_or(item, "type", "function") != "function"
        {
            return invalid("only function tool calls are supported");
        }
        let Some(function) = item
            .get("function")
            .filter(|function| matches!(function, Value::Object(_)))
        else {
            return invalid("assistant tool calls require a function object");
        };
        let id = copy(member_str(item, "id"))?;
        let call = historical(id, function.get("name"), function.get("arguments"))?;
        push(&mut message.calls, call)?;
    }
    Ok(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_keep_spelling_and_tojson_spacing() {
        let parsed = arguments(r#"{"a":"x","b":[1.0,{"c":2.50}],"d":-0}"#).unwrap();
        let values: Vec<_> = parsed
            .iter()
            .map(|a| (a.name.as_str(), a.value.as_str(), a.is_string))
            .collect();
        assert_eq!(
            values,
            [
                ("a", "x", true),
                ("b", r#"[1.0, {"c": 2.50}]"#, false),
                ("d", "-0", false)
            ]
        );
        assert_eq!(
            arguments("[1]"),
            invalid("tool arguments must encode a JSON object")
        );
        assert!(matches!(
            arguments("{"),
            Err(Failure::Invalid(message)) if message.starts_with("tool arguments are not valid JSON: JSON parse error: ")
        ));
    }

    #[test]
    fn roles() {
        assert_eq!(known_role("developer"), Some(Role::Developer));
        assert_eq!(known_role("bot"), None);
        assert_eq!(role_from("bot"), Role::User);
    }
}
