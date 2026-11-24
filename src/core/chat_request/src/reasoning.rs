//! Reasoning controls. Each API keeps its own rules: Chat's `thinking`,
//! `reasoning_effort` and `chat_template_kwargs`, the shared budgets,
//! Responses' `reasoning` and Messages' `thinking` and `output_config`.

use crate::draft::Reasoning;
use crate::error::{Failure, Outcome, bad_request, compatibility, invalid};
use crate::fields::{concat, non_null, number};
use crate::host;
use gufo_json::Value;

/// `ParseReasoningEffortName`: the `ReasoningEffort` value of a name.
pub fn effort_name(value: &str) -> Option<u8> {
    match value {
        "minimal" => Some(host::EFFORT_MINIMAL),
        "low" => Some(host::EFFORT_LOW),
        "medium" => Some(host::EFFORT_MEDIUM),
        "high" => Some(host::EFFORT_HIGH),
        "xhigh" => Some(host::EFFORT_XHIGH),
        "max" => Some(host::EFFORT_MAX),
        _ => None,
    }
}

/// `AssignReasoningEnabled`.
fn assign_enabled(options: &mut Reasoning, enabled: bool) -> Outcome<()> {
    if options.enabled.is_some_and(|current| current != enabled) {
        return invalid("reasoning controls disagree about whether thinking is enabled");
    }
    options.enabled = Some(enabled);
    Ok(())
}

/// `AssignReasoningEffort`.
fn assign_effort(options: &mut Reasoning, value: &str, enable_thinking: bool) -> Outcome<()> {
    if value == "off" || value == "none" {
        return assign_enabled(options, false);
    }
    let Some(effort) = effort_name(value) else {
        return invalid("reasoning_effort must be none, minimal, low, medium, high, xhigh, or max");
    };
    if options.effort.is_some_and(|current| current != effort) {
        return invalid("top-level and chat_template_kwargs reasoning_effort disagree");
    }
    options.effort = Some(effort);
    if enable_thinking {
        assign_enabled(options, true)?;
    }
    Ok(())
}

/// A failed check whose own message wins over the field's, as the C++
/// `if (error->empty()) *error = ...` did.
fn or_message(result: Outcome<()>, message: &str) -> Outcome<()> {
    match result {
        Err(Failure::Invalid(error)) if error.is_empty() => invalid(message),
        other => other,
    }
}

/// `ParseReasoningOptions` (Chat).
pub fn chat_options(body: &Value, options: &mut Reasoning) -> Outcome<()> {
    if let Some(thinking) = body.get("thinking") {
        if !matches!(thinking, Value::Object(_)) {
            return invalid("'thinking' must be an object");
        }
        let kind = thinking.get("type").and_then(Value::as_str);
        let Some(kind @ ("enabled" | "disabled")) = kind else {
            return invalid("'thinking.type' must be enabled or disabled");
        };
        assign_enabled(options, kind == "enabled")?;
    }
    if let Some(effort) = non_null(body, "reasoning_effort") {
        let Some(effort) = effort.as_str() else {
            return invalid("'reasoning_effort' must be a string");
        };
        or_message(
            assign_effort(options, effort, true),
            "'reasoning_effort' must be a string",
        )?;
    }
    let Some(kwargs) = body.get("chat_template_kwargs") else {
        return Ok(());
    };
    if !matches!(kwargs, Value::Object(_)) {
        return invalid("'chat_template_kwargs' must be an object");
    }
    if let Some(enabled) = kwargs.get("enable_thinking") {
        let Some(enabled) = enabled.as_bool() else {
            return invalid("'chat_template_kwargs.enable_thinking' must be a boolean");
        };
        or_message(
            assign_enabled(options, enabled),
            "'chat_template_kwargs.enable_thinking' must be a boolean",
        )?;
    }
    if let Some(mode) = kwargs.get("thinking_mode") {
        let Some(mode) = mode.as_str() else {
            return invalid("'chat_template_kwargs.thinking_mode' must be a string");
        };
        if mode != "auto" {
            let enabled = mode == "thinking" || mode == "on";
            let message = "'chat_template_kwargs.thinking_mode' must be auto, thinking, or chat";
            if !enabled && mode != "chat" && mode != "off" && mode != "none" {
                return invalid(message);
            }
            or_message(assign_enabled(options, enabled), message)?;
        }
    }
    if let Some(effort) = kwargs.get("reasoning_effort") {
        let message = "'chat_template_kwargs.reasoning_effort' must be a string";
        let Some(effort) = effort.as_str() else {
            return invalid(message);
        };
        let enable = options.enabled.unwrap_or(true);
        or_message(assign_effort(options, effort, enable), message)?;
    }
    if let Some(preserve) = kwargs.get("preserve_thinking") {
        let Some(preserve) = preserve.as_bool() else {
            return invalid("'chat_template_kwargs.preserve_thinking' must be a boolean");
        };
        options.preserve_thinking = Some(preserve);
    }
    Ok(())
}

/// `ParseReasoningBudget`: llama.cpp's field or its older spelling; -1
/// keeps the default.
pub fn budget(body: &Value, options: &mut Reasoning) -> Outcome<()> {
    let mut field = "reasoning_budget_tokens";
    let mut value = non_null(body, field);
    if value.is_none() {
        field = "thinking_budget_tokens";
        value = non_null(body, field);
    }
    let Some(value) = value else {
        return Ok(());
    };
    let value = number(value).unwrap_or(-2.0);
    if !value.is_finite() || value < -1.0 || value > f64::from(i32::MAX) || value.floor() != value {
        return Err(bad_request(
            "invalid_reasoning",
            format!("'{field}' must be an integer between -1 and 2147483647"),
        ));
    }
    if value >= 0.0 {
        options.budget_tokens = Some(value as u64);
    }
    Ok(())
}

/// The Responses `reasoning` object, applied to the request's reasoning.
pub fn responses(body: &Value, reasoning: &mut Reasoning) -> Outcome<()> {
    let Some(value) = non_null(body, "reasoning") else {
        return Ok(());
    };
    let Some(object) = value.as_object() else {
        return Err(bad_request(
            "invalid_reasoning",
            "'reasoning' must be an object",
        ));
    };
    for (key, value) in object.members() {
        if key == "summary" {
            continue; // Local reasoning is always sent.
        }
        if key != "effort" {
            return Err(bad_request(
                "invalid_reasoning",
                concat(&["unsupported reasoning member: ", key])?,
            ));
        }
        if matches!(value, Value::Null) {
            continue;
        }
        let mut options = Reasoning::default();
        let result = match value.as_str() {
            Some(effort) => assign_effort(&mut options, effort, true),
            None => invalid(""),
        };
        if let Err(failure) = result {
            let failure = match failure {
                Failure::Invalid(error) if error.is_empty() => {
                    Failure::Invalid("'reasoning.effort' must be a string".into())
                }
                other => other,
            };
            return Err(failure.with(400, "invalid_reasoning"));
        }
        if options.enabled.is_some() {
            reasoning.enabled = options.enabled;
        }
        if options.effort.is_some() {
            reasoning.effort = options.effort;
        }
    }
    Ok(())
}

/// `ReadThinking` (Messages): adaptive leaves enabled to the default; an
/// enabled budget caps the reasoning tokens.
pub fn messages_thinking(body: &Value, reasoning: &mut Reasoning) -> Outcome<()> {
    let Some(thinking) = body.get("thinking") else {
        return Ok(());
    };
    let kind = thinking
        .as_object()
        .and_then(|_| thinking.get("type"))
        .and_then(Value::as_str);
    let Some(kind @ ("enabled" | "adaptive" | "disabled")) = kind else {
        return Err(compatibility(
            "'thinking.type' must be enabled, adaptive, or disabled",
        ));
    };
    if let Some(budget) = thinking.get("budget_tokens") {
        let value = number(budget);
        let Some(value) = value.filter(|value| {
            value.is_finite()
                && *value >= 1.0
                && *value <= f64::from(i32::MAX)
                && value.floor() == *value
        }) else {
            return Err(compatibility(
                "'thinking.budget_tokens' must be a positive integer at most 2147483647",
            ));
        };
        if kind == "enabled" {
            reasoning.budget_tokens = Some(value as u64);
        }
    }
    if let Some(display) = thinking.get("display")
        && !matches!(display.as_str(), Some("summarized" | "omitted" | "updates"))
    {
        return Err(compatibility(
            "'thinking.display' must be summarized, omitted, or updates",
        ));
    }
    if kind != "adaptive" {
        reasoning.enabled = Some(kind == "enabled");
    }
    Ok(())
}

/// `ReadMessagesOutputConfig`: effort never enables thinking, and Messages
/// has no minimal effort.
pub fn messages_output_config(body: &Value, reasoning: &mut Reasoning) -> Outcome<()> {
    let Some(config) = non_null(body, "output_config") else {
        return Ok(());
    };
    let Some(object) = config.as_object() else {
        return Err(compatibility("'output_config' must be an object"));
    };
    for (key, value) in object.members() {
        if key != "effort" {
            return Err(compatibility(concat(&[
                "unsupported output_config member: ",
                key,
            ])?));
        }
        if matches!(value, Value::Null) {
            continue;
        }
        let effort = value
            .as_str()
            .filter(|effort| *effort != "minimal")
            .and_then(effort_name);
        let Some(effort) = effort else {
            return Err(compatibility(
                "'output_config.effort' must be low, medium, high, xhigh, or max",
            ));
        };
        reasoning.effort = Some(effort);
    }
    Ok(())
}

/// Chat merges the server defaults after parsing, so conflicting controls
/// were still rejected.
pub fn merge_defaults(reasoning: &mut Reasoning, defaults: Reasoning) {
    reasoning.enabled = reasoning.enabled.or(defaults.enabled);
    reasoning.effort = reasoning.effort.or(defaults.effort);
    reasoning.preserve_thinking = reasoning.preserve_thinking.or(defaults.preserve_thinking);
    reasoning.budget_tokens = reasoning.budget_tokens.or(defaults.budget_tokens);
}

#[cfg(test)]
mod tests {
    use super::*;
    use gufo_json::{Limits, parse_value};

    fn decode(text: &str) -> Outcome<Reasoning> {
        let body = parse_value(text.as_bytes(), Limits::NONE).unwrap();
        let mut options = Reasoning::default();
        chat_options(&body, &mut options)?;
        budget(&body, &mut options).map(|()| options)
    }

    #[test]
    fn chat_controls_must_agree() {
        let options = decode(
            r#"{"reasoning_effort":"high","chat_template_kwargs":{"enable_thinking":true}}"#,
        )
        .unwrap();
        assert_eq!(options.enabled, Some(true));
        assert_eq!(options.effort, Some(3));
        assert_eq!(
            decode(r#"{"thinking":{"type":"disabled"},"reasoning_effort":"low"}"#),
            invalid("reasoning controls disagree about whether thinking is enabled")
        );
        assert_eq!(
            decode(r#"{"reasoning_effort":7}"#),
            invalid("'reasoning_effort' must be a string")
        );
    }

    #[test]
    fn budgets_keep_minus_one_and_bound_the_rest() {
        assert_eq!(
            decode(r#"{"reasoning_budget_tokens":-1}"#)
                .unwrap()
                .budget_tokens,
            None
        );
        assert_eq!(
            decode(r#"{"reasoning_budget_tokens":null,"thinking_budget_tokens":5}"#)
                .unwrap()
                .budget_tokens,
            Some(5)
        );
        assert!(decode(r#"{"reasoning_budget_tokens":2147483648}"#).is_err());
        assert!(decode(r#"{"reasoning_budget_tokens":1.5}"#).is_err());
        assert!(decode(r#"{"reasoning_budget_tokens":"1"}"#).is_err());
    }
}
