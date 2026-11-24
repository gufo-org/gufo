//! A Chat Completions body (`ParseRequest`), in its check order.

use crate::Defaults;
use crate::draft::{Draft, Reasoning, ToolChoice, Tools};
use crate::error::{Failure, Outcome, Rejection, bad_request};
use crate::fields::{body_view, concat, copy, dump_as_parsed, member_str, non_null, number, push};
use crate::host::{Host, HostField};
use crate::message::{Content, message};
use crate::reasoning;
use crate::tool_choice::parse_tool_controls;
use gufo_json::{Limits, Value, ValueError, parse_value};

/// `json::parse(body)`; a rejection keeps the C++ "JSON parse error: " prefix.
pub fn parse_body(text: &[u8]) -> Outcome<Value> {
    match parse_value(text, Limits::NONE) {
        Ok(value) => Ok(value),
        Err(ValueError::Rejected(reason)) => Err(bad_request(
            "parse_error",
            format!("JSON parse error: {reason}"),
        )),
        Err(ValueError::NoMemory) => Err(Failure::NoMemory),
        Err(ValueError::TooLarge | ValueError::Internal) => Err(Failure::Internal),
    }
}

/// The thinking mode a sampling helper resolves its defaults by.
pub fn thinking_argument(reasoning: &Reasoning) -> i32 {
    reasoning.enabled.map_or(-1, i32::from)
}

/// A helper's value: present values as parsed, absent ones as nothing.
pub fn field(
    host: &mut impl Host,
    field: HostField,
    argument: i32,
    value: Option<&Value>,
) -> Outcome<()> {
    match value {
        Some(value) => host.field(field, argument, Some(dump_as_parsed(value)?.as_bytes())),
        None => host.field(field, argument, None),
    }
}

/// An optional boolean member: null keeps the default.
fn flag(body: &Value, key: &str, code: &'static str) -> Outcome<Option<bool>> {
    match non_null(body, key) {
        None => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(bad_request(code, format!("'{key}' must be a boolean"))),
    }
}

pub fn decode(text: &[u8], defaults: &Defaults, host: &mut impl Host) -> Outcome<Draft> {
    let mut body = parse_body(text)?;
    if !matches!(body, Value::Object(_)) {
        return Err(bad_request(
            "invalid_body",
            "request body must be a JSON object",
        ));
    }
    let mut draft = Draft::new(
        Reasoning::default(),
        Tools::new(Vec::new(), ToolChoice::Auto),
    );
    if non_null(&body, "model").is_some_and(|model| !matches!(model, Value::String(_))) {
        return Err(bad_request("invalid_model", "'model' must be a string"));
    }
    let mut model = copy(member_str(&body, "model"))?;
    if model.is_empty() && !defaults.require_model {
        model = copy(&defaults.model_id)?;
    }
    if model.is_empty() {
        return Err(bad_request("missing_model", "'model' is required"));
    }
    if !host.serves_model(&model)? {
        return Err(Failure::Rejected(Rejection {
            status: 404,
            code: "model_not_found",
            message: concat(&["model '", &model, "' is not served by this process"])?,
        }));
    }
    draft.model = Some(model);

    field(
        host,
        HostField::ResponseFormat,
        0,
        body.get("response_format"),
    )?;
    draft.cache_prompt = flag(&body, "cache_prompt", "invalid_cache_prompt")?;
    draft.cache_persist = flag(&body, "cache_persist", "invalid_cache_persist")?;
    field(host, HostField::Stop, 0, body.get("stop"))?;

    let items = match body.get_mut("messages") {
        Some(Value::Array(items)) if !items.is_empty() => items,
        _ => {
            return Err(bad_request(
                "missing_messages",
                "'messages' must be a non-empty array",
            ));
        }
    };
    draft.messages.try_reserve_exact(items.len())?;
    for item in items.iter_mut() {
        let decoded =
            message(item, Content::Chat, host).map_err(|f| f.with(400, "invalid_messages"))?;
        push(&mut draft.messages, decoded)?;
    }

    parse_tool_controls(&body, &mut draft.tools, false, false, host)?;
    reasoning::chat_options(&body, &mut draft.reasoning)
        .map_err(|f| f.with(400, "invalid_reasoning"))?;
    reasoning::budget(&body, &mut draft.reasoning)?;
    if let Some(kwargs) = body.get("chat_template_kwargs")
        && let Some(vision_id) = kwargs.get("add_vision_id")
    {
        let Some(vision_id) = vision_id.as_bool() else {
            return Err(bad_request(
                "invalid_template_options",
                "'chat_template_kwargs.add_vision_id' must be a boolean",
            ));
        };
        draft.add_vision_id = Some(vision_id);
    }
    reasoning::merge_defaults(&mut draft.reasoning, defaults.reasoning);

    draft.stream = flag(&body, "stream", "invalid_stream")?;
    draft.return_progress = flag(&body, "return_progress", "invalid_return_progress")?;
    if let Some(options) = non_null(&body, "stream_options") {
        if !matches!(options, Value::Object(_)) {
            return Err(bad_request(
                "invalid_stream_options",
                "'stream_options' must be an object",
            ));
        }
        if let Some(include_usage) = options.get("include_usage") {
            let Some(include_usage) = include_usage.as_bool() else {
                return Err(bad_request(
                    "invalid_stream_options",
                    "'stream_options.include_usage' must be a boolean",
                ));
            };
            draft.include_usage = Some(include_usage);
        }
    }

    let mut max_tokens_field = "max_completion_tokens";
    let mut max_tokens = non_null(&body, max_tokens_field);
    if max_tokens.is_none() {
        max_tokens_field = "max_tokens";
        max_tokens = non_null(&body, max_tokens_field);
    }
    if let Some(value) = max_tokens {
        let valid = number(value).filter(|value| {
            value.is_finite()
                && value.floor() == *value
                && *value >= 1.0
                && *value <= f64::from(u32::MAX)
        });
        let Some(value) = valid else {
            return Err(bad_request(
                "invalid_max_tokens",
                format!("'{max_tokens_field}' must be a positive integer"),
            ));
        };
        draft.max_tokens = Some(value as u64);
    }

    let view = body_view(&body)?;
    let thinking = thinking_argument(&draft.reasoning);
    host.field(HostField::Sampling, thinking, Some(view.as_bytes()))?;
    if non_null(&body, "n").is_some_and(|n| number(n) != Some(1.0)) {
        return Err(bad_request("unsupported_n", "only n=1 is supported"));
    }
    // Raw Completions owns the fixed-length benchmark contract.
    if non_null(&body, "ignore_eos").is_some() {
        return Err(bad_request(
            "unsupported_field",
            "request field 'ignore_eos' is not supported on this endpoint",
        ));
    }
    host.field(HostField::Logprobs, 0, Some(view.as_bytes()))?;
    for unsupported in ["modalities", "audio"] {
        let Some(value) = non_null(&body, unsupported) else {
            continue;
        };
        let text_only = unsupported == "modalities"
            && matches!(value.as_array(), Some([only]) if only.as_str() == Some("text"));
        if !text_only {
            return Err(bad_request(
                "unsupported_field",
                format!("request field '{unsupported}' is not implemented"),
            ));
        }
    }
    Ok(draft)
}
