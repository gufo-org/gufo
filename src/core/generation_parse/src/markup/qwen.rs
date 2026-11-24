//! Qwen calls, interpreted against the advertised schemas.

use crate::args::json_out::{arguments_json, partial_string, prefix};
use crate::args::python_literals;
use crate::args::schema::{accepts, resolve, string_only};
use crate::args::{Argument, Call};
use crate::config::{Error, Host, QwenParameter, Tool};
use crate::quote::QuoteTracker;
use crate::text::{NPOS, find, trim, try_parse_json, utf8};
use gufo_json::{Numbers, Value, dump};

/// The schema's exact key. A declared key with surrounding spaces is
/// preserved; otherwise the spelled name is trimmed, as llama.cpp's mapper
/// does.
fn parameter_name(properties: Option<&Value>, spelled: &[u8]) -> Result<String, Error> {
    if let Some(Value::Object(object)) = properties
        && let Ok(key) = std::str::from_utf8(spelled)
        && object.get(key).is_some()
    {
        return utf8(spelled);
    }
    utf8(trim(spelled))
}

/// One scanned parameter against the advertised schema. Qwen's XML-like
/// arguments carry no type marker, so the schema decides whether the value is
/// a string or JSON; without one the raw text is kept. `None` rejects it.
fn interpret(
    schema: Option<&Value>,
    properties: Option<&Value>,
    parameter: &QwenParameter,
) -> Result<Option<Argument>, Error> {
    let name = parameter_name(properties, &parameter.name)?;
    if name.is_empty() {
        return Ok(None);
    }
    let property = properties.and_then(|properties| properties.get(&name));
    let typed_schema = |property| schema.zip(Some(property));
    let string_allowed = match property.and_then(typed_schema) {
        None => true,
        Some((root, property)) => accepts(root, property, &Value::String(utf8(&parameter.value)?))?,
    };
    // Prefer text if the schema permits it; parsing ambiguous scalars as JSON
    // would silently change a caller's declared string type. A union also
    // admitting other types tries those first, as llama.cpp's qwen3-coder
    // parser does.
    let mut typed = None;
    if string_allowed
        && let Some((root, property)) = property.and_then(typed_schema)
        && let Some(value) = try_parse_json(trim(&parameter.value))?
        && !matches!(value, Value::String(_))
        && accepts(root, property, &value)?
    {
        typed = Some(value);
    }
    let is_string = string_allowed && typed.is_none();
    let mut raw = if is_string {
        parameter.value.clone()
    } else {
        trim(&parameter.value).to_vec()
    };
    if !is_string && typed.is_none() {
        let (root, property) = property.and_then(typed_schema).ok_or(Error::Internal)?;
        let mut parsed = match try_parse_json(&raw)? {
            Some(value) if accepts(root, property, &value)? => Some(value),
            _ => None,
        };
        if parsed.is_none() {
            let converted = python_literals::to_json(&raw);
            if let Some(alternative) = try_parse_json(&converted)?
                && accepts(root, property, &alternative)?
            {
                raw = converted;
                parsed = Some(alternative);
            }
        }
        // As in llama.cpp, a value the grammar admitted is kept when its
        // schema cannot be enforced natively (e.g. a recursive or empty one).
        // Recovery without a grammar checks the whole call.
        if parsed.is_none() {
            return Ok(None);
        }
    }
    Ok(Some(Argument {
        name,
        value: utf8(&raw)?,
        is_string,
    }))
}

/// The calls of `text` opened by `open`, appended to `calls`, and with
/// `spans` the byte range of each accepted one. With `partial`, the one
/// unfinished call a declared tool can still complete ends the list.
#[allow(clippy::too_many_arguments)]
pub fn parse_calls(
    text: &[u8],
    open: &[u8],
    tools: &[Tool],
    calls: &mut Vec<Call>,
    mut spans: Option<&mut Vec<(usize, usize)>>,
    require_schema: bool,
    partial: bool,
    host: &mut dyn Host,
) -> Result<(), Error> {
    let mut call_quotes = QuoteTracker::default();
    call_quotes.reset(text, 0);
    let mut cursor = 0;
    loop {
        cursor = find(text, open, cursor);
        if cursor == NPOS || open.is_empty() {
            return Ok(());
        }
        if call_quotes.quoted_at(cursor) || (partial && call_quotes.unclosed_at(cursor)) {
            cursor += open.len();
            continue;
        }
        let marker_begin = cursor;
        let scan = host.qwen_scan(&text[marker_begin..])?;
        let Some(syntax) = scan.call.filter(|_| scan.matched) else {
            cursor = marker_begin + scan.progress;
            if cursor == marker_begin {
                cursor += open.len();
            }
            continue;
        };
        cursor = marker_begin + scan.progress;
        let mut call = Call {
            name: utf8(&syntax.name)?,
            ..Call::default()
        };
        let schema = tools
            .iter()
            .find(|tool| tool.name == call.name)
            .and_then(|tool| tool.schema.as_ref());
        let object = schema.and_then(|schema| resolve(schema, schema));
        let properties = object.and_then(|object| object.get("properties"));
        let mut valid = true;
        let mut complete = false;
        // An implicit end of reasoning is only tentative while the call is
        // unfinished: a declared call under the native grammar may still
        // complete.
        let tentative = partial
            && syntax.canonical_header
            && schema.is_some()
            && !require_schema
            && !call_quotes.unclosed_at(marker_begin);
        if syntax.native {
            for parameter in &syntax.parameters {
                // Recovery resumes after the rejected parameter, not after
                // the call.
                let Some(argument) = interpret(schema, properties, parameter)? else {
                    cursor = marker_begin + parameter.progress;
                    valid = false;
                    break;
                };
                // Models sometimes repeat a parameter. An identical copy is
                // harmless; conflicting copies leave no safe choice.
                match call
                    .arguments
                    .iter()
                    .find(|previous| previous.name == argument.name)
                {
                    None => call.arguments.push(argument),
                    Some(previous)
                        if previous.value != argument.value
                            || previous.is_string != argument.is_string =>
                    {
                        cursor = marker_begin + parameter.progress;
                        valid = false;
                        break;
                    }
                    Some(_) => {}
                }
            }
            if valid && syntax.complete {
                complete = true;
            }
            // Arguments after a rejected parameter are never streamed.
            if !complete
                && tentative
                && valid
                && let Some(unfinished) = &syntax.unfinished
            {
                let name = parameter_name(properties, &unfinished.name)?;
                let property = properties.and_then(|properties| properties.get(&name));
                if unfinished.framed
                    && !name.is_empty()
                    && match (property, schema) {
                        (None, _) => true,
                        (Some(property), Some(root)) => string_only(root, property)?,
                        (Some(_), None) => false,
                    }
                {
                    partial_string(&mut call, &name, &utf8(&unfinished.value)?)?;
                }
            }
        } else if syntax.complete
            && let Some(Value::Object(parsed)) = try_parse_json(&syntax.json_payload)?
        {
            call.name = parsed
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            if !call.name.is_empty()
                && let Some(Value::Object(arguments)) = parsed.get("arguments")
            {
                for (name, value) in arguments.members() {
                    let (value, is_string) = match value {
                        Value::String(text) => (text.clone(), true),
                        other => (dump(other, Numbers::AsParsed)?, false),
                    };
                    call.arguments.push(Argument {
                        name: name.clone(),
                        value,
                        is_string,
                    });
                }
                complete = true;
            }
        }
        if complete && !tools.is_empty() && !tools.iter().any(|tool| tool.name == call.name) {
            complete = false;
        }
        if complete && (require_schema || call_quotes.unclosed_at(marker_begin)) {
            // An unfinished quotation is ambiguous. Restore the legacy
            // fallback only for a declared call whose complete arguments
            // satisfy its schema.
            complete = match tools.iter().position(|tool| tool.name == call.name) {
                None => false,
                Some(tool) => host.grammar_accepts(
                    tool,
                    arguments_json(&call.arguments, Numbers::Canonical)?.as_bytes(),
                )?,
            };
        }
        if complete || tentative {
            if !complete && call.partial_arguments.is_none() {
                call.partial_arguments = Some(prefix(&call)?);
            }
            calls.try_reserve(1)?;
            calls.push(call);
            if !complete {
                return Ok(());
            }
            cursor = marker_begin + scan.end;
            // Parameter bytes belong to the call, not to prose quotation
            // state. An unmatched backtick in a file write must not quote the
            // next call.
            call_quotes.reset(text, cursor);
            if let Some(spans) = spans.as_deref_mut() {
                spans.push((marker_begin, cursor));
            }
        }
    }
}
