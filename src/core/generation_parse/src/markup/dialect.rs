//! Calls of a registered dialect, read by its scanner and checked against
//! the declared schemas.

use crate::args::json_out::{partial_string, prefix};
use crate::args::schema::{accepts, resolve};
use crate::args::{Argument, Call};
use crate::config::{Error, Host, Markup, ScannedArgument, Tool};
use crate::text::{trim, try_parse_json, utf8};
use gufo_json::Value;

/// Counts the argument prefix accepted by the advertised schema. A
/// `string="false"` value that fails its declared type rejects the call, as
/// for Qwen. Streaming keeps only the arguments before that rejection;
/// strings and unknown parameters stay as the dialect wrote them.
fn accepted_count(
    tools: &[Tool],
    name: &[u8],
    arguments: &[ScannedArgument],
) -> Result<usize, Error> {
    let Some(schema) = tools
        .iter()
        .find(|tool| tool.name.as_bytes() == name)
        .and_then(|tool| tool.schema.as_ref())
    else {
        return Ok(arguments.len());
    };
    let Some(properties @ Value::Object(_)) =
        resolve(schema, schema).and_then(|object| object.get("properties"))
    else {
        return Ok(arguments.len());
    };
    for (index, argument) in arguments.iter().enumerate() {
        if argument.is_string {
            continue;
        }
        let Some(property) = std::str::from_utf8(&argument.name)
            .ok()
            .and_then(|name| properties.get(name))
        else {
            continue;
        };
        match try_parse_json(trim(&argument.value))? {
            Some(parsed) if accepts(schema, property, &parsed)? => {}
            _ => return Ok(index),
        }
    }
    Ok(arguments.len())
}

fn convert(arguments: &[ScannedArgument]) -> Result<Vec<Argument>, Error> {
    let mut converted = Vec::new();
    converted.try_reserve(arguments.len())?;
    for argument in arguments {
        converted.push(Argument {
            name: utf8(&argument.name)?,
            value: utf8(&argument.value)?,
            is_string: argument.is_string,
        });
    }
    Ok(converted)
}

/// The calls of `text`, which starts at the first marker, appended to
/// `calls`, and their byte ranges to `spans`. With `partial`, the one
/// unfinished call ends the list.
pub fn parse_calls(
    text: &[u8],
    markup: &Markup,
    tools: &[Tool],
    calls: &mut Vec<Call>,
    spans: &mut Vec<(usize, usize)>,
    partial: bool,
    host: &mut dyn Host,
) -> Result<(), Error> {
    if !markup.dialect {
        return Ok(());
    }
    // The Qwen opener is not an envelope for this dialect even when the JSON
    // fallback grammar admits both.
    let markers: Vec<&[u8]> = markup
        .markers
        .iter()
        .filter(|marker| !markup.qwen_calls || **marker != markup.qwen_open)
        .map(Vec::as_slice)
        .collect();
    let scanned = host.dialect_scan(text, &markers, partial)?;
    spans.try_reserve(scanned.spans.len())?;
    spans.extend_from_slice(&scanned.spans);
    for invocation in scanned.calls {
        let accepted = accepted_count(tools, &invocation.name, &invocation.arguments)?;
        let arguments = &invocation.arguments[..accepted];
        let schema_valid = accepted == invocation.arguments.len();
        let valid = invocation.valid && schema_valid;
        if valid && invocation.complete {
            calls.try_reserve(1)?;
            calls.push(Call {
                name: utf8(&invocation.name)?,
                arguments: convert(arguments)?,
                partial_arguments: None,
            });
            continue;
        }
        if !partial || invocation.name.is_empty() {
            continue;
        }
        let mut call = Call {
            name: utf8(&invocation.name)?,
            arguments: convert(arguments)?,
            partial_arguments: None,
        };
        if let Some(unfinished) = &invocation.unfinished
            && schema_valid
        {
            partial_string(
                &mut call,
                &utf8(&unfinished.name)?,
                &utf8(&unfinished.value)?,
            )?;
        }
        if call.partial_arguments.is_none() {
            call.partial_arguments = Some(prefix(&call)?);
        }
        calls.try_reserve(1)?;
        calls.push(call);
        break;
    }
    Ok(())
}
