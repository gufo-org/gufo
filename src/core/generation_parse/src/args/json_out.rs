//! Arguments as JSON text, on the shared writer.

use super::{Argument, Call};
use crate::config::Error;
use gufo_json::{Limits, Numbers, Object, Value, ValueError, dump, parse_value};

/// The arguments object. Numbers keep the spelling the model generated, as
/// llama.cpp returns them (`Numbers::AsParsed`): above 2^53, and 1.0 or 1e5,
/// a double would change the client's arguments and the replayed turn.
/// Schema checks pass `Numbers::Canonical`, which the native number grammar
/// reads. A repeated name replaces the earlier value in place.
pub fn arguments_json(arguments: &[Argument], numbers: Numbers) -> Result<String, Error> {
    let mut members: Vec<(String, Value)> = Vec::new();
    members.try_reserve(arguments.len())?;
    for argument in arguments {
        let value = if argument.is_string {
            Value::String(argument.value.clone())
        } else {
            match parse_value(argument.value.as_bytes(), Limits::NONE) {
                Ok(value) => value,
                Err(ValueError::NoMemory) => return Err(Error::NoMemory),
                Err(_) => Value::String(argument.value.clone()),
            }
        };
        match members.iter_mut().find(|(name, _)| *name == argument.name) {
            Some((_, slot)) => *slot = value,
            None => members.push((argument.name.clone(), value)),
        }
    }
    let mut object = Object::new();
    for (name, value) in members {
        object.push(name, value)?;
    }
    Ok(dump(&Value::Object(object), numbers)?)
}

/// The complete arguments without their closing brace.
pub fn prefix(call: &Call) -> Result<String, Error> {
    let mut prefix = arguments_json(&call.arguments, Numbers::AsParsed)?;
    prefix.pop();
    Ok(prefix)
}

/// The complete arguments followed by the still-open string `value` of
/// `name`; identical repeated parameters are discarded by the full parser,
/// so a name already present adds nothing.
pub fn partial_string(call: &mut Call, name: &str, value: &str) -> Result<(), Error> {
    if call.arguments.iter().any(|argument| argument.name == name) {
        return Ok(());
    }
    let mut out = prefix(call)?;
    if !call.arguments.is_empty() {
        out.push(',');
    }
    out.push_str(&dump(&Value::String(name.to_owned()), Numbers::Canonical)?);
    out.push(':');
    let mut quoted = dump(&Value::String(value.to_owned()), Numbers::Canonical)?;
    quoted.pop();
    out.push_str(&quoted);
    call.partial_arguments = Some(out);
    Ok(())
}
