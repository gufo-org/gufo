//! Streamed tool-call deltas (`StreamingToolCalls`). Like llama.cpp's
//! task_result_state, calls are kept by index and only the newly parsed
//! argument suffix is published. The final parse remains authoritative.

use crate::args::Call;
use crate::args::json_out::arguments_json;
use crate::config::Error;
use crate::events::{Event, Sink};
use gufo_json::Numbers;

#[derive(Debug)]
struct Sent {
    name: String,
    arguments: String,
    complete: bool,
}

#[derive(Debug, Default)]
pub struct ToolDeltas {
    sent: Vec<Sent>,
}

impl ToolDeltas {
    pub fn update(&mut self, calls: &[Call], sink: &mut dyn Sink) -> Result<(), Error> {
        for (index, call) in calls.iter().enumerate() {
            let arguments = match &call.partial_arguments {
                Some(partial) => partial.clone(),
                None => arguments_json(&call.arguments, Numbers::AsParsed)?,
            };
            let added = index == self.sent.len();
            if added {
                self.sent.try_reserve(1)?;
                self.sent.push(Sent {
                    name: call.name.clone(),
                    arguments: String::new(),
                    complete: false,
                });
            }
            let previous = &mut self.sent[index];
            if previous.name != call.name || !arguments.starts_with(&previous.arguments) {
                // A still-unfinished quotation or value can make a partial
                // parse temporarily disappear. Only the final parse may
                // accept the turn.
                continue;
            }
            let complete = call.partial_arguments.is_none();
            let delta = &arguments[previous.arguments.len()..];
            if (added || !delta.is_empty() || (complete && !previous.complete))
                && !sink.event(Event::ToolDelta {
                    index,
                    name: if added { previous.name.as_bytes() } else { &[] },
                    arguments: delta.as_bytes(),
                    begin: added,
                    complete,
                })
            {
                return Err(Error::Stopped);
            }
            previous.arguments = arguments;
            previous.complete = complete;
        }
        Ok(())
    }

    /// Publishes the final calls' last deltas, and fails when a streamed
    /// call is not what the final parse accepted.
    pub fn finish(
        &mut self,
        calls: &[Call],
        interrupted: bool,
        sink: &mut dyn Sink,
    ) -> Result<(), Error> {
        self.update(calls, sink)?;
        for (index, sent) in self.sent.iter().enumerate() {
            match calls.get(index) {
                None if interrupted && !sent.complete => continue,
                Some(call)
                    if sent.complete
                        && sent.name == call.name
                        && sent.arguments
                            == arguments_json(&call.arguments, Numbers::AsParsed)? =>
                {
                    continue;
                }
                _ => return Err(Error::UnsafeStream),
            }
        }
        Ok(())
    }
}
