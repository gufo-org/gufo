//! The whole-response parsers: plain output and output under a response
//! format or the tool grammar.

pub mod plain;
pub mod structured;

use crate::args::Call;
use crate::args::json_out::arguments_json;
use crate::config::{Config, Error, Host, Mode};
use crate::events::{Event, Sink, Span};
use crate::quote::QuoteTracker;
use crate::text::offset_in;
use gufo_json::Numbers;

/// A parsed response (`ParsedGeneration`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Parsed {
    pub text: Vec<u8>,
    pub reasoning: Vec<u8>,
    pub calls: Vec<Call>,
    pub hide_tool_markup: bool,
    /// Where `text` came from, when tracked.
    pub spans: Vec<Span>,
}

/// Appends the part of `source` that covers `[offset, offset + size)` of the
/// text it maps.
pub fn append_spans(output: &mut Vec<Span>, source: &[Span], mut offset: usize, mut size: usize) {
    for span in source {
        if offset >= span.size {
            offset -= span.size;
            continue;
        }
        let count = size.min(span.size - offset);
        if count > 0 {
            output.push(Span {
                begin: span.begin + offset,
                size: count,
            });
        }
        size -= count;
        if size == 0 {
            break;
        }
        offset = 0;
    }
}

/// Sets the content to `text`, a part of `source`.
pub(crate) fn set_text(
    parsed: &mut Parsed,
    text: &[u8],
    source: &[u8],
    track: bool,
) -> Result<(), Error> {
    parsed.text.clear();
    parsed.text.try_reserve(text.len())?;
    parsed.text.extend_from_slice(text);
    parsed.spans.clear();
    if track && !text.is_empty() {
        parsed.spans.push(Span {
            begin: offset_in(source, text),
            size: text.len(),
        });
    }
    Ok(())
}

pub(crate) fn rebase(parsed: &mut Parsed, offset: usize) {
    for span in &mut parsed.spans {
        span.begin += offset;
    }
}

pub(crate) fn copy(bytes: &[u8]) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    out.try_reserve(bytes.len())?;
    out.extend_from_slice(bytes);
    Ok(out)
}

/// Parses the response `raw` (decoded UTF-8) as the mode of `config` does
/// (`StreamingGeneration`). `partial` admits an unfinished trailing call
/// when the dialect is native.
pub fn parse(
    raw: &[u8],
    config: &Config,
    enforce_required: bool,
    partial: bool,
    track: bool,
    quotes: &mut QuoteTracker,
    host: &mut dyn Host,
) -> Result<Parsed, Error> {
    // Native grammars establish argument framing before the value completes.
    // Recovery from unconstrained/JSON fallback text still requires a full
    // call.
    let partial = partial && config.markup.native;
    match config.mode {
        Mode::Plain => plain::parse(
            raw,
            config.initial,
            config,
            enforce_required,
            quotes,
            partial,
            track,
            host,
        ),
        Mode::Structured | Mode::ToolOnly => structured::parse(
            raw,
            config.initial,
            config,
            enforce_required,
            config.mode == Mode::ToolOnly,
            quotes,
            partial,
            track,
            host,
        ),
    }
}

/// Reports `parsed` as final events.
pub fn emit_final(parsed: &Parsed, sink: &mut dyn Sink) -> Result<(), Error> {
    let stop = |accepted: bool| {
        if accepted {
            Ok(())
        } else {
            Err(Error::Stopped)
        }
    };
    stop(sink.event(Event::FinalReasoning {
        text: &parsed.reasoning,
    }))?;
    stop(sink.event(Event::FinalContent {
        text: &parsed.text,
        spans: &parsed.spans,
    }))?;
    for (index, call) in parsed.calls.iter().enumerate() {
        let arguments = arguments_json(&call.arguments, Numbers::AsParsed)?;
        stop(sink.event(Event::FinalCall {
            index,
            name: call.name.as_bytes(),
            arguments: arguments.as_bytes(),
        }))?;
    }
    stop(sink.event(Event::FinalEnd {
        hide_tool_markup: parsed.hide_tool_markup,
    }))
}
