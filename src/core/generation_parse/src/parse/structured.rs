//! Output under a response format or the tool grammar
//! (`ParseStructuredGeneration`). Markers in JSON strings are data; only the
//! initial reasoning phase has markup semantics.

use super::{Parsed, copy, rebase, set_text};
use crate::config::{Config, Error, Host, Initial, ToolChoice, Unsatisfied};
use crate::events::Span;
use crate::framing::content_before;
use crate::markers::{earliest, held_prefix, is_prefix, reasoning_tool};
use crate::markup::qwen;
use crate::quote::QuoteTracker;
use crate::reasoning::{LEADING, after_separator};
use crate::text::{NPOS, find, find_first_not_of, offset_in, substr, trim};

#[allow(clippy::too_many_arguments)]
pub fn parse(
    raw: &[u8],
    initial: Initial,
    config: &Config,
    enforce_required: bool,
    tool_only: bool,
    quotes: &mut QuoteTracker,
    partial: bool,
    track: bool,
    host: &mut dyn Host,
) -> Result<Parsed, Error> {
    let markup = &config.markup;
    let markers = &markup.markers;
    let closers = &markup.closers;
    let envelope = &markup.envelope;
    let reasoning = &config.reasoning;
    let separator = &markup.reasoning_separator;
    let tools = &config.tools;
    let choice = config.choice;
    let required = choice == ToolChoice::Required;
    let mut parsed = Parsed::default();
    let source = raw;
    let mut raw = raw;
    let mut initial = initial;
    if initial == Initial::Auto {
        let leading = find_first_not_of(raw, LEADING);
        if !reasoning.open.is_empty()
            && leading != NPOS
            && raw[leading..].starts_with(&reasoning.open)
        {
            raw = &raw[leading..];
            initial = Initial::Reasoning;
        }
    }
    if initial == Initial::Reasoning {
        if raw.starts_with(&reasoning.open) {
            raw = &raw[reasoning.open.len()..];
        }
        let end = find(raw, &reasoning.close, 0);
        quotes.reset(raw, 0);
        let call = if config.recognize_tools() {
            reasoning_tool(raw, &markup.reasoning_markers, quotes, 0)
        } else {
            NPOS
        };
        if call < end {
            if partial && quotes.unclosed_at(call) {
                return Ok(parsed);
            }
            parsed.reasoning = copy(&raw[..call])?;
            raw = &raw[call..];
        } else {
            parsed.reasoning = copy(substr(raw, 0, end))?;
            if end == NPOS {
                return Ok(parsed);
            }
            raw = after_separator(&raw[end + reasoning.close.len()..], separator);
        }
    } else if initial == Initial::ContentAfterReasoning {
        raw = after_separator(raw, separator);
    }
    let trimmed = trim(raw);
    let held_opener = required && trimmed.is_empty() && !raw.is_empty() && is_prefix(raw, markers);
    let content = if tool_only || held_opener {
        raw
    } else {
        trimmed
    };
    quotes.reset(content, 0);
    let first = earliest(content, markers, quotes, 0);
    if !tools.is_empty()
        && choice != ToolChoice::None
        && (tool_only || first == 0)
        && first != NPOS
        && !(markup.qwen_calls && content[first..].starts_with(&markup.qwen_open))
    {
        let mut native = super::plain::parse(
            content,
            Initial::Content,
            config,
            enforce_required,
            quotes,
            partial,
            track,
            host,
        )?;
        rebase(&mut native, offset_in(source, content));
        native.reasoning = parsed.reasoning;
        return Ok(native);
    }
    let tool_start = &markup.qwen_open;
    let mut marker = if tool_only {
        earliest(content, markers, quotes, 0)
    } else {
        0
    };
    if tool_only && required && marker == NPOS {
        let held = held_prefix(content, content.len(), markers, quotes);
        if held > 0 {
            marker = content.len() - held;
        }
    }
    if !tools.is_empty()
        && choice != ToolChoice::None
        && !content.is_empty()
        && marker != NPOS
        && (tool_only
            || (markup.qwen_calls && content.starts_with(tool_start))
            || is_prefix(content, markers))
    {
        let mut call = Parsed::default();
        let mut spans = Vec::new();
        qwen::parse_calls(
            content,
            tool_start,
            tools,
            &mut call.calls,
            if tool_only { Some(&mut spans) } else { None },
            false,
            partial,
            host,
        )?;
        if tool_only {
            let quotes = &*quotes;
            let mut append = |part: &[u8]| -> Result<(), Error> {
                call.text.try_reserve(part.len())?;
                call.text.extend_from_slice(part);
                if track && !part.is_empty() {
                    call.spans.push(Span {
                        begin: offset_in(source, part),
                        size: part.len(),
                    });
                }
                Ok(())
            };
            let mut cursor = 0;
            for &(begin, end) in &spans {
                append(content_before(
                    content,
                    cursor,
                    begin,
                    closers,
                    envelope,
                    quotes,
                    tools,
                    cursor > 0,
                ))?;
                cursor = end;
            }
            let tail = &content[cursor..];
            let mut unfinished = earliest(content, markers, quotes, cursor);
            if unfinished != NPOS {
                unfinished -= cursor;
            } else if !enforce_required && required {
                let held = held_prefix(content, content.len(), markers, quotes);
                if held > 0 {
                    // Wraps as C++ size_t does when the held bytes start
                    // before `cursor`; the stop then falls before it.
                    unfinished = tail.len().wrapping_sub(held);
                }
            }
            let limit = if spans.is_empty() { marker } else { unfinished };
            let stop = if limit == NPOS {
                content.len()
            } else {
                cursor.wrapping_add(limit).min(content.len())
            };
            // The model repeats the framing of the call it just made; that
            // echo is markup, not prose the client should replay on the next
            // turn (#383).
            append(content_before(
                content,
                cursor,
                stop,
                closers,
                envelope,
                quotes,
                tools,
                cursor > 0,
            ))?;
        }
        if call.calls.is_empty() && marker != NPOS && quotes.unclosed_at(marker) && !required {
            set_text(&mut parsed, raw, source, track)?;
            return Ok(parsed);
        }
        call.hide_tool_markup = true;
        if call.calls.is_empty() && enforce_required && required {
            return Err(Error::ToolChoiceUnsatisfied(Unsatisfied::NotCompleted));
        }
        call.reasoning = parsed.reasoning;
        return Ok(call);
    }
    if enforce_required && required {
        return Err(Error::ToolChoiceUnsatisfied(Unsatisfied::NotCompleted));
    }
    // A constrained turn that produced no call is still reported without
    // framing: the streamed response drops it as the turn ends, and both
    // transports must report the same content.
    let text = if tool_only {
        content_before(raw, 0, raw.len(), closers, envelope, quotes, tools, false)
    } else {
        raw
    };
    set_text(&mut parsed, text, source, track)?;
    Ok(parsed)
}
