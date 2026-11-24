//! Unconstrained output (`ParseGeneration`): an optional reasoning phase,
//! then content with calls of the request's dialect.

use super::{Parsed, append_spans, copy, rebase, set_text};
use crate::config::{Config, Error, Host, Initial, ToolChoice, Unsatisfied};
use crate::framing::content_before;
use crate::markers::{earliest, held_prefix};
use crate::markup::{dialect, qwen};
use crate::quote::QuoteTracker;
use crate::reasoning::{LEADING, after_separator};
use crate::text::{NPOS, find, find_first_not_of, offset_in, substr, trim};

#[allow(clippy::too_many_arguments)]
pub fn parse(
    raw: &[u8],
    initial: Initial,
    config: &Config,
    enforce_required: bool,
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
    quotes.reset(raw, 0);
    let mut parsed = Parsed::default();
    let mut content = raw;
    let mut reasoning_call_context: &[u8] = &[];
    let mut unfinished_reasoning_quote = false;
    let recognize = config.recognize_tools();
    let tool_marker = |text: &[u8], quotes: &mut QuoteTracker| {
        quotes.reset(text, 0);
        if recognize {
            earliest(text, markers, quotes, 0)
        } else {
            NPOS
        }
    };

    match initial {
        Initial::Reasoning => {
            if content.starts_with(&reasoning.open) {
                content = &content[reasoning.open.len()..];
            }
            let mut think_end = find(content, &reasoning.close, 0);
            if tool_marker(content, quotes) < think_end {
                think_end = NPOS;
            }
            if think_end == NPOS {
                let marker = tool_marker(content, quotes);
                parsed.reasoning = copy(trim(substr(content, 0, marker)))?;
                if marker == NPOS {
                    if enforce_required && required {
                        return Err(Error::ToolChoiceUnsatisfied(Unsatisfied::NotProduced));
                    }
                    return Ok(parsed);
                }
                reasoning_call_context = content;
                unfinished_reasoning_quote = quotes.unclosed_at(marker);
                set_text(&mut parsed, &content[marker..], raw, track)?;
            } else {
                parsed.reasoning = copy(trim(&content[..think_end]))?;
                content = &content[think_end + reasoning.close.len()..];
                set_text(&mut parsed, after_separator(content, separator), raw, track)?;
            }
        }
        Initial::Auto => {
            // Only the initial phase has reasoning markup semantics. A literal
            // tag inside an argument (or quoted ordinary text) must never be
            // stripped.
            let leading = find_first_not_of(content, LEADING);
            let think_start = if !reasoning.open.is_empty()
                && leading != NPOS
                && content[leading..].starts_with(&reasoning.open)
            {
                leading
            } else {
                NPOS
            };
            if think_start != NPOS {
                let think_content_start = think_start + reasoning.open.len();
                let mut think_end = find(content, &reasoning.close, think_content_start);
                let marker = tool_marker(&content[think_content_start..], quotes);
                if marker != NPOS && think_content_start + marker < think_end {
                    think_end = NPOS;
                }
                if think_end != NPOS {
                    parsed.reasoning = copy(trim(&content[think_content_start..think_end]))?;
                    let remaining =
                        after_separator(&content[think_end + reasoning.close.len()..], separator);
                    if think_start > 0 {
                        set_text(&mut parsed, &content[..think_start], raw, track)?;
                        parsed.text.try_reserve(remaining.len())?;
                        parsed.text.extend_from_slice(remaining);
                        if track && !remaining.is_empty() {
                            parsed.spans.push(crate::events::Span {
                                begin: offset_in(raw, remaining),
                                size: remaining.len(),
                            });
                        }
                    } else {
                        set_text(&mut parsed, remaining, raw, track)?;
                    }
                } else {
                    let remaining = &content[think_content_start..];
                    let marker = tool_marker(remaining, quotes);
                    parsed.reasoning = copy(trim(substr(remaining, 0, marker)))?;
                    set_text(&mut parsed, &content[..think_start], raw, track)?;
                    if marker != NPOS {
                        reasoning_call_context = remaining;
                        unfinished_reasoning_quote = quotes.unclosed_at(marker);
                        parsed.text.try_reserve(remaining.len() - marker)?;
                        parsed.text.extend_from_slice(&remaining[marker..]);
                        if track {
                            parsed.spans.push(crate::events::Span {
                                begin: offset_in(raw, remaining) + marker,
                                size: remaining.len() - marker,
                            });
                        }
                    }
                }
            } else {
                set_text(&mut parsed, content, raw, track)?;
            }
        }
        Initial::Content | Initial::ContentAfterReasoning => {
            // After reasoning the prompt closed, the model writes the
            // separator.
            let text = if initial == Initial::ContentAfterReasoning {
                after_separator(content, separator)
            } else {
                content
            };
            set_text(&mut parsed, text, raw, track)?;
        }
    }

    quotes.reset(&parsed.text, 0);
    let marker = earliest(&parsed.text, markers, quotes, 0);
    if choice != ToolChoice::None && !tools.is_empty() && marker != NPOS {
        if partial && (unfinished_reasoning_quote || quotes.unclosed_at(marker)) {
            return Ok(parsed);
        }
        let mut spans = Vec::new();
        {
            let text_from_tools = &parsed.text[marker..];
            // The outer envelope selects the format, as in llama.cpp's model
            // parsers. Scanning both dialects would turn a literal call
            // inside an argument into an additional API invocation.
            if markup.qwen_calls && text_from_tools.starts_with(&markup.qwen_open) {
                qwen::parse_calls(
                    &parsed.text,
                    &markup.qwen_open,
                    tools,
                    &mut parsed.calls,
                    None,
                    unfinished_reasoning_quote,
                    partial,
                    host,
                )?;
            } else {
                dialect::parse_calls(
                    text_from_tools,
                    markup,
                    tools,
                    &mut parsed.calls,
                    &mut spans,
                    partial && !unfinished_reasoning_quote && !quotes.unclosed_at(marker),
                    host,
                )?;
            }
        }
        parsed
            .calls
            .retain(|call| tools.iter().any(|tool| tool.name == call.name));
        if parsed.calls.is_empty() && unfinished_reasoning_quote && !required {
            // The tentative implicit end of reasoning was an invalid quoted
            // example. Preserve it in its original phase, including a later
            // explicit boundary.
            let end = find(reasoning_call_context, &reasoning.close, 0);
            if end != NPOS {
                let after = end + reasoning.close.len();
                parsed = parse(
                    &reasoning_call_context[after..],
                    Initial::Content,
                    config,
                    enforce_required,
                    quotes,
                    partial,
                    track,
                    host,
                )?;
                rebase(&mut parsed, offset_in(raw, reasoning_call_context) + after);
            } else {
                parsed.text.clear();
                parsed.spans.clear();
            }
            parsed.reasoning = copy(substr(reasoning_call_context, 0, end))?;
            return Ok(parsed);
        }
        if parsed.calls.is_empty() && quotes.unclosed_at(marker) && !required {
            return Ok(parsed);
        }
        // An explicit stop can interrupt a call before its closing tags. Keep
        // complete calls, but do not expose an unfinished call as ordinary
        // text.
        if !parsed.calls.is_empty() || !enforce_required {
            // Text outside complete envelopes is content. Framing echoed
            // directly after a call is removed, as for Qwen calls (#383).
            let full = std::mem::take(&mut parsed.text);
            let old_spans = std::mem::take(&mut parsed.spans);
            let mut text = Vec::new();
            let mut append =
                |part: &[u8], spans: &mut Vec<crate::events::Span>| -> Result<(), Error> {
                    text.try_reserve(part.len())?;
                    text.extend_from_slice(part);
                    if !part.is_empty() {
                        append_spans(spans, &old_spans, offset_in(&full, part), part.len());
                    }
                    Ok(())
                };
            let quotes = &*quotes;
            append(
                content_before(&full, 0, marker, closers, envelope, quotes, tools, false),
                &mut parsed.spans,
            )?;
            let mut cursor = marker;
            for &(begin, end) in &spans {
                append(
                    content_before(
                        &full,
                        cursor,
                        marker + begin,
                        closers,
                        envelope,
                        quotes,
                        tools,
                        cursor > marker,
                    ),
                    &mut parsed.spans,
                )?;
                cursor = marker + end;
            }
            if !spans.is_empty() {
                let mut unfinished = earliest(&full, markers, quotes, cursor);
                if unfinished == NPOS && !enforce_required && required {
                    let held = held_prefix(&full, full.len(), markers, quotes);
                    if held > 0 {
                        unfinished = full.len() - held;
                    }
                }
                append(
                    content_before(
                        &full, cursor, unfinished, closers, envelope, quotes, tools, true,
                    ),
                    &mut parsed.spans,
                )?;
            }
            parsed.text = text;
            parsed.hide_tool_markup = true;
        }
    } else if recognize {
        // No marker: the text is content, but the framing it carries still
        // goes. The streamed response drops it as the turn ends, and both
        // transports must report the same content.
        let part = content_before(
            &parsed.text,
            0,
            parsed.text.len(),
            closers,
            envelope,
            quotes,
            tools,
            false,
        );
        let mut spans = Vec::new();
        if !part.is_empty() {
            append_spans(
                &mut spans,
                &parsed.spans,
                offset_in(&parsed.text, part),
                part.len(),
            );
        }
        let part = copy(part)?;
        parsed.text = part;
        parsed.spans = spans;
    }
    if enforce_required && required && parsed.calls.is_empty() {
        return Err(Error::ToolChoiceUnsatisfied(Unsatisfied::NotProduced));
    }
    Ok(parsed)
}
