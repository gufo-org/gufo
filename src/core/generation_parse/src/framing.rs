//! Framing that is removed from content: closer echoes directly after a
//! parsed call, and a client envelope naming a declared tool. Other XML is
//! data.

use crate::config::{Envelope, Tool};
use crate::quote::QuoteTracker;
use crate::text::{NPOS, find, find_byte, find_last_byte, is_space, rfind, trim, trim_trailing};

/// The last of `tags` at or before `stop`, or NPOS when there is none or the
/// last one sits in quoted prose.
fn last_unquoted_tag(
    full: &[u8],
    stop: usize,
    floor: usize,
    tags: &[Vec<u8>],
    quotes: &QuoteTracker,
) -> usize {
    if stop <= floor {
        return NPOS;
    }
    // Search only what the caller can use: a tag starting before `floor` is
    // discarded there.
    let window = &full[floor..stop];
    let mut last = NPOS;
    for tag in tags {
        let found = rfind(window, tag);
        if found != NPOS && (last == NPOS || found > last) {
            last = found;
        }
    }
    if last == NPOS || quotes.quoted_at(floor + last) {
        return NPOS;
    }
    floor + last
}

fn declared(tag: &[u8], envelope: &Envelope, tools: &[Tool]) -> bool {
    let head = envelope.heads.first().map_or(0, Vec::len);
    let value = &tag[head.min(tag.len())..];
    let Some(&quote) = value.first() else {
        return false;
    };
    if quote != b'"' && quote != b'\'' {
        return false;
    }
    let end = find_byte(value, quote, 1);
    end != NPOS
        && tools
            .iter()
            .any(|tool| tool.name.as_bytes() == &value[1..end])
}

/// Find a closed or truncated client envelope whose request context
/// identifies a tool attempt. A bare parameter block is not enough evidence.
#[allow(clippy::too_many_arguments)]
fn block_start(
    full: &[u8],
    stop: usize,
    floor: usize,
    closers: &[Vec<u8>],
    envelope: &Envelope,
    tools: &[Tool],
    after_call: bool,
    quotes: &QuoteTracker,
) -> usize {
    let opening = last_unquoted_tag(full, stop, floor, &envelope.heads, quotes);
    if opening == NPOS {
        return NPOS;
    }
    let tag_end = find_byte(full, b'>', opening);
    if !after_call {
        if tag_end == NPOS || tag_end >= stop {
            return NPOS;
        }
        if !declared(&full[opening..tag_end], envelope, tools) {
            return NPOS;
        }
    }
    let region = trim_trailing(&full[opening..stop]);
    if envelope
        .closers
        .iter()
        .any(|closer| region.ends_with(closer))
    {
        return opening;
    }
    if !envelope
        .parameters
        .iter()
        .any(|parameter| find(region, parameter, 0) != NPOS)
    {
        return NPOS;
    }
    if closers.iter().any(|closer| find(region, closer, 0) != NPOS) {
        return NPOS;
    }
    opening
}

/// The content of `full[begin, end)`. With request context only: a closer
/// echo directly after a parsed call (`after_call`), or a client envelope
/// naming a declared tool, is removed. A dialect without closers removes
/// nothing.
#[allow(clippy::too_many_arguments)]
pub fn content_before<'a>(
    full: &'a [u8],
    begin: usize,
    end: usize,
    closers: &[Vec<u8>],
    envelope: &Envelope,
    quotes: &QuoteTracker,
    tools: &[Tool],
    after_call: bool,
) -> &'a [u8] {
    let stop = end.min(full.len());
    let mut begin = begin;
    if stop <= begin {
        return &[];
    }
    if closers.is_empty() {
        return &full[begin..stop];
    }
    if after_call {
        let mut cursor = begin;
        let mut removed = false;
        while cursor < stop {
            while cursor < stop && is_space(full[cursor]) {
                cursor += 1;
            }
            let tag = closers
                .iter()
                .find(|closer| full[cursor..stop].starts_with(closer));
            match tag {
                Some(tag) if !quotes.quoted_at(cursor) => {
                    cursor += tag.len();
                    removed = true;
                }
                _ => break,
            }
        }
        if removed {
            begin = cursor;
        }
    }
    let slice = &full[begin..stop];
    // The newline a model writes between or after its calls is layout, not a
    // reply: a turn of parallel calls would otherwise carry content "\n"
    // (#673).
    if after_call && trim(slice).is_empty() {
        return &[];
    }
    let adjacent = after_call
        && envelope
            .heads
            .first()
            .is_some_and(|head| trim(slice).starts_with(head));
    let block = block_start(
        full, stop, begin, closers, envelope, tools, adjacent, quotes,
    );
    if block != NPOS {
        return trim_trailing(&full[begin..block]);
    }
    slice
}

/// Hold only a possible client envelope. A declared-tool head must stay
/// pending until final parsing decides whether it is quoted documentation or
/// framing.
pub fn hold(
    full: &[u8],
    end: usize,
    floor: usize,
    closers: &[Vec<u8>],
    envelope: &Envelope,
    quotes: &QuoteTracker,
    tools: &[Tool],
) -> usize {
    if closers.is_empty() {
        return 0;
    }
    let mut held = 0;
    if end > 0 {
        // Hold only a suffix that can still become admitted framing. A bare
        // comparison such as "3 < 5" is already prose once the space arrives.
        let open = find_last_byte(full, b'<', end - 1);
        if open != NPOS && open >= floor && !quotes.quoted_at(open) {
            let suffix = &full[open..end];
            if envelope.heads.iter().any(|head| head.starts_with(suffix)) {
                held = suffix.len();
            }
        }
    }
    let opening = last_unquoted_tag(full, end, floor, &envelope.heads, quotes);
    if opening != NPOS {
        let tag_end = find_byte(full, b'>', opening);
        if tag_end == NPOS || tag_end >= end || declared(&full[opening..tag_end], envelope, tools) {
            held = held.max(end - opening);
        }
    }
    // The separator before framing is removed with it. Keep trailing
    // whitespace undecided until the next piece establishes whether it
    // introduces prose or a held tag, rather than streaming a separator we
    // cannot retract later.
    let ready = &full[floor..end - held];
    held += ready.len() - trim_trailing(ready).len();
    held
}
