//! Where call markers open, and how much of a streamed tail might still
//! become one.

use crate::quote::QuoteTracker;
use crate::text::{NPOS, find_byte};

pub type Markers = [Vec<u8>];

/// Earliest call marker at or after `from` that is not quoted prose. A marker
/// the model writes inside a backtick span or a code fence is the model
/// *explaining* the format; an answer that names its tags must not be parsed
/// or cut at them (#383). `full` is the whole response so the surrounding
/// lines are visible.
pub fn earliest(full: &[u8], markers: &Markers, quotes: &QuoteTracker, from: usize) -> usize {
    let mut position = from;
    while position < full.len() {
        let next = find_byte(full, b'<', position);
        if next == NPOS {
            break;
        }
        // The cheap question first: only a position that carries a marker is
        // worth asking about quoting, which keeps a line full of '<' linear.
        for marker in markers {
            let tag_offset = find_byte(marker, b'<', 0);
            if tag_offset != NPOS
                && next - from >= tag_offset
                && full[next - tag_offset..].starts_with(marker)
                && !quotes.quoted_at(next)
            {
                return next - tag_offset;
            }
        }
        position = next + 1;
    }
    NPOS
}

/// The earliest marker outside quoted prose and outside an unfinished
/// quotation, as a constrained reasoning phase reads it.
pub fn reasoning_tool(full: &[u8], markers: &Markers, quotes: &QuoteTracker, from: usize) -> usize {
    let mut found = earliest(full, markers, quotes, from);
    while found != NPOS && quotes.unclosed_at(found) {
        found = earliest(full, markers, quotes, found + 1);
    }
    found
}

/// Bytes at the end of [0, end) that could still open a call marker. A
/// marker that would open inside quoted prose is prose, so it is not held.
pub fn held_prefix(full: &[u8], end: usize, markers: &Markers, quotes: &QuoteTracker) -> usize {
    let maximum_marker = markers.iter().map(Vec::len).max().unwrap_or(0);
    let maximum = end.min(maximum_marker.saturating_sub(1));
    for length in (1..=maximum).rev() {
        let suffix = &full[end - length..end];
        if markers.iter().any(|marker| marker.starts_with(suffix))
            && !quotes.quoted_at(end - length)
        {
            return length;
        }
    }
    0
}

/// `text` is a prefix of some marker (an empty text is a prefix of any).
pub fn is_prefix(text: &[u8], markers: &Markers) -> bool {
    markers.iter().any(|marker| marker.starts_with(text))
}
