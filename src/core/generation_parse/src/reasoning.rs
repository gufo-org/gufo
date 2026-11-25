//! The boundary after an explicit end of reasoning.

use crate::text::{NPOS, find_first_not_of};

/// `value` after the framing whitespace that follows an explicit reasoning
/// close. Only that boundary is framing; answer and argument interiors are
/// untouched.
pub fn after_separator<'a>(value: &'a [u8], separator: &[u8]) -> &'a [u8] {
    let first = find_first_not_of(value, separator);
    if first == NPOS { &[] } else { &value[first..] }
}

/// Whitespace the parsers skip before a reasoning opener.
pub const LEADING: &[u8] = b" \t\r\n";
