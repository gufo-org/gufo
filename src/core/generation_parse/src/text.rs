//! Byte-string searches with the semantics of `std::string_view`: positions
//! are byte offsets and `NPOS` is "not found", so the ported comparisons
//! (`marker < end` with either missing) read as they do in C++.

/// `std::string_view::npos`.
pub const NPOS: usize = usize::MAX;

/// `std::isspace` in the C locale.
pub fn is_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\x0b' | b'\x0c' | b'\r')
}

/// `haystack.find(needle, from)`.
pub fn find(haystack: &[u8], needle: &[u8], from: usize) -> usize {
    if from > haystack.len() || needle.len() > haystack.len() - from {
        return NPOS;
    }
    let Some((&first, rest)) = needle.split_first() else {
        return from;
    };
    let last = haystack.len() - needle.len();
    let mut at = from;
    while at <= last {
        match haystack[at..=last].iter().position(|&byte| byte == first) {
            None => return NPOS,
            Some(offset) => {
                at += offset;
                if haystack[at + 1..at + needle.len()] == *rest {
                    return at;
                }
                at += 1;
            }
        }
    }
    NPOS
}

/// `haystack.find(byte, from)`.
pub fn find_byte(haystack: &[u8], byte: u8, from: usize) -> usize {
    if from >= haystack.len() {
        return NPOS;
    }
    haystack[from..]
        .iter()
        .position(|&candidate| candidate == byte)
        .map_or(NPOS, |offset| from + offset)
}

/// `haystack.rfind(needle)`.
pub fn rfind(haystack: &[u8], needle: &[u8]) -> usize {
    if needle.len() > haystack.len() {
        return NPOS;
    }
    (0..=haystack.len() - needle.len())
        .rev()
        .find(|&at| haystack[at..].starts_with(needle))
        .unwrap_or(NPOS)
}

/// `haystack.find_last_of(byte, before)`: the last `byte` at or before
/// `before`.
pub fn find_last_byte(haystack: &[u8], byte: u8, before: usize) -> usize {
    if haystack.is_empty() {
        return NPOS;
    }
    let end = before.min(haystack.len() - 1) + 1;
    haystack[..end]
        .iter()
        .rposition(|&candidate| candidate == byte)
        .unwrap_or(NPOS)
}

/// `value.find_first_not_of(set)`.
pub fn find_first_not_of(value: &[u8], set: &[u8]) -> usize {
    value
        .iter()
        .position(|byte| !set.contains(byte))
        .unwrap_or(NPOS)
}

/// `value.substr(from, count)`, clamped as `std::string_view::substr` is.
pub fn substr(value: &[u8], from: usize, count: usize) -> &[u8] {
    let from = from.min(value.len());
    let end = from.saturating_add(count).min(value.len());
    &value[from..end]
}

/// `value` without its trailing whitespace.
pub fn trim_trailing(value: &[u8]) -> &[u8] {
    let end = value
        .iter()
        .rposition(|&byte| !is_space(byte))
        .map_or(0, |last| last + 1);
    &value[..end]
}

pub fn trim(value: &[u8]) -> &[u8] {
    let begin = value
        .iter()
        .position(|&byte| !is_space(byte))
        .unwrap_or(value.len());
    trim_trailing(&value[begin..])
}

/// The offset of `part` inside `whole`; `part` must be a subslice of it.
pub fn offset_in(whole: &[u8], part: &[u8]) -> usize {
    (part.as_ptr() as usize).wrapping_sub(whole.as_ptr() as usize)
}

/// `bytes` as a string. Parsed text is decoded UTF-8 and every cut falls on
/// ASCII framing, so a failure is a host or parser bug.
pub fn utf8(bytes: &[u8]) -> Result<String, crate::config::Error> {
    let text = std::str::from_utf8(bytes).map_err(|_| crate::config::Error::Internal)?;
    let mut out = String::new();
    out.try_reserve(text.len())?;
    out.push_str(text);
    Ok(out)
}

/// `bytes` parsed as JSON, or `None` when they are not JSON.
pub fn try_parse_json(bytes: &[u8]) -> Result<Option<gufo_json::Value>, crate::config::Error> {
    match gufo_json::parse_value(bytes, gufo_json::Limits::NONE) {
        Ok(value) => Ok(Some(value)),
        Err(gufo_json::ValueError::NoMemory) => Err(crate::config::Error::NoMemory),
        Err(_) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn searches_match_string_view() {
        assert_eq!(find(b"abcabc", b"bc", 0), 1);
        assert_eq!(find(b"abcabc", b"bc", 2), 4);
        assert_eq!(find(b"abc", b"", 3), 3);
        assert_eq!(find(b"abc", b"", 4), NPOS);
        assert_eq!(find(b"abc", b"abcd", 0), NPOS);
        assert_eq!(find_byte(b"a<b<", b'<', 2), 3);
        assert_eq!(rfind(b"<a><a>", b"<a>"), 3);
        assert_eq!(rfind(b"", b""), 0);
        assert_eq!(find_last_byte(b"<a<b", b'<', 3), 2);
        assert_eq!(find_last_byte(b"<a<b", b'<', 1), 0);
        assert_eq!(find_first_not_of(b" \tx", b" \t"), 2);
        assert_eq!(trim(b" \x0b a \x0c"), b"a");
        assert_eq!(trim(b"   "), b"");
        assert_eq!(substr(b"abc", 5, NPOS), b"");
    }
}
