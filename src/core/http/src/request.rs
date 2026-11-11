//! What a client's HTTP/1.x request says about itself: its head (RFC 9112,
//! parsed by httparse), how long its body is, its query parameters and the
//! token lists in its header fields.

/// A complete request head.
#[derive(Debug, PartialEq, Eq)]
pub struct Head<'a> {
    /// Bytes up to and including the empty line that ends the head.
    pub len: usize,
    pub method: &'a str,
    /// The target before its first `?`.
    pub path: &'a str,
    /// The target after its first `?`, without it.
    pub query: &'a str,
    pub http11: bool,
    pub headers: Vec<httparse::Header<'a>>,
    /// The body length from Content-Length, 0 without one; `None` when the
    /// framing is invalid: a Transfer-Encoding field (chunked bodies are not
    /// implemented), or Content-Length values that are not decimal digits,
    /// overflow or disagree.
    pub content_length: Option<u64>,
    /// An `Expect: 100-continue` field.
    pub expect_continue: bool,
}

/// The most header fields a head may have, as in cpp-httplib (llama-server).
pub const MAXIMUM_FIELDS: usize = 100;

#[derive(Debug, PartialEq, Eq)]
pub enum Parsed<'a> {
    /// No empty line has ended the head yet.
    Partial,
    /// Not an origin-form HTTP/1.0 or HTTP/1.1 request head.
    Malformed,
    /// More than `MAXIMUM_FIELDS` lines after the request line.
    TooManyFields,
    /// The field slots could not be allocated.
    NoMemory,
    Complete(Head<'a>),
}

/// Parses the head at the start of `buffer` once an empty line has ended
/// it. Bytes before `known_partial` were already parsed as `Partial`; the head
/// can only end in the bytes after them, so a buffer filled a little at a time
/// is searched once and parsed once, not once per read.
pub fn parse_head(buffer: &[u8], known_partial: usize) -> Parsed<'_> {
    // RFC 9112 2.2: one empty line before the request line is ignored. A
    // second one is rejected, so that empty lines cannot end every read.
    let start = match buffer {
        [] | [b'\r'] => return Parsed::Partial,
        [b'\r', b'\n', ..] => 2,
        [b'\n', ..] => 1,
        _ => 0,
    };
    if matches!(buffer.get(start), Some(b'\r' | b'\n')) {
        return Parsed::Malformed;
    }
    // An empty line ends the head: CRLF CRLF, or a bare LF on either line.
    let ends = |from: usize| {
        buffer[from.min(buffer.len())..]
            .windows(2)
            .position(|pair| pair == b"\n\n" || pair == b"\n\r")
            .map(|at| from + at)
    };
    if ends(known_partial.saturating_sub(3)).is_none() {
        return Parsed::Partial;
    }
    // A slot per line of the head: every line after the request line is a
    // field or makes the head malformed. Too many are rejected before any
    // allocation, and a failing allocation is reported, not aborted on.
    let Some(end) = ends(start) else {
        return Parsed::Partial;
    };
    let lines = buffer[start..=end]
        .iter()
        .filter(|&&byte| byte == b'\n')
        .count();
    if lines > MAXIMUM_FIELDS + 1 {
        return Parsed::TooManyFields;
    }
    let mut slots = Vec::new();
    if slots.try_reserve_exact(lines).is_err() {
        return Parsed::NoMemory;
    }
    slots.resize(lines, httparse::EMPTY_HEADER);
    let mut request = httparse::Request::new(&mut slots);
    let len = match request.parse(buffer) {
        Ok(httparse::Status::Complete(len)) => len,
        Ok(httparse::Status::Partial) => return Parsed::Partial,
        Err(_) => return Parsed::Malformed,
    };
    let (Some(method), Some(target), Some(version)) =
        (request.method, request.path, request.version)
    else {
        return Parsed::Malformed;
    };
    // Origin form only: the server has no proxy or OPTIONS * behavior.
    if !target.starts_with('/') {
        return Parsed::Malformed;
    }
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    // The slots become the field list, without a second allocation.
    let fields = request.headers.len();
    let mut headers = slots;
    headers.truncate(fields);
    Parsed::Complete(Head {
        len,
        method,
        path,
        query,
        http11: version == 1,
        content_length: content_length(&headers),
        expect_continue: headers.iter().any(|header| {
            header.name.eq_ignore_ascii_case("expect")
                && header.value.eq_ignore_ascii_case(b"100-continue")
        }),
        headers,
    })
}

fn content_length(headers: &[httparse::Header<'_>]) -> Option<u64> {
    let mut length = None;
    for httparse::Header { name, value } in headers {
        if name.eq_ignore_ascii_case("transfer-encoding") {
            return None;
        }
        if !name.eq_ignore_ascii_case("content-length") {
            continue;
        }
        if value.is_empty() {
            return None;
        }
        let parsed = value.iter().try_fold(0u64, |length, &digit| {
            if !digit.is_ascii_digit() {
                return None;
            }
            length.checked_mul(10)?.checked_add(u64::from(digit - b'0'))
        })?;
        if length.is_some_and(|length| length != parsed) {
            return None;
        }
        length = Some(parsed);
    }
    Some(length.unwrap_or(0))
}

/// `+` as a space and `%XX` as a byte; a `%` without two hex digits stays.
/// `None` when the memory for it cannot be allocated.
pub fn url_decode(text: &[u8]) -> Option<Vec<u8>> {
    let hex = |byte: u8| (byte as char).to_digit(16).map(|digit| digit as u8);
    let mut out = Vec::new();
    out.try_reserve_exact(text.len()).ok()?;
    let mut index = 0;
    while index < text.len() {
        match text[index] {
            b'+' => out.push(b' '),
            b'%' if index + 2 < text.len() => match (hex(text[index + 1]), hex(text[index + 2])) {
                (Some(high), Some(low)) => {
                    out.push(high << 4 | low);
                    index += 2;
                }
                _ => out.push(b'%'),
            },
            byte => out.push(byte),
        }
        index += 1;
    }
    Some(out)
}

/// What `query_param` found.
#[derive(Debug, PartialEq, Eq)]
pub enum Lookup {
    /// The decoded value: empty for a name without `=`.
    Found(Vec<u8>),
    Missing,
    /// The memory for decoding could not be allocated.
    NoMemory,
}

/// The first `&`-separated parameter whose decoded name is `key`. An empty
/// query and the empty end after a final `&` hold no parameter; an empty
/// parameter between two `&` has an empty name.
pub fn query_param(query: &[u8], key: &[u8]) -> Lookup {
    let mut remaining = query;
    while !remaining.is_empty() {
        let (parameter, rest) = match remaining.iter().position(|&byte| byte == b'&') {
            Some(end) => (&remaining[..end], &remaining[end + 1..]),
            None => (remaining, &[][..]),
        };
        let (name, value) = match parameter.iter().position(|&byte| byte == b'=') {
            Some(equals) => (&parameter[..equals], &parameter[equals + 1..]),
            None => (parameter, &[][..]),
        };
        let Some(name) = url_decode(name) else {
            return Lookup::NoMemory;
        };
        if name == key {
            return url_decode(value).map_or(Lookup::NoMemory, Lookup::Found);
        }
        remaining = rest;
    }
    Lookup::Missing
}

/// Whether the comma-separated `value` lists `token` (lowercase), ignoring
/// ASCII case and the spaces and tabs around each item.
pub fn has_token(value: &[u8], token: &[u8]) -> bool {
    let blank = |byte: &u8| *byte == b' ' || *byte == b'\t';
    value.split(|&byte| byte == b',').any(|item| {
        let start = item
            .iter()
            .position(|byte| !blank(byte))
            .unwrap_or(item.len());
        let end = item
            .iter()
            .rposition(|byte| !blank(byte))
            .map_or(start, |end| end + 1);
        let item = &item[start..end];
        item.len() == token.len()
            && item
                .iter()
                .zip(token)
                .all(|(a, b)| a.to_ascii_lowercase() == *b)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn head(text: &[u8]) -> Head<'_> {
        match parse_head(text, 0) {
            Parsed::Complete(head) => head,
            other => panic!("{:?}: {other:?}", String::from_utf8_lossy(text)),
        }
    }

    #[test]
    fn heads_and_framing() {
        let text = b"POST /v1/chat?stream=true HTTP/1.1\r\nHost: a\r\nContent-Length: 5\r\nExpect: 100-Continue\r\n\r\nhello";
        let parsed = head(text);
        assert_eq!(parsed.len, text.len() - 5);
        assert_eq!(
            (parsed.method, parsed.path, parsed.query),
            ("POST", "/v1/chat", "stream=true")
        );
        assert!(parsed.http11 && parsed.expect_continue);
        assert_eq!(parsed.content_length, Some(5));
        let field = |header: &httparse::Header<'_>| (header.name.to_owned(), header.value.to_vec());
        assert_eq!(field(&parsed.headers[0]), ("Host".into(), b"a".to_vec()));
        // RFC 9112 2.2: a bare LF ends a line, and an empty line before the
        // request line is ignored.
        let fields = head(b"\r\nGET / HTTP/1.0\nX: 1\n\n").headers;
        assert_eq!(
            fields.iter().map(field).collect::<Vec<_>>(),
            [("X".into(), b"1".to_vec())]
        );
        assert!(!head(b"GET / HTTP/1.0\r\n\r\n").http11);
        for framing in [
            &b"Transfer-Encoding: chunked"[..],
            b"Content-Length: +5",
            b"Content-Length: 5\r\nContent-Length: 6",
            b"Content-Length: 99999999999999999999",
            b"Content-Length: 0x5",
        ] {
            let text = [&b"POST / HTTP/1.1\r\n"[..], framing, b"\r\n\r\n"].concat();
            assert_eq!(head(&text).content_length, None, "{framing:?}");
        }
        assert_eq!(
            head(b"POST / HTTP/1.1\r\nContent-Length: 05\r\ncontent-length: 5\r\n\r\n")
                .content_length,
            Some(5)
        );
    }

    #[test]
    fn rejections_and_partial_heads() {
        for text in [
            &b"GET  / HTTP/1.1\r\n\r\n"[..],
            b"GET\t/ HTTP/1.1\r\n\r\n",
            b"G@T / HTTP/1.1\r\n\r\n",
            b"GET /a\x01b HTTP/1.1\r\n\r\n",
            b"GET / HTTP/2\r\n\r\n",
            b"GET http://h/ HTTP/1.1\r\n\r\n",
            b"GET / HTTP/1.1\r\nX : v\r\n\r\n",
            b"GET / HTTP/1.1\r\nX: a\r\n b\r\n\r\n",
            b"GET / HTTP/1.1\r\nX: a\x00b\r\n\r\n",
            b"\r\n\r\nGET / HTTP/1.1\r\n\r\n",
            b"\n\n",
        ] {
            assert_eq!(
                parse_head(text, 0),
                Parsed::Malformed,
                "{:?}",
                String::from_utf8_lossy(text)
            );
        }
        let text = b"GET / HTTP/1.1\r\nHost: a\r\n\r\n";
        for len in 0..text.len() {
            assert_eq!(parse_head(&text[..len], 0), Parsed::Partial, "{len}");
        }
        // A head is parsed only once it has ended, and bytes known to be
        // partial are not searched again.
        assert_eq!(
            parse_head(b"GET\t/ HTTP/1.1\r\nHost: a\r\n", 0),
            Parsed::Partial
        );
        assert_eq!(
            parse_head(b"GET / HTTP/1.1\r\nHost: a\r\n", 10),
            Parsed::Partial
        );
        assert!(matches!(
            parse_head(text, text.len() - 2),
            Parsed::Complete(_)
        ));
    }

    #[test]
    fn field_limit() {
        let with_fields = |count: usize, line_end: &str| {
            let mut text = format!("\r\nGET / HTTP/1.1{line_end}");
            for index in 0..count {
                text += &format!("X-{index}: v{line_end}");
            }
            text + line_end
        };
        assert_eq!(
            head(with_fields(MAXIMUM_FIELDS, "\r\n").as_bytes())
                .headers
                .len(),
            MAXIMUM_FIELDS
        );
        assert_eq!(
            head(with_fields(MAXIMUM_FIELDS, "\n").as_bytes())
                .headers
                .len(),
            MAXIMUM_FIELDS
        );
        for line_end in ["\r\n", "\n"] {
            let text = with_fields(MAXIMUM_FIELDS + 1, line_end);
            assert_eq!(parse_head(text.as_bytes(), 0), Parsed::TooManyFields);
        }
        // A line that is no field still counts.
        let text = with_fields(MAXIMUM_FIELDS, "\r\n").replacen("\r\n\r\n", "\r\n x\r\n\r\n", 1);
        assert_eq!(parse_head(text.as_bytes(), 0), Parsed::TooManyFields);
    }

    #[test]
    fn query_parameters_and_tokens() {
        assert_eq!(
            url_decode(b"a+b%41%4a%zz%4").as_deref(),
            Some(&b"a bAJ%zz%4"[..])
        );
        let found = |value: &[u8]| Lookup::Found(value.to_vec());
        assert_eq!(query_param(b"model=x%20y&b", b"model"), found(b"x y"));
        assert_eq!(query_param(b"a&model", b"model"), found(b""));
        assert_eq!(query_param(b"m%6fdel=1", b"model"), found(b"1"));
        assert_eq!(query_param(b"x=1", b"model"), Lookup::Missing);
        // As in 0.33.0: no parameter in an empty query or after a final `&`,
        // an empty one between two `&`, and an empty name before `=`.
        assert_eq!(query_param(b"", b""), Lookup::Missing);
        assert_eq!(query_param(b"a&", b""), Lookup::Missing);
        assert_eq!(query_param(b"a&&b", b""), found(b""));
        assert_eq!(query_param(b"=x", b""), found(b"x"));
        assert!(has_token(b"keep-alive, Upgrade", b"upgrade"));
        assert!(has_token(b" \tWebSocket\t ", b"websocket"));
        assert!(!has_token(b"websockets", b"websocket"));
        assert!(!has_token(b"\x0bwebsocket", b"websocket"));
    }
}
