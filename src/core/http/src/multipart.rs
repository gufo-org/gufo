//! `multipart/form-data` bodies (RFC 7578) for the ASR, image-edit and video
//! APIs, in one parser: RFC 2046 framing, part headers as RFC 9112 fields
//! (through httparse) and RFC 9110 parameters. Each API passes its field
//! limit and whether field names must be unique; what a field means stays
//! with the API.

use std::borrow::Cow;

/// RFC 2046 5.1.1.
const MAXIMUM_BOUNDARY: usize = 70;
/// A part's header lines, without the empty line that ends them.
pub const MAXIMUM_HEADER_BYTES: usize = 16 << 10;
/// Header fields per part; RFC 7578 defines three.
const MAXIMUM_PART_HEADERS: usize = 16;
/// Parameters in a Content-Type or Content-Disposition; RFC 7578 uses up to
/// three. It keeps the search for repeated names linear in the field's length.
const MAXIMUM_PARAMETERS: usize = 16;

const NO_OPENING: &str = "multipart body has no opening boundary";
const NO_CLOSING: &str = "multipart body has no closing boundary";
const HEADERS: &str = "multipart part headers are malformed";
const HEADERS_SIZE: &str = "multipart part headers exceed 16384 bytes";
const ONE_DISPOSITION: &str = "multipart part needs exactly one Content-Disposition";
const DISPOSITION: &str = "multipart part has an invalid Content-Disposition";
const NO_NAME: &str = "multipart part has no field name";

/// What an API allows.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub max_fields: usize,
    pub unique_names: bool,
}

/// Why a form was not read to its end.
#[derive(Debug, PartialEq, Eq)]
pub enum Failure {
    /// The Content-Type is not `multipart/form-data`.
    NotFormData,
    /// Its `boundary` parameter is missing, not 1 to 70 RFC 2046 characters,
    /// or in malformed parameters.
    Boundary,
    /// The body breaks the format; the message says how.
    Malformed(&'static str),
    /// More fields than `Limits::max_fields`.
    TooManyFields,
    /// With `Limits::unique_names`, the name an earlier field had.
    Duplicate(Vec<u8>),
    /// The caller stopped at a field.
    Stopped,
    /// An allocation could not be reserved.
    NoMemory,
}

/// One field of a form.
#[derive(Debug, PartialEq, Eq)]
pub struct Field<'a> {
    pub name: &'a [u8],
    /// Its Content-Disposition has a `filename` (or `filename*`) parameter.
    pub has_filename: bool,
    pub content: &'a [u8],
}

/// Reads the form in `body`, handing each field to `on_field` in order, each
/// only after the checks that come before it; `on_field` returns false to
/// stop. The body starts with its first boundary: RFC 2046 would ignore a
/// preamble, but then a malformed first boundary line would drop the parts up
/// to the next one without an error. An epilogue after the last boundary is
/// ignored.
pub fn parse(
    content_type: &[u8],
    body: &[u8],
    limits: Limits,
    on_field: &mut dyn FnMut(Field<'_>) -> bool,
) -> Result<(), Failure> {
    let boundary = boundary(content_type)?;
    // CRLF "--" boundary; the first delimiter lacks the CRLF.
    let mut delimiter_buffer = [0u8; MAXIMUM_BOUNDARY + 4];
    delimiter_buffer[..4].copy_from_slice(b"\r\n--");
    delimiter_buffer[4..4 + boundary.len()].copy_from_slice(&boundary);
    let delimiter_bytes = &delimiter_buffer[..4 + boundary.len()];
    let mut delimiter = Delimiter {
        start: 0,
        part: delimiter_at(body, &delimiter_bytes[2..], 0).ok_or(Failure::Malformed(NO_OPENING))?,
    };
    let mut names: Vec<Vec<u8>> = Vec::new();
    let mut fields = 0;
    while let Some(part) = delimiter.part {
        let (headers_len, name, has_filename) = part_headers(&body[part..])?;
        let content_begin = part + headers_len;
        // The CRLF that ends the headers may also start the next delimiter:
        // a part without content (RFC 2046 body-part).
        let next = next_delimiter(body, delimiter_bytes, content_begin - 2)
            .ok_or(Failure::Malformed(NO_CLOSING))?;
        let content = &body[content_begin.min(next.start)..next.start];
        fields += 1;
        if fields > limits.max_fields {
            return Err(Failure::TooManyFields);
        }
        if limits.unique_names {
            if names.contains(&name) {
                return Err(Failure::Duplicate(name));
            }
            names.try_reserve(1).map_err(|_| Failure::NoMemory)?;
        }
        let field = Field {
            name: &name,
            has_filename,
            content,
        };
        if !on_field(field) {
            return Err(Failure::Stopped);
        }
        if limits.unique_names {
            names.push(name);
        }
        delimiter = next;
    }
    Ok(())
}

/// A delimiter line in the body.
struct Delimiter {
    /// Where it starts: its CRLF, or the body's start for the first one.
    start: usize,
    /// Where the next part starts; `None` for the close delimiter.
    part: Option<usize>,
}

/// The delimiter line whose `--boundary` starts at `at`: the close delimiter
/// (`--`, transport padding and a line end or EOF follow), or a delimiter
/// followed by transport padding and CRLF. An epilogue may follow CRLF.
fn delimiter_at(body: &[u8], dash_boundary: &[u8], at: usize) -> Option<Option<usize>> {
    let rest = body.get(at..)?.strip_prefix(dash_boundary)?;
    if let Some(after) = rest.strip_prefix(b"--") {
        // Only SP/HTAB are transport padding, not arbitrary ASCII whitespace.
        let padding = after_padding(after);
        return (padding.is_empty() || padding.starts_with(b"\r\n")).then_some(None);
    }
    let padding = rest
        .iter()
        .take_while(|&&byte| byte == b' ' || byte == b'\t')
        .count();
    rest[padding..]
        .starts_with(b"\r\n")
        .then_some(Some(at + dash_boundary.len() + padding + 2))
}

fn after_padding(bytes: &[u8]) -> &[u8] {
    let len = bytes.iter().take_while(|&&byte| is_blank(byte)).count();
    &bytes[len..]
}

/// The first delimiter (`delimiter_bytes`, CRLF "--" boundary) at or after
/// `from`; a `--boundary` that no `--` or CRLF follows is content.
fn next_delimiter(body: &[u8], delimiter_bytes: &[u8], from: usize) -> Option<Delimiter> {
    let mut at = from;
    loop {
        let found = find(body, delimiter_bytes, at)?;
        if let Some(part) = delimiter_at(body, &delimiter_bytes[2..], found + 2) {
            return Some(Delimiter { start: found, part });
        }
        at = found + 1;
    }
}

/// A part's header block: its length with the empty line that ends it, the
/// field name and whether a filename is given.
fn part_headers(part: &[u8]) -> Result<(usize, Vec<u8>, bool), Failure> {
    let window = &part[..part.len().min(MAXIMUM_HEADER_BYTES + 4)];
    let mut slots = [httparse::EMPTY_HEADER; MAXIMUM_PART_HEADERS];
    let (len, headers) = match httparse::parse_headers(window, &mut slots) {
        Ok(httparse::Status::Complete(parsed)) => parsed,
        Ok(httparse::Status::Partial) if window.len() < part.len() => {
            return Err(Failure::Malformed(HEADERS_SIZE));
        }
        Ok(httparse::Status::Partial) | Err(_) => return Err(Failure::Malformed(HEADERS)),
    };
    // httparse also takes a bare LF; RFC 2046 lines end with CRLF, as the
    // delimiters do.
    let block = &part[..len];
    if (0..len).any(|index| block[index] == b'\n' && (index == 0 || block[index - 1] != b'\r')) {
        return Err(Failure::Malformed(HEADERS));
    }
    let mut dispositions = headers
        .iter()
        .filter(|header| header.name.eq_ignore_ascii_case("content-disposition"));
    let (Some(disposition), None) = (dispositions.next(), dispositions.next()) else {
        return Err(Failure::Malformed(ONE_DISPOSITION));
    };
    let (kind, parameters) = parameters(disposition.value).map_err(|error| match error {
        ParameterError::Malformed => Failure::Malformed(DISPOSITION),
        ParameterError::NoMemory => Failure::NoMemory,
    })?;
    if !kind.eq_ignore_ascii_case(b"form-data") {
        return Err(Failure::Malformed(DISPOSITION));
    }
    let name = lookup(&parameters, b"name")
        .filter(|name| !name.is_empty())
        .ok_or(Failure::Malformed(NO_NAME))?;
    let has_filename =
        lookup(&parameters, b"filename").is_some() || lookup(&parameters, b"filename*").is_some();
    Ok((
        len,
        copy_bytes(name).map_err(|_| Failure::NoMemory)?,
        has_filename,
    ))
}

/// The boundary of a `multipart/form-data` Content-Type.
fn boundary(content_type: &[u8]) -> Result<Vec<u8>, Failure> {
    let media_type = content_type
        .split(|&byte| byte == b';')
        .next()
        .unwrap_or_default();
    if !trim(media_type).eq_ignore_ascii_case(b"multipart/form-data") {
        return Err(Failure::NotFormData);
    }
    let (_, parameters) = parameters(content_type).map_err(|error| match error {
        ParameterError::Malformed => Failure::Boundary,
        ParameterError::NoMemory => Failure::NoMemory,
    })?;
    match lookup(&parameters, b"boundary") {
        Some(boundary) if is_boundary(boundary) => {
            copy_bytes(boundary).map_err(|_| Failure::NoMemory)
        }
        _ => Err(Failure::Boundary),
    }
}

/// RFC 2046 5.1.1: 1 to 70 `bchars`, the last one not a space.
fn is_boundary(boundary: &[u8]) -> bool {
    (1..=MAXIMUM_BOUNDARY).contains(&boundary.len())
        && boundary.last() != Some(&b' ')
        && boundary
            .iter()
            .all(|&byte| byte.is_ascii_alphanumeric() || b"'()+_,-./:=? ".contains(&byte))
}

/// Parameter names with their values.
type Parameters<'a> = Vec<(&'a [u8], Cow<'a, [u8]>)>;

fn lookup<'a>(parameters: &'a Parameters<'_>, name: &[u8]) -> Option<&'a [u8]> {
    parameters
        .iter()
        .find(|(candidate, _)| candidate.eq_ignore_ascii_case(name))
        .map(|(_, value)| &value[..])
}

fn is_blank(byte: u8) -> bool {
    byte == b' ' || byte == b'\t'
}

fn trim(value: &[u8]) -> &[u8] {
    let start = value
        .iter()
        .position(|&byte| !is_blank(byte))
        .unwrap_or(value.len());
    let end = value
        .iter()
        .rposition(|&byte| !is_blank(byte))
        .map_or(start, |end| end + 1);
    &value[start..end]
}

/// RFC 9110 5.6.2.
fn is_tchar(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte)
}

/// A value as RFC 9110 5.6.6 writes it after a type (`form-data`,
/// `multipart/form-data`): the type, and each `; name=value` with the value
/// a quoted string or an unquoted run of visible bytes other than `"` and
/// `;`. Parameter names ignore ASCII case and appear at most once, and there
/// are at most `MAXIMUM_PARAMETERS` parameters. As the
/// parsers before this one did, spaces around `=` are allowed and empty
/// parameters and names without a value are skipped. Syntax and allocation
/// failures are reported separately.
#[derive(Debug)]
enum ParameterError {
    Malformed,
    NoMemory,
}

impl From<std::collections::TryReserveError> for ParameterError {
    fn from(_: std::collections::TryReserveError) -> Self {
        Self::NoMemory
    }
}

fn copy_bytes(bytes: &[u8]) -> Result<Vec<u8>, ParameterError> {
    let mut result = Vec::new();
    result.try_reserve_exact(bytes.len())?;
    result.extend_from_slice(bytes);
    Ok(result)
}

fn parameters(value: &[u8]) -> Result<(&[u8], Parameters<'_>), ParameterError> {
    let type_end = value
        .iter()
        .position(|&byte| byte == b';')
        .unwrap_or(value.len());
    let kind = trim(&value[..type_end]);
    if kind.is_empty() {
        return Err(ParameterError::Malformed);
    }
    let mut rest = &value[type_end..];
    let mut parameters: Parameters<'_> = Vec::new();
    loop {
        rest = trim(rest);
        let Some(after) = rest.strip_prefix(b";") else {
            return if rest.is_empty() {
                Ok((kind, parameters))
            } else {
                Err(ParameterError::Malformed)
            };
        };
        rest = trim(after);
        if rest.is_empty() || rest[0] == b';' {
            continue;
        }
        let name_len = rest.iter().take_while(|&&byte| is_tchar(byte)).count();
        let name = &rest[..name_len];
        rest = trim(&rest[name_len..]);
        if name.is_empty() {
            return Err(ParameterError::Malformed);
        }
        if rest.is_empty() || rest[0] == b';' {
            continue;
        }
        rest = trim(rest.strip_prefix(b"=").ok_or(ParameterError::Malformed)?);
        if parameters.len() == MAXIMUM_PARAMETERS || lookup(&parameters, name).is_some() {
            return Err(ParameterError::Malformed);
        }
        let value = if let Some(quoted) = rest.strip_prefix(b"\"") {
            let (value, len) = quoted_string(quoted)?;
            rest = &quoted[len..];
            value
        } else {
            let len = rest
                .iter()
                .take_while(|&&byte| byte != b';' && !is_blank(byte))
                .count();
            let token = &rest[..len];
            if token.is_empty()
                || token
                    .iter()
                    .any(|&byte| byte == b'"' || byte < 0x21 || byte == 0x7f)
            {
                return Err(ParameterError::Malformed);
            }
            rest = &rest[len..];
            Cow::Borrowed(token)
        };
        parameters.try_reserve(1)?;
        parameters.push((name, value));
    }
}

/// Borrow ordinary quoted values; allocate only to remove backslash escapes.
fn quoted_string(text: &[u8]) -> Result<(Cow<'_, [u8]>, usize), ParameterError> {
    let allowed = |byte: u8| byte == b'\t' || (byte >= 0x20 && byte != 0x7f);
    let mut value: Option<Vec<u8>> = None;
    let mut index = 0;
    loop {
        match *text.get(index).ok_or(ParameterError::Malformed)? {
            b'"' => {
                return Ok((
                    value.map_or(Cow::Borrowed(&text[..index]), Cow::Owned),
                    index + 1,
                ));
            }
            b'\\' => {
                let escaped = *text.get(index + 1).ok_or(ParameterError::Malformed)?;
                if !allowed(escaped) {
                    return Err(ParameterError::Malformed);
                }
                if value.is_none() {
                    value = Some(copy_bytes(&text[..index])?);
                }
                let output = value.as_mut().ok_or(ParameterError::Malformed)?;
                output.try_reserve(1)?;
                output.push(escaped);
                index += 2;
            }
            byte if allowed(byte) => {
                if let Some(output) = value.as_mut() {
                    output.try_reserve(1)?;
                    output.push(byte);
                }
                index += 1;
            }
            _ => return Err(ParameterError::Malformed),
        }
    }
}

/// The first index of `needle`, which is not empty, in `haystack` at or
/// after `from`: each occurrence of its first byte is a candidate.
fn find(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    let (&first, rest) = needle.split_first()?;
    let last_start = haystack.len().checked_sub(needle.len())?;
    let starts = &haystack[..=last_start];
    let mut at = from;
    loop {
        at = find_byte(starts, first, at)?;
        if haystack[at + 1..].starts_with(rest) {
            return Some(at);
        }
        at += 1;
    }
}

/// The first index of `byte` in `haystack` at or after `from`, eight bytes
/// per step, as memchr does without SIMD: XOR with `byte` in every lane turns
/// a match into a zero byte, and `(word - 0x01..) & !word & 0x80..` sets the
/// top bit of the lowest zero byte (and maybe of higher ones), so the lowest
/// set bit marks the first match.
fn find_byte(haystack: &[u8], byte: u8, from: usize) -> Option<usize> {
    const LOW: u64 = 0x0101_0101_0101_0101;
    const HIGH: u64 = 0x8080_8080_8080_8080;
    let pattern = LOW * u64::from(byte);
    let mut at = from;
    while let Some(chunk) = haystack.get(at..at + 8) {
        let chunk: [u8; 8] = chunk.try_into().ok()?;
        let word = u64::from_le_bytes(chunk) ^ pattern;
        let zeros = word.wrapping_sub(LOW) & !word & HIGH;
        if zeros != 0 {
            return Some(at + zeros.trailing_zeros() as usize / 8);
        }
        at += 8;
    }
    let tail = haystack.get(at..)?;
    tail.iter()
        .position(|&candidate| candidate == byte)
        .map(|index| at + index)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ANY: Limits = Limits {
        max_fields: 100,
        unique_names: false,
    };
    const CONTENT_TYPE: &[u8] = b"multipart/form-data; boundary=b";

    /// Names, whether each has a filename, and contents.
    type Fields = Vec<(String, bool, Vec<u8>)>;

    fn read(content_type: &[u8], body: &[u8], limits: Limits) -> Result<Fields, Failure> {
        let mut fields = Vec::new();
        parse(content_type, body, limits, &mut |field| {
            let name = String::from_utf8_lossy(field.name).into_owned();
            fields.push((name, field.has_filename, field.content.to_vec()));
            true
        })?;
        Ok(fields)
    }

    fn part(disposition: &str, content: &str) -> String {
        format!("--b\r\nContent-Disposition: {disposition}\r\n\r\n{content}\r\n")
    }

    fn field(name: &str, has_filename: bool, content: &str) -> (String, bool, Vec<u8>) {
        (name.into(), has_filename, content.as_bytes().to_vec())
    }

    #[test]
    fn reads_fields_in_order() {
        let body = [
            part(r#"form-data; name="model""#, "m"),
            part(
                r#"form-data; name="file"; filename="a.wav""#,
                "RIFF\r\n--bx",
            ),
            part(r#"form-data; name="empty""#, ""),
            "--b--\r\n".into(),
        ]
        .concat();
        assert_eq!(
            read(CONTENT_TYPE, body.as_bytes(), ANY),
            Ok(vec![
                field("model", false, "m"),
                field("file", true, "RIFF\r\n--bx"),
                field("empty", false, ""),
            ])
        );
        assert_eq!(read(CONTENT_TYPE, b"--b--", ANY), Ok(vec![]));
    }

    #[test]
    fn follows_rfc_2046_framing() {
        let core = part("form-data; name=a", "1");
        for body in [
            format!("{core}--b--"),
            format!("{core}--b--\r\nepilogue"),
            format!("{core}--b-- \t\r\ntrailing"),
            "--b \t\r\nContent-Disposition: form-data; name=a\r\n\r\n1\r\n--b--".into(),
        ] {
            assert_eq!(
                read(CONTENT_TYPE, body.as_bytes(), ANY),
                Ok(vec![field("a", false, "1")]),
                "{body:?}"
            );
        }
        // A part without content: the headers' CRLF starts the delimiter.
        assert_eq!(
            read(
                CONTENT_TYPE,
                b"--b\r\nContent-Disposition: form-data; name=a\r\n\r\n--b--",
                ANY
            ),
            Ok(vec![field("a", false, "")])
        );
        for (body, message) in [
            ("--bx\r\n", NO_OPENING),
            (
                "preamble\r\n--b\r\nContent-Disposition: form-data; name=a\r\n\r\n1\r\n--b--",
                NO_OPENING,
            ),
            // A malformed first line drops no parts silently.
            (
                "--b\nContent-Disposition: form-data; name=a\r\n\r\n1\r\n--b--",
                NO_OPENING,
            ),
            ("", NO_OPENING),
            (
                "--b\r\nContent-Disposition: form-data; name=a\r\n\r\n1",
                NO_CLOSING,
            ),
            (
                "--b\r\nContent-Disposition: form-data; name=a\r\n\r\n1\r\n--bx",
                NO_CLOSING,
            ),
            ("--b\r\nContent-Disposition: form-data; name=a\r\n", HEADERS),
            (
                "--b\r\nContent-Disposition : form-data; name=a\r\n\r\n\r\n--b--",
                HEADERS,
            ),
            ("--b\r\nX: a\r\n b\r\n\r\n\r\n--b--", HEADERS),
            (
                "--b\r\nContent-Disposition: form-data; name=a\n\n1\r\n--b--",
                HEADERS,
            ),
            (
                "--b\r\nContent-Type: text/plain\r\n\r\n\r\n--b--",
                ONE_DISPOSITION,
            ),
            (
                "--b\r\nContent-Disposition: form-data; name=a\r\ncontent-disposition: form-data; name=a\r\n\r\n\r\n--b--",
                ONE_DISPOSITION,
            ),
            (
                "--b\r\nContent-Disposition: attachment; name=a\r\n\r\n\r\n--b--",
                DISPOSITION,
            ),
            (
                "--b\r\nContent-Disposition: form-data; name=\"a\r\n\r\n\r\n--b--",
                DISPOSITION,
            ),
            (
                "--b\r\nContent-Disposition: form-data; name=\"\"\r\n\r\n\r\n--b--",
                NO_NAME,
            ),
            (
                "--b\r\nContent-Disposition: form-data\r\n\r\n\r\n--b--",
                NO_NAME,
            ),
        ] {
            assert_eq!(
                read(CONTENT_TYPE, body.as_bytes(), ANY),
                Err(Failure::Malformed(message)),
                "{body:?}"
            );
        }
    }

    #[test]
    fn closing_delimiters_need_a_line_end() {
        for content in ["before\r\n--b--junk\r\nafter", "before\r\n--b-- \tjunk"] {
            let body = format!("{}--b--\r\nepilogue", part("form-data; name=f", content));
            assert_eq!(
                read(CONTENT_TYPE, body.as_bytes(), ANY),
                Ok(vec![field("f", false, content)])
            );
        }
        for ending in ["--b--junk", "--b-- \tjunk", "--b--\n"] {
            let body = format!("{}{ending}", part("form-data; name=f", "v"));
            assert_eq!(
                read(CONTENT_TYPE, body.as_bytes(), ANY),
                Err(Failure::Malformed(NO_CLOSING))
            );
        }
        for ending in ["--b--", "--b-- \t", "--b-- \t\r\nepilogue"] {
            let body = format!("{}{ending}", part("form-data; name=f", "v"));
            assert_eq!(
                read(CONTENT_TYPE, body.as_bytes(), ANY),
                Ok(vec![field("f", false, "v")])
            );
        }
    }

    #[test]
    fn reads_rfc_9110_parameters() {
        let named = |disposition: &str| {
            let body = format!("{}--b--", part(disposition, "v"));
            read(CONTENT_TYPE, body.as_bytes(), ANY).map(|fields| fields[0].0.clone())
        };
        let invalid = Err(Failure::Malformed(DISPOSITION));
        assert_eq!(
            named(r#"form-data; filename="a;name=x"; name="n""#),
            Ok("n".into())
        );
        assert_eq!(named(r#"Form-Data; NAME="a\"b\\c""#), Ok(r#"a"b\c"#.into()));
        assert_eq!(named("form-data;name=n;"), Ok("n".into()));
        assert_eq!(named("form-data; name=\"\u{e4}\""), Ok("\u{e4}".into()));
        // As before in all three APIs: spaces around `=`, and parameters
        // without a value are skipped.
        assert_eq!(named("form-data ;; name = n"), Ok("n".into()));
        assert_eq!(named("form-data; x; name=n"), Ok("n".into()));
        assert_eq!(named(r#"form-data; name="n"x"#), invalid);
        assert_eq!(named("form-data; name=a b"), invalid);
        assert_eq!(named(r#"form-data; name="a"; name="b""#), invalid);
        // At most 16 parameters, so that repeated names are found in linear
        // time.
        let others =
            |count: usize| -> String { (1..=count).map(|index| format!("; x{index}=v")).collect() };
        assert_eq!(
            named(&format!("form-data; name=n{}", others(15))),
            Ok("n".into())
        );
        assert_eq!(named(&format!("form-data; name=n{}", others(16))), invalid);
        let file = format!(
            "{}--b--",
            part("form-data; name=f; filename*=UTF-8''a.png", "v")
        );
        assert_eq!(
            read(CONTENT_TYPE, file.as_bytes(), ANY),
            Ok(vec![field("f", true, "v")])
        );

        let body = format!("{}--b--", part("form-data; name=a", "1"));
        for content_type in [
            "multipart/form-data; boundary=b",
            "Multipart/Form-Data; Boundary=\"b\"",
            "multipart/form-data;boundary=b;charset=utf-8",
            " multipart/form-data\t; boundary=b ",
        ] {
            assert!(
                read(content_type.as_bytes(), body.as_bytes(), ANY).is_ok(),
                "{content_type}"
            );
        }
        for (content_type, failure) in [
            ("multipart/mixed; boundary=b", Failure::NotFormData),
            ("multipart/form-datax; boundary=b", Failure::NotFormData),
            ("", Failure::NotFormData),
            ("multipart/form-data", Failure::Boundary),
            ("multipart/form-data; boundary=", Failure::Boundary),
            ("multipart/form-data; boundary=\"\"", Failure::Boundary),
            ("multipart/form-data; boundary=\"b", Failure::Boundary),
            (
                "multipart/form-data; boundary=b; boundary=c",
                Failure::Boundary,
            ),
        ] {
            assert_eq!(
                read(content_type.as_bytes(), body.as_bytes(), ANY),
                Err(failure),
                "{content_type}"
            );
        }
        let content_type = format!("multipart/form-data; boundary=b{}", others(15));
        assert!(read(content_type.as_bytes(), body.as_bytes(), ANY).is_ok());
        for count in [16, 10_000] {
            let content_type = format!("multipart/form-data; boundary=b{}", others(count));
            assert_eq!(
                read(content_type.as_bytes(), body.as_bytes(), ANY),
                Err(Failure::Boundary)
            );
        }
        // RFC 2046: 1 to 70 of its characters, the last one not a space.
        let long = "x".repeat(70);
        let content_type = format!("multipart/form-data; boundary={long}");
        let body = format!("--{long}--");
        assert!(read(content_type.as_bytes(), body.as_bytes(), ANY).is_ok());
        let content_type = "multipart/form-data; boundary=\"a'()+_,-./:=? z\"";
        let body = "--a'()+_,-./:=? z--";
        assert!(read(content_type.as_bytes(), body.as_bytes(), ANY).is_ok());
        for boundary in [
            format!("{long}x"),
            "a ".into(),
            "a\tb".into(),
            "a%b".into(),
            "a!b".into(),
            "\u{e4}".into(),
        ] {
            let content_type = format!("multipart/form-data; boundary=\"{boundary}\"");
            assert_eq!(
                read(content_type.as_bytes(), b"", ANY),
                Err(Failure::Boundary),
                "{boundary:?}"
            );
        }
    }

    #[test]
    fn applies_limits() {
        let form = |names: &[&str]| {
            let parts: String = names
                .iter()
                .map(|name| part(&format!("form-data; name={name}"), "v"))
                .collect();
            parts + "--b--"
        };
        let two = Limits {
            max_fields: 2,
            unique_names: true,
        };
        assert!(read(CONTENT_TYPE, form(&["a", "b"]).as_bytes(), two).is_ok());
        assert_eq!(
            read(CONTENT_TYPE, form(&["a", "b", "c"]).as_bytes(), two),
            Err(Failure::TooManyFields)
        );
        assert_eq!(
            read(CONTENT_TYPE, form(&["a", "a"]).as_bytes(), two),
            Err(Failure::Duplicate(b"a".to_vec()))
        );
        assert!(read(CONTENT_TYPE, form(&["a", "a"]).as_bytes(), ANY).is_ok());
        // Header lines up to 16 KiB.
        for (extra, ok) in [(0, true), (1, false)] {
            let disposition = "Content-Disposition: form-data; name=a\r\nX: ";
            let padding = "p".repeat(MAXIMUM_HEADER_BYTES - disposition.len() + extra);
            let body = format!("--b\r\n{disposition}{padding}\r\n\r\nv\r\n--b--");
            let result = read(CONTENT_TYPE, body.as_bytes(), ANY);
            if ok {
                assert!(result.is_ok(), "{result:?}");
            } else {
                assert_eq!(result, Err(Failure::Malformed(HEADERS_SIZE)));
            }
        }
    }

    #[test]
    fn stops_where_the_caller_says() {
        let body = [
            part("form-data; name=a", "1"),
            part("form-data; name=b", "2"),
            "--b--".into(),
        ]
        .concat();
        let mut seen = 0;
        let result = parse(CONTENT_TYPE, body.as_bytes(), ANY, &mut |_| {
            seen += 1;
            false
        });
        assert_eq!((result, seen), (Err(Failure::Stopped), 1));
    }

    #[test]
    fn finds_needles() {
        assert_eq!(find(b"aab", b"ab", 0), Some(1));
        assert_eq!(find(b"aab", b"ab", 2), None);
        assert_eq!(find(b"ab", b"ab", 3), None);
        assert_eq!(find(b"a", b"ab", 0), None);
        // Every length, start and match position across word boundaries,
        // with neighbours that set the borrow (0x01, the byte itself, 0x80).
        let naive = |haystack: &[u8], byte: u8, from: usize| {
            (from..haystack.len()).find(|&index| haystack[index] == byte)
        };
        for len in 0..34usize {
            for fill in [0x00, 0x01, 0x0c, 0x0e, 0x80, 0xff] {
                for matches in [
                    vec![],
                    vec![0],
                    vec![len / 2],
                    vec![7, 8],
                    vec![len.saturating_sub(1)],
                ] {
                    let mut haystack = vec![fill; len];
                    for &index in matches.iter().filter(|&&index| index < len) {
                        haystack[index] = b'\r';
                    }
                    for from in 0..len + 2 {
                        assert_eq!(
                            find_byte(&haystack, b'\r', from),
                            naive(&haystack, b'\r', from),
                            "{haystack:?} from {from}"
                        );
                    }
                }
            }
        }
        // Needles in pseudo-random bytes, with many candidates.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let haystack: Vec<u8> = (0..4096)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                b"\r\n-b"[(state >> 60) as usize % 4]
            })
            .collect();
        for needle in [&b"\r\n--b"[..], b"\r\n", b"b", b"--b\r\n--b"] {
            for from in [0, 1, 7, 8, 9, 1000, 4090, 4096, 5000] {
                let expected = (from..=haystack.len().saturating_sub(needle.len()))
                    .find(|&index| haystack[index..].starts_with(needle))
                    .filter(|_| haystack.len() >= needle.len());
                assert_eq!(
                    find(&haystack, needle, from),
                    expected,
                    "{needle:?} from {from}"
                );
            }
        }
    }
}
