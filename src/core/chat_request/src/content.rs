//! Message content: Chat content parts, the Responses part rewrite and the
//! Messages text, thinking and image blocks.

use crate::draft::{Image, Message, Role};
use crate::error::{Outcome, invalid};
use crate::fields::{member_str, member_str_or, push, take_text};
use crate::host::Host;
use gufo_json::Value;

/// One content part as `ParseContent` reads it. The Responses rewrite
/// (`ParseOpenAiResponseMessage`) turns `input_image` into an `image_url`
/// part and renames `input_text` and `output_text` to `text`.
#[derive(Debug, Clone, Copy)]
pub enum Part<'a> {
    /// A part as the client sent it.
    Sent(&'a Value),
    /// `{"type":"image_url","image_url":{"url":url,"detail":detail}}`.
    Image {
        url: &'a str,
        detail: Option<&'a Value>,
    },
    /// The part with its type replaced by "text".
    Text(&'a Value),
}

/// `ParseContent`, for content that is not a part array.
pub fn chat_content(
    content: Option<&mut Value>,
    message: &mut Message,
    host: &mut impl Host,
) -> Outcome<()> {
    match content {
        None | Some(Value::Null) => Ok(()),
        Some(value @ Value::String(_)) => {
            message.content = take_text(value);
            Ok(())
        }
        Some(Value::Array(items)) => parts(items.iter().map(Part::Sent), message, host),
        Some(_) => invalid("message content must be a string, null, or content-part array"),
    }
}

/// `ParseContent` over the parts of an array.
pub fn parts<'a>(
    parts: impl IntoIterator<Item = Part<'a>>,
    message: &mut Message,
    host: &mut impl Host,
) -> Outcome<()> {
    for part in parts {
        let (kind, sent) = match part {
            Part::Sent(value) => {
                if !matches!(value, Value::Object(_)) {
                    return invalid("message content parts must be objects");
                }
                (member_str_or(value, "type", "text"), Some(value))
            }
            Part::Image { .. } => ("image_url", None),
            Part::Text(value) => ("text", Some(value)),
        };
        if kind == "image_url" {
            let (url, detail) = match part {
                Part::Image { url, detail } => (Some(url), detail),
                _ => {
                    let image = sent.and_then(|value| value.get("image_url"));
                    let url = image
                        .filter(|image| matches!(image, Value::Object(_)))
                        .and_then(|image| image.get("url"));
                    (
                        url.and_then(Value::as_str),
                        image.and_then(|image| image.get("detail")),
                    )
                }
            };
            let Some(url) = url.filter(|_| matches!(message.role, Role::User | Role::Tool)) else {
                return invalid("image_url requires a user or tool message and a string URL");
            };
            // Resolution is model-owned, except that 'low' bounds it.
            if let Some(detail) = detail
                && !matches!(detail.as_str(), Some("auto" | "low" | "high" | "original"))
            {
                return invalid("image_url.detail must be 'auto', 'low', 'high' or 'original'");
            }
            let low = detail.and_then(Value::as_str) == Some("low");
            let id = host.read_image(url.as_bytes())?;
            let offset = message.content.len();
            push(&mut message.images, Image { id, offset, low })?;
            continue;
        }
        if kind == "thinking" {
            // An earlier thought of an assistant turn, in its place among the
            // text parts (the reference Inkling renderer's form).
            let thinking = sent
                .and_then(|value| value.get("thinking"))
                .and_then(Value::as_str);
            let (Role::Assistant, Some(thinking)) = (message.role, thinking) else {
                return invalid(
                    "thinking content parts require an assistant message and a string 'thinking'",
                );
            };
            message.thought_part(thinking)?;
            continue;
        }
        if kind != "text" && kind != "input_text" {
            return invalid("message content parts must use text or image_url");
        }
        let Some(part_text) = sent
            .and_then(|value| value.get("text"))
            .and_then(Value::as_str)
        else {
            return invalid("text message content parts require a string 'text'");
        };
        message.text_part(part_text)?;
    }
    Ok(())
}

/// How the Responses rewrite reads each part of a content array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rewrite {
    Sent,
    Image,
    Text,
}

/// The Responses rewrite of a content array, checked before the message is.
pub fn responses_parts(items: &[Value]) -> Outcome<Vec<Rewrite>> {
    let mut out = Vec::new();
    out.try_reserve_exact(items.len())?;
    for part in items {
        out.push(match member_str(part, "type") {
            "input_image" => {
                if part.get("image_url").and_then(Value::as_str).is_none()
                    || part.get("file_id").is_some()
                {
                    return invalid("input_image requires an image_url");
                }
                Rewrite::Image
            }
            "input_text" | "output_text" => Rewrite::Text,
            "text" => Rewrite::Sent,
            _ => return invalid("unsupported Responses input content type"),
        });
    }
    Ok(out)
}

/// The part a rewrite makes of `part`.
pub fn rewritten(kind: Rewrite, part: &Value) -> Part<'_> {
    match kind {
        Rewrite::Sent => Part::Sent(part),
        Rewrite::Text => Part::Text(part),
        Rewrite::Image => Part::Image {
            url: part.get("image_url").and_then(Value::as_str).unwrap_or(""),
            detail: part.get("detail"),
        },
    }
}

/// `ParseAnthropicImage`: an image block appended at the content's end.
pub fn anthropic_image(block: &Value, message: &mut Message, host: &mut impl Host) -> Outcome<()> {
    let source = block
        .get("source")
        .filter(|source| matches!(source, Value::Object(_)));
    let kind = source.map_or("", |source| member_str(source, "type"));
    let id = match (kind, source) {
        ("base64", Some(source)) => {
            let media = source.get("media_type").and_then(Value::as_str);
            let data = source.get("data").and_then(Value::as_str);
            let (Some(media), Some(data)) = (media, data) else {
                return invalid("base64 image sources require string media_type and data");
            };
            let mut url = Vec::new();
            url.try_reserve_exact(media.len() + data.len() + 13)?;
            url.extend_from_slice(b"data:");
            url.extend_from_slice(media.as_bytes());
            url.extend_from_slice(b";base64,");
            url.extend_from_slice(data.as_bytes());
            host.read_image(&url)?
        }
        ("url", Some(source)) => {
            let Some(url) = source.get("url").and_then(Value::as_str) else {
                return invalid("url image sources require a string url");
            };
            host.read_image(url.as_bytes())?
        }
        _ => return invalid("image blocks require a base64 or url source"),
    };
    let offset = message.content.len();
    push(
        &mut message.images,
        Image {
            id,
            offset,
            low: false,
        },
    )
}

/// `ReadTextContent` (src/cli/serve/http_server.cpp): `Ok(false)` where it
/// returned false; an image's failure message is `Failure::Invalid`. With
/// `thought`, thinking blocks add to the message's thought; with `images`,
/// image blocks are read into it.
pub fn read_text_content(
    content: Option<&mut Value>,
    message: &mut Message,
    thought: bool,
    images: bool,
    host: &mut impl Host,
) -> Outcome<bool> {
    let Some(content) = content else {
        return Ok(false);
    };
    match content {
        Value::String(_) => {
            message.content = take_text(content);
            Ok(true)
        }
        Value::Array(items) => {
            for part in items.iter() {
                let object = matches!(part, Value::Object(_));
                let kind = member_str(part, "type");
                if thought && object && kind == "thinking" {
                    let Some(thinking) = part.get("thinking").and_then(Value::as_str) else {
                        return Ok(false);
                    };
                    message.thought_part(thinking)?;
                    continue;
                }
                if images && object && kind == "image" {
                    anthropic_image(part, message, host)?;
                    continue;
                }
                let part_text = part.get("text").and_then(Value::as_str);
                match part_text {
                    Some(part_text)
                        if object && matches!(kind, "text" | "input_text" | "output_text") =>
                    {
                        message.text_part(part_text)?;
                    }
                    _ => return Ok(false),
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}
