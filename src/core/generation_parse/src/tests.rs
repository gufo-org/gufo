//! Streams parsed whole and one byte at a time must agree, with stub
//! scanners standing in for the C++ host.

use crate::config::{
    Config, DialectScan, Envelope, Error, Host, Initial, Markup, Mode, QwenCall, QwenScan,
    Reasoning, ScannedArgument, ScannedCall, Tool, ToolChoice, Unsatisfied,
};
use crate::events::Event;
use crate::stream::Stream;
use crate::text::{NPOS, find};

/// A JSON-form Qwen scanner (`<tool_call>{json}</tool_call>`) and a toy
/// dialect: `<call:NAME>key=«value»;...</call>`.
struct Stub;

impl Host for Stub {
    fn qwen_scan(&mut self, text: &[u8]) -> Result<QwenScan, Error> {
        let open = b"<tool_call>";
        let close = b"</tool_call>";
        if !text.starts_with(open) {
            return Ok(QwenScan::default());
        }
        let end = find(text, close, open.len());
        if end == NPOS {
            return Ok(QwenScan {
                matched: true,
                progress: open.len(),
                end: 0,
                call: Some(QwenCall::default()),
            });
        }
        Ok(QwenScan {
            matched: true,
            progress: end + close.len(),
            end: end + close.len(),
            call: Some(QwenCall {
                native: false,
                json_payload: text[open.len()..end].to_vec(),
                complete: true,
                ..QwenCall::default()
            }),
        })
    }

    fn dialect_scan(
        &mut self,
        text: &[u8],
        markers: &[&[u8]],
        partial: bool,
    ) -> Result<DialectScan, Error> {
        assert_eq!(markers, [b"<call:".as_slice()]);
        let mut scan = DialectScan::default();
        let mut cursor = 0;
        loop {
            let begin = find(text, b"<call:", cursor);
            if begin == NPOS {
                return Ok(scan);
            }
            let name_end = find(text, b">", begin);
            if name_end == NPOS {
                return Ok(scan);
            }
            let mut call = ScannedCall {
                name: text[begin + 6..name_end].to_vec(),
                ..ScannedCall::default()
            };
            let mut at = name_end + 1;
            loop {
                if text[at..].starts_with(b"</call>") {
                    call.valid = true;
                    call.complete = true;
                    scan.spans.push((begin, at + 7));
                    scan.calls.push(call);
                    cursor = at + 7;
                    break;
                }
                let equals = find(text, "=«".as_bytes(), at);
                let close = find(text, "»;".as_bytes(), at);
                if equals == NPOS {
                    if partial {
                        scan.calls.push(call);
                    }
                    return Ok(scan);
                }
                let value_begin = equals + "=«".len();
                if close == NPOS {
                    if partial {
                        // Hold back a closer that may be arriving, so the
                        // value only grows.
                        let value = &text[value_begin..];
                        let value = value.strip_suffix("»".as_bytes()).unwrap_or(value);
                        call.unfinished = Some(ScannedArgument {
                            name: text[at..equals].to_vec(),
                            value: value.to_vec(),
                            is_string: true,
                        });
                        scan.calls.push(call);
                    }
                    return Ok(scan);
                }
                call.arguments.push(ScannedArgument {
                    name: text[at..equals].to_vec(),
                    value: text[value_begin..close].to_vec(),
                    is_string: true,
                });
                at = close + "»;".len();
            }
        }
    }

    fn grammar_accepts(&mut self, _tool: usize, _arguments: &[u8]) -> Result<bool, Error> {
        Ok(true)
    }
}

fn config(initial: Initial, choice: ToolChoice, dialect: bool) -> Config {
    let bytes = |values: &[&str]| {
        values
            .iter()
            .map(|value| value.as_bytes().to_vec())
            .collect()
    };
    Config {
        initial,
        mode: Mode::Plain,
        choice,
        track_spans: true,
        buffered: false,
        reasoning: Reasoning {
            open: b"<think>".to_vec(),
            close: b"</think>".to_vec(),
        },
        markup: Markup {
            markers: bytes(if dialect {
                &["<call:"]
            } else {
                &["<tool_call>"]
            }),
            reasoning_markers: Vec::new(),
            closers: bytes(if dialect {
                &["</call>"]
            } else {
                &["</tool_call>"]
            }),
            qwen_calls: !dialect,
            native: dialect,
            dialect,
            reasoning_separator: b"\r\n".to_vec(),
            qwen_open: b"<tool_call>".to_vec(),
            envelope: Envelope {
                heads: bytes(&["<invoke name="]),
                parameters: bytes(&["<parameter name="]),
                closers: bytes(&["</invoke>", "</parameter>"]),
            },
        },
        tools: vec![
            Tool::new(
                "weather".to_owned(),
                br#"{"type":"object","properties":{"city":{"type":"string"}}}"#,
            )
            .unwrap(),
        ],
    }
}

/// What a client assembles: joined deltas per kind and call, then the final
/// events, and the error.
fn run(config: &Config, pieces: &[&[u8]], interrupted: bool) -> (Vec<String>, Option<Error>) {
    let mut stream = Stream::new(config.clone());
    let mut reasoning = Vec::new();
    let mut content = Vec::new();
    let mut tools: Vec<(String, String, bool)> = Vec::new();
    let mut finals = Vec::new();
    let mut sink = |event: Event<'_>| {
        match event {
            Event::ReasoningDelta { text, .. } => reasoning.extend_from_slice(text),
            Event::ContentDelta { text, .. } => content.extend_from_slice(text),
            Event::ToolDelta {
                index,
                name,
                arguments,
                begin,
                complete,
            } => {
                if begin {
                    assert_eq!(index, tools.len());
                    tools.push((
                        String::from_utf8(name.to_vec()).unwrap(),
                        String::new(),
                        false,
                    ));
                }
                tools[index]
                    .1
                    .push_str(std::str::from_utf8(arguments).unwrap());
                tools[index].2 = complete;
            }
            Event::FinalReasoning { text } => finals.push(format!(
                "final reasoning {:?}",
                String::from_utf8_lossy(text)
            )),
            Event::FinalContent { text, .. } => {
                finals.push(format!("final content {:?}", String::from_utf8_lossy(text)))
            }
            Event::FinalCall {
                index,
                name,
                arguments,
            } => finals.push(format!(
                "final call {index} {} {}",
                String::from_utf8_lossy(name),
                String::from_utf8_lossy(arguments)
            )),
            Event::FinalEnd { hide_tool_markup } => {
                finals.push(format!("final end {hide_tool_markup}"))
            }
        }
        true
    };
    let mut result = Ok(());
    for piece in pieces {
        result = stream.feed(piece, &mut Stub, &mut sink);
        if result.is_err() {
            break;
        }
    }
    if result.is_ok() {
        result = stream.finish(!interrupted, interrupted, &mut Stub, &mut sink);
    }
    let mut out = vec![
        format!("reasoning {:?}", String::from_utf8_lossy(&reasoning)),
        format!("content {:?}", String::from_utf8_lossy(&content)),
        format!("tools {tools:?}"),
    ];
    out.extend(finals);
    (out, result.err())
}

/// The stream of `text` whole, one byte at a time and at every two-piece
/// split assembles the same response.
fn check(config: &Config, text: &str) -> (Vec<String>, Option<Error>) {
    let text = text.as_bytes();
    let whole = run(config, &[text], false);
    let bytes: Vec<&[u8]> = text.chunks(1).collect();
    assert_eq!(
        run(config, &bytes, false),
        whole,
        "bytes of {:?}",
        String::from_utf8_lossy(text)
    );
    for at in 1..text.len() {
        assert_eq!(
            run(config, &[&text[..at], &text[at..]], false),
            whole,
            "split {at} of {:?}",
            String::from_utf8_lossy(text)
        );
    }
    whole
}

#[test]
fn reasoning_and_content_split_the_same_one_byte_at_a_time() {
    let (out, error) = check(
        &config(Initial::Auto, ToolChoice::Auto, false),
        "<think>Plan the €.</think>\r\nThe answer is 3 < 5.",
    );
    assert_eq!(error, None);
    assert_eq!(out[0], r#"reasoning "Plan the €.""#);
    assert_eq!(out[1], r#"content "The answer is 3 < 5.""#);
}

#[test]
fn a_qwen_json_call_parses_one_byte_at_a_time() {
    let (out, error) = check(
        &config(Initial::Content, ToolChoice::Auto, false),
        r#"Calling.<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>"#,
    );
    assert_eq!(error, None);
    assert_eq!(out[1], r#"content "Calling.""#);
    assert!(
        out.iter()
            .any(|line| line == r#"final call 0 weather {"city":"Paris"}"#)
    );
}

#[test]
fn a_quoted_marker_stays_prose() {
    let (out, error) = check(
        &config(Initial::Content, ToolChoice::Auto, false),
        "Write `<tool_call>` to call a tool.",
    );
    assert_eq!(error, None);
    assert_eq!(out[1], r#"content "Write `<tool_call>` to call a tool.""#);
}

#[test]
fn dialect_calls_stream_partial_arguments_that_only_grow() {
    let config = config(Initial::Content, ToolChoice::Auto, true);
    let text = "Sure.<call:weather>city=«Paris»;</call>";
    let (whole, _) = run(&config, &[text.as_bytes()], false);
    let bytes: Vec<&[u8]> = text.as_bytes().chunks(1).collect();
    let (streamed, error) = run(&config, &bytes, false);
    assert_eq!(error, None);
    assert_eq!(streamed, whole);
    assert_eq!(
        streamed[2],
        r#"tools [("weather", "{\"city\":\"Paris\"}", true)]"#
    );
}

#[test]
fn required_calls_fail_when_none_is_produced() {
    let (_, error) = check(
        &config(Initial::Content, ToolChoice::Required, false),
        "No call.",
    );
    assert_eq!(
        error,
        Some(Error::ToolChoiceUnsatisfied(Unsatisfied::NotProduced))
    );
}

#[test]
fn a_retracted_streamed_call_ends_the_stream_unsafely() {
    let config = config(Initial::Content, ToolChoice::Auto, true);
    // The call streams while partial; the final parse, which needs it
    // complete, drops it.
    let text = "<call:weather>city=«x";
    let bytes: Vec<&[u8]> = text.as_bytes().chunks(1).collect();
    assert_eq!(run(&config, &bytes, false).1, Some(Error::UnsafeStream));
    assert_eq!(run(&config, &bytes, true).1, None);
}

#[test]
fn a_stopping_sink_stops_the_stream() {
    let mut stream = Stream::new(config(Initial::Content, ToolChoice::None, false));
    let mut sink = |_: Event<'_>| false;
    assert_eq!(
        stream.feed(b"text", &mut Stub, &mut sink),
        Err(Error::Stopped)
    );
    assert_eq!(
        stream.finish(true, false, &mut Stub, &mut sink),
        Err(Error::Internal)
    );
}

/// A host whose Qwen scanner panics.
struct Panicking;

impl Host for Panicking {
    fn qwen_scan(&mut self, _text: &[u8]) -> Result<QwenScan, Error> {
        panic!("scanner failed");
    }
    fn dialect_scan(
        &mut self,
        _text: &[u8],
        _markers: &[&[u8]],
        _partial: bool,
    ) -> Result<DialectScan, Error> {
        panic!("scanner failed");
    }
    fn grammar_accepts(&mut self, _tool: usize, _arguments: &[u8]) -> Result<bool, Error> {
        panic!("grammar failed");
    }
}

#[test]
fn a_panic_inside_feed_leaves_the_stream_refusing_input() {
    let mut config = config(Initial::Content, ToolChoice::Auto, false);
    config.markup.native = true;
    let mut stream = Stream::new(config);
    let mut sink = |_: Event<'_>| true;
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        stream.feed(b"<tool_call>{", &mut Panicking, &mut sink)
    }));
    assert!(panicked.is_err());
    assert_eq!(
        stream.feed(b"x", &mut Stub, &mut sink),
        Err(Error::Internal)
    );
    assert_eq!(
        stream.finish(true, false, &mut Stub, &mut sink),
        Err(Error::Internal)
    );
}

#[test]
fn a_self_intersecting_schema_parses_quickly() {
    // Each part of `a` is `a` again; checking a value once used to visit
    // 3^17 schema nodes on every re-parse.
    let mut config = config(Initial::Content, ToolChoice::Auto, true);
    config.tools = vec![
        Tool::new(
            "t".to_owned(),
            br##"{"type":"object","$defs":{"a":{"allOf":[{"$ref":"#/$defs/a"},{"$ref":"#/$defs/a"},{"$ref":"#/$defs/a"}]}},"properties":{"x":{"$ref":"#/$defs/a"}}}"##,
        )
        .unwrap(),
    ];
    let text = "<call:t>x=«1»;</call>";
    let start = std::time::Instant::now();
    let bytes: Vec<&[u8]> = text.as_bytes().chunks(1).collect();
    let (out, error) = run(&config, &bytes, false);
    assert!(start.elapsed() < std::time::Duration::from_secs(1));
    assert_eq!(error, None);
    assert_eq!(out[2], r#"tools [("t", "{\"x\":\"1\"}", true)]"#);
}
