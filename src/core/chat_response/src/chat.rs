//! OpenAI Chat Completions (`/v1/chat/completions`) response and SSE encoding.

use std::fmt::Write;

use gufo_json::{Object, Value};

use crate::format::{
    SSE_DONE, dump_json, num_i64, num_usize, parse_json, push_json_escaped, push_member, sse_frame,
};
use crate::metrics::{Metrics, usage_json};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    StopSequence,
}

impl FinishReason {
    pub fn as_chat_str(self, has_tool_calls: bool) -> &'static str {
        match self {
            Self::StopSequence => "stop",
            Self::Length => "length",
            Self::Stop => {
                if has_tool_calls {
                    "tool_calls"
                } else {
                    "stop"
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone)]
pub struct ToolDelta<'a> {
    pub index: usize,
    pub begin: bool,
    pub id: &'a str,
    pub name: &'a str,
    pub arguments: &'a str,
    pub complete: bool,
}

pub struct NonStreamingInput<'a> {
    pub id: &'a str,
    pub created: i64,
    pub model: &'a str,
    pub reasoning: &'a str,
    pub text: &'a str,
    pub calls: &'a [ToolCall],
    pub logprobs_json: &'a str,
    pub finish_reason: FinishReason,
    pub cache_persist: bool,
    pub metrics: &'a Metrics,
}

pub fn tool_calls_json(calls: &[ToolCall]) -> Value {
    let mut output = Vec::with_capacity(calls.len());
    for call in calls {
        let mut function = Object::new();
        push_member(&mut function, "name", Value::String(call.name.clone()));
        push_member(
            &mut function,
            "arguments",
            Value::String(call.arguments.clone()),
        );
        let mut item = Object::new();
        push_member(&mut item, "id", Value::String(call.id.clone()));
        push_member(&mut item, "type", Value::String("function".to_string()));
        push_member(&mut item, "function", Value::Object(function));
        output.push(Value::Object(item));
    }
    Value::Array(output)
}

pub fn base_chunk(id: &str, created: i64, model: &str) -> Object {
    let mut chunk = Object::new();
    push_member(&mut chunk, "id", Value::String(id.to_string()));
    push_member(
        &mut chunk,
        "object",
        Value::String("chat.completion.chunk".to_string()),
    );
    push_member(&mut chunk, "created", num_i64(created));
    push_member(&mut chunk, "model", Value::String(model.to_string()));
    chunk
}

pub fn choice_chunk(
    id: &str,
    created: i64,
    model: &str,
    delta: Value,
    finish_reason: Option<&str>,
    logprobs: Option<Value>,
) -> Object {
    let mut chunk = base_chunk(id, created, model);
    let mut choice = Object::new();
    push_member(&mut choice, "index", num_usize(0));
    push_member(&mut choice, "delta", delta);
    if let Some(lp) = logprobs {
        push_member(&mut choice, "logprobs", lp);
    }
    push_member(
        &mut choice,
        "finish_reason",
        match finish_reason {
            Some(reason) => Value::String(reason.to_string()),
            None => Value::Null,
        },
    );
    push_member(
        &mut chunk,
        "choices",
        Value::Array(vec![Value::Object(choice)]),
    );
    chunk
}

pub fn non_streaming_response(input: &NonStreamingInput<'_>) -> String {
    let mut message = Object::new();
    push_member(&mut message, "role", Value::String("assistant".to_string()));
    if !input.reasoning.is_empty() {
        push_member(
            &mut message,
            "reasoning_content",
            Value::String(input.reasoning.to_string()),
        );
    }
    if input.text.is_empty() && !input.calls.is_empty() {
        push_member(&mut message, "content", Value::Null);
    } else {
        push_member(
            &mut message,
            "content",
            Value::String(input.text.to_string()),
        );
    }
    if !input.calls.is_empty() {
        push_member(&mut message, "tool_calls", tool_calls_json(input.calls));
    }

    let mut choice = Object::new();
    push_member(&mut choice, "index", num_usize(0));
    if let Some(logprobs) = parse_json(input.logprobs_json) {
        push_member(&mut choice, "logprobs", logprobs);
    }
    push_member(&mut choice, "message", Value::Object(message));
    push_member(
        &mut choice,
        "finish_reason",
        Value::String(
            input
                .finish_reason
                .as_chat_str(!input.calls.is_empty())
                .to_string(),
        ),
    );

    let mut response = Object::new();
    push_member(&mut response, "id", Value::String(input.id.to_string()));
    push_member(
        &mut response,
        "object",
        Value::String("chat.completion".to_string()),
    );
    push_member(&mut response, "created", num_i64(input.created));
    push_member(
        &mut response,
        "model",
        Value::String(input.model.to_string()),
    );
    push_member(
        &mut response,
        "choices",
        Value::Array(vec![Value::Object(choice)]),
    );
    push_member(&mut response, "usage", usage_json(input.metrics));
    push_member(&mut response, "timings", input.metrics.timings_value());
    push_member(
        &mut response,
        "cache_persist",
        Value::Bool(input.cache_persist),
    );
    dump_json(&Value::Object(response))
}

pub struct ChatStream {
    id: String,
    created: i64,
    model: String,
    prefix: String,
    scratch: String,
}

const CHUNK_SUFFIX: &str = ",\"finish_reason\":null}]}\n\n";

impl ChatStream {
    pub fn new(id: String, created: i64, model: String) -> Self {
        let mut prefix = String::with_capacity(id.len() + model.len() + 96);
        prefix.push_str("data: {\"id\":\"");
        push_json_escaped(&mut prefix, &id);
        prefix.push_str("\",\"object\":\"chat.completion.chunk\",\"created\":");
        let _ = write!(prefix, "{created}");
        prefix.push_str(",\"model\":\"");
        push_json_escaped(&mut prefix, &model);
        prefix.push_str("\",\"choices\":[{\"index\":0,\"delta\":");
        Self {
            id,
            created,
            model,
            prefix,
            scratch: String::with_capacity(256),
        }
    }

    pub fn emit_role(&mut self, write: &mut dyn FnMut(&str) -> bool) -> bool {
        self.emit_raw_delta(r#"{"role":"assistant"}"#, "", write)
    }

    pub fn emit_text_delta(
        &mut self,
        reasoning: bool,
        text: &str,
        logprobs_json: &str,
        write: &mut dyn FnMut(&str) -> bool,
    ) -> bool {
        if text.is_empty() {
            return true;
        }
        let field = if reasoning {
            "reasoning_content"
        } else {
            "content"
        };
        self.scratch.clear();
        self.scratch.push_str(&self.prefix);
        self.scratch.push_str("{\"");
        self.scratch.push_str(field);
        self.scratch.push_str("\":\"");
        push_json_escaped(&mut self.scratch, text);
        self.scratch.push_str("\"}");
        if !logprobs_json.is_empty() {
            self.scratch.push_str(",\"logprobs\":");
            self.scratch.push_str(logprobs_json);
        }
        self.scratch.push_str(CHUNK_SUFFIX);
        write(&self.scratch)
    }

    pub fn emit_empty_delta(
        &mut self,
        logprobs_json: &str,
        write: &mut dyn FnMut(&str) -> bool,
    ) -> bool {
        self.emit_raw_delta("{}", logprobs_json, write)
    }

    pub fn emit_tool_delta(
        &mut self,
        delta: &ToolDelta<'_>,
        logprobs_json: &str,
        write: &mut dyn FnMut(&str) -> bool,
    ) -> bool {
        if !delta.begin && delta.arguments.is_empty() {
            return true;
        }
        self.scratch.clear();
        self.scratch.push_str(&self.prefix);
        let _ = write!(
            self.scratch,
            "{{\"tool_calls\":[{{\"index\":{}",
            delta.index
        );
        if delta.begin {
            self.scratch.push_str(",\"id\":\"");
            push_json_escaped(&mut self.scratch, delta.id);
            self.scratch
                .push_str("\",\"type\":\"function\",\"function\":{\"name\":\"");
            push_json_escaped(&mut self.scratch, delta.name);
            self.scratch.push_str("\",\"arguments\":\"");
        } else {
            self.scratch.push_str(",\"function\":{\"arguments\":\"");
        }
        push_json_escaped(&mut self.scratch, delta.arguments);
        self.scratch.push_str("\"}}]}");
        if !logprobs_json.is_empty() {
            self.scratch.push_str(",\"logprobs\":");
            self.scratch.push_str(logprobs_json);
        }
        self.scratch.push_str(CHUNK_SUFFIX);
        write(&self.scratch)
    }

    pub fn emit_progress(
        &mut self,
        prompt_progress_json: &str,
        write: &mut dyn FnMut(&str) -> bool,
    ) -> bool {
        let mut chunk = choice_chunk(
            &self.id,
            self.created,
            &self.model,
            Value::Object(Object::new()),
            None,
            None,
        );
        if let Some(progress) = parse_json(prompt_progress_json) {
            push_member(&mut chunk, "prompt_progress", progress);
        }
        write(&sse_frame(&Value::Object(chunk)))
    }

    pub fn finish(
        &mut self,
        finish_reason: FinishReason,
        has_tool_calls: bool,
        include_usage: bool,
        cache_persist: bool,
        metrics: &Metrics,
        write: &mut dyn FnMut(&str) -> bool,
    ) -> bool {
        let timings = metrics.timings_value();
        let mut terminal = choice_chunk(
            &self.id,
            self.created,
            &self.model,
            Value::Object(Object::new()),
            Some(finish_reason.as_chat_str(has_tool_calls)),
            None,
        );
        push_member(&mut terminal, "timings", timings.clone());
        push_member(&mut terminal, "cache_persist", Value::Bool(cache_persist));
        if !write(&sse_frame(&Value::Object(terminal))) {
            return false;
        }
        if include_usage {
            let mut usage_chunk = base_chunk(&self.id, self.created, &self.model);
            push_member(&mut usage_chunk, "choices", Value::Array(Vec::new()));
            push_member(&mut usage_chunk, "usage", usage_json(metrics));
            push_member(&mut usage_chunk, "timings", timings);
            push_member(
                &mut usage_chunk,
                "cache_persist",
                Value::Bool(cache_persist),
            );
            if !write(&sse_frame(&Value::Object(usage_chunk))) {
                return false;
            }
        }
        write(SSE_DONE)
    }

    fn emit_raw_delta(
        &mut self,
        delta: &str,
        logprobs_json: &str,
        write: &mut dyn FnMut(&str) -> bool,
    ) -> bool {
        self.scratch.clear();
        self.scratch.push_str(&self.prefix);
        self.scratch.push_str(delta);
        if !logprobs_json.is_empty() {
            self.scratch.push_str(",\"logprobs\":");
            self.scratch.push_str(logprobs_json);
        }
        self.scratch.push_str(CHUNK_SUFFIX);
        write(&self.scratch)
    }
}

pub fn stream_error_sse(message: &str, code: &str, write: &mut dyn FnMut(&str) -> bool) -> bool {
    let mut detail = Object::new();
    push_member(&mut detail, "message", Value::String(message.to_string()));
    push_member(
        &mut detail,
        "type",
        Value::String("server_error".to_string()),
    );
    push_member(&mut detail, "code", Value::String(code.to_string()));
    let mut err = Object::new();
    push_member(&mut err, "error", Value::Object(detail));
    let sent = write(&sse_frame(&Value::Object(err)));
    let _ = write(SSE_DONE);
    sent
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_stream_emits_expected_sse_sequence() {
        let mut stream = ChatStream::new("chatcmpl-1".into(), 100, "qwen".into());
        let mut frames = Vec::new();
        let mut collect = |s: &str| {
            frames.push(s.to_string());
            true
        };
        assert!(stream.emit_role(&mut collect));
        assert!(stream.emit_text_delta(false, "Hi", "", &mut collect));
        assert!(stream.finish(
            FinishReason::Stop,
            false,
            true,
            false,
            &Metrics::default(),
            &mut collect
        ));
        assert_eq!(frames.len(), 5);
        assert!(frames[0].contains(r#""delta":{"role":"assistant"}"#));
        assert!(frames[1].contains(r#""delta":{"content":"Hi"}"#));
        assert!(frames[2].contains(r#""finish_reason":"stop""#));
        assert!(frames[3].contains(r#""usage":{"#));
        assert_eq!(frames[4], SSE_DONE);
    }
}
