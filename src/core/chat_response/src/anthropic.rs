//! Anthropic Messages API (`/v1/messages`) streaming and buffered output state machine.

use std::fmt::Write;

use gufo_json::{Limits, Object, Value, ValueError, parse_value};

use crate::chat::{FinishReason, ToolDelta};
use crate::format::{
    dump_json, named_sse_frame, num_usize, push_json_escaped, push_member, set_member,
};
use crate::metrics::Metrics;

pub struct AnthropicOutput {
    streaming: bool,
    message: Object,
    tool: Option<Object>,
    tool_arguments: String,
    text: String,
    held: String,
    index: usize,
    active: bool,
    reasoning: bool,
    tool_use: bool,
    connected: bool,
    /// Why the arguments of a complete call are not JSON.
    invalid_arguments: Option<String>,
    scratch: String,
}

impl AnthropicOutput {
    pub fn new(id: String, model: String, streaming: bool) -> Self {
        let mut usage = Object::new();
        push_member(&mut usage, "input_tokens", num_usize(0));
        push_member(&mut usage, "output_tokens", num_usize(0));

        let mut message = Object::new();
        push_member(&mut message, "id", Value::String(id));
        push_member(&mut message, "type", Value::String("message".to_string()));
        push_member(&mut message, "role", Value::String("assistant".to_string()));
        push_member(&mut message, "model", Value::String(model));
        push_member(&mut message, "content", Value::Array(Vec::new()));
        push_member(&mut message, "stop_reason", Value::Null);
        push_member(&mut message, "stop_sequence", Value::Null);
        push_member(&mut message, "usage", Value::Object(usage));

        Self {
            streaming,
            message,
            tool: None,
            tool_arguments: String::new(),
            text: String::new(),
            held: String::new(),
            index: 0,
            active: false,
            reasoning: false,
            tool_use: false,
            connected: true,
            invalid_arguments: None,
            scratch: String::with_capacity(256),
        }
    }

    pub fn begin(&mut self, write: &mut dyn FnMut(&str) -> bool) -> bool {
        let mut event = Self::event("message_start");
        push_member(&mut event, "message", Value::Object(self.message.clone()));
        self.emit(event, write)
    }

    pub fn append(
        &mut self,
        text: &str,
        reasoning: bool,
        write: &mut dyn FnMut(&str) -> bool,
    ) -> bool {
        if !text.is_empty() && !self.close_tool(false, write) {
            return false;
        }
        if (!self.active || self.reasoning != reasoning) && !self.close_block(write) {
            return false;
        }
        let owned_piece;
        let text: &str = if reasoning {
            let trimmed = if !self.active || !self.reasoning {
                trim_leading_ascii_whitespace(text)
            } else {
                text
            };
            let body = trim_trailing_ascii_whitespace(trimmed);
            if body.is_empty() {
                self.held.push_str(trimmed);
                return true;
            }
            let trailing = &trimmed[body.len()..];
            let mut piece = std::mem::replace(&mut self.held, trailing.to_string());
            piece.push_str(body);
            owned_piece = piece;
            &owned_piece
        } else {
            text
        };
        if text.is_empty() {
            return true;
        }
        if !self.active || self.reasoning != reasoning {
            self.reasoning = reasoning;
            self.active = true;
            self.text.clear();
            let block = self.block();
            if !self.emit(self.block_start(block), write) {
                return false;
            }
        }
        self.text.push_str(text);
        if !self.streaming {
            return true;
        }
        if !self.connected {
            return false;
        }
        let (delta_type, field) = if reasoning {
            ("thinking_delta", "thinking")
        } else {
            ("text_delta", "text")
        };
        self.scratch.clear();
        let _ = write!(
            self.scratch,
            "event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":{},\"delta\":{{\"type\":\"{delta_type}\",\"{field}\":\"",
            self.index
        );
        push_json_escaped(&mut self.scratch, text);
        self.scratch.push_str("\"}}\n\n");
        self.connected = write(&self.scratch);
        self.connected
    }

    pub fn tool(&mut self, call: &ToolDelta<'_>, write: &mut dyn FnMut(&str) -> bool) -> bool {
        if !call.id.is_empty() {
            if !self.close_block(write) {
                return false;
            }
            let mut tool = Object::new();
            push_member(&mut tool, "type", Value::String("tool_use".to_string()));
            push_member(&mut tool, "id", Value::String(call.id.to_string()));
            push_member(&mut tool, "name", Value::String(call.name.to_string()));
            push_member(&mut tool, "input", Value::Object(Object::new()));
            self.tool_arguments.clear();
            let start = self.block_start(Value::Object(tool.clone()));
            self.tool = Some(tool);
            if !self.emit(start, write) {
                return false;
            }
        }
        self.tool_arguments.push_str(call.arguments);
        if !call.arguments.is_empty() && self.streaming {
            if !self.connected {
                return false;
            }
            self.scratch.clear();
            let _ = write!(
                self.scratch,
                "event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":{},\"delta\":{{\"type\":\"input_json_delta\",\"partial_json\":\"",
                self.index
            );
            push_json_escaped(&mut self.scratch, call.arguments);
            self.scratch.push_str("\"}}\n\n");
            self.connected = write(&self.scratch);
            if !self.connected {
                return false;
            }
        }
        !call.complete || self.close_tool(true, write)
    }

    /// The parse error of a complete call's arguments, once `tool` failed on it.
    pub fn take_invalid_arguments(&mut self) -> Option<String> {
        self.invalid_arguments.take()
    }

    pub fn complete(
        &mut self,
        finish_reason: FinishReason,
        stop_sequence: &str,
        metrics: &Metrics,
        write: &mut dyn FnMut(&str) -> bool,
    ) -> (String, bool) {
        self.close_tool(false, write);
        self.close_block(write);
        let empty_content = self
            .message
            .get("content")
            .and_then(Value::as_array)
            .is_some_and(<[Value]>::is_empty);
        if empty_content {
            self.text.clear();
            self.reasoning = false;
            self.active = true;
            let block = self.block();
            self.emit(self.block_start(block), write);
            self.close_block(write);
        }
        let stop_reason = match finish_reason {
            FinishReason::StopSequence => "stop_sequence",
            FinishReason::Length => "max_tokens",
            FinishReason::Stop => {
                if self.tool_use {
                    "tool_use"
                } else {
                    "end_turn"
                }
            }
        };
        let stop_seq_val = if finish_reason == FinishReason::StopSequence {
            Value::String(stop_sequence.to_string())
        } else {
            Value::Null
        };
        set_member(
            &mut self.message,
            "stop_reason",
            Value::String(stop_reason.to_string()),
        );
        set_member(&mut self.message, "stop_sequence", stop_seq_val.clone());

        let uncached = metrics
            .prompt_tokens
            .saturating_sub(metrics.cached_prompt_tokens);
        let mut usage = Object::new();
        push_member(&mut usage, "input_tokens", num_usize(uncached));
        push_member(
            &mut usage,
            "output_tokens",
            num_usize(metrics.completion_tokens),
        );
        push_member(&mut usage, "cache_creation_input_tokens", num_usize(0));
        push_member(
            &mut usage,
            "cache_read_input_tokens",
            num_usize(metrics.cached_prompt_tokens),
        );
        set_member(&mut self.message, "usage", Value::Object(usage.clone()));
        set_member(&mut self.message, "timings", metrics.timings_value());

        let mut delta_inner = Object::new();
        push_member(
            &mut delta_inner,
            "stop_reason",
            Value::String(stop_reason.to_string()),
        );
        push_member(&mut delta_inner, "stop_sequence", stop_seq_val);
        let mut delta = Self::event("message_delta");
        push_member(&mut delta, "delta", Value::Object(delta_inner));
        push_member(&mut delta, "usage", Value::Object(usage));
        let ok = if self.emit(delta, write) {
            self.emit(Self::event("message_stop"), write)
        } else {
            false
        };
        let body = dump_json(&Value::Object(std::mem::take(&mut self.message)));
        (body, ok)
    }

    pub fn fail(&mut self, message: &str, write: &mut dyn FnMut(&str) -> bool) -> bool {
        let mut err = Object::new();
        push_member(&mut err, "type", Value::String("api_error".to_string()));
        push_member(&mut err, "message", Value::String(message.to_string()));
        let mut event = Self::event("error");
        push_member(&mut event, "error", Value::Object(err));
        self.emit(event, write)
    }

    fn close_tool(&mut self, complete: bool, write: &mut dyn FnMut(&str) -> bool) -> bool {
        let Some(mut tool) = self.tool.take() else {
            return self.connected;
        };
        if complete {
            match parse_value(self.tool_arguments.as_bytes(), Limits::NONE) {
                Ok(parsed) => set_member(&mut tool, "input", parsed),
                Err(error) => {
                    self.invalid_arguments = Some(match error {
                        ValueError::Rejected(reason) => format!("JSON parse error: {reason}"),
                        ValueError::TooLarge | ValueError::NoMemory => "out of memory".to_string(),
                        ValueError::Internal => "JSON parser failed".to_string(),
                    });
                    return false;
                }
            }
            self.tool_use = true;
        }
        if let Some(Value::Array(content)) = self.message.get_mut("content") {
            content.push(Value::Object(tool));
        }
        self.block_stop(write)
    }

    fn event(kind: &str) -> Object {
        let mut event = Object::new();
        push_member(&mut event, "type", Value::String(kind.to_string()));
        event
    }

    fn block(&self) -> Value {
        let mut block = Object::new();
        push_member(
            &mut block,
            "type",
            Value::String(if self.reasoning { "thinking" } else { "text" }.to_string()),
        );
        push_member(
            &mut block,
            if self.reasoning { "thinking" } else { "text" },
            Value::String(self.text.clone()),
        );
        if self.reasoning {
            push_member(&mut block, "signature", Value::String(String::new()));
        }
        Value::Object(block)
    }

    fn block_start(&self, block: Value) -> Object {
        let mut event = Self::event("content_block_start");
        push_member(&mut event, "index", num_usize(self.index));
        push_member(&mut event, "content_block", block);
        event
    }

    fn block_stop(&mut self, write: &mut dyn FnMut(&str) -> bool) -> bool {
        let mut event = Self::event("content_block_stop");
        push_member(&mut event, "index", num_usize(self.index));
        self.index += 1;
        self.emit(event, write)
    }

    fn close_block(&mut self, write: &mut dyn FnMut(&str) -> bool) -> bool {
        self.held.clear();
        if !self.active {
            return self.connected;
        }
        let block = self.block();
        if let Some(Value::Array(content)) = self.message.get_mut("content") {
            content.push(block);
        }
        self.active = false;
        self.block_stop(write)
    }

    fn emit(&mut self, event: Object, write: &mut dyn FnMut(&str) -> bool) -> bool {
        if !self.streaming {
            return true;
        }
        if !self.connected {
            return false;
        }
        let kind = event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let frame = named_sse_frame(&kind, &Value::Object(event));
        self.connected = write(&frame);
        self.connected
    }
}

fn is_c_space(b: u8) -> bool {
    b.is_ascii_whitespace() || b == 0x0b
}

fn trim_leading_ascii_whitespace(s: &str) -> &str {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() && is_c_space(bytes[i]) {
        i += 1;
    }
    &s[i..]
}

fn trim_trailing_ascii_whitespace(s: &str) -> &str {
    let bytes = s.as_bytes();
    let mut end = bytes.len();
    while end > 0 && is_c_space(bytes[end - 1]) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_thinking_whitespace_and_text_blocks() {
        let mut out = AnthropicOutput::new("msg_1".into(), "qwen".into(), true);
        let mut frames = Vec::new();
        let mut collect = |s: &str| {
            frames.push(s.to_string());
            true
        };
        assert!(out.begin(&mut collect));
        assert!(out.append("  think ", true, &mut collect));
        assert!(out.append("more  ", true, &mut collect));
        assert!(out.append("Hi", false, &mut collect));
        let (body, ok) = out.complete(FinishReason::Stop, "", &Metrics::default(), &mut collect);
        assert!(ok);
        assert!(body.contains(r#""thinking":"think more""#));
        assert!(body.contains(r#""text":"Hi""#));
        assert!(frames.last().unwrap().starts_with("event: message_stop\n"));
    }

    fn call<'a>(id: &'a str, arguments: &'a str, complete: bool) -> ToolDelta<'a> {
        ToolDelta {
            index: 0,
            begin: !id.is_empty(),
            id,
            name: "f",
            arguments,
            complete,
        }
    }

    #[test]
    fn anthropic_tool_input_spells_numbers_canonically() {
        let mut out = AnthropicOutput::new("msg_1".into(), "qwen".into(), false);
        let mut ignore = |_: &str| true;
        assert!(out.tool(&call("toolu_1", r#"{"x":1.0,"#, false), &mut ignore));
        assert!(out.tool(&call("", r#""y":1e3}"#, true), &mut ignore));
        let (body, _) = out.complete(FinishReason::Stop, "", &Metrics::default(), &mut ignore);
        assert!(body.contains(r#""input":{"x":1,"y":1000}"#), "{body}");
        assert!(body.contains(r#""stop_reason":"tool_use""#));
    }

    #[test]
    fn anthropic_invalid_complete_arguments_fail() {
        let mut out = AnthropicOutput::new("msg_1".into(), "qwen".into(), false);
        let mut ignore = |_: &str| true;
        assert!(!out.tool(&call("toolu_1", r#"{"x":"#, true), &mut ignore));
        let reason = out.take_invalid_arguments().expect("parse error");
        assert!(reason.starts_with("JSON parse error: "), "{reason}");
        assert!(out.take_invalid_arguments().is_none());
    }
}
